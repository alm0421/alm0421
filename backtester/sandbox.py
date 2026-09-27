"""Sealed evaluation of Python-function rules (f(df, ns) -> Series) and portfolio functions (f(date, history) ->
{ticker: weight}) in a separate process that only ever holds the data up to the day being decided.

Why a process: a function run inside the backtester can walk the interpreter (inspect.stack(), gc.get_objects(),
the namespace's own attributes) and find the whole price history the engine holds, or read the price files. So the
function never runs in the backtester's process. It runs in a fresh child process forked from a small "fork server"
that imported pandas / numpy / backtester but never loaded any data:

  - the function is shipped by value (its code, the globals it names, its closure and defaults; functions of
    installed libraries and of the backtester by reference). Everything shipped is inspected first (pack): a
    pandas object with a date index reaching past the first day the function answers, a numeric array / list / dict
    of 500+ values or a 100 kB string is refused with LeakError - that is data captured before the run
    (FULL = data.load("SPY") at module level, a closure over a dict of full series), which would let the function
    read later prices. Load data inside the function instead (ns["sym"], data.load, the history argument): those
    are cut at each day.
  - the bars are sent one day at a time: at the moment it answers day D the child has received nothing after D,
    so nothing in its memory (frames, stack, gc) is later than D.
  - data.load / sym() and the other point-in-time data functions (market_cap, tbill_rate, treasury_10y, cape...)
    are forwarded to the backtester, which answers them cut at D; every other data function is refused.
  - file, process and network access is refused at the Python level in the child (open, io / os / posix open,
    FileIO, pandas / numpy readers, sockets, subprocess, os.fork / exec / spawn).
  - each stream (one function on one ticker) or call gets its own fresh child, so state a function keeps
    (a global, a default dict) starts empty and only ever saw the days up to the current one.

What is not prevented: native code (ctypes, a C extension) or anything else below Python can read what the
operating system lets the process read (the price files, other processes' memory). No ordinary way of writing a
rule reaches later data.

Long streams are split into consecutive chunks answered by several children at once (each chunk is still fed day
by day); a function whose answers depend on which earlier days it was called on (kept state) is detected at the
chunk boundaries and re-streamed in one pass.
"""
from __future__ import annotations

import atexit
import builtins
import io
import marshal
import os
import pickle
import struct
import subprocess
import sys
import threading
import types
from typing import Any

import numpy as np
import pandas as pd

BIG = 500                 # values in one captured container that count as data
BIG_TEXT = 100_000        # characters / bytes in one captured string
IN_CHILD = False          # True inside a sandbox child (a rule calling evaluate() there runs in-process)
STATS: dict = {"jobs": 0, "steps": 0}


class LeakError(ValueError):
    """A Python rule carries data captured outside the run (would let it read later prices)."""


class SandboxError(RuntimeError):
    pass


# ================================================================ packing a function by value

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _lib_roots() -> tuple[str, ...]:
    import sysconfig
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix}
    for k in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            roots.add(sysconfig.get_paths()[k])
        except KeyError:
            pass
    for p in sys.path:
        if p and ("site-packages" in p or "dist-packages" in p):
            roots.add(p)
    return tuple(os.path.abspath(r) + os.sep for r in roots if r)


_LIB = None


def _library_module(modname: str | None) -> bool:
    """Code the child can import by reference: the backtester itself, the standard library, installed packages."""
    global _LIB
    if not modname or modname in ("__main__", "__mp_main__"):
        return False
    mod = sys.modules.get(modname)
    if mod is None:
        return False
    f = getattr(mod, "__file__", None)
    if f is None:
        return modname in sys.builtin_module_names or modname.split(".")[0] in sys.builtin_module_names
    f = os.path.abspath(f)
    if f.startswith(_PKG_DIR + os.sep):
        return True
    if _LIB is None:
        _LIB = _lib_roots()
    return f.startswith(_LIB)


def _by_value(fn) -> bool:
    q = getattr(fn, "__qualname__", "")
    return "<locals>" in q or "<lambda>" in q or not _library_module(getattr(fn, "__module__", None))


def _code_names(code) -> set[str]:
    out = set(code.co_names)
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            out |= _code_names(c)
    return out


_EMPTY = "<empty cell>"


def _fn_state(fn) -> dict:
    g = fn.__globals__
    names = {n: g[n] for n in sorted(_code_names(fn.__code__)) if n in g and n != "__builtins__"}
    cells = []
    for c in fn.__closure__ or ():
        try:
            cells.append(c.cell_contents)
        except ValueError:
            cells.append(_EMPTY)
    return {"globals": names, "defaults": fn.__defaults__, "kwdefaults": fn.__kwdefaults__, "cells": cells,
            "dict": dict(fn.__dict__), "doc": fn.__doc__}


_GLOBALS: dict = {}     # (child side) one globals dict per source module, shared by its functions


def _make_function(code_bytes: bytes, name: str, qualname: str, modname: str, ncells: int, gkey: int):
    code = marshal.loads(code_bytes)
    g = _GLOBALS.get(gkey)
    if g is None:
        g = _GLOBALS[gkey] = {"__builtins__": builtins, "__name__": modname}
    cells = tuple(types.CellType() for _ in range(ncells)) if ncells else None
    f = types.FunctionType(code, g, name, None, cells)
    f.__qualname__, f.__module__ = qualname, modname
    return f


def _set_function_state(f, state: dict):
    f.__globals__.update(state["globals"])
    f.__defaults__, f.__kwdefaults__, f.__doc__ = state["defaults"], state["kwdefaults"], state["doc"]
    for cell, v in zip(f.__closure__ or (), state["cells"]):
        if not (isinstance(v, str) and v == _EMPTY):
            cell.cell_contents = v
    f.__dict__.update(state["dict"])
    return f


def _import(name: str):
    import importlib
    return importlib.import_module(name)


def _describe(obj, cutoff) -> str | None:
    """Why `obj` (on its own, not its contents) is data that could reach past `cutoff`, or None."""
    if isinstance(obj, (pd.Series, pd.DataFrame, pd.Index)):
        idx = obj if isinstance(obj, pd.Index) else obj.index
        kind = type(obj).__name__
        if isinstance(idx, pd.DatetimeIndex):
            if len(idx) and cutoff is not None:
                last = idx.max()
                try:
                    past = last > pd.Timestamp(cutoff)
                except TypeError:        # tz-aware vs naive
                    past = True
                if past:
                    return f"a {kind} of {len(obj):,} rows dated up to {last.date()}"
            return None
        if len(obj) >= BIG:
            return f"a {kind} of {len(obj):,} rows"
        return None
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fciub" and obj.size >= BIG:
            return f"an array of {obj.size:,} numbers"
        if obj.dtype.kind == "M" and obj.size and cutoff is not None and obj.max() > np.datetime64(pd.Timestamp(cutoff)):
            return f"an array of {obj.size:,} dates up to {pd.Timestamp(obj.max()).date()}"
        return None
    if isinstance(obj, (list, tuple, set, frozenset)) and len(obj) >= BIG:
        nums = sum(1 for x in list(obj)[:2000] if isinstance(x, (int, float, np.number)) and not isinstance(x, bool))
        if nums >= min(len(obj), 2000) // 2:
            return f"a {type(obj).__name__} of {len(obj):,} numbers"
        return None
    if isinstance(obj, dict) and len(obj) >= BIG:
        return f"a dict of {len(obj):,} entries"
    if isinstance(obj, (str, bytes, bytearray, memoryview)) and len(obj) >= BIG_TEXT:
        return f"a {type(obj).__name__} of {len(obj):,} characters"
    return None


def _scan(obj, label: str, cutoff, seen: set, depth: int = 0):
    """Walk what a function carries (named, for the message) and raise LeakError at the first piece of data."""
    if id(obj) in seen or depth > 8:
        return
    seen.add(id(obj))
    d = _describe(obj, cutoff)
    if d:
        raise LeakError(f"{label} is {d}")
    if isinstance(obj, (pd.Series, pd.DataFrame, pd.Index, np.ndarray, str, bytes, bytearray, type, types.ModuleType)):
        return
    if isinstance(obj, types.FunctionType):
        if not _by_value(obj):
            return
        st = _fn_state(obj)
        nm = obj.__name__
        for n, v in st["globals"].items():
            _scan(v, f"global {n!r}" + (f" (used by {nm}())" if depth else ""), cutoff, seen, depth + 1)
        for n, v in zip(obj.__code__.co_freevars, st["cells"]):
            _scan(v, f"variable {n!r} captured by {nm}()", cutoff, seen, depth + 1)
        for v in (st["defaults"] or ()):
            _scan(v, f"a default argument of {nm}()", cutoff, seen, depth + 1)
        for v in (st["kwdefaults"] or {}).values():
            _scan(v, f"a default argument of {nm}()", cutoff, seen, depth + 1)
        for n, v in st["dict"].items():
            _scan(v, f"attribute {n!r} of {nm}()", cutoff, seen, depth + 1)
        return
    if isinstance(obj, types.MethodType):
        _scan(obj.__func__, label, cutoff, seen, depth + 1)
        _scan(obj.__self__, f"the object {label} belongs to", cutoff, seen, depth + 1)
        return
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:5000]:
            _scan(k, label, cutoff, seen, depth + 1)
            _scan(v, f"{label}[{k!r}]" if isinstance(k, (str, int)) else label, cutoff, seen, depth + 1)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in list(obj)[:5000]:
            _scan(v, label, cutoff, seen, depth + 1)
        return
    import functools
    if isinstance(obj, functools.partial):
        for v in (obj.func, *obj.args, *(obj.keywords or {}).values()):
            _scan(v, label, cutoff, seen, depth + 1)
        return
    dct = getattr(obj, "__dict__", None)
    if isinstance(dct, dict) and not isinstance(obj, type):
        _scan(dct, f"{label} (an object)", cutoff, seen, depth + 1)


class _Packer(pickle.Pickler):
    def __init__(self, f, cutoff):
        super().__init__(f, protocol=pickle.HIGHEST_PROTOCOL)
        self.cutoff = cutoff

    def persistent_id(self, obj):     # every object shipped is checked (a backstop behind _scan's named walk)
        d = _describe(obj, self.cutoff)
        if d:
            raise LeakError(f"it carries {d}")
        if isinstance(obj, (pd.Series, pd.DataFrame, pd.Index)):
            # judged as a whole (e.g. a series ending before the cutoff): its own arrays are not judged again
            return ("pandas", pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        return None

    def reducer_override(self, obj):
        if isinstance(obj, types.FunctionType) and _by_value(obj):
            code = obj.__code__
            return (_make_function, (marshal.dumps(code), obj.__name__, obj.__qualname__,
                                     getattr(obj, "__module__", None) or "__main__", len(code.co_freevars),
                                     id(obj.__globals__)),
                    _fn_state(obj), None, None, _set_function_state)
        if isinstance(obj, types.ModuleType):
            if not _library_module(obj.__name__) and obj.__name__ in ("__main__", "__mp_main__"):
                raise LeakError("it uses the script's __main__ module as a value")
            return (_import, (obj.__name__,))
        return NotImplemented


class _Unpacker(pickle.Unpickler):
    def persistent_load(self, pid):
        if isinstance(pid, tuple) and pid and pid[0] == "pandas":
            return pickle.loads(pid[1])
        raise pickle.UnpicklingError(f"unknown persistent id {pid!r}")


def _unpack(payload: bytes):
    return _Unpacker(io.BytesIO(payload)).load()


def pack(fn, cutoff=None, what: str = "Python rule") -> bytes:
    """`fn` by value, refused (LeakError) when it carries data that reaches past `cutoff` (see the module doc)."""
    if not callable(fn):
        raise TypeError(f"{what} must be a function")
    name = getattr(fn, "__name__", "") or "(function)"
    try:
        _scan(fn, f"{name}()", cutoff, set())
        buf = io.BytesIO()
        _Packer(buf, cutoff).dump(fn)
    except LeakError as e:
        raise LeakError(
            f"Lookahead: the {what} {name}() uses future data: {e}, captured outside the run"
            f"{f' and reaching past {pd.Timestamp(cutoff).date()}' if cutoff is not None else ''}. A function "
            "only gets the data up to each day; data loaded before the run (e.g. FULL = data.load('SPY') at "
            "module level, or a closure over full price series) would let it read later prices. Load it inside "
            "the function instead: ns['sym']('SPY') or data.load('SPY') in a rule, the `history` argument in a "
            "portfolio function - both are cut at each day.") from None
    except (pickle.PicklingError, TypeError, AttributeError) as e:
        raise SandboxError(f"The {what} {name}() cannot be sent to the sealed evaluator: {e}. Python rules must be "
                           "plain functions (def or lambda) using picklable values.") from None
    return buf.getvalue()


# ================================================================ messages

class _SafeUnpickler(pickle.Unpickler):
    """What the backtester accepts from a child: plain values only (no classes, no functions)."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"a sandbox child sent {module}.{name}")


def _loads_safe(b: bytes):
    return _SafeUnpickler(io.BytesIO(b)).load()


class _PipeConn:
    """Length-prefixed messages over two pipes (the fallback transport, one process per job)."""

    def __init__(self, r, w, proc=None):
        self.r, self.w, self.proc = r, w, proc

    def send_bytes(self, b: bytes):
        self.w.write(struct.pack("!Q", len(b)))
        self.w.write(b)
        self.w.flush()

    def recv_bytes(self) -> bytes:
        h = self.r.read(8)
        if len(h) < 8:
            raise EOFError
        n = struct.unpack("!Q", h)[0]
        b = self.r.read(n)
        if len(b) < n:
            raise EOFError
        return b

    def close(self):
        for f in (self.w, self.r):
            try:
                f.close()
            except OSError:
                pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()


# ================================================================ the fork server (backtester side)

_SERVER_LOCK = threading.Lock()
_SERVER: dict = {}      # pid -> (Popen, socket path, authkey, tmpdir)


def _child_env() -> dict:
    env = dict(os.environ)
    root = os.path.dirname(_PKG_DIR)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    import json
    env["BACKTESTER_SANDBOX_SYSPATH"] = json.dumps([p for p in sys.path if isinstance(p, str)])
    return env


def _use_fork_server() -> bool:
    import socket
    return hasattr(os, "fork") and hasattr(socket, "AF_UNIX") and os.environ.get("BACKTESTER_SANDBOX") != "spawn"


def _server():
    import secrets
    import tempfile
    pid = os.getpid()
    with _SERVER_LOCK:
        s = _SERVER.get(pid)
        if s is not None and s[0].poll() is None:
            return s
        tmp = tempfile.mkdtemp(prefix="bt-sandbox-")
        path = os.path.join(tmp, "s")
        key = secrets.token_bytes(16)
        proc = subprocess.Popen([sys.executable, "-m", "backtester.sandbox", "serve", path, key.hex()],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=_child_env(),
                                cwd=os.getcwd())
        line = proc.stdout.readline()
        if line.strip() != b"ready":
            proc.kill()
            raise SandboxError("the sealed evaluator for Python rules failed to start")
        s = _SERVER[pid] = (proc, path, key, tmp)
        return s


def _shutdown():
    import shutil
    s = _SERVER.pop(os.getpid(), None)
    if s:
        proc, _, _, tmp = s
        try:
            proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)


atexit.register(_shutdown)


def _connect():
    if not _use_fork_server():
        proc = subprocess.Popen([sys.executable, "-m", "backtester.sandbox", "job"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, env=_child_env(), cwd=os.getcwd())
        return _PipeConn(proc.stdout, proc.stdin, proc)
    from multiprocessing.connection import Client
    for attempt in range(2):
        proc, path, key, _ = _server()
        try:
            return Client(path, family="AF_UNIX", authkey=key)
        except (OSError, EOFError):
            if attempt:
                raise
            with _SERVER_LOCK:
                _SERVER.pop(os.getpid(), None)
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


_BUILTIN_EXC = {n: getattr(builtins, n) for n in dir(builtins)
                if isinstance(getattr(builtins, n), type) and issubclass(getattr(builtins, n), BaseException)}


def _raise_child_error(msg):
    _, cls, text, tb = msg
    from . import expr
    from . import data
    exc_type = {"CallableIOError": expr.CallableIOError, "LeakError": LeakError, "DataError": data.DataError,
                "SandboxError": SandboxError}.get(cls) or _BUILTIN_EXC.get(cls)
    if exc_type is None or not issubclass(exc_type, Exception):
        err = SandboxError(f"{cls}: {text}")
    else:
        try:
            err = exc_type(text)
        except Exception:  # noqa: BLE001
            err = SandboxError(f"{cls}: {text}")
    if tb and hasattr(err, "add_note"):
        err.add_note("In the Python rule (run in the sealed evaluator):\n" + tb)
    raise err


# the data functions a rule may call (answered by the backtester, cut at the day being decided)
DATA_FUNCTIONS = {"load", "load_many", "total_return_close", "quoted_close", "market_cap", "shares_outstanding",
                  "shares_adjusted", "splits", "tbill_rate", "cpi", "shiller_known", "cape_percentile",
                  "treasury_10y", "treasury_2y", "yield_curve", "macro_known", "factors"}


def _cut(v, cut):
    if cut is None:
        return v
    cut = pd.Timestamp(cut)
    if isinstance(v, (pd.Series, pd.DataFrame)):
        if isinstance(v.index, pd.DatetimeIndex):
            return v.loc[v.index <= cut].copy()
        if isinstance(v.index, pd.PeriodIndex):
            return v.loc[v.index.end_time <= cut].copy()
        return v.copy()
    if isinstance(v, dict):
        return {k: _cut(x, cut) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_cut(x, cut) for x in v)
    return v


def _answer_rpc(msg, cut):
    """A child's data request, answered on the backtester's data cut at `cut`."""
    from . import data, expr
    _, name, args, kwargs = msg
    try:
        if name == "sym_override":
            v = expr._SYM_OVERRIDE.get(args[0])
            return ("ok", _cut(v, cut) if v is not None else None)
        if name not in DATA_FUNCTIONS or not hasattr(data, name):
            raise expr.CallableIOError(f"data.{name}() is not available to a Python rule (only {', '.join(sorted(DATA_FUNCTIONS))})")
        tok = data.LOAD_CUTOFF.set(pd.Timestamp(cut) if cut is not None else None)
        try:
            with expr.io_allowed():
                v = getattr(data, name)(*args, **kwargs)
        finally:
            data.LOAD_CUTOFF.reset(tok)
        return ("ok", _cut(v, cut))
    except Exception as e:  # noqa: BLE001 - sent back to the rule, raised there
        return ("err", type(e).__name__, str(e), "")


class Job:
    """One fresh child running one packed function."""

    def __init__(self, payload: bytes, spec: dict, feed=None):
        self.conn = _connect()
        self.cut = None
        self.feed, self.limit = feed, 0      # a streamed rule: its data, and the rows sent so far
        STATS["jobs"] += 1
        self._send(("job", payload, {**spec, "sys_path": [p for p in sys.path if isinstance(p, str)]}))
        self._wait()

    def _send(self, msg):
        self.conn.send_bytes(pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL))

    def _wait(self):
        while True:
            try:
                msg = _loads_safe(self.conn.recv_bytes())
            except (EOFError, OSError) as e:
                raise SandboxError(f"the sealed evaluator for a Python rule stopped unexpectedly ({e!r})") from None
            if msg[0] == "rpc":
                if msg[1] == "memo" and self.feed is not None:
                    key, k = msg[2][0]
                    vals = self.feed.memo(key)
                    # never more rows than the child has been sent
                    self._send(("ok", None if vals is None else vals[: min(int(k), self.limit)].copy()))
                else:
                    self._send(_answer_rpc(msg, self.cut))
                continue
            if msg[0] == "err":
                _raise_child_error(msg)
            return msg[1] if len(msg) == 2 else msg[1:]

    def ask(self, msg, cut):
        self.cut = cut
        self._send(msg)
        return self._wait()

    def close(self):
        try:
            self._send(("end",))
        except Exception:  # noqa: BLE001
            pass
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self):
        c = getattr(self, "conn", None)
        if c is not None:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------- rules on one ticker

_EXTRA_KEYS = ("bars_held", "entry_price", "pnl", "highest_since_entry", "lowest_since_entry")


def _dt_values(idx: pd.DatetimeIndex) -> np.ndarray:
    """datetime64 values (in the index's own unit; UTC wall times for a tz-aware index)."""
    return (idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx).to_numpy()


def _frame_parts(df: pd.DataFrame):
    """(columns, dtypes, float matrix) of a frame: the rows are sent as slices of the matrix."""
    cols = list(df.columns)
    dtypes = [str(df[c].dtype) for c in cols]
    m = np.column_stack([df[c].to_numpy(dtype=float, na_value=np.nan) if df[c].dtype.kind in "fiub"
                         else np.full(len(df), np.nan) for c in cols]) if cols else np.zeros((len(df), 0))
    other = {c: df[c].tolist() for c in cols if df[c].dtype.kind not in "fiub"}
    return cols, dtypes, m, other


class _RuleFeed:
    """The pieces of a ticker's data a child is fed, row block by row block."""

    def __init__(self, ns, extras: dict):
        from . import expr  # noqa: F401
        self.q = _frame_parts(ns.quoted_df)
        self.a = None if ns.df is ns.quoted_df else _frame_parts(ns.df)
        self.idx = ns.df.index
        self.ts = _dt_values(self.idx) if isinstance(self.idx, pd.DatetimeIndex) else None
        self.extras = {k: v.to_numpy(dtype=float, na_value=np.nan) for k, v in extras.items()}
        self.ns = ns
        self._memo: dict = {}
        self._lock = threading.Lock()

    def memo(self, key):
        """The full-history value of a built-in on the ticker (expr.memo_full), computed once per stream; the
        caller cuts it at the rows the child has."""
        from . import expr
        with self._lock:
            if key not in self._memo:
                self._memo[key] = expr.memo_full(self.ns, key) if not self.extras else None
            return self._memo[key]

    def spec(self, ns, kind):
        return {"mode": "rule", "kind": kind, "ticker": ns.ticker, "price_basis": ns.price_basis,
                "month_lookbacks": ns.month_lookbacks, "close_fill": bool(getattr(ns, "close_fill", False)),
                "n": len(self.idx), "q_cols": self.q[:2], "a_cols": self.a[:2] if self.a else None,
                "extras": list(self.extras), "index_name": self.idx.name, "tz": str(getattr(self.idx, "tz", None) or "")}

    def block(self, a: int, b: int):
        """Rows a..b-1."""
        def part(p):
            if p is None:
                return None
            return (p[2][a:b].copy(), {c: v[a:b] for c, v in p[3].items()})
        return (self.ts[a:b].copy() if self.ts is not None else list(self.idx[a:b]), part(self.q), part(self.a),
                {k: v[a:b].copy() for k, v in self.extras.items()})


def _extras_of(ns) -> dict:
    return {k: v for k in _EXTRA_KEYS if isinstance(v := dict.get(ns, k), pd.Series) and len(v) == len(ns.df)}


def _run_chunk(payload, feed: _RuleFeed, spec: dict, pos: np.ndarray, kind: str) -> np.ndarray:
    job = Job(payload, spec, feed)
    try:
        out = np.zeros(len(pos), bool) if kind == "bool" else np.full(len(pos), np.nan)
        have = 0
        for n, i in enumerate(pos):
            i = int(i)
            job.limit = i + 1
            v = job.ask(("step", feed.block(have, i + 1)), feed.idx[i])
            have = i + 1
            out[n] = v
        STATS["steps"] += len(pos)
        return out
    finally:
        job.close()


def _workers(n_bars: int) -> int:
    from . import expr
    if expr.STREAM_WORKERS == 1 or n_bars < expr.STREAM_PARALLEL_MIN:
        return 1
    cpus = expr.STREAM_WORKERS or min(8, os.cpu_count() or 1)
    return max(1, min(cpus, n_bars // (expr.STREAM_PARALLEL_MIN // 2)))


def stream(fn, ns, kind: str, pos: np.ndarray, what: str = "Python rule") -> np.ndarray:
    """fn's answer at each bar position in `pos` (increasing), each computed in a child that holds only the bars up
    to that position. Long streams are split into chunks answered in parallel."""
    idx = ns.df.index
    payload = pack(fn, idx[int(pos[0])] if len(pos) else None, what)
    feed = _RuleFeed(ns, _extras_of(ns))
    spec = feed.spec(ns, kind)
    w = _workers(len(pos))
    if w <= 1:
        return _run_chunk(payload, feed, spec, pos, kind)
    n = len(pos)
    k = max(w * 3, 2)
    cuts = sorted({int(round(n * ((j / k) ** 0.75))) for j in range(k + 1)} | {0, n})
    parts = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    from concurrent.futures import ThreadPoolExecutor
    # each chunk also answers the next chunk's first bar: a function whose answer depends on which earlier bars it
    # was called on (state kept between calls) disagrees there, and is then streamed in one sequential pass
    with ThreadPoolExecutor(w) as ex:
        outs = list(ex.map(lambda ab: _run_chunk(payload, feed, spec, pos[ab[0]: min(ab[1] + 1, n)], kind), parts))
    for (a, b), o, nxt in zip(parts[:-1], outs[:-1], outs[1:]):
        x, y = o[-1], nxt[0]
        if not (x == y or (kind != "bool" and np.isnan(x) and np.isnan(y))):
            return _run_chunk(payload, feed, spec, pos, kind)
    return np.concatenate([o[: b - a] for (a, b), o in zip(parts, outs)])


def call(fn, ns, kind: str, cutoff=None, what: str = "Python rule", payload: bytes | None = None) -> np.ndarray:
    """fn called once on all of ns's bars (a vectorized_causal rule, or the whole-history comparison of a check), in
    a fresh child; its data requests are cut at the last bar. Returns the answers aligned to the bars."""
    idx = ns.df.index
    if payload is None:
        payload = pack(fn, cutoff if cutoff is not None else (idx[0] if len(idx) else None), what)
    feed = _RuleFeed(ns, _extras_of(ns))
    job = Job(payload, {**feed.spec(ns, kind), "mode": "call"})
    try:
        raw = job.ask(("call", feed.block(0, len(idx))), idx[-1] if len(idx) else None)
    finally:
        job.close()
    return np.frombuffer(raw, dtype=bool if kind == "bool" else float).copy()


def call_prefixes(fn, ns, kind: str, cuts, payload: bytes, feed: "_RuleFeed | None" = None) -> dict:
    """fn called on the bars up to (and including) each position in `cuts`, in increasing order, in one fresh child
    that is sent more rows before each call (so at every call it holds nothing after that call's last bar). Returns
    {position: answers array, or the exception the call raised}."""
    idx = ns.df.index
    feed = feed or _RuleFeed(ns, _extras_of(ns))
    job = Job(payload, {**feed.spec(ns, kind), "mode": "call"})
    out: dict = {}
    have = 0
    try:
        for i in sorted({int(x) for x in cuts}):
            try:
                raw = job.ask(("call", feed.block(have, i + 1)), idx[i])
                out[i] = np.frombuffer(raw, dtype=bool if kind == "bool" else float).copy()
            except (SandboxError, LeakError):
                raise
            except Exception as e:  # noqa: BLE001 - e.g. a function that needs more rows than the cut has
                out[i] = e
            have = i + 1
    finally:
        job.close()
    return out


# ---------------------------------------------------------------- portfolio functions

class PortfolioSession:
    """A portfolio function f(date, history) -> {ticker: weight} answered in one child fed day by day: at each call
    the child receives the tickers' rows up to `date` (only the new ones) and nothing later."""

    def __init__(self, fn, frames: dict[str, pd.DataFrame], first_date):
        self.frames = frames
        self.parts = {t: _frame_parts(df) for t, df in frames.items()}
        self.ts = {t: _dt_values(df.index) for t, df in frames.items()}
        self.have = {t: 0 for t in frames}
        self.last = None
        spec = {"mode": "portfolio", "tickers": {t: (p[0], p[1], frames[t].index.name) for t, p in self.parts.items()}}
        self.job = Job(pack(fn, first_date, "portfolio function"), spec)

    def __call__(self, date) -> dict:
        date = pd.Timestamp(date)
        if self.last is not None and date < self.last:
            raise SandboxError("portfolio function sessions only move forward in time")
        self.last = date
        rows = {}
        for t, df in self.frames.items():
            n = int(df.index.searchsorted(date, side="right"))
            a = self.have[t]
            if n > a:
                p = self.parts[t]
                rows[t] = (self.ts[t][a:n].copy(), p[2][a:n].copy(), {c: v[a:n] for c, v in p[3].items()})
                self.have[t] = n
        got = self.job.ask(("pstep", date.value, rows), date)
        return {str(k): float(v) for k, v in got}

    def close(self):
        self.job.close()

    def __del__(self):
        j = getattr(self, "job", None)
        if j is not None:
            j.close()


# ================================================================ the child side

class _Chan:
    conn = None


def _send(msg):
    _Chan.conn.send_bytes(pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL))


def _recv():
    return pickle.loads(_Chan.conn.recv_bytes())


def _rpc(name, *args, **kwargs):
    _send(("rpc", name, args, kwargs))
    msg = _recv()
    if msg[0] == "ok":
        return msg[1]
    _, cls, text, _tb = msg
    from . import expr
    if cls == "CallableIOError":
        raise expr.CallableIOError(text)
    exc = _BUILTIN_EXC.get(cls)
    if cls == "DataError":
        from . import data
        raise data.DataError(text)
    raise (exc(text) if exc is not None and issubclass(exc, Exception) else RuntimeError(f"{cls}: {text}"))


def _memo_request(key, k):
    return _rpc("memo", (key, k))


class _OverrideProxy:
    def get(self, key, default=None):
        v = _rpc("sym_override", key)
        return default if v is None else v


def _harden():
    """Child: data functions forwarded to the backtester (cut at the current day), files / processes / network
    refused for good."""
    from . import data, expr
    expr._install_io_guard()
    expr._IO.blocked = 1
    for name in DATA_FUNCTIONS:
        if hasattr(data, name):
            def proxy(*a, _n=name, **k):
                return _rpc(_n, *a, **k)
            proxy.__name__ = name
            setattr(data, name, proxy)

    def refuse_data(*a, _n="", **k):
        raise expr.CallableIOError("a Python rule may not read the backtester's data files directly")
    for name, v in list(vars(data).items()):
        if callable(v) and getattr(v, "__module__", None) == data.__name__ and name not in DATA_FUNCTIONS \
                and name not in ("canonical", "is_fund", "unknown_ticker_message") and not isinstance(v, type):
            setattr(data, name, refuse_data)
    expr._SYM_OVERRIDE = _OverrideProxy()

    import _io
    import posix

    real_fileio, real_open = _io.FileIO, _io.open

    def code_file(file, mode) -> bool:
        # the import system reads module code through _io.open / _io.FileIO: Python source, bytecode, extensions
        name = os.fsdecode(file) if isinstance(file, (str, bytes, os.PathLike)) else ""
        return (name.endswith((".py", ".pyc", ".so", ".pyd")) and "w" not in mode and "a" not in mode
                and "+" not in mode and not name.startswith("/proc"))

    class FileIO(real_fileio):
        def __init__(self, file, mode="r", *a, **k):
            if not code_file(file, mode):
                expr._refuse("open a file")
            super().__init__(file, mode, *a, **k)

    def open_code_only(file, mode="r", *a, **k):
        if not code_file(file, mode):
            expr._refuse("open a file")
        return real_open(file, mode, *a, **k)
    _io.FileIO = FileIO
    io.FileIO = FileIO
    _io.open = open_code_only

    def no(what):
        def f(*a, **k):
            expr._refuse(what)
        return f
    for mod, names, what in ((posix, ("open", "fork", "forkpty", "execv", "execve", "posix_spawn", "posix_spawnp",
                                      "system", "popen", "openpty"), "start a process or open a file"),
                             (os, ("open", "fork", "forkpty", "execv", "execve", "execl", "execle", "execlp",
                                   "execlpe", "execvp", "execvpe", "posix_spawn", "posix_spawnp", "system", "popen",
                                   "spawnv", "spawnve", "spawnl", "spawnle", "openpty"), "start a process or open a file"),
                             ):
        for n in names:
            if hasattr(mod, n):
                setattr(mod, n, no(what))
    subprocess.Popen = no("start a process")    # type: ignore[assignment]
    try:
        import _posixsubprocess
        _posixsubprocess.fork_exec = no("start a process")
    except ImportError:
        pass


def _frame_from(ts, cols_dtypes, index_name, tz, cols_arr, other) -> pd.DataFrame:
    """A frame from column arrays (cols_arr[j]: column j's values as floats; copied, the rule may change them)."""
    cols, dtypes = cols_dtypes
    if isinstance(ts, np.ndarray):
        idx = pd.DatetimeIndex(ts, name=index_name)
        if tz:
            idx = idx.tz_localize("UTC").tz_convert(tz)
    else:
        idx = pd.Index(ts, name=index_name)
    data = {}
    for j, (c, dt) in enumerate(zip(cols, dtypes)):
        if c in other:
            data[c] = pd.Series(other[c], index=idx, dtype=dt if dt != "object" else object)
            continue
        col = cols_arr[j]
        if dt == "bool":
            data[c] = col != 0
        elif dt == "float64" or (dt.startswith(("int", "uint")) and np.isnan(col).any()):
            data[c] = col.copy()
        else:
            try:
                data[c] = col.astype(dt)
            except (TypeError, ValueError):
                data[c] = col.copy()
    return pd.DataFrame(data, index=idx, columns=cols, copy=False)


class _Buf:
    """Rows received so far, stored column by column (grown in place; nothing later exists in the child)."""

    def __init__(self, ncols: int):
        self.ts = None
        self.idx_list: list = []
        self.m = np.zeros((ncols, 0))
        self.other: dict = {}
        self.n = 0

    def add(self, ts, mat, other):
        k = len(mat)
        if isinstance(ts, np.ndarray):
            if self.ts is None:
                self.ts = np.empty(max(k, 64), dtype=ts.dtype)
            if self.n + k > len(self.ts):
                grown = np.empty(max(64, 2 * (self.n + k)), dtype=self.ts.dtype)
                grown[: self.n] = self.ts[: self.n]
                self.ts = grown
            self.ts[self.n: self.n + k] = ts
        else:
            self.idx_list += list(ts)
        if self.n + k > self.m.shape[1]:
            grown = np.empty((self.m.shape[0], max(64, 2 * (self.n + k))))
            grown[:, : self.n] = self.m[:, : self.n]
            self.m = grown
        if k:
            self.m[:, self.n: self.n + k] = np.asarray(mat, dtype=float).T
        for c, v in other.items():
            self.other.setdefault(c, []).extend(v)
        self.n += k

    def frame(self, cols_dtypes, index_name, tz, is_dt=True) -> pd.DataFrame:
        if is_dt:
            ts = self.ts[: self.n] if self.ts is not None else np.array([], dtype="datetime64[ns]")
        else:
            ts = self.idx_list
        return _frame_from(ts, cols_dtypes, index_name, tz, self.m[:, : self.n],
                           {c: list(v) for c, v in self.other.items()})


def _to_plain(out, kind: str, idx) -> bytes:
    from . import expr
    s = expr._as_bool(out, idx) if kind == "bool" else expr._as_value(out, idx)
    return np.ascontiguousarray(s.to_numpy(dtype=bool if kind == "bool" else float)).tobytes()


def _job(conn, hardened: bool = False):
    """Child: serve one job (see Job)."""
    global IN_CHILD
    IN_CHILD = True
    _Chan.conn = conn
    from . import expr
    tb_text = ""
    try:
        if not hardened:
            _harden()
        msg = _recv()
        _, payload, spec = msg
        for p in reversed(spec.get("sys_path", [])):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            fn = _unpack(payload)
        except Exception as e:  # noqa: BLE001
            raise SandboxError(f"could not rebuild the function in the sealed evaluator: {type(e).__name__}: {e}") \
                from None
        _send(("ready", None))
        mode, kind = spec["mode"], spec.get("kind")
        if mode in ("rule", "call"):
            qb = _Buf(len(spec["q_cols"][0]))
            ab = _Buf(len(spec["a_cols"][0])) if spec["a_cols"] else None
            ex: dict = {k: np.zeros(0) for k in spec["extras"]}
            is_dt = True
        else:
            pb = {t: _Buf(len(v[0])) for t, v in spec["tickers"].items()}
        while True:
            msg = _recv()
            op = msg[0]
            if op == "end":
                break
            try:
                if op in ("step", "call"):
                    ts, q, a, e = msg[1]
                    is_dt = isinstance(ts, np.ndarray)
                    qb.add(ts, q[0], q[1])
                    if ab is not None:
                        ab.add(ts, a[0], a[1])
                    for k, v in e.items():
                        ex[k] = np.concatenate([ex[k], v])
                    quoted = qb.frame(spec["q_cols"], spec["index_name"], spec["tz"], is_dt)
                    df = ab.frame(spec["a_cols"], spec["index_name"], spec["tz"], is_dt) if ab is not None else quoted
                    extra = {k: pd.Series(v.copy(), index=df.index) for k, v in ex.items()} or None
                    ns = expr._SealedNamespace(quoted, df, spec["ticker"], spec["price_basis"],
                                               spec["month_lookbacks"], spec["close_fill"], extra,
                                               memo=_memo_request if op == "step" and extra is None else None)
                    out = fn(ns.df, ns)
                    if op == "step":
                        _send(("val", expr._last(out, kind)))
                    else:
                        _send(("arr", _to_plain(out, kind, ns.df.index)))
                elif op == "pstep":
                    _, date, rows = msg
                    for t, (ts, mat, other) in rows.items():
                        pb[t].add(ts, mat, other)
                    hist = {t: pb[t].frame(spec["tickers"][t][:2], spec["tickers"][t][2], "") for t in pb}
                    w = fn(pd.Timestamp(date), hist) or {}
                    if not isinstance(w, dict):
                        raise TypeError(f"a portfolio function must return a dict {{ticker: weight}}, not {type(w).__name__}")
                    _send(("w", [(str(k), float(v)) for k, v in w.items()]))
                else:
                    raise SandboxError(f"unknown request {op!r}")
            except BaseException as e:  # noqa: BLE001 - reported to the backtester, which raises it
                import traceback
                tb_text = "".join(traceback.format_exception(type(e), e, e.__traceback__)[-6:])
                _send(("err", type(e).__name__, str(e), tb_text))
    except (EOFError, OSError, KeyboardInterrupt):
        pass
    except BaseException as e:  # noqa: BLE001
        try:
            _send(("err", type(e).__name__, str(e), tb_text))
        except Exception:  # noqa: BLE001
            pass


def _warm_up():
    """Fork server: run a typical workload once on made-up bars (no real data), so the lazy imports and first-call
    costs of pandas and the rule language are paid here and not in every child."""
    from . import expr
    n = 300
    idx = pd.bdate_range("2000-01-03", periods=n, name="date")
    c = 100 * np.exp(np.cumsum(np.sin(np.arange(n)) * 0.01))
    df = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6, "dividend": 0.0,
                       "adj_close": c, "quote_close": c, "split": 1.0, "open_ok": True}, index=idx)
    try:
        parts = _frame_parts(df)
        b = _Buf(len(parts[0]))
        b.add(_dt_values(idx), parts[2], parts[3])
        f = b.frame(parts[:2], "date", "")
        ns = expr._SealedNamespace(f, f, "X")
        for rule in ("rsi(2) < 30 and close > sma(200)", "ret(20) > 0 and atr(14) > 0", "ema(close, 9) > wma(close, 9)",
                     "macd() > macd_signal() and cross(close, sma(50))", "down_days >= 3 or ibs < 0.2"):
            expr._last(expr.evaluate(rule, ns), "bool")
        x = f.close
        _ = (x > x.shift(1)) & (x.rolling(10).min() < x) & (x.pct_change(5) > x.ewm(span=5).mean())
        _to_plain(_, "bool", f.index)
        pickle.loads(pickle.dumps(f))
    except Exception:  # noqa: BLE001 - only a speed-up
        pass


IDLE_CHILDREN = 4


def _serve(path: str, key: bytes):
    """The fork server: imports the backtester (no data) and keeps a few forked, hardened children waiting for a
    connection; each serves one job and exits, and is replaced."""
    import json
    import select
    import signal
    import socket
    from multiprocessing.connection import Listener
    extra = json.loads(os.environ.get("BACKTESTER_SANDBOX_SYSPATH", "[]"))
    for p in reversed(extra):
        if p not in sys.path:
            sys.path.insert(0, p)
    from . import expr, data, calendar  # noqa: F401 - preloaded for every child
    try:
        from . import portfolio, engine  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    _warm_up()
    listener = Listener(path, family="AF_UNIX", authkey=key)
    lsock: socket.socket = listener._listener._socket
    lsock.setblocking(False)      # idle children race for each connection: the losers go back to waiting
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)       # children are reaped automatically
    alive_r, alive_w = os.pipe()     # EOF in the children when the server is gone
    taken_r, taken_w = os.pipe()     # a child took a connection: fork a replacement

    def idle_child():
        os.close(alive_w)
        os.close(taken_r)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        try:
            _harden()
            while True:
                r, _, _ = select.select([lsock, alive_r], [], [])
                if alive_r in r:
                    return
                try:
                    conn = listener.accept()
                except (BlockingIOError, InterruptedError):
                    continue            # another child took it
                except Exception:  # noqa: BLE001 - a failed handshake: this child is spent
                    os.write(taken_w, b"x")
                    return
                os.write(taken_w, b"x")
                lsock.close()
                os.close(alive_r)
                _job(conn, hardened=True)
                return
        finally:
            os._exit(0)

    def spawn():
        if os.fork() == 0:
            idle_child()

    def watch():                                         # exit with the backtester
        try:
            sys.stdin.buffer.read()
        finally:
            os._exit(0)
    for _ in range(IDLE_CHILDREN):
        spawn()
    threading.Thread(target=watch, daemon=True).start()
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    os.dup2(2, 1)          # a rule's print() goes to stderr (nobody reads the pipe after "ready")
    while True:
        try:
            got = os.read(taken_r, 64)
        except InterruptedError:
            continue
        if not got:
            break
        for _ in got:
            spawn()


def _stdio_job():
    """Fallback transport: one job over stdin / stdout in this process."""
    r, w = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr           # prints of the rule must not corrupt the channel
    import json
    for p in reversed(json.loads(os.environ.get("BACKTESTER_SANDBOX_SYSPATH", "[]"))):
        if p not in sys.path:
            sys.path.insert(0, p)
    _job(_PipeConn(r, w))


if __name__ == "__main__":
    # run the importable module's functions (not this __main__ copy), so the child's state lives in
    # backtester.sandbox, where the rest of the backtester looks for it
    from backtester import sandbox as _sandbox
    if sys.argv[1] == "serve":
        _sandbox._serve(sys.argv[2], bytes.fromhex(sys.argv[3]))
    else:
        _sandbox._stdio_job()
