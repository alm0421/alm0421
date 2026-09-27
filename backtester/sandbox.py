"""Sealed evaluation of Python-function rules (f(df, ns) -> Series) and portfolio functions (f(date, history) ->
{ticker: weight}) in a separate process that only ever holds the data up to the day being decided.

Why a process: a function run inside the backtester can walk the interpreter (inspect.stack(), gc.get_objects(),
the namespace's own attributes) and find the whole price history the engine holds, or read the price files. So the
function never runs in the backtester's process. It runs in a fresh child process forked from a small "fork server"
that imported pandas / numpy / backtester but never loaded any data:

  - the function is shipped by value (its code, the globals it names, its closure and defaults; functions of
    installed libraries and of the backtester by reference). Everything it carries is measured first (_Scanner,
    with no depth limit: globals, closure cells, defaults, keyword defaults, attributes, the functions it calls,
    held objects / bound methods / partials, its own classes' and modules' attributes, its code's constants) and
    must fit a rule's parameters: at most MAX_ITEMS (64) values in one container / array / pandas object and
    MAX_TOTAL_ITEMS (256) in all, strings of MAX_TEXT (256) characters (MAX_TEXT_TOTAL in all), integers of
    MAX_INT_BITS (64) bits, MAX_CODE_BYTES of bytecode. Anything more - a price history, a list / string / bytes /
    bitmask of later outcomes - is refused with LeakError naming the variable; a pandas object dated past the first
    day the function answers is refused as future data. Every object actually pickled is checked again on its own
    (_Packer). Load data inside the function instead (ns["sym"], data.load, the history argument): those are cut
    at each day.
  - the process starts with an almost empty environment (ENV_KEEP; a variable the script set is not passed), no
    inherited file descriptors but its pipe (and stderr), and an import path of existing folders only; a module
    of the user's own that it imports is measured the same way after it runs (_UserModuleGuard), and its files
    are readable only while the import system loads it.
  - the bars are sent one day at a time: at the moment it answers day D the child has received nothing after D,
    so nothing in its memory (frames, stack, gc) is later than D.
  - data.load / sym() and the other point-in-time data functions (market_cap, tbill_rate, treasury_10y, cape...)
    are forwarded to the backtester, which answers them cut at D; every other data function is refused.
  - file, process and network access is refused at the Python level in the child (open, io / os / posix open,
    io.open_code, FileIO - except the standard library's, installed packages' and the backtester's own files -,
    pandas / numpy readers, sockets, subprocess, os.fork / exec / spawn).
  - each stream (one function on one ticker) or call gets its own fresh child, so state a function keeps
    (a global, a default dict) starts empty and only ever saw the days up to the current one.

What is not prevented: native code (ctypes, a C extension) or anything else below Python can read what the
operating system lets the process read (the price files, other processes' memory); module code of the user's own
can read its own source while it is imported. The limits bound how much a function carries; they cannot stop
knowledge typed into a rule by hand. No ordinary way of writing a rule reaches later data.

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

# What a function may carry by value (see _Scanner): enough for a rule's parameters (a few numbers, a small lookup
# table, a list of tickers), not enough for data (a price history, a list of later up / down days, a bitmask).
MAX_ITEMS = 64            # values in one container / array / pandas object (a small lookup table fits)
MAX_TOTAL_ITEMS = 256     # values in all: numbers / strings it holds, container entries, array / pandas elements
MAX_TEXT = 256            # characters / bytes in one string
MAX_TEXT_TOTAL = 2048     # characters / bytes in all its strings
MAX_INT_BITS = 64         # bits in one integer
MAX_CODE_BYTES = 32_768   # bytecode of the function and the functions it calls (shipped by value)
MAX_CODE_CONSTS = 2_000   # constants in that code
MAX_CODE_TEXT = 8_192     # characters in the code's string literals
MAX_PAYLOAD = 1_000_000   # bytes in the whole packed function (a backstop)
IN_CHILD = False          # True inside a sandbox child (a rule calling evaluate() there runs in-process)
STATS: dict = {"jobs": 0, "steps": 0}


class LeakError(ValueError):
    """A Python rule carries data captured outside the run (would let it read later prices)."""


class SandboxError(RuntimeError):
    pass


# ================================================================ packing a function by value

_PKG_DIR = os.path.dirname(os.path.realpath(__file__))


def _lib_roots() -> tuple[str, ...]:
    """Where the standard library and installed packages live (never a folder holding the user's own code: a root
    containing the working directory or the backtester's checkout is left out)."""
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
    mine = [os.path.realpath(os.getcwd()) + os.sep, os.path.dirname(_PKG_DIR) + os.sep]
    out = []
    for r in roots:
        if not r:
            continue
        r = os.path.realpath(r) + os.sep
        if not any(m.startswith(r) for m in mine):
            out.append(r)
    return tuple(out)


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
    f = os.path.realpath(f)
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


class _Dated(LeakError):
    """A captured pandas object / date array reaching past the cutoff (certainly later data)."""


class _TooMuch(LeakError):
    """More captured information than a rule's parameters need (could be data, e.g. a list of later outcomes)."""


def _dated(obj, cutoff) -> str | None:
    """A pandas object / datetime array whose dates reach past `cutoff`: a description, or None."""
    if isinstance(obj, (pd.Series, pd.DataFrame, pd.Index)):
        idx = obj if isinstance(obj, pd.Index) else obj.index
        if isinstance(idx, pd.DatetimeIndex) and len(idx) and cutoff is not None:
            last = idx.max()
            try:
                past = last > pd.Timestamp(cutoff)
            except TypeError:        # tz-aware vs naive
                past = True
            if past:
                return f"a {type(obj).__name__} of {len(obj):,} rows dated up to {last.date()}"
        return None
    if isinstance(obj, np.ndarray) and obj.dtype.kind == "M" and obj.size and cutoff is not None:
        if obj.max() > np.datetime64(pd.Timestamp(cutoff)):
            return f"an array of {obj.size:,} dates up to {pd.Timestamp(obj.max()).date()}"
    return None


def _is_docstring_code(code) -> bool:
    """Whether co_consts[0] of `code` is its docstring (a def's code; not a comprehension / class body)."""
    if sys.version_info >= (3, 14):
        return bool(code.co_flags & 0x4000000)          # CO_HAS_DOCSTRING
    return bool(code.co_consts) and isinstance(code.co_consts[0], str) and \
        (not code.co_name.startswith("<") or code.co_name == "<lambda>") and bool(code.co_flags & 0x2)  # NEWLOCALS


def _ship_code(code):
    """`code` with docstrings longer than MAX_TEXT dropped (documentation is not shipped: it could carry data)."""
    consts = list(code.co_consts)
    changed = False
    for i, c in enumerate(consts):
        if isinstance(c, types.CodeType):
            n = _ship_code(c)
            if n is not c:
                consts[i], changed = n, True
    if _is_docstring_code(code) and len(consts[0]) > MAX_TEXT:
        consts[0], changed = None, True
    return code.replace(co_consts=tuple(consts)) if changed else code


def limits_text() -> str:
    return (f"a Python rule or portfolio function may carry at most {MAX_ITEMS} values in one list / tuple / dict / "
            f"array / pandas object and {MAX_TOTAL_ITEMS} in all, strings of up to {MAX_TEXT} characters "
            f"({MAX_TEXT_TOTAL:,} in all) and integers of up to {MAX_INT_BITS} bits; its code at most "
            f"{MAX_CODE_BYTES // 1024} kB of bytecode, {MAX_CODE_CONSTS:,} constants and literals of up to "
            f"{MAX_ITEMS} values / {MAX_TEXT} characters")


class _Scanner:
    """Measures everything a function carries by value against one budget and raises LeakError naming the first
    variable over it: the globals its code names, its closure cells, defaults, keyword defaults and attributes, and
    the same recursively for the functions it calls; the objects, bound methods and partials it holds (what pickle
    would ship for them); user classes' attributes; user modules' attributes (modules outside the standard library,
    installed packages and the backtester, named as globals or imported in the code); and the constants and size of
    the code itself. Library modules, functions and classes are shipped by reference and carry nothing."""

    def __init__(self, cutoff):
        self.cutoff = cutoff
        self.items = 0
        self.text = 0
        self.consts = 0
        self.ctext = 0
        self.cbytes = 0
        self.seen: set = set()
        self.keep: list = []           # keeps scanned temporaries alive (so their ids stay unique)

    # ---- budget
    def _items(self, n: int, label: str, desc: str):
        if n > MAX_ITEMS:
            raise _TooMuch(f"{label} is {desc}")
        self.items += n
        if self.items > MAX_TOTAL_ITEMS:
            raise _TooMuch(f"{label} is {desc}, which brings what it carries to over {MAX_TOTAL_ITEMS} values")

    def _text(self, n: int, label: str, kind: str):
        if n > MAX_TEXT:
            raise _TooMuch(f"{label} is a {kind} of {n:,} characters")
        self.text += n
        if self.text > MAX_TEXT_TOTAL:
            raise _TooMuch(f"{label} is a {kind} of {n:,} characters, which takes the text it carries past "
                           f"{MAX_TEXT_TOTAL:,} characters")

    @staticmethod
    def _int_bits(v, label: str):
        b = abs(int(v)).bit_length()
        if b > MAX_INT_BITS:
            raise _TooMuch(f"{label} is an integer of {b:,} bits")

    # ---- code
    def _code(self, code, owner: str):
        if id(code) in self.seen:
            return
        self.seen.add(id(code))
        self.keep.append(code)
        self.cbytes += len(code.co_code)
        if self.cbytes > MAX_CODE_BYTES:
            raise _TooMuch(f"the code of {owner}() (with the functions it calls) is over {MAX_CODE_BYTES:,} bytes of "
                           "bytecode")
        doc = _is_docstring_code(code)
        todo = [c for i, c in enumerate(code.co_consts) if not (i == 0 and doc)]
        while todo:
            c = todo.pop()
            if isinstance(c, types.CodeType):
                self._code(c, owner)
                continue
            self.consts += 1
            if self.consts > MAX_CODE_CONSTS:
                raise _TooMuch(f"the code of {owner}() holds over {MAX_CODE_CONSTS:,} constants")
            if isinstance(c, bool) or c is None:
                continue
            if isinstance(c, int):
                self._int_bits(c, f"a constant in the code of {owner}()")
            elif isinstance(c, (str, bytes)):
                if len(c) > MAX_TEXT:
                    raise _TooMuch(f"a literal in the code of {owner}() is a {type(c).__name__} of {len(c):,} "
                                   "characters")
                self.ctext += len(c)
                if self.ctext > MAX_CODE_TEXT:
                    raise _TooMuch(f"the literals in the code of {owner}() hold over {MAX_CODE_TEXT:,} characters")
            elif isinstance(c, (tuple, frozenset)):
                if len(c) > MAX_ITEMS:
                    raise _TooMuch(f"a literal in the code of {owner}() has {len(c):,} values")
                todo.extend(c)

    # ---- values
    def walk(self, root, label: str):
        stack = [(root, label, True)]
        while stack:
            obj, lab, top = stack.pop()
            stack.extend(reversed(self._visit(obj, lab, top) or ()))

    def _visit(self, obj, lab: str, top: bool):
        if obj is None or isinstance(obj, (bool, np.bool_)) or obj is Ellipsis or obj is NotImplemented:
            return None
        if id(obj) in self.seen:
            return None
        self.seen.add(id(obj))
        self.keep.append(obj)
        if isinstance(obj, (int, np.integer)):
            self._int_bits(obj, lab)
            if top:
                self._items(1, lab, "a number")
            return None
        if isinstance(obj, (float, complex, np.floating, np.complexfloating)):
            if top:
                self._items(1, lab, "a number")
            return None
        if isinstance(obj, (str, bytes, bytearray, memoryview)):
            n = obj.nbytes if isinstance(obj, memoryview) else len(obj)
            self._text(n, lab, type(obj).__name__)
            if top:
                self._items(1, lab, f"a {type(obj).__name__}")
            return None
        if isinstance(obj, (pd.Series, pd.DataFrame, pd.Index)):
            d = _dated(obj, self.cutoff)
            if d:
                raise _Dated(f"{lab} is {d}")
            self._items(int(obj.size), lab, f"a {type(obj).__name__} of {int(obj.size):,} values")
            kids = []
            if isinstance(obj, (pd.Series, pd.Index)) and obj.dtype == object:
                kids += [(v, lab, False) for v in list(obj)]
            if isinstance(obj, (pd.Series, pd.DataFrame)) and obj.index.dtype == object:
                kids.append((obj.index, f"the index of {lab}", False))       # its text (its length is counted)
            if isinstance(obj, pd.DataFrame):
                kids.append((obj.columns, f"the columns of {lab}", False))
            return kids
        if isinstance(obj, np.ndarray):
            d = _dated(obj, self.cutoff)
            if d:
                raise _Dated(f"{lab} is {d}")
            self._items(int(obj.size), lab, f"an array of {obj.size:,} values")
            if obj.dtype == object:
                return [(v, lab, False) for v in obj.ravel().tolist()]
            return None
        if isinstance(obj, np.generic):
            return None
        if isinstance(obj, types.ModuleType):
            if _library_module(obj.__name__):
                return None
            if obj.__name__ in ("__main__", "__mp_main__"):
                raise _TooMuch(f"{lab} is the script's __main__ module (pass values, not the module)")
            return [(v, f"attribute {k!r} of module {obj.__name__}", True) for k, v in list(vars(obj).items())
                    if not (k.startswith("__") and k.endswith("__"))]
        if isinstance(obj, types.FunctionType):
            if not _by_value(obj):
                return None
            nm = obj.__name__
            self._code(obj.__code__, nm)
            st = _fn_state(obj)
            used = f" (used by {nm}())" if not lab.startswith(f"{nm}(") else ""
            kids = [(v, f"global {n!r}{used}", True) for n, v in st["globals"].items()]
            kids += [(v, f"variable {n!r} captured by {nm}()", True)
                     for n, v in zip(obj.__code__.co_freevars, st["cells"]) if not (isinstance(v, str) and v == _EMPTY)]
            kids += [(v, f"a default argument of {nm}()", True) for v in (st["defaults"] or ())]
            kids += [(v, f"default argument {k!r} of {nm}()", True) for k, v in (st["kwdefaults"] or {}).items()]
            kids += [(v, f"attribute {k!r} of {nm}()", True) for k, v in st["dict"].items()]
            for n in sorted(_code_names(obj.__code__)):          # `import mymodule` inside the function
                m = sys.modules.get(n)
                if isinstance(m, types.ModuleType) and not _library_module(n):
                    kids.append((m, f"module {n} (imported by {nm}())", True))
            return kids
        if isinstance(obj, types.CodeType):
            self._code(obj, lab)
            return None
        if isinstance(obj, types.MethodType):
            return [(obj.__func__, lab, True), (obj.__self__, f"the object {lab} belongs to", True)]
        if isinstance(obj, (types.BuiltinFunctionType, types.MethodWrapperType)):
            s = getattr(obj, "__self__", None)
            if s is None or isinstance(s, types.ModuleType):
                return None
            return [(s, f"the object {lab} is bound to", True)]
        if isinstance(obj, (types.WrapperDescriptorType, types.MethodDescriptorType, types.GetSetDescriptorType,
                            types.MemberDescriptorType, types.ClassMethodDescriptorType)):
            return None
        import functools
        if isinstance(obj, functools.partial):
            self._items(len(obj.args) + len(obj.keywords or {}), lab, "a partial")
            return [(v, lab, False) for v in (obj.func, *obj.args, *(obj.keywords or {}).values())]
        if isinstance(obj, (staticmethod, classmethod)):
            return [(obj.__func__, lab, True)]
        if isinstance(obj, property):
            return [(f, lab, True) for f in (obj.fget, obj.fset, obj.fdel) if f is not None]
        if isinstance(obj, dict):
            self._items(len(obj), lab, f"a {type(obj).__name__} of {len(obj):,} entries")
            kids = []
            for k, v in list(obj.items()):
                kids.append((k, lab, False))
                kids.append((v, f"{lab}[{k!r}]" if isinstance(k, (str, int)) and len(repr(k)) < 40 else lab, False))
            fac = getattr(obj, "default_factory", None)
            if fac is not None:
                kids.append((fac, lab, True))
            return kids
        if isinstance(obj, (list, tuple, set, frozenset)):
            self._items(len(obj), lab, f"a {type(obj).__name__} of {len(obj):,} values")
            return [(v, lab, False) for v in list(obj)]
        if isinstance(obj, type):
            if _library_module(getattr(obj, "__module__", None)):
                return None
            return [(v, f"class attribute {obj.__name__}.{k}", True) for k, v in list(vars(obj).items())
                    if not (k.startswith("__") and k.endswith("__")) or k in ("__call__", "__init__", "__slots__")]
        # any other object: what pickle would ship for it (its class, constructor arguments, state, items)
        try:
            rv = obj.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
        except Exception:  # noqa: BLE001 - it will not pickle either (reported when it is packed)
            return None
        if not isinstance(rv, tuple):
            return None
        import itertools
        kids = [(type(obj), lab, True)]
        if len(rv) > 1 and isinstance(rv[1], tuple):
            self._items(len(rv[1]), lab, f"a {type(obj).__name__}")
            kids += [(v, lab, False) for v in rv[1]]
        if len(rv) > 2 and rv[2] is not None:
            kids.append((rv[2], f"{lab} (a {type(obj).__name__})", False))
        if len(rv) > 3 and rv[3] is not None:
            got = list(itertools.islice(rv[3], MAX_ITEMS + 1))
            self._items(len(got), lab, f"a {type(obj).__name__} of {len(got):,}+ values")
            kids += [(v, lab, False) for v in got]
        if len(rv) > 4 and rv[4] is not None:
            got = list(itertools.islice(rv[4], MAX_ITEMS + 1))
            self._items(len(got), lab, f"a {type(obj).__name__} of {len(got):,}+ entries")
            for k, v in got:
                kids += [(k, lab, False), (v, lab, False)]
        return kids


class _Packer(pickle.Pickler):
    """Pickles a function by value. Backstop behind _Scanner: every object actually shipped is checked on its own (a
    container / array of more than MAX_ITEMS values, a string over MAX_TEXT characters, an integer over MAX_INT_BITS
    bits, a pandas object dated past the cutoff), except the packer's own records (code bytes, function state)."""

    def __init__(self, f, cutoff):
        super().__init__(f, protocol=pickle.HIGHEST_PROTOCOL)
        self.cutoff = cutoff
        self.internal: set = set()
        self.keep: list = []

    def _mine(self, *objs):
        for o in objs:
            self.internal.add(id(o))
            self.keep.append(o)

    def persistent_id(self, obj):
        if id(obj) in self.internal:
            return None
        d = _dated(obj, self.cutoff)
        if d:
            raise _Dated(f"it carries {d}")
        if isinstance(obj, (pd.Series, pd.DataFrame, pd.Index, np.ndarray)):
            if obj.size > MAX_ITEMS:
                raise _TooMuch(f"it carries a {type(obj).__name__} of {obj.size:,} values")
            if isinstance(obj, np.ndarray):
                return None
            # judged as a whole: its own arrays are not judged again
            return ("pandas", pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        if isinstance(obj, (list, tuple, set, frozenset, dict)) and len(obj) > MAX_ITEMS:
            raise _TooMuch(f"it carries a {type(obj).__name__} of {len(obj):,} values")
        if isinstance(obj, (str, bytes, bytearray)) and len(obj) > MAX_TEXT:
            raise _TooMuch(f"it carries a {type(obj).__name__} of {len(obj):,} characters")
        if isinstance(obj, int) and not isinstance(obj, bool) and abs(obj).bit_length() > MAX_INT_BITS:
            raise _TooMuch(f"it carries an integer of {abs(obj).bit_length():,} bits")
        return None

    def reducer_override(self, obj):
        if isinstance(obj, types.FunctionType) and _by_value(obj):
            code = marshal.dumps(_ship_code(obj.__code__))
            st = _fn_state(obj)
            if isinstance(st["doc"], str) and len(st["doc"]) > MAX_TEXT:
                st["doc"] = None
            args = (code, obj.__name__, obj.__qualname__, getattr(obj, "__module__", None) or "__main__",
                    len(obj.__code__.co_freevars), id(obj.__globals__))
            self._mine(code, args, st, st["globals"], st["cells"], *args[1:4])
            return (_make_function, args, st, None, None, _set_function_state)
        if isinstance(obj, types.ModuleType):
            if not _library_module(obj.__name__) and obj.__name__ in ("__main__", "__mp_main__"):
                raise _TooMuch("it uses the script's __main__ module as a value")
            return (_import, (obj.__name__,))
        return NotImplemented


class _Unpacker(pickle.Unpickler):
    def persistent_load(self, pid):
        if isinstance(pid, tuple) and pid and pid[0] == "pandas":
            return pickle.loads(pid[1])
        raise pickle.UnpicklingError(f"unknown persistent id {pid!r}")


def _unpack(payload: bytes):
    return _Unpacker(io.BytesIO(payload)).load()


_ADVICE = ("A function only gets the data up to each day; data loaded or computed before the run (e.g. FULL = "
           "data.load('SPY') at module level, a list of later outcomes, a closure over full price series) would let "
           "it read later prices. Load it inside the function instead: ns['sym']('SPY') or data.load('SPY') in a "
           "rule, the `history` argument in a portfolio function - both are cut at each day.")


def leak_message(what: str, name: str | None, e: LeakError, cutoff=None) -> str:
    who = f"the {what} {name}()" if name else f"the {what}"
    past = f" and reaching past {pd.Timestamp(cutoff).date()}" if cutoff is not None else ""
    if isinstance(e, _Dated):
        return f"Lookahead: {who} uses future data: {e}, captured outside the run{past}. {_ADVICE}"
    return (f"Lookahead: {who} may use future data: {e}, captured outside the run. To keep data out, "
            f"{limits_text()}. {_ADVICE}")


def pack(fn, cutoff=None, what: str = "Python rule") -> bytes:
    """`fn` by value, refused (LeakError) when it carries data captured outside the run: more than a rule's
    parameters need (see _Scanner and the MAX_ limits), or a pandas object dated past `cutoff`."""
    if not callable(fn):
        raise TypeError(f"{what} must be a function")
    name = getattr(fn, "__name__", "") or "(function)"
    try:
        _Scanner(cutoff).walk(fn, f"{name}()")
        buf = io.BytesIO()
        _Packer(buf, cutoff).dump(fn)
        if buf.tell() > MAX_PAYLOAD:
            raise _TooMuch(f"it packs into {buf.tell():,} bytes")
    except LeakError as e:
        raise LeakError(leak_message(what, name, e, cutoff)) from None
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


# the only environment variables a sandbox process starts with (the rest of the backtester's environment - which a
# script could fill with data: os.environ["X"] = ... - never reaches it); each at most 4,096 characters
ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR", "SYSTEMROOT", "PYTHONHOME",
            "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "VIRTUAL_ENV")


def _sys_path() -> list[str]:
    """The backtester's import path as the child gets it: existing directories and archives only (no other text)."""
    return [p for p in sys.path if isinstance(p, str) and len(p) <= 4096 and (p == "" or os.path.exists(p))]


def _child_env() -> dict:
    env = {k: v for k in ENV_KEEP if (v := os.environ.get(k)) is not None and len(v) <= 4096}
    root = os.path.dirname(_PKG_DIR)
    extra = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and os.path.isdir(p)]
    env["PYTHONPATH"] = os.pathsep.join([root] + extra)
    import json
    env["BACKTESTER_SANDBOX_SYSPATH"] = json.dumps(_sys_path())
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
                                cwd=os.getcwd(), close_fds=True)
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
                                stdout=subprocess.PIPE, env=_child_env(), cwd=os.getcwd(), close_fds=True)
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


# data helpers that read no files (usable as they are inside the sealed process)
PURE_DATA_FUNCTIONS = {"canonical", "stale_from", "stale_note", "not_investable"}


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

    def __init__(self, payload: bytes, spec: dict, feed=None, cut=None):
        self.conn = _connect()
        self.cut = cut           # data requests while the function is rebuilt (a user module's import) are cut too
        self.feed, self.limit = feed, 0      # a streamed rule: its data, and the rows sent so far
        STATS["jobs"] += 1
        self._send(("job", payload, {**spec, "sys_path": _sys_path(),
                                     "cutoff": None if cut is None else pd.Timestamp(cut).value}))
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
    job = Job(payload, spec, feed, cut=feed.idx[int(pos[0])] if len(pos) else None)
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
    first = cutoff if cutoff is not None else (idx[0] if len(idx) else None)
    job = Job(payload, {**feed.spec(ns, kind), "mode": "call"}, cut=first)
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
    first = min((int(x) for x in cuts), default=None)
    job = Job(payload, {**feed.spec(ns, kind), "mode": "call"}, cut=idx[first] if first is not None else None)
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
        self.job = Job(pack(fn, first_date, "portfolio function"), {**spec, "what": "portfolio function"},
                       cut=first_date)

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


_CHILD: dict = {"cutoff": None, "what": "Python rule"}
_IMPORTING: set = set()      # (child) files of the user module the import system is loading right now


def _read_roots() -> tuple[str, ...]:
    global _LIB
    if _LIB is None:
        _LIB = _lib_roots()
    return _LIB + (_PKG_DIR + os.sep,)


def _may_read(file, mode) -> bool:
    """(child) Files a sealed process may read: the code and data of the standard library, installed packages and
    the backtester package, and the source / bytecode of a user module while the import system loads it (then
    measured, see _UserModuleGuard). Nothing else (no data files, no other source files as text)."""
    if not isinstance(file, (str, bytes, os.PathLike)) or any(c in str(mode) for c in "wax+"):
        return False
    path = os.path.realpath(os.fsdecode(file))
    return path.startswith(_read_roots()) or path in _IMPORTING


class _GuardedLoader:
    """(child) Loads a user module with its own files readable, then measures it like a captured value."""

    def __init__(self, loader, spec):
        self._loader = loader
        self._paths = {os.path.realpath(p) for p in (spec.origin, spec.cached) if p}

    def create_module(self, spec):
        return self._loader.create_module(spec) if hasattr(self._loader, "create_module") else None

    def exec_module(self, module):
        new = self._paths - _IMPORTING
        _IMPORTING.update(new)
        try:
            self._loader.exec_module(module)
        finally:
            _IMPORTING.difference_update(new)
        try:
            _Scanner(_CHILD["cutoff"]).walk(module, f"module {module.__name__}")
        except LeakError as e:
            raise LeakError(leak_message(_CHILD["what"], None, e, _CHILD["cutoff"]) + " (A module of your own that "
                            "a function imports is measured the same way when the sealed evaluator imports it.)") \
                from None

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _UserModuleGuard:
    """(child) A meta path finder in front of the others: a module imported from outside the library roots (a
    module of your own, found through sys.path) is loaded by _GuardedLoader."""

    def find_spec(self, name, path=None, target=None):
        spec = None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None or not spec.origin or not spec.has_location:
            return spec
        if os.path.realpath(spec.origin).startswith(_read_roots()):
            return spec
        spec.loader = _GuardedLoader(spec.loader, spec)
        return spec


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
                and name not in PURE_DATA_FUNCTIONS and not isinstance(v, type):
            setattr(data, name, refuse_data)
    expr._SYM_OVERRIDE = _OverrideProxy()

    # the environment: only the basics (the backtester already starts the server with ENV_KEEP; this also covers
    # the one-process fallback)
    keep = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ") if k in os.environ}
    os.environ.clear()
    os.environ.update(keep)
    sys.dont_write_bytecode = True

    import _io
    import posix

    real_fileio, real_open, real_open_code = _io.FileIO, _io.open, _io.open_code

    class FileIO(real_fileio):
        def __init__(self, file, mode="r", *a, **k):
            if not _may_read(file, mode):
                expr._refuse("open a file")
            super().__init__(file, mode, *a, **k)

    def open_readable(file, mode="r", *a, **k):
        if not _may_read(file, mode):
            expr._refuse("open a file")
        return real_open(file, mode, *a, **k)

    def open_code(path):
        if not _may_read(path, "rb"):
            expr._refuse("open a file")
        return real_open_code(path)
    _io.FileIO = FileIO
    io.FileIO = FileIO
    _io.open = open_readable
    _io.open_code = open_code
    io.open_code = open_code
    sys.meta_path.insert(0, _UserModuleGuard())

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
        global _LIB
        _LIB = None
        _CHILD["cutoff"] = pd.Timestamp(spec["cutoff"]) if spec.get("cutoff") is not None else None
        _CHILD["what"] = spec.get("what", "Python rule")
        try:
            fn = _unpack(payload)
        except LeakError:
            raise
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

    null_fd = os.open(os.devnull, os.O_RDONLY)

    def idle_child():
        os.close(alive_w)
        os.close(taken_r)
        os.dup2(null_fd, 0)          # nothing of the server's is readable: stdin is /dev/null
        os.close(null_fd)
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
