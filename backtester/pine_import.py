"""TradingView Pine Script strategies -> a Strategy spec (a practical subset of Pine v4-v6).

Supported:
  strategy(...)            initial_capital, default_qty_type / default_qty_value, commission_type / commission_value,
                           pyramiding, process_orders_on_close, margin_long (the other display settings are ignored)
  input.*() / input()      the default value (inputs are constants of the backtest)
  x = <expression>         variables of ta.* / math.* / request.security expressions, conditions, numbers; used
                           anywhere later (with history: x[1])
  cond ? a : b             the ternary operator (-> where(cond, a, b))
  f(x) => <expression>     one-line custom functions, inlined where they are called
  var x = <init>           state kept from bar to bar, updated with x := ... / x += ... (at the top level or inside
                           ifs): computed bar by bar from the bars up to each bar (causal), then read like any series
  [a, b, c] = ta.macd(...) tuple results of ta.macd, ta.bb, ta.supertrend, ta.dmi, ta.kc
  strategy.entry(id, strategy.long / strategy.short, when=cond, limit= / stop=)   or inside `if cond` (nested ifs,
                           else); limit / stop entries are working orders that stay until filled or replaced
  strategy.close(id, when=...), strategy.close_all(...)             a rule exit
  strategy.exit(id, from_entry, stop=, limit=, profit=, loss=, trail_price=, trail_points=, trail_offset=, qty_percent=)
                           stops and targets as a percentage of strategy.position_avg_price, a multiple of ta.atr (the
                           current ATR on every bar, as TradingView re-evaluates strategy.exit), a price expression
                           (re-evaluated on every bar), or ticks (syminfo.mintick = $0.01 for US stocks and ETFs);
                           trailing stops by ATR, percent or ticks with an activation level; partial exits
                           (qty_percent) as scale-outs, with a stop that covers only the part it is attached to
  strategy.position_size   position checks that the backtest already applies (flat before an entry, in a position
                           before an exit)
  request.security(syminfo.tickerid, "W" / "M" / "D", x)           weekly(x) / monthly(x) / x (lookahead off)
  time >= timestamp(...)   a date filter becomes the start (end) of the test
Display-only calls (plot, bgcolor, alertcondition, label.new...) are skipped. Anything else - varip, loops, multi-line
functions, strategy.order / strategy.cancel, state that depends on the position - is refused with its line number,
never guessed. The result runs in TradingView-compatible mode.
"""
from __future__ import annotations

import ast
import copy
import math
import re
import threading

import numpy as np

import pandas as pd

from . import data
from .parser import ParseError
from .strategy import Strategy


class PineImportError(ParseError):
    pass


_TLS = threading.local()        # .funcs: the script's one-line functions while it is translated

MINTICK = 0.01                  # TradingView's tick size for US stocks and ETFs (syminfo.mintick)


def looks_like_pine(text) -> bool:
    """A pasted Pine script: starts with //@version, or declares strategy(...) and calls strategy.entry."""
    if not isinstance(text, str):
        return False
    t = text.lstrip()
    return t.startswith("//@version") or bool(re.search(r"(?m)^\s*strategy\s*\(", t) and "strategy.entry" in t)


DISPLAY = {"plot", "plotshape", "plotchar", "plotarrow", "plotcandle", "plotbar", "bgcolor", "barcolor", "fill", "hline",
           "alertcondition", "alert", "label.new", "line.new", "box.new", "table.new", "table.cell", "log.info",
           "log.warning", "log.error", "label.delete", "line.delete", "box.delete", "runtime.error"}
IGNORED_SETTINGS = {"title", "shorttitle", "overlay", "format", "precision", "scale", "max_bars_back", "currency",
                    "calc_on_every_tick", "max_lines_count", "max_labels_count", "max_boxes_count", "max_polylines_count",
                    "explicit_plot_zorder", "calc_bars_count", "risk_free_rate", "fill_orders_on_standard_ohlc",
                    "dynamic_requests", "behind_chart", "close_entries_rule", "margin_short"}
TUPLES = {
    "macd": lambda a: [f"macd({a[1]}, {a[2]})" if a[0] == "close" else f"macd({a[1]}, {a[2]}, {a[0]})",
                       f"macd_signal({a[1]}, {a[2]}, {a[3]})", f"macd_hist({a[1]}, {a[2]}, {a[3]})"],
    "bb": lambda a: [f"sma({a[0]}, {a[1]})", f"(sma({a[0]}, {a[1]}) + {a[2]} * stdev({a[0]}, {a[1]}))",
                     f"(sma({a[0]}, {a[1]}) - {a[2]} * stdev({a[0]}, {a[1]}))"],
    "supertrend": lambda a: [f"supertrend({a[1]}, {a[0]})", f"supertrend_dir({a[1]}, {a[0]})"],
    "dmi": lambda a: [f"plus_di({a[0]})", f"minus_di({a[0]})", f"adx({a[1]})"],
    "kc": lambda a: [f"ema({a[0]}, {a[1]})", f"keltner_upper({a[1]}, {a[2]})", f"keltner_lower({a[1]}, {a[2]})"],
}
TUPLE_ARGS = {"macd": 4, "bb": 3, "supertrend": 2, "dmi": 2, "kc": 3}


# ------------------------------------------------------------------ lexing

def _strip_comment(line: str) -> tuple[str, str]:
    """(code, comment) of one line: // outside a string starts a comment."""
    q = None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch == "/" and line[i:i + 2] == "//":
            return line[:i], line[i + 2:]
    return line, ""


def _depth(s: str) -> int:
    d, q = 0, None
    for ch in s:
        if q:
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch in "([":
            d += 1
        elif ch in ")]":
            d -= 1
    return d


_CONT_END = re.compile(r"(?:[-+*/%,=<>?:(\[]|\band|\bor|\bnot)\s*$")


def _statements(src: str) -> tuple[list[tuple[int, int, str]], list[tuple[int, str]]]:
    """[(line number, indent, code)] with continuation lines joined, and [(line number, comment)]."""
    lines = src.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").split("\n")
    out, comments = [], []
    buf = None
    for n, raw in enumerate(lines, 1):
        code, com = _strip_comment(raw)
        if com.strip():
            comments.append((n, com.strip()))
        if not code.strip():
            continue
        ind = len(code) - len(code.lstrip(" "))
        if buf is not None:
            cont = _depth(buf[2]) > 0 or _CONT_END.search(buf[2]) or (ind > buf[1] and ind % 4 != 0)
            if cont:
                buf = (buf[0], buf[1], buf[2].rstrip() + " " + code.strip())
                continue
            out.append(buf)
        buf = (n, ind, code.rstrip())
    if buf is not None:
        out.append(buf)
    return out, comments


# ------------------------------------------------------------------ expressions

def _refuse(line: int, msg: str):
    raise PineImportError(f"Pine script line {line}: {msg}")


def _close_of(t: str, i: int) -> int:
    """The index of the bracket closing the one opened at t[i] (strings skipped), or -1."""
    d, q = 0, None
    for j in range(i, len(t)):
        ch = t[j]
        if q:
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch in "([":
            d += 1
        elif ch in ")]":
            d -= 1
            if d == 0:
                return j
    return -1


def _split_top(t: str, sep: str) -> list[str]:
    """t split at the top-level (outside brackets and strings) occurrences of the one-character sep."""
    out, d, q, last = [], 0, None, 0
    for j, ch in enumerate(t):
        if q:
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch in "([":
            d += 1
        elif ch in ")]":
            d -= 1
        elif ch == sep and d == 0:
            out.append(t[last:j])
            last = j + 1
    out.append(t[last:])
    return out


def _ternary(t: str, line: int) -> str:
    """Pine's cond ? a : b (right-associative, below `or`) as where(cond, a, b), at every bracket depth."""
    if "?" not in t:
        return t
    out, j = [], 0
    while j < len(t):
        ch = t[j]
        if ch in "\"'":
            k = t.find(ch, j + 1)
            k = len(t) - 1 if k < 0 else k
            out.append(t[j:k + 1])
            j = k + 1
            continue
        if ch in "([":
            k = _close_of(t, j)
            if k < 0:
                _refuse(line, f"unbalanced brackets in {t.strip()!r}.")
            inner = ",".join(_ternary(x, line) for x in _split_top(t[j + 1:k], ","))
            out.append(ch + inner + t[k])
            j = k + 1
            continue
        out.append(ch)
        j += 1
    s = "".join(out)
    parts = _split_top(s, "?")
    if len(parts) == 1:
        return s
    cond, rest = parts[0], "?".join(parts[1:])
    # the ':' that belongs to this '?': the first top-level one with as many '?' as ':' before it
    d, q, depth = 0, None, 0
    for k, ch in enumerate(rest):
        if q:
            if ch == q:
                q = None
            continue
        if ch in "\"'":
            q = ch
        elif ch in "([":
            d += 1
        elif ch in ")]":
            d -= 1
        elif d == 0 and ch == "?":
            depth += 1
        elif d == 0 and ch == ":" and rest[k + 1:k + 2] != "=":
            if depth == 0:
                a, b = rest[:k], rest[k + 1:]
                return f"where({cond.strip()}, {_ternary(a, line).strip()}, {_ternary(b, line).strip()})"
            depth -= 1
    _refuse(line, f"the ternary in {t.strip()!r} has no ':'.")


class _Inline(ast.NodeTransformer):
    """Calls of the script's one-line functions f(x) => expr replaced by expr with the arguments put in."""

    def __init__(self, funcs, line):
        self.funcs, self.line = funcs, line

    def visit_Call(self, n):
        self.generic_visit(n)
        if isinstance(n.func, ast.Name) and n.func.id in self.funcs:
            params, body = self.funcs[n.func.id]
            args = list(n.args)
            kw = {k.arg: k.value for k in n.keywords}
            vals = {}
            for i, pname in enumerate(params):
                pn, default = pname
                if i < len(args):
                    vals[pn] = args[i]
                elif pn in kw:
                    vals[pn] = kw[pn]
                elif default is not None:
                    vals[pn] = default
                else:
                    _refuse(self.line, f"{n.func.id}() is called without its argument {pn}.")
            if len(args) > len(params):
                _refuse(self.line, f"{n.func.id}() takes {len(params)} argument(s).")

            class Put(ast.NodeTransformer):
                def visit_Name(self, m):
                    return copy.deepcopy(vals[m.id]) if m.id in vals else m
            return self.visit(Put().visit(copy.deepcopy(body)))
        return n


def _pyexpr(text: str, line: int) -> ast.AST:
    """A Pine expression as a Python AST (the operators and literals are the same for the supported subset; the
    ternary becomes where(cond, a, b) and the script's one-line functions are inlined)."""
    t = text.strip()
    if re.search(r":=", t):
        _refuse(line, "reassignment (:=) is only supported for a variable declared with var (state kept from bar to bar).")
    t = _ternary(t, line)
    t = re.sub(r"\btrue\b", "True", t)
    t = re.sub(r"\bfalse\b", "False", t)
    try:
        node = ast.parse(t, mode="eval").body
    except SyntaxError:
        _refuse(line, f"could not read the expression {text.strip()!r}.")
    funcs = getattr(_TLS, "funcs", None)
    if funcs:
        node = _Inline(funcs, line).visit(node)
    return node


def _attr(n) -> str:
    """'strategy.position_size' for the attribute chain, '' otherwise."""
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
        return ".".join(reversed(parts))
    return ""


def _call_name(n) -> str:
    return _attr(n.func) if isinstance(n, ast.Call) else ""


def _const(n):
    """The value of a constant expression (numbers, strings, booleans, arithmetic on them), else None."""
    try:
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = _const(n.operand)
            return None if not isinstance(v, (int, float)) else (-v if isinstance(n.op, ast.USub) else v)
        if isinstance(n, ast.BinOp):
            a, b = _const(n.left), _const(n.right)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
                return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b,
                        ast.Div: a / b if b else None}.get(type(n.op))
    except Exception:  # noqa: BLE001
        return None
    return None


class _Ctx:
    def __init__(self):
        self.sym: dict[str, ast.AST] = {}
        self.start = None
        self.end = None
        self.notes: list[str] = []
        self.state: dict[str, dict] = {}        # var name -> {"type", "init", "line"}
        self.updates: list[dict] = []           # {"var", "when": [(cond id, node)], "value": node, "line"}
        self.uses: list[tuple[int, set]] = []   # (line, names read) of statements that read state


def _input_default(call: ast.Call, line: int):
    name = _call_name(call)
    kw = {k.arg: k.value for k in call.keywords}
    v = kw.get("defval", call.args[0] if call.args else None)
    if v is None:
        _refuse(line, f"{name}() has no default value.")
    if name in ("input.session", "input.color", "input.text_area", "input.enum"):
        _refuse(line, f"{name}() is not supported.")
    return v


class _Subst(ast.NodeTransformer):
    """Inputs and variables replaced by their definitions; strategy.* position state mapped or refused."""

    def __init__(self, ctx: _Ctx, line: int, where: str):
        self.ctx, self.line, self.where = ctx, line, where     # where: "entry-long" / "entry-short" / "exit-long"...

    def visit_UnaryOp(self, n):
        # not (strategy.position_size > 0) -> strategy.position_size <= 0 (also through a variable)
        if isinstance(n.op, ast.Not):
            inner = n.operand
            while isinstance(inner, ast.Name) and inner.id in self.ctx.sym:
                inner = self.ctx.sym[inner.id]
            if (isinstance(inner, ast.Compare) and len(inner.ops) == 1
                    and any(_attr(x) in ("strategy.position_size", "strategy.opentrades") for x in [inner.left] + inner.comparators)):
                neg = {ast.Gt: ast.LtE, ast.GtE: ast.Lt, ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Eq: ast.NotEq,
                       ast.NotEq: ast.Eq}[type(inner.ops[0])]()
                return self.visit(ast.Compare(left=copy.deepcopy(inner.left), ops=[neg],
                                              comparators=copy.deepcopy(inner.comparators)))
        return self.generic_visit(n)

    def visit_Name(self, n):
        if n.id in self.ctx.sym:
            return self.visit(copy.deepcopy(self.ctx.sym[n.id]))
        if n.id in self.ctx.state:
            return ast.Name(id="pv_" + n.id, ctx=ast.Load())
        if n.id == "na" and self.where == "state":
            return n
        if n.id == "bar_index":
            _refuse(self.line, "bar_index is only supported as bar_index - strategy.opentrades.entry_bar_index(0) (the "
                               "bars since the entry) in an exit.")
        if n.id in ("time", "time_close", "timenow"):
            _refuse(self.line, f"{n.id} is only supported in a date filter: time >= timestamp(2015, 1, 1).")
        if n.id == "na":
            _refuse(self.line, "na as a value is only supported in a var's value (var float x = na, x := na).")
        return n

    def visit_Call(self, n):
        name = _call_name(n)
        if name == "input" or name.startswith("input."):
            return self.visit(copy.deepcopy(_input_default(n, self.line)))
        if name in ("timestamp",):
            return n
        if name.startswith("strategy."):
            _refuse(self.line, f"{name}() can't be used inside an expression.")
        if name == "na" and not n.args and self.where == "state":
            return ast.Name(id="na", ctx=ast.Load())
        return self.generic_visit(n)

    def visit_BinOp(self, n):
        # bar_index - strategy.opentrades.entry_bar_index(0): the bars since the entry bar
        if (isinstance(n.op, ast.Sub) and isinstance(n.left, ast.Name) and n.left.id == "bar_index"
                and _call_name(n.right) == "strategy.opentrades.entry_bar_index"):
            if not self.where.startswith("exit"):
                _refuse(self.line, "bars since the entry can only be used in an exit.")
            return ast.Name(id="bars_held", ctx=ast.Load())
        return self.generic_visit(n)

    def visit_Attribute(self, n):
        a = _attr(n)
        if a == "strategy.position_avg_price":
            if self.where == "display":
                return n
            if not self.where.startswith("exit"):
                _refuse(self.line, "strategy.position_avg_price can only be used in an exit.")
            return ast.Name(id="entry_price", ctx=ast.Load())
        if a == "barstate.isconfirmed":
            return ast.Constant(True)
        if a.startswith(("strategy.", "barstate.")) and a not in ("strategy.long", "strategy.short"):
            if self.where == "state":
                _refuse(self.line, f"{a} in a var's update: state that depends on the position is not supported (the "
                                   "state is computed from the bars before the simulation).")
            _refuse(self.line, f"{a} is not supported here.")
        return self.generic_visit(n)

    def visit_Compare(self, n):
        ops = [n.left] + list(n.comparators)
        # date filters: time >= timestamp(...) -> the start of the test
        if len(n.ops) == 1 and any(isinstance(x, ast.Name) and x.id in ("time", "time_close") for x in ops):
            return n if self.where == "display" else self._date(n)
        pos = [i for i, x in enumerate(ops) if _attr(x) in ("strategy.position_size", "strategy.opentrades")]
        if pos and self.where == "display":
            return n
        if pos and self.where == "state":
            _refuse(self.line, f"'{ast.unparse(n)}' in a var's update: state that depends on the position is not "
                               "supported (the state is computed from the bars before the simulation).")
        if pos:
            if len(n.ops) != 1:
                _refuse(self.line, "a chained comparison of strategy.position_size is not supported.")
            v = _const(ops[1 - pos[0]])
            op = type(n.ops[0])
            if pos[0] == 1:   # 0 < strategy.position_size -> strategy.position_size > 0
                op = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE}.get(op, op)
            flat = (op is ast.Eq and v == 0)
            long_ = (op is ast.Gt and v == 0) or (op is ast.GtE and v == 1)
            short_ = (op is ast.Lt and v == 0)
            inpos = (op is ast.NotEq and v == 0) or (_attr(ops[pos[0]]) == "strategy.opentrades" and long_)
            ok = {"entry-long": flat or (op is ast.LtE and v == 0), "entry-short": flat or (op is ast.GtE and v == 0),
                  "exit-long": long_ or inpos, "exit-short": short_ or inpos, "exit-any": inpos or long_ or short_}
            if v is None or not ok.get(self.where, False):
                _refuse(self.line, f"'{ast.unparse(n)}' is not supported here: position checks are only read as 'no "
                                   "position' before an entry or 'in a position' before an exit (which the backtest "
                                   "already applies).")
            if "position check" not in " ".join(self.ctx.notes):
                self.ctx.notes.append(f"line {self.line}: the position check {ast.unparse(n)} is implied (entries are "
                                      "only taken as pyramiding allows, exits only apply to an open position).")
            return ast.Constant(True)
        return self.generic_visit(n)

    def _date(self, n):
        a, b = n.left, n.comparators[0]
        op = type(n.ops[0])
        if isinstance(b, ast.Name) and b.id in ("time", "time_close"):
            a, b = b, a
            op = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE}.get(op, op)
        b = self.visit(b)
        when = None
        if _call_name(b) == "timestamp":
            args = [_const(x) for x in b.args]
            try:
                if len(args) == 1 and isinstance(args[0], str):
                    when = pd.Timestamp(args[0]).tz_localize(None) if pd.Timestamp(args[0]).tz else pd.Timestamp(args[0])
                else:
                    nums = [x for x in args if isinstance(x, (int, float)) and not isinstance(x, bool)]
                    if len(nums) >= 3:
                        when = pd.Timestamp(int(nums[0]), int(nums[1]), int(nums[2]))
                    elif len(args) >= 4 and isinstance(args[0], str):   # timestamp("GMT-5", 2020, 1, 1, ...)
                        when = pd.Timestamp(int(args[1]), int(args[2]), int(args[3]))
            except (ValueError, TypeError):
                when = None
        if when is None:
            _refuse(self.line, f"the date filter {ast.unparse(n)!r} is not supported; use time >= timestamp(2015, 1, 1).")
        if op in (ast.Gt, ast.GtE):
            self.ctx.start = str(when.date())
        elif op in (ast.Lt, ast.LtE):
            self.ctx.end = str(when.date())
        else:
            _refuse(self.line, f"the date filter {ast.unparse(n)!r} is not supported.")
        self.ctx.notes.append(f"line {self.line}: the date filter {ast.unparse(n)} sets the "
                              f"{'start' if op in (ast.Gt, ast.GtE) else 'end'} of the test ({when.date()}).")
        return ast.Constant(True)


def _simplify(n: ast.AST) -> ast.AST:
    """`True and x` -> x, `x and True` -> x (the position checks and date filters replaced by True)."""
    class S(ast.NodeTransformer):
        def visit_BoolOp(self, b):
            self.generic_visit(b)
            if isinstance(b.op, ast.And):
                vals = [v for v in b.values if not (isinstance(v, ast.Constant) and v.value is True)]
                if not vals:
                    return ast.Constant(True)
                if len(vals) == 1:
                    return vals[0]
                b.values = vals
            return b
    return S().visit(n)


def _rule(node: ast.AST, ctx: _Ctx, line: int, where: str) -> str:
    """An expression node as rule-language text (TradingView spellings translated), checked by the rule compiler."""
    from .expr import compile_expr, pine_to_rule
    n = _simplify(_Subst(ctx, line, where).visit(copy.deepcopy(node)))
    text = ast.unparse(ast.fix_missing_locations(n))
    try:
        out = pine_to_rule(text)
        compile_expr(out)
    except ValueError as e:
        _refuse(line, str(e))
    return out


# ------------------------------------------------------------------ strategy.exit levels

def _level(node: ast.AST, kind: str, side: int, ctx: _Ctx, line: int) -> tuple[str, object, int | None]:
    """A strategy.exit stop / limit price as ('pct', fraction, None) or ('atr', multiple, period), from
    strategy.position_avg_price * (1 -/+ x), * k, or -/+ n * ta.atr(m) (the ATR as of each bar: TradingView re-evaluates
    strategy.exit on every bar); anything else as ('expr', the level in the rule language, None), re-evaluated on every
    bar (entry_price = strategy.position_avg_price)."""
    n = _Inputs(ctx, line).visit(copy.deepcopy(node))

    def is_p(x):
        return _attr(x) == "strategy.position_avg_price"

    want_up = (kind == "limit") == (side == 1)       # a long's target is above the entry, its stop below
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) and (is_p(n.left) or is_p(n.right)):
        k = _const(n.right if is_p(n.left) else n.left)
        if isinstance(k, (int, float)) and k > 0 and k != 1 and (k > 1) == want_up:
            return "pct", abs(k - 1), None
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)) and is_p(n.left):
        up = isinstance(n.op, ast.Add)
        r = n.right
        atr, mult = None, None
        for a, b in ((r.left, r.right), (r.right, r.left)) if isinstance(r, ast.BinOp) and isinstance(r.op, ast.Mult) else []:
            if _call_name(a) == "ta.atr" and _const(b) is not None:
                atr, mult = a, _const(b)
        if _call_name(r) == "ta.atr":
            atr, mult = r, 1.0
        if atr is not None and up == want_up and mult and mult > 0:
            per = _const(atr.args[0]) if atr.args else 14
            if isinstance(per, (int, float)) and float(per).is_integer():
                ctx.notes.append(f"line {line}: the {'target' if kind == 'limit' else 'stop'} {ast.unparse(node)} follows the "
                                 f"current ATR({int(per)}): TradingView re-evaluates strategy.exit() on every bar, so the "
                                 "level moves with the ATR as of the previous close (current_atr).")
                return "atr", float(mult), int(per)
        # P + x (a price distance) or P * x/100 written as P + P * x
        if isinstance(r, ast.BinOp) and isinstance(r.op, ast.Mult) and (is_p(r.left) or is_p(r.right)) and up == want_up:
            k = _const(r.right if is_p(r.left) else r.left)
            if isinstance(k, (int, float)) and k > 0:
                return "pct", float(k), None
    text = _rule(node, ctx, line, "exit-long" if side == 1 else "exit-short")
    if text in ("True", "False"):
        _refuse(line, f"the {'target (limit)' if kind == 'limit' else 'stop'} {ast.unparse(node)!r} is not a price.")
    return "expr", text, None


def _ticks(node: ast.AST, ctx: _Ctx, line: int, what: str, ticker: str) -> tuple[str, object, int | None]:
    """profit= / loss= / trail_offset= / trail_points= in ticks: a number of ticks (syminfo.mintick = $0.01 for US
    stocks and ETFs) -> ('pts', price distance); a price distance divided by syminfo.mintick: close * x -> ('pct', x),
    n * ta.atr(m) -> ('atr', n, m), any other expression -> ('dist', the distance in the rule language)."""
    from .strategy import _fractional_market
    n = _Inputs(ctx, line).visit(copy.deepcopy(node))
    k0 = _const(n)
    if isinstance(k0, (int, float)) and not isinstance(k0, bool):
        if _fractional_market(ticker):
            _refuse(line, f"{what}={ast.unparse(node)} is in ticks, and the tick size of {ticker} is not known; write it as "
                          "a price distance divided by syminfo.mintick, e.g. close * 0.05 / syminfo.mintick.")
        if k0 < 0:
            _refuse(line, f"{what}={ast.unparse(node)}: a number of ticks can't be negative.")
        note = (f"{what}={ast.unparse(node)} ticks: syminfo.mintick is $0.01 for US stocks and ETFs, so "
                f"{k0:g} ticks = ${k0 * MINTICK:g}.")
        if note not in ctx.notes:
            ctx.notes.append(f"line {line}: {note}")
        return "pts", float(k0) * MINTICK, None
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and _attr(n.right) == "syminfo.mintick":
        d = n.left
        pair = (d.left, d.right) if isinstance(d, ast.BinOp) and isinstance(d.op, ast.Mult) else None
        if _call_name(d) == "ta.atr":
            pair = (d, ast.Constant(1.0))
        for a, b in ((pair, pair[::-1]) if pair else ()):
            k = _const(b)
            if k is None or k <= 0:
                continue
            if _attr(a) in ("close", "strategy.position_avg_price") or (isinstance(a, ast.Name) and a.id == "close"):
                if _attr(a) != "strategy.position_avg_price":
                    ctx.notes.append(f"line {line}: {what} = {ast.unparse(node)} is {k:.4g} x the close when strategy.exit() is "
                                     f"called; the backtest takes {k:.2%} of the entry price.")
                return "pct", float(k), None
            if _call_name(a) == "ta.atr":
                per = _const(a.args[0]) if a.args else 14
                if isinstance(per, (int, float)) and float(per).is_integer():
                    return "atr", float(k), int(per)
        return "dist", _rule(d, ctx, line, "exit-long"), None
    _refuse(line, f"{what}={ast.unparse(node)} is in ticks: give a number of ticks (e.g. {what}=500, $5 at a $0.01 tick) "
                  "or a price distance divided by syminfo.mintick, e.g. close * 0.05 / syminfo.mintick (5%) or "
                  "2 * ta.atr(14) / syminfo.mintick.")


GTC_BARS = 100_000      # a working entry order with no expiry (TradingView keeps it until filled or replaced)


def expr_names(text) -> set:
    """The names a rule-language expression reads."""
    from .expr import names_in
    return names_in(text)


class _Inputs(ast.NodeTransformer):
    """Variables and input.*() calls replaced by their definitions / default values, and nothing else (the numbers
    inside a stop / limit, a qty, a setting)."""

    def __init__(self, ctx, line):
        self.ctx, self.line = ctx, line

    def visit_Name(self, n):
        v = self.ctx.sym.get(n.id)
        if v is not None:
            return self.visit(copy.deepcopy(v))
        if n.id in self.ctx.state:
            return ast.Name(id="pv_" + n.id, ctx=ast.Load())
        return n

    def visit_Call(self, n):
        name = _call_name(n)
        if name == "input" or name.startswith("input."):
            return self.visit(copy.deepcopy(_input_default(n, self.line)))
        return self.generic_visit(n)


# ------------------------------------------------------------------ the script

def _args(call: ast.Call, names: list[str]) -> dict:
    """Positional and keyword arguments of a call by name."""
    out = {names[i]: a for i, a in enumerate(call.args) if i < len(names)}
    if len(call.args) > len(names):
        raise ValueError("too many arguments")
    for k in call.keywords:
        out[k.arg] = k.value
    return out


ENTRY_ARGS = ["id", "direction", "qty", "limit", "stop", "oca_name", "oca_type", "comment", "alert_message", "disable_alert"]
EXIT_ARGS = ["id", "from_entry", "qty", "qty_percent", "profit", "limit", "loss", "stop", "trail_price", "trail_points",
             "trail_offset", "oca_name", "comment", "comment_profit", "comment_loss", "comment_trailing", "alert_message",
             "alert_profit", "alert_loss", "alert_trailing", "disable_alert"]
CLOSE_ARGS = ["id", "comment", "qty", "qty_percent", "alert_message", "immediately", "disable_alert"]


def translate(text: str, ticker: str | None = None) -> Strategy:
    """A Pine strategy script as a Strategy (TradingView-compatible mode), with the translation in its notes."""
    _TLS.funcs = {}
    try:
        return _translate(text, ticker)
    finally:
        _TLS.funcs = None


class _StateNames(ast.NodeTransformer):
    """The script's var names as the series the backtest computes (pv_<name>, so they never shadow a rule-language
    name such as atr or high)."""

    def __init__(self, ctx):
        self.ctx = ctx

    def visit_Name(self, n):
        return ast.Name(id="pv_" + n.id, ctx=ast.Load()) if n.id in self.ctx.state else n


def _state_uses(node, ctx) -> set:
    """The var names an expression reads, directly or through the variables it uses."""
    out, seen = set(), set()

    def walk(n):
        for x in ast.walk(n):
            if isinstance(x, ast.Name):
                if x.id in ctx.state:
                    out.add(x.id)
                elif x.id in ctx.sym and x.id not in seen:
                    seen.add(x.id)
                    walk(ctx.sym[x.id])
    walk(node)
    return out


def _translate(text: str, ticker: str | None = None) -> Strategy:
    stmts, comments = _statements(text)
    ctx = _Ctx()
    tick = None
    for n, c in comments:
        m = re.match(r"(?i)\s*(?:@?ticker|@?symbol)\s*[:=]?\s*([A-Za-z0-9.^$=_:-]+)\s*$", c)
        if m:
            tick = m.group(1).split(":")[-1]
            break
    tick = tick or (ticker.strip() if isinstance(ticker, str) and ticker.strip() else None)
    if not tick:
        raise PineImportError("Which ticker should this Pine script run on? Add a comment line such as '// ticker: SPY' "
                              "to the script, or give the ticker (the Ticker setting on the site, --tickers SPY on the "
                              "command line).")
    if "," in tick:
        raise PineImportError("A Pine script runs on one ticker (the chart's symbol); give one, e.g. '// ticker: SPY'.")
    tick = data.canonical(tick)
    settings: dict = {}
    title = ""
    seen_strategy = False
    entries: dict[str, dict] = {}         # entry id -> {"side", "conds": [rule], "line"}
    closes: list[dict] = []               # {"ids": [entry ids] or None (all), "cond", "line"}
    exits: list[dict] = []                # {"from": id or None, "args", "line", "cond"}
    shown: list[str] = []
    skipped: list[int] = []
    blocks: list[tuple[int, list]] = []   # (indent, [condition nodes]) of the enclosing ifs
    last_if: dict[int, list] = {}         # indent -> the conditions of the last if/else-if at that indent

    def conds_here(ind):
        while blocks and blocks[-1][0] >= ind:
            blocks.pop()
        out = []
        for _, cs in blocks:
            out.extend(cs)
        return out

    for line, ind, code in stmts:
        s = code.strip()
        head = conds_here(ind)
        m_if = re.match(r"^(else\s+if|if)\s+(.+)$", s)
        if m_if or s == "else":
            if m_if and m_if.group(1) == "if":
                c = _pyexpr(m_if.group(2), line)
                blocks.append((ind, [c]))
                last_if[ind] = [c]
            else:
                prev = last_if.get(ind)
                if prev is None:
                    _refuse(line, "'else' without an 'if' before it.")
                negs = [ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(p)) for p in prev]
                if m_if:
                    c = _pyexpr(m_if.group(2), line)
                    blocks.append((ind, negs + [c]))
                    last_if[ind] = prev + [c]
                else:
                    blocks.append((ind, negs))
            continue
        # (a statement at this indent ends any if-chain deeper than it)
        for k in [k for k in last_if if k > ind]:
            del last_if[k]
        if re.match(r"^(for|while|switch|import|export|method|type|enum)\b", s):
            _refuse(line, f"'{s.split()[0]}' is not supported by the importer.")
        # one-line custom functions: f(x, y = 2) => expression (inlined where called)
        m = re.match(r"^([A-Za-z_]\w*)\s*\(([^()]*)\)\s*=>\s*(.*)$", s)
        if m:
            if head or ind > 0:
                _refuse(line, "a function defined inside a block is not supported.")
            if not m.group(3).strip() or re.search(r"(?<![=!<>:])=(?![=>])", m.group(3)):
                _refuse(line, f"the function {m.group(1)}() has a body of several lines; only one-line functions "
                              "(name(x) => expression) are supported.")
            params = []
            for part in [x.strip() for x in _split_top(m.group(2), ",") if x.strip()]:
                pm = re.match(r"^(?:(?:simple|series|const)\s+)?(?:(?:float|int|bool)\s+)?([A-Za-z_]\w*)\s*(?:=\s*(.+))?$", part)
                if not pm:
                    _refuse(line, f"the parameter {part!r} of {m.group(1)}() is not supported.")
                params.append((pm.group(1), _pyexpr(pm.group(2), line) if pm.group(2) else None))
            body = _pyexpr(m.group(3), line)
            for x in ast.walk(body):
                if isinstance(x, ast.Name) and x.id in ctx.state:
                    _refuse(line, f"the function {m.group(1)}() reads the var {x.id}; functions may only use their "
                                  "arguments, prices and indicators.")
            _TLS.funcs[m.group(1)] = (params, body)
            shown.append(f"line {line}: {m.group(1)}({', '.join(p_ for p_, _ in params)}) => {ast.unparse(body)} (inlined where called)")
            continue
        # var: state kept from bar to bar
        m = re.match(r"^(var|varip)\s+(?:(float|int|bool|string|color|label|line|box|table)\s+)?([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$", s)
        if m or re.match(r"^(var|varip)\b", s):
            if not m or m.group(1) == "varip":
                _refuse(line, f"'{s.split()[0]}' is not supported here (varip updates within a bar, which daily bars can't "
                              "show); use var <type> name = <value>.")
            typ, nm, rhs = m.group(2), m.group(3), m.group(4)
            if typ in ("string", "color", "label", "line", "box", "table"):
                if typ in ("label", "line", "box", "table", "color"):
                    skipped.append(line)
                    continue
                _refuse(line, "a var of type string is not supported (numbers and true / false only).")
            if head or ind > 0:
                _refuse(line, f"the var {nm} is declared inside a block; declare it at the top level.")
            if nm in ctx.sym or nm in ctx.state:
                _refuse(line, f"{nm} is declared twice.")
            init = _pyexpr(rhs, line)
            if typ is None:
                typ = "bool" if isinstance(init, ast.Constant) and isinstance(init.value, bool) else "float"
            ctx.state[nm] = {"type": typ, "init": init, "line": line}
            continue
        # updates of a var: x := value, x += value (at the top level or inside ifs)
        m = re.match(r"^([A-Za-z_]\w*)\s*(:=|\+=|-=|\*=|/=)\s*(.+)$", s)
        if m:
            nm, op, rhs = m.groups()
            if nm not in ctx.state:
                _refuse(line, f"reassignment ({op}) of {nm}: only a variable declared with var (var float {nm} = ...) can be "
                              "updated; it keeps its value from bar to bar.")
            val = _pyexpr(rhs, line)
            if op != ":=":
                val = ast.BinOp(left=ast.Name(id=nm, ctx=ast.Load()),
                                op={"+=": ast.Add(), "-=": ast.Sub(), "*=": ast.Mult(), "/=": ast.Div()}[op], right=val)
            ctx.updates.append({"var": nm, "when": [(id(c), c) for c in head], "value": val, "line": line})
            continue
        # strategy(...) / indicator(...)
        m = re.match(r"^(strategy|indicator|study)\s*\(", s)
        if m:
            if m.group(1) != "strategy":
                _refuse(line, "this is an indicator, not a strategy: it places no orders. Paste a script that starts "
                              "with strategy(...).")
            node = _pyexpr(s, line)
            try:
                kw = _args(node, ["title", "shorttitle", "overlay", "format", "precision", "scale", "pyramiding",
                                  "calc_on_order_fills", "calc_on_every_tick", "max_bars_back", "backtest_fill_limits_assumption",
                                  "default_qty_type", "default_qty_value", "initial_capital", "currency", "slippage",
                                  "commission_type", "commission_value", "process_orders_on_close", "close_entries_rule",
                                  "margin_long", "margin_short"])
            except ValueError:
                _refuse(line, "strategy(): too many arguments.")
            settings = {k: _Inputs(ctx, line).visit(v) for k, v in kw.items()}
            title = _const(settings.get("title")) or ""
            seen_strategy = True
            continue
        # display-only calls
        m = re.match(r"^([A-Za-z_][\w.]*)\s*\(", s)
        if m and m.group(1) in DISPLAY:
            skipped.append(line)
            continue
        if m and m.group(1).startswith("strategy."):
            call = _pyexpr(s, line)
            name = _call_name(call)
            used = _state_uses(call, ctx)
            for c in head:
                used |= _state_uses(c, ctx)
            if used:
                ctx.uses.append((line, used))
            if name == "strategy.entry":
                try:
                    a = _args(call, ENTRY_ARGS + ["when"])
                except ValueError:
                    _refuse(line, "strategy.entry(): too many arguments.")
                order = None
                if "limit" in a and "stop" in a:
                    _refuse(line, "strategy.entry(..., limit=, stop=) (a stop-limit entry order) is not supported; use a "
                                  "limit or a stop.")
                for kind_ in ("limit", "stop"):
                    if kind_ in a:
                        order = (kind_, a[kind_], line)
                if "qty" in a:
                    q = _const(_Inputs(ctx, line).visit(copy.deepcopy(a["qty"])))
                    if not isinstance(q, (int, float)) or q <= 0:
                        _refuse(line, "strategy.entry(qty=...) must be a fixed number of shares.")
                    settings.setdefault("_qty", {})[line] = float(q)
                d = _attr(a.get("direction")) if a.get("direction") is not None else ""
                if d not in ("strategy.long", "strategy.short"):
                    if isinstance(a.get("direction"), ast.Constant) and isinstance(a["direction"].value, bool):
                        d = "strategy.long" if a["direction"].value else "strategy.short"   # Pine v3: long=true
                    else:
                        _refuse(line, "strategy.entry() needs strategy.long or strategy.short as its direction.")
                eid = _const(a.get("id"))
                if not isinstance(eid, str):
                    _refuse(line, "strategy.entry() needs an id in quotes, e.g. \"Long\".")
                side = 1 if d == "strategy.long" else -1
                cs = head + ([a["when"]] if "when" in a else [])
                e = entries.setdefault(eid, {"side": side, "conds": [], "line": line, "orders": []})
                if e["side"] != side:
                    _refuse(line, f"the entry id {eid!r} is used for both long and short entries.")
                e["conds"].append((cs, line))
                e["orders"].append(order)
                continue
            if name in ("strategy.close", "strategy.close_all"):
                try:
                    a = _args(call, (CLOSE_ARGS if name == "strategy.close" else CLOSE_ARGS[1:]) + ["when"])
                except ValueError:
                    _refuse(line, f"{name}(): too many arguments.")
                if "qty" in a or "qty_percent" in a:
                    _refuse(line, f"{name}(qty=...) (a partial market exit) is not supported; use strategy.exit(..., "
                                  "qty_percent=, limit=) for a partial exit at a price.")
                if "immediately" in a and _const(a["immediately"]):
                    _refuse(line, f"{name}(immediately=true) is not supported.")
                ids = None
                if name == "strategy.close":
                    eid = _const(a.get("id"))
                    if not isinstance(eid, str):
                        _refuse(line, "strategy.close() needs the entry id in quotes, e.g. \"Long\".")
                    ids = [eid]
                closes.append({"ids": ids, "cs": head + ([a["when"]] if "when" in a else []), "line": line})
                continue
            if name == "strategy.exit":
                try:
                    a = _args(call, EXIT_ARGS + ["when"])
                except ValueError:
                    _refuse(line, "strategy.exit(): too many arguments.")
                fe = _const(a["from_entry"]) if "from_entry" in a else None
                if "from_entry" in a and not isinstance(fe, str):
                    _refuse(line, "strategy.exit(from_entry=...) must be an entry id in quotes.")
                exits.append({"from": fe, "a": a, "line": line, "cs": head + ([a["when"]] if "when" in a else [])})
                continue
            _refuse(line, f"{name}() is not supported by the importer (supported: strategy.entry, strategy.close, "
                          "strategy.close_all, strategy.exit).")
        # tuple assignment
        m = re.match(r"^\[\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\]\s*=\s*(.+)$", s)
        if m:
            names = [x.strip() for x in m.group(1).split(",")]
            call = _pyexpr(m.group(2), line)
            fn = _call_name(call)
            base = fn.split(".")[-1]
            if not fn.startswith("ta.") or base not in TUPLES:
                _refuse(line, f"a tuple from {fn or 'this expression'} is not supported (supported: "
                              f"{', '.join('ta.' + k for k in TUPLES)}).")
            args = [ast.unparse(_Subst(ctx, line, "display").visit(copy.deepcopy(x))) for x in call.args]
            if len(args) != TUPLE_ARGS[base]:
                _refuse(line, f"ta.{base}() takes {TUPLE_ARGS[base]} arguments here.")
            if base == "dmi":
                args = [args[0], args[1]]
            vals = TUPLES[base](args)
            if len(names) > len(vals):
                _refuse(line, f"ta.{base}() returns {len(vals)} values.")
            for nm, v in zip(names, vals):
                if nm != "_":
                    ctx.sym[nm] = ast.parse(v, mode="eval").body
            shown.append(f"line {line}: [{', '.join(names)}] = {', '.join(vals[:len(names)])}")
            continue
        # variable assignment
        m = re.match(r"^(?:(?:const|simple|series)\s+)?(?:(?:float|int|bool|string|color)\s+)?([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$", s)
        if m:
            nm, rhs = m.group(1), m.group(2)
            node = _pyexpr(rhs, line)
            if head:
                _refuse(line, f"'{nm}' is assigned inside an if block; the importer only reads variables defined at "
                              "the top level (a value kept from bar to bar is declared with var and updated with :=).")
            if nm in ctx.state:
                _refuse(line, f"{nm} is a var: update it with {nm} := ... (= would declare it again).")
            used = _state_uses(node, ctx)
            if used:
                ctx.uses.append((line, used))
            if _call_name(node) == "input" or _call_name(node).startswith("input."):
                v = _input_default(node, line)
                v = _Inputs(ctx, line).visit(copy.deepcopy(v))
                ctx.sym[nm] = v
                shown.append(f"line {line}: {nm} = {ast.unparse(v)} (the input's default)")
                continue
            if _call_name(node).startswith("color.") or (isinstance(node, ast.Attribute) and _attr(node).startswith("color.")):
                continue
            ctx.sym[nm] = node
            if _const(_Inputs(ctx, line).visit(copy.deepcopy(node))) is None:
                try:
                    shown.append(f"line {line}: {nm} = {_rule(node, ctx, line, 'display')}")
                except PineImportError:
                    shown.append(f"line {line}: {nm} = {ast.unparse(node)}")
            continue
        _refuse(line, f"{s[:60]!r} is not supported by the importer.")

    if not seen_strategy:
        raise PineImportError("This is not a Pine strategy: there is no strategy(...) declaration.")
    if not entries:
        raise PineImportError("The Pine script places no strategy.entry() orders.")

    # ---- settings
    notes = list(ctx.notes)
    kw: dict = {}
    poc = bool(_const(settings.get("process_orders_on_close", ast.Constant(False))))
    for bad in ("calc_on_order_fills", "use_bar_magnifier", "calc_on_every_history_tick"):
        if _const(settings.get(bad, ast.Constant(False))):
            raise PineImportError(f"strategy({bad} = true) is not supported (it needs intrabar data).")
    slip = _const(settings.get("slippage", ast.Constant(0)))
    if slip:
        raise PineImportError("strategy(slippage=...) is in ticks, which the backtest can't convert (no tick size); set "
                              "slippage in basis points instead, e.g. add '5 bps slippage' as a setting.")
    bfl = _const(settings.get("backtest_fill_limits_assumption", ast.Constant(0)))
    if bfl:
        raise PineImportError("strategy(backtest_fill_limits_assumption=...) is not supported.")
    cap = _const(settings.get("initial_capital")) if "initial_capital" in settings else 1_000_000
    if not isinstance(cap, (int, float)) or cap <= 0:
        raise PineImportError("strategy(initial_capital=...) must be a positive number.")
    kw["capital"] = float(cap)
    qt = _attr(settings.get("default_qty_type")) if "default_qty_type" in settings else "strategy.fixed"
    qv = _const(settings.get("default_qty_value")) if "default_qty_value" in settings else 1
    if not isinstance(qv, (int, float)) or qv <= 0:
        raise PineImportError("strategy(default_qty_value=...) must be a positive number.")
    qtys = set((settings.get("_qty") or {}).values())
    if len(qtys) > 1:
        raise PineImportError("strategy.entry() calls use different qty values; one size per strategy is supported.")
    if qtys:
        kw["sizing"], kw["fixed_amount"] = "fixed_shares", qtys.pop()
        size_txt = f"{kw['fixed_amount']:g} shares per entry (qty)"
    elif qt == "strategy.percent_of_equity":
        kw["position_size"] = qv / 100
        size_txt = f"{qv:g}% of equity per entry"
    elif qt == "strategy.cash":
        kw["sizing"], kw["fixed_amount"] = "fixed_dollars", float(qv)
        size_txt = f"${qv:,.0f} per entry"
    elif qt == "strategy.fixed":
        kw["sizing"], kw["fixed_amount"] = "fixed_shares", float(qv)
        size_txt = f"{qv:g} share{'s' if qv != 1 else ''} per entry" + (" (TradingView's default order size)" if "default_qty_type" not in settings else "")
    else:
        raise PineImportError(f"default_qty_type {qt or '?'} is not supported.")
    pyr = _const(settings.get("pyramiding", ast.Constant(1)))
    kw["pyramiding"] = max(int(pyr or 1), 1)
    ct = _attr(settings.get("commission_type")) if "commission_type" in settings else "strategy.commission.percent"
    cv = _const(settings.get("commission_value", ast.Constant(0))) or 0
    com_txt = "no commission"
    if cv:
        if ct == "strategy.commission.percent":
            kw["commission_pct"], com_txt = cv / 100, f"commission {cv:g}% of value"
        elif ct == "strategy.commission.cash_per_order":
            kw["commission"], com_txt = float(cv), f"commission ${cv:g} per order"
        elif ct == "strategy.commission.cash_per_contract":
            kw["commission_per_share"], com_txt = float(cv), f"commission ${cv:g} per share"
        else:
            raise PineImportError(f"commission_type {ct} is not supported.")
    ml = _const(settings.get("margin_long", ast.Constant(100)))
    if isinstance(ml, (int, float)) and ml not in (0, 100):
        kw["leverage"] = 100.0 / ml
    fill = "close" if poc else "next_open"
    notes.insert(0, f"strategy(): ${kw['capital']:,.0f} initial capital{' (TradingView default)' if 'initial_capital' not in settings else ''}, "
                    f"{size_txt}, {com_txt}, pyramiding {kw['pyramiding']}, orders fill "
                    f"{'at the close of the signal bar (process_orders_on_close = true)' if poc else 'at the next bar open (process_orders_on_close = false)'}.")
    notes.extend(shown)

    # ---- entries
    sides = {e["side"] for e in entries.values()}

    def or_rules(items, where):
        rules = []
        for cs, ln in items:
            if not cs:
                rules.append("True")
                continue
            node = cs[0] if len(cs) == 1 else ast.BoolOp(op=ast.And(), values=[copy.deepcopy(c) for c in cs])
            r = _rule(node, ctx, ln, where)
            rules.append(r)
        rules = list(dict.fromkeys(rules))
        return rules[0] if len(rules) == 1 else " or ".join(f"({r})" for r in rules)

    long_items = [x for e in entries.values() if e["side"] == 1 for x in e["conds"]]
    short_items = [x for e in entries.values() if e["side"] == -1 for x in e["conds"]]
    long_rule = or_rules(long_items, "entry-long") if long_items else None
    short_rule = or_rules(short_items, "entry-short") if short_items else None
    for r, it in ((long_rule, long_items), (short_rule, short_items)):
        if r:
            notes.append(f"line {', '.join(str(ln) for _, ln in it)}: strategy.entry -> {'buy' if r is long_rule else 'short'} "
                         f"when {r}")
    side = "both" if sides == {1, -1} else ("long" if sides == {1} else "short")

    # ---- limit / stop entry orders
    orders = [o for e in entries.values() for o in e["orders"]]
    if any(orders):
        kinds = {o[0] for o in orders if o}
        if None in orders or len(kinds) > 1:
            bad = next(o for o in orders if o) if None in orders else orders[-1]
            _refuse(bad[2], "a script that mixes market and limit / stop entries (or limit and stop entries) is not supported: "
                            "the backtest's entries are all one order type.")
        if side == "both":
            _refuse(orders[0][2], "limit / stop entries in a long/short script are not supported (one order level applies "
                                  "to both sides).")
        lv = list(dict.fromkeys(_rule(o[1], ctx, o[2], "entry-long" if side == "long" else "entry-short") for o in orders))
        if len(lv) > 1:
            _refuse(orders[-1][2], "the entries use different limit / stop levels; one level per strategy is supported.")
        kind_ = orders[0][0]
        kw["entry_order"], kw["entry_level"], kw["order_valid_bars"] = kind_, lv[0], GTC_BARS
        notes.append(f"line {', '.join(str(o[2]) for o in orders)}: strategy.entry(..., {kind_}=...) -> a {kind_} order at "
                     f"{lv[0]} (as of the signal bar's close), working from the next bar until it fills; a new signal "
                     "replaces it (TradingView keeps an unfilled entry order until it fills, is replaced or cancelled).")

    # ---- var state
    if ctx.state:
        for ln, used in ctx.uses:
            for nm in used:
                later = [u["line"] for u in ctx.updates if u["var"] == nm and u["line"] > ln]
                if later:
                    _refuse(ln, f"this line reads the var {nm} before its update on line {later[0]}; move the update above "
                                "it (the backtest reads each var after the bar's updates).")
        spec_vars, cids = [], {}
        for nm, v in ctx.state.items():
            init = _rule(v["init"], ctx, v["line"], "state")
            ups = []
            for u in ctx.updates:
                if u["var"] != nm:
                    continue
                ups.append({"line": u["line"], "value": _rule(u["value"], ctx, u["line"], "state"),
                            "when": [[f"c{cids.setdefault(cid, len(cids) + 1)}", _rule(c, ctx, u["line"], "state")]
                                     for cid, c in u["when"]]})
            spec_vars.append({"name": "pv_" + nm, "type": v["type"], "init": init, "updates": ups})
        kw["state_vars"] = spec_vars
        try:
            check_state_vars(kw["state_vars"])
        except ValueError as e:
            raise PineImportError(f"Pine script: {e}") from None
        for v in spec_vars:
            notes.append(f"line {ctx.state[v['name'][3:]]['line']}: var {v['name'][3:]} = {v['init']} (kept from bar to bar as "
                         f"{v['name']}; updated on line{'s' if len(v['updates']) != 1 else ''} "
                         f"{', '.join(str(u['line']) for u in v['updates']) or '(none)'}), computed bar by bar from the bars "
                         "up to each bar.")

    # ---- rule exits
    def ids_side(ids, line):
        if ids is None:
            return sides
        out = set()
        for i in ids:
            if i not in entries:
                _refuse(line, f"strategy.close({i!r}): no strategy.entry() has that id.")
            out.add(entries[i]["side"])
        return out

    exit_rule, exit_sides = None, set()
    rules = []
    for c in closes:
        sd = ids_side(c["ids"], c["line"])
        where = "exit-long" if sd == {1} else "exit-short" if sd == {-1} else "exit-any"
        node = (c["cs"][0] if len(c["cs"]) == 1 else ast.BoolOp(op=ast.And(), values=[copy.deepcopy(x) for x in c["cs"]])
                if c["cs"] else ast.Constant(True))
        r = _rule(node, ctx, c["line"], where)
        rules.append(r)
        exit_sides |= sd
        notes.append(f"line {c['line']}: strategy.close -> sell when {r}")
    if rules:
        if side == "both" and exit_sides != {1, -1}:
            raise PineImportError("strategy.close() applies to only one side of a long/short script; the backtest's exit "
                                  "rule applies to both sides. Close both (strategy.close_all) or use one side.")
        rules = list(dict.fromkeys(rules))
        exit_rule = rules[0] if len(rules) == 1 else " or ".join(f"({r})" for r in rules)

    # ---- strategy.exit brackets
    ex: dict = {}
    scale: list[tuple[float, float, int]] = []       # (level fraction, qty fraction of the entry, line)
    rest_stop, rest_trail = {}, {}                   # the stop / trailing stop of the exit(s) for the rest
    part_stop: list[tuple[dict, int]] = []           # (stop, line) of each partial exit (qty_percent < 100)
    part_trail: list[tuple[dict, int]] = []
    lvl_expr: dict = {"stop_level": {}, "target_level": {}}   # side -> level text

    def put(field_, v, line):
        if field_ in ex and ex[field_] != v and not (isinstance(v, float) and abs(ex[field_] - v) < 1e-12):
            _refuse(line, "two different stops (or targets) for the same entries are not supported.")
        ex[field_] = v

    def as_fields(k, v, per, what, sides_):
        """A level of the given kind as Strategy fields: {field: value} (+ per-side level expressions)."""
        out = {}
        if k == "pct":
            out[{"stop": "stop_loss", "limit": "take_profit"}[what]] = float(v)
        elif k == "atr":
            out[{"stop": "stop_atr", "limit": "take_profit_atr"}[what]] = float(v)
            out["atr_period"] = per
            out["current_atr"] = True
        else:
            out[{"stop": "stop_level", "limit": "target_level"}[what]] = dict(v)     # side -> text
        return out

    for x in exits:
        a, line = x["a"], x["line"]
        sd = ids_side([x["from"]] if x["from"] else None, line)
        if side == "both" and sd != {1, -1} and x["from"]:
            others = [y for y in exits if y is not x]
            if not any(set(ids_side([y["from"]] if y["from"] else None, y["line"])) - sd for y in others):
                raise PineImportError(f"Pine script line {line}: strategy.exit() for one side only of a long/short script: "
                                      "the backtest's stops and targets apply to both sides. Give both sides the same exit.")
        s1 = 1 if sd == {1} else (-1 if sd == {-1} else (1 if side != "short" else -1))
        sides_x = sorted(sd) if sd else [s1]
        for c in x["cs"]:
            r = _rule(c, ctx, line, "exit-long" if s1 == 1 else "exit-short")
            if r != "True":
                _refuse(line, "a strategy.exit() placed only under a condition (other than 'in a position') is not "
                              "supported: the backtest's stops and targets apply to every trade.")
        if "qty" in a:
            _refuse(line, "strategy.exit(qty=...) is not supported (use qty_percent).")
        qp = _const(_Inputs(ctx, line).visit(copy.deepcopy(a["qty_percent"]))) if "qty_percent" in a else 100
        if not isinstance(qp, (int, float)) or not 0 < qp <= 100:
            _refuse(line, "strategy.exit(qty_percent=...) must be a number between 0 and 100.")
        for k2 in ("profit", "loss"):
            if k2 in a and k2.replace("profit", "limit").replace("loss", "stop") in a:
                _refuse(line, f"strategy.exit() has both {k2}= and "
                              f"{k2.replace('profit', 'limit').replace('loss', 'stop')}=; keep one.")
        got = []

        def level_of(what):
            """{side: (kind, value, per)} of this exit's stop ('stop') or target ('limit')."""
            out = {}
            for sg in sides_x:
                if what in a:
                    k, v, per = _level(a[what], what, sg, ctx, line)
                else:
                    k, v, per = _ticks(a["loss" if what == "stop" else "profit"], ctx, line,
                                       "loss" if what == "stop" else "profit", tick)
                    if k in ("pts", "dist"):
                        d = f"{v:g}" if k == "pts" else f"({v})"
                        up = (what == "limit") == (sg == 1)
                        k, v = "expr", f"entry_price {'+' if up else '-'} {d}"
                out[sg] = (k, v, per)
            ks = {v[0] for v in out.values()}
            if len(ks) > 1 or (ks != {"expr"} and len({(v[1], v[2]) for v in out.values()}) > 1):
                _refuse(line, "the long and short levels of this exit are not mirror images; give each side its own "
                              "strategy.exit().")
            k0, v0, p0 = next(iter(out.values()))
            if k0 == "expr":
                return "expr", {sg: v[1] for sg, v in out.items()}, None
            return k0, v0, p0

        def describe(k, v, per, what):
            if k == "pct":
                return f"{'stop loss ' + format(v, '.2%') if what == 'stop' else 'take profit +' + format(v, '.2%')}"
            if k == "atr":
                return f"{'stop' if what == 'stop' else 'take profit'} {v:g} x ATR({per}) (the current ATR)"
            return f"{'stop' if what == 'stop' else 'target'} at {' / '.join(v.values())}"

        stop_f, trail_f = None, None
        if "stop" in a or "loss" in a:
            k, v, per = level_of("stop")
            stop_f = as_fields(k, v, per, "stop", sides_x)
            got.append(describe(k, v, per, "stop"))
        if "limit" in a or "profit" in a:
            k, v, per = level_of("limit")
            if qp < 100:
                if k != "pct":
                    _refuse(line, "a partial exit (qty_percent) is supported at a percentage target only.")
                scale.append((v, qp / 100, line))
                got.append(f"sell {qp:g}% of the entry at +{v:.2%}")
            else:
                for f_, v_ in as_fields(k, v, per, "limit", sides_x).items():
                    if f_ == "target_level":
                        for sg, t_ in v_.items():
                            if lvl_expr["target_level"].get(sg, t_) != t_:
                                _refuse(line, "two different targets for the same entries are not supported.")
                            lvl_expr["target_level"][sg] = t_
                    elif f_ == "take_profit" or f_ == "take_profit_atr":
                        put(f_, v_, line)
                    else:
                        put(f_, v_, line)
                got.append(describe(k, v, per, "limit"))
        if "trail_offset" in a or "trail_points" in a or "trail_price" in a:
            if "trail_offset" not in a:
                _refuse(line, "strategy.exit(trail_points= / trail_price=) needs trail_offset (the trailing distance).")
            if "trail_points" in a and "trail_price" in a:
                _refuse(line, "strategy.exit() has both trail_points= and trail_price=; keep one.")
            trail_f = {}
            k, v, per = _ticks(a["trail_offset"], ctx, line, "trail_offset", tick)
            if k == "dist":
                _refuse(line, f"trail_offset={ast.unparse(a['trail_offset'])}: a trailing distance that changes on every "
                              "bar is not supported (use a number of ticks, close * x / syminfo.mintick or n * ta.atr(m) / "
                              "syminfo.mintick).")
            if k == "pct":
                trail_f["trailing_stop"] = float(v)
                desc = f"trailing stop {v:.2%} from the best price"
            elif k == "atr":
                trail_f["trailing_atr"], trail_f["atr_period"], trail_f["current_atr"] = float(v), per, True
                desc = f"trailing stop {v:g} x ATR({per}) (the current ATR) from the best price"
                notes.append(f"line {line}: the trailing distance follows the current ATR({per}) (TradingView re-evaluates "
                             "strategy.exit on every bar).")
            else:
                trail_f["trailing_points"] = float(v)
                desc = f"trailing stop ${v:g} from the best price"
            if "trail_points" in a:
                k2, v2, _ = _ticks(a["trail_points"], ctx, line, "trail_points", tick)
                if k2 == "pct" and v2 > 0:
                    trail_f["trail_activation"] = float(v2)
                    desc += f", once the price is +{v2:.2%} from the entry"
                elif k2 == "pts" and v2 > 0:
                    trail_f["trail_activation_points"] = float(v2)
                    desc += f", once the price is ${v2:g} from the entry"
                elif k2 not in ("pct", "pts"):
                    _refuse(line, f"trail_points={ast.unparse(a['trail_points'])} is not supported (use a number of ticks "
                                  "or close * x / syminfo.mintick).")
            if "trail_price" in a:
                k2, v2, _ = _level(a["trail_price"], "limit", s1, ctx, line)
                mm = re.fullmatch(r"entry_price ([-+]) (\d+(?:\.\d+)?)", str(v2).strip()) if k2 == "expr" and len(sides_x) == 1 else None
                if k2 == "pct":
                    trail_f["trail_activation"] = float(v2)
                    desc += f", once the price reaches +{v2:.2%} from the entry (trail_price)"
                elif mm and (mm.group(1) == "+") == (s1 == 1):
                    trail_f["trail_activation_points"] = float(mm.group(2))
                    desc += f", once the price is ${float(mm.group(2)):g} from the entry (trail_price)"
                else:
                    _refuse(line, f"trail_price={ast.unparse(a['trail_price'])} is not supported: write it from the entry "
                                  "price, e.g. strategy.position_avg_price * 1.05.")
            got.append(desc)
        # which part of the position the stop / trailing stop covers
        if qp < 100:
            if stop_f is not None:
                part_stop.append((stop_f, line))
            if trail_f is not None:
                part_trail.append((trail_f, line))
            if "limit" not in a and "profit" not in a:
                _refuse(line, "a partial exit (qty_percent) needs a target (limit= or profit=): it is read as a scale-out.")
        else:
            if stop_f is not None:
                for f_, v_ in stop_f.items():
                    if f_ == "stop_level":
                        for sg, t_ in v_.items():
                            if lvl_expr["stop_level"].get(sg, t_) != t_:
                                _refuse(line, "two different stops for the same entries are not supported.")
                            lvl_expr["stop_level"][sg] = t_
                    else:
                        put(f_, v_, line)
                rest_stop.update({k_: v_ for k_, v_ in stop_f.items()})
            if trail_f is not None:
                for f_, v_ in trail_f.items():
                    put(f_, v_, line)
                rest_trail.update(trail_f)
        if not got:
            _refuse(line, "strategy.exit() without a stop, limit, profit, loss or trailing stop has no effect.")
        notes.append(f"line {line}{' (qty_percent=' + format(qp, 'g') + ')' if qp < 100 else ''}: strategy.exit -> {', '.join(got)}")
    # a stop on part of the position: TradingView's exits each cover their own quantity
    uncovered = []
    for lst, rest, what in ((part_stop, rest_stop, "stop"), (part_trail, rest_trail, "trailing stop")):
        for f_, ln in lst:
            if not rest:
                _refuse(ln, f"a {what} on a partial exit (qty_percent) only, with none on the rest of the position, is not "
                            "supported; put it on the exit for the rest too (or on every exit).")
            if f_ != rest:
                _refuse(ln, f"the partial exit's {what} differs from the {what} on the rest of the position; different "
                            f"{what}s for parts of one position are not supported.")
    for lvl, frac, ln in scale:
        if (rest_stop or rest_trail) and not any(l2 == ln for _, l2 in part_stop + part_trail):
            uncovered.append(ln)
    if uncovered:
        if any(l2 not in uncovered for _, _, l2 in scale):
            _refuse(uncovered[0], "some partial exits carry the stop and others don't; give every partial exit the same "
                                  "stop, or none.")
        ex["stop_covers_scale_outs"] = False
        notes.append(f"line {', '.join(map(str, uncovered))}: the partial exit (qty_percent) has no stop, so - as in "
                     "TradingView, where each strategy.exit covers its own quantity - the stop and target cover only the "
                     "rest of the position; the scale-out's shares leave only at its limit (or by a rule exit / the end).")
    for f_ in ("stop_level", "target_level"):
        v_ = lvl_expr[f_]
        if not v_:
            continue
        if side == "both":
            if set(v_) != {1, -1}:
                raise PineImportError(f"Pine script: the {f_.replace('_', ' ')} is given for one side only of a long/short "
                                      "script; give both sides one.")
            ex[f_] = v_[1] if v_[1] == v_[-1] else f"where(side > 0, {v_[1]}, {v_[-1]})"
        else:
            ex[f_] = next(iter(v_.values()))
        dyn = bool(expr_names(ex[f_]) - {"entry_price", "side", "stop_price"})
        if dyn:
            ex["dynamic_levels"] = True
    if ex.get("dynamic_levels"):
        notes.append("The stop / target level is a price expression: as TradingView re-evaluates strategy.exit() on every "
                     "bar, it is recomputed on every bar from the previous close (dynamic levels); entry_price is "
                     "strategy.position_avg_price.")
    if scale:
        if sum(f for _, f, _ in scale) > 1 + 1e-9:
            raise PineImportError("The partial exits (qty_percent) add up to more than 100%.")
        left = 1.0
        so = []
        for lvl, frac, _ in sorted(scale):
            so.append({"at": lvl, "fraction": min(frac / left, 1.0)})
            left -= frac
        ex["scale_out"] = so
    if ex.get("atr_period") is None:
        ex.pop("atr_period", None)

    # ---- the spec
    notes += [n for n in ctx.notes if n not in notes]     # notes the stop / target translation added
    if not (exit_rule or ex or side == "both"):
        exit_rule = "False"
        notes.append("The script never exits (no strategy.close / strategy.exit): the position is held to the end, as in "
                     "TradingView.")
    if skipped:
        notes.append(f"Display-only lines skipped (plots, colours, alerts): {', '.join(map(str, skipped[:12]))}"
                     f"{' and more' if len(skipped) > 12 else ''}.")
    notes.append("TradingView-compatible mode: the Pine script was translated to the rules above and runs with "
                 "TradingView's conventions (next-open fills unless process_orders_on_close, no interest, no dividends).")
    strat = Strategy(
        universe=[tick], entry=long_rule if side != "short" else short_rule,
        short_entry=short_rule if side == "both" else None, side=side, reverse=True,
        entry_fill=fill, exit_when=exit_rule, exit_when_fill="close" if poc else "next_open",
        tv_compat=True, cash_rate=None, dividends=False, start=ctx.start, end=ctx.end,
        name=str(title)[:80] if title else "", description=f"Pine script: {title or 'strategy'} on {tick}",
        notes=[f"Pine import: {n}" for n in notes],
        **ex, **kw)
    return strat


# ------------------------------------------------------------------ var state, bar by bar

_STATE_FUNCS = {"maximum", "minimum", "abs", "nz", "na", "where", "log", "sqrt", "ref"}
_STATE_CACHE: dict = {}


def _truthy(v) -> bool:
    return v == v and v != 0


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _state_node(node, names: set, build, line_txt: str):
    """A closure i -> value of a var's value / condition: the parts that read no var are whole series (computed once,
    causal indicators), read at bar i; the rest is evaluated on bar i's numbers (a var reads its value so far this bar;
    ref(var, k) its value at the end of the bar k bars back)."""
    def uses(n):
        return any(isinstance(x, ast.Name) and (x.id in names or x.id == "na") for x in ast.walk(n))

    def comp(n):
        if isinstance(n, ast.Constant):
            v = _num(n.value)
            return lambda i: v
        if isinstance(n, ast.Name):
            if n.id in names:
                return build("var", n.id)
            if n.id == "na":
                return lambda i: np.nan
            if n.id in ("True", "False"):
                v = 1.0 if n.id == "True" else 0.0
                return lambda i: v
        if not uses(n):
            arr = build("series", ast.unparse(n))
            return lambda i: arr[i]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _STATE_FUNCS and not n.keywords:
            f, args = n.func.id, n.args
            if f == "ref":
                if not (len(args) == 2 and isinstance(args[0], ast.Name) and args[0].id in names
                        and isinstance(args[1], ast.Constant) and isinstance(args[1].value, int) and args[1].value >= 0):
                    raise ValueError(f"{line_txt}: {ast.unparse(n)}: the history of a var is read as name[k] with a whole "
                                     "number k.")
                if args[1].value == 0:
                    return build("var", args[0].id)
                return build("hist", (args[0].id, args[1].value))
            cs = [comp(a) for a in args]
            if f == "maximum" and len(cs) >= 2:
                return lambda i: (lambda vs: np.nan if any(v != v for v in vs) else max(vs))([c(i) for c in cs])
            if f == "minimum" and len(cs) >= 2:
                return lambda i: (lambda vs: np.nan if any(v != v for v in vs) else min(vs))([c(i) for c in cs])
            if f == "abs" and len(cs) == 1:
                return lambda i: abs(cs[0](i))
            if f in ("log", "sqrt") and len(cs) == 1:
                g = math.log if f == "log" else math.sqrt
                return lambda i: (lambda v: g(v) if v == v and v > 0 else np.nan)(cs[0](i))
            if f == "nz" and len(cs) in (1, 2):
                d = cs[1] if len(cs) == 2 else (lambda i: 0.0)
                return lambda i: (lambda v: v if v == v else d(i))(cs[0](i))
            if f == "na" and len(cs) == 1:
                return lambda i: 0.0 if cs[0](i) == cs[0](i) else 1.0
            if f == "where" and len(cs) == 3:
                return lambda i: cs[1](i) if _truthy(cs[0](i)) else cs[2](i)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)):
            l_, r_ = comp(n.left), comp(n.right)
            op = type(n.op)

            def bin_(i):
                a_, b_ = l_(i), r_(i)
                if a_ != a_ or b_ != b_:
                    return np.nan
                try:
                    if op is ast.Add:
                        return a_ + b_
                    if op is ast.Sub:
                        return a_ - b_
                    if op is ast.Mult:
                        return a_ * b_
                    if op is ast.Div:
                        return a_ / b_
                    if op is ast.Mod:
                        return a_ % b_
                    return a_ ** b_
                except (ZeroDivisionError, OverflowError, ValueError):
                    return np.nan
            return bin_
        if isinstance(n, ast.UnaryOp):
            c_ = comp(n.operand)
            if isinstance(n.op, ast.USub):
                return lambda i: -c_(i)
            if isinstance(n.op, ast.UAdd):
                return c_
            if isinstance(n.op, (ast.Not, ast.Invert)):
                return lambda i: 0.0 if _truthy(c_(i)) else 1.0
        if isinstance(n, ast.BoolOp):
            cs = [comp(v) for v in n.values]
            if isinstance(n.op, ast.And):
                return lambda i: 1.0 if all(_truthy(c(i)) for c in cs) else 0.0
            return lambda i: 1.0 if any(_truthy(c(i)) for c in cs) else 0.0
        if isinstance(n, ast.Compare):
            cs = [comp(n.left)] + [comp(x) for x in n.comparators]
            ops = [type(o) for o in n.ops]
            fn = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b,
                  ast.GtE: lambda a, b: a >= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}

            def cmp_(i):
                vs = [c(i) for c in cs]
                if any(v != v for v in vs):
                    return 0.0
                return 1.0 if all(fn[o](vs[j], vs[j + 1]) for j, o in enumerate(ops)) else 0.0
            return cmp_
        raise ValueError(f"{line_txt}: {ast.unparse(n)} is not supported in a var's update (a var may be combined "
                         "with + - * /, comparisons, and / or / not, math.max / min / abs, nz, na and cond ? a : b; "
                         "an indicator of a var would need its whole history).")
    return comp(node)


def check_state_vars(spec) -> None:
    """A Strategy.state_vars list: [{"name", "type" (float / int / bool), "init" (rule text or number), "updates":
    [{"value": rule text, "when": [[condition id, rule text], ...], "line": order}]}], names read by the rules."""
    from .expr import compile_expr, pine_to_rule
    if not isinstance(spec, list):
        raise ValueError("state_vars must be a list of var definitions")
    names = set()
    for v in spec:
        if not isinstance(v, dict) or not re.fullmatch(r"[A-Za-z_]\w*", str(v.get("name", ""))):
            raise ValueError(f"state_vars: each var needs a name (got {v!r})")
        if v.get("type", "float") not in ("float", "int", "bool"):
            raise ValueError(f"state_vars {v['name']}: the type must be float, int or bool")
        names.add(v["name"])
    for v in spec:
        texts = [v.get("init")] + [u.get("value") for u in v.get("updates") or []]
        texts += [w[1] for u in v.get("updates") or [] for w in u.get("when") or []]
        for t in texts:
            if t is None or (isinstance(t, (int, float)) and not isinstance(t, bool)):
                continue
            if not isinstance(t, str):
                raise ValueError(f"state_vars {v['name']}: {t!r} is not an expression")
            compile_expr(t)
            _state_node(ast.parse(pine_to_rule(t).strip(), mode="eval").body, names,
                        lambda kind, x: (lambda i: 0.0) if kind != "series" else np.zeros(1), f"var {v['name']}")


def state_series(spec: list, ns) -> dict:
    """The vars of a Pine script as daily series: from each var's initial value, the updates run in script order on
    every bar, reading only that bar and the bars before it (causal by construction), and the value at the end of each
    bar is the series' value. Booleans come back as bool series."""
    import json
    from .expr import evaluate_value, pine_to_rule
    df = ns.quoted_df
    key = (json.dumps(spec, sort_keys=True, default=str), ns.ticker, id(df), len(ns.df.index))
    hit = _STATE_CACHE.get(key)
    if hit is not None and hit[0] is df:
        return {k: v.copy() for k, v in hit[1].items()}
    names = {v["name"] for v in spec}
    n = len(ns.df.index)
    cur = {nm: np.nan for nm in names}
    hist = {nm: np.full(n, np.nan) for nm in names}
    arrays: dict = {}

    def build(kind, x):
        if kind == "var":
            return lambda i: cur[x]
        if kind == "hist":
            nm, k = x
            h = hist[nm]
            return lambda i: h[i - k] if i - k >= 0 else np.nan
        if x not in arrays:
            arrays[x] = evaluate_value(x, ns).to_numpy(dtype=float)
        return arrays[x]

    def compile_text(t, what):
        if t is None:
            return lambda i: np.nan
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            return lambda i, _v=float(t): _v
        return _state_node(ast.parse(pine_to_rule(str(t)).strip(), mode="eval").body, names, build, what)

    steps = []
    for vi, v in enumerate(spec):
        for ui, u in enumerate(v.get("updates") or []):
            whens = [(w[0], compile_text(w[1], f"var {v['name']}")) for w in u.get("when") or []]
            steps.append((u.get("line", 0), vi, ui, v["name"], whens, compile_text(u.get("value"), f"var {v['name']}"),
                          v.get("type", "float")))
    steps.sort(key=lambda x: (x[0], x[1], x[2]))
    inits = [(v["name"], compile_text(v.get("init"), f"var {v['name']}"), v.get("type", "float")) for v in spec]
    for i in range(n):
        if i == 0:
            for nm, f, typ in inits:
                val = f(0)
                cur[nm] = (1.0 if _truthy(val) else 0.0) if typ == "bool" else val
        seen: dict = {}
        for _, _, _, nm, whens, f, typ in steps:
            ok = True
            for cid, w in whens:
                if cid not in seen:
                    seen[cid] = _truthy(w(i))
                if not seen[cid]:
                    ok = False
                    break
            if ok:
                val = f(i)
                if typ == "int" and val == val:
                    val = float(int(val))
                cur[nm] = (1.0 if _truthy(val) else 0.0) if typ == "bool" else val
        for nm in names:
            hist[nm][i] = cur[nm]
    out = {}
    for v in spec:
        s_ = pd.Series(hist[v["name"]], index=ns.df.index)
        out[v["name"]] = s_.fillna(0).astype(bool) if v.get("type") == "bool" else s_
    if len(_STATE_CACHE) > 256:
        _STATE_CACHE.clear()
    _STATE_CACHE[key] = (df, out)
    return {k: v.copy() for k, v in out.items()}
