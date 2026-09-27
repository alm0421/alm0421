"""TradingView Pine Script strategies -> a Strategy spec (a practical subset of Pine v4-v6).

Supported:
  strategy(...)            initial_capital, default_qty_type / default_qty_value, commission_type / commission_value,
                           pyramiding, process_orders_on_close, margin_long (the other display settings are ignored)
  input.*() / input()      the default value (inputs are constants of the backtest)
  x = <expression>         variables of ta.* / math.* / request.security expressions, conditions, numbers; used
                           anywhere later (with history: x[1])
  [a, b, c] = ta.macd(...) tuple results of ta.macd, ta.bb, ta.supertrend, ta.dmi, ta.kc
  strategy.entry(id, strategy.long / strategy.short, when=cond)   or inside `if cond` (nested ifs, else)
  strategy.close(id, when=...), strategy.close_all(...)             a rule exit
  strategy.exit(id, from_entry, stop=, limit=, profit=, loss=, trail_points=, trail_offset=, qty_percent=)
                           stops and targets as a percentage of strategy.position_avg_price or a multiple of ta.atr,
                           trailing stops by ATR or percent, partial exits (qty_percent) as scale-outs
  request.security(syminfo.tickerid, "W" / "M" / "D", x)           weekly(x) / monthly(x) / x (lookahead off)
  time >= timestamp(...)   a date filter becomes the start (end) of the test
Display-only calls (plot, bgcolor, alertcondition, label.new...) are skipped. Anything else - var / := state, custom
functions, loops, ternaries, strategy.order, limit / stop entries, dynamic stop levels - is refused with its line
number, never guessed. The result runs in TradingView-compatible mode.
"""
from __future__ import annotations

import ast
import copy
import re

import pandas as pd

from . import data
from .parser import ParseError
from .strategy import Strategy


class PineImportError(ParseError):
    pass


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


def _pyexpr(text: str, line: int) -> ast.AST:
    """A Pine expression as a Python AST (the operators and literals are the same for the supported subset)."""
    t = text.strip()
    if re.search(r":=", t):
        _refuse(line, "reassignment (:=) keeps state from bar to bar, which the importer does not support.")
    q = None
    for ch in t:
        if q:
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch == "?":
            _refuse(line, "the ternary operator (cond ? a : b) is not supported; write the condition out, e.g. with and / or.")
    t = re.sub(r"\btrue\b", "True", t)
    t = re.sub(r"\bfalse\b", "False", t)
    try:
        return ast.parse(t, mode="eval").body
    except SyntaxError:
        _refuse(line, f"could not read the expression {text.strip()!r}.")


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

    def visit_Name(self, n):
        if n.id in self.ctx.sym:
            return self.visit(copy.deepcopy(self.ctx.sym[n.id]))
        if n.id == "bar_index":
            _refuse(self.line, "bar_index is only supported as bar_index - strategy.opentrades.entry_bar_index(0) (the "
                               "bars since the entry) in an exit.")
        if n.id in ("time", "time_close", "timenow"):
            _refuse(self.line, f"{n.id} is only supported in a date filter: time >= timestamp(2015, 1, 1).")
        if n.id == "na":
            _refuse(self.line, "na as a value is not supported.")
        return n

    def visit_Call(self, n):
        name = _call_name(n)
        if name == "input" or name.startswith("input."):
            return self.visit(copy.deepcopy(_input_default(n, self.line)))
        if name in ("timestamp",):
            return n
        if name.startswith("strategy."):
            _refuse(self.line, f"{name}() can't be used inside an expression.")
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

def _level(node: ast.AST, kind: str, side: int, ctx: _Ctx, line: int) -> tuple[str, float, int | None]:
    """A strategy.exit stop / limit price as ('pct', fraction, None) or ('atr', multiple, period), from
    strategy.position_avg_price * (1 -/+ x), * k, or -/+ n * ta.atr(m)."""
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
                ctx.notes.append(f"line {line}: the {'target' if kind == 'limit' else 'stop'} {ast.unparse(node)} uses the "
                                 f"ATR({int(per)}) of the entry (the backtest fixes it when the position opens; a strategy.exit "
                                 "called on every bar in TradingView moves it with the current ATR).")
                return "atr", float(mult), int(per)
        # P + x (a price distance) or P * x/100 written as P + P * x
        if isinstance(r, ast.BinOp) and isinstance(r.op, ast.Mult) and (is_p(r.left) or is_p(r.right)) and up == want_up:
            k = _const(r.right if is_p(r.left) else r.left)
            if isinstance(k, (int, float)) and k > 0:
                return "pct", float(k), None
    _refuse(line, f"the {'target (limit)' if kind == 'limit' else 'stop'} {ast.unparse(node)!r} is not supported: write it "
                  "from the entry price, e.g. strategy.position_avg_price * (1 - 0.05), or strategy.position_avg_price - 2 * "
                  "ta.atr(14) (a level that moves with the market, recomputed on every bar, can't be imported).")


def _ticks(node: ast.AST, ctx: _Ctx, line: int, what: str) -> tuple[str, float, int | None]:
    """profit= / loss= / trail_offset= in ticks, written as a price distance / syminfo.mintick: close * x (a percentage)
    or n * ta.atr(m)."""
    n = _Inputs(ctx, line).visit(copy.deepcopy(node))
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
                return "pct", float(k), None
            if _call_name(a) == "ta.atr":
                per = _const(a.args[0]) if a.args else 14
                if isinstance(per, (int, float)) and float(per).is_integer():
                    return "atr", float(k), int(per)
    _refuse(line, f"{what}={ast.unparse(node)} is in ticks; write it as a price distance divided by syminfo.mintick, "
                  "e.g. close * 0.05 / syminfo.mintick (5%) or 2 * ta.atr(14) / syminfo.mintick (a fixed tick count "
                  "depends on the symbol's tick size, which the backtest does not have).")


class _Inputs(ast.NodeTransformer):
    """Variables and input.*() calls replaced by their definitions / default values, and nothing else (the numbers
    inside a stop / limit, a qty, a setting)."""

    def __init__(self, ctx, line):
        self.ctx, self.line = ctx, line

    def visit_Name(self, n):
        v = self.ctx.sym.get(n.id)
        if v is not None:
            return self.visit(copy.deepcopy(v))
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
        if re.match(r"^[A-Za-z_]\w*\s*\([^()]*\)\s*=>", s):
            _refuse(line, "custom functions (name(...) =>) are not supported; write the expression where it is used.")
        if re.match(r"^(var|varip)\b", s):
            _refuse(line, f"'{s.split()[0]}' declares a variable that keeps its value from bar to bar; the importer "
                          "only reads expressions recomputed on every bar.")
        if re.match(r"^[A-Za-z_][\w.]*\s*:=", s) or re.search(r"(?<![=!<>:])[-+*/]=", s):
            _refuse(line, "reassignment (:=, +=) keeps state from bar to bar, which the importer does not support.")
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
            if name == "strategy.entry":
                try:
                    a = _args(call, ENTRY_ARGS + ["when"])
                except ValueError:
                    _refuse(line, "strategy.entry(): too many arguments.")
                if "limit" in a or "stop" in a:
                    _refuse(line, "strategy.entry(..., limit= / stop=) entry orders are not supported by the importer; "
                                  "describe the entry as a sentence instead (e.g. 'buy SPY with a limit order 2% below "
                                  "the close when ...').")
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
                e = entries.setdefault(eid, {"side": side, "conds": [], "line": line})
                if e["side"] != side:
                    _refuse(line, f"the entry id {eid!r} is used for both long and short entries.")
                e["conds"].append((cs, line))
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
                              "the top level.")
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
    scale: list[tuple[float, float]] = []            # (level fraction, qty fraction of the entry)
    for x in exits:
        a, line = x["a"], x["line"]
        sd = ids_side([x["from"]] if x["from"] else None, line)
        if side == "both" and sd != {1, -1} and x["from"]:
            others = [y for y in exits if y is not x]
            if not any(set(ids_side([y["from"]] if y["from"] else None, y["line"])) - sd for y in others):
                raise PineImportError(f"Pine script line {line}: strategy.exit() for one side only of a long/short script: "
                                      "the backtest's stops and targets apply to both sides. Give both sides the same exit.")
        s1 = 1 if sd == {1} else (-1 if sd == {-1} else (1 if side != "short" else -1))
        for c in x["cs"]:
            r = _rule(c, ctx, line, "exit-long" if s1 == 1 else "exit-short")
            if r != "True":
                _refuse(line, "a strategy.exit() placed only under a condition (other than 'in a position') is not "
                              "supported: the backtest's stops and targets apply to every trade.")
        for bad in ("qty", "trail_price"):
            if bad in a:
                _refuse(line, f"strategy.exit({bad}=...) is not supported.")
        qp = _const(_Inputs(ctx, line).visit(copy.deepcopy(a["qty_percent"]))) if "qty_percent" in a else 100
        if not isinstance(qp, (int, float)) or not 0 < qp <= 100:
            _refuse(line, "strategy.exit(qty_percent=...) must be a number between 0 and 100.")
        got = []
        if "stop" in a or "loss" in a:
            k, v, per = _level(a["stop"], "stop", s1, ctx, line) if "stop" in a else _ticks(a["loss"], ctx, line, "loss")
            key = ("stop_loss", v) if k == "pct" else ("stop_atr", v)
            if qp < 100:
                _refuse(line, "a stop on part of the position (qty_percent) is not supported; put the stop on the whole "
                              "position.")
            if key[0] in ex and abs(ex[key[0]] - v) > 1e-12:
                _refuse(line, "two different stops for the same entries are not supported.")
            ex[key[0]] = v
            if per:
                ex["atr_period"] = per
            got.append(f"{'stop loss ' + format(v, '.2%') if k == 'pct' else f'stop {v:g} x ATR({per})'}")
        if "limit" in a or "profit" in a:
            k, v, per = _level(a["limit"], "limit", s1, ctx, line) if "limit" in a else _ticks(a["profit"], ctx, line, "profit")
            if qp < 100:
                if k != "pct":
                    _refuse(line, "a partial exit (qty_percent) is supported at a percentage target only.")
                scale.append((v, qp / 100))
                got.append(f"sell {qp:g}% of the entry at +{v:.2%}")
            else:
                name = "take_profit" if k == "pct" else "take_profit_atr"
                if name in ex and abs(ex[name] - v) > 1e-12:
                    _refuse(line, "two different targets for the same entries are not supported.")
                ex[name] = v
                if per:
                    ex["atr_period"] = per
                got.append(f"take profit {'+' + format(v, '.2%') if k == 'pct' else f'{v:g} x ATR({per})'}")
        if "trail_offset" in a or "trail_points" in a:
            tp = _const(_Inputs(ctx, line).visit(copy.deepcopy(a["trail_points"]))) if "trail_points" in a else 0
            if tp not in (0, 0.0):
                _refuse(line, "strategy.exit(trail_points=...) (a trailing stop activated after a profit) is not "
                              "supported; use trail_points=0 (active from the entry) and trail_offset.")
            if "trail_offset" not in a:
                _refuse(line, "strategy.exit(trail_points=...) needs trail_offset.")
            k, v, per = _ticks(a["trail_offset"], ctx, line, "trail_offset")
            if qp < 100:
                _refuse(line, "a trailing stop on part of the position (qty_percent) is not supported.")
            ex["trailing_stop" if k == "pct" else "trailing_atr"] = v
            if per:
                ex["atr_period"] = per
            got.append(f"trailing stop {format(v, '.2%') if k == 'pct' else f'{v:g} x ATR({per})'} from the best price")
            if k == "atr":
                notes.append(f"line {line}: the trailing distance uses the ATR({per}) of the entry (TradingView uses the value "
                             "when strategy.exit() was last called).")
        for k2 in ("profit", "loss"):
            if k2 in a and k2.replace("profit", "limit").replace("loss", "stop") in a:
                _refuse(line, f"strategy.exit() has both {k2}= and "
                              f"{k2.replace('profit', 'limit').replace('loss', 'stop')}=; keep one.")
        if not got:
            _refuse(line, "strategy.exit() without a stop, limit, profit, loss or trailing stop has no effect.")
        notes.append(f"line {line}: strategy.exit -> {', '.join(got)}")
    if scale:
        if len({f for _, f in scale}) and sum(f for _, f in scale) > 1 + 1e-9:
            raise PineImportError("The partial exits (qty_percent) add up to more than 100%.")
        left = 1.0
        so = []
        for lvl, frac in sorted(scale):
            so.append({"at": lvl, "fraction": min(frac / left, 1.0)})
            left -= frac
        ex["scale_out"] = so
    if ex.get("atr_period") is None:
        ex.pop("atr_period", None)

    # ---- the spec
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
