"""Export a Portfolio spec as a Composer (composer.trade) symphony: the inverse of composer_import.

Composer's editor supports a subset of what the portfolio engine does. This module writes that subset:

  asset                                  {"step": "asset", "ticker": "SPY"}
  equal / specified / inverse_vol        wt-cash-equal / wt-cash-specified (weights as num/den) / wt-inverse-vol
  if ... then ... else                   an "if" block with if-children: one indicator of a ticker compared with a
                                         fixed number or with an indicator of a ticker (gt, gte, lt, lte);
                                         "and" / "or" / "not" become nested ifs and flipped comparators
  filter (top/bottom N by a metric)      a "filter" block: sort-by-fn over sort-by-window-days, select-fn, select-n
  cash in an else branch                 left out (Composer holds nothing there, the importer reads it as cash)

The indicators are Composer's: current price, moving average of price, exponential moving average of price,
cumulative return, moving average of return, standard deviation of price / return, max drawdown and RSI (the
return-based ones are in percent in Composer). Anything else - shorts, leverage, volatility targeting, index
universes, requirements on a filter, other indicators or weightings, cash as a holding - raises
ComposerExportError saying what Composer lacks. Settings that do not change the allocation (capital, costs,
dates, cash flows) are not part of a symphony; `export` returns notes naming what was left out.

    python -m backtester composer-export spec.json|"sentence" [--out symphony.json]
"""
from __future__ import annotations

import ast
import json

from . import data


class ComposerExportError(ValueError):
    pass


# rule-language function -> (Composer function, series it reads, percent-valued in Composer)
FUNCS = {
    "rsi": ("relative-strength-index", "close", False),
    "tret": ("cumulative-return", "tr", True),
    "sma": ("moving-average-price", "close", False),
    "ma": ("moving-average-price", "close", False),
    "ema": ("exponential-moving-average-price", "close", False),
    "ma_return": ("moving-average-return", "tr", True),
    "stdev_return": ("standard-deviation-return", "tr", True),
    "stdev": ("standard-deviation-price", "close", False),
    "max_drawdown": ("max-drawdown", "tr", True),
}
PCT_FNS = {v[0] for v in FUNCS.values() if v[2]}
CMP = {ast.Gt: "gt", ast.GtE: "gte", ast.Lt: "lt", ast.LtE: "lte"}
FLIP = {"gt": "lt", "gte": "lte", "lt": "gt", "lte": "gte"}       # a op b  ==  b FLIP[op] a
NEGATE = {"gt": "lte", "gte": "lt", "lt": "gte", "lte": "gt"}     # not (a op b)  ==  a NEGATE[op] b
REBALANCE = {"daily", "weekly", "monthly", "quarterly", "yearly", "none"}
SUPPORTED = ("current price, moving average of price (sma), exponential moving average of price (ema), cumulative "
             "return (tret), moving average of return, standard deviation of price or return, max drawdown and RSI")


def _ticker(t: str) -> str:
    t = data.canonical(t)
    if t.endswith("-USD"):
        return f"CRYPTO::{t[:-4]}//USD"
    return t


def _num(x) -> float | None:
    if isinstance(x, ast.UnaryOp) and isinstance(x.op, (ast.USub, ast.UAdd)):
        v = _num(x.operand)
        return None if v is None else (-v if isinstance(x.op, ast.USub) else v)
    if isinstance(x, ast.Constant) and isinstance(x.value, (int, float)) and not isinstance(x.value, bool):
        return float(x.value)
    return None


def _fmt(x: float):
    return int(x) if float(x).is_integer() else round(float(x), 10)


class _Exporter:
    def __init__(self):
        self.notes: list[str] = []

    def note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)

    # ------------------------------------------------------------ indicators
    def series_of(self, node, own: str | None, where: str) -> tuple[str | None, str]:
        """close / tr / sym("X").close -> (ticker or None for the rule's own, field)."""
        if isinstance(node, ast.Name) and node.id in ("close", "tr", "price"):
            return own, "close" if node.id == "price" else node.id
        if (isinstance(node, ast.Attribute) and node.attr in ("close", "tr") and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "sym" and len(node.value.args) == 1
                and isinstance(node.value.args[0], ast.Constant) and isinstance(node.value.args[0].value, str)):
            return data.canonical(node.value.args[0].value), node.attr
        raise ComposerExportError(f"{where}: `{ast.unparse(node)}` is not a price series Composer can read (close, tr or "
                                  "sym(\"X\").close).")

    def indicator(self, node, own: str | None, where: str) -> tuple[str, str | None, int | None]:
        """An expression -> (Composer function, ticker (None = the rule's own), window)."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "quoted" and len(node.args) == 1:
            return self.indicator(node.args[0], own, where)     # Composer compares prices as they are
        if isinstance(node, (ast.Name, ast.Attribute)):
            t, field = self.series_of(node, own, where)
            if field != "close":
                raise ComposerExportError(f"{where}: `{ast.unparse(node)}` (the total-return index level) has no "
                                          "Composer function; use its price or a return.")
            return "current-price", t, None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords:
            fn, want, _ = FUNCS[node.func.id]
            args = node.args
            if len(args) != 2 or _num(args[1]) is None:
                raise ComposerExportError(f"{where}: `{ast.unparse(node)}` needs a series and a whole-number window "
                                          f"for Composer, e.g. {node.func.id}({want}, 20).")
            t, field = self.series_of(args[0], own, where)
            if field != want and not (node.func.id == "tret" and field == "close"):
                raise ComposerExportError(f"{where}: Composer's {fn} reads {'total-return' if want == 'tr' else 'price'} "
                                          f"data; `{ast.unparse(node)}` reads {field}.")
            n = _num(args[1])
            if n < 1 or not float(n).is_integer():
                raise ComposerExportError(f"{where}: the window of `{ast.unparse(node)}` must be a whole number of days.")
            return fn, t, int(n)
        raise ComposerExportError(f"{where}: `{ast.unparse(node)}` has no Composer equivalent. Composer's conditions "
                                  f"and filters use: {SUPPORTED}.")

    # ------------------------------------------------------------ conditions
    def compare(self, node, own: str, where: str) -> dict:
        """One comparison -> the condition fields of an if-child."""
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or type(node.ops[0]) not in CMP:
            raise ComposerExportError(f"{where}: `{ast.unparse(node)}` is not a single >, >=, < or <= comparison "
                                      "(Composer has no ==, != or chained comparisons).")
        op = CMP[type(node.ops[0])]
        lhs, rhs = node.left, node.comparators[0]
        if _num(lhs) is not None and _num(rhs) is None:
            lhs, rhs, op = rhs, lhs, FLIP[op]
        fn, t, win = self.indicator(lhs, own, where)
        out = {"lhs-fn": fn, "lhs-val": _ticker(t or own), "comparator": op}
        if win is not None:
            out["lhs-window-days"] = win
        v = _num(rhs)
        if v is not None:
            out["rhs-fixed-value?"] = True
            out["rhs-val"] = _fmt(v * 100 if fn in PCT_FNS else v)
        else:
            rfn, rt, rwin = self.indicator(rhs, own, where)
            if (fn in PCT_FNS) != (rfn in PCT_FNS) and fn != rfn:
                self.note(f"{where}: {fn} is compared with {rfn}; Composer compares them as they are.")
            out.update({"rhs-fixed-value?": False, "rhs-fn": rfn, "rhs-val": _ticker(rt or own)})
            if rwin is not None:
                out["rhs-window-days"] = rwin
        return out

    def cond_tree(self, node, own: str, then: dict, other: dict | None, where: str) -> dict:
        """A boolean rule -> nested Composer if blocks ('and' / 'or' as nesting, 'not' by flipping)."""
        while isinstance(node, ast.Expression):
            node = node.body
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
            inner = node.operand
            if isinstance(inner, ast.BoolOp):   # De Morgan
                flipped = ast.BoolOp(op=ast.Or() if isinstance(inner.op, ast.And) else ast.And(),
                                     values=[ast.UnaryOp(op=ast.Not(), operand=v) for v in inner.values])
                return self.cond_tree(flipped, own, then, other, where)
            if isinstance(inner, ast.UnaryOp) and isinstance(inner.op, (ast.Not, ast.Invert)):
                return self.cond_tree(inner.operand, own, then, other, where)
            c = self.compare(inner, own, where)
            c["comparator"] = NEGATE[c["comparator"]]
            return self.if_block(c, then, other)
        if isinstance(node, ast.BoolOp):
            first, rest = node.values[0], node.values[1:]
            rest_node = rest[0] if len(rest) == 1 else ast.BoolOp(op=node.op, values=rest)
            if isinstance(node.op, ast.And):     # if A then (if B then X else Y) else Y
                return self.cond_tree(first, own, self.cond_tree(rest_node, own, then, other, where), other, where)
            return self.cond_tree(first, own, then, self.cond_tree(rest_node, own, then, other, where), where)
        return self.if_block(self.compare(node, own, where), then, other)

    @staticmethod
    def if_block(cond: dict, then: dict, other: dict | None) -> dict:
        kids = [{"step": "if-child", "is-else-condition?": False, **cond, "children": [then]}]
        if other is not None:
            kids.append({"step": "if-child", "is-else-condition?": True, "children": [other]})
        return {"step": "if", "children": kids}

    # ------------------------------------------------------------ nodes
    def node(self, n: dict, where: str) -> dict | None:
        """A tree node -> a Composer block (None for cash in an else branch, which Composer leaves empty)."""
        if not isinstance(n, dict):
            raise ComposerExportError(f"{where}: not a node")
        if "asset" in n:
            return {"step": "asset", "ticker": _ticker(n["asset"])}
        if n.get("cash"):
            raise ComposerExportError(f"{where}: Composer has no cash holding. Hold a T-bill fund such as BIL or SHV "
                                      "instead (an 'otherwise cash' branch is fine: Composer leaves it empty).")
        if "custom" in n:
            raise ComposerExportError(f"{where}: a Python function node cannot be exported to Composer.")
        if "weights" in n:
            kids = n.get("children") or []
            m = n["weights"]
            if m == "equal":
                return {"step": "wt-cash-equal", "children": [self.node(k, f"{where} > {i + 1}") for i, k in enumerate(kids)]}
            if m == "specified":
                ws = [float(w) for w in n["w"]]
                if any(w < 0 for w in ws):
                    raise ComposerExportError(f"{where}: negative weights are short positions, which Composer does not "
                                              "support.")
                if sum(ws) > 1 + 1e-9:
                    raise ComposerExportError(f"{where}: the weights add up to {sum(ws):.0%}; Composer cannot borrow "
                                              "(leverage). Use leveraged funds instead.")
                out = []
                cash = 0.0
                for i, (w, k) in enumerate(zip(ws, kids)):
                    if isinstance(k, dict) and k.get("cash"):
                        cash += w
                        continue
                    b = self.node(k, f"{where} > {i + 1}")
                    b["weight"] = {"num": _fmt(round(w * 100, 8)), "den": 100}
                    out.append(b)
                if cash > 1e-9 or sum(ws) < 1 - 1e-9:
                    raise ComposerExportError(f"{where}: {max(cash, 1 - sum(ws) + cash):.0%} is held in cash; Composer "
                                              "has no cash holding. Give that slice to BIL (T-bills) instead.")
                return {"step": "wt-cash-specified", "children": out}
            if m == "inverse_vol":
                return {"step": "wt-inverse-vol", "window-days": int(n.get("lookback") or 20),
                        "children": [self.node(k, f"{where} > {i + 1}") for i, k in enumerate(kids)]}
            raise ComposerExportError(f"{where}: {m.replace('_', ' ')} weighting is not available in Composer (it has "
                                      "equal, specified and inverse-volatility weights).")
        if "if" in n:
            on = data.canonical(n.get("on", "SPY"))
            try:
                tree = ast.parse(str(n["if"]).strip(), mode="eval")
            except SyntaxError as e:
                raise ComposerExportError(f"{where}: the condition `{n['if']}` is not valid: {e.msg}")
            then = self.branch(n["then"], f"{where} > then")
            other = self.branch(n["else"], f"{where} > else")
            if then is None:
                raise ComposerExportError(f"{where}: the 'then' branch holds cash; Composer has no cash holding. Swap the "
                                          "condition so cash is the else branch, or hold BIL.")
            return self.cond_tree(tree, on, then, other, where)
        if "filter" in n:
            f = n["filter"]
            u = n.get("universe", "children")
            if u in ("NDX", "nasdaq100"):
                raise ComposerExportError(f"{where}: Composer has no point-in-time Nasdaq-100 universe; list the "
                                          "tickers to choose from instead.")
            if f.get("require"):
                raise ComposerExportError(f"{where}: the requirement `{f['require']}` on the chosen assets has no "
                                          "Composer equivalent (a filter only ranks and selects).")
            if (f.get("weights") or "equal") != "equal":
                raise ComposerExportError(f"{where}: Composer weights a filter's picks equally, not by "
                                          f"{f['weights'].replace('_', ' ')}.")
            fb = n.get("fallback")
            if fb and not (isinstance(fb, dict) and fb.get("cash")):
                raise ComposerExportError(f"{where}: a filter's fallback holding has no Composer equivalent.")
            try:
                by = ast.parse(str(f["by"]).strip(), mode="eval").body
            except SyntaxError as e:
                raise ComposerExportError(f"{where}: the ranking `{f['by']}` is not valid: {e.msg}")
            fn, t, win = self.indicator(by, None, where)
            if t is not None:
                raise ComposerExportError(f"{where}: a filter ranks each candidate by its own indicator; "
                                          f"`{f['by']}` reads another ticker.")
            kids = n.get("children") if u == "children" else [{"asset": x} for x in u]
            blk = {"step": "filter", "sort-by-fn": fn, "select-fn": f.get("select", "top"), "select-n": int(f.get("n", 1)),
                   "children": [self.node(k, f"{where} > {i + 1}") for i, k in enumerate(kids or [])]}
            if win is not None:
                blk["sort-by-window-days"] = win
            return blk
        raise ComposerExportError(f"{where}: unknown node {sorted(n)}")

    def branch(self, n: dict, where: str) -> dict | None:
        return None if isinstance(n, dict) and n.get("cash") else self.node(n, where)


def export(spec) -> tuple[dict, list[str]]:
    """A Portfolio (or its dict) -> (Composer symphony JSON as a dict, notes on what was left out)."""
    from .portfolio import Portfolio
    if isinstance(spec, dict):
        spec = Portfolio.from_dict({k: v for k, v in spec.items() if k != "kind"})
    if not isinstance(spec, Portfolio):
        raise ComposerExportError("Only allocation portfolios can be exported to Composer; a trading rule "
                                  "(buy when ..., sell when ...) has no Composer equivalent.")
    ex = _Exporter()
    if spec.leverage != 1:
        raise ComposerExportError(f"Composer cannot borrow: leverage {spec.leverage:g}x has no equivalent (use leveraged "
                                  "funds instead).")
    if getattr(spec, "target_vol", None):
        raise ComposerExportError("Volatility targeting has no Composer equivalent.")
    if spec.rebalance not in REBALANCE:
        raise ComposerExportError(f"Composer rebalances daily, weekly, monthly, quarterly or yearly (or on a threshold), "
                                  f"not {spec.rebalance.replace('_', ' ')}.")
    if spec.drift_band_relative:
        raise ComposerExportError("Composer's threshold rebalancing uses an absolute corridor, not a relative one.")
    root: dict = {"step": "root", "name": (spec.name or spec.description or "Exported portfolio")[:80],
                  "description": spec.description or "", "rebalance": spec.rebalance}
    if spec.drift_band:
        if spec.rebalance != "none":
            raise ComposerExportError("Composer uses a rebalance corridor only instead of a calendar schedule; say "
                                      "'never rebalance' together with the drift band.")
        root["rebalance-corridor-width"] = _fmt(round(spec.drift_band * 100, 8))
    elif spec.rebalance == "none":
        raise ComposerExportError("Composer has no buy-and-hold (never rebalance) setting without a corridor width.")
    top = ex.branch(spec.tree, "portfolio")
    if top is None:
        raise ComposerExportError("The portfolio holds only cash.")
    root["children"] = [top]
    if spec.price_basis != "adjusted":
        ex.note("Composer computes indicators on total-return (dividend-adjusted) prices; this spec used prices as "
                "quoted, so RSI and moving averages of dividend payers can differ slightly.")
    if spec.fill != "close":
        ex.note("Composer trades near the close of the rebalance day; this spec traded at the next open.")
    if spec.contribution or spec.withdrawal or spec.withdrawal_pct:
        ex.note("Cash flows (contributions, withdrawals) are account settings, not part of a symphony: left out.")
    if spec.slippage_bps or spec.commission or spec.commission_pct:
        ex.note("Trading costs are not part of a symphony: left out.")
    if spec.start or spec.end:
        ex.note("The backtest period is chosen in Composer's backtest view, not stored in the symphony.")
    return root, ex.notes


def export_json(spec, indent: int = 2) -> str:
    sym, _ = export(spec)
    return json.dumps(sym, indent=indent)
