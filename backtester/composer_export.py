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
import uuid

from . import data

# namespace of the stable block ids generated for nodes that carry no Composer id (uuid5 of the symphony's name and
# the node's place in the tree: exporting the same portfolio twice gives the same ids)
ID_NAMESPACE = uuid.UUID("5b0c3f0e-6a52-4d1e-9f4c-1f2a8f3c9d01")


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
    def __init__(self, seed: str = ""):
        self.notes: list[str] = []
        self.seed = seed
        self._seen: dict[str, int] = {}

    def block_id(self, n, key: str, where: str) -> str:
        """The Composer id of the block node `n` became: the one it was imported with (n["composer"][key]), else a
        stable UUID made from the symphony's name and the node's place in the tree."""
        cm = n.get("composer") if isinstance(n, dict) else None
        v = cm.get(key) if isinstance(cm, dict) else None
        if isinstance(v, str) and v.strip():
            return v
        k = f"{where}|{key}"
        self._seen[k] = self._seen.get(k, 0) + 1
        return str(uuid.uuid5(ID_NAMESPACE, f"{self.seed}|{k}|{self._seen[k]}"))

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
        # Composer's current exports give a window both as "lhs-fn-params": {"window": n} and as the older
        # "lhs-window-days": "n" (text), and a fixed value as text ("rhs-val": "79")
        out = {"lhs-fn": fn, "lhs-val": _ticker(t or own), "comparator": op}
        if win is not None:
            out["lhs-fn-params"] = {"window": win}
            out["lhs-window-days"] = str(win)
        v = _num(rhs)
        if v is not None:
            out["rhs-fixed-value?"] = True
            out["rhs-val"] = str(_fmt(v * 100 if fn in PCT_FNS else v))
        else:
            rfn, rt, rwin = self.indicator(rhs, own, where)
            if (fn in PCT_FNS) != (rfn in PCT_FNS) and fn != rfn:
                self.note(f"{where}: {fn} is compared with {rfn}; Composer compares them as they are.")
            out.update({"rhs-fixed-value?": False, "rhs-fn": rfn, "rhs-val": _ticker(rt or own)})
            if rwin is not None:
                out["rhs-fn-params"] = {"window": rwin}
                out["rhs-window-days"] = str(rwin)
        return out

    def cond_tree(self, node, own: str, then: dict, other: dict | None, where: str, ids: dict | None = None) -> dict:
        """A boolean rule -> nested Composer if blocks ('and' / 'or' as nesting, 'not' by flipping). `ids` (the
        if node's own: id, then_id, else_id) go to the outermost block; nested ones get generated ids."""
        while isinstance(node, ast.Expression):
            node = node.body
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
            inner = node.operand
            if isinstance(inner, ast.BoolOp):   # De Morgan
                flipped = ast.BoolOp(op=ast.Or() if isinstance(inner.op, ast.And) else ast.And(),
                                     values=[ast.UnaryOp(op=ast.Not(), operand=v) for v in inner.values])
                return self.cond_tree(flipped, own, then, other, where, ids)
            if isinstance(inner, ast.UnaryOp) and isinstance(inner.op, (ast.Not, ast.Invert)):
                return self.cond_tree(inner.operand, own, then, other, where, ids)
            c = self.compare(inner, own, where)
            c["comparator"] = NEGATE[c["comparator"]]
            return self.if_block(c, then, other, where, ids)
        if isinstance(node, ast.BoolOp):
            first, rest = node.values[0], node.values[1:]
            rest_node = rest[0] if len(rest) == 1 else ast.BoolOp(op=node.op, values=rest)
            # the shared branch appears twice in Composer's nesting: the second copy gets ids of its own
            if isinstance(node.op, ast.And):     # if A then (if B then X else Y) else Y
                inner = self.cond_tree(rest_node, own, then, self.copy_of(other, where), where)
                return self.cond_tree(first, own, inner, other, where, ids)
            inner = self.cond_tree(rest_node, own, self.copy_of(then, where), other, where)
            return self.cond_tree(first, own, then, inner, where, ids)
        return self.if_block(self.compare(node, own, where), then, other, where, ids)

    def copy_of(self, b: dict | None, where: str) -> dict | None:
        """A copy of an exported block with new (stable) ids on it and every block inside it."""
        if b is None:
            return None
        out = json.loads(json.dumps(b))

        def walk(x):
            if isinstance(x, dict):
                if "id" in x:
                    x["id"] = self.block_id(None, f"copy:{x['id']}", where)
                for k in x.get("children") or []:
                    walk(k)
        walk(out)
        return out

    def if_block(self, cond: dict, then: dict, other: dict | None, where: str = "", ids: dict | None = None) -> dict:
        ids = ids or {}
        kids = [{"id": ids.get("then_id") or self.block_id(None, "then_id", where), "step": "if-child",
                 "is-else-condition?": False, **cond, "children": [then]}]
        if other is not None:
            kids.append({"id": ids.get("else_id") or self.block_id(None, "else_id", where), "step": "if-child",
                         "is-else-condition?": True, "children": [other]})
        return {"id": ids.get("id") or self.block_id(None, "id", where), "step": "if", "children": kids}

    # ------------------------------------------------------------ nodes
    def node(self, n: dict, where: str) -> dict:
        """A tree node -> a Composer block with its id; a named node (a group) inside a "group" block."""
        b = self._node(n, where)
        if b.get("step") != "if":
            b = {"id": self.block_id(n, "id", where), **b}
        cm = n.get("composer")
        if n.get("name") or (isinstance(cm, dict) and cm.get("group_id")):
            g = {"id": self.block_id(n, "group_id", where), "step": "group"}
            if n.get("name"):
                g["name"] = str(n["name"])
            b = {**g, "children": [b]}
        return b

    def _node(self, n: dict, where: str) -> dict:
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
                return {"step": "wt-inverse-vol", "window-days": str(int(n.get("lookback") or 20)),
                        "children": [self.node(k, f"{where} > {i + 1}") for i, k in enumerate(kids)]}
            raise ComposerExportError(f"{where}: {m.replace('_', ' ')} weighting is not available in Composer (it has "
                                      "equal, specified and inverse-volatility weights).")
        if "if" in n:
            on = data.canonical(n.get("on", "SPY"))
            try:
                tree = ast.parse(str(n["if"]).strip(), mode="eval")
            except SyntaxError as e:
                raise ComposerExportError(f"{where}: the condition `{n['if']}` is not valid: {e.msg}")
            chain = self.elif_chain(n)
            if len(chain) > 1:
                return self.multi_if(chain, where)
            then = self.branch(n["then"], f"{where} > then")
            other = self.branch(n["else"], f"{where} > else")
            if then is None:
                raise ComposerExportError(f"{where}: the 'then' branch holds cash; Composer has no cash holding. Swap the "
                                          "condition so cash is the else branch, or hold BIL.")
            cm = n.get("composer") if isinstance(n.get("composer"), dict) else {}
            ids = {k: cm[k] for k in ("id", "then_id", "else_id") if cm.get(k)}
            return self.cond_tree(tree, on, then, other, where, ids)
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
                blk["sort-by-fn-params"] = {"window": win}
                blk["sort-by-window-days"] = str(win)
            return blk
        raise ComposerExportError(f"{where}: unknown node {sorted(n)}")

    @staticmethod
    def _one_compare(rule: str):
        """The single comparison of a rule (or None): what one if-child can hold."""
        try:
            b = ast.parse(str(rule).strip(), mode="eval").body
        except SyntaxError:
            return None
        return b if isinstance(b, ast.Compare) else None

    def elif_chain(self, n: dict) -> list[dict]:
        """An imported Composer if block with several conditions became nested if nodes (each else holds the next
        condition's node, which carries its if-child id but no block id of its own). Those nodes, in order, so the
        export writes them back as one if block; [n] when there is no such chain."""
        chain = [n]
        x = n
        while True:
            e = x.get("else")
            cm = e.get("composer") if isinstance(e, dict) else None
            if not (isinstance(e, dict) and "if" in e and isinstance(cm, dict) and cm.get("then_id") and not cm.get("id")
                    and not cm.get("group_id") and not e.get("name")):
                break
            chain.append(e)
            x = e
        if len(chain) > 1 and all(self._one_compare(c["if"]) is not None for c in chain):
            return chain
        return [n]

    def multi_if(self, chain: list[dict], where: str) -> dict:
        kids = []
        w = where
        for c in chain:
            cm = c.get("composer") or {}
            then = self.branch(c["then"], f"{w} > then")
            if then is None:
                raise ComposerExportError(f"{w}: the 'then' branch holds cash; Composer has no cash holding. Hold BIL "
                                          "instead.")
            cond = self.compare(self._one_compare(c["if"]), data.canonical(c.get("on", "SPY")), w)
            kids.append({"id": cm.get("then_id") or self.block_id(None, "then_id", w), "step": "if-child",
                         "is-else-condition?": False, **cond, "children": [then]})
            w = f"{w} > else"
        last = chain[-1]
        other = self.branch(last["else"], w)
        if other is not None:
            kids.append({"id": (last.get("composer") or {}).get("else_id") or self.block_id(None, "else_id", where),
                         "step": "if-child", "is-else-condition?": True, "children": [other]})
        return {"id": self.block_id(chain[0], "id", where), "step": "if", "children": kids}

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
    ex = _Exporter(spec.name or spec.description or "")
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
    cm = spec.tree.get("composer") if isinstance(spec.tree, dict) else None
    root: dict = {"id": (cm or {}).get("root_id") or ex.block_id(None, "root_id", "symphony"), "step": "root",
                  "name": (spec.name or spec.description or "Exported portfolio")[:80],
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
