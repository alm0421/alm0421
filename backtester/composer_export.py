"""Export a Portfolio spec as a Composer (composer.trade) symphony: the inverse of composer_import.

Composer's editor supports a subset of what the portfolio engine does. This module writes that subset:

  asset                                  {"step": "asset", "ticker": "SPY"}
  equal / specified / inverse_vol        wt-cash-equal / wt-cash-specified (weights as num/den) / wt-inverse-vol
  market_cap (of single assets)          wt-marketcap
  if ... then ... else                   an "if" block with if-children: one indicator of a ticker compared with a
                                         fixed number or with an indicator of a ticker (gt, gte, lt, lte, eq);
                                         "and" / "or" become a "condition" block (compound all / any; the same
                                         comparison on several tickers as binary-compound), "not" flips comparators
  filter (top/bottom N by a metric)      a "filter" block: sort-by-fn over sort-by-window-days, select-fn, select-n
  cash in a branch of an if              Composer's "empty" block
  a drift band                           threshold rebalancing: "rebalance": "none" with "rebalance-corridor-width"
                                         as a fraction (0.05 = 5%), as Composer stores it

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
CMP = {ast.Gt: "gt", ast.GtE: "gte", ast.Lt: "lt", ast.LtE: "lte", ast.Eq: "eq"}
FLIP = {"gt": "lt", "gte": "lte", "lt": "gt", "lte": "gte", "eq": "eq"}       # a op b  ==  b FLIP[op] a
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
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
                "macd", "macd_signal", "ppo", "ppo_signal", "bb_upper", "bb_lower"):
            # Checked (Sept 2026): no public export shows their params. Composer's schema as mirrored by
            # github.com/SolarWolf-Code/composer-trade-py (composer/models/common/symphony.py: "lhs-fn-params":
            # dict[str, Any]) and github.com/tanwithme/composer-trade-mcp (schemas/symphony_score_schema.py:
            # WindowParams, a window only) lists the function names but not their parameter keys.
            raise ComposerExportError(f"{where}: `{ast.unparse(node)}`: Composer has MACD, PPO and Bollinger bands, but its "
                                      "published symphony schema does not name the keys of their parameters (fast / slow "
                                      "/ signal window, standard deviations), so the export does not guess them. They "
                                      "import from Composer; build this condition in Composer's editor.")
        raise ComposerExportError(f"{where}: `{ast.unparse(node)}` has no Composer equivalent. Composer's conditions "
                                  f"and filters use: {SUPPORTED}.")

    # ------------------------------------------------------------ conditions
    def compare(self, node, own: str, where: str) -> dict:
        """One comparison -> the condition fields of an if-child."""
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or type(node.ops[0]) not in CMP:
            raise ComposerExportError(f"{where}: `{ast.unparse(node)}` is not a single >, >=, <, <= or == comparison "
                                      "(Composer has no != or chained comparisons).")
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

    def cond_fields(self, node, own: str, where: str) -> dict:
        """A rule -> the condition fields of one if-child. A single comparison (or its negation) is written in the
        single-comparison form (lhs-fn, lhs-val, comparator, rhs-...); 'and' / 'or' of several as Composer's "condition"
        block: an "all" / "any" compound, 'not' pushed inward by flipping comparators (De Morgan)."""
        while isinstance(node, ast.Expression):
            node = node.body
        neg = False
        while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
            node, neg = node.operand, not neg
        if isinstance(node, ast.BoolOp):
            cond = self.cond_json(node, own, where, neg)
            return {**self._legacy_mirror(cond), "condition": cond}
        c = self.compare(node, own, where)
        if neg:
            c["comparator"] = self._negate(c["comparator"], node, where)
        return c

    @staticmethod
    def _legacy_mirror(cond: dict) -> dict:
        """The single-comparison fields Composer's editor keeps beside a "condition" block: they repeat the block's
        last comparison ("lhs-fn", "lhs-window-days", "lhs-val", "comparator", "rhs-val" as text, "rhs-fixed-value?").
        Seen in a real export (tests/fixtures/composer_frontrunner_2026.json, from
        https://backtest-api.composer.trade/api/v1/public/symphonies/4aI4kVT5cEc0XJpTLei3/score, shown by
        https://composeratlas.com/converter): an any-of-12 RSI condition whose if-child also carries lhs-val "XLY",
        the last ticker. Composer reads the "condition" block; the mirror keeps the if-child in the shape it writes."""
        leaf = cond
        while leaf.get("condition-type") == "compound" and leaf.get("conditions"):
            leaf = leaf["conditions"][-1]
        lhs, rhs = leaf.get("lhs") or {}, leaf.get("rhs") or {}
        tk = lhs.get("ticker")
        if tk == "%" and leaf.get("tickers"):
            tk = leaf["tickers"][-1]
        out = {"lhs-fn": lhs.get("fn"), "lhs-val": tk, "comparator": leaf.get("comparator")}
        if isinstance(lhs.get("params"), dict) and "window" in lhs["params"]:
            out["lhs-window-days"] = str(lhs["params"]["window"])
        if "constant" in rhs:
            out["rhs-fixed-value?"] = True
            out["rhs-val"] = str(_fmt(float(rhs["constant"])))
        else:
            rt = rhs.get("ticker")
            out.update({"rhs-fixed-value?": False, "rhs-fn": rhs.get("fn"),
                        "rhs-val": leaf["tickers"][-1] if rt == "%" and leaf.get("tickers") else rt})
            if isinstance(rhs.get("params"), dict) and "window" in rhs["params"]:
                out["rhs-window-days"] = str(rhs["params"]["window"])
        return out

    @staticmethod
    def _negate(op: str, node, where: str) -> str:
        if op not in NEGATE:
            raise ComposerExportError(f"{where}: `not ({ast.unparse(node)})` has no Composer comparator (it has no "
                                      "'not equal').")
        return NEGATE[op]

    def cond_json(self, node, own: str, where: str, neg: bool = False) -> dict:
        """A rule -> a Composer condition block (binary / compound / binary-compound)."""
        while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
            node, neg = node.operand, not neg
        if isinstance(node, ast.BoolOp):
            is_and = isinstance(node.op, ast.And) != neg          # De Morgan
            parts = []
            for v in node.values:
                j = self.cond_json(v, own, where, neg)
                # all(a, all(b, c)) = all(a, b, c)
                if j["condition-type"] == "compound" and j["operator"] == ("all" if is_and else "any"):
                    parts.extend(j["conditions"])
                else:
                    parts.append(j)
            return self._merge_tickers({"condition-type": "compound", "operator": "all" if is_and else "any",
                                        "conditions": parts})
        c = self.compare(node, own, where)
        if neg:
            c["comparator"] = self._negate(c["comparator"], node, where)
        lhs = {"fn": c["lhs-fn"], "ticker": c["lhs-val"]}
        if "lhs-fn-params" in c:
            lhs["params"] = c["lhs-fn-params"]
        if c["rhs-fixed-value?"]:
            rhs = {"constant": _fmt(float(c["rhs-val"]))}
        else:
            rhs = {"fn": c["rhs-fn"], "ticker": c["rhs-val"]}
            if "rhs-fn-params" in c:
                rhs["params"] = c["rhs-fn-params"]
        return {"condition-type": "binary", "lhs": lhs, "comparator": c["comparator"], "rhs": rhs}

    @staticmethod
    def _merge_tickers(comp: dict) -> dict:
        """any/all of the same comparison on several tickers (RSI(10) > 79 of SPY, QQQ and SMH) -> Composer's
        "binary-compound" form (the comparison once, "%" for each of its tickers), as its editor writes it."""
        cs = comp["conditions"]
        if len(cs) < 2 or not all(c["condition-type"] == "binary" and "constant" in c["rhs"] for c in cs):
            return comp
        shape = lambda c: (c["lhs"]["fn"], json.dumps(c["lhs"].get("params"), sort_keys=True), c["comparator"],  # noqa: E731
                           c["rhs"]["constant"])
        tickers = [c["lhs"]["ticker"] for c in cs]
        if len({shape(c) for c in cs}) != 1 or len(set(tickers)) != len(tickers):
            return comp
        lhs = {**cs[0]["lhs"], "ticker": "%"}
        return {"condition-type": "binary-compound", "operator": comp["operator"], "tickers": tickers, "lhs": lhs,
                "comparator": cs[0]["comparator"], "rhs": cs[0]["rhs"]}

    def if_block(self, cond: dict, then: dict, other: dict, where: str = "", ids: dict | None = None) -> dict:
        ids = ids or {}
        kids = [{"id": ids.get("then_id") or self.block_id(None, "then_id", where), "step": "if-child",
                 "is-else-condition?": False, **cond, "children": [then]},
                {"id": ids.get("else_id") or self.block_id(None, "else_id", where), "step": "if-child",
                 "is-else-condition?": True, "children": [other]}]
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
            raise ComposerExportError(f"{where}: Composer has no cash holding here. Hold a T-bill fund such as BIL or "
                                      "SHV instead (a branch of an if that holds cash is fine: it becomes Composer's "
                                      "empty block).")
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
            if m == "market_cap":
                if not kids or not all(isinstance(k, dict) and "asset" in k for k in kids):
                    raise ComposerExportError(f"{where}: Composer's market-cap weighting applies to single assets only.")
                return {"step": "wt-marketcap", "children": [self.node(k, f"{where} > {i + 1}") for i, k in enumerate(kids)]}
            raise ComposerExportError(f"{where}: {m.replace('_', ' ')} weighting is not available in Composer (it has "
                                      "equal, specified, inverse-volatility and market-cap weights).")
        if "if" in n:
            chain = self.elif_chain(n)
            if len(chain) > 1:
                return self.multi_if(chain, where)
            cond = self.condition_of(n, where)
            then = self.branch(n["then"], f"{where} > then")
            other = self.branch(n["else"], f"{where} > else")
            cm = n.get("composer") if isinstance(n.get("composer"), dict) else {}
            ids = {k: cm[k] for k in ("id", "then_id", "else_id") if cm.get(k)}
            return self.if_block(cond, then, other, where, ids)
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

    def condition_of(self, n: dict, where: str) -> dict:
        """An if node's rule -> the condition fields of its if-child."""
        try:
            tree = ast.parse(str(n["if"]).strip(), mode="eval")
        except SyntaxError as e:
            raise ComposerExportError(f"{where}: the condition `{n['if']}` is not valid: {e.msg}")
        return self.cond_fields(tree, data.canonical(n.get("on", "SPY")), where)

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
        return chain

    def multi_if(self, chain: list[dict], where: str) -> dict:
        kids = []
        w = where
        for c in chain:
            cm = c.get("composer") or {}
            cond = self.condition_of(c, w)
            then = self.branch(c["then"], f"{w} > then")
            kids.append({"id": cm.get("then_id") or self.block_id(None, "then_id", w), "step": "if-child",
                         "is-else-condition?": False, **cond, "children": [then]})
            w = f"{w} > else"
        last = chain[-1]
        other = self.branch(last["else"], w)
        kids.append({"id": (last.get("composer") or {}).get("else_id") or self.block_id(None, "else_id", where),
                     "step": "if-child", "is-else-condition?": True, "children": [other]})
        return {"id": self.block_id(chain[0], "id", where), "step": "if", "children": kids}

    def branch(self, n: dict, where: str) -> dict:
        """A branch of an if: cash is Composer's empty block."""
        if isinstance(n, dict) and n.get("cash") and not n.get("name"):
            return {"id": self.block_id(n, "id", where), "step": "empty"}
        return self.node(n, where)


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
        # Composer's threshold rebalancing evaluates the rules at every close and trades when they change the target or
        # a holding leaves its corridor: this spec's "daily, trading only when the target changes or a holding drifts"
        # (and "never rebalance" with a band) exactly
        if spec.rebalance not in ("none", "daily"):
            raise ComposerExportError(f"Composer uses a rebalance corridor instead of a calendar schedule, not both: "
                                      f"'rebalance {spec.rebalance} or when a weight drifts' has no equivalent. Drop the "
                                      "schedule (the rules are then checked every day, as Composer's threshold "
                                      "rebalancing does) or the drift band.")
        if not 0 < spec.drift_band < 1:
            raise ComposerExportError(f"A {spec.drift_band:.0%} drift band is not a corridor Composer can hold.")
        root["rebalance"] = "none"
        # a fraction of the portfolio, as Composer stores it: 0.05 is a 5% corridor
        root["rebalance-corridor-width"] = _fmt(round(spec.drift_band, 10))
    elif spec.rebalance == "none":
        raise ComposerExportError("Composer has no buy-and-hold (never rebalance) setting without a corridor width.")
    if isinstance(spec.tree, dict) and spec.tree.get("cash"):
        raise ComposerExportError("The portfolio holds only cash.")
    root["children"] = [ex.node(spec.tree, "portfolio")]
    if spec.price_basis != "adjusted":
        ex.note("Composer computes indicators on total-return (dividend-adjusted) prices; this spec used prices as "
                "quoted, so RSI and moving averages of dividend payers can differ slightly.")
    if spec.fill != "close":
        ex.note("Composer trades near the close of the rebalance day; this spec traded at the next open.")
    day = getattr(spec, "rebalance_day", None)
    if spec.rebalance in ("weekly", "monthly", "quarterly", "yearly") and not spec.drift_band and day != "start":
        # Composer has no rebalance-day setting: its weekly / monthly / quarterly / yearly symphonies trade on the first
        # trading day of each period (help.composer.trade/article/54-create-tutorial)
        per = {"weekly": "week", "monthly": "month", "quarterly": "quarter", "yearly": "year"}[spec.rebalance]
        when = (f"every {day.capitalize()}" if day and day not in ("end", "start") else f"on the last trading day of each {per}")
        ex.note(f"Rebalance timing: Composer runs a {spec.rebalance} symphony on the first trading day of each {per}; this "
                f"spec rebalanced {when}, so Composer's backtest trades on different days (set rebalance_day \"start\" "
                "to match it here).")
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
