"""Import a Composer (composer.trade) symphony into a Portfolio spec.

Composer exports and shares a symphony as a JSON tree of nodes, each with a "step":

  root               the symphony: name, description, "rebalance" (daily ... yearly, or "none" with a
                     "rebalance-corridor-width" threshold) and one child
  wt-cash-equal      equal weights of its children
  wt-cash-specified  fixed weights; every child carries "weight": {"num": 60, "den": 100}
  wt-inverse-vol     inverse-volatility weights over "window-days"
  group              a named box around its children (equal weights)
  if / if-child      a condition ("lhs-fn" of "lhs-val" compared with a fixed "rhs-val" or with
                     "rhs-fn" of another ticker) and an else branch ("is-else-condition?": true)
  filter             rank the children by "sort-by-fn" over "sort-by-window-days" and keep the
                     "select-fn" (top/bottom) "select-n"
  asset              one "ticker"

The result is a dict for `Portfolio.from_dict` (see portfolio.py). Composer's return-based functions
(cumulative return, moving average of return, standard deviation of return, max drawdown) are in
percent, so fixed thresholds on them are divided by 100. Anything the importer does not understand
raises ComposerImportError naming the field and where it is in the tree; nothing is dropped silently.

    python -m backtester import-composer symphony.json [--out spec.json] [--run]
"""
from __future__ import annotations

import json
import re
from pathlib import Path


class ComposerImportError(ValueError):
    pass


# Composer function -> (rule-language function or None for the series itself, series, percent-valued)
FUNCTIONS = {
    "relative-strength-index": ("rsi", "close", False),
    "cumulative-return": ("tret", "tr", True),
    "moving-average-price": ("sma", "close", False),
    "exponential-moving-average-price": ("ema", "close", False),
    "moving-average-return": ("ma_return", "tr", True),
    "standard-deviation-return": ("stdev_return", "tr", True),
    "standard-deviation-price": ("stdev", "close", False),
    "max-drawdown": ("max_drawdown", "tr", True),
    "current-price": (None, "close", False),
}
COMPARATORS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
REBALANCE = {"daily": "daily", "weekly": "weekly", "monthly": "monthly", "quarterly": "quarterly",
             "yearly": "yearly", "annually": "yearly", "none": "none", "threshold": "none"}

# The allow-list of descriptive fields Composer attaches to nodes (seen in real exports, API responses and shared
# links). None of them changes the allocation: labels, display settings, ids, timestamps, versions, the asset
# class and quote metadata. They are matched with "-" and "_" treated alike ("asset_class" = "asset-class"),
# because exports use both spellings. Any other field raises ComposerImportError: an unknown field might change
# what the symphony holds, so the importer stops rather than ignore it.
META = {"id", "name", "description", "collapsed?", "suppress-description?", "exchange", "price", "dollar-volume",
        "has-marketcap", "children-count", "asset-class", "asset-classes", "color", "version-id", "version",
        "created-at", "updated-at", "last-updated-at", "last-backtest-at", "symphony-id", "tags", "notes", "hashtag",
        "hashtags", "benchmarks", "share-with-everyone?", "copied-from", "sid", "hidden?", "comment",
        "window-days-label", "owner", "owner-id", "user-id", "is-public?", "public?", "source", "icon", "emoji",
        "display-name", "ticker-name", "company-name", "long-name", "short-name", "currency", "position"}


def _norm(k: str) -> str:
    return str(k).replace("_", "-")
# fields each step understands (besides META and "step"/"children")
FIELDS = {
    "root": {"rebalance", "rebalance-corridor-width", "rebalance-frequency", "corridor-width"},
    "wt-cash-equal": set(),
    "wt-cash-specified": set(),
    "wt-inverse-vol": {"window-days", "window-days-params"},
    "group": set(),
    "if": set(),
    "if-child": {"is-else-condition?", "lhs-fn", "lhs-val", "lhs-window-days", "lhs-fn-params", "comparator",
                 "rhs-val", "rhs-fixed-value?", "rhs-fn", "rhs-window-days", "rhs-fn-params"},
    "filter": {"sort-by-fn", "sort-by-window-days", "sort-by-fn-params", "select-fn", "select-n"},
    "asset": {"ticker"},
}
WEIGHT_KEYS = {"weight"}   # on children of wt-cash-specified


def ticker(v, where: str) -> str:
    """'SPY', 'EQUITIES::SPY//USD' or 'CRYPTO::BTC//USD' -> this tool's symbol."""
    if not isinstance(v, str) or not v.strip():
        raise ComposerImportError(f"{where}: expected a ticker, got {v!r}")
    s = v.strip().upper()
    crypto = s.startswith("CRYPTO::")
    s = s.split("::", 1)[-1].split("//", 1)[0]
    if crypto and not s.endswith("-USD"):
        s += "-USD"
    if not re.fullmatch(r"\^?[A-Z0-9.\-]{1,15}", s):
        raise ComposerImportError(f"{where}: {v!r} is not a ticker")
    return s.replace(".", "-") if re.fullmatch(r"[A-Z]+\.[A-Z]", s) else s


def _int(v, what: str, where: str) -> int:
    if isinstance(v, dict):
        v = v.get("window", v.get("value"))
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ComposerImportError(f"{where}: {what} must be a number, got {v!r}")
    if f < 1 or not f.is_integer():
        raise ComposerImportError(f"{where}: {what} must be a whole number of days (got {v!r})")
    return int(f)


def _num(v, what: str, where: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ComposerImportError(f"{where}: {what} must be a number, got {v!r}")


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(round(x, 10))


def _window(node: dict, prefix: str, where: str) -> int | None:
    for k in (f"{prefix}-window-days", f"{prefix}-fn-params"):
        v = node.get(k)
        if v not in (None, "", {}):
            return _int(v, f"{k}", where)
    return None


def indicator(fn: str, t: str | None, window: int | None, own: str | None, where: str) -> tuple[str, bool]:
    """A Composer function of ticker t as a rule on `own` (sym("T") when t is another ticker)."""
    if fn not in FUNCTIONS:
        raise ComposerImportError(f"{where}: unknown Composer function {fn!r} "
                                  f"(supported: {', '.join(sorted(FUNCTIONS))})")
    name, ser, pct = FUNCTIONS[fn]
    series = ser if (t is None or t == own) else f'sym("{t}").{ser}'
    if name is None:
        return series, pct
    if window is None:
        raise ComposerImportError(f"{where}: {fn} needs a window (days)")
    return f"{name}({series}, {window})", pct


def _check(node: dict, step: str, where: str, extra: set = frozenset()) -> None:
    ok = FIELDS.get(step, set()) | {"step", "children"} | set(extra)
    bad = sorted(k for k in node if k not in ok and _norm(k) not in META)
    if bad:
        raise ComposerImportError(f"{where} ({step}): unknown field(s) {', '.join(bad)} — the importer does not "
                                  "know what they do, so it stops rather than ignore them")


class _Importer:
    def __init__(self):
        self.notes: list[str] = []

    def note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)

    def kids(self, node: dict, where: str) -> list:
        k = node.get("children")
        if k is None:
            return []
        if not isinstance(k, list):
            raise ComposerImportError(f"{where}: 'children' must be a list")
        return k

    def group_of(self, kids: list, where: str, weights: str = "equal") -> dict:
        """Several children as one node: a single child stands alone, several are equal-weighted."""
        nodes = [self.node(k, f"{where} > {i + 1}") for i, k in enumerate(kids)]
        if not nodes:
            raise ComposerImportError(f"{where}: empty block (it has no children)")
        if len(nodes) == 1:
            return nodes[0]
        return {"weights": weights, "children": nodes}

    def node(self, n, where: str, parent_specified: bool = False) -> dict:
        if not isinstance(n, dict):
            raise ComposerImportError(f"{where}: expected a node object, got {type(n).__name__}")
        step = n.get("step")
        if step not in FIELDS or step == "root":
            raise ComposerImportError(f"{where}: unknown step {step!r} (supported: "
                                      f"{', '.join(s for s in FIELDS if s != 'root')})")
        label = n.get("name") or n.get("ticker") or step
        here = f"{where} [{label}]" if label != step else where
        _check(n, step, here, WEIGHT_KEYS if parent_specified else set())
        kids = self.kids(n, here)
        if step == "asset":
            if kids:
                raise ComposerImportError(f"{here}: an asset has no children")
            return {"asset": ticker(n.get("ticker"), here)}
        if step in ("wt-cash-equal", "group"):
            return self.group_of(kids, here)
        if step == "wt-inverse-vol":
            win = n.get("window-days", n.get("window-days-params"))
            lb = _int(win, "window-days", here) if win not in (None, "") else 20
            nodes = [self.node(k, f"{here} > {i + 1}") for i, k in enumerate(kids)]
            if not nodes:
                raise ComposerImportError(f"{here}: empty block (it has no children)")
            return {"weights": "inverse_vol", "lookback": lb, "children": nodes}
        if step == "wt-cash-specified":
            if not kids:
                raise ComposerImportError(f"{here}: empty block (it has no children)")
            nodes, w = [], []
            for i, k in enumerate(kids):
                kw = f"{here} > {i + 1}"
                if not isinstance(k, dict) or "weight" not in k:
                    raise ComposerImportError(f"{kw}: a child of wt-cash-specified needs a 'weight'")
                wt = k["weight"]
                if isinstance(wt, dict):
                    num, den = _num(wt.get("num"), "weight num", kw), _num(wt.get("den", 100), "weight den", kw)
                    if den <= 0:
                        raise ComposerImportError(f"{kw}: weight denominator must be positive")
                    w.append(num / den)
                else:
                    x = _num(wt, "weight", kw)
                    w.append(x / 100 if x > 1 else x)
                nodes.append(self.node(k, kw, parent_specified=True))
            tot = sum(w)
            if tot <= 0:
                raise ComposerImportError(f"{here}: the weights add up to {tot:g}")
            if abs(tot - 1) > 1e-6:
                self.note(f"Specified weights under {here} added up to {tot:.2%}; they were scaled to 100%.")
                w = [x / tot for x in w]
            return {"weights": "specified", "w": [round(x, 10) for x in w], "children": nodes}
        if step == "if":
            return self.if_node(kids, here)
        if step == "if-child":
            raise ComposerImportError(f"{here}: an if-child must sit directly inside an 'if' block")
        if step == "filter":
            return self.filter_node(n, kids, here)
        raise ComposerImportError(f"{here}: unsupported step {step!r}")  # pragma: no cover

    def condition(self, c: dict, where: str) -> tuple[str, str]:
        fn = c.get("lhs-fn")
        if not fn:
            raise ComposerImportError(f"{where}: the condition has no 'lhs-fn'")
        on = ticker(c.get("lhs-val"), f"{where} lhs-val")
        lhs, pct = indicator(fn, on, _window(c, "lhs", where), on, where)
        cmp = c.get("comparator")
        if cmp not in COMPARATORS:
            raise ComposerImportError(f"{where}: unknown comparator {cmp!r} (supported: gt, gte, lt, lte)")
        fixed = c.get("rhs-fixed-value?")
        if fixed is None:
            fixed = not c.get("rhs-fn")
        if fixed:
            v = _num(c.get("rhs-val"), "rhs-val", where)
            rhs = _fmt(v / 100 if pct else v)
        else:
            rfn = c.get("rhs-fn")
            if not rfn:
                raise ComposerImportError(f"{where}: 'rhs-fixed-value?' is false but there is no 'rhs-fn'")
            rt = ticker(c.get("rhs-val"), f"{where} rhs-val")
            rhs, _ = indicator(rfn, rt, _window(c, "rhs", where), on, where)
        return f"{lhs} {COMPARATORS[cmp]} {rhs}", on

    def if_node(self, kids: list, where: str) -> dict:
        conds, other = [], None
        for i, k in enumerate(kids):
            kw = f"{where} > {i + 1}"
            if not isinstance(k, dict) or k.get("step") != "if-child":
                raise ComposerImportError(f"{kw}: an 'if' block may only contain if-child blocks")
            _check(k, "if-child", kw)
            body = self.group_of(self.kids(k, kw), kw)
            if k.get("is-else-condition?"):
                if other is not None:
                    raise ComposerImportError(f"{kw}: more than one else branch")
                other = body
            else:
                rule, on = self.condition(k, kw)
                conds.append((rule, on, body))
        if not conds:
            raise ComposerImportError(f"{where}: an 'if' block needs a condition")
        if other is None:
            self.note(f"{where} had no else branch: cash is held when no condition is true.")
            other = {"cash": True}
        out = other
        for rule, on, body in reversed(conds):
            out = {"if": rule, "on": on, "then": body, "else": out}
        return out

    def filter_node(self, n: dict, kids: list, where: str) -> dict:
        fn = n.get("sort-by-fn")
        if not fn:
            raise ComposerImportError(f"{where}: the filter has no 'sort-by-fn'")
        by, _ = indicator(fn, None, _window(n, "sort-by", where), None, where)
        sel = n.get("select-fn", "top")
        if sel not in ("top", "bottom"):
            raise ComposerImportError(f"{where}: select-fn must be 'top' or 'bottom', got {sel!r}")
        k = _int(n.get("select-n", 1), "select-n", where)
        nodes = [self.node(c, f"{where} > {i + 1}") for i, c in enumerate(kids)]
        if not nodes:
            raise ComposerImportError(f"{where}: the filter has nothing to choose from")
        return {"filter": {"select": sel, "n": k, "by": by, "weights": "equal"}, "universe": "children",
                "children": nodes, "fallback": {"cash": True}}


def _unwrap(obj):
    """Accept the symphony itself or the wrappers Composer's API and exports put around it."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError as e:
            raise ComposerImportError(f"Not valid JSON: {e.msg} (line {e.lineno}, column {e.colno}). "
                                      "Paste the symphony's JSON (Composer: ⋯ → Export / copy JSON).")
    for _ in range(3):
        if isinstance(obj, dict) and "step" not in obj:
            for k in ("symphony", "score", "fields", "data", "raw_value"):
                if isinstance(obj.get(k), (dict, str)):
                    obj = obj[k]
                    if isinstance(obj, str):
                        obj = json.loads(obj)
                    break
            else:
                break
    if not isinstance(obj, dict) or "step" not in obj:
        raise ComposerImportError("This does not look like a Composer symphony: expected a JSON object with "
                                  '"step": "root" and "children".')
    return obj


def convert(obj) -> dict:
    """Composer symphony (dict or JSON text) -> a Portfolio spec dict (for Portfolio.from_dict)."""
    sym = _unwrap(obj)
    imp = _Importer()
    # Composer computes every indicator on dividend-adjusted (total-return) prices
    spec: dict = {"kind": "allocation", "price_basis": "adjusted"}
    if sym.get("step") == "root":
        _check(sym, "root", "symphony")
        rb = sym.get("rebalance", sym.get("rebalance-frequency", "daily"))
        if isinstance(rb, dict):
            rb = rb.get("frequency") or rb.get("value")
        if rb not in REBALANCE:
            raise ComposerImportError(f"symphony: unknown rebalance {rb!r} (daily, weekly, monthly, quarterly, "
                                      "yearly, or none with a corridor width)")
        spec["rebalance"] = REBALANCE[rb]
        cw = sym.get("rebalance-corridor-width", sym.get("corridor-width"))
        if cw not in (None, "") and spec["rebalance"] != "none":
            # Composer only uses the corridor with threshold rebalancing; exports of calendar-rebalanced
            # symphonies still carry the field (the last value set in the editor)
            imp.note(f"The symphony rebalances {spec['rebalance']}, so its rebalance-corridor-width ({cw}) is not used "
                     "(Composer applies it only to threshold rebalancing).")
        elif cw not in (None, ""):
            band = _num(cw, "rebalance-corridor-width", "symphony")
            spec["drift_band"] = band / 100 if band >= 1 else band
        elif spec["rebalance"] == "none" and rb == "threshold":
            raise ComposerImportError("symphony: threshold rebalancing needs 'rebalance-corridor-width'")
        spec["tree"] = imp.group_of(imp.kids(sym, "symphony"), "symphony")
        name = str(sym.get("name") or "").strip()
    else:
        spec["rebalance"] = "daily"
        spec["tree"] = imp.node(sym, "symphony")
        name = str(sym.get("name") or "").strip()
    spec["name"] = (name or "Composer symphony")[:80]
    spec["description"] = f"Composer symphony: {name}" if name else "Composer symphony"
    spec["notes"] = ["Imported from a Composer symphony: conditions and filters are evaluated on each "
                     "rebalance day's close, on total-return (dividend-adjusted) prices as Composer does, and "
                     "traded at that close."] + imp.notes
    return spec


def load(path: str | Path) -> dict:
    return convert(Path(path).read_text())


def to_portfolio(obj):
    from .portfolio import Portfolio
    return Portfolio.from_dict(convert(obj))
