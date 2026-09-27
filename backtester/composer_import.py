"""Import a Composer (composer.trade) symphony into a Portfolio spec.

Composer exports and shares a symphony as a JSON tree of nodes, each with a "step":

  root               the symphony: name, description, "rebalance" (daily ... yearly, or "none" with a
                     "rebalance-corridor-width" threshold, a fraction: 0.05 = 5%) and one child
  wt-cash-equal      equal weights of its children
  wt-cash-specified  fixed weights; every child carries "weight": {"num": 60, "den": 100}
  wt-inverse-vol     inverse-volatility weights over "window-days"
  wt-marketcap       market-cap weights of its assets
  group              a named box around its children (equal weights)
  if / if-child      a condition and an else branch ("is-else-condition?": true). The condition is either
                     a "condition" block - "binary" (lhs {fn, ticker, params} compared with rhs {constant}
                     or another {fn, ticker, params}), "compound" (any/all of several conditions) or
                     "binary-compound" (one comparison applied to each of "tickers", any/all) - or, in
                     older exports, "lhs-fn" of "lhs-val" compared with a fixed "rhs-val" or with "rhs-fn"
                     of another ticker. Comparators gt, gte, lt, lte, eq.
  filter             rank the children by "sort-by-fn" over "sort-by-window-days" and keep the
                     "select-fn" (top/bottom) "select-n"
  asset              one "ticker"
  empty              nothing: cash

The result is a dict for `Portfolio.from_dict` (see portfolio.py). Composer's return-based functions
(cumulative return, moving average of return, standard deviation of return, max drawdown) are in
percent, so fixed thresholds on them are divided by 100. MACD / MACD signal, PPO / PPO signal (in percent)
and the lower / upper Bollinger band map to macd, macd_signal, ppo, ppo_signal, bb_lower and bb_upper. Anything the importer does not understand
raises ComposerImportError naming the field and where it is in the tree: an unknown step, function, comparator or
condition type, and any unknown field that could change the allocation (its name speaks of weights, windows,
conditions, tickers, rebalancing ... or it holds blocks). An unknown field that looks like metadata (a new label,
display setting or timestamp) is ignored with a warning in the notes; nothing is dropped silently.

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
    # several parameters (see MULTI): the series is their last argument
    "moving-average-convergence-divergence": ("macd", "close", False),
    "moving-average-convergence-divergence-signal": ("macd_signal", "close", False),
    "percentage-price-oscillator": ("ppo", "close", False),          # in percent, as Composer shows it
    "percentage-price-oscillator-signal": ("ppo_signal", "close", False),
    "lower-bollinger": ("bb_lower", "close", False),
    "upper-bollinger": ("bb_upper", "close", False),
}
# Composer's API schema gives these functions (fast, slow), (fast, slow, signal) and (window, standard deviations)
# parameters but does not name the keys of their params object: the spellings below are accepted, any other key is
# refused. Each entry: (parameter, accepted keys, default used - with a note - when the params leave it out).
_FAST = ("fast window", ("fast-window", "fast-window-days", "short-window", "fast-period", "fast", "short"), 12)
_SLOW = ("slow window", ("slow-window", "slow-window-days", "long-window", "slow-period", "slow", "long"), 26)
_SIG = ("signal window", ("signal-window", "signal-window-days", "signal-period", "signal"), 9)
_BBW = ("window", ("window", "window-days", "period"), 20)
_BBK = ("standard deviations", ("std-dev", "stddev", "num-std-dev", "num-std", "standard-deviations", "std", "k",
                                "multiplier"), 2)
MULTI = {"macd": (_FAST, _SLOW), "ppo": (_FAST, _SLOW), "macd_signal": (_FAST, _SLOW, _SIG),
         "ppo_signal": (_FAST, _SLOW, _SIG), "bb_lower": (_BBW, _BBK), "bb_upper": (_BBW, _BBK)}
COMPARATORS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "=="}
REBALANCE = {"daily": "daily", "weekly": "weekly", "monthly": "monthly", "quarterly": "quarterly",
             "yearly": "yearly", "annually": "yearly", "none": "none", "threshold": "none"}

# The allow-list of descriptive fields Composer attaches to nodes (seen in real exports, API responses and shared
# links). None of them changes the allocation: labels, display settings, ids, timestamps, versions, the asset
# class and quote metadata. They are matched with "-" and "_" treated alike ("asset_class" = "asset-class"),
# because exports use both spellings. Any other field is checked by unknown_fields: refused when it could change
# what the symphony holds, ignored with a warning when it looks like more metadata.
META = {"id", "name", "description", "collapsed?", "collapsed-specified-weight?", "suppress-incomplete-warnings?", "suppress-incomplete-warnings", "suppress-description?", "exchange", "price", "dollar-volume",
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
                 "rhs-val", "rhs-fixed-value?", "rhs-fn", "rhs-window-days", "rhs-fn-params", "condition"},
    "filter": {"sort-by-fn", "sort-by-window-days", "sort-by-fn-params", "select-fn", "select-n"},
    "asset": {"ticker"},
    "wt-marketcap": set(),
    "empty": set(),
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
        extra = sorted(k for k in v if k not in ("window", "value"))
        if extra or v.get("window", v.get("value")) is None:
            raise ComposerImportError(f"{where}: {what} should be {{\"window\": <days>}} for this function, got {v!r}")
        v = v.get("window", v.get("value"))
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ComposerImportError(f"{where}: {what} must be a number, got {v!r}")
    if f < 1 or not f.is_integer():
        raise ComposerImportError(f"{where}: {what} must be a positive whole number of days (got {v!r})")
    return int(f)


def _num(v, what: str, where: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ComposerImportError(f"{where}: {what} must be a number, got {v!r}")


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(round(x, 10))


def _params(node: dict, prefix: str) -> tuple[object, str]:
    """The raw parameters of a function in the single-condition form: "<prefix>-fn-params" (current exports: a dict,
    {"window": 10}) or the older "<prefix>-window-days" ("10"), and the field it came from."""
    for k in (f"{prefix}-fn-params", f"{prefix}-window-days"):
        v = node.get(k)
        if v not in (None, "", {}):
            return v, k
    return None, f"{prefix}-fn-params"


def _multi(name: str, fn: str, raw, what: str, where: str, notes: list | None) -> list[str]:
    """The (fast, slow[, signal]) or (window, standard deviations) parameters of MACD / PPO / Bollinger bands."""
    spec = MULTI[name]
    if raw not in (None, "", {}) and not isinstance(raw, dict):
        raise ComposerImportError(f"{where}: {fn} takes several parameters ({', '.join(p for p, _, _ in spec)}); "
                                  f"{what} should be an object such as {{\"{spec[0][1][0]}\": {spec[0][2]}, "
                                  f"\"{spec[1][1][0]}\": {spec[1][2]}}}, got {raw!r}")
    raw = dict(raw or {})
    known = {k for _, keys, _ in spec for k in keys}
    extra = sorted(k for k in raw if _norm(k) not in known)
    if extra:
        raise ComposerImportError(f"{where}: {what} of {fn}: unknown parameter(s) {', '.join(map(str, extra))} (expected "
                                  + "; ".join(f"{p}: {' / '.join(keys[:3])}" for p, keys, _ in spec) + ")")
    out, defaulted = [], []
    for label, keys, default in spec:
        v = next((raw[k] for k in raw if _norm(k) in keys), None)
        if v in (None, ""):
            v = default
            defaulted.append((label, default))
        if label == "standard deviations":
            x = _num(v, f"{fn} {label}", where)
            if not x > 0:
                raise ComposerImportError(f"{where}: {fn} {label} must be positive (got {v!r})")
            out.append(_fmt(x))
        else:
            out.append(str(_int(v, f"{fn} {label}", where)))
    if defaulted and notes is not None:
        msg = (f"{where}: {fn} does not give its {', '.join(l for l, _ in defaulted)}; the standard "
               f"{', '.join(f'{l} {d}' for l, d in defaulted)} was used.")
        if msg not in notes:
            notes.append(msg)
    return out


def indicator(fn: str, t: str | None, raw, own: str | None, where: str, what: str = "params",
              notes: list | None = None) -> tuple[str, bool]:
    """A Composer function of ticker t as a rule on `own` (sym("T") when t is another ticker). `raw` is its
    parameters as Composer writes them: a window (10, "10", {"window": 10}) or, for MACD / PPO / Bollinger bands, an
    object of several."""
    if fn not in FUNCTIONS:
        raise ComposerImportError(f"{where}: unknown Composer function {fn!r} "
                                  f"(supported: {', '.join(sorted(FUNCTIONS))})")
    name, ser, pct = FUNCTIONS[fn]
    series = ser if (t is None or t == own) else f'sym("{t}").{ser}'
    if name is None:
        return series, pct
    if name in MULTI:
        args = _multi(name, fn, raw, what, where, notes)
        return f"{name}({', '.join(args)}{'' if series == 'close' else ', ' + series})", pct
    if raw in (None, "", {}):
        raise ComposerImportError(f"{where}: {fn} needs a window (days)")
    return f"{name}({series}, {_int(raw, what, where)})", pct


# An unknown field is either metadata Composer added since this importer was written (a label, a display setting, a
# timestamp: ignored with a warning, so a new export still opens) or something that could change what the symphony
# holds (refused, naming where it is). A field is taken as structural when its name has a word that describes
# holdings, weights, conditions, rankings or rebalancing, or when its value holds blocks of its own (a "step").
STRUCTURAL_WORDS = re.compile(
    r"(?:^|[-_?])(fn|params?|windows?|days|n|weights?|comparator|select|sort|conditions?|lhs|rhs|child|children|"
    r"rebalance|rebalancing|corridor|threshold|operator|tickers?|val|values?|step|if|else|filter|period|leverage|"
    r"alloc|allocation|short|hedge|cash|assets?|lookback|rank|top|bottom|max|min|limit|cap|scale|percent|pct|"
    r"ratio|fraction|num|den|order|trade|signal|rules?|when|stop|margin|target|override|amount|size|holdings?)"
    r"(?=[-_?]|$)", re.I)


def _has_blocks(v) -> bool:
    if isinstance(v, dict):
        return "step" in v or any(_has_blocks(x) for x in v.values())
    if isinstance(v, list):
        return any(_has_blocks(x) for x in v)
    return False


def structural(key: str, value) -> bool:
    """Could this unknown field change the allocation? (see STRUCTURAL_WORDS)"""
    return bool(STRUCTURAL_WORDS.search(str(key))) or _has_blocks(value)


def unknown_fields(node: dict, ok: set, where: str, what: str, notes: list | None) -> None:
    """Refuse unknown structural fields; note (and ignore) unknown metadata."""
    bad = sorted((k for k in node if k not in ok and _norm(k) not in META), key=str)
    if not bad:
        return
    hard = [k for k in bad if structural(k, node[k]) or notes is None]
    if hard:
        raise ComposerImportError(f"{where} ({what}): unknown field(s) {', '.join(map(str, hard))} — they may change what "
                                  "the symphony holds and the importer does not know what they do, so it stops rather "
                                  "than ignore them")
    msg = (f"Warning: {where} ({what}): unknown field(s) {', '.join(map(str, bad))} ignored — they look like metadata "
           "(labels, display settings, timestamps) that do not change the allocation. Check the result if Composer "
           "gave them a meaning.")
    if msg not in notes:
        notes.append(msg)


def _check(node: dict, step: str, where: str, extra: set = frozenset(), notes: list | None = None) -> None:
    ok = FIELDS.get(step, set()) | {"step", "children"} | set(extra)
    unknown_fields(node, ok, where, step, notes)


def _with_id(node: dict, key: str, block_id) -> dict:
    """Keep the id of the Composer block a node came from (node["composer"][key]), so an export keeps it."""
    if isinstance(block_id, (str, int)) and not isinstance(block_id, bool) and str(block_id).strip():
        node.setdefault("composer", {})[key] = str(block_id).strip()[:100]
    return node


class _Importer:
    def __init__(self):
        self.notes: list[str] = []
        self.ignored_weights = False

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

    def group_of(self, kids: list, where: str, weights: str = "equal", block_id=None) -> dict:
        """Several children as one node: a single child stands alone, several are equal-weighted."""
        nodes = [self.node(k, f"{where} > {i + 1}") for i, k in enumerate(kids)]
        if not nodes:
            raise ComposerImportError(f"{where}: empty block (it has no children)")
        if len(nodes) == 1:
            return nodes[0]
        return _with_id({"weights": weights, "children": nodes}, "id", block_id)

    def node(self, n, where: str, parent_specified: bool = False) -> dict:
        if not isinstance(n, dict):
            raise ComposerImportError(f"{where}: expected a node object, got {type(n).__name__}")
        step = n.get("step")
        if step not in FIELDS or step == "root":
            raise ComposerImportError(f"{where}: unknown step {step!r} (supported: "
                                      f"{', '.join(s for s in FIELDS if s != 'root')})")
        label = n.get("name") or n.get("ticker") or step
        here = f"{where} [{label}]" if label != step else where
        _check(n, step, here, WEIGHT_KEYS, self.notes)
        if "weight" in n and not parent_specified:
            self.ignored_weights = True
        kids = self.kids(n, here)
        bid = n.get("id")
        if step == "asset":
            if kids:
                raise ComposerImportError(f"{here}: an asset has no children")
            return _with_id({"asset": ticker(n.get("ticker"), here)}, "id", bid)
        if step == "empty":
            if kids:
                raise ComposerImportError(f"{here}: an empty block has no children")
            return {"cash": True}
        if step == "wt-cash-equal":
            return self.group_of(kids, here, block_id=bid)
        if step == "wt-marketcap":
            nodes = [self.node(k, f"{here} > {i + 1}") for i, k in enumerate(kids)]
            if not nodes:
                raise ComposerImportError(f"{here}: empty block (it has no children)")
            if not all(isinstance(x, dict) and "asset" in x for x in nodes):
                raise ComposerImportError(f"{here}: market-cap weighting applies to assets only")
            from .portfolio import MCAP_FUND_HINT, _funds_named
            funds = _funds_named([x["asset"] for x in nodes])
            if funds:
                raise ComposerImportError(
                    f"{here}: market-cap weighting of {', '.join(funds[:6])}: "
                    f"{'it is a fund' if len(funds) == 1 else 'they are funds'} (ETF, mutual fund or index), and funds have "
                    f"no market cap (no shares outstanding of a company), so the weights would quietly be equal. "
                    + MCAP_FUND_HINT)
            return _with_id({"weights": "market_cap", "children": nodes}, "id", bid)
        if step == "group":
            out = self.group_of(kids, here)
            name = str(n.get("name") or "").strip()
            if not name and bid in (None, ""):
                return out
            if name and out.get("name"):      # a named group around a named group: keep both labels
                out = {"weights": "equal", "children": [out]}
            if name:
                out["name"] = name[:120]
            return _with_id(out, "group_id", bid)
        if step == "wt-inverse-vol":
            win = n.get("window-days", n.get("window-days-params"))
            lb = _int(win, "window-days", here) if win not in (None, "") else 20
            nodes = [self.node(k, f"{here} > {i + 1}") for i, k in enumerate(kids)]
            if not nodes:
                raise ComposerImportError(f"{here}: empty block (it has no children)")
            return _with_id({"weights": "inverse_vol", "lookback": lb, "children": nodes}, "id", bid)
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
            return _with_id({"weights": "specified", "w": [round(x, 10) for x in w], "children": nodes}, "id", bid)
        if step == "if":
            return _with_id(self.if_node(kids, here), "id", bid)
        if step == "if-child":
            raise ComposerImportError(f"{here}: an if-child must sit directly inside an 'if' block")
        if step == "filter":
            return _with_id(self.filter_node(n, kids, here), "id", bid)
        raise ComposerImportError(f"{here}: unsupported step {step!r}")  # pragma: no cover

    # ------------------------------------------------------------ conditions
    def condition(self, c: dict, where: str) -> tuple[str, str]:
        """An if-child's condition -> (rule, the ticker it is evaluated on). Current Composer exports write it as a
        "condition" block (binary, or any/all of several: "compound" and "binary-compound"); older ones as lhs-/rhs-
        fields on the if-child itself."""
        cond = c.get("condition")
        if cond not in (None, {}):
            if c.get("lhs-fn"):
                self.note(f"{where}: the if-child's 'condition' block is used; the lhs-/rhs- fields beside it are what "
                          "Composer's editor keeps from the single-comparison form.")
            try:
                on = self._first_ticker(cond, where)
            except ComposerImportError:
                self.cond_block(cond, "SPY", where)     # a malformed block: say what is wrong with it
                raise
            return self.cond_block(cond, on, where), on
        fn = c.get("lhs-fn")
        if not fn:
            raise ComposerImportError(f"{where}: the condition has no 'lhs-fn'")
        lraw, lwhat = _params(c, "lhs")
        lhs = {"fn": fn, "ticker": c.get("lhs-val"), "raw": lraw, "what": lwhat}
        fixed = c.get("rhs-fixed-value?")
        if fixed is None:
            fixed = not c.get("rhs-fn")
        if fixed:
            rhs = {"constant": c.get("rhs-val"), "what": "rhs-val"}
        else:
            rfn = c.get("rhs-fn")
            if not rfn:
                raise ComposerImportError(f"{where}: 'rhs-fixed-value?' is false but there is no 'rhs-fn'")
            rraw, rwhat = _params(c, "rhs")
            rhs = {"fn": rfn, "ticker": c.get("rhs-val"), "raw": rraw, "what": rwhat}
        on = ticker(c.get("lhs-val"), f"{where} lhs-val")
        return self.compare(lhs, c.get("comparator"), rhs, on, where), on

    def _first_ticker(self, cond, where: str) -> str:
        """The ticker a condition block is evaluated on: the first one it names."""
        if isinstance(cond, dict):
            ct = cond.get("condition-type")
            if ct == "binary-compound":
                ts = cond.get("tickers")
                if isinstance(ts, list) and ts:
                    return ticker(ts[0], f"{where} tickers")
            if ct == "compound":
                for x in cond.get("conditions") or []:
                    try:
                        return self._first_ticker(x, where)
                    except ComposerImportError:
                        continue
            for side in ("lhs", "rhs"):
                s = cond.get(side)
                if isinstance(s, dict) and s.get("ticker") not in (None, "%"):
                    return ticker(s["ticker"], f"{where} {side} ticker")
        raise ComposerImportError(f"{where}: the condition names no ticker")

    def cond_block(self, cond, on: str, where: str) -> str:
        """A Composer condition block -> a rule. compound: any/all of its conditions; binary-compound: one comparison
        applied to each of several tickers ("%" stands for each), any/all of them; binary: one comparison."""
        if not isinstance(cond, dict):
            raise ComposerImportError(f"{where}: a condition must be an object, got {type(cond).__name__}")
        ct = cond.get("condition-type")
        allowed = {"binary": {"condition-type", "lhs", "comparator", "rhs"},
                   "compound": {"condition-type", "operator", "conditions"},
                   "binary-compound": {"condition-type", "operator", "tickers", "lhs", "comparator", "rhs"}}
        if ct not in allowed:
            raise ComposerImportError(f"{where}: unknown condition-type {ct!r} (supported: binary, compound, "
                                      "binary-compound)")
        unknown_fields(cond, allowed[ct], where, f"{ct} condition", self.notes)
        if ct == "binary":
            return self.compare(self._side(cond.get("lhs"), f"{where} lhs"), cond.get("comparator"),
                                self._side(cond.get("rhs"), f"{where} rhs"), on, where)
        op = cond.get("operator")
        if op not in ("any", "all"):
            raise ComposerImportError(f"{where}: a {ct} condition's operator must be 'any' or 'all', got {op!r}")
        if ct == "compound":
            parts = cond.get("conditions")
            if not isinstance(parts, list) or not parts:
                raise ComposerImportError(f"{where}: a compound condition needs a list of conditions")
            rules = [self.cond_block(x, on, f"{where} > condition {i + 1}") for i, x in enumerate(parts)]
        else:
            ts = cond.get("tickers")
            if not isinstance(ts, list) or not ts:
                raise ComposerImportError(f"{where}: a binary-compound condition needs a list of tickers")
            lhs, rhs = self._side(cond.get("lhs"), f"{where} lhs"), self._side(cond.get("rhs"), f"{where} rhs")
            if lhs.get("ticker") != "%" and rhs.get("ticker") != "%":
                raise ComposerImportError(f"{where}: a binary-compound condition applies its comparison to each of its "
                                          "tickers, written \"%\" in the lhs")
            rules = []
            for i, t in enumerate(ts):
                tk = ticker(t, f"{where} tickers[{i}]")
                sub = lambda s: {**s, "ticker": tk} if s.get("ticker") == "%" else s  # noqa: E731
                rules.append(self.compare(sub(lhs), cond.get("comparator"), sub(rhs), on, f"{where} ({tk})"))
        if len(rules) == 1:
            return rules[0]
        return f" {'or' if op == 'any' else 'and'} ".join(f"({r})" for r in rules)

    @staticmethod
    def _side(s, where: str) -> dict:
        """One side of a binary condition: {"fn", "ticker", "params"} or {"constant"}."""
        if not isinstance(s, dict):
            raise ComposerImportError(f"{where}: expected {{\"fn\", \"ticker\", \"params\"}} or {{\"constant\"}}, got {s!r}")
        if "constant" in s:
            if set(s) != {"constant"}:
                raise ComposerImportError(f"{where}: unknown field(s) beside 'constant': "
                                          f"{', '.join(sorted(k for k in s if k != 'constant'))}")
            return {"constant": s["constant"], "what": "constant"}
        bad = sorted(k for k in s if k not in ("fn", "ticker", "params"))
        if bad or not s.get("fn"):
            raise ComposerImportError(f"{where}: expected {{\"fn\", \"ticker\", \"params\"}}, got {s!r}")
        return {"fn": s["fn"], "ticker": s.get("ticker"), "raw": s.get("params"), "what": "params"}

    def compare(self, lhs: dict, cmp, rhs: dict, on: str, where: str) -> str:
        """One comparison of an indicator with a fixed number or another indicator, as a rule evaluated on `on`."""
        if cmp not in COMPARATORS:
            raise ComposerImportError(f"{where}: unknown comparator {cmp!r} (supported: {', '.join(COMPARATORS)})")
        op = COMPARATORS[cmp]
        if "constant" in lhs and "constant" in rhs:
            raise ComposerImportError(f"{where}: the condition compares two fixed numbers")
        if "constant" in lhs:    # 70 < RSI  ==  RSI > 70
            lhs, rhs = rhs, lhs
            op = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "==": "=="}[op]
        fn = lhs["fn"]
        lt = ticker(lhs.get("ticker"), f"{where} ticker")
        lx, pct = indicator(fn, lt, lhs.get("raw"), on, where, lhs["what"], self.notes)
        if "constant" in rhs:
            v = _num(rhs["constant"], rhs["what"], where)
            rx = _fmt(v / 100 if pct else v)
            rfn = None
        else:
            rfn = rhs["fn"]
            rt = ticker(rhs.get("ticker"), f"{where} rhs ticker")
            rx, _ = indicator(rfn, rt, rhs.get("raw"), on, where, rhs["what"], self.notes)
        rule = f"{lx} {op} {rx}"
        if op == "==":
            self.note(f"{where}: Composer's 'eq' comparator is an exact equality (`{rule}`), which a computed indicator "
                      "rarely meets exactly.")
        # a price level (current price, its moving average or standard deviation, a Bollinger band) against a fixed
        # number means the quoted price, and price levels of two different tickers are only comparable as quoted:
        # each ticker's total-return level starts at its own first close and grows with its own dividends
        from .portfolio import quote_levels_why
        q, why = quote_levels_why(rule, on)
        if q != rule:
            if "fixed" in why:
                self.note(f"{where}: {fn} of {lt} is compared with the fixed value {rx}, so it is read on quoted prices "
                          f"(`{q}`); the other indicators use total-return prices.")
            if "cross" in why:
                self.note(f"{where}: {fn} of {lt} is compared with {rfn} of another ticker, so both price "
                          f"levels are read on quoted prices (`{q}`); total-return levels of two tickers are not "
                          "comparable.")
            rule = q
        return rule

    def if_node(self, kids: list, where: str) -> dict:
        conds, other = [], None
        for i, k in enumerate(kids):
            kw = f"{where} > {i + 1}"
            if not isinstance(k, dict) or k.get("step") != "if-child":
                raise ComposerImportError(f"{kw}: an 'if' block may only contain if-child blocks")
            _check(k, "if-child", kw, notes=self.notes)
            body = self.group_of(self.kids(k, kw), kw)
            if k.get("is-else-condition?"):
                if other is not None:
                    raise ComposerImportError(f"{kw}: more than one else branch")
                other, else_id = body, k.get("id")
            else:
                rule, on = self.condition(k, kw)
                conds.append((rule, on, body, k.get("id")))
        if not conds:
            raise ComposerImportError(f"{where}: an 'if' block needs a condition")
        if other is None:
            self.note(f"{where} had no else branch: cash is held when no condition is true.")
            other, else_id = {"cash": True}, None
        out = other
        for j, (rule, on, body, cid) in enumerate(reversed(conds)):
            out = _with_id({"if": rule, "on": on, "then": body, "else": out}, "then_id", cid)
            if j == 0:
                _with_id(out, "else_id", else_id)
        return out

    def filter_node(self, n: dict, kids: list, where: str) -> dict:
        fn = n.get("sort-by-fn")
        if not fn:
            raise ComposerImportError(f"{where}: the filter has no 'sort-by-fn'")
        raw, what = _params(n, "sort-by")
        by, _ = indicator(fn, None, raw, None, where, what, self.notes)
        from .portfolio import quote_metric
        if quote_metric(by) != by:
            self.note(f"{where}: ranking by {fn} compares price levels across tickers, so it uses quoted prices "
                      f"(`{quote_metric(by)}`); total-return levels of different tickers are not comparable.")
            by = quote_metric(by)
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
        _check(sym, "root", "symphony", notes=imp.notes)
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
            # Composer stores the corridor as a fraction of the portfolio: 0.05 is a 5% threshold (its symphony
            # database lists "Weekly Grind Popped (Threshold 10%)" with a corridor width of 0.1)
            band = _num(cw, "rebalance-corridor-width", "symphony")
            if not 0 < band < 1:
                raise ComposerImportError(
                    f"symphony: rebalance-corridor-width {cw!r} is not a threshold Composer can hold: it is a fraction of "
                    "the portfolio (0.05 = a 5% corridor), between 0 and 1.")
            spec["drift_band"] = band
        elif spec["rebalance"] == "none" and rb == "threshold":
            raise ComposerImportError("symphony: threshold rebalancing needs 'rebalance-corridor-width'")
        top = imp.kids(sym, "symphony")
        spec["tree"] = _with_id(imp.group_of(top, "symphony"), "root_id", sym.get("id"))
        # Composer puts the content under one weighting block (wt-cash-equal with a single child is a no-op): its id
        # is kept so an export writes the same wrapper back
        if (len(top) == 1 and isinstance(top[0], dict) and top[0].get("step") == "wt-cash-equal"
                and isinstance(top[0].get("children"), list) and len(top[0]["children"]) == 1):
            from .composer_export import wrapper_id
            if str(top[0].get("id")) != wrapper_id(str(sym.get("id"))):    # (not one this tool's export made)
                _with_id(spec["tree"], "wrap_id", top[0].get("id"))
        name = str(sym.get("name") or "").strip()
    else:
        spec["rebalance"] = "daily"
        spec["tree"] = imp.node(sym, "symphony")
        name = str(sym.get("name") or "").strip()
    spec["name"] = (name or "Composer symphony")[:80]
    spec["description"] = f"Composer symphony: {name}" if name else "Composer symphony"
    if spec["rebalance"] == "none":
        from .portfolio import _is_dynamic
        band = spec.get("drift_band")
        when = (("the rules are evaluated at every close, on total-return (dividend-adjusted) prices as Composer does; "
                 "it trades at that close when their target allocation changes (a different branch or selection)"
                 if _is_dynamic(spec["tree"]) else "it trades at the close")
                + (f" or when a holding drifts more than {band * 100:g} percentage points from its target" if band else ""))
        head = f"Imported from a Composer symphony with threshold rebalancing: {when}."
    else:
        head = ("Imported from a Composer symphony: conditions and filters are evaluated on each rebalance day's close, "
                "on total-return (dividend-adjusted) prices as Composer does, and traded at that close.")
    if spec["rebalance"] in ("weekly", "monthly", "quarterly", "yearly"):
        # Composer's help center: "we set the trading frequency of this symphony to quarterly, which means that
        # Composer will execute the symphony on the first trading day of each quarter" (Create a Symphony,
        # help.composer.trade/article/54-create-tutorial); its trading period is 3:45-4:00 PM ET, near the close
        # (help.composer.trade/article/63-trading-period)
        spec["rebalance_day"] = "start"
        per = {"weekly": "week", "monthly": "month", "quarterly": "quarter", "yearly": "year"}[spec["rebalance"]]
        imp.note(f"Rebalance timing: a {spec['rebalance']} symphony trades on the first trading day of each {per}, near "
                 "the close, as Composer documents it (not on the last day of the period); rebalance_day \"start\" in the "
                 "spec. Set it to \"end\" (or null) for period-end rebalancing.")
    if imp.ignored_weights:
        imp.note("Some blocks carried a 'weight' although their parent is not a specified-weight block; Composer "
                 "ignores it there (it is left over from the editor), and so does the import.")
    spec["notes"] = [head] + imp.notes
    return spec


def load(path: str | Path) -> dict:
    return convert(Path(path).read_text())


def to_portfolio(obj):
    from .portfolio import Portfolio
    return Portfolio.from_dict(convert(obj))
