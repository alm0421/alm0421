"""A small, safe expression language for entry/exit rules.

Expressions are Python-like and evaluated per ticker on daily bars, e.g.

    down_days >= 5
    rsi(2) < 10 and close > sma(close, 200)
    ret(1) <= -0.03 and sym("SPY").close > sma(sym("SPY").close, 200)

`and`, `or`, `not` and chained comparisons (0.1 < ibs < 0.3) work on whole
series.  All prices are split- and dividend-adjusted.  See VARIABLES and
FUNCTIONS below (or `python -m backtester --help-expr`) for the vocabulary.
"""
from __future__ import annotations

import ast
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import data

# ---------------------------------------------------------------- indicators


def _s(x: Any, like: pd.Series) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x, index=like.index, dtype=float)


def streak(close: pd.Series, direction: int) -> pd.Series:
    """Number of consecutive closes strictly down (direction=-1) or up (+1)."""
    chg = np.sign(close.diff()).fillna(0).to_numpy()
    hit = chg == direction
    out = np.zeros(len(hit), dtype=float)
    run = 0
    for i, h in enumerate(hit):
        run = run + 1 if h else 0
        out[i] = run
    return pd.Series(out, index=close.index)


def rsi_wilder(x: pd.Series, n: int) -> pd.Series:
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(d.notna())


class Bars:
    """Price fields of another ticker aligned to the current ticker's dates."""

    def __init__(self, df: pd.DataFrame, index: pd.Index):
        a = df.reindex(index.union(df.index)).ffill().reindex(index)
        self.open, self.high, self.low, self.close = a["open"], a["high"], a["low"], a["close"]
        self.volume = a["volume"]


class Namespace(dict):
    """Evaluation namespace for one ticker; derived variables are lazy."""

    def __init__(self, df: pd.DataFrame, extra: dict[str, Any] | None = None):
        super().__init__()
        self.df = df
        c = df["close"]
        self.update({
            "open": df["open"], "high": df["high"], "low": df["low"], "close": c,
            "volume": df["volume"], "price": c,
            "True": True, "False": False,
        })
        self.update(self._functions())
        if extra:
            self.update(extra)

    # lazily computed variables
    def __missing__(self, key: str) -> Any:
        df, c = self.df, self.df["close"]
        idx = df.index
        lazy: dict[str, Callable[[], Any]] = {
            "down_days": lambda: streak(c, -1),
            "up_days": lambda: streak(c, +1),
            "ibs": lambda: ((c - df["low"]) / (df["high"] - df["low"])).where(df["high"] > df["low"], 0.5),
            "gap": lambda: df["open"] / c.shift(1) - 1,
            "range": lambda: df["high"] / df["low"] - 1,
            "change": lambda: c.pct_change(fill_method=None),
            "dow": lambda: pd.Series(idx.dayofweek, index=idx),
            "day": lambda: pd.Series(idx.day, index=idx),
            "month": lambda: pd.Series(idx.month, index=idx),
            "year": lambda: pd.Series(idx.year, index=idx),
            "dollar_volume": lambda: c * df["volume"],
            "trading_day_of_month": lambda: pd.Series(idx.to_period("M"), index=idx).groupby(idx.to_period("M")).cumcount() + 1,
            "trading_days_left_in_month": lambda: pd.Series(1, index=idx).groupby(idx.to_period("M")).transform(lambda s: np.arange(len(s), 0, -1)),
        }
        if key in lazy:
            v = lazy[key]()
            self[key] = v
            return v
        raise NameError(f"unknown name {key!r} in expression (see --help-expr)")

    def _functions(self) -> dict[str, Callable]:
        df = self.df
        c = df["close"]

        def pick(args, default_x, default_n):
            """Accept f(n), f(x, n), f(n, x) or f()."""
            x, n = default_x, default_n
            for a in args:
                if isinstance(a, pd.Series):
                    x = a
                else:
                    n = a
            return x, int(n)

        def sma(*a):
            x, n = pick(a, c, 20)
            return x.rolling(n, min_periods=n).mean()

        def ema(*a):
            x, n = pick(a, c, 20)
            return x.ewm(span=n, adjust=False, min_periods=n).mean()

        def highest(*a):
            x, n = pick(a, df["high"], 20)
            return x.rolling(n, min_periods=n).max()

        def lowest(*a):
            x, n = pick(a, df["low"], 20)
            return x.rolling(n, min_periods=n).min()

        def stdev(*a):
            x, n = pick(a, c, 20)
            return x.rolling(n, min_periods=n).std()

        def zscore(*a):
            x, n = pick(a, c, 20)
            return (x - x.rolling(n).mean()) / x.rolling(n).std()

        def ref(x, n=1):
            x = _s(x, c)
            if x.dtype == bool:
                return x.shift(int(n), fill_value=False)
            return x.shift(int(n))

        def ret(*a):
            x, n = pick(a, c, 1)
            return x / x.shift(n) - 1

        def rsi(*a):
            x, n = pick(a, c, 14)
            return rsi_wilder(x, n)

        def atr(n=14):
            n = int(n)
            tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
            return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

        def natr(n=14):
            return atr(n) / c

        def volatility(n=20):
            return c.pct_change(fill_method=None).rolling(int(n)).std() * np.sqrt(252)

        def bb_upper(n=20, k=2.0):
            return sma(c, n) + k * stdev(c, n)

        def bb_lower(n=20, k=2.0):
            return sma(c, n) - k * stdev(c, n)

        def pct_rank(*a):
            x, n = pick(a, c, 252)
            return x.rolling(n, min_periods=n).rank(pct=True)

        def crossover(a, b):
            a, b = _s(a, c), _s(b, c)
            return (a > b) & (a.shift() <= b.shift())

        def crossunder(a, b):
            a, b = _s(a, c), _s(b, c)
            return (a < b) & (a.shift() >= b.shift())

        def count(cond, n):
            return _s(cond, c).astype(float).rolling(int(n), min_periods=1).sum()

        def down_streak(x=None):
            return streak(c if x is None else _s(x, c), -1)

        def up_streak(x=None):
            return streak(c if x is None else _s(x, c), +1)

        def sym(ticker: str) -> Bars:
            return Bars(data.load(ticker), df.index)

        return {
            "sma": sma, "ma": sma, "ema": ema, "highest": highest, "lowest": lowest,
            "stdev": stdev, "zscore": zscore, "ref": ref, "ret": ret, "roc": ret,
            "rsi": rsi, "atr": atr, "natr": natr, "volatility": volatility,
            "bb_upper": bb_upper, "bb_lower": bb_lower, "pct_rank": pct_rank,
            "crossover": crossover, "crossunder": crossunder, "count": count,
            "down_streak": down_streak, "up_streak": up_streak,
            "cummax": lambda x: _s(x, c).cummax(), "cummin": lambda x: _s(x, c).cummin(),
            "sym": sym, "abs": np.abs, "maximum": np.maximum, "minimum": np.minimum,
            "log": np.log,
        }


HELP = """
Variables (per bar, adjusted prices):
  open high low close volume        OHLCV
  change                            1-day % change of close (0.01 = 1%)
  down_days / up_days               consecutive down / up closes ending today
  ibs                               internal bar strength (close-low)/(high-low)
  gap                               open / previous close - 1
  range                             high/low - 1
  dow month day year                calendar (dow: 0=Mon .. 4=Fri)
  trading_day_of_month, trading_days_left_in_month
  dollar_volume                     close * volume
Position variables (exit rules only):
  bars_held  entry_price  pnl (open trade return, 0.05 = +5%)
Functions (x defaults to close; n = lookback in bars):
  sma(x,n) ema(x,n) highest(x,n) lowest(x,n) stdev(x,n) zscore(x,n)
  ret(x,n) ref(x,n) rsi(x,n) atr(n) natr(n) volatility(n) pct_rank(x,n)
  bb_upper(n,k) bb_lower(n,k) crossover(a,b) crossunder(a,b) count(cond,n)
  down_streak(x) up_streak(x)       consecutive down/up count of any series
  cummax(x) cummin(x)               running max/min (all-time high/low)
  sym("SPY").close  -> another ticker's series aligned to this one
Operators: + - * / < <= > >= == != and or not, e.g. 0.1 < ibs < 0.3
"""


# ---------------------------------------------------------------- evaluator

_ALLOWED = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name,
    ast.Load, ast.Constant, ast.Attribute, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.BitAnd, ast.BitOr, ast.Invert,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.keyword,
)


_ATTRS = {"open", "high", "low", "close", "volume"}


class _Vectorize(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        out = node.values[0]
        for v in node.values[1:]:
            out = ast.BinOp(left=out, op=op, right=v)
        return out

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.UnaryOp(op=ast.Invert(), operand=node.operand)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        parts, left = [], node.left
        for op, right in zip(node.ops, node.comparators):
            parts.append(ast.Compare(left=left, ops=[op], comparators=[right]))
            left = right
        out = parts[0]
        for p in parts[1:]:
            out = ast.BinOp(left=out, op=ast.BitAnd(), right=p)
        return out


def compile_expr(text: str):
    tree = ast.parse(text.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"not allowed in expression: {type(node).__name__} in {text!r}")
        if isinstance(node, ast.Attribute) and node.attr not in _ATTRS:
            raise ValueError(f"only {sorted(_ATTRS)} can follow a '.', e.g. sym(\"SPY\").close")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("private names are not allowed")
    tree = ast.fix_missing_locations(_Vectorize().visit(tree))
    return compile(tree, "<rule>", "eval")


POSITION_VARS = {"bars_held", "entry_price", "pnl"}


def names_in(text: str) -> set[str]:
    return {n.id for n in ast.walk(ast.parse(text.strip(), mode="eval")) if isinstance(n, ast.Name)}


def evaluate(text: str, ns: Namespace) -> pd.Series:
    """Evaluate a rule to a boolean Series (NaN -> False)."""
    code = compile_expr(text)
    out = eval(code, {"__builtins__": {}}, ns)  # noqa: S307 - AST is whitelisted above
    idx = ns.df.index
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=idx)
    if out.dtype != bool:
        out = out.fillna(0).astype(bool)
    return out.reindex(idx, fill_value=False)


def evaluate_value(text: str, ns: Namespace) -> pd.Series:
    """Evaluate a numeric expression (used for ranking)."""
    out = eval(compile_expr(text), {"__builtins__": {}}, ns)  # noqa: S307
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=ns.df.index, dtype=float)
    return out.astype(float)


# ---------------------------------------------------------------- lookahead guard

OPEN_SAFE_NAMES = {"gap", "dow", "month", "day", "year", "trading_day_of_month",
                   "trading_days_left_in_month", "open", "True", "False"}
# functions whose series argument defaults to today's close/high/low when omitted
_DEFAULTS_TO_CLOSE = {"sma", "ma", "ema", "highest", "lowest", "stdev", "zscore", "ret", "roc", "rsi",
                      "pct_rank", "down_streak", "up_streak"}
# functions that always read today's close/high/low
_ALWAYS_CLOSE = {"atr", "natr", "volatility", "bb_upper", "bb_lower"}


def open_safe(rule: str) -> bool:
    """True if `rule` can be evaluated at the bar's open, i.e. uses no data from later in the bar."""
    def ok(node) -> bool:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            f = node.func.id
            if f == "ref":
                n = node.args[1] if len(node.args) > 1 else None
                if n is None:
                    return True
                return isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and n.value >= 1
            if f in _ALWAYS_CLOSE:
                return False
            if f in _DEFAULTS_TO_CLOSE:
                series_args = [a for a in node.args if not isinstance(a, ast.Constant)]
                if not series_args:
                    return False
            if f == "sym":
                return True
            return all(ok(a) for a in node.args)
        if isinstance(node, ast.Attribute):
            return node.attr == "open" and ok(node.value)
        if isinstance(node, ast.Name):
            return node.id in OPEN_SAFE_NAMES
        return all(ok(ch) for ch in ast.iter_child_nodes(node))

    return ok(ast.parse(rule.strip(), mode="eval"))
