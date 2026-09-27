"""Adversarial tests of the open-fill lookahead guard (expr.open_safe, compile-time argument shapes, the runtime
argument checks and the empirical expr.open_time_probe).

Every rule here is either refused for fills at the open, or shown to be knowable at the open: truncating the
data at a day D and replacing D's close/high/low/volume by other valid values never changes the decision on D.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr
from backtester.strategy import Strategy

HAVE = bool(data.available_tickers()) and "QQQ" in data.available_tickers()
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")

REVIEWER_RULE = "highest(abs(1)) > lowest(abs(1))*1.1 or (year < 2017 and sma(abs(1)) > open) or year >= 2017"
REVIEWER_EXIT = "highest(abs(1)) > lowest(abs(1))*1.1 or (year < 2017 and sma(abs(1)) < open) or year >= 2017"


def frame(n: int = 600, seed: int = 0, start: str = "2015-01-02") -> pd.DataFrame:
    """Random daily bars with valid OHLC (low <= open, close <= high), volume and total-return close."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0, 0.006, n))
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    v = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v, "dividend": 0.0,
                         "adj_close": c, "open_ok": True}, index=idx)


def perturbed_cut(df: pd.DataFrame, i: int, rng) -> pd.DataFrame:
    """Data up to bar i with bar i's close/high/low/volume replaced by random values consistent with its open."""
    cut = df.iloc[: i + 1].copy()
    d = cut.index[-1]
    o = float(cut.at[d, "open"])
    size = float(rng.choice([1e-5, 1e-3, 0.01, 0.05, 0.2]))
    c = o * (1 + rng.uniform(-1, 1) * size)
    cut.at[d, "close"] = c
    cut.at[d, "adj_close"] = c
    cut.at[d, "high"] = max(o, c) * (1 + rng.uniform(0, size))
    cut.at[d, "low"] = min(o, c) * (1 - rng.uniform(0, min(size, 0.5)))
    cut.at[d, "volume"] = float(rng.uniform(1, 1e7))
    return cut


def decisions_invariant(rule: str, df: pd.DataFrame, days: int = 12, reps: int = 3, seed: int = 0,
                        other: pd.DataFrame | None = None) -> None:
    """The decision on each sampled day D is the same on the full data and on data cut at D with D's unknown-at-
    the-open fields replaced (for this ticker and, if given, the sym("ZZZ") ticker)."""
    rng = np.random.default_rng(seed)
    try:
        if other is not None:
            expr._SYM_OVERRIDE["ZZZ"] = other
        full = expr.evaluate(rule, expr.Namespace(df)).to_numpy()
        for i in rng.choice(np.arange(1, len(df)), size=days, replace=False):
            for _ in range(reps):
                cut = perturbed_cut(df, int(i), rng)
                if other is not None:
                    day = cut.index[-1]
                    oc = other.loc[:day]
                    if day in oc.index:
                        oc = perturbed_cut(oc, len(oc) - 1, rng)
                    expr._SYM_OVERRIDE["ZZZ"] = oc
                got = bool(expr.evaluate(rule, expr.Namespace(cut)).iloc[-1])
                assert got == bool(full[i]), f"{rule!r} changed on {df.index[i].date()} (lookahead)"
                if other is not None:
                    expr._SYM_OVERRIDE["ZZZ"] = other
    finally:
        expr._SYM_OVERRIDE.clear()


# ------------------------------------------------------------------ refused at the open
REFUSED = [
    REVIEWER_RULE,
    REVIEWER_EXIT,
    # computed lookbacks: abs / sqrt / log / min / max / pow / constant folding / bitwise / booleans
    "open > sma(abs(1))", "open > sma(sqrt(400))", "open > highest(log(20))", "open > sma(maximum(20, 1))",
    "open > sma(minimum(20, 30))", "open > sma(2**4)", "open > sma(2*10)", "open > sma(abs(-20))", "open > sma(1+0)",
    "open > sma(-(-1))", "open > sma(20 | 0)", "open > sma(~-21)", "open > sma(True + 19)", "open > sma(2 and 20)",
    "open > sma(round(20))", "open > sma(int(20))", "open > sma(min(1, 2))", "open > sma(pow(2, 2))",
    "open > lowest(open, abs(3))", "open > sma(open, sqrt(25))", "open > rsi(open, 1 + 1)",
    # keyword arguments
    "open > sma(x=open, n=5)", "open < ref(x=close, n=1)", "open > sma(open, n=abs(5))",
    # computed / zero / fractional offsets
    "open < ref(close, abs(1))", "open < ref(close, 1 - 1)", "open < ref(close, 0)", "open < ref(close, 0.5)",
    "open < ref(close, 2 and 1)", "open > ref(close, ~-2)", "open < ref(close, year - year + 1)",
    "open < ref(close, sma(abs(1)))",
    # rolling functions with the default (today's close/high/low) series
    "open > sma(20)", "open > rsi(2)", "ret(1) < 0", "open > highest(5)", "open < lowest(5)", "open > ema(5)",
    "open > drawdown(252)", "open > stdev(20)", "zscore(20) < -2", "pct_rank(20) > 0.5", "tret(5) > 0",
    "down_streak() >= 2", "up_streak() >= 2", "diff() > 0", "diff(5) > 0", "diff(close) > 0", "diff(high, 2) > 0",
    "diff(open, abs(1)) > 0", "diff(open * 0 + close) > 0", "ta.change(close) > 0", "ta.mom(close, 3) > 0",
    # today's bar, directly or through variables / other indicators
    "gap < -0.01 and close > open", "open < close", "high > open", "low < open", "volume > 0", "ibs < 0.2",
    "range > 0.01", "change < 0", "down_days >= 1", "dollar_volume > 0", "tr > 0", "price > open", "market_cap > 0",
    "cummax(close) > open", "open > maximum(close, 0)", "open > abs(close)", "open > sma(open * 0 + close, 3)",
    "open > sma(high, 5)", "crossover(open, close)", "open > atr(14)", "open > volatility(20)", "open > vwap(20)",
    "open > bb_upper(20, 2)", "macd() > 0", "count(gap < 0, 5) > 2", "bars_since(gap < 0) > 2", "obv() > 0",
    "open > supertrend(10, 3)", "open > sar(0.02, 0.2)", "tbill_ret(5) > 0",
    # chained comparisons and bitwise operators hiding the close
    "0 < gap < close", "(open > 1) & (close > 1)", "(gap < 0) | (close < open)", "~(close > open)",
    "open > 1 > close - open", "not close > open",
    # other tickers: only sym(...).open is known at the open
    'sym("SPY").close > sym("SPY").open', 'sym("SPY").high > 0', 'open > sma(sym("SPY").close, 5)',
    'sym("SPY").volume > 0', 'sym("SPY").tr > 0',
    # weekly / monthly bars and period-end flags
    "open > weekly(close)", "open > monthly(close)", "open > weekly_close()", "open > monthly_close()",
    "weekly_rsi(14) > 50", "open > weekly_sma(10)", "is_month_end() and close > open", "is_week_end(close)",
    # position variables at the open
    "pnl > 0.05", "bars_held >= 3", "open > highest_since_entry", "open < lowest_since_entry",
]


@pytest.mark.parametrize("rule", REFUSED)
def test_refused_at_the_open(rule):
    assert not expr.open_safe(rule)
    s = Strategy(universe=["QQQ"], entry=rule, entry_fill="open", hold_bars=1)
    with pytest.raises((ValueError, SyntaxError)):
        s.validate()
    s = Strategy(universe=["QQQ"], entry="gap < 0", exit_when=rule, exit_when_fill="open")
    with pytest.raises((ValueError, SyntaxError)):
        s.validate()


# ------------------------------------------------------------------ allowed at the open, and really knowable
SAFE = [
    "gap < -0.02", "open > ref(close, 1)", "open > sma(open, 5)", "sma(ref(close, 1), 20) < open",
    "rsi(ref(close, 1), 2) < 10", "year >= 2017 and gap > 0", "crossover(open, ref(high, 1))", "abs(gap) > 0.01",
    "down_streak(ref(close, 1)) >= 3", "ref(close, 1) > ref(sma(close, 20), 1)", "open > highest(ref(high, 1), 20)",
    "open < lowest(ref(low, 1), 5)", "ref(ibs, 1) < 0.2 and gap < 0", "ref(rsi(2), 1) < 10", "ref(close, 2) > ref(close)",
    "dow == 0 and month != 12", "trading_day_of_month <= 3", "trading_days_left_in_month <= 2",
    # scheduled-calendar flags: from the published NYSE schedule, known before the open
    "is_month_end()", "is_week_end() and gap < 0", "is_quarter_end() or is_year_end()",
    "maximum(gap, 0) > 0.001", "sqrt(abs(gap)) > 0.05", "open > ref(weekly(close), 1)", "ref(macd_hist(), 1) > 0",
    "open / ref(close, 1) - 1 < -0.01 or day == 15", "ret(open, 3) < -0.05", "zscore(ref(close, 1), 20) < -1",
    "cummax(open) > open * 1.2", "drawdown(ref(close, 1), 60) < -0.1", "gap > 0",
    'sym("ZZZ").open > ref(sym("ZZZ").close, 1)', 'open > ref(sym("ZZZ").high, 1) and gap < 0',
    "diff(open) > 0", "diff(open, 3) < 0", "diff(2, open) > 0", "ta.change(open) > 0", "ta.mom(open, 5) < 0",
    "diff(ref(close, 1), 5) > 0", "diff(gap, 1) > 0", 'diff(sym("ZZZ").open) > 0',
]


@pytest.mark.parametrize("rule", SAFE)
def test_open_safe_rules_are_knowable_at_the_open(rule):
    assert expr.open_safe(rule), rule
    df = frame(700, seed=3)
    other = frame(700, seed=4)
    decisions_invariant(rule, df, other=other if "ZZZ" in rule else None)


# ------------------------------------------------------------------ runtime: scalars and computed lookbacks refused
def test_runtime_refuses_scalars_where_series_or_lookbacks_belong():
    ns = expr.Namespace(frame(100))
    for bad in (lambda: ns["sma"](np.int64(5)), lambda: ns["sma"](np.abs(5)), lambda: ns["sma"](1, 2),
                lambda: ns["highest"](np.float64(3.0)), lambda: ns["ref"](ns["close"], np.int64(1)),
                lambda: ns["cummax"](5), lambda: ns["down_streak"](3), lambda: ns["drawdown"](np.int64(5)),
                lambda: ns["sma"](True)):
        with pytest.raises(ValueError):
            bad()
    assert np.isfinite(ns["sma"](5).iloc[-1]) and np.isfinite(ns["sma"](ns["open"], 5.0).iloc[-1])
    for rule in ("sma(abs(1)) > 0", "ref(close, abs(1)) > 0", "atr(abs(14)) > 0", "volatility(n=abs(20)) > 0",
                 "sym(1).close > 0", "cummax(1) > 0"):
        with pytest.raises(ValueError):
            expr.evaluate(rule, ns)


# ------------------------------------------------------------------ the empirical probe
def test_probe_catches_leaks_without_a_fingerprint():
    df = frame(900, seed=5)
    for rule in ("close > open * 1.00001", "close < open * 0.99999", "range > 0.02", "high > open * 1.001",
                 "volume > 3000000", "ibs < 0.5", "open > sma(4)", "open < highest(2) * 0.999",
                 "(high / low - 1 > 0.109) or close > open"):
        assert expr.open_time_probe(rule, df) is not None, rule
    for rule in SAFE[:12]:
        assert expr.open_time_probe(rule, df) is None, rule


def test_probe_samples_the_whole_history():
    df = frame(3000, seed=6, start="2008-01-02")    # the leak is only in the first year, > 2000 bars before the end
    assert expr.open_time_probe("year < 2009 and close > open", df) is not None
    assert expr.open_time_probe("year >= 2009 or ref(close, 1) > open", df) is None


def test_probe_perturbs_other_tickers_separately():
    df, other = frame(800, seed=7), frame(800, seed=8)
    try:
        orig = data.load
        data.load = lambda t: other if data.canonical(t) == "ZZZ" else orig(t)
        assert expr.open_time_probe('sym("ZZZ").close > sym("ZZZ").open', df) is not None
        assert expr.open_time_probe('gap < 0 and sym("ZZZ").low < ref(sym("ZZZ").low, 1)', df) is not None
        assert expr.open_time_probe('sym("ZZZ").open > ref(sym("ZZZ").close, 1)', df) is None
    finally:
        data.load = orig


# ------------------------------------------------------------------ generated rules (property test)
SAFE_ATOMS = ["open", "gap", "dow", "month", "day", "year", "trading_day_of_month", "trading_days_left_in_month",
              'sym("ZZZ").open', "ref(close, 1)", "ref(high, 1)", "ref(low, 2)", "ref(volume, 1)", 'ref(sym("ZZZ").close, 1)']


def _gen(rng, depth=0, bias=False) -> str:
    """A random numeric expression from the rule grammar (bias: mostly atoms known at the open, so that plenty
    of generated rules pass open_safe and get checked)."""
    atoms = ["open", "close", "high", "low", "volume", "gap", "dow", "month", "day", "year", "ibs", "range",
             "change", "trading_day_of_month", "trading_days_left_in_month", "down_days", 'sym("ZZZ").open',
             'sym("ZZZ").close', "tr"]
    if bias and rng.random() < 0.9:
        atoms = SAFE_ATOMS
    if depth > 2 or rng.random() < 0.3:
        return str(rng.choice(atoms)) if rng.random() < 0.85 else str(round(float(rng.uniform(-2, 150)), 3))
    n = int(rng.integers(1, 30))
    k = int(rng.integers(0, 4))
    x = _gen(rng, depth + 1, bias)
    y = _gen(rng, depth + 1, bias)
    choices = [
        f"ref({x}, {k})", f"ref({x})", f"sma({x}, {n})", f"ema({x}, {n})", f"rsi({x}, {n})", f"highest({x}, {n})",
        f"lowest({x}, {n})", f"ret({x}, {n})", f"stdev({x}, {n})", f"zscore({x}, {n})", f"drawdown({x})",
        f"cummax({x})", f"down_streak({x})", f"abs({x})", f"maximum({x}, {y})", f"minimum({x}, {y})",
        f"({x} + {y})", f"({x} - {y})", f"({x} * {y})", f"({x} / ({y}))", f"sma({n})", f"rsi({n})", f"highest({n})",
        f"atr({n})", f"crossover({x}, {y})", f"weekly({x})", f"count({x} > {y}, {n})", f"-{x}",
        f"sma({x}, abs({n}))", f"ref({x}, {k} + 0)", f"sqrt(abs({x}))", f"diff({x}, {k})", f"diff({n})",
    ]
    return str(rng.choice(choices))


def _gen_rule(rng) -> str:
    ops = [">", "<", ">=", "<="]
    parts = []
    bias = rng.random() < 0.6
    for _ in range(int(rng.integers(1, 4))):
        a, b = _gen(rng, bias=bias), _gen(rng, bias=bias)
        parts.append(f"{a} {rng.choice(ops)} {b}" if rng.random() < 0.8
                     else f"{a} {rng.choice(ops)} {b} {rng.choice(ops)} {_gen(rng, bias=bias)}")
    out = parts[0]
    for p in parts[1:]:
        out = f"({out}) {rng.choice(['and', 'or'])} ({'not ' if rng.random() < 0.2 else ''}{p})"
    return out


def test_generated_open_safe_rules_never_see_the_rest_of_the_bar():
    rng = np.random.default_rng(2024)
    df, other = frame(400, seed=11), frame(400, seed=12)
    checked = refused = 0
    for _ in range(1000):
        rule = _gen_rule(rng)
        if not expr.open_safe(rule):
            refused += 1
            continue
        try:
            expr._SYM_OVERRIDE["ZZZ"] = other
            expr.evaluate(rule, expr.Namespace(df))
        except Exception:  # noqa: BLE001 - a rule that cannot run (bool arithmetic, 1/0, ...) is not a leak
            continue
        finally:
            expr._SYM_OVERRIDE.clear()
        decisions_invariant(rule, df, days=6, reps=2, seed=checked, other=other)
        checked += 1
    print("generated rules checked", checked, "refused", refused)
    assert checked >= 80 and refused >= 200, (checked, refused)


# ------------------------------------------------------------------ the reviewer's repros, end to end
@needs_data
def test_reviewer_entry_repro_is_refused():
    s = Strategy(universe=["QQQ"], entry=REVIEWER_RULE, entry_fill="open", hold_bars=0, hold_exit_fill="close",
                 end="2016-12-31")
    with pytest.raises(ValueError):
        engine.run(s)


@needs_data
def test_reviewer_exit_repro_is_refused():
    s = Strategy(universe=["QQQ"], entry="gap > -1", exit_when=REVIEWER_EXIT, exit_when_fill="open",
                 start="2000", end="2016")
    with pytest.raises(ValueError):
        engine.run(s)


@needs_data
def test_open_fill_truncation_invariance_on_real_data():
    rule = 'gap < -0.005 and ref(close, 1) < sma(ref(close, 1), 10) and sym("SPY").open > ref(sym("SPY").low, 1)'
    assert expr.open_safe(rule)
    s = dict(universe=["QQQ"], entry=rule, entry_fill="open", hold_bars=0, hold_exit_fill="close", start="2015-01-01")
    full = engine.run(Strategy(**s, end="2020-12-31"))
    cut = engine.run(Strategy(**s, end="2018-06-29"))
    a = full.trades[pd.to_datetime(full.trades["exit_date"]) < pd.Timestamp("2018-06-29")].reset_index(drop=True)
    b = cut.trades[pd.to_datetime(cut.trades["exit_date"]) < pd.Timestamp("2018-06-29")].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["entry_date", "exit_date", "pnl"]], b[["entry_date", "exit_date", "pnl"]])
    assert expr.open_time_probe(rule, data.load("QQQ"), "QQQ") is None
