"""Python-function rules are causal by construction: evaluated bar by bar on the data up to each bar, with file and
network access blocked, so caching the largest frame seen or reading the price file cannot reach later data."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr
from backtester.strategy import Strategy


def walk(seed, n=800):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    o = c * np.exp(rng.normal(0, 0.003, n))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                         "adj_close": c, "volume": 1e6, "dividend": 0.0, "split": 1.0, "open_ok": True,
                         "quote_close": c}, index=idx)


@pytest.fixture
def fake(monkeypatch):
    frames = {}
    monkeypatch.setattr(data, "load", lambda t: frames[data.canonical(t)])
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): frames[data.canonical(t)] for t in ts})
    expr._STREAMED.clear()
    expr._CALLABLE_PROBES.clear()
    return frames



def test_memo_cache_attack_is_neutralised(fake):
    df = walk(1)
    fake["X"] = df
    seen = {}

    def memo(d, ns):            # keeps the largest frame it was ever given and answers from it
        if "big" not in seen or len(d) > len(seen["big"]):
            seen["big"] = d
        big = seen["big"]
        return (big.close.shift(-1) > big.close).reindex(d.index).fillna(False)
    s = expr.evaluate(memo, expr.Namespace(df, ticker="X"))
    # bar by bar, the largest frame seen at bar i ends at bar i: "tomorrow" is always unknown -> never true
    assert not s.any()
    # a run refuses it (its whole-history answer differs from its causal one) instead of trading the leak
    seen.clear()
    expr._STREAMED.clear()
    with pytest.raises(ValueError, match="uses future data"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=memo, hold_bars=1))


def test_file_read_attack_is_blocked(fake, tmp_path):
    df = walk(2)
    fake["X"] = df
    p = tmp_path / "X.csv"
    df.to_csv(p)

    def reads(d, ns):
        full = pd.read_csv(p, index_col=0, parse_dates=True)
        return (full.close.shift(-1) > full.close).reindex(d.index).fillna(False)

    def opens(d, ns):
        with open(p) as f:
            f.read()
        return d.close > 0
    for fn in (reads, opens):
        with pytest.raises(expr.CallableIOError, match="tried to"):
            expr.evaluate(fn, expr.Namespace(df, ticker="X"))
    # the guard is only on while a rule runs
    assert open(p).read(4)


def test_data_load_inside_a_rule_is_cut_at_the_bar(monkeypatch):
    if not (data.PRICES / "SPY.csv").exists():
        pytest.skip("no data")
    spy = data.load("SPY").loc["2020-01-01":"2020-06-30"]

    def peek(d, ns):
        full = data.load("SPY")          # the rule loads the file-backed history itself
        return pd.Series(full.index[-1] > d.index[-1], index=d.index)
    s = expr.evaluate(peek, expr.Namespace(spy, ticker="SPY"))
    assert not s.any()


@pytest.mark.parametrize("fn", [lambda d, ns: d.close.shift(-1) > d.close,
                                lambda d, ns: d.close > d.close.rolling(5, center=True).mean(),
                                lambda d, ns: d.close < d.close.mean()])
def test_future_reading_rules_are_refused_and_neutralised(fake, fn):
    df = walk(3)
    fake["X"] = df
    with pytest.raises(ValueError, match="uses future data"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=fn, hold_bars=1))
    # evaluated on its own it only ever sees the data up to each bar: equal to the causal reading
    s = expr.evaluate(fn, expr.Namespace(df, ticker="X"))
    ref = pd.Series([bool(expr._last(fn(df.iloc[:i + 1], expr.Namespace(df.iloc[:i + 1], ticker="X")), "bool"))
                     for i in range(len(df))], index=df.index)
    assert (s == ref).all()


def test_legitimate_callables_give_the_same_answers(fake):
    df = walk(4)
    fake["X"] = df

    def causal(d, ns):
        return (d.close < d.close.rolling(10).min().shift(1)) & (ns["rsi"](2) < 30)

    def rank(d, ns):
        return d.close.pct_change(5)
    ns = expr.Namespace(df, ticker="X")
    assert (expr.evaluate(causal, ns) == expr._as_bool(causal(df, ns), df.index)).all()
    v = expr.evaluate_value(rank, expr.Namespace(df, ticker="X"))
    pd.testing.assert_series_equal(v, rank(df, ns).astype(float), check_names=False)
    st = Strategy(cash_rate=None, universe=["X"], entry=causal, hold_bars=2, rank_by=rank)
    r = engine.run(st)
    assert len(r.trades) > 0
    assert any(n.startswith("Python rule causal(): evaluated bar by bar") for n in st.notes)


def test_trust_vectorized_skips_streaming_but_is_still_probed(fake):
    # (round 13: `vectorized_causal` is checked exactly - streamed - so the one-call path is `trust_vectorized`)
    df = walk(5)
    fake["X"] = df
    from backtester import sandbox

    def fast(d, ns):
        return d.close > d.close.shift(1)
    fast.trust_vectorized = True
    jobs, steps = sandbox.STATS["jobs"], sandbox.STATS["steps"]
    s = expr.evaluate(fast, expr.Namespace(df, ticker="X"))
    # one call on the whole history (in one sealed process), not bar by bar
    assert sandbox.STATS["jobs"] == jobs + 1 and sandbox.STATS["steps"] == steps and s.sum() > 0

    def leaky(d, ns):
        return d.close.shift(-1) > d.close
    leaky.trust_vectorized = True
    with pytest.raises(ValueError, match="uses future data"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=leaky, hold_bars=1))


def test_streaming_speed_on_5000_bars():
    df = walk(6, n=5000)

    def causal(d, ns):
        return (d.close < d.close.rolling(10).min().shift(1)) & (d.close > d.close.rolling(200).mean())
    t = time.perf_counter()
    expr._STREAMED.clear()
    expr.evaluate(causal, expr.Namespace(df, ticker="X"))
    assert time.perf_counter() - t < 15
