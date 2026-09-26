import numpy as np
import pandas as pd
import pytest

from backtester import data, expr


def _frame(n=300, seed=3):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, n))), index=idx)
    o = c.shift(1).fillna(100)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005, "low": np.minimum(o, c) * 0.995, "close": c,
                         "volume": 1e6, "dividend": 0.0, "adj_close": c})


def test_ema_seeded_with_sma_like_tradingview():
    df = _frame()
    e = expr.evaluate_value("ema(close, 10)", expr.Namespace(df))
    assert e.iloc[:9].isna().all()
    assert abs(e.iloc[9] - df["close"].iloc[:10].mean()) < 1e-12
    a = 2 / 11
    assert abs(e.iloc[10] - (a * df["close"].iloc[10] + (1 - a) * e.iloc[9])) < 1e-12


def test_sar_flips_and_stays_on_the_correct_side():
    df = _frame()
    s = expr.evaluate_value("sar(0.02, 0.2)", expr.Namespace(df))
    below = s < df["low"]
    above = s > df["high"]
    assert (below | above).iloc[2:].mean() > 0.95  # SAR sits outside the bar except on flip bars
    assert below.any() and above.any()


def test_weekly_expression_uses_completed_weeks_only():
    df = _frame()
    ns = expr.Namespace(df)
    full = expr.evaluate_value("weekly(atr(14))", ns)
    for cut in ("2020-06-10", "2020-08-19", "2020-10-02"):
        part = expr.evaluate_value("weekly(atr(14))", expr.Namespace(df.loc[:cut]))
        common = part.index
        assert (full.reindex(common) - part).abs().max() < 1e-12 or part.dropna().empty
    # values change only on a week's last session
    ch = full.diff().fillna(0).ne(0)
    assert (full.index[ch].dayofweek == 4).mean() > 0.9


def test_weekly_function_matches_dedicated_weekly_rsi():
    df = _frame(600)
    ns = expr.Namespace(df)
    a = expr.evaluate_value("weekly(rsi(14))", ns)
    b = expr.evaluate_value("weekly_rsi(14)", ns)
    assert (a - b).abs().max() < 1e-9


def test_weekly_is_not_open_safe():
    assert not expr.open_safe("open > weekly(close)")
