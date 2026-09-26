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


HAVE = bool(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


@pytest.mark.parametrize("rule", ['ref(sym("SPY").close, 2 and 1) > sym("SPY").open * 1.003', "ref(close, 2 and 1) > open",
                                  "open < sma(1+0)", "open < ref(close, 3 - 2 - 1 + 1 and 1)", "open > ref(close, ~-2)"])
def test_open_guard_rejects_computed_offsets(rule):
    assert not expr.open_safe(rule)


@needs_data
def test_probe_perturbs_other_tickers():
    df = data.load("QQQ")
    assert expr.open_time_probe('sym("SPY").close > sym("SPY").open', df, "QQQ") is not None
    assert expr.open_time_probe('sym("SPY").open > ref(sym("SPY").close, 1)', df, "QQQ") is None


@needs_data
@pytest.mark.parametrize("rule", ["is_week_end()", "weekly(close) > ref(weekly(close), 1)", "weekly_rsi(14) > 50",
                                  "close > weekly_sma(10)", "trading_days_left_in_month <= 1", "is_month_end()"])
def test_period_ends_have_no_hindsight_about_unscheduled_closures(rule):
    df = data.load("SPY")
    full = expr.evaluate(rule, expr.Namespace(df))
    for cut in ("2001-09-10", "2012-10-26", "2018-12-04"):
        part = expr.evaluate(rule, expr.Namespace(df.loc[:cut]))
        assert (full.reindex(part.index) == part).all(), cut
    assert not bool(full.loc["2001-09-10"]) if rule == "is_week_end()" else True


@needs_data
def test_large_stocks_are_not_stripped_from_membership_as_etfs():
    m = data.membership()
    assert "ORCL" in m.columns and m["ORCL"].sum() > 100
    for etf in ("TQQQ", "SQQQ", "QQQ"):
        assert etf not in m.columns


@needs_data
def test_stale_opens_are_flagged():
    assert not data.load("^GSPC").loc["1990"]["open_ok"].any()
    assert data.load("SPY").loc["2010":]["open_ok"].all()


@needs_data
def test_unadjusted_corporate_actions_are_flagged():
    assert pd.Timestamp("2011-12-21") in data.corporate_action_days("EXPE")
    assert len(data.corporate_action_days("SPY")) == 0
