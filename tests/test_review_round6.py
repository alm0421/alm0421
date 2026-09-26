"""Reviewer round 6: Nasdaq-100 open entries, delistings in both engines, the equity anchor row, market-cap
phrases and spin-off distributions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import calendar as cal_
from backtester import data, engine, metrics, parser, portfolio
from backtester.strategy import Strategy

AVAIL = set(data.available_tickers())
needs = lambda *t: pytest.mark.skipif(not set(t) <= AVAIL, reason="price data not downloaded")  # noqa: E731


# ------------------------------------------------------------------ equity anchor row
def test_anchor_day_is_the_previous_session():
    assert cal_.anchor_day("2019-01-02") == pd.Timestamp("2018-12-31")
    assert cal_.anchor_day("2019-01-07") == pd.Timestamp("2019-01-04")
    traded = pd.DatetimeIndex(["2019-01-04", "2019-01-05", "2019-01-07"])   # a crypto-style calendar
    assert cal_.anchor_day("2019-01-07", traded) == pd.Timestamp("2019-01-05")


@needs("SPY")
def test_equity_curves_anchor_on_a_trading_session():
    p = parser.parse("hold SPY since 2019-01-02")
    r = portfolio.run(p)
    assert r.equity.index[0] == pd.Timestamp("2018-12-31") and r.equity.index[1] == pd.Timestamp("2019-01-02")
    s = Strategy(universe=["SPY"], entry="close > 0", exit_when="close < 0", start="2019-01-02", end="2019-12-31")
    e = engine.run(s)
    assert e.equity.index[0] == pd.Timestamp("2018-12-31")
    # the anchor's year has no row of its own in the yearly tables
    nv = r.equity / r.equity.iloc[0]
    assert 2018 not in metrics.yearly_detail(nv, r.trades, r.exposure).index
    assert 2018 not in metrics.yearly_returns({"x": nv}).index
    y = metrics.yearly_detail(nv, r.trades, r.exposure)
    assert abs(y.loc[2019, "return"] - (nv.loc["2019"].iloc[-1] - 1)) < 1e-12


# ------------------------------------------------------------------ delistings
@pytest.fixture
def intc_delisted(monkeypatch):
    orig = data.load.__wrapped__

    def load(t):
        df = orig(t)
        return df.loc[:"2015-06-30"] if data.canonical(t) == "INTC" else df

    import functools
    monkeypatch.setattr(data, "load", functools.lru_cache(maxsize=None)(load))
    data.quality.cache_clear()
    yield
    data.quality.cache_clear()


@needs("INTC", "SPY")
def test_portfolio_sells_a_delisted_holding_at_its_last_close(intc_delisted):
    p = parser.parse("hold 50% INTC and 50% SPY, rebalance quarterly, since 2014")
    r = portfolio.run(p)
    od = r.orders
    last = od[(od.ticker == "INTC")].iloc[-1]
    assert str(last.date) == "2015-06-30" and last.reason == "delisted" and last.side == "sell"
    assert abs(last.price - data.load("INTC")["close"].iloc[-1]) < 1e-9
    assert (r.holdings.loc["2015-06-30":, "INTC"] == 0).all()
    assert not any("before its history starts" in n for n in p.notes)
    assert any(n.startswith("Delisted: INTC delisted/acquired on 2015-06-30") for n in p.notes)
    assert not (od[pd.to_datetime(od.date) > "2015-06-30"].ticker == "INTC").any()


@needs("INTC", "SPY")
def test_signal_engine_closes_a_delisted_position(intc_delisted):
    s = Strategy(universe=["INTC", "SPY"], entry="close > 0", exit_when="close < 0", start="2014-01-01",
                 position_size=0.5, max_positions=2)
    r = engine.run(s)
    t = r.trades[r.trades.ticker == "INTC"]
    assert len(t) == 1 and str(t.iloc[0].exit_date) == "2015-06-30" and t.iloc[0].exit_reason == "delisted"
    assert any("INTC delisted/acquired on 2015-06-30" in n for n in s.notes)
    assert not any(n.startswith("Coverage:") for n in s.notes)
    assert abs(r.equity.iloc[-1] - (s.capital + r.trades.pnl.sum() + r.interest)) < 1e-6


# ------------------------------------------------------------------ Nasdaq-100 entries at the open
@pytest.mark.skipif(data.membership() is None or "SPY" not in AVAIL, reason="membership data not downloaded")
def test_ndx_open_entries_run_and_skip_never_member_junk():
    ever = set(data.nasdaq100_ever())
    assert "SSCC" not in ever
    mem = data.membership()
    for t in ever:
        if t in mem.columns and t not in set(data.nasdaq100()):
            m, _ = data.member_mask([t], data.load(t).index)
            assert m.any(), t
    text = "buy Nasdaq 100 stocks at the open when they gap down 3%, sell at the close, max 5 positions, since 2025"
    s = parser.parse(text)
    r = engine.run(s)
    assert len(r.trades) > 0


# ------------------------------------------------------------------ market cap phrases
@pytest.mark.skipif(data.membership() is None, reason="membership data not downloaded")
@pytest.mark.parametrize("text", [
    "hold the top 5 Nasdaq 100 stocks by market cap, market cap weighted, rebalance monthly, since 2021",
    "hold the top 5 Nasdaq 100 stocks by market-cap, market-cap weighted, rebalance monthly, since 2021",
    "hold the largest 5 Nasdaq 100 stocks by market cap, market cap weighted, rebalance monthly, since 2021",
    "hold the 5 largest Nasdaq 100 stocks, weighted by market cap, rebalance monthly, since 2021",
])
def test_market_cap_phrases(text):
    p = parser.parse(text)
    f = p.tree["filter"]
    assert f["select"] == "top" and f["n"] == 5 and f["by"] == "market_cap" and f["weights"] == "market_cap"
    assert p.tree.get("universe") == "NDX"


def test_market_cap_value_phrase_is_not_read_as_a_moving_average():
    assert parser.value_phrase("market cap")[0] == "market_cap"
    assert parser.value_phrase("market capitalization")[0] == "market_cap"


# ------------------------------------------------------------------ spin-off distributions
@needs("MDLZ")
def test_spinoff_distribution_is_labelled_not_called_a_dividend():
    assert pd.Timestamp("2012-10-02") in data.spinoff_days("MDLZ")
    assert pd.Timestamp("2012-09-17") not in data.spinoff_days("MDLZ")      # an ordinary dividend
    p = parser.parse("hold MDLZ from 2012-06-01 to 2013-06-01")
    r = portfolio.run(p)
    assert any("MDLZ 2012-10-02" in n and "spin-off" in n for n in p.notes)
    # the value is kept: the holding's income includes the distribution
    div = data.load("MDLZ")["dividend"]
    assert r.trades["income"].sum() > div.loc["2012-10-02"] * 0.99 * r.orders.iloc[0].shares
    s = Strategy(universe=["MDLZ"], entry="close > 0", exit_when="close < 0", start="2012-06-01", end="2013-06-01")
    engine.run(s)
    assert any("MDLZ 2012-10-02" in n and "spin-off" in n for n in s.notes)
