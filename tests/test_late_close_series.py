"""Series that close after the 4:00pm stock close (crypto at 8pm New York, Cboe's VIX at 4:15pm) are read as of their
previous bar by rules acted on at a US close; portfolio sentences accept 'rebalance never' and short-selling costs."""
from __future__ import annotations

import pandas as pd
import pytest

from backtester import data, engine, expr, parser, portfolio as pf
from backtester.strategy import Strategy

HAVE = all((data.PRICES / f"{t}.csv").exists() for t in ("SPY", "^VIX", "BTC-USD", "TLT"))
needs = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


@needs
def test_vix_close_lags_a_day_for_same_day_close_fills_only():
    spy = data.load("SPY")
    vix = data.load("^VIX")["close"]
    ns = expr.Namespace(spy, ticker="SPY", close_fill=True)
    got = ns["sym"]("^VIX").close
    want = vix.shift(1).reindex(spy.index.union(vix.index)).ffill().reindex(spy.index)
    assert (got.dropna() == want.reindex(got.dropna().index)).all()
    assert any("4:15pm" in n for n in ns.notes)
    ns2 = expr.Namespace(spy, ticker="SPY")                 # next-open fills: that day's value is known by then
    pd.testing.assert_series_equal(ns2["sym"]("^VIX").close, vix.reindex(spy.index), check_names=False)


@needs
def test_engine_lags_vix_at_the_close_not_at_the_next_open():
    rule = 'sym("^VIX").close > 30'
    at_close = Strategy(universe=["SPY"], entry=rule, hold_bars=1, start="2020-01-01", end="2020-12-31")
    r1 = engine.run(at_close)
    nxt = Strategy(universe=["SPY"], entry=rule, entry_fill="next_open", hold_bars=1, start="2020-01-01", end="2020-12-31")
    r2 = engine.run(nxt)
    vix = data.load("^VIX")["close"]
    first_close = pd.Timestamp(r1.trades["entry_date"].iloc[0])
    first_open_signal = vix.loc["2020-01-01":][vix.loc["2020-01-01":] > 30].index[0]
    assert first_close == data.load("SPY").index[data.load("SPY").index.get_loc(first_open_signal) + 1]
    assert pd.Timestamp(r2.trades["entry_date"].iloc[0]) > first_open_signal
    assert any("4:15pm" in n for n in at_close.notes)


@needs
def test_portfolio_rule_on_crypto_uses_the_previous_crypto_bar():
    tree = {"if": "close > sma(close, 50)", "on": "BTC-USD", "then": {"asset": "SPY"}, "else": {"asset": "TLT"}}
    p = pf.Portfolio(tree=tree, rebalance="daily", start="2021-01-01", end="2021-12-31")
    pf.run(p)
    assert any("BTC-USD" in n and "previous day's bar" in n for n in p.notes)


def test_rebalance_never_and_borrow_fee_phrases():
    p = parser.parse("hold 60% SPY and 40% TLT, rebalance never")
    assert p.rebalance == "none"
    p = parser.parse("hold 150% QQQ and -50% TLT, rebalance monthly, 1% borrow fee")
    assert p.borrow_fee == pytest.approx(0.01) and p.rebalance == "monthly"
    p = parser.parse("hold 150% QQQ and -50% TLT, rebalance monthly, borrow fee of 2%, margin rate 1%, "
                     "short rebate 0.5% below the t-bill rate")
    assert (p.borrow_fee, p.margin_rate, p.short_rebate_spread) == pytest.approx((0.02, 0.01, 0.005))
