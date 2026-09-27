"""Round 9 (TradingView review): signal-strategy phrases, Stochastic RSI, date validation, and the Pine importer."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr, parser, pine_import
from backtester.parser import ParseError

AVAIL = set(data.available_tickers())
needs_data = pytest.mark.skipif(not {"SPY", "QQQ", "NVDA", "^VIX"} <= AVAIL, reason="price data not downloaded")
PINE = Path(__file__).parent / "fixtures" / "pine"


def rand_bars(n=300, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0, 0.006, n))
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    idx = pd.bdate_range("2019-12-30", periods=n)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": 1e6, "adj_close": c, "dividend": 0.0},
                        index=idx)


# ------------------------------------------------------------ 5. phrases

@needs_data
@pytest.mark.parametrize("text,entry,exit_", [
    ("buy SPY when close crosses above the 50 day MA, sell when it crosses below it",
     "(crossover(close, sma(close, 50)))", "close < sma(close, 50)"),
    ("buy SPY when RSI(2) is below 10, sell when VIX crosses above 30", None, '(crossover(sym("^VIX").close, 30))'),
    ("buy SPY when VIX crosses below 20, sell when VIX crosses above 30", '(crossunder(sym("^VIX").close, 20))', None),
    ("buy SPY when RSI(2) is below 10, sell when it crosses above the 20 day mean", None, "(crossover(close, sma(close, 20)))"),
    ("buy SPY when %K is below 20, hold 3 days", "(stoch_k(14, 1) < 20)", None),
    ("buy SPY when %D crosses above 20, sell when %K is above 80", "(crossover(stoch_d(14, 1, 3), 20))", "(stoch_k(14, 1) > 80)"),
    ("buy SPY when supertrend direction flips to up, sell when it flips to down",
     "(supertrend_dir(10, 3) < 0 and ref(supertrend_dir(10, 3), 1) > 0)",
     "(supertrend_dir(10, 3) > 0 and ref(supertrend_dir(10, 3), 1) < 0)"),
    ("buy SPY when supertrend flips to up, sell when it flips to down", None, "(crossunder(close, supertrend(10, 3)))"),
    ("buy SPY when Heikin Ashi turns green, sell when Heikin Ashi turns red",
     "(ha_close > ha_open and ref(ha_close, 1) <= ref(ha_open, 1))",
     "(ha_close < ha_open and ref(ha_close, 1) >= ref(ha_open, 1))"),
    ("buy SPY when Heikin Ashi turns green, sell when it turns red", None,
     "(ha_close < ha_open and ref(ha_close, 1) >= ref(ha_open, 1))"),
    ("buy SPY when stochastic RSI is below 20, hold 3 days", "(stoch_rsi_k(3, 14, 14) < 20)", None),
    ("buy SPY when stochastic RSI below 20, hold 3 days", "(stoch_rsi_k(3, 14, 14) < 20)", None),
    ("buy SPY when stoch RSI %K crosses above %D, sell when stoch RSI is above 80",
     "(crossover(stoch_rsi_k(3, 14, 14), stoch_rsi_d(3, 3, 14, 14)))", "(stoch_rsi_k(3, 14, 14) > 80)"),
    ("buy SPY when `ta.stoch(ta.rsi(close, 14), ta.rsi(close, 14), ta.rsi(close, 14), 14) < 20`, hold 3 days",
     "(stoch(rsi(close, 14), rsi(close, 14), rsi(close, 14), 14) < 20)", None),
    ('buy SPY when request.security(syminfo.tickerid, "W", close) > request.security(syminfo.tickerid, "W", '
     'ta.sma(close, 10)), hold 3 days', "(weekly(close) > weekly(sma(close, 10)))", None),
    ('buy SPY when close > request.security(syminfo.tickerid, "D", ta.sma(close, 200)), hold 3 days',
     "(close > sma(close, 200))", None),
])
def test_round9_phrases(text, entry, exit_):
    s = parser.parse(text)
    if entry:
        assert s.entry == entry
    if exit_:
        assert s.exit_when == exit_


@needs_data
def test_cost_and_stop_word_order():
    assert parser.parse("buy SPY when RSI(2) is below 10, hold 3 days, commission 0.1%").commission_pct == pytest.approx(0.001)
    assert parser.parse("buy SPY when RSI(2) is below 10, hold 3 days, 0.1% commission").commission_pct == pytest.approx(0.001)
    assert parser.parse("buy SPY when RSI(2) is below 10, stop loss at 2 ATR below entry, hold 5 days").stop_atr == 2
    assert parser.parse("buy SPY when RSI(2) is below 10, stop loss at 2 ATR below the entry price, hold 5 days").stop_atr == 2


@needs_data
def test_it_flips_without_a_signal_to_reverse_is_refused():
    with pytest.raises(ParseError, match="what flips"):
        parser.parse("buy SPY when RSI(2) is below 10, sell when it flips to bearish")


def test_ta_stoch_of_other_sources():
    assert expr.pine_to_rule("ta.stoch(open, high, low, 14) < 20") == "stoch(open, high, low, 14) < 20"
    assert expr.pine_to_rule("ta.stoch(close, high, low, 14) < 20") == "stoch_k(14, 1) < 20"


def test_stoch_rsi_matches_tradingview_definition():
    df = rand_bars(300, seed=5)
    ns = expr.Namespace(df)
    r = expr.rsi_wilder(df["close"], 14)
    raw = 100 * (r - r.rolling(14).min()) / (r.rolling(14).max() - r.rolling(14).min())
    k = raw.rolling(3).mean()
    got = expr.evaluate_value("stoch_rsi_k()", ns)
    assert np.allclose(got.dropna(), k.dropna())
    assert np.allclose(expr.evaluate_value("stoch_rsi_d()", ns).dropna(), k.rolling(3).mean().dropna())
    assert not expr.open_safe("stoch_rsi_k() < 20") and expr.open_safe("ref(stoch_rsi_k(), 1) < 20")


# ------------------------------------------------------------ 6. dates validated up front

@needs_data
def test_bad_dates_are_refused_clearly():
    with pytest.raises(ParseError, match="start date 'garbage' is not a date"):
        parser.parse("buy SPY when RSI(2) is below 10, hold 3 days", start="garbage")
    with pytest.raises(ParseError, match="end date '2016-13-45' is not a date"):
        parser.parse("buy SPY when RSI(2) is below 10, hold 3 days", end="2016-13-45")
    from backtester import web
    with pytest.raises(web.ClientError, match="start date 'garbage' is not a date"):
        web._options({"options": {"start": "garbage"}})
    assert web._options({"options": {"start": "2015-01-02"}})["start"] == "2015-01-02"


@needs_data
def test_cli_bad_date(capsys):
    from backtester.__main__ import main
    rc = main(["buy SPY when RSI(2) is below 10, hold 3 days", "--start", "garbage", "--dry-run"])
    out = capsys.readouterr()
    assert rc != 0 and "not a date" in (out.out + out.err)


# ------------------------------------------------------------ 7. Pine importer

def test_pine_detection():
    assert pine_import.looks_like_pine("//@version=5\nstrategy('x')")
    assert pine_import.looks_like_pine("strategy('x')\nif close > open\n    strategy.entry('L', strategy.long)")
    assert not pine_import.looks_like_pine("buy SPY when RSI(2) is below 10, hold 3 days")
    assert len(glob.glob(str(PINE / "*.pine"))) >= 5


def pine(name, **kw):
    return parser.parse((PINE / name).read_text(), **kw)


@needs_data
def test_pine_needs_a_ticker():
    with pytest.raises(ParseError, match="Which ticker"):
        pine("ma_cross.pine")
    s = pine("ma_cross.pine", ticker="QQQ")
    assert s.universe == ["QQQ"]


@needs_data
def test_pine_ma_cross():
    s = pine("ma_cross.pine", ticker="SPY")
    assert s.entry == "crossover(sma(close, 10), sma(close, 30))"
    assert s.exit_when == "crossunder(sma(close, 10), sma(close, 30))"
    assert (s.entry_fill, s.exit_when_fill, s.tv_compat) == ("next_open", "next_open", True)
    assert s.capital == 10_000 and s.position_size == 1.0 and s.commission_pct == pytest.approx(0.0005)
    assert s.fractional_shares is False and s.cash_rate is None and s.dividends is False
    r = engine.run(s)
    assert len(r.trades) > 20
    assert r.equity.iloc[-1] == pytest.approx(s.capital + r.trades.pnl.sum() + r.interest, abs=1e-6)


@needs_data
def test_pine_rsi_mean_reversion_with_stop_and_limit():
    s = pine("rsi_mean_reversion.pine")
    assert s.universe == ["SPY"]
    assert s.entry == "rsi(close, 2) < 10 and close > sma(close, 200)"
    assert s.exit_when == "crossover(rsi(close, 2), 70)"
    assert s.stop_loss == pytest.approx(0.05) and s.take_profit == pytest.approx(0.06)
    assert s.entry_fill == "close" and s.exit_when_fill == "close" and s.position_size == pytest.approx(0.95)
    assert s.capital == 100_000


@needs_data
def test_pine_breakout_with_atr_trailing():
    s = pine("atr_breakout.pine", ticker="NVDA")
    assert s.entry == "close > ref(highest(high, 20), 1)"
    assert s.trailing_atr == 3 and s.stop_atr == 2 and s.atr_period == 14
    assert s.exit_when is None
    engine.run(s)


@needs_data
def test_pine_long_short_reversal():
    s = pine("macd_reversal.pine")
    assert s.universe == ["QQQ"] and s.side == "both" and s.reverse
    assert s.entry == "crossover(macd(12, 26), macd_signal(12, 26, 9))"
    assert "crossunder(macd(12, 26), macd_signal(12, 26, 9))" in s.short_entry
    assert s.commission == 1
    r = engine.run(s)
    assert {"long", "short"} <= set(r.trades.side)


@needs_data
def test_pine_higher_timeframe_filter():
    s = pine("htf_filter.pine", ticker="SPY")
    assert s.entry == "close > weekly(ema(close, 20)) and crossover(rsi(close, 14), 50)"
    assert s.exit_when == "bars_held >= 10 or close < weekly(ema(close, 20))"
    assert s.start == "2015-01-01" and s.sizing == "fixed_dollars" and s.fixed_amount == 20_000
    engine.run(s)


@needs_data
def test_pine_partial_exit():
    s = pine("scale_out.pine")
    assert s.universe == ["NVDA"] and s.scale_out == [{"at": pytest.approx(0.03), "fraction": pytest.approx(0.5),
                                                          "after_fill": True}]   # from position_avg_price (round 11)
    assert s.take_profit == pytest.approx(0.06) and s.stop_loss == pytest.approx(0.05) and s.start == "2012-01-01"


H = '//@version=5\nstrategy("x", overlay=true)\n'


@pytest.mark.parametrize("body,line,msg", [
    # (var state, one-line functions, ternaries, limit entries, dynamic stops and ticks are supported since round 10)
    ("varip float lvl = na\nif close > open\n    strategy.entry(\"L\", strategy.long)\n", 3, "varip"),
    ("x = 0.0\nx := close\nif close > x\n    strategy.entry(\"L\", strategy.long)\n", 4, "reassignment"),
    ("f(x) =>\n    y = x * 2\n    y + 1\nif close > open\n    strategy.entry(\"L\", strategy.long)\n", 3, "several lines"),
    ("if close > open\n    strategy.order(\"L\", strategy.long)\n", 4, "strategy.order"),
    ("if close > open\n    strategy.entry(\"L\", strategy.long, limit=close * 0.98, stop=close * 1.02)\n", 4, "stop-limit"),
    ("var float x = 0\nif strategy.position_size > 0\n    x := close\nif close > open\n    strategy.entry(\"L\", strategy.long)\n",
     5, "depends on the position"),
    ("for i = 0 to 10\n    x = i\nif close > open\n    strategy.entry(\"L\", strategy.long)\n", 3, "for"),
])
def test_pine_refuses_unsupported_with_line_number(body, line, msg):
    with pytest.raises(ParseError, match=rf"line {line}:.*{msg}"):
        parser.parse(H + body, ticker="SPY")


def test_pine_refuses_indicators_and_scripts_without_entries():
    with pytest.raises(ParseError, match="indicator, not a strategy"):
        parser.parse('//@version=5\nindicator("x")\nplot(close)\n', ticker="SPY")
    with pytest.raises(ParseError, match="no strategy.entry"):
        parser.parse(H + "plot(close)\n", ticker="SPY")


@needs_data
def test_pine_on_the_site_uses_the_ticker_option():
    from backtester import web
    spec = web._spec({"text": (PINE / "ma_cross.pine").read_text(), "options": {"ticker": "QQQ"}})
    assert spec.universe == ["QQQ"] and spec.tv_compat
    with pytest.raises(ParseError, match="Which ticker"):
        web._spec({"text": (PINE / "ma_cross.pine").read_text()})
