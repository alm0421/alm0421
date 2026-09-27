"""Round 13 (a TradingView expert's review): an exit clause with no comma before it is its own clause (never merged
into the entry as AND), 'A then B' and 'sell ..., buy back ...' are refused with a question, candle colours are
close vs open, 'the RSI(14) is oversold' keeps its period, Pine's slippage in ticks / dayofweek /
strategy.risk.allow_entry_in, the TradingView-mode warning when many entries are skipped for cash, and new phrases
('on a 20 day breakout', 'the 3 day average of the RSI(2)', the linear regression slope, 'ichimoku tenkan',
'the histogram turns negative', 'crosses above the mean', 'otherwise sell', 'touches the lower keltner channel')."""
import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr, parser, pine_import
from backtester.parser import ParseError
from backtester.strategy import Strategy

AVAIL = set(data.available_tickers())
needs_data = pytest.mark.skipif(not {"SPY", "AAPL"} <= AVAIL, reason="price data not downloaded")


def bars(rows, start="2019-12-30", volume=1e6):
    idx = pd.bdate_range(start, periods=len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = volume
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    return df


@pytest.fixture
def fake(monkeypatch):
    frames = {}
    real = data.load

    def load(t):
        if t in frames:
            return frames[t]
        return real(t)
    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {t: load(t) for t in ts})
    return frames


def books(r):
    s = r.strategy
    pnl = r.trades["pnl"].sum() if len(r.trades) else 0.0
    assert r.equity.iloc[-1] == pytest.approx(s.capital + pnl + r.interest, abs=1e-6)


# ------------------------------------------------------------------ 1. clause boundaries

@needs_data
@pytest.mark.parametrize("verb", ["sell", "exit", "close the position", "take profit", "sell at the close"])
def test_exit_clause_without_comma_is_the_exit_not_part_of_the_entry(verb):
    s = parser.parse(f"buy AAPL when RSI(2) is below 50 {verb} when RSI(2) is above 80")
    assert s.entry == "(rsi(close, 2) < 50)"
    assert s.exit_when == "(rsi(close, 2) > 80)"


@needs_data
def test_exit_clause_without_comma_with_a_holding_period():
    s = parser.parse("buy AAPL when RSI(2) is below 50 sell when RSI(2) is above 10, hold 3 days")
    assert s.entry == "(rsi(close, 2) < 50)" and s.exit_when == "(rsi(close, 2) > 10)" and s.hold_bars == 3


@needs_data
def test_cover_and_short_clauses_without_comma():
    s = parser.parse("short AAPL when RSI(2) is above 90 cover when RSI(2) is below 50")
    assert s.side == "short" and s.entry == "(rsi(close, 2) > 90)" and s.exit_when == "(rsi(close, 2) < 50)"
    s = parser.parse("buy AAPL when RSI(2) is below 20 short when RSI(2) is above 80")
    assert s.side == "both" and s.entry == "(rsi(close, 2) < 20)" and s.short_entry == "(rsi(close, 2) > 80)"
    for text in ("buy SPY when RSI(2) is below 10, go short when RSI(2) is above 90",
                 "buy SPY when RSI(2) is below 10 go short when RSI(2) is above 90",
                 "go long SPY when RSI(2) is below 10 sell short when RSI(2) is above 90"):
        s = parser.parse(text)
        assert s.side == "both" and s.entry == "(rsi(close, 2) < 10)" and s.short_entry == "(rsi(close, 2) > 90)", text


@needs_data
@pytest.mark.parametrize("text", ["buy AAPL when RSI(2) is below 50 then RSI(2) is above 10",
                                  "buy AAPL when RSI(2) is below 50 then RSI(2) is above 10, hold 3 days",
                                  "buy AAPL when RSI(2) is below 50 and then RSI(2) is above 10, hold 3 days"])
def test_then_between_conditions_asks_sequence_or_both(text):
    with pytest.raises(ParseError, match="sequence .* or both on the same day"):
        parser.parse(text)


@needs_data
def test_then_before_an_action_is_still_a_new_clause():
    s = parser.parse("buy SPY when RSI(2) is below 10 then sell at the close")
    assert s.entry == "(rsi(close, 2) < 10)" and s.hold_bars == 1
    s = parser.parse("buy SPY when RSI(2) is below 10 and then sell when RSI(2) is above 70")
    assert s.exit_when == "(rsi(close, 2) > 70)"
    s = parser.parse("buy SPY when RSI(2) is below 10, hold 3 days then sell at the open")
    assert s.hold_bars == 3


@needs_data
@pytest.mark.parametrize("word", ["until", "unless", "before", "followed by", "after that"])
def test_other_connectives_are_refused_not_read_as_and(word):
    with pytest.raises(ParseError):
        parser.parse(f"buy AAPL when RSI(2) is below 50 {word} RSI(2) is above 80, hold 3 days")


@needs_data
def test_sell_then_buy_back_asks_short_or_held_position():
    with pytest.raises(ParseError) as e:
        parser.parse("sell AAPL when RSI(2) is above 90, buy back when RSI(2) is below 50")
    msg = str(e.value)
    assert "short AAPL when RSI(2) is above 90, cover when RSI(2) is below 50" in msg
    assert "buy AAPL when RSI(2) is below 50, sell when RSI(2) is above 90" in msg
    # the short reading, said explicitly, runs
    s = parser.parse("short AAPL when RSI(2) is above 90, cover when RSI(2) is below 50")
    assert s.side == "short" and s.entry == "(rsi(close, 2) > 90)" and s.exit_when == "(rsi(close, 2) < 50)"


# ------------------------------------------------------------------ 2. candle colours

@needs_data
@pytest.mark.parametrize("phrase,rule", [
    ("it closes red", "close < open"), ("it closes green", "close > open"), ("it closed red", "close < open"),
    ("there is a red candle", "close < open"), ("it is a green candle", "close > open"),
    ("a bearish candle", "close < open"), ("a bullish candle", "close > open"),
    ("it closes red 3 days in a row", "count(close < open, 3) == 3"),
    ("3 green candles in a row", "count(close > open, 3) == 3")])
def test_candle_colour_is_close_against_open(phrase, rule):
    s = parser.parse(f"buy AAPL when {phrase}, hold 1 day")
    assert s.entry == f"({rule})"
    assert any("TradingView colours it" in n and "closes down" in n or "closes up" in n for n in s.notes)


@needs_data
def test_candle_colour_of_another_ticker_and_the_previous_close_readings_stay():
    s = parser.parse("buy AAPL when SPY closes red, hold 1 day")
    assert s.entry == '(sym("SPY").close < sym("SPY").open)'
    assert parser.parse("buy AAPL when it closes down, hold 1 day").entry == "(change < 0)"
    assert parser.parse("buy AAPL when it closes down 3 days in a row, hold 1 day").entry == "(down_days >= 3)"
    # a bullish engulfing candle is still the two-candle pattern
    assert "ref(close, 1) < ref(open, 1)" in parser.parse("buy AAPL after a bullish engulfing candle, hold 2 days").entry


def test_red_candle_is_true_on_a_down_candle_that_closed_up(fake):
    # gaps up then closes below its open, still above the previous close: red candle, not a down day
    fake["CND"] = bars([[100, 101, 99, 100], [103, 104, 101, 102], [101, 103, 100, 102.5]])
    ns = expr.Namespace(fake["CND"], ticker="CND")
    red = expr.evaluate("close < open", ns)
    assert list(red) == [False, True, False]


# ------------------------------------------------------------------ 3. oversold / overbought with an RSI named

@needs_data
def test_rsi_oversold_keeps_the_named_period_and_the_sentence_runs():
    s = parser.parse("buy SPY when the RSI(14) is oversold and the price is above the 200 EMA, target 2R with stop at "
                     "the 10 day low")
    assert s.entry == "(rsi(close, 14) < 30) and (close > ema(close, 200))"
    assert s.target_r == 2.0 and s.stop_level == "lowest(low, 10)"
    assert any(n.startswith("Warning: 'the RSI(14) is oversold'") for n in s.notes)
    assert parser.parse("buy SPY when RSI(2) is oversold, hold 3 days").entry == "(rsi(close, 2) < 30)"
    assert parser.parse("buy SPY when the 10 day RSI is overbought, hold 3 days").entry == "(rsi(close, 10) > 70)"
    assert parser.parse("buy SPY when it is oversold, hold 3 days").entry == "(rsi(close, 14) < 30)"


# ------------------------------------------------------------------ 4. Pine: slippage ticks, dayofweek, allow_entry_in

H = '//@version=5\nstrategy("x", overlay=true, initial_capital=100000)\n'


def test_pine_slippage_ticks_convert_at_one_cent(fake):
    fake["PSL"] = bars([[100, 101, 99, 100]] * 3 + [[100, 101, 99, 101]] + [[102, 103, 101, 102]] * 3
                       + [[102, 103, 95, 96]] + [[96, 97, 95, 96]] * 2)
    src = ('//@version=5\nstrategy("x", overlay=true, initial_capital=100000, slippage=3)\n'
           'if close > close[1] and close[1] <= close[2]\n    strategy.entry("L", strategy.long)\n'
           'strategy.exit("x", "L", loss=300)\n')
    s = pine_import.translate(src, "PSL")
    assert s.slippage_price == pytest.approx(0.03)
    assert any("slippage=3" in n and "$0.03 per share" in n and "not to limit fills" in n for n in s.notes)
    r = engine.run(s)
    t = r.trades.iloc[0]
    assert t.entry_price == pytest.approx(102 + 0.03)                  # the next open, 3 ticks worse
    assert t.exit_reason == "stop loss" or t.exit_reason == "stop level"
    assert t.exit_price == pytest.approx(102.03 - 3.0 - 0.03)          # the stop level, 3 ticks worse
    books(r)


def test_slippage_price_is_not_applied_to_limit_fills(fake):
    fake["PLM"] = bars([[100, 101, 99, 100]] * 3 + [[100, 101, 99, 101]] + [[101, 112, 100, 111]] * 3)
    s = Strategy(universe=["PLM"], entry="close > ref(close, 1)", entry_fill="next_open", take_profit=0.05,
                 slippage_price=0.05, cash_rate=None, fractional_shares=True, start="2019-12-30")
    r = engine.run(s)
    t = r.trades.iloc[0]
    assert t.entry_price == pytest.approx(101 + 0.05)                  # a market order: slipped
    assert t.exit_reason == "take profit" and t.exit_price == pytest.approx(101.05 * 1.05)   # a limit: not slipped
    books(r)
    s2 = Strategy(universe=["PLM"], entry="close > ref(close, 1)", entry_order="limit", entry_level="close * 0.999",
                  hold_bars=1, slippage_price=0.05, cash_rate=None, fractional_shares=True, start="2019-12-30")
    t2 = engine.run(s2).trades.iloc[0]
    assert t2.entry_fill in ("limit", "open") and t2.entry_price == pytest.approx(min(101 * 0.999, 101.0))


def test_pine_slippage_refused_for_crypto():
    with pytest.raises(pine_import.PineImportError, match="tick size of BTC-USD"):
        pine_import.translate('//@version=5\nstrategy("x", slippage=2)\nif close > open\n    strategy.entry("L", strategy.long)\n'
                              'if close < open\n    strategy.close("L")\n', "BTC-USD")


def test_pine_dayofweek_uses_tradingview_numbering(fake):
    src = H + ('if dayofweek == dayofweek.monday\n    strategy.entry("L", strategy.long)\n'
               'if dayofweek(time) == dayofweek.friday\n    strategy.close("L")\n')
    s = pine_import.translate(src, "SPY")
    assert s.entry == "dow + 2 == 2" and s.exit_when == "dow + 2 == 6"
    assert any("Sunday = 1" in n and "dow + 2" in n for n in s.notes)
    fake["DOW"] = bars([[100, 101, 99, 100]] * 10, start="2020-01-06")      # a Monday
    ns = expr.Namespace(fake["DOW"], ticker="DOW")
    mon = expr.evaluate("dow + 2 == 2", ns)
    assert [d.day_name() for d in mon[mon].index] == ["Monday", "Monday"]
    s = pine_import.translate(H + 'if dayofweek >= 5 and close > open\n    strategy.entry("L", strategy.long)\n'
                                  'if dayofweek == 2\n    strategy.close("L")\n', "SPY")
    assert s.entry == "dow + 2 >= 5 and close > open"


def test_pine_allow_entry_in_long_only_turns_short_entries_into_exits():
    body = ('if close > open\n    strategy.entry("L", strategy.long)\nif close < open\n    strategy.entry("S", strategy.short)\n')
    s = pine_import.translate(H + 'strategy.risk.allow_entry_in(strategy.direction.long)\n' + body, "SPY")
    assert s.side == "long" and s.entry == "close > open" and s.exit_when == "close < open" and not s.short_entry
    assert any("allow_entry_in(strategy.direction.long)" in n and "only closes" in n for n in s.notes)
    s = pine_import.translate(H + 'strategy.risk.allow_entry_in(strategy.direction.short)\n' + body, "SPY")
    assert s.side == "short" and s.entry == "close < open" and s.exit_when == "close > open"
    s = pine_import.translate(H + 'strategy.risk.allow_entry_in(strategy.direction.all)\n' + body, "SPY")
    assert s.side == "both"


@pytest.mark.parametrize("call", ["strategy.risk.max_drawdown(10, strategy.percent_of_equity)",
                                  "strategy.risk.max_intraday_loss(5, strategy.percent_of_equity)",
                                  "strategy.risk.max_cons_loss_days(3)"])
def test_pine_other_risk_rules_are_refused_with_an_explanation(call):
    with pytest.raises(pine_import.PineImportError, match="no such kill switch"):
        pine_import.translate(H + call + '\nif close > open\n    strategy.entry("L", strategy.long)\n'
                                         'if close < open\n    strategy.close("L")\n', "SPY")


# ------------------------------------------------------------------ 8. TradingView mode: many entries skipped for cash

def test_tv_mode_warns_when_many_entries_are_skipped(fake):
    rows = []
    for _ in range(6):   # a signal (volume) and a gap up at the next open, then flat
        rows += [[100, 101, 99, 100], [105, 106, 104, 105], [105, 106, 104, 105]]
    df = bars(rows)
    df.loc[df.index[::3], "volume"] = 2e6
    fake["TVW"] = df
    s = Strategy(universe=["TVW"], entry="volume > 1.5e6", entry_fill="next_open", hold_bars=1, tv_compat=True,
                 capital=10_000, cash_rate=None, start="2019-12-30")
    engine.run(s)
    w = [n for n in s.notes if n.startswith("Warning: TradingView skipped")]
    assert w and "of 6 entry orders" in w[0] and "95% of equity" in w[0]
    s = Strategy(universe=["TVW"], entry="volume > 1.5e6", entry_fill="next_open", hold_bars=1, tv_compat=True,
                 capital=10_000, position_size=0.95, cash_rate=None, start="2019-12-30")
    engine.run(s)
    assert not any(n.startswith("Warning: TradingView skipped") for n in s.notes)


# ------------------------------------------------------------------ 9. vocabulary

@needs_data
@pytest.mark.parametrize("text,entry", [
    ("buy SPY on a 20 day breakout, hold 5 days", "(close > ref(highest(high, 20), 1))"),
    ("buy SPY after a 55-day breakdown, hold 5 days", "(close < ref(lowest(low, 55), 1))"),
    ("buy SPY when the 3 day average of the RSI(2) is below 20, hold 3 days", "(sma(rsi(close, 2), 3) < 20)"),
    ("buy SPY when the 3 day moving average of the 2 day RSI is below 20, hold 3 days", "(sma(rsi(close, 2), 3) < 20)"),
    ("buy SPY when the 20 day linear regression slope is positive, hold 5 days", "(linreg(close, 20, 0) > linreg(close, 20, 1))"),
    ("buy SPY when the 50 day linear regression slope is negative, hold 5 days", "(linreg(close, 50, 0) < linreg(close, 50, 1))"),
    ("buy SPY when ichimoku tenkan crosses above kijun, hold 5 days", "(crossover(tenkan(), kijun()))"),
    ("buy SPY when the price touches the lower keltner channel, hold 3 days", "(low <= keltner_lower(20, 2))"),
    ("buy SPY when it touches the upper bollinger band, hold 3 days", "(high >= bb_upper(20, 2))"),
])
def test_new_phrases(text, entry):
    s = parser.parse(text)
    assert s.entry == entry


def test_linreg_slope_is_the_regression_slope(fake):
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, 1, 80))
    fake["LRS"] = bars(np.column_stack([c, c + 1, c - 1, c]))
    ns = expr.Namespace(fake["LRS"], ticker="LRS")
    d = expr.evaluate_value("linreg(close, 20, 0) - linreg(close, 20, 1)", ns)
    slope = pd.Series(c).rolling(20).apply(lambda w: np.polyfit(np.arange(20), w, 1)[0], raw=True).to_numpy()
    assert np.allclose(d.to_numpy()[19:], slope[19:])


@needs_data
def test_histogram_exit_refers_to_the_entry_histogram():
    s = parser.parse("buy SPY when the MACD histogram turns positive, exit when the histogram turns negative")
    assert s.entry == "(crossover(macd_hist(), 0))" and s.exit_when == "(crossunder(macd_hist(), 0))"
    with pytest.raises(ParseError, match="which histogram"):
        parser.parse("buy SPY when RSI(2) is below 10, exit when the histogram turns negative")


@needs_data
def test_crosses_above_the_mean_is_the_entry_mean_or_refused():
    s = parser.parse("buy SPY when it closes below the lower bollinger band, sell when it crosses above the mean")
    assert s.exit_when == "(crossover(close, sma(close, 20)))"
    assert any(n.startswith("Warning: 'the mean' was read as the entry's mean, sma(close, 20)") for n in s.notes)
    s = parser.parse("buy SPY when it closes 2 standard deviations below its 50 day mean, sell when it closes above the mean")
    assert s.exit_when == "(close > sma(close, 50))"
    with pytest.raises(ParseError, match="which mean"):
        parser.parse("buy SPY when it closes below the lower bollinger band and it is above its 200 day moving average, "
                     "sell when it crosses above the mean")
    with pytest.raises(ParseError, match="which mean"):
        parser.parse("buy SPY when RSI(2) is below 10, sell when it crosses above the mean")


@needs_data
def test_otherwise_sell_exits_when_the_entry_rule_is_false():
    s = parser.parse("buy SPY when RSI(2) is below 10, otherwise sell")
    assert s.entry == "(rsi(close, 2) < 10)" and s.exit_when == "not ((rsi(close, 2) < 10))"
    assert any("'otherwise sell' was read as" in n for n in s.notes)
    with pytest.raises(ParseError, match="otherwise what"):
        parser.parse("buy SPY when RSI(2) is below 10, otherwise short it")
