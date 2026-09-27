"""Round 11 (a TradingView expert's review): trailing stops that start once the trade is up (never read as an entry
condition), Pine var state through the CLI preflight and the site's rule probe, Pine variables named like built-ins,
exits from strategy.position_avg_price that TradingView only places after the fill bar, the non-repainting
request.security idiom, --tickers over the script's '// ticker:' comment, scale-outs at ticks / points, and phrases."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr, parser, pine_import, web
from backtester import __main__ as cli
from backtester.parser import ParseError
from backtester.strategy import Strategy

AVAIL = set(data.available_tickers())
needs_data = pytest.mark.skipif(not {"SPY", "AAPL", "TSLA"} <= AVAIL, reason="price data not downloaded")


def bars(rows, start="2019-12-30", volume=1e6):
    idx = pd.bdate_range(start, periods=len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = volume
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    return df


def rand_bars(n=900, seed=0, start="2015-01-02", vol=0.02):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0, vol * 0.4, n))
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol * 0.5, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol * 0.5, n)))
    return bars(np.column_stack([o, hi, lo, c]), start=start)


def osc_bars(n=1500, seed=0):
    """Prices swinging around 100 (a scale-out's limit is reached again and again)."""
    rng = np.random.default_rng(seed)
    c = 100 + 8 * np.sin(np.arange(n) / 6.0) + rng.normal(0, 1.5, n)
    o = np.r_[100.0, c[:-1]] + rng.normal(0, 0.5, n)
    hi = np.maximum(o, c) + np.abs(rng.normal(0, 0.8, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.8, n))
    return bars(np.column_stack([o, hi, lo, c]), start="2015-01-02")


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


def pine(body, ticker="SYN"):
    return f'//@version=5\nstrategy("t", initial_capital=100000)\n// ticker: {ticker}\n' + body


# ------------------------------------------------------------ 1. a trailing stop that starts once the trade is up

@pytest.mark.parametrize("text,act", [
    ("buy SPY when RSI(2) < 10, 2% stop, 6% target, trailing stop 1% once up 3%", ("trail_activation", 0.03)),
    ("buy SPY when RSI(2) < 10, 10% trailing stop once it is up 5%", ("trail_activation", 0.05)),
    ("buy SPY when RSI(2) < 10, trailing stop of 1% once the trade is up 3%", ("trail_activation", 0.03)),
    ("buy SPY when RSI(2) < 10, 5% trailing stop after the position is up 2%", ("trail_activation", 0.02)),
    ("buy SPY when RSI(2) < 10, with a 4% trailing stop that activates once up 2%", ("trail_activation", 0.02)),
    ("buy SPY when RSI(2) < 10, 10% trailing stop, once it is up 5%", ("trail_activation", 0.05)),
    ("buy SPY when RSI(2) < 10, 3% trailing stop when in profit by 4%", ("trail_activation", 0.04)),
    ("buy SPY when RSI(2) < 10, 2% stop, 5% trailing stop, activate trail after 1R", ("trail_activation_r", 1.0)),
    ("buy SPY when RSI(2) < 10, 2% stop, 5% trailing stop, activate the trailing stop at 1.5R", ("trail_activation_r", 1.5)),
    ("buy SPY when RSI(2) < 10, 2% stop, 3 ATR trailing stop after 1R", ("trail_activation_r", 1.0)),
    ("buy SPY when RSI(2) < 10, 2% stop, 5% trailing stop, start trailing once up 4%", ("trail_activation", 0.04)),
    ("buy SPY when RSI(2) < 10, trailing stop 1% once up $5", ("trail_activation_points", 5.0)),
])
def test_trailing_stop_activation_is_an_exit_not_an_entry_condition(text, act):
    s = parser.parse(text)
    assert s.entry == "(rsi(close, 2) < 10)"
    assert getattr(s, act[0]) == pytest.approx(act[1])
    assert s.trailing_stop or s.trailing_atr
    assert any("Trail activation" in n for n in s.notes)
    assert "starts once the price is" in s.summary()


@pytest.mark.parametrize("text", [
    "buy SPY when RSI(2) < 10, 2% stop, 6% target once up 3%",
    "buy SPY when RSI(2) < 10, 5% stop when the trade is 3% in profit",
    "buy SPY when RSI(2) < 10, 2% stop loss, 8% take profit after it has gained 4%",
])
def test_exit_words_are_never_absorbed_into_the_entry(text):
    with pytest.raises(ParseError, match="belongs to the exit|in the exit"):
        parser.parse(text)


@pytest.mark.parametrize("text,entry", [
    ("buy SPY with a 2% stop if it closes up 3 days in a row", "(up_days >= 3)"),
    ("buy SPY with a 2% stop when RSI(2) < 10", "(rsi(close, 2) < 10)"),
])
def test_entry_after_a_stop_phrase_still_reads(text, entry):
    assert parser.parse(text).entry == entry


def test_trail_activation_r_arms_the_trailing_stop_only_after_1r(fake):
    fake["SYN"] = rand_bars(1500, seed=3)
    common = dict(universe=["SYN"], cash_rate=None, dividends=False, entry="rsi(close, 2) < 15", stop_loss=0.03)
    never = engine.run(Strategy(**common, trailing_stop=0.02, trail_activation_r=1000))
    plain = engine.run(Strategy(**common))
    books(never)
    pd.testing.assert_frame_equal(never.trades[["entry_date", "exit_date", "exit_price"]],
                                  plain.trades[["entry_date", "exit_date", "exit_price"]])
    r = engine.run(Strategy(**common, trailing_stop=0.02, trail_activation_r=1))
    books(r)
    tr = r.trades[r.trades.exit_reason == "trailing stop"]
    assert len(tr) > 5
    fake_ = fake["SYN"]
    for _, t in tr.iterrows():      # the best price since entry reached entry + 1R (R = 3% of the entry) first
        seg = fake_.loc[t.entry_date: t.exit_date]
        assert seg.high.max() >= t.entry_price * 1.03 - 1e-9


# ------------------------------------------------------------ 2. Pine var state through the preflight and the site

COUNTER = """var int dc = 0
if close < close[1]
    dc += 1
else
    dc := 0
if dc >= 3
    strategy.entry("L", strategy.long)
if dc == 0
    strategy.close("L")
"""
COUNTER_TERNARY = """var int dc = 0
dc := close < close[1] ? dc + 1 : 0
if dc >= 3
    strategy.entry("L", strategy.long)
if dc == 0
    strategy.close("L")
"""


@needs_data
@pytest.mark.parametrize("body", [COUNTER, COUNTER_TERNARY])
def test_pine_var_state_passes_the_preflight_dry_run_and_full_run(body, tmp_path):
    f = tmp_path / "t.pine"
    f.write_text(pine(body, "AAPL"))
    a = cli.build_run_parser().parse_args([str(f), "--dry-run"])
    spec = cli.make_spec(a.text, a, a.spec)
    assert spec.state_vars and "pv_dc" in spec.entry
    cli.preflight(spec)                       # used to fail: unknown name 'pv_dc'
    web._probe_rules(spec)
    r = engine.run(spec)
    books(r)
    assert len(r.trades) > 50
    out = web.api_parse({"text": pine(body, "AAPL")})
    assert "pv_dc >= 3" in out["interpretation"]


@needs_data
def test_cli_dry_run_of_a_var_script_exits_cleanly(tmp_path, capsys):
    f = tmp_path / "t.pine"
    f.write_text(pine(COUNTER_TERNARY, "AAPL"))
    assert cli.main([str(f), "--dry-run"]) == 0
    assert "pv_dc >= 3" in capsys.readouterr().out


# ------------------------------------------------------------ 3. Pine variables named like built-ins

MACD = """[macd, signal, hist] = ta.macd(close, 12, 26, 9)
if ta.crossover(macd, signal)
    strategy.entry("L", strategy.long)
if ta.crossunder(macd, signal)
    strategy.close("L")
"""
DMI = """[diplus, diminus, adx] = ta.dmi(14, 14)
if diplus > diminus and adx > 25
    strategy.entry("L", strategy.long)
if diplus < diminus
    strategy.close("L")
"""
SHADOW = """atr = ta.atr(14)
rsi = ta.rsi(close, 14)
sma = ta.sma(close, 50)
high = ta.highest(high, 20)
if rsi < 30 and close > sma and atr > 0
    strategy.entry("L", strategy.long)
if close > high[1]
    strategy.close("L")
"""


@pytest.mark.parametrize("body,entry,exit_", [
    (MACD, "crossover(macd(12, 26), macd_signal(12, 26, 9))", "crossunder(macd(12, 26), macd_signal(12, 26, 9))"),
    (DMI, "plus_di(14) > minus_di(14) and adx(14) > 25", "plus_di(14) < minus_di(14)"),
    (SHADOW, "rsi(close, 14) < 30 and close > sma(close, 50) and (atr(14) > 0)", "close > ref(highest(high, 20), 1)"),
])
def test_pine_variables_named_like_builtins(body, entry, exit_, fake):
    fake["SYN"] = rand_bars(700)
    s = pine_import.translate(pine(body))
    assert s.entry == entry and s.exit_when == exit_
    books(engine.run(s))


# ------------------------------------------------------------ 4. exits from strategy.position_avg_price

AVG_EXIT = """if ta.rsi(close, 2) < 10
    strategy.entry("L", strategy.long)
strategy.exit("x", "L", stop = strategy.position_avg_price * 0.97, limit = strategy.position_avg_price * 1.03)
"""


def test_avg_price_exits_are_placed_after_the_fill_bar(fake):
    fake["SYN"] = rand_bars(1500, seed=5, vol=0.03)
    s = pine_import.translate(pine(AVG_EXIT))
    assert s.exits_after_fill == ["stop", "target"] and s.stop_loss == pytest.approx(0.03)
    assert any("never on the fill bar" in n for n in s.notes)
    r = engine.run(s)
    books(r)
    t = r.trades
    assert len(t) > 20 and (t.bars_held >= 1).all()
    # the same levels live from the fill (the backtester's own default) exit on the fill bar at times
    s0 = pine_import.translate(pine(AVG_EXIT))
    s0.exits_after_fill = []
    t0 = engine.run(s0).trades
    assert (t0.bars_held == 0).sum() > 0
    # process_orders_on_close: the entry fills at the signal bar's close; the script sees the position at the next
    # bar's close, so the stop and target work from the bar after that
    s2 = pine_import.translate(pine(AVG_EXIT).replace("initial_capital=100000", "initial_capital=100000, process_orders_on_close=true"))
    t2 = engine.run(s2).trades
    assert len(t2) > 20 and (t2[t2.exit_reason.isin(["stop loss", "take profit"])].bars_held >= 2).all()


def test_tick_exits_stay_live_from_the_fill():
    s = pine_import.translate(pine("""if ta.rsi(close, 2) < 10
    strategy.entry("L", strategy.long)
strategy.exit("x", "L", loss = 300, profit = 500)
"""))
    assert s.exits_after_fill == []


def test_exits_after_fill_validation():
    with pytest.raises(ValueError, match="exits_after_fill"):
        Strategy(universe=["SPY"], entry="close > 0", stop_loss=0.05, exits_after_fill=["sometimes"]).validate()


# ------------------------------------------------------------ 5. the non-repainting request.security idiom

def test_security_lookahead_on_with_an_offset_is_the_previous_period():
    t = expr.pine_to_rule('request.security(syminfo.tickerid, "W", high[1], lookahead = barmerge.lookahead_on)')
    assert t == "ref(weekly(high), 1)"
    assert expr.pine_to_rule('request.security(syminfo.tickerid, "M", close[2], lookahead=barmerge.lookahead_on)') == \
        "ref(monthly(ref(close, 1)), 1)"
    assert expr.pine_to_rule('request.security(syminfo.tickerid, "D", close[1], barmerge.gaps_off, barmerge.lookahead_on)') == \
        "ref(close, 1)"
    with pytest.raises(ValueError, match="lookahead"):
        expr.pine_to_rule('request.security(syminfo.tickerid, "W", high, lookahead = barmerge.lookahead_on)')
    with pytest.raises(ValueError, match="lookahead"):
        expr.pine_to_rule('request.security(syminfo.tickerid, "W", high[1] + 1, lookahead = barmerge.lookahead_on)')


def test_security_idiom_values_and_no_lookahead():
    df = rand_bars(400)
    ns = expr.Namespace(df)
    v = expr.evaluate_value(expr.pine_to_rule(
        'request.security(syminfo.tickerid, "W", high[1], lookahead = barmerge.lookahead_on)'), ns)
    wk = df.index.to_period("W-FRI")
    whigh = df.high.groupby(wk).max()
    for i in range(10, len(df)):
        prev = wk[i] - 1
        if prev in whigh.index:
            assert v.iloc[i] == pytest.approx(whigh[prev]), df.index[i]
    for cut in (123, 250, 333):          # truncating the data never changes an earlier value
        vt = expr.evaluate_value("ref(weekly(high), 1)", expr.Namespace(df.iloc[:cut]))
        pd.testing.assert_series_equal(vt, v.iloc[:cut], check_names=False)


# ------------------------------------------------------------ 6. --tickers over the script's comment

@needs_data
def test_tickers_option_overrides_the_ticker_comment(tmp_path):
    f = tmp_path / "p1.pine"
    f.write_text(pine("""if ta.rsi(close, 2) < 10
    strategy.entry("L", strategy.long)
if ta.rsi(close, 2) > 70
    strategy.close("L")
""", "SPY"))
    a = cli.build_run_parser().parse_args([str(f), "--tickers", "AAPL", "--dry-run"])
    s = cli.make_spec(a.text, a, a.spec)
    assert s.universe == ["AAPL"]
    assert any("AAPL (as given with the run)" in n and "// ticker: SPY" in n for n in s.notes)
    a = cli.build_run_parser().parse_args([str(f), "--dry-run"])
    assert cli.make_spec(a.text, a, a.spec).universe == ["SPY"]
    out = web.api_parse({"text": f.read_text(), "options": {"ticker": "AAPL"}})
    assert out["spec"]["universe"] == ["AAPL"]


# ------------------------------------------------------------ 7. scale-outs at ticks / points

SCALE = """if ta.rsi(close, 2) < 10
    strategy.entry("L", strategy.long)
strategy.exit("tp", "L", qty_percent = 50, profit = 300)
strategy.exit("tp2", "L", qty_percent = 25, limit = strategy.position_avg_price + 6)
strategy.exit("rest", "L", loss = 500, profit = 1000)
"""


def test_scale_outs_at_ticks_and_points(fake):
    fake["SYN"] = osc_bars(1500, seed=7)
    s = pine_import.translate(pine(SCALE).replace('initial_capital=100000', 'initial_capital=100000, default_qty_value=10'))
    assert s.scale_out == [{"points": 3.0, "fraction": 0.5}, {"points": 6.0, "fraction": 0.5, "after_fill": True}]
    assert s.stop_covers_scale_outs is False
    r = engine.run(s)
    books(r)
    so = r.trades[r.trades.exit_reason == "scale out"]
    assert len(so) > 5
    for _, t in so.iterrows():
        gain = t.exit_price - t.entry_price
        assert gain == pytest.approx(3.0, abs=1e-6) or gain == pytest.approx(6.0, abs=1e-6) or gain > 3.0
    second = so[(so.exit_price - so.entry_price).round(6) == 6.0]
    assert len(second) and (second.bars_held >= 1).all()


def test_scale_outs_mixing_percent_and_points_are_refused():
    with pytest.raises(ParseError, match="mix"):
        pine_import.translate(pine("""if ta.rsi(close, 2) < 10
    strategy.entry("L", strategy.long)
strategy.exit("a", "L", qty_percent = 50, profit = 300)
strategy.exit("b", "L", qty_percent = 25, limit = strategy.position_avg_price * 1.05)
strategy.exit("c", "L", loss = 500)
"""))


def test_points_scale_out_validation():
    Strategy(universe=["SPY"], entry="close > 0", stop_loss=0.05, scale_out=[{"points": 2, "fraction": 0.5}]).validate()
    with pytest.raises(ValueError, match="points"):
        Strategy(universe=["SPY"], entry="close > 0", stop_loss=0.05,
                 scale_out=[{"points": 2, "at": 0.1, "fraction": 0.5}]).validate()


# ------------------------------------------------------------ 8. phrases

@pytest.mark.parametrize("text,entry,extra", [
    ("buy SPY when the 10 day SMA crosses above the 30 day SMA, exit on the opposite cross",
     "(crossover(sma(close, 10), sma(close, 30)))", {"exit_when": "sma(close, 10) < sma(close, 30)"}),
    ("short SPY when the 10 day SMA crosses below the 30 day SMA, cover on the opposite cross",
     "(crossunder(sma(close, 10), sma(close, 30)))", {"exit_when": "sma(close, 10) > sma(close, 30)"}),
    ("buy SPY when the price breaks out above the Donchian channel (20), hold 5 days", "(close > ref(highest(high, 20), 1))", {}),
    ("buy SPY when it breaks above the 55 day Donchian channel, hold 5 days", "(close > ref(highest(high, 55), 1))", {}),
    ("buy SPY when stochastic crosses above 20 from below, hold 5 days", "(crossover(stoch_k(14, 1), 20))", {}),
    ("buy SPY when RSI(2) < 10, 2% stop, 1:3 risk reward", "(rsi(close, 2) < 10)", {"target_r": 3.0, "stop_loss": 0.02}),
    ("buy SPY when RSI(2) < 10, 2% stop, risk-reward of 1:2.5", "(rsi(close, 2) < 10)", {"target_r": 2.5}),
    ("buy SPY when it closes above the open, hold 5 days", "(close > open)", {}),
    ("buy SPY when it closes higher than it opened, hold 5 days", "(close > open)", {}),
    ("buy SPY when it closes below its open, hold 5 days", "(close < open)", {}),
    ("buy SPY when it closes up on the day, hold 5 days", "(close > ref(close, 1))", {}),
    ("buy SPY when the 50 SMA is rising, hold 5 days", "sma(close, 50) > ref(sma(close, 50), 1)", {}),
    ("buy SPY when an inside bar breaks to the upside, hold 5 days", "(high < ref(high, 1) and low > ref(low, 1))",
     {"entry_order": "stop", "entry_level": "high"}),
])
def test_phrases(text, entry, extra):
    s = parser.parse(text)
    assert s.entry == entry or (s.short_entry == entry)
    for k, v in extra.items():
        assert getattr(s, k) == (pytest.approx(v) if isinstance(v, float) else v), k


@pytest.mark.parametrize("text,msg", [
    ("buy SPY when VIX closes back inside, hold 5 days", "back inside what"),
    ("buy SPY when stochastic crosses above 20 from above, hold 5 days", "which one"),
    ("buy SPY when RSI(2) < 10, exit on the opposite cross", "opposite of which cross"),
])
def test_phrases_that_ask(text, msg):
    with pytest.raises(ParseError, match=msg):
        parser.parse(text)


@needs_data
def test_inside_bar_breakout_fills_at_the_inside_bar_high():
    s = parser.parse("buy SPY when an inside bar breaks to the upside, hold 5 days, since 2015")
    r = engine.run(s)
    books(r)
    px = data.load("SPY")
    t = r.trades
    assert len(t) > 20
    for _, row in t.head(20).iterrows():
        i = px.index.get_loc(pd.Timestamp(row.entry_date))
        sig = px.iloc[i - 1]                      # the inside bar, the day before the fill
        prev = px.iloc[i - 2]
        assert sig.high < prev.high and sig.low > prev.low
        assert row.entry_price >= sig.high - 1e-6 or row.entry_fill == "open"
