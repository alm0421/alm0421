"""Round 12 (a TradingView expert's review): TradingView-compatible mode charges no borrow fee and makes no margin calls,
today's entries at the close are BUY actions in the signal scan (a same-day sell and re-buy is a net hold), entries
that compare two values both ways are refused, Pine arrays / per-side exits / counting for loops, clearer error
messages, and phrases ('at today's open', 'hold 3 days' with no comma, '3 positions', 'it crosses below zero', the
weekly / monthly moving averages, Heikin Ashi runs)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr, parser, signals, web
from backtester.parser import ParseError
from backtester.strategy import Strategy

AVAIL = set(data.available_tickers())
needs_data = pytest.mark.skipif(not {"SPY", "QQQ"} <= AVAIL, reason="price data not downloaded")


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


H = '//@version=5\nstrategy("x", overlay=true, initial_capital=100000)\n'
BB = H + """[mid, up, lo] = ta.bb(close, 20, 2)
if ta.crossover(close, lo)
    strategy.entry("L", strategy.long)
if ta.crossunder(close, up)
    strategy.entry("S", strategy.short)
"""


# ------------------------------------------------------------ 1. no borrow fee, no margin calls in TradingView mode

LS = "long QQQ when the close crosses above its 20 day SMA, short when the close crosses below its 20 day SMA"


def test_tradingview_mode_has_no_borrow_fee_and_no_margin_calls():
    s = parser.parse(LS, tv_compat=True)
    s.validate()
    assert s.borrow_fee == 0.0 and s.maintenance_margin == 0.0
    note = next(n for n in s.notes if n.startswith("TradingView-compatible returns:"))
    assert "no borrow fee on shorts" in note and "no maintenance-margin calls" in note
    # the default mode keeps the assumed fees and the 25% maintenance margin
    d = parser.parse(LS)
    assert d.borrow_fee is None and d.maintenance_margin == 0.25


@pytest.mark.parametrize("extra,fee,mm", [
    (", with a 0.3% borrow fee", 0.003, 0.0),
    (", with a 25% maintenance margin", 0.0, 0.25),
])
def test_tradingview_mode_keeps_stated_financing(extra, fee, mm):
    s = parser.parse(LS + extra, tv_compat=True)
    assert s.borrow_fee == pytest.approx(fee) and s.maintenance_margin == pytest.approx(mm)


def test_tradingview_json_spec_and_pine_import_defaults():
    s = Strategy.from_dict({"universe": ["QQQ"], "side": "both", "entry": "close > open", "short_entry": "close < open",
                            "tv_compat": True})
    assert s.borrow_fee == 0.0 and s.maintenance_margin == 0.0 and s.cash_rate is None
    s = Strategy.from_dict({"universe": ["QQQ"], "side": "both", "entry": "close > open", "short_entry": "close < open",
                            "tv_compat": True, "borrow_fee": 0.01, "maintenance_margin": 0.3})
    assert s.borrow_fee == 0.01 and s.maintenance_margin == 0.3
    p = parser.parse(BB, ticker="QQQ")
    assert p.tv_compat and p.borrow_fee == 0.0 and p.maintenance_margin == 0.0
    assert any("no borrow fees on shorts, no margin calls" in n for n in p.notes)


def test_site_switch_to_tradingview_mode_drops_borrow_fee_and_margin_calls():
    spec = parser.parse(LS)
    web._tv_switch(spec, {"tv_compat": True})
    spec.tv_compat = True                 # (web._spec sets the option after the switch)
    assert spec.borrow_fee == 0.0 and spec.maintenance_margin == 0.0 and spec.cash_rate is None
    web._tv_switch(spec, {"tv_compat": False})
    spec.tv_compat = False
    assert spec.borrow_fee is None and spec.maintenance_margin == 0.25


def test_tradingview_mode_charges_shorts_nothing(fake):
    fake["SYN"] = rand_bars(900, seed=3)
    base = dict(universe=["SYN"], side="both", entry="crossover(close, sma(close, 20))",
                short_entry="crossunder(close, sma(close, 20))", cash_rate=None, dividends=False)
    tv = engine.run(Strategy.from_dict({**base, "tv_compat": True}))
    books(tv)
    shorts = tv.trades[tv.trades["side"] == "short"]
    assert len(shorts) > 5 and (shorts["income"] == 0).all()      # no borrow fee
    plain = engine.run(Strategy(**base))
    assert (plain.trades[plain.trades["side"] == "short"]["income"] < 0).any()


@needs_data
def test_pine_bollinger_long_short_on_qqq_runs_without_fees_or_margin_calls():
    s = parser.parse(BB, ticker="QQQ")
    s.start = "2015-01-01"
    r = engine.run(s)
    books(r)
    shorts = r.trades[r.trades["side"] == "short"]
    assert len(shorts) > 10 and (shorts["income"] == 0).all()
    assert not any(n.startswith("Margin call") for n in s.notes)


def test_readme_lists_the_financing_defaults():
    txt = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert "Shorts pay no borrow fee and there are no maintenance-margin calls" in txt


# ------------------------------------------------------------ 3. today's entries at the close in the signal scan

def _scan_frames(fake):
    n = 300
    c_a = np.full(n, 100.0)
    c_a[-2:] = 80.0                  # below 90 on the last two bars: bought at T-1's close, re-bought at T's close
    c_b = np.full(n, 100.0)
    c_b[-1] = 80.0                   # a new signal on the last bar only
    for t, c in (("AAA", c_a), ("BBB", c_b)):
        fake[t] = bars(np.column_stack([c, c * 1.01, c * 0.99, c]), start="2020-01-01")


def test_scan_reports_todays_close_entries_as_buys(fake):
    _scan_frames(fake)
    s = Strategy(universe=["AAA", "BBB"], entry="close < 90", hold_bars=1, max_positions=2, cash_rate=None)
    out = signals.scan(s)
    ent = {e["ticker"]: e for e in out["entry_signals"]}
    assert set(ent) == {"AAA", "BBB"}
    assert ent["BBB"]["action"] == "BUY" and ent["BBB"]["done"] and not ent["BBB"]["rebought"]
    assert ent["AAA"]["rebought"]                       # sold (time exit) and bought back at the same close
    assert any(e["ticker"] == "AAA" and e["done"] for e in out["exit_signals"])
    act = out["action"]
    assert "No new entry signals" not in act
    assert "BUY BBB at today's close" in act
    assert "AAA: SELL at today's close (time exit) and BUY again at the same close" in act and "net, keep holding AAA" in act
    assert "SELL AAA at today's close" not in act     # said once, as the net hold
    txt = signals.format_alert(out)
    assert "BUY BBB at today's close" in txt and "sold and bought back at the same close" in txt
    assert out["skipped_already_held"] == []            # neither was held before today's close


def test_scan_counts_positions_held_before_today_as_held(fake):
    _scan_frames(fake)
    fake["AAA"].iloc[-3:, :4] = 80.0                    # AAA signals on T-2, T-1 and T: held since T-2 (hold 5 bars)
    s = Strategy(universe=["AAA", "BBB"], entry="close < 90", hold_bars=5, max_positions=2, cash_rate=None)
    out = signals.scan(s)
    assert [e["ticker"] for e in out["entry_signals"]] == ["BBB"]
    assert out["skipped_already_held"] == ["AAA"]


def test_daily_signals_script_writes_entries(fake, monkeypatch, tmp_path):
    import importlib.util
    _scan_frames(fake)
    s = Strategy(universe=["AAA", "BBB"], entry="close < 90", hold_bars=1, max_positions=2, cash_rate=None)
    row = {"name": "t", "registered": "2020-01-01", "return": 0.0, "max_drawdown": 0.0, "days": 1, "today": signals.scan(s)}
    monkeypatch.setattr(signals, "paper_report", lambda: [row])
    path = Path(__file__).resolve().parent.parent / "scripts" / "daily_signals.py"
    spec = importlib.util.spec_from_file_location("daily_signals_r12", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    mod.main()
    md = (tmp_path / "signals" / "latest.md").read_text()
    assert "entry: BUY BBB at today's close" in md and "net, keep holding it" in md


# ------------------------------------------------------------ 4. contradictory entries

@pytest.mark.parametrize("text", [
    "buy SPY when the 200 day SMA is above the 50 day SMA and the 50 day SMA is above the 200 day SMA, hold 3 days",
    "buy SPY when the close is above the 50 day SMA and the close is below the 50 day SMA, hold 3 days",
    "buy SPY when `close > open and open > close`, hold 3 days",
    "buy SPY when RSI(2) is above 50 and RSI(2) is below 30, hold 3 days",
])
def test_contradictory_entries_are_refused(text):
    with pytest.raises(ParseError, match="can never be true"):
        parser.parse(text)


def test_pair_bounds_in_expr():
    assert expr.bound_conflicts("sma(close, 200) > sma(close, 50) and sma(close, 50) > sma(close, 200)")[0] is False
    assert expr.bound_conflicts("close > open and close < open")[0] is False
    assert expr.bound_conflicts("close >= open and close <= open")[0] is None       # equal is possible
    assert expr.bound_conflicts("close > open and high > low")[0] is None
    t, msgs = expr.bound_conflicts("close > open or close <= open")
    assert t is True and "always true" in msgs[0]
    # the message keeps the order the rule was written in
    msg = expr.bound_conflicts("sma(close, 200) > sma(close, 50) and sma(close, 50) > sma(close, 200)")[1][0]
    assert "sma(close, 200) cannot be above sma(close, 50) and below sma(close, 50)" in msg
    with pytest.raises(ValueError, match="never be true"):
        Strategy(universe=["SPY"], entry="close > open and open > close", hold_bars=1).validate()


# ------------------------------------------------------------ 5. Pine: arrays, per-side exits, for loops

@pytest.mark.parametrize("line", ["var float[] highs = array.new_float()", "highs = array.new_float(0)",
                                  "array.push(highs, high)", "var highs = array.new<float>()"])
def test_pine_arrays_are_refused_with_alternatives(line):
    with pytest.raises(ParseError, match=r"line 3: arrays .* not supported.*ta\.highest"):
        parser.parse(H + line + "\nif close > open\n    strategy.entry(\"L\", strategy.long)\n", ticker="SPY")


LS_EXITS = H + """[mid, up, lo] = ta.bb(close, 20, 2)
if ta.crossover(close, lo)
    strategy.entry("L", strategy.long)
if ta.crossunder(close, up)
    strategy.entry("S", strategy.short)
strategy.exit("XL", "L", profit=800, loss=400)
strategy.exit("XS", "S", trail_points=300, trail_offset=200)
"""


def test_pine_long_short_exits_per_side(fake):
    fake["SYN"] = rand_bars(1500, seed=5, vol=0.015)
    s = parser.parse(LS_EXITS, ticker="SYN")
    assert s.stop_level == "entry_price - 4" and s.target_level == "entry_price + 8"
    assert s.side_exits == {"short": {"trailing_points": 2.0, "trail_activation_points": 3.0,
                                      "stop_level": None, "target_level": None}}
    assert s.trailing_points is None
    assert "short positions only: trailing stop 2 points" in s.summary()
    r = engine.run(s)
    books(r)
    t = r.trades
    longs, shorts = t[t["side"] == "long"], t[t["side"] == "short"]
    assert set(longs["exit_reason"]) <= {"stop level", "take profit", "reversal", "open at end"}
    assert set(shorts["exit_reason"]) <= {"trailing stop", "reversal", "open at end"}
    assert (longs["exit_reason"] == "take profit").any() and (shorts["exit_reason"] == "trailing stop").any()
    # a long's target is 8 above its entry, a short has none
    tp = longs[longs["exit_reason"] == "take profit"]
    assert (tp["exit_price"] >= tp["entry_price"] + 8 - 1e-6).all()


def test_side_exits_in_the_engine_and_validation(fake):
    fake["SYN"] = rand_bars(900, seed=7)
    base = dict(universe=["SYN"], side="both", entry="crossover(close, sma(close, 20))",
                short_entry="crossunder(close, sma(close, 20))", cash_rate=None)
    r = engine.run(Strategy(**base, side_exits={"long": {"stop_loss": 0.02}, "short": {"take_profit": 0.03}}))
    books(r)
    t = r.trades
    assert set(t[t["side"] == "long"]["exit_reason"]) <= {"stop loss", "reversal", "open at end"}
    assert set(t[t["side"] == "short"]["exit_reason"]) <= {"take profit", "reversal", "open at end"}
    with pytest.raises(ValueError, match="can't be set for one side"):
        Strategy(**base, side_exits={"long": {"hold_bars": 3}}).validate()
    with pytest.raises(ValueError, match="positive number"):
        Strategy(**base, side_exits={"long": {"stop_loss": -0.02}}).validate()
    with pytest.raises(ValueError, match="no short positions"):
        Strategy(universe=["SYN"], entry="close > open", hold_bars=2, side_exits={"short": {"stop_loss": 0.02}}).validate()
    # a side_exits-only exit counts as an exit rule
    Strategy(universe=["SYN"], entry="close > open", side_exits={"long": {"stop_loss": 0.02}}).validate()


LOOP = H + """len = input.int(10)
total = 0.0
for i = 0 to len - 1
    total := total + close[i]
avg = total / len
if ta.crossover(close, avg)
    strategy.entry("L", strategy.long)
if ta.crossunder(close, avg)
    strategy.close("L")
"""


def test_pine_counting_for_loop_is_unrolled(fake):
    fake["SYN"] = rand_bars(400, seed=2)
    s = parser.parse(LOOP, ticker="SYN")
    assert any("for i = 0 to 9 unrolled (10 iterations), updating total" in n for n in s.notes)
    ns = expr.Namespace(fake["SYN"])
    a = expr.evaluate(s.entry, ns)
    b = expr.evaluate("crossover(close, sma(close, 10))", ns)
    assert (a == b).all() and b.sum() > 5
    s2 = parser.parse(LOOP.replace("total := total + close[i]", "total += close[i]"), ticker="SYN")
    assert (expr.evaluate(s2.entry, ns) == b).all()


@pytest.mark.parametrize("body,msg", [
    ("t = 0.0\nfor i = 0 to 500\n    t += close[i]\n", "runs 501 times"),
    ("t = 0.0\nfor i = 0 to bar_index\n    t += close[i]\n", "whole number fixed"),
    ("t = 0.0\nfor i = 0 to 9\n    if close[i] > open[i]\n        t += 1\n", "only updates of a variable"),
    ("for i = 0 to 9\n    t += close[i]\n", "not declared before"),
    ("t = 0.0\nfor x in closes\n    t += x\n", "counting loop"),
])
def test_pine_loops_that_are_refused(body, msg):
    with pytest.raises(ParseError, match=rf"line 4:.*{msg}"):
        parser.parse(H + body + "if close > t\n    strategy.entry(\"L\", strategy.long)\n", ticker="SPY")


# ------------------------------------------------------------ 6. error messages

def test_backticked_value_compared_with_a_number():
    assert parser.parse("buy SPY when `ta.highest(high, 20)` crosses above 0, hold 3 days").entry == \
        "(crossover(highest(high, 20), 0))"
    assert parser.parse("buy SPY when `rsi(close, 2)` is below 10, hold 3 days").entry == "(rsi(close, 2) < 10)"
    assert parser.parse("buy SPY when `rsi(close, 2)` crosses below 10, hold 3 days").entry == "(crossunder(rsi(close, 2), 10))"
    with pytest.raises(ParseError, match="already a condition"):
        parser.parse("buy SPY when `rsi(close, 2) < 10` is above 5, hold 3 days")


def test_opposite_cross_with_a_compound_entry():
    s = parser.parse("buy SPY when the 10 day SMA crosses above the 50 day SMA and RSI(14) is above 50, exit on the opposite cross")
    assert s.exit_when == "crossunder(sma(close, 10), sma(close, 50))"
    s = parser.parse("buy SPY when MACD crosses above its signal line and the close is above the 200 day SMA, exit on the "
                     "opposite cross")
    assert s.exit_when == "crossunder(macd(), macd_signal())"
    with pytest.raises(ParseError, match=r"'exit on the opposite cross': the entry has 2 crosses") as e:
        parser.parse("buy SPY when the close crosses above the 200 day SMA or the 10 day SMA crosses above the 50 day SMA, "
                     "exit on the opposite cross")
    assert "it crosses below it" not in str(e.value)


def test_dunder_names_get_no_nonsense_suggestion():
    with pytest.raises(ParseError) as e:
        parser.parse('buy SPY when `__import__("os")`, hold 3 days')
    msg = str(e.value)
    assert ("not a function of the rule language" in msg or "not allowed in rules" in msg) and "__import__(close" not in msg
    with pytest.raises(ValueError) as e2:      # a known window function keeps its example
        expr.compile_expr('sma(close, "20") > 1')
    assert "sma(close, 20)" in str(e2.value)


def test_negative_stop_message_says_positive():
    with pytest.raises(ValueError, match=r"stop_loss cannot be negative \(got -0.05\).*must be positive.*5 for 5%"):
        Strategy(universe=["SPY"], entry="close > open", stop_loss=-0.05).validate()


# ------------------------------------------------------------ 7. phrases

@pytest.mark.parametrize("text,check", [
    ("buy SPY at today's open when it gaps down 1%, sell at the close", {"entry_fill": "open", "entry": "(gap <= -0.01)"}),
    ("buy SPY at the open today when it gaps down 1%, sell at the close", {"entry_fill": "open"}),
    ("buy SPY when RSI(2) is below 10 hold 3 days", {"entry": "(rsi(close, 2) < 10)", "hold_bars": 3}),
    ("buy SPY when RSI(2) is below 10 hold for 3 days", {"hold_bars": 3}),
    ("buy SPY when RSI(2) is below 10, hold 3 days, 3 positions", {"max_positions": 3}),
    ("buy SPY when RSI(2) is below 10, hold 3 days, with 4 positions", {"max_positions": 4}),
    ("buy SPY when the MACD histogram crosses above zero, sell when it crosses below zero",
     {"exit_when": "(crossunder(macd_hist(), 0))"}),
    ("buy SPY when the weekly 20 EMA is rising, hold 5 days", {"entry": "(weekly(ema(close, 20) > ref(ema(close, 20), 1)))"}),
    ("buy SPY when the weekly 20 EMA slope is positive, hold 5 days",
     {"entry": "(weekly(ema(close, 20) > ref(ema(close, 20), 1)))"}),
    ("buy SPY when the slope of the monthly 10 SMA is negative, hold 5 days",
     {"entry": "(monthly(sma(close, 10) < ref(sma(close, 10), 1)))"}),
    ("buy SPY when the close is above the 20 EMA of the weekly chart, hold 5 days", {"entry": "(close > weekly_ema(20))"}),
    ("buy SPY when the monthly SMA(10) is below the close, hold 5 days", {"entry": "(close > monthly_sma(10))"}),
    ("buy SPY when heikin ashi turns green two days in a row, hold 5 days",
     {"entry": "(count(ha_close > ha_open, 2) == 2 and ref(ha_close, 2) <= ref(ha_open, 2))"}),
    ("buy SPY when heikin ashi candles are green 2 days in a row, hold 5 days",
     {"entry": "count(((ha_close > ha_open)), 2) == 2"}),
])
def test_round12_phrases(text, check):
    s = parser.parse(text)
    for k, v in check.items():
        assert getattr(s, k) == v, (k, getattr(s, k))


def test_today_open_is_open_safe_and_explained():
    s = parser.parse("buy SPY at today's open when RSI(2) is below 10, sell at the close")
    s.validate()
    assert s.entry_fill == "open" and expr.open_safe(s.entry)
    assert any("at the open of the signal day" in n for n in s.notes)
    # 'buy and hold' and other uses of 'hold' are left alone
    assert parser.parse("buy SPY and hold").__class__.__name__ == "Portfolio"


def test_weekly_rising_holds_through_the_week(fake):
    fake["SYN"] = rand_bars(400, seed=4)
    ns = expr.Namespace(fake["SYN"])
    a = expr.evaluate("weekly(ema(close, 20) > ref(ema(close, 20), 1))", ns)
    wk = expr.evaluate_value("weekly_ema(20)", ns)
    # the value changes only at week ends, and agrees with the weekly EMA's week-over-week change
    ends = wk[wk.diff().fillna(0) != 0].index
    assert len(ends) > 50
    for d in ends[25:40]:
        assert bool(a.loc[d]) == bool(wk.loc[d] > wk.loc[:d].iloc[:-1][wk.loc[:d].iloc[:-1] != wk.loc[d]].iloc[-1])
