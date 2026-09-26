"""Round 6 (a TradingView user's review): pronouns in exits resolve to the entry's subject, today's exit orders
from `signals`, pyramiding without dust trades, SMA-seeded weekly/monthly EMAs, slot contention, gap entries at
the open, equal slots for a short ticker list, Pine spellings in the rule language, breakeven stops and
valuewhen / bars_since."""
import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr, parser, report, signals
from backtester.parser import ParseError
from backtester.strategy import Strategy

HAVE = {"QQQ", "SPY", "AAPL", "TSLA", "NVDA"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def bars(rows, start="2019-12-30", volume=1e6):
    idx = pd.bdate_range(start, periods=len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = volume
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    df["split"] = 0.0
    return df


@pytest.fixture
def fake(monkeypatch):
    frames = {}
    real = data.load

    def load(t):
        if t in frames:
            return frames[t]
        if t in ("SPY", "QQQ", "SPYSIM"):
            raise FileNotFoundError(t)
        return real(t)
    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {t: frames[t] for t in ts})
    return frames


def check_books(r):
    s = r.strategy
    assert r.equity.iloc[-1] == pytest.approx(s.capital + r.trades.pnl.sum() + r.interest)


# ------------------------------------------------------------ 1. pronouns in the exit

@pytest.mark.parametrize("text,want", [
    ("buy AAPL when the 50 day moving average is rising, sell when it is falling, since 2010",
     "sma(close, 50) < ref(sma(close, 50), 1)"),
    ("buy AAPL when the 50 day moving average is rising, sell when they are falling",
     "sma(close, 50) < ref(sma(close, 50), 1)"),
    ("buy AAPL when the 20 day EMA is falling, sell when it turns up", "ema(close, 20) > ref(ema(close, 20), 1)"),
    ("buy AAPL when the 10 day RSI is rising, sell when it's falling", "rsi(close, 10) < ref(rsi(close, 10), 1)"),
    ("buy AAPL when the 50 day moving average is rising, sell when it crosses below 100",
     "crossunder(sma(close, 50), 100)"),
    ("buy AAPL when the 50 day moving average is rising, sell when it is below 100", "sma(close, 50) < 100"),
    ("buy AAPL when RSI(2) < 10, sell when it is above 70", "rsi(close, 2) > 70"),
    ("buy AAPL when the 14 day ROC crosses above 0, sell when it crosses below 0", "crossunder(ret(close, 14), 0)"),
])
def test_it_in_the_exit_is_the_entrys_subject(text, want):
    s = parser.parse(text)
    assert want in s.exit_when
    assert "close < ref(close, 1)" not in s.exit_when
    assert any(n.startswith("Warning:") for n in s.notes)


@pytest.mark.parametrize("text,want", [
    ("buy AAPL when it is rising, sell when it is falling", "close < ref(close, 1)"),
    ("buy SPY when it closes above its 200 day moving average, sell when it crosses back below", "close < sma(close, 200)"),
    ("buy SPY when it is above its 200 day moving average, sell when it is falling", "close < ref(close, 1)"),
])
def test_it_is_the_price_when_the_entry_is_about_the_price(text, want):
    assert want in parser.parse(text).exit_when


@pytest.mark.parametrize("text", [
    "buy AAPL when RSI(2) < 10 and the 50 day moving average is rising, sell when it is falling",
    "buy AAPL when the 20 day EMA is rising and the 50 day moving average is rising, sell when it is falling",
    "buy AAPL when the 20 day EMA is rising and the 14 day ROC is above 0, sell when it is below 100",
])
def test_ambiguous_it_is_refused(text):
    with pytest.raises(ParseError, match="ambiguous"):
        parser.parse(text)


def test_reinterpretation_warnings_are_listed_first(capsys):
    from backtester.__main__ import main
    main(["buy AAPL when the 50 day moving average is rising, sell when it is falling", "--dry-run"])
    out = capsys.readouterr().out
    lines = [x for x in out.splitlines() if x.startswith(("WARNING:", "Note:"))]
    assert lines[0].startswith("WARNING: 'it' in 'it is falling' was read as the entry's sma(close, 50)")


def test_warning_notes_become_report_warnings(fake):
    fake["X"] = bars([(100 + i, 101 + i, 99 + i, 100 + i) for i in range(80)])
    s = parser.parse("buy AAPL when the 5 day moving average is rising, sell when it is falling")
    s.universe, s.cash_rate = ["X"], None
    A = report.analyze(engine.run(s), sensitivity=False)
    assert any(w["code"] == "interpretation" and "sma(close, 5)" in w["detail"] for w in A["warnings"])
    assert "Note: Warning" not in report.console_summary(A)


# ------------------------------------------------------------ 2. exits in today's signals

def _cut(monkeypatch, day):
    real = data.load.__wrapped__ if hasattr(data.load, "__wrapped__") else data.load

    def load(t):
        return real(t).loc[:day]
    monkeypatch.setattr(data, "load", load)


@needs_data
def test_signals_say_sell_at_the_next_open_for_an_exit_rule():
    out = signals.scan(parser.parse("buy QQQ at the close when RSI(2) is below 95, sell at the next open when RSI(2) is above 60"))
    df = data.load("QQQ")
    r2 = expr.evaluate_value("rsi(close, 2)", expr.Namespace(df)).iloc[-1]
    nxt = [e for e in out["exit_signals"] if e["when"] == "at the next open"]
    if out["open_positions"]:
        assert bool(nxt) == (r2 > 60)
    if nxt:
        assert out["action"].startswith("SELL QQQ at the next open")
        assert "SELL QQQ at the next open" in signals.format_alert(out)


@needs_data
@pytest.mark.parametrize("text", [
    "buy QQQ at the close when RSI(2) is below 30, sell at the next open when RSI(2) is above 60, since 2024",
    "buy QQQ when RSI(2) is below 30, sell when RSI(2) is above 70, since 2024",
    "buy QQQ when RSI(2) is below 30, hold 4 days and sell at the open, since 2024",
    "buy QQQ when RSI(2) is below 30, hold 3 days, since 2024",
])
def test_signal_exits_on_day_d_equal_the_engines(monkeypatch, text):
    """Truncated at D, the scan's exits must be exactly what the full backtest does after D."""
    full = engine.run(parser.parse(text))
    tr = full.trades
    cal = full.equity.index[1:]
    days = [d for d in cal[-60:-2]]
    rng = np.random.default_rng(1)
    picks = sorted(set(rng.choice(len(days), size=8, replace=False)))
    # always include days just before exits
    ex_days = [cal[cal.get_loc(pd.Timestamp(d)) - 1] for d in tr["exit_date"] if pd.Timestamp(d) in cal[-58:-2]]
    check = sorted({days[i] for i in picks} | set(ex_days[-4:]))
    for D in check:
        with monkeypatch.context() as m:
            _cut(m, D)
            out = signals.scan(parser.parse(text))
        assert out["as_of"] == str(D.date())
        nxt = cal[cal.get_loc(D) + 1]
        # exits the engine made at D's close (MOC) and what it does at the next open
        done_close = {(e["ticker"], e["reason"]) for e in out["exit_signals"] if e.get("done") and e["order"] == "MOC"}
        want_close = {(r.ticker, r.exit_reason) for r in tr.itertuples() if pd.Timestamp(r.exit_date) == D and r.exit_fill == "close"}
        assert done_close == want_close
        at_open = {(e["ticker"], e["reason"], e["date"]) for e in out["exit_signals"] if e["when"] == "at the next open"}
        want_open = {(r.ticker, r.exit_reason, str(nxt.date())) for r in tr.itertuples()
                     if pd.Timestamp(r.exit_date) == nxt and r.exit_fill == "open" and r.exit_reason != "open at end"}
        assert at_open == want_open
        # scheduled time exits carry the engine's exit date
        for e in out["exit_signals"]:
            if e["reason"] == "time exit" and not e.get("done"):
                match = tr[(tr["ticker"] == e["ticker"]) & (tr["exit_reason"] == "time exit")
                           & (pd.to_datetime(tr["entry_date"]) <= D) & (pd.to_datetime(tr["exit_date"]) > D)]
                if len(match) and pd.Timestamp(e["date"]) <= cal[-1]:
                    assert str(match["exit_date"].iloc[0]) == e["date"]


@needs_data
def test_signal_stop_and_target_levels_are_the_engines(monkeypatch):
    text = "buy SPY when RSI(2) is below 20, with a 3% trailing stop and a 4% profit target, since 2023"
    full = engine.run(parser.parse(text))
    tr = full.trades
    cal = full.equity.index[1:]
    hits = tr[tr["exit_reason"].isin(["trailing stop", "take profit"])]
    hits = hits[pd.to_datetime(hits["exit_date"]) > cal[0] + pd.Timedelta(days=30)].tail(4)
    assert len(hits)
    for r in hits.itertuples():
        E = pd.Timestamp(r.exit_date)
        D = cal[cal.get_loc(E) - 1]
        if pd.Timestamp(r.entry_date) > D:
            continue
        with monkeypatch.context() as m:
            _cut(m, D)
            out = signals.scan(parser.parse(text))
        orders = {o["reason"]: o for o in out["exit_orders"] if o["ticker"] == r.ticker}
        o = orders[r.exit_reason]
        assert o["order"] == ("STOP" if r.exit_reason == "trailing stop" else "LIMIT")
        assert o["oca"]           # stop and target cancel each other
        o_px = data.load("SPY").loc[E, "open"]
        if r.exit_fill == "open":   # gapped through the level: filled at the open
            assert r.exit_price == pytest.approx(o_px)
        else:
            assert r.exit_price == pytest.approx(o["price"], rel=1e-6)


@needs_data
def test_orders_turn_next_open_exits_into_sells(monkeypatch):
    from backtester import orders
    text = "buy QQQ at the close when RSI(2) is below 95, sell at the next open when RSI(2) is above 60, since 2024"
    full = engine.run(parser.parse(text))
    tr = full.trades
    cal = full.equity.index[1:]
    opens = tr[(tr["exit_fill"] == "open") & (tr["exit_reason"] == "exit rule")]
    E = pd.Timestamp(opens["exit_date"].iloc[-1])
    D = cal[cal.get_loc(E) - 1]
    with monkeypatch.context() as m:
        _cut(m, D)
        o = orders.todays_orders(parser.parse(text), 100_000, "QQQ,100")
    row = next(x for x in o["orders"] if x["ticker"] == "QQQ")
    assert row["side"] == "SELL" and row["target_shares"] == 0
    assert any("at the next open" in n for n in o["notes"])


# ------------------------------------------------------------ 3. pyramiding without dust

@needs_data
def test_pyramiding_splits_the_default_size_and_makes_no_dust():
    s = parser.parse("buy SPY when RSI(14) < 35, pyramid up to 3 entries, sell when RSI(14) > 70, since 2018")
    r = engine.run(s)
    assert s.position_size == pytest.approx(1 / 3)
    assert any(n.startswith("Pyramiding: no size per entry given") for n in s.notes)
    assert (r.trades["position_value"] > 1000).all()
    assert (r.trades.groupby("exit_date").size() <= 3).all()
    check_books(r)


def test_min_order_skips_dust_and_warns_without_headroom(fake):
    rows = [(100, 100, 100, 100)] * 3 + [(100, 101, 99, 100 + 1e-9 * i) for i in range(6)] + [(100, 100, 100, 100)] * 3
    fake["X"] = bars(rows)
    s = Strategy(universe=["X"], entry="True", pyramiding=3, position_size=1.0, hold_bars=5, cash_rate=None)
    r = engine.run(s)
    assert (r.trades["shares"] * r.trades["entry_price"] >= 1.0).all()
    assert any(n.startswith("Warning: pyramiding add-ons were skipped") for n in s.notes)
    check_books(r)
    s2 = Strategy(universe=["X"], entry="True", hold_bars=1, sizing="fixed_dollars", fixed_amount=0.5, cash_rate=None)
    r2 = engine.run(s2)
    assert r2.trades.empty and any(n.startswith("Min order:") for n in s2.notes)


# ------------------------------------------------------------ 4. weekly / monthly EMA seeding

def _ema_ref(v, n):
    out = [np.nan] * len(v)
    if len(v) < n:
        return out
    e = float(np.mean(v[:n]))
    out[n - 1] = e
    a = 2 / (n + 1)
    for i in range(n, len(v)):
        e = a * v[i] + (1 - a) * e
        out[i] = e
    return out


@pytest.mark.parametrize("fn,per,n", [("weekly_ema", "weekly_close", 10), ("monthly_ema", "monthly_close", 6)])
def test_weekly_and_monthly_ema_are_sma_seeded(fn, per, n):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-05", "2019-12-31")
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6, "adj_close": c, "dividend": 0.0})
    ns = expr.Namespace(df)
    wc = ns[per]()
    chg = wc.ne(wc.shift()) & wc.notna()
    closes = wc[chg].to_numpy()
    ref = pd.Series(_ema_ref(closes, n), index=wc.index[chg])
    got = ns[fn](n)[chg]
    assert got.iloc[: n - 1].isna().all()
    assert np.allclose(got.iloc[n - 1:].to_numpy(), ref.iloc[n - 1:].to_numpy(), rtol=1e-12)


# ------------------------------------------------------------ 5. slot contention

@needs_data
def test_slot_contention_says_how_ties_were_broken():
    s = parser.parse("buy SPY, QQQ and AAPL when RSI(2) < 50, hold 1 day, max 1 position, since 2024")
    r = engine.run(s)
    assert any(n.startswith("Slots:") and "20-day average dollar volume" in n for n in s.notes)
    ranked = parser.parse("buy SPY, QQQ and AAPL when RSI(2) < 50, hold 1 day, max 1 position, prefer the lowest RSI, since 2024")
    engine.run(ranked)
    assert not any(n.startswith("Slots:") for n in ranked.notes)
    assert len(r.trades)


# ------------------------------------------------------------ 6. gap rules enter at the open

@needs_data
def test_gap_rule_without_timing_enters_at_the_open():
    s = parser.parse("buy QQQ when it gaps down 2%, sell at the close")
    assert s.entry_fill == "open" and s.hold_bars == 0
    assert any("known at the open" in n for n in s.notes)
    r = engine.run(s)
    assert (r.trades["entry_fill"] == "open").all()
    assert (r.trades["entry_date"] == r.trades["exit_date"]).all()
    assert parser.parse("buy QQQ at the close when it gaps down 2%, sell at the close").entry_fill == "close"
    assert parser.parse("buy QQQ when it gaps down 2% and RSI(2) < 10, sell at the close").entry_fill == "close"


# ------------------------------------------------------------ 7. a short explicit list gets one slot per ticker

@needs_data
def test_short_ticker_list_gets_equal_slots():
    s = parser.parse("buy Tesla and Nvidia when RSI(2) < 10, hold 1 day")
    assert s.max_positions == 2 and s.position_size == pytest.approx(0.5)
    assert any(n.startswith("2 tickers and no position limit given") for n in s.notes)
    big = parser.parse("buy SPY, QQQ, AAPL, TSLA, NVDA, MSFT, AMZN, GOOGL, META, NFLX, AMD when RSI(2) < 10, hold 1 day")
    assert big.max_positions == 10
    assert parser.parse("buy Tesla and Nvidia when RSI(2) < 10, hold 1 day, max 1 position").max_positions == 1


# ------------------------------------------------------------ 8. minor

def test_short_chandelier_is_from_the_low():
    s = Strategy(universe=["X"], entry="True", side="short", trailing_atr=3)
    assert "from the low" in s.summary()
    assert "from the high" in Strategy(universe=["X"], entry="True", trailing_atr=3).summary()


@needs_data
def test_lookback_longer_than_history_says_never_warmed_up(capsys):
    from backtester.__main__ import main
    assert main(["buy SPY when it is above its 9000 day moving average, hold 1 day", "--dry-run"]) != 0
    err = capsys.readouterr()
    assert "never warmed up" in (err.out + err.err) and "needs 9000 bars" in (err.out + err.err)


def test_report_says_never_warmed_up(fake):
    fake["X"] = bars([(100, 101, 99, 100)] * 50)
    s = Strategy(universe=["X"], entry="close > sma(close, 200)", hold_bars=1, cash_rate=None)
    A = report.analyze(engine.run(s), sensitivity=False)
    w = next(w for w in A["warnings"] if w["code"] == "no_trades")
    assert "never warmed up" in w["detail"] and "200 bars" in w["detail"]


@needs_data
@pytest.mark.parametrize("text,exit_", [
    ("short SPY when it is below its 50 day moving average, cover when it closes above it", "close > sma(close, 50)"),
    ("short SPY when it crosses below its 20 day EMA, cover when it closes back above it", "close > ema(close, 20)"),
])
def test_cover_when_it_closes_above_it(text, exit_):
    s = parser.parse(text)
    assert s.side == "short" and s.exit_when == exit_


@needs_data
def test_on_ticker_go_long_and_exit():
    s = parser.parse("on QQQ, go long when RSI(2) < 10 and exit when RSI(2) > 70")
    assert s.universe == ["QQQ"] and s.entry == "(rsi(close, 2) < 10)" and s.exit_when == "(rsi(close, 2) > 70)"
    s2 = parser.parse("for SPY and QQQ: buy when RSI(2) < 10, sell when RSI(2) > 70")
    assert s2.universe == ["SPY", "QQQ"]


# ------------------------------------------------------------ 9. Pine spellings

@pytest.mark.parametrize("pine,rule", [
    ("close[1]", "ref(close, 1)"),
    ("close[0] > open", "close > open"),
    ("ta.sma(close, 5) > close[2]", "sma(close, 5) > ref(close, 2)"),
    ("ta.crossover(ta.ema(close, 9), ta.ema(close, 21))", "crossover(ema(close, 9), ema(close, 21))"),
    ("ta.crossunder(ta.rsi(close, 14), 70)", "crossunder(rsi(close, 14), 70)"),
    ("ta.atr(14) > ta.highest(high, 20)[1] - ta.lowest(20)", "atr(14) > ref(highest(high, 20), 1) - lowest(20)"),
    ("ta.change(close) > 0", "diff(close) > 0"),
    ("ta.stdev(close, 20) < 5", "stdev(close, 20) < 5"),
    ("ta.macd(close, 12, 26, 9) > 0", "macd(12, 26) > 0"),
    ("math.abs(ta.change(close, 3)) > 1 and not (close < open)", "abs(diff(close, 3)) > 1 and (not close < open)"),
    ("ta.barssince(close > open) > 2", "bars_since(close > open) > 2"),
])
def test_pine_spellings(pine, rule):
    assert expr.pine_to_rule(pine) == rule


@pytest.mark.parametrize("bad", ["close[-1] > 0", "close[n] > 0", "ta.vwma(close, 5) > 0"])
def test_pine_lookahead_and_unknowns_are_refused(bad):
    with pytest.raises(ValueError):
        expr.compile_expr(bad)


def test_pine_indexing_matches_ref_and_is_open_safe(fake):
    rows = [(100 + np.sin(i), 101 + np.sin(i), 99 + np.sin(i), 100 + np.cos(i)) for i in range(120)]
    fake["X"] = bars(rows)
    a = engine.run(Strategy(universe=["X"], entry="ta.sma(close, 5) > close[1]", hold_bars=1, cash_rate=None))
    b = engine.run(Strategy(universe=["X"], entry="sma(close, 5) > ref(close, 1)", hold_bars=1, cash_rate=None))
    pd.testing.assert_frame_equal(a.trades, b.trades)
    assert a.strategy.entry == "sma(close, 5) > ref(close, 1)"
    assert expr.open_safe("open < close[1]") and not expr.open_safe("close > close[1]")


@needs_data
def test_pine_in_backticks_and_macd_atr_phrases():
    s = parser.parse("buy SPY when `ta.crossover(ta.ema(close, 9), ta.ema(close, 21)) and close > close[1]`, hold 3 days")
    assert s.entry == "(crossover(ema(close, 9), ema(close, 21)) and close > ref(close, 1))"
    m = parser.parse("buy SPY when MACD(12,26,9) crosses above signal, sell when MACD(12,26,9) crosses below signal")
    assert m.entry == "(crossover(macd(12, 26), macd_signal(12, 26, 9)))"
    assert m.exit_when == "(crossunder(macd(12, 26), macd_signal(12, 26, 9)))"
    a = parser.parse("buy SPY when the 14 day ATR is above its 50 day average, hold 5 days")
    assert a.entry == "(atr(14) > sma(atr(14), 50))"
    k = parser.parse("buy SPY when it closes 2 ATR below its 20 day moving average, hold 5 days")
    assert k.entry == "(close < sma(close, 20) - 2 * atr(14))"


# ------------------------------------------------------------ 10. breakeven stop, valuewhen, bars_since

def test_breakeven_stop_is_armed_from_the_next_bar(fake):
    rows = [(100, 100, 100, 100)] * 3 + [
        (100, 100, 100, 100),     # 3: entry at the close (100)
        (100.5, 103, 99.5, 101),  # 4: +3% high, but the stop is not armed on the bar that arms it
        (101, 101.5, 99, 99.5),   # 5: low 99 < 100 -> breakeven stop at 100
        (99, 99, 99, 99)]
    fake["X"] = bars(rows)
    s = Strategy(universe=["X"], entry="dow == 3", hold_bars=10, breakeven_after=0.02, cash_rate=None)
    r = engine.run(s)
    t = r.trades.iloc[0]
    assert t.exit_reason == "breakeven stop" and t.exit_price == pytest.approx(100)
    assert str(t.exit_date) == str(fake["X"].index[5].date())
    check_books(r)
    # never reached +2%: no breakeven stop
    fake["X"] = bars(rows[:4] + [(100, 101, 99, 100), (100, 101, 98, 99), (99, 99, 99, 99)])
    r2 = engine.run(Strategy(universe=["X"], entry="dow == 3", hold_bars=2, breakeven_after=0.02, cash_rate=None))
    assert r2.trades.iloc[0].exit_reason == "time exit"


def test_breakeven_short_and_gap_through(fake):
    rows = [(100, 100, 100, 100)] * 3 + [(100, 100, 100, 100), (99.5, 100.5, 97, 98), (102, 103, 101, 102), (102,) * 4]
    fake["X"] = bars(rows)
    r = engine.run(Strategy(universe=["X"], entry="dow == 3", side="short", hold_bars=10, breakeven_after=0.02,
                            cash_rate=None, maintenance_margin=0.0))
    t = r.trades.iloc[0]
    assert t.exit_reason == "breakeven stop" and t.exit_price == pytest.approx(102) and t.exit_fill == "open"
    check_books(r)


def test_breakeven_parses():
    s = parser.parse("buy SPY when RSI(2) < 10, move stop to breakeven after +2%, sell when RSI(2) > 70")
    assert s.breakeven_after == pytest.approx(0.02)
    assert "breakeven" in s.summary()
    s2 = parser.parse("buy SPY when RSI(2) < 10 with a 3% stop loss and move the stop to breakeven once it is up 1.5%, hold 10 days")
    assert s2.breakeven_after == pytest.approx(0.015) and s2.stop_loss == pytest.approx(0.03)
    with pytest.raises(ParseError, match="how much"):
        parser.parse("buy SPY when RSI(2) < 10, move stop to breakeven, hold 10 days")


def test_valuewhen_and_bars_since_are_causal():
    idx = pd.bdate_range("2020-01-01", periods=12)
    c = pd.Series([10, 11, 9, 12, 8, 13, 7, 14, 6, 15, 5, 16], index=idx, dtype=float)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0, "adj_close": c, "dividend": 0.0})
    ns = expr.Namespace(df)
    up = "close > ref(close, 1)"
    v0 = expr.evaluate_value(f"valuewhen({up}, close, 0)", ns)
    v1 = expr.evaluate_value(f"valuewhen({up}, close, 1)", ns)
    bs = expr.evaluate_value(f"bars_since({up})", ns)
    assert np.isnan(v0.iloc[0]) and v0.iloc[1] == 11 and v0.iloc[2] == 11 and v0.iloc[3] == 12
    assert np.isnan(v1.iloc[2]) and v1.iloc[3] == 11 and v1.iloc[5] == 12
    assert list(bs.iloc[1:5]) == [0, 1, 0, 1]
    # truncation at D changes nothing before D
    for d in range(4, 12):
        cut = expr.Namespace(df.iloc[:d])
        for rule in (f"valuewhen({up}, close, 1)", f"bars_since({up})"):
            a = expr.evaluate_value(rule, cut)
            b = expr.evaluate_value(rule, ns).iloc[:d]
            pd.testing.assert_series_equal(a, b, check_names=False)
    with pytest.raises(ValueError, match="negative"):
        expr.evaluate_value(f"valuewhen({up}, close, -1)", ns)
