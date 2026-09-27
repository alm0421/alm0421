"""Round 8 (QuantConnect review): an adversarial lookahead probe for Python-function rules, limit/stop entries and
brackets at the broker, ranking phrases, "today's members", market-cap coverage, the point-in-time NYSE schedule
(calendar functions at the open, pre-announced closures, pre-1971 holidays) and exact Nasdaq-100 change days."""
import datetime as dt
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import broker, calendar, data, engine, expr, orders, parser, runner, signals
from backtester.strategy import Strategy

HAVE = {"QQQ", "SPY", "AAPL", "SMCI"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def frame(closes, start="2005-01-03"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    opens = np.r_[closes[0], closes[:-1]] * 1.001
    df = pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * 1.004,
                       "low": np.minimum(opens, closes) * 0.996, "close": closes}, index=idx)
    df["volume"] = 1e6
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    df["split"] = 1.0
    return df


def walk(seed, n=2000, drift=0.0003, vol=0.015):
    r = np.random.default_rng(seed).normal(drift, vol, n)
    r[0] = 0.0
    return 100 * np.cumprod(1 + r)


@pytest.fixture
def fake(monkeypatch):
    frames = {}

    def load(t):
        return frames[data.canonical(t)]
    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): load(t) for t in ts})
    expr._CALLABLE_PROBES.clear()
    return frames


# ------------------------------------------------------------ 1. the callable lookahead probe

def repro(df, ns):          # the reviewer's rule: "oversold, and tomorrow is not a down day"
    return (ns["rsi"](2) < 10) & ~(df.close.shift(-1) < df.close)


def five_ahead(df, ns):
    return (ns["rsi"](2) < 10) & ~(df.close.shift(-5) < df.close)


def causal(df, ns):
    return (ns["rsi"](2) < 10) & (df.close > df.close.rolling(200).mean())


@pytest.mark.parametrize("fn", [repro, five_ahead])
def test_reviewer_style_leaks_are_refused(fake, fn):
    fake["X"] = frame(walk(1))
    with pytest.raises(ValueError, match="uses future data"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=fn, hold_bars=1))


def test_leak_far_from_the_data_edge_is_found_through_fire_days(fake):
    fake["X"] = frame(walk(2))
    cut = fake["X"].index[1200]

    def middle(df, ns):         # leaks only in the middle of the history: the latest bars are clean
        return (ns["rsi"](2) < 10) & ~((df.close.shift(-1) < df.close) & (df.index < cut))
    with pytest.raises(ValueError, match="uses future data"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=middle, hold_bars=1))


def test_leak_inside_a_narrow_run_window_is_refused(fake):
    fake["X"] = frame(walk(3))
    idx = fake["X"].index
    lo, hi = idx[800], idx[860]

    def narrow(df, ns):
        return (ns["rsi"](2) < 30) & ~((df.close.shift(-1) < df.close) & (df.index >= lo) & (df.index <= hi))
    with pytest.raises(ValueError, match="uses future data"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=narrow, hold_bars=1,
                            start=str(lo.date()), end=str(hi.date())))


def test_causal_callables_pass_and_the_probe_is_fast(fake):
    fake["X"] = frame(walk(4, n=5000))
    t = time.perf_counter()
    assert expr.callable_lookahead_probe(causal, fake["X"], "X") is None
    assert expr.callable_lookahead_probe(lambda df, ns: df.close.pct_change(5), fake["X"], "X", "value") is None
    assert time.perf_counter() - t < 15          # ~2-5 s: at most 1,200 cuts each
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry=causal, hold_bars=2))
    assert len(r.trades) > 0


@needs_data
@pytest.mark.parametrize("fn", [repro, five_ahead])
def test_reviewer_repro_on_qqq_is_refused(fn):
    from backtester import api
    expr._CALLABLE_PROBES.clear()
    with pytest.raises(ValueError, match="uses future data"):
        api.backtest(Strategy(universe=["QQQ"], entry=fn, hold_bars=1, start="2005-01-01"))
    with pytest.raises(ValueError, match="uses future data"):
        api.backtest(Strategy(universe=["QQQ"], entry=fn, hold_bars=1, start="2020-02-01", end="2020-03-31"))


# ------------------------------------------------------------ 2. limit / stop entries reach the broker

LIMIT = "buy SPY at a limit 1% below the close when `change > 0`, hold 3 days, 3% stop loss, 5% take profit"


def _cut(monkeypatch, day):
    real = data.load

    def load(t):
        return real(t).loc[:day]
    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): load(t) for t in ts})


def _signal_days(full, n=3):
    """Signal days D of the engine's last `n` limit entries (filled the session after D)."""
    spy = data.load("SPY")
    out = []
    for d in pd.to_datetime(full.trades["entry_date"]).drop_duplicates().iloc[-n:]:
        out.append(spy.index[spy.index.get_loc(d) - 1])
    return out


@needs_data
def test_limit_entry_is_an_order_at_the_engines_level(monkeypatch):
    spy = data.load("SPY")
    full = engine.run(parser.parse(LIMIT))
    for D in _signal_days(full):
        k = spy.index.get_loc(D)
        with monkeypatch.context() as m:
            _cut(m, D)
            sc = signals.scan(parser.parse(LIMIT))
            pl = broker.plan(parser.parse(LIMIT), 100_000, {}, fractional=False)
        lvl = spy["close"].iloc[k] * 0.99
        e = sc["entry_orders"][0]
        assert e["order"] == "LIMIT" and e["price"] == pytest.approx(lvl, abs=1e-4) and e["valid_sessions"] == 1
        assert "BUY LIMIT SPY" in sc["action"] and "market-on-close" not in sc["action"]
        o = pl.orders[0]
        assert (o["type"], o["order_class"], o["time_in_force"], o["side"]) == ("limit", "bracket", "day", "buy")
        assert o["limit_price"] == broker._px(lvl)
        assert o["stop_loss"] == {"stop_price": broker._px(lvl * 0.97)}
        assert o["take_profit"] == {"limit_price": broker._px(lvl * 1.05)}
        assert int(o["qty"]) == int(100_000 / lvl)
        assert not any("fill at the close of the signal day" in n for n in pl.notes)
        # the engine, on the full data, fills that very order the next session at the level (or a better open)
        nxt = spy.index[k + 1]
        t = full.trades[pd.to_datetime(full.trades["entry_date"]) == nxt]
        assert len(t) == 1
        want = min(spy["open"].iloc[k + 1], lvl)
        assert float(t["entry_price"].iloc[0]) == pytest.approx(want, rel=1e-9)


@needs_data
def test_stop_entry_and_multi_session_orders(monkeypatch):
    s = parser.parse("buy SPY with a stop order 1% above the close when `change > 0`, hold 3 days, 3% stop loss")
    res = runner.run(s)
    pend = res.extras["pending_entries"]
    if not pend:
        pytest.skip("no signal on the last bar")
    pl = broker.plan(s, 100_000, {}, fractional=False)
    o = pl.orders[0]
    assert o["type"] == "stop" and o["order_class"] == "oto" and o["stop_price"] == broker._px(pend[0]["level"])
    assert o["stop_loss"] == {"stop_price": broker._px(pend[0]["level"] * 0.97)}


def test_build_orders_for_engine_level_orders_and_market_entry_brackets():
    rows = [{"ticker": "AAPL", "side": "BUY", "shares": 10, "current_shares": 0, "target_shares": 10}]
    notes = []
    lv = {"AAPL": {"price": 100.0, "valid_sessions": 3, "until": "2026-10-01", "stop_loss": 97.0, "take_profit": 105.0}}
    o = broker.build_orders(rows, kind="signal", as_of="d", tag="t", fill="close", entry_order="limit",
                            entry_levels=lv, notes=notes)[0]
    assert (o["type"], o["limit_price"], o["time_in_force"], o["order_class"]) == ("limit", "100.00", "gtc", "bracket")
    assert o["stop_loss"] == {"stop_price": "97.00"} and any("3 more session" in n for n in notes)
    # a market entry at the next open: a day market bracket, protected from the entry session
    ex = {"AAPL": {"stop": 95.0, "target": 110.0}}
    o = broker.build_orders(rows, kind="signal", as_of="d", tag="t", fill="next_open", entry_exits=ex,
                            fractional=False)[0]
    assert (o["type"], o["time_in_force"], o["order_class"]) == ("market", "day", "bracket")
    # market-on-close: no bracket at Alpaca; the OCO pair is shown for after the fill
    after = []
    o = broker.build_orders(rows, kind="signal", as_of="d", tag="t", fill="close", entry_exits=ex,
                            fractional=False, after_fill=after)[0]
    assert o["time_in_force"] == "cls" and "order_class" not in o
    assert after and after[0]["order_class"] == "oco" and after[0]["qty"] == "10" and after[0]["side"] == "sell"


def test_submit_cancels_the_open_legs_of_our_earlier_brackets():
    from tests.test_review_round7_quantconnect import FakeResponse, FakeSession
    p = broker.Plan(as_of="2026-09-25", account_value=1e4, orders=[
        {"symbol": "SPY", "qty": "1", "side": "sell", "type": "limit", "order_class": "oco", "time_in_force": "day",
         "client_order_id": "bt-x-2026-09-25-SPY-oco"}])
    open_orders = [
        {"id": "p1", "client_order_id": "bt-x-2026-09-24-SPY-entry", "symbol": "SPY",
         "legs": [{"id": "l1", "status": "new"}, {"id": "l2", "status": "held"}]},
        {"id": "l3", "client_order_id": "auto-123", "symbol": "SPY", "order_class": "bracket", "type": "stop"},
        {"id": "m1", "client_order_id": "manual", "symbol": "QQQ", "type": "limit"}]
    s = FakeSession({("GET", "/v2/orders"): FakeResponse(200, open_orders),
                     ("POST", "/v2/orders"): FakeResponse(200, {"id": "n1", "status": "new"})})
    broker.submit(broker.AlpacaClient("k", "s", session=s), p, dry_run=False, log=lambda *_: None)
    deleted = [c["url"].rsplit("/", 1)[1] for c in s.calls if c["method"] == "DELETE"]
    assert sorted(deleted) == ["l1", "l2", "l3", "p1"]
    assert s.calls[0]["params"]["nested"] == "true"


@needs_data
def test_market_entry_dry_run_shows_the_exit_orders(monkeypatch):
    s = parser.parse("buy SPY at the next open when `change > -1`, hold 3 days, 3% stop loss, 5% take profit")
    pl = broker.plan(s, 100_000, {}, fractional=False)
    e = [o for o in pl.orders if o["client_order_id"].endswith("entry")]
    assert e and e[0]["order_class"] == "bracket" and e[0]["time_in_force"] == "day"
    s = parser.parse("buy SPY at the close when `change > -1`, hold 3 days, 3% stop loss, 5% take profit")
    pl = broker.plan(s, 100_000, {}, fractional=False)
    out = []
    broker.submit(broker.AlpacaClient(None, None), pl, dry_run=True, log=out.append)
    assert pl.after_fill and any("AFTER THE ENTRY FILLS" in x and '"oco"' in x for x in out)


# ------------------------------------------------------------ 3. ranking phrases

@pytest.mark.parametrize("text,rank,asc", [
    ("prefer the lowest RSI", "rsi(2)", True),
    ("prefer the Lowest RSI(2)", "rsi(close, 2)", True),
    ("rank by lowest RSI(2)", "rsi(close, 2)", True),
    ("prefer the highest RSI 3", "rsi(close, 3)", False),
    ("prefer the highest 20-day volatility", "volatility(20)", False),
    ("prefer the lowest 10 day return", "ret(10)", True),
])
def test_ranking_phrases_are_case_insensitive_and_take_parameters(text, rank, asc):
    s = parser.parse(f"buy nasdaq 100 stocks when RSI(2) < 10, hold 3 days, max 5 positions, {text}")
    assert (s.rank_by, s.rank_ascending, s.hold_bars, s.max_positions) == (rank, asc, 3, 5)


# ------------------------------------------------------------ 4. today's members

@needs_data
def test_todays_members_trade_only_the_current_list_with_a_warning():
    cur = set(data.current_members())
    s = parser.parse("buy nasdaq 100 stocks when RSI(2) < 5, hold 3 days, using today's members only, since 2018")
    assert set(s.universe) <= cur and "today's members (survivorship-biased by construction)" in s.summary()
    r = runner.run(s)
    assert set(r.trades["ticker"]) <= cur
    assert any(n.startswith("Warning: survivorship bias") for n in s.notes)
    p = parser.parse("hold the top 5 Nasdaq 100 stocks by 3 month momentum, rebalance monthly, "
                     "using today's members only, since 2018")
    r = runner.run(p)
    held = set(r.holdings.columns[(r.holdings.abs() > 0).any()]) - {"cash"}
    assert held <= cur and any(n.startswith("Warning: survivorship bias") for n in p.notes)
    assert "today's members" in p.summary()


# ------------------------------------------------------------ 5. market-cap coverage

@needs_data
def test_market_cap_gap_is_noted_and_etfs_are_refused():
    s = parser.parse("buy AAPL at the close when `market_cap > 1e9`, hold 5 days, since 2005")
    r = runner.run(s)
    first = data.market_cap("AAPL").dropna().index[0]
    assert pd.Timestamp(r.trades["entry_date"].min()) >= first
    assert any(f"market_cap has no value before {first.date()} for AAPL" in n for n in s.notes)
    with pytest.raises(ValueError, match="ETFs have no market cap"):
        runner.run(parser.parse("buy SPY at the close when `market_cap > 1e9`, hold 5 days"))


# ------------------------------------------------------------ 6. calendar functions and the open

def test_trading_days_left_in_month_counts_sessions_after_today():
    idx = pd.bdate_range("2026-08-24", "2026-09-30")
    idx = idx[[calendar.is_session(d) for d in idx]]
    left = pd.Series(calendar.scheduled_sessions_left(idx, "M"), index=idx)
    assert left["2026-08-31"] == 0 and left["2026-09-30"] == 0 and left["2026-09-29"] == 1
    assert left["2026-09-04"] == 17          # Labor Day (Sep 7) skipped
    s = parser.parse("buy SPY at the close on the last trading day of the month, sell at the close 5 days later")
    assert s.entry == "(trading_days_left_in_month == 0)"


def test_period_end_flags_are_known_at_the_open(fake):
    fake["X"] = frame(walk(5, n=400), start="2024-01-02")
    for rule in ("is_month_end()", "is_week_end() and gap < 0", "trading_days_left_in_month == 0"):
        assert expr.open_safe(rule)
        Strategy(universe=["X"], entry=rule, entry_fill="open", hold_bars=0).validate()
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="is_month_end()", entry_fill="open", hold_bars=0))
    days = pd.to_datetime(r.trades["entry_date"])
    days = [d for d in days if calendar.is_session(d)]    # (the synthetic bars include holidays)
    assert len(days) and all(d.month != calendar.next_sessions(d)[0].month for d in days)   # same day, not lagged
    assert not expr.open_safe("is_month_end() and close > open")


# ------------------------------------------------------------ 7. pre-announced closures and old holidays

def test_pre_announced_closures_are_in_the_schedule_from_their_announcement():
    for d in ("2004-06-11", "2007-01-02", "2018-12-05", "2025-01-09"):
        assert not calendar.is_session(d)
    idx = pd.DatetimeIndex(["2004-06-04", "2004-06-07", "2018-12-04", "2024-12-27", "2024-12-31"])
    left = calendar.scheduled_sessions_left(idx, "M")
    # June 2004: Jun 4 (before the June 6 announcement) does not know Jun 11 yet (18 weekdays left); on Jun 7
    # it is known (Jun 8..30 without Jun 11: 16)
    assert left[0] == 18 and left[1] == 16
    nxt = calendar.next_scheduled(pd.DatetimeIndex(["2018-12-04", "2025-01-08", "2024-12-27"]))
    assert list(nxt.strftime("%Y-%m-%d")) == ["2018-12-06", "2025-01-10", "2024-12-30"]
    wk = calendar.scheduled_period_end(pd.DatetimeIndex(["2004-06-10", "2001-09-10"]), "W-FRI")
    assert list(wk) == [True, False]          # Reagan: known; 9/11: never scheduled


@pytest.mark.parametrize("day,open_", [
    ("1968-11-05", False), ("1965-02-22", False), ("1969-05-30", False), ("1953-07-03", True), ("1976-11-02", False),
    ("1984-11-06", True), ("1939-11-23", False), ("1941-11-20", False), ("1935-02-12", False), ("1950-10-12", False),
    ("1968-02-12", False), ("1954-12-24", False), ("1970-05-29", True), ("1970-02-23", False), ("1972-11-07", False),
])
def test_historical_holiday_rules(day, open_):
    assert calendar.is_session(day) == open_


@needs_data
def test_holidays_agree_with_the_sp500_bars_since_1928():
    bars = set(d.date() for d in data.load("^GSPC").index)
    for y in range(1928, 2026):
        assert not (calendar.closures(y) & bars), y


# ------------------------------------------------------------ 8. exact membership change days

def _fetch_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("fetch_data_r8", root / "scripts" / "fetch_data.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


WIKI = """Intro text.

{| class="wikitable sortable" id="changes"
! rowspan="2" data-sort-type="date" |Date
! colspan="2" |Added
! colspan="2" |Removed
! rowspan="2" |Reason
|-
!Ticker
!Security
!Ticker
!Security
|-
|December 23, 2024
|PLTR
|[[Palantir Technologies]]
|ILMN
|[[Illumina, Inc.|Ilumina]]
|Annual index reconstitution.<ref name=":0">{{cite news |title=x}}</ref>
|-
|July 22, 2024 || SMCI || [[Supermicro]] || WBA || [[Walgreens Boots Alliance]] || weight
|-
|September 14, 2026
|
|
|KHC
|[[Kraft Heinz]]
|Moved to the NYSE.
|}
"""


def test_change_table_parser():
    rows = _fetch_module().parse_changes(WIKI)
    assert [(r["date"], r["added"], r["removed"]) for r in rows] == [
        ("2024-12-23", "PLTR", "ILMN"), ("2024-07-22", "SMCI", "WBA"), ("2026-09-14", "", "KHC")]


def test_change_days_override_the_monthly_snapshots(tmp_path, monkeypatch):
    f = tmp_path / "ch.csv"
    f.write_text("date,added,removed,reason\n2024-07-22,AAA,BBB,x\n")
    monkeypatch.setattr(data, "CHANGES_FILE", f)
    data.ndx_changes.cache_clear()
    try:
        idx = pd.bdate_range("2024-06-03", "2024-09-30")
        snap = np.zeros((len(idx), 3), bool)
        snap[:, 0] = idx >= "2024-08-01"          # monthly snapshot: AAA from August
        snap[:, 1] = idx < "2024-08-01"           # BBB until July
        snap[:, 2] = True                          # CCC: no change
        out = data.apply_changes(snap.copy(), ["AAA", "BBB", "CCC"], idx)
        m = pd.DataFrame(out, index=idx, columns=["AAA", "BBB", "CCC"])
        assert not m.loc[:"2024-07-19", "AAA"].any() and m.loc["2024-07-22":, "AAA"].all()
        assert m.loc[:"2024-07-19", "BBB"].all() and not m.loc["2024-07-22":, "BBB"].any()
        assert m["CCC"].all()
    finally:
        data.ndx_changes.cache_clear()


def test_without_a_change_table_the_snapshots_stand(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CHANGES_FILE", tmp_path / "missing.csv")
    data.ndx_changes.cache_clear()
    try:
        idx = pd.bdate_range("2024-06-03", "2024-09-30")
        snap = np.zeros((len(idx), 1), bool)
        snap[:, 0] = idx >= "2024-08-01"
        assert (data.apply_changes(snap.copy(), ["AAA"], idx) == snap).all()
    finally:
        data.ndx_changes.cache_clear()


@needs_data
def test_smci_and_pltr_are_members_from_their_effective_days():
    if data.ndx_changes() is None:
        pytest.skip("no change table")
    idx = data.load("SMCI").index
    idx = idx[(idx >= "2024-07-01") & (idx <= "2025-01-31")]
    m, _ = data.member_mask(["SMCI"], idx)
    s = pd.Series(m[:, 0], index=idx)
    assert not s.loc[:"2024-07-19"].any() and s.loc["2024-07-22":"2024-07-31"].all()
    assert s.loc["2024-12-20"] and not s.loc["2024-12-23":].any()
    if "PLTR" in data.available_tickers():
        pidx = data.load("PLTR").index
        pidx = pidx[(pidx >= "2024-12-01") & (pidx <= "2025-01-15")]
        pm, _ = data.member_mask(["PLTR"], pidx)
        ps = pd.Series(pm[:, 0], index=pidx)
        assert not ps.loc[:"2024-12-20"].any() and ps.loc["2024-12-23":"2024-12-31"].all()
