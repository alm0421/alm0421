"""Scalable price-data checks: security breaks (bankruptcy re-listing splices: detection, forced close, indicators that
do not span them, truncation invariance), untraded stale runs repaired as bad ticks, and the price-flag file
(data/price_flags.json: kept up to date with the price files, warnings in backtests held across a flagged day)."""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, integrity, parser, portfolio, price_flags, runner
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy


def have(*ts):
    return pytest.mark.skipif(not all((data.PRICES / f"{t}.csv").exists() for t in ts), reason="price data not downloaded")


# ------------------------------------------------------------------ synthetic splices

N_OLD, N_NEW = 300, 200


def splice(jump_to: float = 25.0, new_volume: float = 5e5, seed: int = 3, gap_days: int = 0,
           volume: bool = True) -> pd.DataFrame:
    """An old security that collapses from ~$20 to $0.10, then (bar N_OLD on) a new one around `jump_to`."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=N_OLD + N_NEW)
    if gap_days:
        idx = idx[:N_OLD].append(idx[N_OLD:] + pd.Timedelta(days=gap_days))
    old = np.r_[20 * np.exp(np.cumsum(rng.normal(0, 0.01, 150))), np.geomspace(18, 0.1, N_OLD - 150)]
    new = jump_to * np.exp(np.cumsum(rng.normal(0, 0.01, N_NEW)))
    c = np.r_[old, new]
    v = np.r_[np.full(N_OLD, 5e6), np.full(N_NEW, new_volume)] if volume else np.zeros(N_OLD + N_NEW)
    df = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": v, "dividend": 0.0,
                       "adj_close": c, "quote_close": c, "split": 1.0, "open_ok": True}, index=idx)
    return df


def test_find_breaks_on_a_synthetic_splice_is_causal():
    df = splice()
    b = integrity.find_breaks("ZB", df)
    assert list(b["date"]) == [df.index[N_OLD]]
    assert b["source"].iloc[0] == "detected" and b["jump"].iloc[0] > 100
    # decided from the bars up to the break day only: the same with the data cut at any day
    for cut in (N_OLD - 1, N_OLD, N_OLD + 1, N_OLD + 30):
        got = list(integrity.find_breaks("ZB", df.iloc[:cut + 1])["date"])
        assert got == ([df.index[N_OLD]] if cut >= N_OLD else [])


def test_what_is_not_a_break():
    # a 1-for-20 reverse split the data missed: 20x on share volume / 20 (the integrity gate's case, not a new security)
    assert integrity.find_breaks("ZB", splice(jump_to=2.0, new_volume=5e6 / 20)).empty
    # news on a penny stock: 12x on 30x the volume
    assert integrity.find_breaks("ZB", splice(jump_to=1.2, new_volume=1.5e8)).empty
    # a split booked that day
    df = splice()
    df.iloc[N_OLD, df.columns.get_loc("split")] = 1 / 200
    assert integrity.find_breaks("ZB", df).empty
    # no volume and no gap: nothing but the price to go on (flagged instead); with a 3-week gap: the old line stopped
    assert integrity.find_breaks("ZB", splice(volume=False)).empty
    assert len(integrity.find_breaks("ZB", splice(volume=False, gap_days=21))) == 1
    # a stock that never collapsed does not "splice" by jumping (a real 10x day is flagged, not closed out)
    df = splice()
    df.iloc[:N_OLD, df.columns.get_loc("close")] = 2.0
    assert integrity.find_breaks("ZB", df).empty
    # overrides decide known cases either way
    day = str(splice().index[N_OLD].date())
    assert integrity.find_breaks("ZB", splice(), {day: {"action": "no_break"}}).empty
    forced = integrity.find_breaks("ZB", splice(jump_to=2.0, new_volume=5e6 / 20), {day: {"action": "break", "why": "x"}})
    assert list(forced["source"]) == ["override"]


@pytest.fixture
def fake(monkeypatch):
    frames = {}
    monkeypatch.setattr(data, "load", lambda t: frames[data.canonical(t)])
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): frames[data.canonical(t)] for t in ts})
    return frames


def test_engine_closes_a_position_held_into_a_break_at_the_old_price(fake):
    fake["ZB"] = df = splice()
    b = df.index[N_OLD]
    r = engine.run(Strategy(cash_rate=None, universe=["ZB"], entry="year == 2015", hold_bars=10_000, commission=5.0,
                            slippage_bps=10))
    t = r.trades.iloc[0]
    assert t.exit_reason == "security break" and pd.Timestamp(t.exit_date) == b
    assert t.exit_price == pytest.approx(df["close"].iloc[N_OLD - 1])       # the old shares' last close, no slippage
    assert t.commission == pytest.approx(5.0)                               # the entry's only: the settlement is free
    # no 250x "return"; the equity on the break day is the settled cash, and the books balance
    assert r.equity.pct_change().abs().max() < 0.5
    assert r.equity.loc[b] == pytest.approx(r.equity.loc[df.index[N_OLD - 1]])
    assert r.equity.iloc[-1] == pytest.approx(10_000 + r.trades["pnl"].sum() + r.interest, rel=1e-9)
    assert any(n.startswith("Security break: ZB on") for n in r.strategy.notes)


def test_indicators_do_not_span_a_break(fake):
    fake["ZB"] = df = splice()
    b = N_OLD
    r = engine.run(Strategy(cash_rate=None, universe=["ZB"], entry="sma(close, 20) > 0", hold_bars=3))
    entries = pd.to_datetime(r.trades["entry_date"])
    after = entries[entries >= df.index[b]]
    # a 20-bar average of the new security needs 20 of its own bars: none spanning the old one's
    assert after.min() == df.index[b + 19]
    # the old security traded up to the day before the break
    assert (entries < df.index[b]).any()


def _cut(df: pd.DataFrame, d) -> pd.DataFrame:
    """The data truncated at d (the bars up to and including it)."""
    return df.loc[df.index <= pd.Timestamp(d)]


@pytest.mark.parametrize("k", [-1, 0, 1, 5])
def test_truncation_invariance_around_a_break(fake, k):
    # the CLAUDE.md invariant: the data truncated at D gives the same trades and equity before D - with D the day
    # before the break (nothing is known yet), the break day itself (known from its own first print) and after it
    full_df = splice()
    D = full_df.index[N_OLD + k]
    rules = dict(cash_rate=None, universe=["ZB"], entry="close > ref(close, 1)", hold_bars=8, commission=1.0)
    fake["ZB"] = full_df
    full = engine.run(Strategy(**rules))
    fake["ZB"] = _cut(full_df, D)
    cut = engine.run(Strategy(**rules))
    pd.testing.assert_series_equal(full.equity[full.equity.index < D], cut.equity[cut.equity.index < D])
    a = full.trades[pd.to_datetime(full.trades["exit_date"]) < D].reset_index(drop=True)
    c = cut.trades[pd.to_datetime(cut.trades["exit_date"]) < D].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["entry_date", "exit_date", "pnl"]], c[["entry_date", "exit_date", "pnl"]])
    # the same for an allocation
    fake["ZB"] = full_df
    pf = portfolio.run(Portfolio(tree={"asset": "ZB"}, rebalance="monthly", cash_rate=None))
    fake["ZB"] = _cut(full_df, D)
    pc = portfolio.run(Portfolio(tree={"asset": "ZB"}, rebalance="monthly", cash_rate=None))
    pd.testing.assert_series_equal(pf.equity[pf.equity.index < D], pc.equity[pc.equity.index < D])


def test_portfolio_settles_the_old_holding_and_rebuys_the_new_security(fake):
    fake["ZB"] = df = splice()
    b = df.index[N_OLD]
    r = portfolio.run(Portfolio(tree={"asset": "ZB"}, rebalance="none", cash_rate=None))
    o = r.orders
    brk = o[o["reason"] == "security break"].iloc[0]
    assert pd.Timestamp(brk["date"]) == b and brk["price"] == pytest.approx(df["close"].iloc[N_OLD - 1])
    assert brk["commission"] == 0
    assert r.equity.pct_change().abs().max() < 0.5
    assert r.equity.loc[b:].nunique() == 1                                  # held as cash from then on (no rebalance)
    # a monthly rebalance buys the new security at the next month-end, at its own price
    r = portfolio.run(Portfolio(tree={"asset": "ZB"}, rebalance="monthly", cash_rate=None))
    buys = r.orders[(r.orders["side"] == "buy") & (pd.to_datetime(r.orders["date"]) > b)]
    assert len(buys) and buys["price"].iloc[0] > 20
    assert r.equity.pct_change().abs().max() < 0.5


# ------------------------------------------------------------------ CHRD (Oasis Petroleum's Chapter 11)

@have("CHRD", "SPY")
def test_chrd_bankruptcy_splice_is_a_security_break():
    df = data.load("CHRD")
    found = integrity.find_breaks("CHRD", df)                              # detected without the curated override
    assert list(pd.to_datetime(found["date"]).astype(str)) == ["2020-11-20"]
    assert list(data.security_breaks("CHRD").astype(str)) == ["2020-11-20"]
    # a buy-and-hold across it: settled at the old shares' $0.12, no 258x gain
    s = parser.parse("buy and hold CHRD from 2020-06-01 to 2021-01-31")
    r = runner.run(s)
    assert r.equity.pct_change().abs().max() < 3
    assert r.equity.iloc[-1] < r.equity.iloc[0]
    o = r.orders[r.orders["reason"] == "security break"]
    assert len(o) == 1 and o["price"].iloc[0] == pytest.approx(0.12)
    assert any(n.startswith("Security break: CHRD on 2020-11-20") for n in s.notes)
    # a signal strategy: the trade held into it exits there, at the old price; the books balance
    s = parser.parse("buy CHRD when RSI(2) is below 30, hold 60 days, from 2020-06-01 to 2021-06-30")
    r = runner.run(s)
    tr = r.trades
    br = tr[tr["exit_reason"] == "security break"]
    assert len(br) == 1 and str(br["exit_date"].iloc[0]) == "2020-11-20" and br["exit_price"].iloc[0] == pytest.approx(0.12)
    assert r.equity.pct_change().abs().max() < 3
    assert r.equity.iloc[-1] == pytest.approx(10_000 + tr["pnl"].sum() + r.interest, rel=1e-9)
    # truncation: ending the run on either side of the break changes nothing before it
    rules = "buy CHRD when RSI(2) is below 30, hold 60 days, from 2020-06-01 to {end}"
    full = runner.run(parser.parse(rules.format(end="2021-06-30")))
    for end, D in (("2020-11-19", "2020-11-19"), ("2020-11-20", "2020-11-20"), ("2020-11-24", "2020-11-24")):
        cut = runner.run(parser.parse(rules.format(end=end)))
        pd.testing.assert_series_equal(full.equity[full.equity.index < D], cut.equity[cut.equity.index < D])


# ------------------------------------------------------------------ stale runs (bad ticks lasting 2-3 bars)

def test_untraded_stale_run_is_repaired_but_a_traded_one_is_not():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2016-01-01", periods=200)
    c = 40 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    raw = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "adj_close": c,
                        "volume": 1e6, "dividend": 0.0, "split": 0.0}, index=idx)
    bad = raw.copy()
    for col in ("open", "high", "low", "close", "adj_close"):
        bad.iloc[120:122, bad.columns.get_loc(col)] = c[119] / 6             # two quotes at 1/6 of the price ...
    bad.iloc[120:122, bad.columns.get_loc("volume")] = 0.0                    # ... on which nothing traded
    out, ev = integrity.check("ZZZ", bad, lambda t: None, fund=False)
    assert list(ev["kind"]) == ["bad_tick", "bad_tick"] and list(ev["date"]) == list(idx[120:122])
    assert np.log(out["close"]).diff().abs().max() < 0.1
    traded = bad.copy()
    traded.iloc[120:122, traded.columns.get_loc("volume")] = 5e6               # real trading: a real (wild) move
    assert integrity.check("ZZZ", traded, lambda t: None, fund=False)[1].empty


@have("INDV", "SPY")
def test_indv_untraded_quotes_repaired():
    ev = data.price_repairs("INDV")
    days = set(pd.to_datetime(ev.loc[ev["kind"] == "bad_tick", "date"]).astype(str))
    assert {"2022-11-23", "2022-11-25"} <= days
    c = data.load("INDV")["close"]
    assert abs(c.loc["2022-11-28"] / c.loc["2022-11-22"] - 1) < 0.1
    assert not data.load("INDV")["open_ok"].loc["2022-11-23"]


# ------------------------------------------------------------------ the flags file and the warning

@pytest.mark.skipif(not data.available_tickers(), reason="price data not downloaded")
def test_price_flags_file_is_up_to_date_with_the_price_files():
    # a data refresh that forgets to regenerate the flags fails here (scripts/fetch_data.py does it after each run)
    doc = price_flags.read()
    assert doc.get("version") == price_flags.VERSION, "data/price_flags.json missing: python -m backtester.price_flags"
    stale, gone = price_flags.stale(doc)
    assert not stale and not gone, ("data/price_flags.json does not match the price files (regenerate: python -m "
                                    f"backtester.price_flags): stale {stale[:20]}, deleted {gone[:20]}")
    for t, e in doc["files"].items():
        for f in e.get("flags", []):
            assert set(f) >= {"date", "kinds", "move", "gap"} and f["kinds"], (t, f)


@have("CLSK")
def test_backtests_are_warned_across_flagged_days():
    assert any(str(f["date"].date()) == "2018-09-19" for f in price_flags.flags("CLSK"))
    s = parser.parse("hold 100% CLSK, rebalance monthly, from 2018-06-01 to 2018-12-31")
    runner.run(s)
    w = [n for n in s.notes if n.startswith("Warning: unexplained price moves")]
    assert w and "CLSK 2018-09-19" in w[0] and "+387%" in w[0]
    s = parser.parse("hold 100% CLSK, rebalance monthly, from 2019-01-01 to 2019-12-31")     # nothing flagged then
    runner.run(s)
    assert not [n for n in s.notes if n.startswith("Warning: unexplained price moves")]


def test_flag_note_names_only_days_held_across(monkeypatch):
    idx = pd.bdate_range("2021-01-04", periods=10)
    monkeypatch.setattr(price_flags, "flags", lambda t: [{"date": idx[5], "kinds": ["move"], "move": 2.5, "gap": 0.0}]
                        if t == "ZF" else [])
    held = pd.DataFrame({"ZF": [0, 0, 1, 1, 1, 1, 1, 0, 0, 0.0], "ZG": 1.0}, index=idx)
    n = price_flags.flag_note(held)
    assert n.startswith("Warning: unexplained price moves") and f"ZF {idx[5].date()} (+250% in one day)" in n
    held["ZF"] = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0.0]            # bought at the flagged day's close: not held across it
    assert price_flags.flag_note(held) is None


@pytest.fixture
def tmp_prices(tmp_path, monkeypatch):
    prices = tmp_path / "prices"
    prices.mkdir()
    monkeypatch.setattr(data, "PRICES", prices)
    data.clear_caches()
    yield prices
    monkeypatch.undo()
    data.clear_caches()


@have("CHRD", "CLSK", "SPY")
def test_write_regenerates_only_what_changed(tmp_prices, monkeypatch):
    real = data.ROOT / "data" / "prices"
    for t in ("CHRD", "CLSK", "SPY"):
        shutil.copy(real / f"{t}.csv", tmp_prices / f"{t}.csv")
    doc = price_flags.write(quiet=True)
    assert (tmp_prices.parent / "price_flags.json").exists() and set(doc["files"]) == {"CHRD", "CLSK", "SPY"}
    assert doc["files"]["CHRD"]["breaks"][0]["date"] == "2020-11-20" and doc["files"]["CLSK"]["flags"]
    assert price_flags.stale() == ([], [])
    # a refresh rewrites CLSK and deletes SPY: exactly those are stale, and write() rescans only them
    with open(tmp_prices / "CLSK.csv", "a") as f:
        f.write("2099-01-02,1,1,1,1,1,0,0,0\n")
    (tmp_prices / "SPY.csv").unlink()
    data.clear_caches()
    assert price_flags.stale() == (["CLSK"], ["SPY"])
    scanned = []
    real_scan = price_flags.scan
    monkeypatch.setattr(price_flags, "scan", lambda t: scanned.append(t) or real_scan(t))
    doc = price_flags.write(quiet=True)
    assert scanned == ["CLSK"] and set(doc["files"]) == {"CHRD", "CLSK"}
    assert price_flags.stale() == ([], [])
