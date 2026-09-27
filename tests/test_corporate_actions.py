"""Corporate actions booked twice (DHR, EXPE, TMUS), as-traded share units for per-share costs and whole-share
sizing, SEC share counts, market-cap coverage for index rankings, liquidity and survivorship notes."""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, parser, portfolio, report, runner
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy

ROOT = pathlib.Path(__file__).parents[1]


def have(*ts):
    return pytest.mark.skipif(not all((data.PRICES / f"{t}.csv").exists() for t in ts), reason="price data not downloaded")


def one_day_tr(t, day):
    df = data.load(t)
    i = df.index.get_loc(pd.Timestamp(day))
    return float((df["close"].iloc[i] + df["dividend"].iloc[i]) / df["close"].iloc[i - 1] - 1)


def adj_tr(t, day):
    a = data.load(t)["adj_close"]
    i = a.index.get_loc(pd.Timestamp(day))
    return float(a.iloc[i] / a.iloc[i - 1] - 1)


# ------------------------------------------------------------ reconciled corporate actions

@have("DHR", "SPY")
def test_dhr_fortive_spinoff_is_not_counted_twice():
    # Danaher's published figures (investors.danaher.com, Fortive separation, Canadian shareholders): DHR
    # $101.91 before (2016-07-01 high-low average), $78.94 after and $24.56 of FTV per DHR share (2016-07-05)
    truth = (78.94 + 24.56) / 101.91 - 1                                    # +1.6%
    tr = one_day_tr("DHR", "2016-07-05")
    assert abs(tr - truth) < 0.015                                          # was +39.4%
    assert abs(tr - adj_tr("DHR", "2016-07-05")) < 0.01                     # adj_close agrees after reconciling
    fx = data.corporate_action_fixes("DHR")
    row = fx[pd.to_datetime(fx["date"]) == "2016-07-05"].iloc[0]
    assert row["reading"] == "no_split" and row["return_was"] > 0.35
    # the "split" is gone, so prices before it are back on their as-traded basis
    assert pd.Timestamp("2016-07-05") not in data.splits("DHR").index
    q = data.quoted_close("DHR")
    assert q.loc["2016-07-01"] == pytest.approx(101.91, rel=0.03)
    assert q.loc["2016-07-05"] == pytest.approx(78.94, rel=0.03)
    # the engine: buy and hold across the ex-date has no phantom jump
    r = runner.run(parser.parse("buy and hold DHR from 2016-06-01 to 2016-07-31"))
    assert r.equity.pct_change().abs().max() < 0.05
    assert any(n.startswith("Corporate actions:") and "DHR 2016-07-05" in n for n in r.strategy.notes)


@have("EXPE", "TMUS", "SPY")
@pytest.mark.parametrize("t, day", [("EXPE", "2011-12-21"), ("TMUS", "2013-05-01")])
def test_payout_per_pre_split_share(t, day):
    # EXPE: 1-for-2 reverse split, then one TRIP per new share (the data books half a TRIP, $15.125, per old
    # share); TMUS: MetroPCS 1-for-2 reverse split plus $4.06 cash per pre-split share (T-Mobile / SEC 8-K)
    tr = one_day_tr(t, day)
    assert abs(tr) < 0.06                                                    # was -24.7% / -13.1%
    assert abs(tr - adj_tr(t, day)) < 0.01
    fx = data.corporate_action_fixes(t)
    row = fx[pd.to_datetime(fx["date"]) == day].iloc[0]
    assert row["reading"] == "per_presplit" and row["payout_now"] == pytest.approx(2 * row["payout_was"])
    assert data.splits(t).get(pd.Timestamp(day)) == 0.5                     # the reverse split is real


@have("EXPE", "TRIP")
def test_expe_matches_what_a_holder_owned():
    # per old EXPE share after the event: half a new EXPE share plus half a TRIP share (TRIP's own file)
    e, trip = data.quoted_close("EXPE"), data.quoted_close("TRIP")
    before = e.loc["2011-12-20"]                                             # an old share, as quoted
    assert before == pytest.approx(28.55, rel=0.01)
    after = 0.5 * e.loc["2011-12-21"] + 0.5 * trip.loc["2011-12-21"]
    truth = after / before - 1
    assert abs(one_day_tr("EXPE", "2011-12-21") - truth) < 0.06             # the payout is booked at TRIP's prior close


def test_reconcile_synthetic_spinoff_booked_twice(monkeypatch):
    idx = pd.bdate_range("2020-01-01", periods=4)
    # $100 stock spins off $21.30 of value per share; the vendor books a $21.30 payout AND a 100/78.7 "split"
    r = 100 / 78.7
    raw = pd.DataFrame({"open": [78.7, 78.7, 78.7, 79.0], "high": [78.7, 78.7, 79.2, 79.5],
                        "low": [78.7, 78.7, 78.7, 78.7], "close": [78.7, 78.7, 79.2, 79.5],
                        "adj_close": [1.0, 1.0, 79.2 / (78.7 - 21.3), 79.5 / (78.7 - 21.3)],
                        "volume": [1e6] * 4, "dividend": [0, 0, 21.3, 0], "split": [0, 0, r, 0]}, index=idx)
    monkeypatch.setattr(data, "_market_day_returns", lambda: pd.Series(0.0, index=idx))
    out, log = data.reconcile_actions("ZZSPIN", raw)
    assert list(log["reading"]) == ["no_split"]
    assert out["close"].iloc[1] == pytest.approx(100.0)                     # back on the as-traded basis
    assert out["volume"].iloc[1] == pytest.approx(1e6 / r)
    want = (79.2 + 21.3) / 100 - 1
    assert (out["close"].iloc[2] + out["dividend"].iloc[2]) / out["close"].iloc[1] - 1 == pytest.approx(want)
    assert out["adj_close"].iloc[2] / out["adj_close"].iloc[1] - 1 == pytest.approx(want)
    assert out["split"].iloc[2] == 0
    # a genuine 2:1 split with a normal dividend is left alone
    raw2 = raw.copy()
    raw2.loc[idx[2], ["dividend", "split"]] = [0.2, 2.0]
    raw2["close"] = [150, 150, 75.0, 76.0]
    raw2["adj_close"] = [150, 150, 75.0 + 0.2, 76.2]
    assert data.reconcile_actions("ZZSPLIT", raw2)[1].empty


def test_whole_ratio():
    assert all(data._whole_ratio(r) for r in (2, 3, 1.5, 0.5, 7, 4, 1 / 15, 1.25, 10))
    assert not any(data._whole_ratio(r) for r in (1.319, 1.128))


def _all_days():
    rows = []
    for t in data.available_tickers():
        try:
            df = data.load(t)
        except data.DataError:
            continue
        if len(df) < 2:
            continue
        c, d, a = df["close"], df["dividend"], df["adj_close"]
        diff = ((c + d) / c.shift(1) - a / a.shift(1)).abs()
        for day in df.index[(diff > data.CA_TOLERANCE).fillna(False).to_numpy()]:
            rows.append((t, str(day.date()), float(diff[day])))
    return rows


@pytest.mark.skipif(not data.available_tickers(), reason="price data not downloaded")
def test_no_day_disagrees_with_the_total_return_unless_whitelisted():
    # mutual funds' free histories often misreport capital-gains distributions: those are not reconciled but every
    # backtest that holds one across such a day gets a data-quality warning (data.distribution_note), tested below
    from backtester import fund_lists
    bad = [r for r in _all_days() if (r[0], r[1]) not in data.CA_WHITELIST and r[0] not in fund_lists.MUTUAL_FUNDS]
    assert not bad, f"engine total return differs from adj_close by more than 2% (reconcile or whitelist): {bad[:20]}"
    for why in data.CA_WHITELIST.values():
        assert len(why) > 20                                                 # every exception has a reason


@pytest.mark.skipif("PCRAX" not in data.available_tickers(), reason="PCRAX not downloaded")
def test_mutual_fund_distribution_mismatch_is_warned():
    days = data.distribution_mismatch_days("PCRAX")
    assert days, "PCRAX's 2008 distribution disagrees with its adjusted close"
    spec = parser.parse("hold 100% PCRAX from 2008 to 2009")
    runner.run(spec)
    assert any(n.startswith("Warning: data quality") and "PCRAX" in n for n in spec.notes)


# ------------------------------------------------------------ as-traded share units

@have("AAPL")
def test_ibkr_commissions_use_as_traded_shares():
    r = runner.run(parser.parse("buy AAPL when RSI(2) is below 10, hold 3 days, IBKR commissions, from 1995 to 2000"))
    tr = r.trades
    first = tr.iloc[0]
    # AAPL traded around $40 in early 1995 (later splits: 2:1 2000, 2:1 2005, 7:1 2014, 4:1 2020 = 112x)
    assert 30 < first.entry_price < 50 and first.shares < 400                # not ~26,000 "shares" at $0.38
    # IBKR fixed: $0.005 per share as traded, $1 minimum, 1% cap, on each side
    want = (np.minimum(np.maximum(1.0, 0.005 * tr["shares"]), 0.01 * tr["shares"] * tr["entry_price"])
            + np.minimum(np.maximum(1.0, 0.005 * tr["exit_shares"]), 0.01 * tr["exit_shares"] * tr["exit_price"]))
    assert tr["commission"].to_numpy() == pytest.approx(want.to_numpy(), rel=1e-6)
    assert tr["commission"].mean() < 5                                       # was ~$192 a trade
    assert r.equity.iloc[-1] == pytest.approx(10_000 + tr["pnl"].sum() + r.interest, rel=1e-9)
    assert r.equity.iloc[-1] > 3_000                                         # was $851


def synth(rows, split_at=None, ratio=2.0):
    idx = pd.bdate_range("2019-12-30", periods=len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = 1e6
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    df["split"] = 0.0
    if split_at is not None:
        df.iloc[split_at, df.columns.get_loc("split")] = ratio
    return df


@pytest.fixture
def fake(monkeypatch):
    frames = {}
    monkeypatch.setattr(data, "load", lambda t: frames[data.canonical(t)])
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): frames[data.canonical(t)] for t in ts})
    return frames


def test_whole_shares_and_fixed_shares_are_as_traded(fake):
    # split-adjusted $3: a 10:1 split comes later, so the stock traded at $30 on these days
    fake["ZX"] = synth([(3, 3, 3, 3)] * 4 + [(3, 3, 3, 3)], split_at=4, ratio=10.0)
    r = engine.run(Strategy(cash_rate=None, universe=["ZX"], entry="dow == 0", hold_bars=1, fractional_shares=False,
                            capital=1_000))
    t = r.trades.iloc[0]
    assert t.shares == 33 and t.entry_price == pytest.approx(30)            # 33 x $30, not 333 x $3
    assert t.position_value == pytest.approx(990)
    r = engine.run(Strategy(cash_rate=None, universe=["ZX"], entry="dow == 0", hold_bars=1, sizing="fixed_shares",
                            fixed_amount=10, commission_per_share=0.01))
    t = r.trades.iloc[0]
    assert t.shares == pytest.approx(10) and t.position_value == pytest.approx(300)
    assert t.commission == pytest.approx(2 * 0.01 * 10)
    assert r.equity.iloc[-1] == pytest.approx(10_000 + t.pnl)


def test_split_while_held_reports_both_counts(fake):
    # held across a 2:1 split: 100 shares at $20 become 200 at $10 (split-adjusted $10 throughout)
    fake["ZY"] = synth([(10, 10, 10, 10)] * 3 + [(10.5, 10.5, 10.5, 10.5)] * 2, split_at=2, ratio=2.0)
    r = engine.run(Strategy(cash_rate=None, universe=["ZY"], entry="dow == 0", hold_bars=3, capital=2_000))
    t = r.trades.iloc[0]
    assert t.shares == pytest.approx(100) and t.entry_price == pytest.approx(20)
    assert t.exit_shares == pytest.approx(200) and t.exit_price == pytest.approx(10.5)
    assert t.pnl == pytest.approx(t.exit_shares * t.exit_price - t.shares * t.entry_price)


def test_portfolio_whole_shares_are_as_traded(fake):
    fake["ZX"] = synth([(3, 3, 3, 3)] * 5, split_at=4, ratio=10.0)
    r = portfolio.run(Portfolio(tree={"asset": "ZX"}, rebalance="none", cash_rate=None, capital=1_000,
                                fractional_shares=False))
    o = r.orders.iloc[0]
    assert o["shares"] == 33 and o["price"] == pytest.approx(30)


# ------------------------------------------------------------ SEC share counts

def fetch_module():
    pytest.importorskip("yfinance")
    spec = importlib.util.spec_from_file_location("fetch_data_sec", ROOT / "scripts" / "fetch_data.py")
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    return fd


def test_sec_companyfacts_parser():
    fd = fetch_module()
    # a trimmed copy of data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json (filings 2013-06..2015-01)
    facts = json.loads((ROOT / "tests" / "fixtures" / "sec_companyfacts_AAPL_2013_2015.json").read_text())
    s = fd.sec_share_counts(facts)
    assert s.index.is_monotonic_increasing and s.index[0] >= pd.Timestamp("2013-06-01")
    # dated by the filing date, in the units of that date: 7:1 split on 2014-06-09
    assert s.loc["2014-04-24"] == pytest.approx(861_381_000, rel=1e-3)
    assert s.loc["2014-07-23"] == pytest.approx(5_987_867_000, rel=1e-3)
    assert fd.sec_ticker_map({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                              "1": {"cik_str": 1067983, "ticker": "BRK.B", "title": "x"}}) == {"AAPL": 320193, "BRK-B": 1067983}
    ua = fd.SEC_UA["User-Agent"]
    # SEC needs a contact e-mail: the repo's GitHub no-reply address (never a personal one) unless SEC_CONTACT is set
    assert "backtester" in ua and ("SEC_CONTACT" in __import__("os").environ or ua.endswith("@users.noreply.github.com"))


def test_sec_counts_fill_before_yahoo(monkeypatch, tmp_path):
    (tmp_path / "shares").mkdir()
    (tmp_path / "shares_sec").mkdir()
    pd.DataFrame({"date": ["2016-01-28", "2016-04-28", "2017-01-30"], "shares": [5.5e9, 5.5e9, 5.3e9]}) \
        .to_csv(tmp_path / "shares" / "ZQ.csv", index=False)
    pd.DataFrame({"date": ["2010-01-25", "2014-07-23", "2016-04-27", "2016-10-26"], "shares": [9e8, 6e9, 5.6e9, 5.4e9]}) \
        .to_csv(tmp_path / "shares_sec" / "ZQ.csv", index=False)
    monkeypatch.setattr(data, "DATA", tmp_path)
    monkeypatch.setattr(data, "SHARES_SEC", tmp_path / "shares_sec")
    data.shares_outstanding.cache_clear()
    try:
        s = data.shares_outstanding("ZQ")
    finally:
        data.shares_outstanding.cache_clear()
    # SEC before Yahoo's first count and in Yahoo's 2016-04..2017-01 gap (> 120 days); not next to a Yahoo count
    assert list(s.index.strftime("%Y-%m-%d")) == ["2010-01-25", "2014-07-23", "2016-01-28", "2016-04-28",
                                                  "2016-10-26", "2017-01-30"]


# ------------------------------------------------------------ market-cap coverage

def test_mcap_start_waits_for_coverage(monkeypatch):
    idx = pd.bdate_range("2010-01-01", periods=10)
    names = [f"T{k}" for k in range(10)]
    known_from = {t: idx[min(k, 9)] for k, t in enumerate(names)}          # one more name gains a count each day
    monkeypatch.setattr(data, "market_cap", lambda t: pd.Series(np.where(idx >= known_from[t], 1e9, np.nan), index=idx))
    elig = np.ones((10, 10), bool)
    cov = data.mcap_coverage(elig, names, idx)
    assert cov[0] == pytest.approx(0.1) and cov[7] == pytest.approx(0.8)
    d, note = data.mcap_start(elig, names, idx)
    assert d == idx[7] and "too little of the universe before" in note and str(idx[7].date()) in note
    d, note = data.mcap_start(elig[:, :3] & False, names[:3], idx)
    assert d is None and "at most 0%" in note


@have("AAPL", "MSFT", "QQQ")
def test_ndx_top_by_market_cap_starts_when_counts_cover_the_universe(monkeypatch):
    real = data.market_cap
    gate = pd.Timestamp("2019-06-03")

    def mc(t):
        s = real(t)
        if s.empty:
            s = data.load(t)["close"] * 1e9
        return s.where(s.index >= gate)
    monkeypatch.setattr(data, "market_cap", mc)
    p = parser.parse("hold the top 10 Nasdaq 100 stocks by market cap, market cap weighted, rebalance quarterly, "
                     "from 2019-01-01 to 2019-12-31")
    r = runner.run(p)
    assert r.equity.index[1] >= gate
    assert any(n.startswith("Market cap: market-cap data covers too little") for n in p.notes)
    A = report.analyze(r, sensitivity=False, mc=False, detail=False)
    w = A["warnings"][0]
    assert w["code"] == "survivorship" and w["message"].startswith("Survivorship: ") and "biased upward" in w["message"]
    assert "!! Survivorship:" in report.console_summary(A)


# ------------------------------------------------------------ notes

def test_liquidity_ignores_stale_bars(fake, monkeypatch):
    idx = pd.bdate_range("2012-01-02", periods=300)
    c = np.r_[np.full(200, 10.0), 50 + np.arange(100) * 0.1]
    v = np.r_[np.zeros(200), np.full(100, 1e6)]
    fake["ZF"] = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": v, "adj_close": c,
                               "dividend": 0.0}, index=idx)
    monkeypatch.setattr(data, "membership", lambda: None)
    assert not [n for n in data.identity_notes("ZF") if n.startswith("Liquidity:")]
    fake["ZF"]["volume"] = np.r_[np.zeros(200), np.full(100, 10.0)]
    assert [n for n in data.identity_notes("ZF") if n.startswith("Liquidity:")]


@pytest.mark.skipif(data.membership() is None, reason="membership data not downloaded")
def test_missing_members_and_tiingo_hint():
    # EA's history was restored from archives (data/delisted_sources.json), so it is no longer missing
    assert "history" not in data.delisted()["EA"] and data.load("EA").index[0].year <= 1990
    assert "EA" not in dict(data.missing_members("2013-01-01", "2020-12-31", n=12))
    # companies no free archive carries are still named, with the Tiingo hint
    miss = dict(data.missing_members("2004-01-01", "2008-12-31", n=12))
    assert {"PIXR", "SEBL", "MERQ"} <= set(miss)
    note = data.coverage_note("2004-01-01", "2008-12-31")
    assert "Biggest missing members" in note and "TIINGO_API_KEY" in note and "PIXR (" in note
