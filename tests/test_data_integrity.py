"""Market cap on the right basis, the data job never losing histories, bar repairs and identity warnings."""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from backtester import data, runner
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy

ROOT = pathlib.Path(__file__).parents[1]
HAVE = (data.PRICES / "AAPL.csv").exists() and (data.DATA / "shares" / "AAPL.csv").exists()
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def fetch_module():
    pytest.importorskip("yfinance")
    spec = importlib.util.spec_from_file_location("fetch_data", ROOT / "scripts" / "fetch_data.py")
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    return fd


# ------------------------------------------------------------ market cap

def cap(t, d):
    return float(data.market_cap(t).asof(pd.Timestamp(d)))


@needs_data
def test_market_caps_are_quoted_price_times_point_in_time_shares():
    assert 1.6e12 <= cap("AMZN", "2021-06-30") <= 1.8e12
    assert 2.0e12 <= cap("NVDA", "2024-03-28") <= 2.45e12
    assert 2.85e12 <= cap("AAPL", "2023-12-29") <= 3.05e12
    aep = data.market_cap("AEP")
    assert aep.loc["2021"].max() < 60e9 and aep.max() < 100e9    # a utility, not a $1.4T company
    # right after a split Yahoo still reports pre-split share counts for a few weeks: put on the right basis
    assert 1.8e12 <= cap("AAPL", "2020-09-15") <= 2.2e12
    assert 3.5e11 <= cap("TSLA", "2020-09-15") <= 4.8e11


@needs_data
def test_market_cap_has_no_split_or_dividend_artifacts():
    for t in ("AAPL", "AMZN", "GOOG", "WMT", "AEP", "MSFT"):
        mc = data.market_cap(t).dropna()
        assert len(mc) > 500
        assert (np.log(mc).diff().dropna().abs() < 0.2).all(), t      # no day where it jumps by a split ratio


@needs_data
def test_market_cap_is_point_in_time():
    sh = data.shares_adjusted("AAPL")
    d = sh.index[len(sh) // 2]
    mc = data.market_cap("AAPL")
    day = mc.index[mc.index.searchsorted(d)]          # the session on or after the report date
    if day == d:
        # a count dated d is only known from the next session
        prev_count = sh[sh.index < d].iloc[-1]
        assert mc[day] == pytest.approx(data.load("AAPL")["close"][day] * prev_count)


@needs_data
def test_top5_nasdaq100_by_market_cap_2021():
    p = Portfolio(tree={"filter": {"select": "top", "n": 5, "by": "market_cap", "weights": "market_cap"}, "universe": "NDX"},
                  rebalance="monthly", start="2021-01-01", end="2021-12-31")
    h = runner.run(p).holdings
    alphabet = 0
    for d in ("2021-02-01", "2021-07-01", "2021-12-01"):
        row = h.loc[h.index >= d].iloc[0]
        held = set(row[row > 1e-6].index)
        assert len(held) == 5
        assert {"AAPL", "MSFT", "AMZN"} <= held
        assert held <= {"AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "FB", "META", "TSLA", "NVDA"}, held
        assert "AEP" not in held and "EXC" not in held
        alphabet += bool(held & {"GOOG", "GOOGL"})
    # each Alphabet class is about half the company (~$0.7-0.9T in 2021), close to TSLA and FB
    assert alphabet >= 2
    assert any(data.MCAP_NOTE == n for n in p.notes)


@needs_data
def test_market_cap_does_not_depend_on_price_basis():
    picks = []
    for basis in ("adjusted", "quoted"):
        p = Portfolio(tree={"filter": {"select": "top", "n": 3, "by": "market_cap", "weights": "equal"},
                            "children": [{"asset": t} for t in ("AAPL", "AEP", "XEL", "EXC", "MSFT")]},
                      rebalance="monthly", start="2021-01-01", end="2021-06-30", price_basis=basis)
        h = runner.run(p).holdings
        picks.append(tuple(sorted(h.iloc[-1][h.iloc[-1] > 0].index)))
    assert picks[0] == picks[1]
    assert "AAPL" in picks[0] and "MSFT" in picks[0]


# ------------------------------------------------------------ the data job keeps histories

def frame(dates, closes, adj=None):
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    c = np.asarray(closes, float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "adj_close": c if adj is None else adj,
                         "volume": 1e6, "dividend": 0.0, "split": 0.0}, index=idx)


def test_merge_keeps_old_history_when_download_is_truncated():
    fd = fetch_module()
    days = pd.bdate_range("2020-01-01", periods=300)
    old = frame(days, np.linspace(100, 200, 300))
    # a single bar after the saved history (how Yahoo returned EA after it was taken private)
    new = frame([days[-1] + pd.offsets.BDay(1)], [201.0])
    out, how = fd.merge_history("EA", new, old)
    assert how == "appended" and len(out) == 301 and out.index[0] == days[0]
    # a later start overlapping the file on a new split basis (2:1): old rows rescaled and kept
    new = frame(days[200:], np.linspace(100, 200, 300)[200:] / 2)
    out, how = fd.merge_history("X", new, old)
    assert how == "spliced" and len(out) == 300
    assert out["close"].iloc[0] == pytest.approx(50.0) and out["volume"].iloc[0] == pytest.approx(2e6)
    # an overlap that disagrees (another company): the saved file wins
    new = frame(days[200:], np.random.default_rng(0).uniform(1, 2, 100))
    out, how = fd.merge_history("X", new, old)
    assert how.startswith("kept-old") and out is old
    # a full refresh replaces the file
    out, how = fd.merge_history("X", frame(days, np.linspace(100, 200, 300)), old)
    assert how == "new"


def test_fetch_job_never_deletes_price_files():
    src = (ROOT / "scripts" / "fetch_data.py").read_text()
    assert ".unlink(" not in src


def test_fallback_history_identity_check():
    fd = fetch_module()
    months = [str(p) for p in pd.period_range("2008-01", "2012-12", freq="M")]
    days = pd.bdate_range("2006-01-01", "2014-12-31")
    good = frame(days, np.full(len(days), 40.0))
    assert fd.plausible_member_series("X", good, months) is None
    later = frame(days[days > "2013-06-01"], np.full((days > "2013-06-01").sum(), 40.0))
    assert "covers only" in fd.plausible_member_series("X", later, months)
    penny = frame(days, np.full(len(days), 0.01))
    assert "dollar volume" in fd.plausible_member_series("X", penny, months)


@needs_data
def test_ea_history_is_kept_and_marked_delisted():
    assert (data.PRICES / "EA.csv").exists()
    info = json.loads((data.DATA / "delisted.json").read_text())["EA"]
    assert info["last_date"] == "2026-08-04" and "acquired" in info["reason"]


# ------------------------------------------------------------ bar sanity pass

@needs_data
def test_goog_bad_open_on_2014_04_02_is_repaired_and_flagged():
    g = data.load("GOOG")
    d = pd.Timestamp("2014-04-02")
    assert 28.0 < g.at[d, "open"] < 28.8          # was 29.92, 6% above anything that traded
    assert g.at[d, "high"] < 29.0
    assert not g.at[d, "open_ok"]
    assert g.at[pd.Timestamp("2014-04-03"), "open_ok"]


def test_repair_bars_is_causal_and_clips_opens_outside_the_range():
    days = pd.bdate_range("2021-01-01", periods=10)
    raw = frame(days, np.linspace(10, 11, 10))
    raw["high"] = raw["close"] * 1.01
    raw["low"] = raw["close"] * 0.99
    raw.loc[days[5], "open"] = raw.loc[days[5], "close"] * 1.2
    fixed, mask = data.repair_bars("ZZZTEST", raw)
    assert mask[days[5]] and mask.sum() == 1
    assert fixed.loc[days[5], "open"] == pytest.approx(raw.loc[days[5], "high"])
    part, _ = data.repair_bars("ZZZTEST", raw.iloc[:6])
    pd.testing.assert_frame_equal(part, fixed.iloc[:6])


@needs_data
def test_open_anomalies_report():
    rep = data.open_anomalies("GOOG", jump=0.05)
    assert isinstance(rep, pd.DataFrame) and set(rep.columns) >= {"date", "open", "prev_close", "next_open"}


# ------------------------------------------------------------ identity warnings

@needs_data
def test_identity_notes_for_recycled_symbols():
    assert any("later company" in n for n in data.identity_notes("CPWR", "2005-01-01", "2005-12-31"))
    assert any("thin series" in n for n in data.identity_notes("CPWR", "2018-01-01", "2020-01-01"))
    assert any("2012-01-09" in n for n in data.identity_notes("MNST", "2008-01-01", "2012-12-31"))
    assert data.identity_notes("MNST", "2015-01-01", "2020-01-01") == []
    assert data.identity_notes("SPOT", "2019-01-01", "2020-01-01") == []    # Spotify, as meant
    assert data.identity_notes("AAPL", "2000-01-01", "2020-01-01") == []
    assert any(n.startswith("Delisted: EA") for n in data.identity_notes("EA", "2026-01-01", "2026-09-01"))


@needs_data
def test_runner_adds_identity_note():
    s = Strategy(universe=["MNST"], entry="close > sma(close, 50)", hold_bars=5, start="2009-01-01", end="2010-12-31")
    runner.run(s)
    assert any(n.startswith("Identity: MNST") for n in s.notes)


def test_fetch_job_keeps_files_when_downloads_fail(tmp_path, monkeypatch):
    """A whole run of the data job with the network mocked: failed and truncated downloads never remove or
    shorten a saved history, and ended histories are listed in delisted.json."""
    fd = fetch_module()
    prices = tmp_path / "data" / "prices"
    prices.mkdir(parents=True)
    days = pd.bdate_range("2020-01-01", "2026-08-04")
    frame(days, np.linspace(50, 210, len(days))).to_csv(prices / "EA.csv")        # taken private
    frame(days, np.linspace(10, 20, len(days))).to_csv(prices / "OLDCO.csv")      # a former member
    frame(days, np.linspace(10, 20, len(days))).to_csv(prices / "JUNK.csv")       # referenced by nobody
    full = pd.bdate_range("2020-01-01", "2026-09-25")
    monkeypatch.setattr(fd, "ROOT", tmp_path)
    monkeypatch.setattr(fd, "PRICES", prices)
    monkeypatch.setattr(fd, "DELISTED_FILE", tmp_path / "data" / "delisted.json")
    monkeypatch.setattr(fd, "KEYED_FILE", tmp_path / "data" / "delisted_sources.json")
    monkeypatch.setattr(fd, "EXTRA", ["SPY", "QQQ"])
    monkeypatch.setattr(fd, "constituents", lambda: (["AAPL", "MSFT"], "test"))
    mem = pd.DataFrame({"month": ["2026-07"], "revid": ["1"], "timestamp": [""], "count": ["4"],
                        "tickers": ["AAPL EA MSFT OLDCO"], "parser": ["3"]})
    monkeypatch.setattr(fd, "update_membership", lambda: mem)
    monkeypatch.setattr(fd, "MEMBERSHIP", tmp_path / "data" / "ndx_membership.csv")

    def fake_fetch(t, tries=3):
        if t in ("SPY", "QQQ", "AAPL", "MSFT"):
            return frame(full, np.linspace(100, 200, len(full)))
        if t == "EA":
            return frame([pd.Timestamp("2026-08-04")], [209.7])   # Yahoo's reset history
        return None
    monkeypatch.setattr(fd, "fetch_with_retry", fake_fetch)
    for name in ("fetch_macro", "fetch_factors"):
        monkeypatch.setattr(fd, name, lambda: None)
    monkeypatch.setattr(fd, "fetch_shares", lambda ts: None)
    monkeypatch.setattr(fd, "fetch_sec_shares", lambda ts: None)
    monkeypatch.setattr(fd, "build_sims", lambda: [])
    monkeypatch.setattr(fd, "fetch_delisted_keyed", lambda t: None)
    monkeypatch.setattr(fd, "fetch_stooq", lambda t: None)
    fd.main()
    for t in ("EA", "OLDCO", "JUNK"):
        assert (prices / f"{t}.csv").exists()
    ea = pd.read_csv(prices / "EA.csv")
    assert len(ea) == len(days)                     # not replaced by the single bar
    dl = json.loads((tmp_path / "data" / "delisted.json").read_text())
    assert dl["EA"]["last_date"] == "2026-08-04" and "acquired" in dl["EA"]["reason"]
    assert dl["OLDCO"]["last_date"] == "2026-08-04"
    meta = json.loads((tmp_path / "data" / "universe.json").read_text())
    assert {"EA", "OLDCO"} <= set(meta["former_members"])
