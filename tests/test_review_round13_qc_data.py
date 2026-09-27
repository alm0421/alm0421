"""QuantConnect reviewer, data round: market-cap share counts (units errors, missing mega-caps, a size-aware coverage
gate), fake delistings of truncated rebuilt histories, multi-bar level-shift junk, and price-only rebuilt histories."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import data, parser, runner

needs = lambda *t: pytest.mark.skipif(not all((data.PRICES / f"{x}.csv").exists() for x in t),  # noqa: E731
                                      reason="no price data")


# ------------------------------------------------------------ 1. share counts

def test_clean_share_counts_rescales_units_errors_and_drops_junk():
    idx = pd.to_datetime(["2010-01-05", "2010-04-05", "2010-07-05", "2010-10-05", "2011-01-05", "2011-04-05",
                          "2011-07-05", "2011-10-05"])
    v = np.array([300e6, 301e6, 300e9, 299e9, 100.0, 298e6, 297e6, 296e6])   # two counts in thousands, a shell's 100
    clean, problems = data.clean_share_counts("ZZNOFILE", pd.Series(v, index=idx))
    assert list(clean.round(-3)) == [300e6, 301e6, 300e6, 299e6, 298e6, 297e6, 296e6]
    acts = {d.date().isoformat(): what for d, _, what in problems}
    assert acts["2010-07-05"].startswith("rescaled by 1/1000") and acts["2010-10-05"].startswith("rescaled by 1/1000")
    assert acts["2011-01-05"].startswith("dropped") and len(problems) == 3


def test_clean_share_counts_keeps_a_count_reported_on_the_other_side_of_a_split(monkeypatch):
    idx = pd.to_datetime(["2024-01-05", "2024-04-05", "2024-06-08", "2024-07-05", "2024-10-05"])
    v = np.array([2.46e9, 2.46e9, 24.6e9, 24.6e9, 24.5e9])   # NVDA-like: reported post-split two days before it
    monkeypatch.setattr(data, "splits", lambda t: pd.Series([10.0], index=pd.to_datetime(["2024-06-10"])))
    monkeypatch.setattr(data, "price_path", lambda t: data.PRICES / "SPY.csv")   # "has a price file"
    monkeypatch.setattr(data, "quoted_close", lambda t: pd.Series(dtype=float))
    clean, problems = data.clean_share_counts("ZZSPLIT", pd.Series(v, index=idx))
    assert problems == [] and len(clean) == 5


@needs("MXIM")
def test_mxim_market_cap_is_not_a_thousand_times_too_big():
    mc = data.market_cap("MXIM")
    for d in ("2011-02-15", "2011-06-30", "2011-09-30"):
        assert 4e9 < mc.asof(pd.Timestamp(d)) < 20e9, d


def test_sec_share_count_files_are_clean():
    bad = {}
    for p in sorted(data.SHARES_SEC.glob("*.csv")):
        pr = [x for x in data.share_count_problems(p.stem) if x[0] == "sec"]
        if pr:
            bad[p.stem] = pr
    assert bad == {}


@needs("GOOG", "GOOGL", "FB")
def test_google_and_facebook_market_caps_reach_back():
    for t in ("GOOG", "GOOGL"):
        mc = data.market_cap(t)
        assert mc.dropna().index[0] <= pd.Timestamp("2009-08-10")
        company = mc.asof(pd.Timestamp("2011-06-30")) * data._class_divisor(t)
        assert 130e9 < company < 200e9                     # Google: about $163B at the end of June 2011
        assert mc.loc["2014-03-03":"2014-04-30"].notna().all()
    fb = data.market_cap("FB")
    assert fb.dropna().index[0] <= pd.Timestamp("2012-08-02")
    assert 100e9 < fb.asof(pd.Timestamp("2013-12-31")) < 170e9   # about $139B


@needs("GOOG")
def test_goog_stays_a_member_through_the_class_c_listing():
    idx = pd.bdate_range("2014-02-20", "2014-04-10")
    m, _ = data.member_mask(["GOOG"], idx)
    assert m[:, 0].all()


@needs("VOD", "BIDU", "TEVA", "AVGO", "NWSA-2013")
def test_foreign_and_former_members_have_counts():
    for t, d, lo, hi in (("VOD", "2012-06-29", 100e9, 180e9), ("BIDU", "2012-06-29", 25e9, 60e9),
                         ("TEVA", "2011-06-30", 35e9, 60e9), ("AVGO", "2013-06-28", 6e9, 15e9),
                         ("NWSA-2013", "2012-12-31", 40e9, 75e9)):
        v = data.market_cap(t).asof(pd.Timestamp(d))
        assert lo < v < hi, (t, v)


def test_mcap_start_waits_for_the_largest_members(monkeypatch):
    idx = pd.bdate_range("2010-01-01", periods=10)
    names = [f"T{k}" for k in range(10)]
    # nine small names known from day 0; the giant T9 (half the index) only from day 6
    known = {t: idx[0] for t in names[:9]} | {"T9": idx[6]}
    monkeypatch.setattr(data, "market_cap", lambda t: pd.Series(np.where(idx >= known[t], 1e9, np.nan), index=idx))
    monkeypatch.setattr(data, "estimated_market_cap",
                        lambda t: pd.Series(9e9 if t == "T9" else 1e9, index=idx))
    elig = np.ones((10, 10), bool)
    d, note = data.mcap_start(elig, names, idx)
    assert d == idx[6]
    assert "large members have none: T9" in note and "estimated value" in note
    w = data.mcap_gap_warning(elig, names, idx, 3)
    assert w.startswith("Warning: Market cap: probable top-3 members") and "T9 (2010-01-01..2010-01-08" in w
    assert data.mcap_gap_warning(elig[6:], names, idx[6:], 3) is None


@needs("AAPL", "MSFT", "GOOG", "FB", "MXIM")
def test_top10_nasdaq100_by_market_cap_holds_the_mega_caps():
    p = parser.parse("hold the top 10 Nasdaq 100 stocks by market cap, rebalance monthly, from 2005-01-01 to 2015-12-31")
    r = runner.run(p)
    start = r.equity.index[1]
    assert pd.Timestamp("2010-06-01") <= start <= pd.Timestamp("2011-01-31")
    assert any("market-cap data covers too little of the universe before" in n for n in p.notes)
    h = r.holdings.drop(columns="cash", errors="ignore")
    held = lambda day: set(h.loc[:day].iloc[-1][lambda s: s > 0.01].index)  # noqa: E731
    for day in ("2011-10-10", "2012-06-11", "2013-06-10"):
        s = held(day)
        assert {"AAPL", "MSFT", "GOOG"} <= s and "MXIM" not in s, (day, s)
    for day in ("2014-01-10", "2014-04-10", "2015-06-10"):
        s = held(day)
        assert ("GOOG" in s or "GOOGL" in s) and ("FB" in s or "META" in s), (day, s)
    assert "MXIM" not in set(h.columns[(h > 0.01).any()])


# ------------------------------------------------------------ 2. truncated rebuilt histories are not delistings

TRUNCATED = ("AMLN", "BEAS", "BMET", "CDWC", "CKFR", "SEPR", "XMSR", "NIHD", "NTLI", "MEDI-2007", "CHIR", "IACI",
             "DELL-2013", "LIFE-2014")


def test_truncated_histories_record_their_true_end():
    for t in TRUNCATED:
        e = data.end_of_data(t)
        assert e is not None and e["kind"] == "data_ends", t
        assert e["event"], t
        assert data.end_label(t) == "data ends"
    xm = data.end_of_data("XMSR")
    assert xm["ended"] == pd.Timestamp("2008-07-28") and "Sirius" in xm["event"]
    assert "$31.00" in data.end_of_data("AMLN")["event"]
    # a history that ends where the listing did stays a delisting
    assert data.end_label("CEPH") == "delisted" and data.end_of_data("CEPH")["kind"] == "delisted"


@needs("AMLN", "QQQ")
def test_a_truncated_history_is_closed_with_a_data_ends_note():
    p = parser.parse("hold AMLN and QQQ equally, rebalance monthly, from 2005 to 2008")
    r = runner.run(p)
    notes = " ".join(p.notes)
    assert "Data ends: AMLN's data ends on 2006-12-29" in notes and "2012-08-08" in notes
    assert "AMLN delisted/acquired" not in notes and not any(n.startswith("Delisted: AMLN") for n in p.notes)
    assert not any("Identity: during AMLN" in n for n in p.notes)   # no data after 2006 is not junk data
    orders = pd.DataFrame(r.orders)
    assert "delisted" not in set(orders["reason"])
    assert "data ends" in set(orders.loc[orders["ticker"] == "AMLN", "reason"])


@needs("XMSR")
def test_signal_run_on_a_truncated_history_names_the_data_end():
    p = parser.parse("buy XMSR when its close is above its 10 day moving average, sell when below, from 2006 to 2008")
    runner.run(p)
    assert any(n.startswith("Data ends: XMSR's price history ends on 2006-12-29") and "Sirius" in n for n in p.notes)


# ------------------------------------------------------------ 3. multi-bar level-shift junk

def test_level_junk_detector_finds_flipping_blocks_and_ignores_real_moves():
    from backtester import integrity
    rng = np.random.default_rng(0)
    base = np.log(20.0) + np.cumsum(rng.normal(0, 0.01, 300))
    junk = base.copy()
    for k in range(120, 180, 5):                    # two-day blocks 1.33x off, every five days
        junk[k:k + 2] += np.log(1.33)
    found = integrity.level_junk_stretches(junk, np.diff(junk, prepend=np.nan))
    assert len(found) == 1 and 118 <= found[0][0] <= 121 and 170 <= found[0][1] <= 182
    crash = base.copy()
    crash[150:] -= np.log(2)                         # one real crash: not flipping
    assert integrity.level_junk_stretches(crash, np.diff(crash, prepend=np.nan)) == []
    ticks = np.log(np.where(np.arange(300) % 3 == 0, 5.0, 6.0))   # a thin stock bouncing between two ticks
    assert integrity.level_junk_stretches(ticks, np.diff(ticks, prepend=np.nan)) == []
    # a thin stock quoted on a few tick prices, unchanged for days in between (OFG 1990): real, if sparse, trading
    ofg = np.log([                                   # OFG's closes 1989-10..1990-03 (prices file rows 640-759)
        0.8367, 0.8367, 0.8367, 0.8008, 0.8008, 0.8606, 0.8128, 0.8128, 0.8128, 0.8367, 0.8367, 0.7889, 0.8128, 0.765,
        0.8128, 0.8367, 0.8367, 0.8367, 0.8367, 0.7889, 0.9562, 0.9562, 0.9562, 0.9084, 0.9084, 0.9562, 0.8606, 0.8606,
        0.8606, 0.8606, 0.8606, 0.8606, 0.9562, 0.9562, 0.8128, 0.8845, 0.8845, 0.8845, 0.8845, 0.8845, 0.8845, 0.8845,
        0.9084, 0.9084, 0.9562, 0.9084, 0.9084, 0.9084, 0.9084, 0.9562, 0.9562, 0.8367, 0.8367, 0.8367, 0.8367, 0.8128,
        0.8128, 0.8128, 0.8128, 0.9562, 0.7889, 0.7889, 0.8367, 0.7889, 0.7889, 0.7889, 0.7889, 0.9562, 0.8008, 0.8008,
        0.8008, 0.8008, 0.8008, 0.8008, 0.8008, 0.8008, 0.9562, 0.9562, 0.9323, 0.7889, 0.7889, 0.7889, 0.7889, 0.7889,
        0.7889, 0.9562, 0.7889, 0.7889, 0.8128, 0.8128, 0.8606, 0.8606, 0.8606, 0.9084, 0.9084, 0.8606, 0.8606, 0.8128,
        0.8486, 0.8128, 0.9084, 0.9084, 0.9084, 0.9084, 0.9084, 0.9084, 0.8128, 0.8128, 0.8128, 0.765, 0.765, 0.765,
        0.765, 0.765, 0.8008, 0.8008, 0.8008, 0.8367, 0.8367, 0.8367])
    thin = np.concatenate([ofg[0] + base[:180] - base[179], ofg])
    assert integrity.level_junk_stretches(thin, np.diff(thin, prepend=np.nan)) == []
    old = integrity.LEVEL_SEG_DISTINCT
    try:                                                 # (the stale-tick check is what keeps it out)
        integrity.LEVEL_SEG_DISTINCT = 0.0
        assert integrity.level_junk_stretches(thin, np.diff(thin, prepend=np.nan))
    finally:
        integrity.LEVEL_SEG_DISTINCT = old


@needs("WFM")
def test_wfm_1996_junk_is_repaired_and_makes_no_phantom_trades():
    ev = data.price_repairs("WFM")
    junk = ev[ev["kind"] == "level_junk"]
    assert len(junk) and junk["date"].min() <= pd.Timestamp("1996-08-06")
    assert junk["date"].max() >= pd.Timestamp("1996-10-15")
    c = data.load("WFM")["close"]["1996-07-25":"1996-10-31"]
    assert (c.pct_change().abs().dropna() < 0.12).all()
    assert not data.load("WFM").loc["1996-08-01":"1996-10-18", "open_ok"].any()
    p = parser.parse("buy WFM when it drops 20% in a day, hold 1 day, from 1996 to 1997")
    r = runner.run(p)
    assert len(r.trades) == 0
    assert "WFM 1996-08-01..1996-10-18" in (data.integrity_note(["WFM"], "1996-01-01", "1997-12-31") or "")


# Level-shift junk known in the core universe (Nasdaq-100 members past and present, core ETFs: price_flags.core_tickers):
# WFM/WFMI 1996 (alternating two-day blocks ~1.33x apart on normal volume). SZK 2014-15 (zero-volume quotes flipping
# 150 <-> 210) is outside it. The data job keeps adding files, so outside the core only consistency is asserted.
KNOWN_CORE_JUNK = {("WFM", 1996), ("WFMI", 1996)}


def _level_junk_hits(raw: pd.DataFrame) -> list:
    from backtester import integrity
    c = pd.to_numeric(raw["close"], errors="coerce")
    with np.errstate(invalid="ignore", divide="ignore"):
        lc = np.log(c.where(c > 0)).to_numpy()
        lh = np.log(pd.to_numeric(raw["high"], errors="coerce").where(lambda x: x > 0)).to_numpy()
        ll = np.log(pd.to_numeric(raw["low"], errors="coerce").where(lambda x: x > 0)).to_numpy()
    return integrity.level_junk_stretches(lc, np.diff(lc, prepend=np.nan), lh, ll)


def test_only_the_known_junk_stretches_are_found_in_the_price_files():
    from backtester import price_flags
    core = price_flags.core_tickers()
    hits = {}
    for p in sorted(data.PRICES.glob("*.csv")):
        raw = pd.read_csv(p, index_col=0, parse_dates=True)
        found = _level_junk_hits(raw)
        if found:
            hits[p.stem] = [(raw.index[a], raw.index[b]) for a, b, *_ in found]
    print("level-shift junk found:", {t: [(a.date(), b.date()) for a, b in v] for t, v in sorted(hits.items())})
    core_hits = {(t, a.year) for t, v in hits.items() if t in core for a, _ in v}
    assert core_hits <= KNOWN_CORE_JUNK, f"new level-shift junk in the core universe (check it): {core_hits - KNOWN_CORE_JUNK}"
    # every detection, anywhere: the integrity gate repairs it, and the loaded series no longer flips there
    for t, spans in hits.items():
        ev = data.price_repairs(t)
        junk = pd.to_datetime(ev.loc[ev["kind"] == "level_junk", "date"]) if len(ev) else pd.Series([], dtype="datetime64[ns]")
        df = data.load(t)
        after = [(df.index[a], df.index[b]) for a, b, *_ in _level_junk_hits(df)]
        for a, b in spans:
            assert ((junk >= a) & (junk < b)).any(), (t, a.date(), b.date(), "detected but not repaired")
            assert not any(x <= b and y >= a for x, y in after), (t, a.date(), b.date(), "still flipping after repair")


# ------------------------------------------------------------ 4. dividends of rebuilt histories

@needs("LLTC", "BRCM", "WFM", "KRFT")
def test_rebuilt_histories_carry_their_dividends():
    for t, lo in (("LLTC", 80), ("BRCM", 20), ("WFM", 40), ("KRFT", 8)):
        df = data.load(t)
        assert (df["dividend"] > 0).sum() >= lo, t
        # the total-return index moves by the dividend on its ex-date
        d = df.index[df["dividend"] > 0][-1]
        i = df.index.get_loc(d)
        tr = df["adj_close"].iloc[i] / df["adj_close"].iloc[i - 1]
        pr = (df["close"].iloc[i] + df["dividend"].iloc[i]) / df["close"].iloc[i - 1]
        assert abs(tr / pr - 1) < 0.002, t
    assert data.load("LLTC").loc["2017-02-22", "dividend"] == pytest.approx(0.33)


@needs("ALTR", "MOLX")
def test_price_only_histories_are_named_in_runs_holding_them():
    assert data.price_only_ranges("ALTR")[0]["yield"] == pytest.approx(0.008)
    p = parser.parse("hold ALTR and MOLX equally, rebalance monthly, from 2008 to 2012")
    runner.run(p)
    n = [x for x in p.notes if x.startswith("Price-only history")]
    assert any("ALTR" in x and "0.8%/yr" in x for x in n) and any("MOLX" in x for x in n)
    assert data.price_only_note(["LLTC"], "2005-01-01", "2010-01-01") is None
