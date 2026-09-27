"""QuantConnect review, round 11: share-class market-cap weights, stale membership snapshots, non-investable
indexes, stale macro series, margin messages and day-one margin calls, default borrow fees, same-day yields,
spin-off ratios booked as splits, and rule-language additions (cross-sectional rank, yield curve, holidays)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import api, data, parser

needs_data = pytest.mark.skipif(not (data.PRICES / "SPY.csv").exists(), reason="no price data")


def _weights_on(r, day):
    h = r.holdings
    row = h.loc[:day].iloc[-1]
    row = row[row.abs() > 1e-9]
    return row / row.sum()


# ------------------------------------------------------------ 6. market-cap weights of share classes

@needs_data
@pytest.mark.skipif(not (data.PRICES / "GOOGL.csv").exists(), reason="no GOOGL data")
def test_market_cap_weight_counts_a_share_class_at_its_company_value():
    day = pd.Timestamp("2024-06-03")
    goog, msft = data.market_cap("GOOGL"), data.market_cap("MSFT")
    company = goog.loc[:day].iloc[-2] * data._class_divisor("GOOGL")     # the value known at the rebalance close
    r = api.backtest(parser.parse("hold GOOGL and MSFT, market cap weighted, rebalance monthly, since 2024"))
    w = _weights_on(r, day)
    ratio = w["GOOGL"] / w["MSFT"]
    assert ratio == pytest.approx(company / msft.loc[:day].iloc[-2], rel=0.05)
    assert ratio > 0.55          # not half of Alphabet (~0.35)
    # both classes held: each at half the company, together the company's weight
    r2 = api.backtest(parser.parse("hold GOOGL, GOOG and MSFT, market cap weighted, rebalance monthly, since 2024"))
    w2 = _weights_on(r2, day)
    assert (w2["GOOGL"] + w2["GOOG"]) / w2["MSFT"] == pytest.approx(ratio, rel=0.05)


# ------------------------------------------------------------ 7. stale snapshots do not bring removed members back

@needs_data
@pytest.mark.skipif(data.membership() is None or data.ndx_changes() is None, reason="no membership history")
@pytest.mark.parametrize("ticker,start,end", [("CSGP", "2020-07-20", "2020-08-31"), ("AVGO", "2015-11-11", "2016-01-29"),
                                              ("PRGO", "2013-06-06", "2013-08-30"), ("BATRK", "2016-06-20", "2016-12-30")])
def test_a_dated_removal_keeps_the_name_out_until_a_dated_addition(ticker, start, end):
    idx = data.load("SPY").loc[start:end].index
    m, _ = data.member_mask([ticker], idx)
    assert not m.any()
    if ticker == "AVGO":           # re-added on 2016-02-01
        idx2 = data.load("SPY").loc["2016-02-01":"2016-03-31"].index
        assert data.member_mask(["AVGO"], idx2)[0].all()


@needs_data
@pytest.mark.skipif(data.membership() is None or data.ndx_changes() is None, reason="no membership history")
def test_scan_no_member_between_a_dated_removal_and_the_next_addition():
    idx = data.load("SPY").loc["2007":].index
    names = sorted(data.membership().columns)
    m, _ = data.member_mask(names, idx)
    bad = []
    for t, lst in data.change_events(names).items():
        j = names.index(t)
        for n, (d, joined) in enumerate(lst):
            if joined:
                continue
            nxt = next((e for e, jn in lst[n + 1:] if jn), None)
            span = (idx >= d) & ((idx < nxt) if nxt is not None else True)
            if m[span, j].any():
                bad.append((t, str(d.date()), int(m[span, j].sum())))
    assert not bad


def test_a_stale_snapshot_after_an_addition_does_not_drop_it_but_a_long_absence_does(tmp_path, monkeypatch):
    f = tmp_path / "ch.csv"
    f.write_text("date,added,removed,reason\n2024-07-22,AAA,BBB,x\n2024-09-16,BBB,,x\n")
    monkeypatch.setattr(data, "CHANGES_FILE", f)
    data.ndx_changes.cache_clear()
    try:
        idx = pd.bdate_range("2024-06-03", "2025-06-30")
        snap = np.zeros((len(idx), 3), bool)
        snap[:, 0] = (idx >= "2024-09-01") & (idx < "2024-10-01")    # AAA: listed a month late, then gone for good
        snap[:, 1] = True                                             # BBB: stale snapshots never drop it
        snap[:, 2] = (idx >= "2024-08-01")
        out = pd.DataFrame(data.apply_changes(snap.copy(), ["AAA", "BBB", "CCC"], idx), index=idx,
                           columns=["AAA", "BBB", "CCC"])
        assert out.loc["2024-07-22":"2024-09-30", "AAA"].all()
        assert not out.loc["2024-12-01":, "AAA"].any()      # absent from the snapshots for > half a year: they win
        assert out.loc[:"2024-07-19", "BBB"].all()
        assert not out.loc["2024-07-22":"2024-09-13", "BBB"].any() and out.loc["2024-09-16":, "BBB"].all()
        assert (out["CCC"] == snap[:, 2]).all()
    finally:
        data.ndx_changes.cache_clear()


# ------------------------------------------------------------ 8. indexes cannot be held or traded

@needs_data
@pytest.mark.skipif(not (data.PRICES / "^VIX.csv").exists(), reason="no ^VIX data")
@pytest.mark.parametrize("sentence,proxy", [
    ("buy ^VIX at the close when it is down 3 days in a row, hold 2 days", "VIXY"),
    ("hold 50% ^GSPC and 50% ^VIX", "SPY"),
])
def test_holding_or_trading_an_index_is_refused_with_a_proxy(sentence, proxy):
    with pytest.raises(ValueError, match=r"is an index.*cannot be bought") as e:
        api.backtest(parser.parse(sentence))
    assert proxy in str(e.value) and "sym(" in str(e.value)


@needs_data
@pytest.mark.skipif(not (data.PRICES / "^VIX.csv").exists(), reason="no ^VIX data")
def test_indexes_stay_usable_in_conditions():
    r = api.backtest(parser.parse("hold SPY when ^VIX is below 20, otherwise TLT, since 2020"))
    assert r.equity.iloc[-1] > 0
    s = api.backtest(parser.parse("buy SPY when `sym(\"^VIX\").close > 30`, hold 5 days, since 2018"))
    assert len(s.trades)


def test_not_investable():
    assert data.not_investable("SPY") is None and data.not_investable("vix")      # the alias VIX is ^VIX
    assert "QQQ" in data.not_investable("^NDX")
