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


# ------------------------------------------------------------ 9. stale macro series are unknown, not frozen

def _bars(start, end):
    idx = pd.bdate_range(start, end)
    c = np.linspace(100, 120, len(idx))
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6, "dividend": 0.0, "adj_close": c,
                         "split": 1.0, "open_ok": True, "quote_close": c}, index=idx)


def test_a_stale_cape_is_nan_with_a_note(monkeypatch):
    from backtester import expr
    known = pd.Series([25.0, 26.0, 27.0], index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))
    monkeypatch.setattr(data, "shiller_known", lambda col, lag_months=None: known)
    ns = expr.Namespace(_bars("2024-01-02", "2024-12-31"), ticker="X")
    v = ns["cape"]()
    assert v.loc["2024-04-30"] == 27.0                       # within the allowed lag
    assert v.loc["2024-05-03":].isna().all()                 # 62 days after the last value: unknown
    assert not expr.evaluate("cape() < 100", ns).loc["2024-05-03":].any()
    assert any(n.startswith("Stale data: Shiller CAPE") and "2023-10" in n and "2024-05-02" in n for n in ns.notes)


def test_a_stale_yield_is_nan_and_tbill_return_stops(monkeypatch):
    from backtester import expr
    y = pd.Series(0.04, index=pd.bdate_range("2023-01-02", "2024-03-01"))
    monkeypatch.setattr(data, "treasury_10y", lambda: y)
    monkeypatch.setattr(data, "tbill_rate", lambda: y)
    ns = expr.Namespace(_bars("2023-06-01", "2024-06-28"), ticker="X")
    t10 = ns["treasury_10y"]()
    assert t10.loc["2024-03-08"] == 0.04 and t10.loc["2024-03-11":].isna().all()   # 10 days after 03-01
    tb = ns["tbill_ret"](21)
    assert tb.loc["2024-03-11":].isna().all() and tb.loc["2024-02-01"] > 0


def test_the_shiller_link_is_found_in_an_escaped_page():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fd", data.ROOT / "scripts" / "fetch_data.py")
    fd = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(fd)
    except ImportError:
        pytest.skip("fetch dependencies missing")
    url = "https://img1.wsimg.com/blobby/go/e5e/downloads/70fec4f5/ie_data.xls?ver=1788371540009"
    for page in (f'<a href="{url}">', '{"href":"' + url.replace("/", "\\/") + '"}',
                 '{"href":"' + url.replace("/", "\\u002F") + '"}'):
        assert fd.shiller_link(page) == url


# ------------------------------------------------------------ 13. same-day yields at a close fill

def test_fred_yields_lag_a_session_for_close_fills(monkeypatch):
    from backtester import expr
    idx = pd.bdate_range("2024-01-02", "2024-02-29")
    y = pd.Series(np.arange(len(idx)) / 1000.0, index=idx)
    monkeypatch.setattr(data, "treasury_10y", lambda: y)
    bars = _bars("2024-01-02", "2024-02-29")
    at_open = expr.Namespace(bars, ticker="X")["treasury_10y"]()
    at_close = expr.Namespace(bars, ticker="X", close_fill=True)
    v = at_close["treasury_10y"]()
    assert v.loc["2024-01-10"] == y.loc["2024-01-09"] and at_open.loc["2024-01-10"] == y.loc["2024-01-10"]
    assert expr.FRED_CLOSE_NOTE in at_close.notes


# ------------------------------------------------------------ 10. / 11. margin messages and day-one margin calls

def test_a_book_of_leveraged_etfs_is_described_as_a_book():
    with pytest.raises(ValueError) as e:
        parser.parse("hold 200% TQQQ and -100% SQQQ since 2012").validate()
    msg = str(e.value)
    assert "200% TQQQ, -100% SQQQ" in msg and "225% of the equity" in msg and "0.444 times this size" in msg
    assert "1x leverage on it" not in msg


def test_leverage_that_would_be_margin_called_at_once_is_refused():
    from backtester import margin
    from backtester.portfolio import Portfolio
    from backtester.strategy import Strategy
    with pytest.raises(ValueError, match=r"97.5% of the equity.*fall of 7.7%.*at most about 1.2x"):
        Portfolio(tree={"asset": "TQQQ"}, leverage=1.3).validate()
    with pytest.raises(ValueError, match="97.5% of the equity"):
        Strategy(universe=["TQQQ"], entry="True", hold_bars=1, leverage=1.3).validate()
    Portfolio(tree={"asset": "TQQQ"}, leverage=1.2).validate()           # 90%: the 10% buffer is kept
    Portfolio(tree={"asset": "TQQQ"}, leverage=1.3, maintenance_margin=0).validate()   # margin calls off
    assert margin.call_drop(0.975, 1.3, 0.25) == pytest.approx(0.025 / (1.3 * 0.25))


# ------------------------------------------------------------ 12. default borrow fees

def test_default_borrow_fees():
    from backtester import margin
    assert margin.default_borrow_fee("SQQQ") == 0.05 and margin.default_borrow_fee("UVXY") == 0.05
    assert margin.default_borrow_fee("SH") == 0.05 and margin.default_borrow_fee("QQQ") == 0.003
    assert margin.borrow_fee_of("SQQQ", 0.0) == 0.0 and margin.borrow_fee_of("SQQQ", 0.02) == 0.02
    assert parser.parse("hold 100% SPY and -30% SQQQ, rebalance monthly, no borrow fee").borrow_fee == 0.0
    assert parser.parse("hold 100% SPY and -30% SQQQ, rebalance monthly").borrow_fee is None


@needs_data
@pytest.mark.skipif(not (data.PRICES / "SQQQ.csv").exists(), reason="no SQQQ data")
def test_shorting_sqqq_pays_the_assumed_fee_with_a_note():
    base = "hold 100% SPY and -30% SQQQ, rebalance monthly, from 2020 to 2021"
    spec = parser.parse(base)
    dflt = api.backtest(spec)
    free = api.backtest(parser.parse(base + ", no borrow fee"))
    set5 = api.backtest(parser.parse(base + ", borrow fee 5%"))
    assert dflt.equity.iloc[-1] < free.equity.iloc[-1]
    assert dflt.equity.iloc[-1] == pytest.approx(set5.equity.iloc[-1], rel=1e-9)
    assert any(n.startswith("Borrow fee (assumed") and "SQQQ 5%" in n for n in spec.notes)


# ------------------------------------------------------------ 13. a spin-off booked as a split

def test_split_like_ratios():
    for r in (2, 3, 1.5, 4 / 3, 1.25, 0.5, 0.1, 1 / 15, 1.05, 1.1, 1.02):
        assert data._split_like(r), r
    for r in (2.376, 1.319, 2.0842, 1.758, 1.998):
        assert not data._split_like(r), r


def test_a_spin_off_booked_as_a_split_becomes_a_distribution():
    idx = pd.bdate_range("2015-07-13", periods=6)
    close = [26.0, 26.5, 26.7, 27.6, 27.9, 28.57]
    adj = [c * 0.9 for c in close[:5]] + [27.9 * 0.9 * 1.024]
    raw = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "adj_close": adj,
                        "volume": 1e6, "dividend": 0.0, "split": [0, 0, 0, 0, 0, 2.376]}, index=idx)
    out, log = data.reconcile_actions("ZZZ", raw)
    assert out["split"].iloc[-1] == 0 and out["close"].iloc[-2] == pytest.approx(27.9 * 2.376)
    assert out["volume"].iloc[0] == pytest.approx(1e6 / 2.376)
    pay = out["dividend"].iloc[-1]
    assert pay == pytest.approx(27.9 * 2.376 * 1.024 - 28.57)
    assert (28.57 + pay) / out["close"].iloc[-2] == pytest.approx(1.024)      # the total return is unchanged
    assert list(log["reading"]) == ["spinoff_as_split"]


@needs_data
@pytest.mark.skipif(not (data.PRICES / "EBAY.csv").exists(), reason="no EBAY data")
def test_ebay_paypal_spin_off():
    df = data.load("EBAY")
    assert df.loc["2015-07-17", "close"] > 60                    # as traded, not divided by 2.376
    assert df.loc["2015-07-20", "split"] == 1.0 and df.loc["2015-07-20", "dividend"] > 30
    f = data.as_traded_factor("EBAY", pd.DatetimeIndex(["2015-07-17", "2015-07-21"]), df)
    assert f[0] == pytest.approx(f[1])                           # no split between them
    assert pd.Timestamp("2015-07-20") in data.spinoff_days("EBAY")


# ------------------------------------------------------------ 14. rule-language additions

def _walks(n=300, k=6, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    out = {}
    for j in range(k):
        c = 100 * np.exp(np.cumsum(rng.normal(0.0003 * j, 0.01, n)))
        out[f"T{j}"] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "adj_close": c,
                                     "volume": 1e6, "dividend": 0.0, "split": 1.0, "open_ok": True, "quote_close": c},
                                    index=idx)
    return out


def test_xrank_is_the_cross_sectional_percentile_and_causal(monkeypatch):
    from backtester import engine, expr
    from backtester.strategy import Strategy
    frames = _walks()
    monkeypatch.setattr(data, "load", lambda t: frames[data.canonical(t)])
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): frames[data.canonical(t)] for t in ts})
    s = Strategy(universe=list(frames), entry="xrank(ret(20)) <= 0.2", hold_bars=1, cash_rate=None)
    tables = engine._xrank_tables(s, frames, list(frames), frames["T0"].index)
    src = expr.xrank_calls(s.entry)[0]
    day = frames["T0"].index[100]
    r20 = pd.Series({t: df["close"].iloc[100] / df["close"].iloc[80] - 1 for t, df in frames.items()})
    want = r20.rank(pct=True)
    got = pd.Series({t: tables[t][src].loc[day] for t in frames})
    pd.testing.assert_series_equal(got, want, check_names=False)
    r = engine.run(s)
    assert len(r.trades) and set(r.trades["entry_date"].map(pd.Timestamp)) <= set(frames["T0"].index)
    # truncating the data changes nothing before the cut
    cut = {t: df.iloc[:200] for t, df in frames.items()}
    tc = engine._xrank_tables(s, cut, list(cut), cut["T0"].index)
    for t in frames:
        pd.testing.assert_series_equal(tc[t][src], tables[t][src].iloc[:200])
    with pytest.raises(ValueError, match="several tickers"):
        Strategy(universe=["T0"], entry="xrank(ret(20)) <= 0.2", hold_bars=1).validate()
    assert expr.open_safe("xrank(ref(ret(20), 1)) < 0.5") and not expr.open_safe("xrank(ret(20)) < 0.5")


def test_phrases_for_xrank_yield_curve_and_holidays():
    s = parser.parse("buy Nasdaq 100 stocks when they are in the bottom decile of 20 day return, hold 5 days")
    assert "xrank(ret(close, 20)) <= 0.1" in s.entry
    s = parser.parse("buy Nasdaq 100 stocks when they are in the top quintile of 3 month return, hold 20 days")
    assert "xrank(ret(close, 63)) > 0.8" in s.entry
    assert "yield_curve() < 0" in parser.parse("buy SPY when the yield curve is inverted, hold 20 days").entry
    s = parser.parse("buy SPY at the close when it is the day before Thanksgiving, hold 1 day")
    assert 'days_to_holiday() == 0 and next_holiday_is("thanksgiving")' in s.entry
    assert "days_to_holiday() == 0" in parser.parse(
        "buy SPY at the close on the last trading day before a holiday, hold 1 day").entry
    assert "days_since_holiday() == 0" in parser.parse(
        "buy SPY at the open when it is the first trading day after a holiday, hold 1 day").entry


def test_holiday_functions_are_scheduled_and_open_safe():
    from backtester import expr
    idx = pd.bdate_range("2024-11-18", "2024-12-31")
    from backtester import calendar as cal
    idx = idx[[cal.is_session(d) for d in idx]]
    c = pd.Series(100.0, index=idx)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6})
    ns = expr.Namespace(df, ticker="X")
    before = expr.evaluate('days_to_holiday() == 0 and next_holiday_is("thanksgiving")', ns)
    assert list(before[before].index) == [pd.Timestamp("2024-11-27")]
    after = expr.evaluate('days_since_holiday() == 0 and last_holiday_is("Thanksgiving Day")', ns)
    assert list(after[after].index) == [pd.Timestamp("2024-11-29")]
    xmas = expr.evaluate("days_to_holiday() == 0", ns)
    assert pd.Timestamp("2024-12-24") in xmas[xmas].index
    assert expr.open_safe('days_to_holiday() == 0 and next_holiday_is("christmas")')
    with pytest.raises(ValueError, match="unknown market holiday"):
        expr.evaluate('next_holiday_is("easter monday")', ns)


@needs_data
@pytest.mark.skipif(not (data.ROOT / "data" / "macro" / "DGS2.csv").exists(), reason="no FRED data")
def test_yield_curve_matches_dgs10_minus_dgs2():
    yc = data.yield_curve()
    a, b = data.treasury_10y(), data.treasury_2y()
    d = yc.index[-100]
    if (data.DATA / "macro" / "T10Y2Y.csv").exists():
        assert abs(yc.loc[d] - (a.loc[d] - b.loc[d])) < 0.0003
    else:
        assert yc.loc[d] == pytest.approx(a.loc[d] - b.loc[d])
