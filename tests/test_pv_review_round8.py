"""Round 8 (Portfolio Visualizer expert review): historical withdrawal rates independent of the amount entered,
weight-first / N/M / asset-class phrases and month-name dates, more lazy portfolios, one frequency in the
correlation tool's asset table, calendar-month lookbacks, and report cosmetics (monthly max drawdown, labels)."""
import numpy as np
import pandas as pd
import pytest

from backtester import data, metrics, parser, report, runner
from backtester.portfolio import Portfolio

AVAIL = set(data.available_tickers())
needs = lambda *ts: pytest.mark.skipif(not set(ts) <= AVAIL, reason="price data not downloaded")  # noqa: E731


def port(text) -> Portfolio:
    p = parser.parse(text)
    assert isinstance(p, Portfolio), text
    return p


# ------------------------------------------------------------------ 1. historical SWR / PWR

def _independent_swr_6040_1966() -> float:
    """60/40 SPYSIM/IEFSIM rebalanced on the first day of each year, from 1966; month-end returns; the highest
    CPI-indexed yearly withdrawal (share of the start, taken at the start of each year) that never runs out."""
    df = pd.concat([data.load("SPYSIM")["close"], data.load("IEFSIM")["close"]], axis=1, keys=["s", "b"],
                   sort=True).dropna()
    r = df.pct_change().dropna()
    r = r[r.index >= "1966-01-01"]
    hs, hb, yr, nav = 0.6, 0.4, None, []
    for d, s_, b_ in zip(r.index, r["s"].to_numpy(), r["b"].to_numpy()):
        if yr is not None and d.year != yr:
            t = hs + hb
            hs, hb = 0.6 * t, 0.4 * t
        yr = d.year
        hs *= 1 + s_
        hb *= 1 + b_
        nav.append(hs + hb)
    me = pd.Series(nav, r.index).resample("ME").last()
    mr = me.pct_change()
    mr.iloc[0] = me.iloc[0] - 1
    c = data.cpi()
    c.index = c.index.to_period("M").to_timestamp("M")
    inf = c.pct_change().reindex(mr.index.to_period("M").to_timestamp("M")).fillna(0).to_numpy()
    ci = np.concatenate([[1.0], np.cumprod(1 + inf)])
    R = mr.to_numpy()

    def lasts(rate):
        bal = 1.0
        for m in range(len(R)):
            if m % 12 == 0:
                bal -= rate * ci[m]
                if bal <= 0:
                    return False
            bal *= 1 + R[m]
        return True
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if lasts(mid) else (lo, mid)
    return lo


@needs("SPYSIM", "IEFSIM")
def test_historical_swr_does_not_depend_on_the_amount_withdrawn():
    want = _independent_swr_6040_1966()
    assert 0.03 < want < 0.04
    got = {}
    for w in ("$60,000 a year adjusted for inflation", "4% per year adjusted for inflation",
              "$30,000 a year adjusted for inflation"):
        p = port(f"hold 60% SPYSIM and 40% IEFSIM since 1966, rebalance yearly, starting with $1,000,000, withdraw {w}")
        A = report.analyze(runner.run(p), sensitivity=False, mc=False, detail=False)
        wr = A["withdrawal_rates"]
        got[w] = wr["swr"]
        # the full requested period, even when $60,000 a year empties the account around 1981
        assert str(wr["from"]).startswith("1966") and wr["to"].year >= 2020, wr
        assert wr["basis"] == "flow_free_returns"
        assert "over the full period 1966" in report.console_summary(A)
    vals = list(got.values())
    assert max(vals) - min(vals) < 1e-9, got
    assert vals[0] == pytest.approx(want, abs=5e-4)


@needs("VTISIM", "BNDSIM")
def test_contribute_then_withdraw_rate_uses_the_whole_withdrawal_phase():
    rates = []
    for amt in ("$50,000", "$20,000"):
        p = port(f"add $1,000 a month for 20 years, then withdraw {amt} a year adjusted for inflation, "
                 "hold 60% VTISIM and 40% BNDSIM, since 1980, starting with $10,000")
        A = report.analyze(runner.run(p), sensitivity=False, mc=False, detail=False)
        wr = A["withdrawal_rates"]
        assert wr["base"] == "withdrawal_start" and wr["base_date"].year == 2000 and wr["to"].year >= 2020
        rates.append((wr["swr"], wr["pwr"]))
    assert rates[0] == pytest.approx(rates[1], abs=1e-12)
    assert rates[0][0] < 0.07          # not the 7.72% measured only up to the 2012 depletion


def test_monthly_withdrawals_pay_the_same_yearly_amount_in_instalments():
    from backtester import montecarlo as mc
    P = np.zeros((1, 120))
    ci = np.ones((1, 121))
    # no growth, no inflation: 10 years of withdrawals last exactly at 10% a year, monthly or yearly
    assert mc.safe_withdrawal_rate(P, ci, 1.0, 1.0, every=12) == pytest.approx(0.1, abs=1e-6)
    assert mc.safe_withdrawal_rate(P, ci, 1.0, 1.0, every=1) == pytest.approx(0.1, abs=1e-6)


# ------------------------------------------------------------------ 2. phrases

def _w(p) -> dict:
    t = p.tree
    return {c["asset"]: w for c, w in zip(t["children"], t["w"])} if "children" in t else {t["asset"]: 1.0}


@pytest.mark.parametrize("text,want,start,end", [
    ("60% VTI 40% BND since 2010", {"VTI": 0.6, "BND": 0.4}, "2010-01-01", None),
    ("60% VTI 40% BND from 2010 to 2020", {"VTI": 0.6, "BND": 0.4}, "2010-01-01", "2020-12-31"),
    ("50% VTSMX 50% VBMFX since 1993", {"VTSMX": 0.5, "VBMFX": 0.5}, "1993-01-01", None),
    ("VTI 60 BND 40 since 2010", {"VTI": 0.6, "BND": 0.4}, "2010-01-01", None),
    ("60% SPY 40% TLT from March 2005 to June 2015", {"SPY": 0.6, "TLT": 0.4}, "2005-03-01", "2015-06-30"),
    ("60% SPY 40% TLT since Jan 1999", {"SPY": 0.6, "TLT": 0.4}, "1999-01-01", None),
    ("60% SPY 40% TLT until 2020-06", {"SPY": 0.6, "TLT": 0.4}, None, "2020-06-30"),
    ("hold 60% SPY and 40% TLT from 2010-03 through Feb. 2016", {"SPY": 0.6, "TLT": 0.4}, "2010-03-01", "2016-02-29"),
])
def test_weight_first_lists_and_month_dates(text, want, start, end):
    ts = [t for t in want if t not in AVAIL]
    if ts:
        pytest.skip(f"no data for {ts}")
    p = port(text)
    assert _w(p) == pytest.approx(want) and p.start == start and p.end == end


@needs("VTISIM", "BNDSIM")
@pytest.mark.parametrize("text,stocks", [
    ("80/20 portfolio since 1972", 0.8), ("US stocks and bonds 70/30 since 1950", 0.7), ("Stocks/Bonds 60/40", 0.6),
    ("hold 40/60 stocks/bonds, rebalance yearly", 0.4), ("the 20/80 mix", 0.2),
])
def test_stock_bond_splits(text, stocks):
    p = port(text)
    w = _w(p)
    s_key = next(k for k in w if k.startswith("VTI"))
    b_key = next(k for k in w if k.startswith("BND"))
    assert w[s_key] == pytest.approx(stocks) and w[b_key] == pytest.approx(1 - stocks)
    if p.start and p.start < "2007":
        assert (s_key, b_key) == ("VTISIM", "BNDSIM")
    assert any("stocks/bonds" in n for n in p.notes)


def test_stock_bond_split_must_add_up():
    with pytest.raises(parser.ParseError):
        parser.parse("70/40 portfolio")


def test_signal_sentences_stay_signals():
    s = parser.parse("buy SPY when RSI(2) is below 10, sell when RSI(2) is above 70, since 2010")
    assert not isinstance(s, Portfolio)


@needs("VCLTSIM", "VBRSIM")
def test_bare_asset_class_is_100_percent():
    p = port("long-term corporate bonds since 1955")
    assert _w(p) == {"VCLTSIM": 1.0} and p.start == "1955-01-01"
    assert _w(port("hold US small cap value")) == {"VBRSIM": 1.0}
    # round 12: one rule for asset-class words - alone and with no start, the long-history series (TLTSIM is TLT
    # itself from TLT's first day); with a start after the fund existed, the fund
    assert _w(port("hold long-term treasuries")) == {"TLTSIM": 1.0}
    assert _w(port("hold long-term treasuries since 2010")) == {"TLT": 1.0}


def test_drift_phrases():
    for t in ("60% SPY 40% TLT, rebalance when drift exceeds 5%", "60% SPY 40% TLT, rebalance at 5% drift",
              "hold 60% SPY and 40% TLT, rebalance if the drift is more than 5%"):
        p = port(t)
        assert p.drift_band == pytest.approx(0.05) and p.rebalance == "none", t
    p = port("hold 60% SPY and 40% TLT, rebalance yearly or when drift exceeds 10%")
    assert p.drift_band == pytest.approx(0.10) and p.rebalance == "yearly"


def test_target_volatility_phrase_rescales_monthly():
    p = port("hold SPY with a 10% target volatility using 60 day volatility")
    assert p.target_vol == pytest.approx(0.10) and p.target_vol_lookback == 60 and p.rebalance == "monthly"
    assert port("hold SPY with a 10% target volatility, rebalance weekly").rebalance == "weekly"


# ------------------------------------------------------------------ 3. lazy portfolios

@pytest.mark.parametrize("text,n,first", [
    ("7Twelve portfolio", 12, "VV"), ("couch potato portfolio", 2, "VTI"), ("Merriman 4-fund combo", 4, "SPY"),
    ("Frank Armstrong ideal index", 7, "VV"), ("Bogleheads four-fund portfolio", 4, "VTI"),
    ("second grader's starter portfolio", 3, "VTI"), ("Dave Ramsey portfolio", 4, "VUG"),
    ("Aronson family taxable portfolio", 11, "TIP"),
])
def test_new_lazy_portfolios(text, n, first):
    p = port(text)
    kids = p.tree["children"]
    assert len(kids) == n and kids[0]["asset"] == first and sum(p.tree["w"]) == pytest.approx(1.0, abs=1e-8)
    assert any(":" in x and "%" in x for x in p.notes)


@needs("SPYSIM", "VTVSIM", "VBSIM", "VBRSIM", "VPLSIM")
def test_lazy_portfolio_early_start_uses_long_series():
    p = port("Merriman 4-fund combo since 1970")
    assert [c["asset"] for c in p.tree["children"]] == ["SPYSIM", "VTVSIM", "VBSIM", "VBRSIM"]
    p = port("aronson family taxable since 1995")
    # the Pacific sleeve's long series: VPLSIM (Japan + Asia-Pacific ex Japan, then VPACX) since the fund-history
    # review; before it, Japan alone (EWJSIM) stood in
    assert "VPLSIM" in [c["asset"] for c in p.tree["children"]]
    assert any("VPLSIM for VPL" in x for x in p.notes)


# ------------------------------------------------------------------ 4. correlation tool: one frequency

@needs("SPYSIM", "EFASIM", "VNQSIM")
def test_correlation_asset_stats_follow_the_matrix_frequency():
    from backtester import correlation
    R = correlation.analyze(["SPYSIM", "EFASIM", "VNQSIM"], "monthly", None, "1980-01-01", "1995-12-31")
    assert R["freq"] == "monthly" and R["stats_basis"] == "monthly"
    s = {x["ticker"]: x for x in R["stats"]}
    px = data.load("SPYSIM")["adj_close"]
    px = px[(px.index >= "1980-01-01") & (px.index <= "1995-12-31")]
    mr = metrics.monthly_returns(px)
    assert s["SPYSIM"]["volatility"] == pytest.approx(mr.std() * np.sqrt(12), rel=1e-9)
    assert s["SPYSIM"]["max_drawdown"] == pytest.approx(metrics.monthly_max_drawdown(px), abs=1e-12)
    assert s["SPYSIM"]["max_drawdown"] > s["SPYSIM"]["max_drawdown_daily"]        # 1987: month-end is shallower
    assert "monthly returns, like the matrix" in correlation.console(R)
    D = correlation.analyze(["SPYSIM", "TLTSIM"], "daily", None, "1990-01-01", "1995-12-31")
    assert D["stats_basis"] == "daily" and "return_basis" not in D["stats"][0]


def test_monthly_max_drawdown_uses_month_ends():
    idx = pd.bdate_range("2020-01-01", "2020-04-30")
    v = pd.Series(100.0, index=idx)
    v[(v.index >= "2020-02-10") & (v.index <= "2020-02-20")] = 60.0      # a dip inside February
    v[v.index >= "2020-03-01"] = 90.0
    assert metrics.monthly_max_drawdown(v) == pytest.approx(-0.10)
    assert (v / v.cummax() - 1).min() == pytest.approx(-0.40)


# ------------------------------------------------------------------ 5. calendar-month lookbacks

DM = ("hold the top 1 of SPYSIM and EFASIM by 12 month return, only if their 12 month return is above BILSIM's "
      "12 month return, otherwise hold IEFSIM, rebalance monthly, since 1976")


def _dual_momentum_mismatches(p) -> tuple[int, int]:
    """Months where the backtest's holding differs from a pandas rebuild on month-end total-return prices."""
    from backtester import expr
    h = runner.run(p).holdings
    me = {}
    for t in ("SPYSIM", "EFASIM", "BILSIM"):
        s = expr.adjusted_frame(data.load(t))["close"]
        m = s[expr.month_end_flags(s.index)]
        m.index = m.index.to_period("M")
        me[t] = m
    M = pd.DataFrame(me).dropna()
    R = M / M.shift(12) - 1
    n = bad = 0
    for d in h.index[expr.month_end_flags(h.index)]:
        per = d.to_period("M")
        if per not in R.index or R.loc[per].isna().any():
            continue
        r = R.loc[per]
        best = "SPYSIM" if r["SPYSIM"] >= r["EFASIM"] else "EFASIM"
        want = best if r[best] > r["BILSIM"] else "IEFSIM"
        n += 1
        bad += h.loc[d].idxmax() != want
    return n, bad


@needs("SPYSIM", "EFASIM", "BILSIM", "IEFSIM")
def test_calendar_month_dual_momentum_matches_a_month_end_rebuild():
    p = port(DM + ", using calendar months")
    assert p.month_lookbacks == "calendar"
    assert "calendar months, month-end to month-end" in p.summary()
    n, bad = _dual_momentum_mismatches(p)
    assert n > 500 and bad == 0
    # the default (21 trading days a month) is a different measurement, and the interpretation says so
    q = port(DM)
    assert q.month_lookbacks == "trading" and "12 months = 252 sessions" in q.summary()
    assert _dual_momentum_mismatches(q)[1] > 0


def test_calendar_month_phrases_and_spec_round_trip():
    for t in ("hold the top 1 of SPY and EFA by 12 month return, rebalance monthly, using month-end prices",
              "hold the top 1 of SPY and EFA by 6 month return, rebalance monthly, month-end to month-end"):
        assert port(t).month_lookbacks == "calendar", t
    p = port("hold the top 1 of SPY and EFA by 12 month return, rebalance monthly, using calendar months")
    assert runner.from_dict(runner.to_dict(p)).month_lookbacks == "calendar"
    with pytest.raises(ValueError):
        Portfolio(tree={"asset": "SPY"}, month_lookbacks="weeks").validate()


def test_calendar_month_return_is_causal():
    from backtester import expr
    idx = pd.bdate_range("2019-01-01", "2021-06-30")
    x = pd.Series(np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, len(idx)))), index=idx)
    full = expr.calendar_month_return(x, 3)
    for cut in ("2020-03-17", "2020-03-31", "2020-12-31", "2021-02-01"):
        part = expr.calendar_month_return(x[x.index <= cut], 3)
        pd.testing.assert_series_equal(part, full[full.index <= cut])
    # on a month-end it is that month over three months earlier; mid-month it holds the last month-end's value
    me = x.resample("ME").last()
    assert full.loc["2020-06-30"] == pytest.approx(me.loc["2020-06-30"] / me.loc["2020-03-31"] - 1)
    assert full.loc["2020-07-15"] == full.loc["2020-06-30"]


# ------------------------------------------------------------------ 6. cosmetics

def test_short_labels_keep_blends_readable():
    assert report.short_label("60% SPYSIM / 40% IEFSIM buy & hold") == "60/40 SPYSIM/IEFSIM"
    assert report.short_label("60% SPY, 40% AGG blend") == "60/40 SPY/AGG"
    assert report.short_label("SPY buy & hold") == "SPY"


@needs("SPYSIM", "IEFSIM")
def test_console_shows_blend_names_and_month_end_drawdown():
    p = port("hold 60% SPYSIM and 40% IEFSIM since 1966, rebalance yearly, vs 60/40 SPYSIM/IEFSIM")
    A = report.analyze(runner.run(p), sensitivity=False, mc=False, detail=False)
    txt = report.console_summary(A)
    assert "~" not in txt and "60/40 SPYSIM/IEFSIM" in txt and "month-end" in txt
    st = A["stats"]
    assert st["max_drawdown"] <= st["max_drawdown_monthly"] < 0


def test_optimiser_console_does_not_cut_names():
    from backtester import research_report
    name = "Min tracking error (≥ +1.0% over the benchmark)"
    R = {"fit_start": "2010-01-01", "fit_end": "2020-12-31", "rf": 0.01, "benchmark": None,
         "portfolios": {name: {"exp_return": 0.08, "exp_vol": 0.1, "exp_sharpe": 0.7, "sentence": ""}}}
    assert name in research_report.optimize_console(R)


def test_no_minus_zero_dollars():
    idx = pd.bdate_range("2020-01-01", periods=30)
    eq = pd.Series(np.linspace(100, 130, 30), index=idx)
    flows = pd.Series(0.0, index=idx)
    flows.iloc[5] = 10.0
    c = metrics.cashflow_stats(eq, flows)
    assert c["total_withdrawals"] == 0 and f"{c['total_withdrawals']:,.0f}" == "0"


def test_monte_carlo_labels_and_stress_basis(monkeypatch):
    from backtester import montecarlo as mc
    idx = pd.bdate_range("2000-01-03", periods=3000)
    r = np.full(len(idx), 0.0004)
    r[1200:1400] = -0.003
    df = pd.DataFrame({"close": 100 * np.cumprod(1 + r)}, index=idx)
    for k in ("open", "high", "low", "adj_close"):
        df[k] = df["close"]
    df["volume"] = 1e6
    df["dividend"] = 0.0
    real_load = data.load
    monkeypatch.setattr(data, "load", lambda t, *a, **k: df if t == "ZZZ" else real_load(t, *a, **k))
    base = dict(weights={"ZZZ": 1}, sims=200, years=15, stress="worst_sequence", stress_years=2)
    S = mc.run(mc.Settings(**base, inflation=0.0))
    assert S["stress"]["basis"] == "nominal" and any("nominal" in n for n in S["notes"])
    txt = mc.console(mc.run(mc.Settings(weights={"ZZZ": 1}, sims=200, years=15, inflation=0.0,
                                        flows=[mc.CashFlow(amount=-400)])))
    assert "95% of paths" in txt and "median path" in txt
