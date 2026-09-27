"""Portfolio Visualizer review, round 11: cash flows made pro rata (never a hidden rebalance), depleted
portfolios beside benchmarks that keep running, benchmark warnings and labels, income on total-return series,
blended benchmarks on the portfolio's schedule, and portfolio phrases (GTAA, GEM, expense ratio word order,
relative drift, fed funds margin rate)."""
import numpy as np
import pandas as pd
import pytest

from backtester import data, metrics
from backtester import portfolio as pf


def frame(closes, start="2015-01-01", div_every=0, div=0.0):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes}, index=idx)
    df["volume"] = 1e6
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    if div_every:
        df.iloc[div_every::div_every, df.columns.get_loc("dividend")] = div
    df["split"] = 0.0
    return df


def walk(seed, n=1500, drift=0.0003, vol=0.012):
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
    return frames


TREE = {"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "A"}, {"asset": "B"}]}
FLOWS = [dict(withdrawal=150.0, withdrawal_freq="monthly"),
         dict(contribution=400.0, contribution_freq="monthly"),
         dict(withdrawal_pct=0.04, withdrawal_freq="yearly"),
         dict(contribution=1000.0, contribution_freq="quarterly", withdrawal=250.0, withdrawal_freq="monthly")]


def twr(res):
    nv = metrics.nav(res.equity, res.extras["flows"])
    return (nv / nv.iloc[0]).to_numpy()


# ------------------------------------------------------------ 1. flows are pro rata, never a rebalance

@pytest.mark.parametrize("rebalance", ["none", "yearly", "quarterly"])
@pytest.mark.parametrize("fl", FLOWS)
def test_flows_do_not_change_the_time_weighted_return(fake, rebalance, fl):
    """Invariant: a portfolio's time-weighted return is the same (1e-9) with or without cash flows, both buy and
    hold (never rebalanced) and on a rebalancing schedule: a flow buys or sells every holding pro rata."""
    fake["A"], fake["B"] = frame(walk(1, drift=0.0006, vol=0.015)), frame(walk(2, drift=0.0001, vol=0.004))
    kw = dict(tree=TREE, rebalance=rebalance, cash_rate=None, capital=100_000)
    base = pf.run(pf.Portfolio(**kw))
    flowed = pf.run(pf.Portfolio(**kw, **fl))
    assert flowed.extras["flows"].abs().sum() > 0
    np.testing.assert_allclose(twr(flowed), twr(base), rtol=1e-9)
    # a withdrawal never shows up as a rebalance
    assert "raise cash" not in set(flowed.orders["reason"])


@pytest.mark.parametrize("rebalance", ["none", "yearly"])
def test_flows_with_dividends_stay_close(fake, rebalance):
    fake["A"] = frame(walk(3, drift=0.0005), div_every=63, div=0.5)
    fake["B"] = frame(walk(4, drift=0.0001, vol=0.004), div_every=21, div=0.25)
    kw = dict(tree=TREE, rebalance=rebalance, cash_rate=None, capital=100_000)
    base = pf.run(pf.Portfolio(**kw))
    flowed = pf.run(pf.Portfolio(**kw, withdrawal=300.0, withdrawal_freq="monthly"))
    np.testing.assert_allclose(twr(flowed), twr(base), rtol=2e-4)


def test_buy_and_hold_with_withdrawals_has_no_rebalancing_turnover(fake):
    fake["A"], fake["B"] = frame(walk(5)), frame(walk(6, vol=0.004))
    res = pf.run(pf.Portfolio(tree=TREE, rebalance="none", cash_rate=None, capital=1_000_000,
                              withdrawal=2_000, withdrawal_freq="monthly"))
    reasons = set(res.orders["reason"])
    assert reasons == {"initial", "withdrawal"}, reasons
    # the mix drifts exactly as without flows
    base = pf.run(pf.Portfolio(tree=TREE, rebalance="none", cash_rate=None, capital=1_000_000))
    np.testing.assert_allclose(res.holdings["A"].to_numpy(), base.holdings["A"].to_numpy(), atol=1e-9)
    # flow trades are not rebalancing turnover
    assert res.extras["turnover_annual"] == pytest.approx(base.extras["turnover_annual"], abs=1e-9)



# ------------------------------------------------------------ real data: the reviewer's repros

from backtester import report  # noqa: E402

HAVE = {"SPY", "QQQ", "AGG", "VTISIM", "BNDSIM"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def run_sentence(text):
    from backtester import parser, runner
    return runner.run(parser.parse(text))


@needs_data
def test_withdrawals_leave_a_buy_and_hold_60_40_alone():
    base = run_sentence("hold 60% VTISIM and 40% BNDSIM since 1990, never rebalance, starting with $1,000,000")
    wd = run_sentence("hold 60% VTISIM and 40% BNDSIM since 1990, never rebalance, starting with $1,000,000, "
                      "withdraw $2,000 a month")
    y0 = report.analyze(base, sensitivity=False, mc=False, detail=False)["yearly"]
    y1 = report.analyze(wd, sensitivity=False, mc=False, detail=False)["yearly"]
    assert y1.loc[2008, "return"] == pytest.approx(y0.loc[2008, "return"], abs=1e-6)
    assert y1.loc[2008, "return"] == pytest.approx(-0.250, abs=0.002)
    assert wd.extras["turnover_annual"] == pytest.approx(base.extras["turnover_annual"], abs=1e-9)
    assert "raise cash" not in set(wd.orders["reason"])
    assert set(wd.orders["reason"]) <= {"initial", "withdrawal"}


@needs_data
def test_depleted_portfolio_keeps_benchmarks_and_warns_about_a_blend():
    res = run_sentence("hold 100% QQQ since 2000, withdraw $120,000 a year adjusted for inflation, starting with "
                       "$1,000,000, vs 60/40 SPY/AGG")
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    dep = pd.Timestamp(A["depleted"])
    assert dep.year == 2003
    end = res.equity.index[-1]
    # the blend (AGG starts 2003-09-29, after the money ran out) is shown from its own start, to the end
    blend = next(k for k in A["benchmarks"] if k.endswith(" blend"))
    assert A["benchmarks"][blend].index[0] > dep and A["benchmarks"][blend].index[-1] == end
    for b in A["benchmarks"].values():
        assert b.index[-1] == end
    y = A["yearly"]
    assert y.index.max() == end.year
    assert bool(y.loc[2010, "depleted"]) and pd.isna(y.loc[2010, "return"]) and y.loc[2010, "end_balance"] == 0
    assert bool(y.loc[2003, "partial"]) and str(y.loc[2003, "to"]) == str(dep.date())
    bb = A["benchmarks"][blend]
    assert y.loc[2010, blend] == pytest.approx(bb.loc["2010"].iloc[-1] / bb.loc[:"2009"].iloc[-1] - 1, abs=1e-9)
    tr = A["trailing"]["Strategy"]
    assert tr["depleted_on"] == dep.date() and tr["1Y"] is None and str(tr["as_of"]) == str(end.date())
    assert A["trailing"][blend]["1Y"] is not None
    assert len(A["equity_full"]) == len(res.equity) and A["equity_full"].iloc[-1] == 0
    txt = report.console_summary(A)
    assert "ran out" in txt and "the report stops there" not in txt


@needs_data
def test_a_benchmark_that_cannot_be_shown_is_announced():
    res = run_sentence("hold 100% SPY from 1995 to 1998, vs 60/40 SPY/AGG")
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    assert not any(k.endswith(" blend") for k in A["benchmarks"])
    w = [x for x in A["warnings"] if x.get("code") == "benchmark_missing"]
    assert w and "60/40 SPY/AGG" in w[0]["detail"] and "2003-09-29" in w[0]["detail"]


@needs_data
def test_grid_run_out_does_not_cut_the_other_columns():
    from backtester import runner, web
    specs, _, problems = web.grid_specs({
        "rows": [{"asset": "QQQ", "w": [100, None]}, {"asset": "SPY", "w": [None, 60]}, {"asset": "AGG", "w": [None, 40]}],
        "names": ["QQQ", "60/40"], "start": "2004", "capital": 1_000_000,
        "flows": [{"kind": "withdraw", "amount": 30000, "freq": "annual"}], "rebalance": "yearly"})
    assert not problems
    for s in specs:          # one column empties the account fast, the other never does
        if s.name == "QQQ":
            s.withdrawal = 400_000.0
    As = [report.analyze(runner.run(s), sensitivity=False, mc=False, detail=False) for s in specs]
    assert As[0]["depleted"] is not None and As[1]["depleted"] is None
    C = report.common_window_stats(As)
    end = As[1]["result"].equity.index[-1]
    assert C["end"] == end.date() and C["depleted"] == {"QQQ": str(As[0]["depleted"])}
    assert C["columns"]["60/40"]["end"] == end.date()
    assert any(k.endswith("buy & hold") and v["end"] == end.date() for k, v in C["columns"].items())
    # the depleted column's partial year is marked, and the other column's rows go on
    y0 = As[0]["yearly"]
    d = pd.Timestamp(As[0]["depleted"])
    assert bool(y0.loc[d.year, "partial"])
    assert As[1]["yearly"].index.max() == end.year


@needs_data
def test_blended_benchmark_follows_the_portfolio_schedule():
    res = run_sentence("hold 60% SPY and 40% AGG since 2005, rebalance yearly, vs 60/40 SPY/AGG, no interest on cash")
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    blend = next(k for k in A["benchmarks"] if k.endswith(" blend"))
    b, nv = A["benchmarks"][blend], A["nav"]
    rb = b.pct_change().dropna()
    rn = nv.reindex(rb.index).pct_change().dropna()
    # (the portfolio is valued on quoted prices plus cash dividends, the blend on total-return prices)
    np.testing.assert_allclose(rb.reindex(rn.index).to_numpy(), rn.to_numpy(), atol=2e-4)
    assert any("rebalanced yearly like the portfolio" in n for n in res.strategy.notes)
    # stated: its own schedule
    res2 = run_sentence("hold 60% SPY and 40% AGG since 2005, rebalance yearly, vs 60/40 SPY/AGG rebalanced monthly")
    assert res2.strategy.benchmark_rebalance == "monthly"
    b2 = report.analyze(res2, sensitivity=False, mc=False, detail=False)["benchmarks"][blend]
    assert abs(float(b2.iloc[-1]) - float(b.iloc[-1])) > 1.0


@needs_data
def test_grid_blend_benchmark_is_rebalanced_like_the_columns():
    from backtester import web
    specs, _, problems = web.grid_specs({
        "rows": [{"asset": "US Stock Market", "w": [60]}, {"asset": "Total Bond Market", "w": [40]}],
        "start": "1990", "rebalance": "yearly", "benchmark": "60% US Stock Market 40% Total Bond Market"})
    assert not problems
    assert report.blend_rebalance(specs[0]) == ("yearly", "the portfolio's schedule")


@needs_data
def test_income_on_total_return_series_is_not_zero_dividends():
    res = run_sentence("hold 60% VTISIM and 40% BNDSIM since 1990, rebalance yearly")
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    assert A["income_all_total_return"] and set(A["income_total_return"]) == {"VTISIM", "BNDSIM"}
    assert A["income_yearly"]["dividends"].isna().all()
    assert report.pct(-0.0) == "0.00%" and report.pct(-1e-9) == "0.00%" and report.pct(-0.012) == "-1.20%"
    res2 = run_sentence("hold 60% SPY and 40% AGG since 2010, rebalance yearly")
    A2 = report.analyze(res2, sensitivity=False, mc=False, detail=False)
    assert not A2["income_all_total_return"] and A2["income_yearly"]["dividends"].sum() > 0


# ------------------------------------------------------------ 7. portfolio phrases

def parse(text):
    from backtester import parser
    return parser.parse(text)


def test_expense_ratio_either_word_order():
    for t in ("hold 60% SPY and 40% AGG, expense ratio 0.5%", "hold 60% SPY and 40% AGG, expense ratio of 0.5%",
              "hold 60% SPY and 40% AGG, 0.5% expense ratio", "hold 60% SPY and 40% AGG, expense ratio: 0.5%"):
        assert parse(t).expense_ratio == pytest.approx(0.005), t


def test_relative_drift_after_exceeds():
    p = parse("hold 60% SPY and 40% AGG, rebalance when drift exceeds 25% relative")
    assert p.drift_band_relative == pytest.approx(0.25) and not p.drift_band


def test_margin_rate_over_fed_funds():
    p = parse("hold 60% SPY and 40% AGG with 1.5x leverage, margin rate of fed funds plus 1%")
    assert p.margin_rate == pytest.approx(0.01)
    assert any("no fed funds series" in n for n in p.notes)
    p2 = parse("hold 60% SPY and 40% AGG with 1.5x leverage, margin rate T-bills + 1.5%")
    assert p2.margin_rate == pytest.approx(0.015)


@pytest.mark.parametrize("text", ["GTAA", "Faber GTAA", "Ivy timing", "Faber's Ivy timing", "GTAA 5 since 1990"])
def test_gtaa_named_model(text):
    p = parse(text)
    kids = p.tree["children"]
    assert p.tree["weights"] == "equal" and len(kids) == 5
    assert all(k["else"] == {"cash": True} and "monthly_sma(10)" in k["if"] for k in kids)
    held = [k["then"]["asset"] for k in kids]
    assert held == (["SPYSIM", "EFASIM", "IEFSIM", "VNQSIM", "DBCSIM"] if "1990" in text else ["SPY", "EFA", "IEF", "VNQ", "DBC"])
    assert p.rebalance == "monthly"


@pytest.mark.parametrize("text", ["Antonacci GEM", "global equities momentum", "dual momentum GEM", "Antonacci's GEM model"])
def test_gem_named_model(text):
    p = parse(text)
    t = p.tree
    assert t["on"] == "SPY" and t["if"] == "tret(252) > tbill_ret(252)"
    assert [c["asset"] for c in t["then"]["children"]] == ["SPY", "VEU"] and t["then"]["filter"]["by"] == "tret(252)"
    assert t["else"] == {"asset": "AGG"} and p.rebalance == "monthly"


def test_ivy_portfolio_without_timing_is_still_the_static_mix():
    p = parse("ivy portfolio")
    assert p.tree["weights"] == "specified" and len(p.tree["children"]) == 5


def test_weighted_lookbacks_skipping_the_last_month():
    p = parse("hold the top 1 of SPY, EFA and IEF by 3 month return weighted 50%, 6 month weighted 30% and 12 month "
              "weighted 20%, skipping the last month, rebalance monthly")
    assert p.tree["filter"]["by"] == "ref(0.5 * tret(tr, 42) + 0.3 * tret(tr, 105) + 0.2 * tret(tr, 231), 21)"


def test_blend_benchmark_schedule_phrases():
    p = parse("hold 60% SPY and 40% AGG, rebalance yearly, vs 60/40 SPY/AGG never rebalanced")
    assert p.benchmark_rebalance == "none"
    assert parse("hold 60% SPY and 40% AGG, rebalance quarterly, vs 60/40 SPY/AGG").benchmark_rebalance is None
    from backtester.parser import ParseError
    with pytest.raises(ParseError):
        parse("buy SPY when RSI(2) < 10, hold 3 days, vs 60/40 SPY/AGG rebalanced yearly")


@pytest.mark.parametrize("rebalance", ["yearly", "quarterly", "none"])
def test_blend_benchmark_matches_the_same_mix_on_the_same_schedule(fake, rebalance):
    """Invariant (synthetic data, no dividends): a 60/40 blended benchmark on the portfolio's schedule has the
    portfolio's daily returns to 1e-9."""
    fake["A"], fake["B"] = frame(walk(11, drift=0.0005, vol=0.015)), frame(walk(12, vol=0.004))
    fake["SPY"], fake["QQQ"] = fake["A"], fake["B"]
    res = pf.run(pf.Portfolio(tree=TREE, rebalance=rebalance, cash_rate=None, capital=10_000, benchmark="60 A 40 B"))
    b = report.benchmark_series(res)["60% A / 40% B blend"]
    rb, rp = b.pct_change().iloc[2:], res.equity.reindex(b.index).pct_change().iloc[2:]
    np.testing.assert_allclose(rb.to_numpy(), rp.to_numpy(), atol=1e-9)
