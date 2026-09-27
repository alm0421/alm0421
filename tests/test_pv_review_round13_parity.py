"""Round 13 Portfolio Visualizer review: annualising on the calendar's own bar frequency (weekend bars), the glide
path (dynamic allocation) in historical backtests, PV phrasings ("100% SPY", "25% each of ...", dual momentum's
hurdle restated), PV conventions (4% rule indexed once a year, expense ratio net of leverage, time- vs
money-weighted headline), the grid example reaching 1987, honest stress-test drawdowns, the faster (identical)
Monte Carlo and backtest paths, and the PCA and financial-goals tools."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import data, metrics, montecarlo as mc, parser, portfolio as P, report, web

AVAIL = set(data.available_tickers())


def need(*t):
    return pytest.mark.skipif(not set(t) <= AVAIL, reason=f"price data for {', '.join(t)} not available")


# ------------------------------------------------------------------ 1. annualisation on the series' own frequency

def _series(days: pd.DatetimeIndex, sd=0.02, mu=0.0005, seed=1) -> pd.Series:
    r = np.random.default_rng(seed).normal(mu, sd, len(days) - 1)
    return pd.Series(10_000 * np.concatenate([[1.0], np.cumprod(1 + r)]), index=days)


def test_periods_per_year_follows_the_calendar():
    seven = pd.date_range("2015-01-01", "2020-12-31", freq="D")
    five = pd.bdate_range("2015-01-01", "2020-12-31")
    assert metrics.periods_per_year(seven) == 365
    assert metrics.periods_per_year(five) == 252
    assert metrics.periods_per_year(pd.date_range("2000-01-07", periods=300, freq="W-FRI")) == 52
    assert metrics.periods_per_year(pd.date_range("2000-01-31", periods=120, freq="ME")) == 12
    assert metrics.periods_per_year(seven[:12]) == 365          # short: told apart by its weekend bars
    assert metrics.periods_per_year(five[:3]) == 252


def test_seven_day_series_annualised_with_365_bars():
    days = pd.date_range("2015-01-01", "2022-12-31", freq="D")
    eq = _series(days)
    st = metrics.equity_stats(eq, rf=0.0)
    r = eq.pct_change().iloc[1:]
    assert st["periods_per_year"] == 365
    assert st["volatility"] == pytest.approx(r.std() * np.sqrt(365), rel=1e-12)
    assert st["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(365), rel=1e-12)
    down = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(365)
    assert st["sortino"] == pytest.approx(r.mean() * 365 / down, rel=1e-12)
    # with a risk-free rate: 2%/yr over 365 bars, so a year of bars carries 2%
    st2 = metrics.equity_stats(eq, rf=0.02)
    assert st2["sharpe"] == pytest.approx((r - 0.02 / 365).mean() / r.std() * np.sqrt(365), rel=1e-12)
    # the same returns on a stock calendar keep 252
    eq5 = pd.Series(eq.to_numpy()[: len(pd.bdate_range("2015-01-01", periods=len(eq)))],
                    index=pd.bdate_range("2015-01-01", periods=len(eq)))
    st5 = metrics.equity_stats(eq5, rf=0.0)
    r5 = eq5.pct_change().iloc[1:]
    assert st5["periods_per_year"] == 252 and st5["volatility"] == pytest.approx(r5.std() * np.sqrt(252), rel=1e-12)


def test_relative_rolling_and_bootstrap_use_the_bar_frequency():
    days = pd.date_range("2016-01-01", "2023-12-31", freq="D")
    a, b = _series(days, seed=2), _series(days, seed=3)
    rel = metrics.relative_stats(a, b, rf=0.0)
    ra, rb = a.pct_change().dropna(), b.pct_change().dropna()
    beta = np.cov(ra, rb)[0, 1] / rb.var()
    assert rel["alpha_annual"] == pytest.approx((ra.mean() - beta * rb.mean()) * 365, rel=1e-9)
    assert rel["tracking_error"] == pytest.approx((ra - rb).std() * np.sqrt(365), rel=1e-9)
    ro = metrics.rolling_series(a, b, rf=0.0)
    first = ro["return_12m"].first_valid_index()
    assert first == a.index[365]                           # a year of 365 bars, not 252
    assert ro["vol_3m"].iloc[200] == pytest.approx(a.pct_change().iloc[200 - 90:201].std() * np.sqrt(365), rel=1e-9)
    mcr = metrics.monte_carlo(a, sims=200)           # the bootstrap's years are bars / 365
    g = (a.iloc[-1] / a.iloc[0])
    years = (len(a) - 1) / 365
    assert abs(np.log(1 + mcr["cagr_p50"]) - np.log(g) / years) < 0.2


def test_seven_day_interest_accrues_one_year_per_year():
    days = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    r = P._daily_rate(days, 0.05, metrics.periods_per_year(days))
    assert r.sum() == pytest.approx(0.05 * len(days) / 365)


@need("BTC-USD", "SPY")
def test_btc_spy_portfolio_annualised_on_365_bars():
    res = P.run(parser.parse("hold 50% BTC-USD and 50% SPY, rebalance monthly"))
    st = metrics.equity_stats(res.equity)
    r = res.equity.pct_change().iloc[1:]
    assert st["periods_per_year"] == 365
    assert st["volatility"] == pytest.approx(r.std() * np.sqrt(365), rel=1e-9)
    assert 0.33 < st["volatility"] < 0.45                  # was 30% with sqrt(252)
    A = report.analyze(res, sensitivity=False)
    assert "Annualised on     365 bars a year" in report.console_summary(A)


# ------------------------------------------------------------------ 2. glide path (dynamic allocation)

def _glide_pf(**g):
    return P.Portfolio(tree={"weights": "specified", "w": [0.9, 0.1], "children": [{"asset": "VTI"}, {"asset": "BND"}]},
                       rebalance="yearly", glide={"to": {"VTI": 0.4, "BND": 0.6}, **g})


@need("VTI", "BND")
def test_glide_targets_follow_the_date_linearly():
    p = _glide_pf(years=30)
    res = P.run(p)
    gt = res.extras["glide_targets"]
    s0 = res.equity.index[1] if res.extras["day0"] else res.equity.index[1]
    for d, row in gt.iloc[1:].iterrows():
        f = min((d - s0).days / 365.25 / 30, 1.0)
        assert row["VTI"] == pytest.approx(0.9 - 0.5 * f, abs=1e-12)
        assert row["BND"] == pytest.approx(0.1 + 0.5 * f, abs=1e-12)
    assert np.allclose(gt.sum(axis=1), 1.0)
    # right after a rebalance the holdings are at that day's target
    for d in gt.index[1:4]:
        assert res.holdings.loc[d, "VTI"] == pytest.approx(gt.loc[d, "VTI"], abs=1e-9)
    assert "Glide path: from 90% VTI, 10% BND to 40% VTI, 60% BND over 30 years" in p.summary()


@need("VTI", "BND")
def test_glide_has_no_lookahead():
    full = P.run(_glide_pf(years=20))
    cut = P.run(dataclass_replace(_glide_pf(years=20), end="2016-06-30"))
    common = cut.equity.index[:-1]
    assert np.allclose(full.equity.reindex(common).to_numpy(), cut.equity.reindex(common).to_numpy(), rtol=0, atol=1e-9)
    g_full, g_cut = full.extras["glide_targets"], cut.extras["glide_targets"]
    assert g_full.loc[g_cut.index].equals(g_cut)


def dataclass_replace(p, **kw):
    import dataclasses
    return dataclasses.replace(p, **kw)


@need("VTI", "BND")
def test_glide_per_year_end_date_and_target_date_shapes():
    p = _glide_pf(per_year=0.02)                           # 50 points at 2 a year: 25 years
    P.check_glide(p)
    assert P.glide_years(p, pd.Timestamp("2010-01-01")) == pytest.approx(25)
    p2 = _glide_pf(end="2040")
    assert P.glide_years(p2, pd.Timestamp("2010-01-01")) == pytest.approx((pd.Timestamp("2040-01-01") - pd.Timestamp("2010-01-01")).days / 365.25)
    p3 = _glide_pf(years=30, shape="target_date")
    s0 = pd.Timestamp("2010-01-01")
    assert P.glide_fraction_at(p3, s0 + pd.Timedelta(days=int(365.25 * 5)), s0) == 0.0   # the first fifth holds
    assert P.glide_fraction_at(p3, s0 + pd.Timedelta(days=int(365.25 * 40)), s0) == 1.0
    p4 = _glide_pf(years=10, step="yearly")
    assert P.glide_fraction_at(p4, s0 + pd.Timedelta(days=200), s0) == 0.0
    assert P.glide_fraction_at(p4, s0 + pd.Timedelta(days=int(365.25 * 9) + 2), s0) == 1.0
    for bad in ({"years": 30, "end": "2040"}, {}, {"years": 30, "shape": "bogus"}):
        with pytest.raises(ValueError):
            P.check_glide(_glide_pf(**bad))
    with pytest.raises(ValueError, match="fixed mix"):
        P.check_glide(P.Portfolio(tree={"if": "close > sma(close, 200)", "on": "VTI", "then": {"asset": "VTI"},
                                        "else": {"asset": "BND"}}, glide={"to": {"BND": 1.0}, "years": 10}))
    with pytest.raises(ValueError, match="rebalanc"):
        P.check_glide(dataclass_replace(_glide_pf(years=10), rebalance="none"))


@need("VTI", "BND")
def test_glide_sentences():
    p = parser.parse("hold 90% VTI and 10% BND gliding to 40% VTI and 60% BND over 30 years")
    assert p.glide["to"] == {"VTI": 0.4, "BND": 0.6} and p.glide["years"] == 30
    p = parser.parse("hold 90% VTI and 10% BND gliding to 40% VTI and 60% BND linearly by 2% a year, rebalance yearly")
    assert p.glide["per_year"] == pytest.approx(0.02) and p.rebalance == "yearly"
    p = parser.parse("hold 90% VTI and 10% BND, target date 2050 glide path")
    assert p.glide["shape"] == "target_date" and p.glide["end"] == "2050" and p.glide["to"] == {"VTI": 0.4, "BND": 0.6}
    assert any("no end mix was given" in n for n in p.notes)
    p = parser.parse("hold 90% VTI and 10% BND gliding to 40% VTI and 60% BND by 2040, never rebalance")
    assert p.rebalance == "yearly" and any("moves the target at rebalances" in n for n in p.notes)
    # a spec round trip keeps it
    from backtester import runner
    q = runner.from_dict(json.loads(p.to_json()))
    assert q.glide == p.glide


@need("VTI", "BND")
def test_glide_in_the_report_and_the_grid():
    res = P.run(_glide_pf(years=30))
    A = report.analyze(res, sensitivity=False)
    g = report.glide_payload(res)
    assert g["columns"] == ["VTI", "BND"] and g["values"]["VTI"][0] == pytest.approx(0.9)
    assert g["text"].startswith("Glide path:")
    assert "Glide targets" in report.console_summary(A)
    grid = {"rows": [{"asset": "VTI", "w": [90, 40]}, {"asset": "BND", "w": [10, 60]}], "names": ["TD", "End"],
            "start": "2010", "rebalance": "yearly", "glide": {"from": 0, "to": 1, "years": 30}}
    specs, notes, problems = web.grid_specs(grid)
    assert problems == [] and len(specs) == 1
    assert specs[0].glide["to"] == {"VTI": 0.4, "BND": 0.6} and specs[0].glide["years"] == 30
    grid["glide"] = {"from": 0, "to": 0, "years": 30}
    assert web.grid_specs(grid)[2]


def test_glide_step_is_in_the_page_and_report():
    html = (Path(web.__file__).parent / "report_template.html").read_text()
    assert 'id="glideTbl"' in html and "r.glide" in html
    app = (Path(web.__file__).parent / "webapp.html").read_text()
    assert 'id="g_glide"' in app and "b.glide = {" in app


# ------------------------------------------------------------------ 3. parser phrasings

@need("SPY", "VTI")
def test_single_holding_as_a_weight():
    for s, t in (("100% SPY", "SPY"), ("100% VTI", "VTI"), ("VTI 100", "VTI"), ("VTI 100%", "VTI")):
        p = parser.parse(s)
        assert P._fixed_mix(p.tree) == {t: 1.0}, s


@need("SPY", "AGG")
def test_weights_after_tickers():
    p = parser.parse("SPY 60 AGG 40, rebalance yearly")
    assert P._fixed_mix(p.tree) == {"SPY": 0.6, "AGG": 0.4} and p.rebalance == "yearly"
    with pytest.raises(parser.ParseError, match="add up to 90%"):
        parser.parse("SPY 60 AGG 30")


def test_x_percent_each_of_a_list():
    p = parser.parse("hold 25% each of US large cap growth, US large cap value, US small cap growth, US small cap value, since 1930")
    mix = P._fixed_mix(p.tree)
    assert len(mix) == 4 and all(v == pytest.approx(0.25) for v in mix.values())
    assert p.start == "1930-01-01"
    assert any("4 x 25% = 100%" in n for n in p.notes)
    with pytest.raises(parser.ParseError, match="adds up to 90%, not 100%"):
        parser.parse("hold 30% each of US large cap growth, US large cap value and US small cap value")
    with pytest.raises(parser.ParseError, match="not 100%"):      # tickers: the parser's own reading, same check
        parser.parse("hold 20% each of SPY, AGG and GLD")


@need("SPY", "EFA", "AGG", "BIL")
def test_dual_momentum_hurdle_restated():
    base = "dual momentum between SPY and EFA with AGG as the safe asset"
    p = parser.parse(base + ", only if their 12 month return is above BIL's 12 month return")
    assert p.tree["filter"]["require"] == 'tret(252) > tret(sym("BIL").tr, 252)'
    assert any("applied as written" in n for n in p.notes)
    q = parser.parse(base + ", only if their 12 month return is above T-bills")
    assert q.tree["filter"]["require"] == parser.parse(base).tree["filter"]["require"]
    assert any("restates dual momentum's built-in" in n for n in q.notes)


# ------------------------------------------------------------------ 4. PV conventions

@need("SPY", "AGG")
def test_annual_inflation_indexing_pays_exactly_the_amount_in_year_one():
    def flows(ind):
        p = P.Portfolio(tree={"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "AGG"}]},
                        rebalance="yearly", capital=1_000_000, withdrawal=40_000 / 12, withdrawal_freq="monthly",
                        withdrawal_inflation=True, start="2005-01-01", end="2012-12-31", inflation_indexing=ind)
        r = P.run(p)
        f = -r.extras["flows"]
        return f[f > 0], p
    fa, pa = flows("annual")
    by_year = fa.groupby(fa.index.year).sum()
    assert by_year.loc[2005] == pytest.approx(40_000, abs=1e-6)
    for y, g in fa.groupby(fa.index.year):
        assert g.max() - g.min() < 1e-6                      # fixed within a year of the flow
    assert by_year.loc[2006] > by_year.loc[2005]
    fp, _ = flows("published")
    assert fp.groupby(fp.index.year).sum().loc[2005] != pytest.approx(40_000, abs=1)
    assert any("indexed once a year (Portfolio Visualizer's convention" in n for n in pa.notes)


@need("SPY", "AGG")
def test_pv_defaults_choose_annual_indexing_and_say_so():
    p = parser.parse("hold 60% SPY and 40% AGG, start with $1,000,000, withdraw 4% a year adjusted for inflation, "
                     "with Portfolio Visualizer defaults")
    assert p.inflation_indexing == "annual"
    assert any("stepped up once a year" in n for n in p.notes)
    assert parser.parse("hold 60% SPY and 40% AGG, withdraw $40,000 a year adjusted for inflation annually").inflation_indexing == "annual"
    assert parser.parse("hold 60% SPY and 40% AGG, withdraw $40,000 a year adjusted for inflation").inflation_indexing == "published"


@need("SPY", "AGG")
def test_expense_ratio_net_of_leverage():
    kw = dict(tree={"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "AGG"}]},
              rebalance="monthly", leverage=2.0, expense_ratio=0.01, start="2015-01-01", end="2019-12-31")
    gross = P.run(P.Portfolio(**kw)).extras["fees"]
    net = P.run(P.Portfolio(expense_on="equity", **kw)).extras["fees"]
    assert net == pytest.approx(gross / 2, rel=0.08)          # half the gross exposure at 2x
    one = dict(kw, leverage=1.0)
    # unlevered the account value is the holdings plus a little cash (either way): about the same fee
    assert P.run(P.Portfolio(**one)).extras["fees"] == pytest.approx(P.run(P.Portfolio(expense_on="equity", **one)).extras["fees"], rel=2e-3)
    p = parser.parse("hold 60% SPY and 40% AGG with 2x leverage and a 0.5% expense ratio net of leverage")
    assert p.expense_on == "equity" and p.expense_ratio == pytest.approx(0.005)
    assert "net of leverage" in p.summary()
    assert "expense_on 'equity' charges it net of leverage" in parser.parse(
        "hold 60% SPY and 40% AGG with 2x leverage and a 0.5% expense ratio").summary()


@need("SPY", "AGG")
def test_headline_labels_time_weighted_and_money_weighted():
    res = P.run(parser.parse("hold 60% SPY and 40% AGG, start with $1,000,000, withdraw $60,000 a year, since 2005"))
    A = report.analyze(res, sensitivity=False)
    txt = report.console_summary(A)
    assert re.search(r"Time-weighted     total [-\d.]+%", txt)
    assert re.search(r"Money-weighted    IRR [-\d.]+%/yr", txt)
    assert "Total return      " not in txt
    tpl = (Path(report.__file__).parent / "report_template.html").read_text()
    assert "'Time-weighted return'" in tpl and "money-weighted (IRR)" in tpl


# ------------------------------------------------------------------ 5. the grid example reaches its start

def test_grid_example_uses_asset_classes_back_to_1987():
    app = (Path(web.__file__).parent / "webapp.html").read_text()
    m = re.search(r"\$\('gExample'\)\.addEventListener\('click', \(\) => fill\((\{.*?\})\)\);", app, re.S)
    assert m
    js = m.group(1)
    assert "AGG" not in js and "SPY" not in js and "start: '1987'" in js
    rows = [{"asset": a, "w": [float(x) if x != "null" else None for x in w.split(", ")]}
            for a, w in re.findall(r"asset: '([^']+)', w: \[([^\]]+)\]", js)]
    specs, notes, problems = web.grid_specs({"rows": rows, "names": ["60/40", "Three-fund"], "start": "1987", "rebalance": "yearly"})
    assert problems == [] and len(specs) == 2
    assert all(s.start == "1987-01-01" for s in specs)
    assert not any("constrained by the available data" in n for n in notes)
    for s in specs:
        assert max(data.load(t).index[0] for t in s.universe) <= pd.Timestamp("1987-01-01")


# ------------------------------------------------------------------ 6. stress-test drawdowns

@need("SPY", "AGG")
def test_stress_drawdown_is_shown_on_its_own():
    s = mc.Settings(weights={"SPY": 0.6, "AGG": 0.4}, years=30, sims=600, stress="worst_sequence",
                    flows=[mc.CashFlow(amount=-40_000)])
    R = mc.run(s)
    st = R["stress"]
    assert st["max_drawdown"] < 0 and 0 <= st["deepest_share"] <= 1
    after = list(R["max_drawdown_after_stress"].values())
    assert len(set(round(v, 6) for v in after)) > 1          # varies by path
    assert len(set(round(v, 6) for v in R["max_drawdown_unstressed"].values())) > 1
    assert all(v <= st["max_drawdown"] + 1e-12 for v in R["max_drawdown"].values())   # a whole path holds the stress
    txt = mc.console(R)
    assert "of the stressed months" in txt and "after the stress" in txt and "without the stress" in txt
    assert any("Max drawdown under the stress test" in n for n in R["notes"])


# ------------------------------------------------------------------ 7. faster, identical

def _alive_bruteforce(P_, ci, start, rate, every):
    sims, months = P_.shape
    b = np.full(sims, float(start))
    alive = np.ones(sims, bool)
    for m in range(months):
        if m % every == 0:
            b = b - rate * start * every / 12 * ci[:, m]
            alive &= b > 0
            b = np.maximum(b, 0.0)
        b = b * (1 + P_[:, m])
    return alive, b / ci[:, months]


def test_withdrawal_rates_closed_form_match_the_simulation():
    rng = np.random.default_rng(5)
    P_ = rng.normal(0.006, 0.04, (400, 240))
    ci = np.concatenate([np.ones((400, 1)), np.cumprod(1 + rng.normal(0.002, 0.002, (400, 240)), axis=1)], axis=1)
    for every in (12, 1):
        x = mc._withdrawal_thresholds(P_, ci, 0, every, every / 12)[0]
        for rate in (0.02, 0.04, 0.06, 0.09):
            alive, _ = _alive_bruteforce(P_, ci, 1.0, rate, every)
            assert ((rate < x) == alive).mean() > 0.999
        swr = mc.safe_withdrawal_rate(P_, ci, 1.0, 0.9, every)
        assert _alive_bruteforce(P_, ci, 1.0, swr, every)[0].mean() >= 0.9
        assert _alive_bruteforce(P_, ci, 1.0, swr + 1e-6, every)[0].mean() < 0.9 + 1e-9
    swr_p, pwr_p = mc.withdrawal_rates_by_path(P_, ci, np.full(400, 1e6))
    for i in range(0, 400, 50):
        a = _alive_bruteforce(P_[i:i + 1], ci[i:i + 1], 1.0, swr_p[i], 12)[0][0]
        b = _alive_bruteforce(P_[i:i + 1], ci[i:i + 1], 1.0, swr_p[i] + 1e-9, 12)[0][0]
        assert a and not b


def _mc_flows_reference(equity, flows, sims=300, block=20, seed=7):
    """The day-by-day flow replay of metrics.monte_carlo before round 13."""
    r = metrics.twr_returns(equity, flows).to_numpy()
    n, m = len(equity) - 1, len(r)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    S = np.array([rng.integers(0, m - block, nb) for _ in range(sims)]).reshape(sims, nb)
    fl = flows.reindex(equity.index).fillna(0.0).to_numpy()[1:]
    v = np.full(sims, float(equity.iloc[0]))
    alive = np.ones(sims, bool)
    for t in range(n):
        nxt = v * (1 + r[S[:, t // block] + t % block]) + fl[t]
        v = np.where(alive, nxt, v)
        dead = alive & (v <= 0)
        if dead.any():
            v[dead] = 0.0
            alive &= ~dead
    return v, alive


@need("SPY", "AGG")
def test_bootstrap_flow_replay_is_bit_identical():
    for kw in (dict(contribution=500, contribution_freq="monthly", withdrawal=30_000, withdrawal_start=8),
               dict(withdrawal=120_000)):
        p = P.Portfolio(tree={"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "AGG"}]},
                        rebalance="yearly", capital=1_000_000, start="2004-01-01", **kw)
        res = P.run(p)
        fl = res.extras["flows"]
        out = metrics.monte_carlo(res.equity, fl, sims=300)
        v, alive = _mc_flows_reference(res.equity, fl)
        assert out["final_p50"] == float(np.percentile(v, 50)) and out["final_p5"] == float(np.percentile(v, 5))
        assert out["success_rate"] == 1 - (~alive).sum() / 300


@pytest.mark.parametrize("sentence", [
    "hold 60% SPY and 40% AGG, rebalance yearly",
    "hold 33% QQQ, 33% TLT and 34% GLD, rebalance monthly, withdraw 4% a year",
    "hold 60% SPY and 40% TLT, never rebalance, add $500 a month",
    "hold 50% AAPL and 50% MSFT, rebalance quarterly, since 1995",
])
def test_quiet_day_fast_path_is_bit_identical(sentence, monkeypatch):
    if not set(parser.find_tickers(sentence)) <= AVAIL:
        pytest.skip("data")
    out = []
    for fast in (False, True):
        monkeypatch.setattr(P, "FAST_QUIET_DAYS", fast)
        out.append(P.run(parser.parse(sentence)))
    a, b = out
    assert a.equity.to_numpy().tobytes() == b.equity.to_numpy().tobytes()
    assert a.holdings.equals(b.holdings) and a.interest == b.interest
    assert a.extras["income"].equals(b.extras["income"])


@need("SPY", "AGG")
def test_flow_free_rerun_reuses_evaluations_with_the_same_curve():
    p = parser.parse("hold 60% SPY and 40% AGG, rebalance yearly, start with $500,000, withdraw $30,000 a year")
    res = P.run(p)
    a = report.flow_free_nav(p, res.equity.index[0])
    b = report.flow_free_nav(p, res.equity.index[0], res)
    assert a.equals(b)


# ------------------------------------------------------------------ 8. PCA and financial goals

def test_pca_components_on_synthetic_returns():
    from backtester import pca
    rng = np.random.default_rng(0)
    f = rng.normal(0, 0.04, 300)
    r = pd.DataFrame({"A": f + rng.normal(0, 0.005, 300), "B": f + rng.normal(0, 0.005, 300),
                      "C": rng.normal(0, 0.03, 300)}, index=pd.date_range("2000-01-31", periods=300, freq="ME"))
    comps, sd, share = pca.components(r, "correlation")
    assert share.sum() == pytest.approx(1.0) and np.all(np.diff(share) <= 1e-12)
    assert comps[0]["explained"] > 0.6                       # A and B move together
    L = np.array([[c["loadings"][t] for t in "ABC"] for c in comps])
    assert np.allclose(L @ L.T, np.eye(3), atol=1e-10)         # orthonormal
    assert sum(comps[0]["loadings"].values()) > 0
    assert abs(comps[0]["loadings"]["C"]) < 0.2
    cc, _, sh2 = pca.components(r, "covariance")
    ev = np.linalg.eigvalsh(np.cov(r.to_numpy(), rowvar=False))[::-1]
    assert np.allclose([c["eigenvalue"] for c in cc], ev) and cc[0]["volatility_annual"] == pytest.approx(np.sqrt(ev[0] * 12))


@need("SPY", "AGG", "GLD")
def test_pca_cli_and_api():
    from backtester import pca
    R = pca.analyze(["SPY", "AGG", "GLD"])
    assert R["basis"] == "correlation" and len(R["components"]) == 3
    assert "Principal components of monthly total returns" in pca.console(R)
    J = web.api_pca({"tickers": ["SPY", "AGG", "GLD"], "basis": "covariance"})
    assert J["components"][0]["volatility_annual"] > 0
    with pytest.raises(web.ClientError):
        web.api_pca({"tickers": ["SPY"]})


def test_parse_goals():
    from backtester import goals as G
    g = G.parse_goal("College: withdraw 60000 a year from year 8 to year 11")
    assert (g.name, g.kind, g.amount, g.freq, g.start_year, g.end_year) == ("College", "withdraw", 60000, "yearly", 8, 11)
    g = G.parse_goal("House: withdraw $150k in year 12")
    assert (g.amount, g.freq, g.start_year) == (150000, "once", 12)
    g = G.parse_goal("Savings: contribute 20,000 a year for 15 years")
    assert (g.kind, g.start_year, g.end_year) == ("contribute", 1, 15)
    g = G.parse_goal("Retirement: withdraw 70000 a year from 2045, fixed dollars", this_year=2026)
    assert (g.start_year, g.end_year, g.inflation_adjusted) == (20, None, False)
    with pytest.raises(ValueError):
        G.parse_goal("buy a boat someday")


@need("SPY", "AGG")
def test_goals_probabilities_on_the_monte_carlo_paths():
    from backtester import goals as G
    s = mc.Settings(weights={"SPY": 0.6, "AGG": 0.4}, start_balance=500_000, years=25, sims=500, inflation=0.025)
    goals = [G.Goal("Small", 1_000, "withdraw", 3, None, "once"), G.Goal("Huge", 5_000_000, "withdraw", 10, None, "once"),
             G.Goal("Save", 10_000, "contribute", 1, 10, "yearly"), G.Goal("Income", 30_000, "withdraw", 15, None, "yearly")]
    R = G.run(s, goals)
    by = {g["name"]: g for g in R["goals"]}
    assert by["Small"]["probability"] == 1.0 and by["Huge"]["probability"] == 0.0
    assert by["Save"]["probability"] is None
    assert 0 <= by["Income"]["probability"] <= 1 and 0 <= R["all_goals_met"] <= by["Income"]["probability"]
    assert "_paths" not in R
    assert "Financial goals" in G.console(R)
    # the per-goal replay has the same balances as the Monte Carlo run
    rng = np.random.default_rng(1)
    P_ = rng.normal(0.005, 0.03, (50, 120))
    ci = np.concatenate([np.ones((50, 1)), np.cumprod(np.full((50, 120), 1.002), axis=1)], axis=1)
    flows = [g.cash_flow() for g in goals]
    a = mc.simulate(P_, ci, 100_000, flows)
    b = mc.simulate(P_, ci, 100_000, flows, by_flow=True)
    assert np.array_equal(a["B"], b["B"])
    wd = [i for i, g in enumerate(goals) if g.kind == "withdraw"]
    assert np.allclose(b["flow_paid"][wd].sum(axis=0), a["withdrawn"], rtol=1e-9)


@need("SPY", "AGG")
def test_goals_api():
    J = web.api_goals({"weights": "SPY 60, AGG 40", "balance": 300000, "years": 20, "sims": 300,
                       "goals": [{"name": "Car", "kind": "withdraw", "amount": 30000, "freq": "once", "start_year": 5},
                                 "Travel: withdraw 10000 a year from year 2 to year 6"]})
    assert [g["name"] for g in J["goals"]] == ["Travel", "Car"]
    assert all(0 <= g["probability"] <= 1 for g in J["goals"])
    with pytest.raises(web.ClientError):
        web.api_goals({"weights": "SPY 60, AGG 40", "goals": []})
    app = (Path(web.__file__).parent / "webapp.html").read_text()
    assert "api('/api/goals'" in app and "api('/api/pca'" in app
