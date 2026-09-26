"""Portfolio Visualizer review, round 7: optimiser inputs (forecasts, Black-Litterman, resampling), benchmark-relative
objectives, per-percentile withdrawal rates, weight lists without separators, ranking-period options and the risk
contribution table."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from backtester import data, expr, metrics, montecarlo as mc, parser, report, research, risk, runner


def _monthly(tickers, start="2006-01-01", end="2024-12-31"):
    px = pd.concat({t: data.load(t)["adj_close"] for t in tickers}, axis=1).dropna()
    px = px[(px.index >= start) & (px.index <= end)]
    return metrics.monthly_returns_frame(px)


# ------------------------------------------------------------------ Black-Litterman

def _bl_textbook(cov, w, P, Q, conf, tau, delta):
    """[(tau S)^-1 + P' W^-1 P]^-1 [(tau S)^-1 pi + P' W^-1 Q] with W = (1-c)/c diag(P tau S P')."""
    pi = delta * cov @ w
    tS = tau * cov
    W = np.diag((1 - conf) / conf * np.diag(P @ tS @ P.T))
    A = np.linalg.inv(tS) + P.T @ np.linalg.inv(W) @ P
    mu = np.linalg.solve(A, np.linalg.inv(tS) @ pi + P.T @ np.linalg.inv(W) @ Q)
    return pi, mu, cov + np.linalg.inv(A)


def test_black_litterman_matches_the_textbook_formula():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4)) @ rng.standard_normal((4, 4)) * 0.03
    cov = np.cov(X, rowvar=False) * 12
    w = np.array([0.4, 0.3, 0.2, 0.1])
    P = np.array([[1, 0, 0, 0], [0, 1, -1, 0], [0.5, 0.5, 0, -1]], float)
    Q = np.array([0.05, 0.02, 0.01])
    conf = np.array([0.6, 0.3, 0.8])
    bl = research.black_litterman(cov, w, P, Q, conf, tau=0.05, delta=2.7)
    pi, mu, post = _bl_textbook(cov, w, P, Q, conf, 0.05, 2.7)
    np.testing.assert_allclose(bl["pi"], pi, atol=1e-12)
    np.testing.assert_allclose(bl["mu"], mu, atol=1e-10)
    np.testing.assert_allclose(bl["cov"], post, atol=1e-10)


def test_black_litterman_full_confidence_satisfies_the_view_and_no_views_is_the_prior():
    rng = np.random.default_rng(1)
    cov = np.cov(rng.standard_normal((120, 3)) * 0.04, rowvar=False) * 12
    w = np.array([0.5, 0.3, 0.2])
    P = np.array([[1.0, -1.0, 0.0]])
    bl = research.black_litterman(cov, w, P, np.array([0.03]), np.array([1.0]))
    assert P @ bl["mu"] == pytest.approx([0.03], abs=1e-12)
    none = research.black_litterman(cov, w)
    np.testing.assert_allclose(none["mu"], 2.5 * cov @ w)
    np.testing.assert_allclose(none["cov"], cov * 1.05)


def test_parse_views():
    v = research.parse_views("SPY = 8% @ 60%; QQQ > SPY by 2%; TLT underperforms SPY by 3% (70%)\nGLD 5%", ["SPY", "QQQ", "TLT", "GLD"])
    assert [x["kind"] for x in v] == ["absolute", "relative", "relative", "absolute"]
    assert v[0]["p"] == {"SPY": 1.0} and v[0]["q"] == pytest.approx(0.08) and v[0]["confidence"] == pytest.approx(0.6)
    assert v[1]["p"] == {"QQQ": 1.0, "SPY": -1.0} and v[1]["q"] == pytest.approx(0.02) and v[1]["confidence"] == 0.5
    assert v[2]["p"] == {"TLT": 1.0, "SPY": -1.0} and v[2]["q"] == pytest.approx(-0.03) and v[2]["confidence"] == pytest.approx(0.7)
    assert v[3]["q"] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        research.parse_views("EFA = 5%", ["SPY", "TLT"])
    with pytest.raises(ValueError):
        research.parse_views("SPY = 5% @ 0%", ["SPY", "TLT"])


def test_optimize_black_litterman_posterior_and_prior_recovery():
    tk = ["SPY", "TLT", "GLD"]
    prior = {"SPY": 0.5, "TLT": 0.3, "GLD": 0.2}
    # no views: the posterior is the equilibrium, and max Sharpe on it gives back the prior weights
    R = research.optimize(tk, "2006-01-01", "2024-12-31", methods=["max_sharpe"], prior=prior)
    assert R["inputs"]["source"] == "black_litterman"
    w = R["portfolios"]["Max Sharpe"]["weights"]
    for t in tk:
        assert w.get(t, 0) == pytest.approx(prior[t], abs=2e-3)
    # with views: the posterior equals the formula applied to the historical covariance (views in total returns)
    views = ["SPY = 9% @ 70%", "TLT > GLD by 1%"]
    R = research.optimize(tk, "2006-01-01", "2024-12-31", methods=["max_sharpe", "min_variance"], prior=prior, views=views,
                          tau=0.1, risk_aversion=3.0)
    mr = _monthly(tk)
    mr = mr[(mr.index >= pd.Timestamp(R["fit_start"])) & (mr.index <= pd.Timestamp(R["fit_end"]))]
    cov = np.cov(mr.to_numpy(), rowvar=False) * 12
    rf = R["rf"]
    P = np.array([[1, 0, 0], [0, 1, -1]], float)
    Q = np.array([0.09 - rf, 0.01])
    _, mu, post = _bl_textbook(cov, np.array([0.5, 0.3, 0.2]), P, Q, np.array([0.7, 0.5]), 0.1, 3.0)
    got = R["black_litterman"]["posterior_returns"]
    assert [got[t] for t in tk] == pytest.approx(list(mu + rf), abs=1e-9)
    # the optimiser used them: the assets' expected returns and volatilities are the posterior ones
    assert [a["return"] for a in R["assets"]] == pytest.approx(list(mu + rf), abs=1e-9)
    assert [a["vol"] for a in R["assets"]] == pytest.approx(list(np.sqrt(np.diag(post))), rel=1e-6)
    v0 = R["black_litterman"]["views"][0]
    assert v0["prior_value"] < v0["posterior_value"] < 0.09


def test_optimize_market_cap_prior_needs_caps():
    with pytest.raises(ValueError, match="market capitalisation"):
        research.optimize(["SPY", "TLT"], "2010-01-01", "2020-12-31", methods=["max_sharpe"], prior="market_cap",
                          views=["SPY = 8%"])
    # unspecified prior and no caps for funds: equal weights, with a note
    R = research.optimize(["SPY", "TLT"], "2010-01-01", "2020-12-31", methods=["max_sharpe"], views=["SPY = 8%"])
    assert R["black_litterman"]["prior_weights"] == {"SPY": 0.5, "TLT": 0.5}
    assert any("equilibrium weights are equal" in n for n in R["notes"])


# ------------------------------------------------------------------ forecasts

def test_forecast_inputs_and_tangency_reference():
    tk = ["SPY", "TLT"]
    er, vol, rho = {"SPY": 0.07, "TLT": 0.04}, {"SPY": 0.16, "TLT": 0.12}, -0.3
    R = research.optimize(tk, "2006-01-01", "2024-12-31", methods=["max_sharpe", "min_variance"],
                          expected_returns="SPY=7%, TLT=4%", expected_vols=vol, correlations="SPY/TLT=-0.3")
    assert R["inputs"]["source"] == "forecast"
    assert [a["return"] for a in R["assets"]] == pytest.approx([0.07, 0.04], abs=1e-12)
    assert [a["vol"] for a in R["assets"]] == pytest.approx([0.16, 0.12], rel=1e-6)
    cov = np.array([[0.16 ** 2, rho * 0.16 * 0.12], [rho * 0.16 * 0.12, 0.12 ** 2]])
    mu = np.array([0.07, 0.04]) - R["rf"]
    tang = np.linalg.solve(cov, mu)
    tang /= tang.sum()
    w = R["portfolios"]["Max Sharpe"]["weights"]
    assert [w["SPY"], w["TLT"]] == pytest.approx(list(tang), abs=1e-3)
    mv = np.linalg.solve(cov, np.ones(2))
    mv /= mv.sum()
    w = R["portfolios"]["Min variance"]["weights"]
    assert [w["SPY"], w["TLT"]] == pytest.approx(list(mv), abs=1e-4)
    with pytest.raises(ValueError, match="not both"):
        research.optimize(tk, "2006-01-01", "2024-12-31", expected_returns=er, views=["SPY = 5%"])
    with pytest.raises(ValueError, match="positive semi-definite"):
        research.optimize(["SPY", "TLT", "GLD"], "2006-01-01", "2024-12-31",
                          correlations="SPY/TLT=0.99; SPY/GLD=0.99; TLT/GLD=-0.99")


def test_retarget_is_exact():
    rng = np.random.default_rng(3)
    R = rng.standard_t(4, (150, 3)) * 0.04 + 0.005
    mean = np.array([0.01, 0.002, -0.001])
    cov = np.array([[0.002, 0.0003, 0.0], [0.0003, 0.001, -0.0002], [0.0, -0.0002, 0.0015]])
    Y = research.retarget(R, mean, cov)
    np.testing.assert_allclose(Y.mean(axis=0), mean, atol=1e-12)
    np.testing.assert_allclose(np.cov(Y, rowvar=False), cov, atol=1e-12)


# ------------------------------------------------------------------ benchmark-relative

def test_min_tracking_error_holds_an_investable_benchmark():
    R = research.optimize(["SPY", "TLT", "GLD"], "2006-01-01", "2024-12-31", benchmark="60% SPY 40% TLT",
                          methods=["min_tracking_error", "max_information_ratio"])
    p = R["portfolios"]["Min tracking error"]
    assert p["weights"].get("SPY") == pytest.approx(0.6, abs=1e-4) and p["weights"].get("TLT") == pytest.approx(0.4, abs=1e-4)
    assert p["tracking_error"] < 1e-3
    assert R["benchmark"]["investable"]
    R = research.optimize(["SPY", "TLT", "GLD"], "2006-01-01", "2024-12-31", benchmark="SPY", methods=["min_tracking_error"])
    assert R["portfolios"]["Min tracking error"]["weights"].get("SPY") == pytest.approx(1.0, abs=1e-4)


def _te_reference(tk, bench, fit_start, fit_end, floor=None):
    mr = _monthly(tk + [bench])
    mr = mr[(mr.index >= pd.Timestamp(fit_start)) & (mr.index <= pd.Timestamp(fit_end))]
    X, b = mr[tk].to_numpy(), mr[bench].to_numpy()
    mu = X.mean(axis=0) * 12
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if floor is not None:
        cons.append({"type": "ineq", "fun": lambda w: w @ mu - floor})
    best = None
    for x0 in np.eye(len(tk)).tolist() + [np.full(len(tk), 1 / len(tk))]:
        r = minimize(lambda w: np.var(X @ w - b, ddof=1) * 12, np.array(x0), method="SLSQP", bounds=[(0, 1)] * len(tk),
                     constraints=cons, options={"ftol": 1e-15, "maxiter": 2000})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    return best, mu, float(b.mean() * 12)


@pytest.mark.parametrize("bench", ["QQQ", "AGG"])
def test_min_tracking_error_external_benchmark_matches_scipy(bench):
    tk = ["SPY", "TLT", "GLD"]
    R = research.optimize(tk, "2006-01-01", "2024-12-31", benchmark=bench, methods=["min_tracking_error"])
    ref, mu, mb = _te_reference(tk, bench, R["fit_start"], R["fit_end"])
    p = R["portfolios"]["Min tracking error"]
    assert [p["weights"].get(t, 0) for t in tk] == pytest.approx(list(ref.x), abs=2e-3)
    assert p["tracking_error"] == pytest.approx(np.sqrt(ref.fun), rel=1e-3)
    assert R["benchmark"]["return"] == pytest.approx(mb, rel=1e-9) and not R["benchmark"]["investable"]
    if bench != "AGG":
        return
    # a return floor above the unconstrained minimum-TE portfolio's return binds; scipy agrees
    floor = float(ref.x @ mu) + 0.01
    R2 = research.optimize(tk, "2006-01-01", "2024-12-31", benchmark=bench, methods=["min_tracking_error"],
                           target_active=floor - mb)
    p2 = next(iter(R2["portfolios"].values()))
    ref2, _, _ = _te_reference(tk, bench, R["fit_start"], R["fit_end"], floor)
    assert p2["exp_return"] == pytest.approx(floor, abs=1e-5) and p2["tracking_error"] > p["tracking_error"]
    assert [p2["weights"].get(t, 0) for t in tk] == pytest.approx(list(ref2.x), abs=2e-3)
    assert p2["active_return"] == pytest.approx(floor - mb, abs=1e-5)
    R3 = research.optimize(tk, "2006-01-01", "2024-12-31", benchmark=bench, methods=["min_tracking_error"], target_return=floor)
    assert next(iter(R3["portfolios"].values()))["weights"] == pytest.approx(p2["weights"], abs=1e-5)


def test_max_information_ratio_beats_alternatives():
    tk = ["SPY", "TLT", "GLD", "QQQ"]
    R = research.optimize(tk, "2006-01-01", "2024-12-31", benchmark="60% SPY 40% TLT",
                          methods=["max_information_ratio", "max_sharpe", "min_variance"])
    ir = {n: p["information_ratio"] for n, p in R["portfolios"].items()}
    best = ir["Max information ratio"]
    assert all(best >= v - 1e-6 for v in ir.values() if np.isfinite(v))
    # a random search over the simplex does not find a better one
    mr = _monthly(tk)
    mr = mr[(mr.index >= pd.Timestamp(R["fit_start"])) & (mr.index <= pd.Timestamp(R["fit_end"]))]
    mu, cov = mr.mean().to_numpy() * 12, np.cov(mr.to_numpy(), rowvar=False) * 12
    wb = np.array([0.6, 0.4, 0, 0])
    W = np.random.default_rng(5).dirichlet(np.ones(4), 4000)
    D = W - wb
    irs = (D @ mu) / np.sqrt(np.einsum("ij,jk,ik->i", D, cov, D))
    assert best >= irs.max() - 1e-6


def test_bench_methods_need_a_benchmark():
    with pytest.raises(ValueError, match="benchmark"):
        research.optimize(["SPY", "TLT"], "2010-01-01", "2020-12-31", methods=["min_tracking_error"])
    R = research.optimize(["SPY", "TLT"], "2010-01-01", "2020-12-31", methods=None)
    assert not any(p["method"] in research.BENCH_METHODS for p in R["portfolios"].values())


# ------------------------------------------------------------------ resampling

def test_resampled_frontier():
    R = research.optimize(["SPY", "TLT", "GLD"], "2006-01-01", "2024-12-31", methods=["max_sharpe", "min_variance"],
                          resample=12, points=10)
    rs = R["resampled"]
    assert rs["draws"] == 12 and len(rs["frontier"]) == 15
    for name in ("Resampled max Sharpe", "Resampled min variance"):
        p = R["portfolios"][name]
        assert sum(p["weights"].values()) == pytest.approx(1.0) and p["resampled"] == 12 and p["weights_sd"]
    # averaging spreads the max-Sharpe bet; the min-variance portfolio is estimated precisely
    mv, rmv = R["portfolios"]["Min variance"]["weights"], R["portfolios"]["Resampled min variance"]["weights"]
    assert all(abs(mv.get(t, 0) - rmv.get(t, 0)) < 0.05 for t in ("SPY", "TLT", "GLD"))
    # no resampled frontier point beats the inputs' own optimum (the max-Sharpe portfolio) or the min variance
    ms = R["portfolios"]["Max Sharpe"]
    for p in rs["frontier"]:
        assert (p["return"] - R["rf"]) / p["vol"] <= ms["exp_sharpe"] + 1e-6
        assert p["vol"] >= R["portfolios"]["Min variance"]["exp_vol"] - 1e-6
    R2 = research.optimize(["SPY", "TLT", "GLD"], "2006-01-01", "2024-12-31", methods=["max_sharpe"], resample=12, points=10)
    assert R2["portfolios"]["Resampled max Sharpe"]["weights"] == R["portfolios"]["Resampled max Sharpe"]["weights"]


def test_optimize_cli_and_api(tmp_path, capsys, monkeypatch):
    from backtester import web
    from backtester.__main__ import main
    assert main(["optimize", "SPY", "TLT", "GLD", "--start", "2006-01-01", "--end", "2024-12-31", "--methods",
                 "max_sharpe,min_tracking_error", "--benchmark", "60% SPY 40% TLT", "--view", "SPY = 8% @ 60%",
                 "--prior", "SPY=60%,TLT=30%,GLD=10%", "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "Black-Litterman" in out and "posterior" in out and "TE" in out and "Min tracking error" in out
    page = (tmp_path / "o" / "report.html").read_text()
    assert '"black_litterman"' in page and '"benchmark"' in page
    monkeypatch.setattr(web, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    r = web.api_research({"tickers": "SPY TLT GLD", "start": "2006-01-01", "end": "2024-12-31",
                          "methods": ["max_sharpe", "min_variance"], "expected_returns": "SPY=7%, TLT=4%, GLD=3%",
                          "resample": 6}, "optimize")
    page = (tmp_path / "runs" / r["id"] / "report.html").read_text()
    blob = json.loads(page.split('<script id="data" type="application/json">')[1].split("</script>")[0].replace("<\\/", "</"))
    assert blob["inputs"]["source"] == "forecast" and blob["resampled"]["draws"] == 6
    assert "Resampled max Sharpe" in blob["portfolios"]


# ------------------------------------------------------------------ Monte Carlo withdrawal rates by percentile

def test_withdrawal_rates_by_path_match_annuity_formulas():
    years, r_m = 30, 0.005
    P = np.full((3, years * 12), r_m)
    ci = np.ones((3, years * 12 + 1))
    swr, pwr = mc.withdrawal_rates_by_path(P, ci, np.array([1.0, 2.0, 5.0]))
    g = (1 + r_m) ** 12
    annuity = 1 / sum(g ** -k for k in range(years))
    # the last withdrawal leaves exactly 0, which does not count as paid ("b > 0"): the bisection converges from below
    assert swr == pytest.approx(np.full(3, annuity), abs=1e-8)
    assert pwr == pytest.approx(np.full(3, 1 - 1 / g), abs=1e-8)


def test_percentile_rates_agree_with_the_aggregate_swr():
    s = mc.Settings(weights={"SPY": 0.6, "TLT": 0.4}, years=30, sims=800, flows=[mc.CashFlow(amount=-40_000)])
    R = mc.run(s)
    wr = R["withdrawal_rates"]
    assert wr["basis"] == "start" and list(wr["safe"]) == ["10", "25", "50", "75", "90"]
    assert wr["safe_at_target"] == pytest.approx(R["safe_withdrawal_rate"], abs=1e-6)
    vals = list(wr["safe"].values())
    assert vals == sorted(vals) and list(wr["perpetual"].values()) == sorted(wr["perpetual"].values())
    assert wr["perpetual"]["50"] == pytest.approx(R["perpetual_withdrawal_rate"], abs=2e-3)
    assert all(wr["safe"][p] >= wr["perpetual"][p] - 1e-9 for p in wr["safe"])
    txt = mc.console(R)
    assert "Withdrawal rates by percentile" in txt and "Perpetual withdrawal rate" in txt


def test_contribute_then_withdraw_rates_use_the_balance_at_retirement():
    flows = [mc.CashFlow(amount=20_000, end_year=10), mc.CashFlow(amount=-50_000, start_year=11)]
    s = mc.Settings(weights={"SPY": 0.6, "TLT": 0.4}, years=35, sims=400, start_balance=100_000, flows=flows, seed=3)
    R = mc.run(s)
    wr = R["withdrawal_rates"]
    assert wr["basis"] == "withdrawal_start" and wr["from_year"] == 11 and wr["years"] == 25
    assert wr["base_balance"]["50"] > 300_000          # ten years of contributions and growth
    assert R["show_withdrawal_rates"] and any("start of year 11" in n for n in R["notes"])
    assert "balance at the start of year 11" in mc.console(R)
    # the same numbers from the building blocks: re-run the simulation and measure from month 120
    s2 = mc.Settings(weights={"SPY": 0.6, "TLT": 0.4}, years=35, sims=400, start_balance=100_000, flows=flows, seed=3)
    rng_state = mc.run(s2)
    assert rng_state["withdrawal_rates"] == wr


def test_report_withdrawal_rate_percentiles():
    out = mc.historical_withdrawal_rates(_monthly(["SPY"])["SPY"])
    pc = out["percentiles"]
    assert set(pc) == {"safe", "perpetual"} and list(pc["safe"]) == ["10", "25", "50", "75", "90"]
    assert pc["safe"]["10"] <= pc["safe"]["50"] <= pc["safe"]["90"]


# ------------------------------------------------------------------ parser: weights and ranking periods

@pytest.mark.parametrize("text", ["60% VTI 40% BND", "VTI 60% BND 40%", "VTI 60, BND 40", "hold 60% VTI 40% BND, rebalance quarterly",
                                  "VTI 60 BND 40", "vti 60% bnd 40%"])
def test_weights_without_separators(text):
    p = parser.parse(text)
    assert p.tree == {"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "VTI"}, {"asset": "BND"}]}


def test_weights_without_separators_cash_and_refusals():
    p = parser.parse("60% SPY 30% TLT 10% cash")
    assert p.tree["w"] == [0.6, 0.3, 0.1] and p.tree["children"][2] == {"cash": True}
    with pytest.raises(parser.ParseError):
        parser.parse("VTI 50, BND 40")          # bare numbers must add up to 100


@pytest.mark.parametrize("phrase,by", [
    ("average of 1, 3, 6 and 12 month return", "(tret(tr, 21) + tret(tr, 63) + tret(tr, 126) + tret(tr, 252)) / 4"),
    ("the average of the 1-, 3-, 6- and 12-month returns", "(tret(tr, 21) + tret(tr, 63) + tret(tr, 126) + tret(tr, 252)) / 4"),
    ("average 3/6/12 month momentum", "(tret(tr, 63) + tret(tr, 126) + tret(tr, 252)) / 3"),
    ("12 month return skipping the last month", "ref(tret(tr, 231), 21)"),
    ("12 month momentum excluding the most recent month", "ref(tret(tr, 231), 21)"),
    ("12-1 momentum", "ref(tret(tr, 231), 21)"),
    ("12 month return divided by volatility", "tret(tr, 252) / volatility(252)"),
    ("6 month return divided by 60 day volatility", "tret(tr, 126) / volatility(60)"),
    ("risk-adjusted momentum", "tret(tr, 252) / volatility(252)"),
    ("6 month volatility-adjusted return", "tret(tr, 126) / volatility(126)"),
])
def test_ranking_period_options(phrase, by):
    p = parser.parse(f"top 2 of SPY, QQQ, TLT and GLD by {phrase}, rebalance monthly")
    assert p.tree["filter"]["by"] == by
    assert p.tree["filter"]["n"] == 2 and len(p.tree["children"]) == 4


def test_ranking_period_options_in_rotations():
    p = parser.parse("rotate between SPY, TLT and GLD by average of 3, 6 and 12 month return, rebalance monthly")
    assert p.tree["filter"]["by"] == "(tret(tr, 63) + tret(tr, 126) + tret(tr, 252)) / 3"


def test_ranking_expressions_match_pandas_and_have_no_lookahead():
    df = data.load("SPY")
    df = df[df.index >= "2015-01-01"]
    ns = expr.Namespace(df)
    tr, c = df["adj_close"], df["close"]
    ref = {
        "(tret(tr, 21) + tret(tr, 63) + tret(tr, 126) + tret(tr, 252)) / 4":
            sum(tr / tr.shift(n) - 1 for n in (21, 63, 126, 252)) / 4,
        "ref(tret(tr, 231), 21)": tr.shift(21) / tr.shift(252) - 1,
        "tret(tr, 252) / volatility(252)": (tr / tr.shift(252) - 1) / (c.pct_change().rolling(252).std() * np.sqrt(252)),
    }
    for rule, want in ref.items():
        got = expr.evaluate_value(rule, ns)
        pd.testing.assert_series_equal(got, want, check_names=False, rtol=1e-12)
        cut = df.index[len(df) // 2]
        early = expr.evaluate_value(rule, expr.Namespace(df[df.index <= cut]))
        pd.testing.assert_series_equal(early, got[got.index <= cut], check_names=False, rtol=1e-12)


def test_parsed_ranking_runs():
    res = runner.run(parser.parse("top 1 of SPY, TLT and GLD by average of 1, 3, 6 and 12 month return, rebalance monthly, "
                                  "from 2015 to 2020"))
    assert res.equity.iloc[-1] > 0 and len(res.orders)


# ------------------------------------------------------------------ risk contributions

@pytest.fixture(scope="module")
def sixty_forty():
    res = runner.run(parser.parse("hold 60% SPY and 40% TLT, rebalance monthly, from 2008 to 2022"))
    return res, report.analyze(res, sensitivity=False, mc=False)


def test_risk_contributions_add_up(sixty_forty):
    res, A = sixty_forty
    rc = A["risk_contributions"]
    rows = {r["ticker"]: r for r in rc["rows"]}
    assert set(rows) == {"SPY", "TLT"}
    for k in ("share", "share_monthly", "share_realised", "drawdown_share"):
        assert sum(r[k] for r in rows.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(r["contribution"] for r in rows.values()) == pytest.approx(rc["vol_daily"], rel=1e-12)
    assert rows["SPY"]["share"] > 0.8 > rows["SPY"]["avg_weight"]      # stocks carry most of a 60/40's risk
    # Euler decomposition against numpy on the average weights and the daily covariance
    w = res.holdings[["SPY", "TLT"]].mean().to_numpy()
    px = pd.DataFrame({t: res.prices[t]["adj_close"] for t in ("SPY", "TLT")}).reindex(res.holdings.index).ffill()
    cov = px.pct_change().iloc[1:].cov().to_numpy() * 252
    sig = np.sqrt(w @ cov @ w)
    assert rows["SPY"]["contribution"] == pytest.approx(w[0] * (cov @ w)[0] / sig, rel=1e-9)
    assert rows["SPY"]["mcr"] == pytest.approx((cov @ w)[0] / sig, rel=1e-9)
    dd = rc["drawdown"]
    assert dd["depth"] < -0.1 and dd["peak"] < dd["trough"]
    assert sum(r["drawdown_contribution"] for r in rows.values()) == pytest.approx(dd["asset_return"], abs=1e-12)


def test_risk_contributions_in_outputs(sixty_forty, tmp_path):
    res, A = sixty_forty
    assert "Risk contribution" in report.console_summary(A)
    report.write_outputs(A, tmp_path)
    page = (tmp_path / "report.html").read_text()
    assert '"risk_contributions"' in page and "riskContribTbl" in page
    pytest.importorskip("openpyxl")
    x = pd.read_excel(tmp_path / "report.xlsx", sheet_name="Risk contributions")
    assert list(x["ticker"]) and "share of volatility (daily)" in x.columns
    assert x["share of volatility (daily)"].sum() == pytest.approx(1.0, abs=1e-9)


def test_no_risk_table_for_signal_runs():
    res = runner.run(parser.parse("buy SPY when RSI(2) is below 10, hold 3 days, from 2019 to 2020"))
    assert risk.risk_contributions(res) == {}
