"""Monte Carlo, optimiser, factor analysis, P&L attribution, benchmark alignment and long-history
(SIM) series."""
import numpy as np
import pandas as pd
import pytest

from backtester import data, metrics, montecarlo as mc, parser, report, research, runner

HAVE = {"SPY", "TLT", "QQQ", "GLD"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


# ------------------------------------------------------------ Monte Carlo maths

def test_balance_recursion_and_withdrawal_rates_on_a_known_path():
    # a constant 5% a year with no inflation: withdrawing w each year (start of year) lasts n years
    # while the annuity factor covers it; the perpetual rate keeps the balance flat
    r_m = 1.05 ** (1 / 12) - 1
    P = np.full((1, 30 * 12), r_m)
    ci = np.ones((1, 30 * 12 + 1))
    B = mc.simulate_balances(P, ci, 100.0, [mc.CashFlow(amount=-5.0, freq="yearly", inflation_adjusted=False)])
    # withdrawing 5 at the start of each year from 100 growing 5%: (100 - 5) * 1.05 = 99.75, ...
    assert B[0, 12] == pytest.approx(95 * 1.05)
    assert B[0, 24] == pytest.approx((95 * 1.05 - 5) * 1.05)
    # perpetual rate with start-of-year withdrawals: w such that (1 - w) * 1.05 = 1 -> w = 1 - 1/1.05
    assert mc.perpetual_withdrawal_rate(P, ci, 1.0) == pytest.approx(1 - 1 / 1.05, abs=1e-6)
    # safe rate over 30 years = annuity-due payment: sum_{k=0}^{29} w / 1.05^k = 1
    ann_due = sum(1.05 ** -k for k in range(30))
    assert mc.safe_withdrawal_rate(P, ci, 1.0, 1.0) == pytest.approx(1 / ann_due, abs=1e-6)


def test_inflation_indexed_flows_and_pct_withdrawals():
    P = np.zeros((1, 24))
    ci = np.array([[1.1 ** (m // 12) for m in range(25)]])
    B = mc.simulate_balances(P, ci, 1000.0, [mc.CashFlow(amount=-100, freq="yearly", inflation_adjusted=True)])
    assert B[0, -1] == pytest.approx(1000 - 100 - 110)
    B = mc.simulate_balances(P, ci, 1000.0, [mc.CashFlow(pct=-0.12, freq="monthly")])
    assert B[0, 1] == pytest.approx(990.0) and B[0, 2] == pytest.approx(990 * 0.99)


def test_portfolio_returns_rebalancing():
    # two assets alternating +100% / -50%: rebalanced monthly 50/50 earns 25% then -25%...
    A = np.array([[[1.0, -0.5], [-0.5, 1.0]]])
    w = np.array([0.5, 0.5])
    r = mc._portfolio_returns(A, w, 1)
    assert r[0, 0] == pytest.approx(0.25) and r[0, 1] == pytest.approx(0.25)
    r = mc._portfolio_returns(A, w, 0)          # never rebalanced: each asset ends at x2 * x0.5 = x1
    assert (1 + r[0]).prod() == pytest.approx(0.5 * 2 * 0.5 + 0.5 * 0.5 * 2)


def test_t_df_fit_recovers_fat_tails():
    rng = np.random.default_rng(0)
    x = rng.standard_t(5, size=(20000, 1)) * 0.04
    assert 3.8 < mc.fit_t_df(x) < 6.5
    z = rng.standard_normal((20000, 2))
    assert mc.fit_t_df(z) > 30


@needs_data
@pytest.mark.parametrize("model", mc.MODELS)
def test_montecarlo_models_run_and_are_consistent(model):
    s = mc.Settings(weights={"SPY": 60, "TLT": 40}, years=20, sims=600, model=model,
                    flows=[mc.CashFlow(amount=-40_000)], forecast={"SPY": (0.06, 0.16)})
    R = mc.run(s)
    f = R["final"]
    assert f["10"] <= f["25"] <= f["50"] <= f["75"] <= f["90"]
    assert 0 <= R["prob_success"] <= 1
    assert R["success_by_year"][0] == 1.0 and all(a >= b - 1e-12 for a, b in zip(R["success_by_year"], R["success_by_year"][1:]))
    assert len(R["bands"]["50"]) == 21 and R["bands"]["50"][0] == 1_000_000
    assert 0 <= R["safe_withdrawal_rate"] <= R["perpetual_withdrawal_rate"] + 0.05
    assert all(v <= 0 for v in R["max_drawdown"].values())


@needs_data
def test_bootstrap_preserves_cross_asset_correlation():
    hist = mc.monthly_asset_returns(["SPY", "QQQ"])
    rng = np.random.default_rng(1)
    idx = mc._draw_blocks(rng, len(hist), 240, 400, 12)
    A = hist.to_numpy()[idx].reshape(-1, 2)
    assert np.corrcoef(A.T)[0, 1] == pytest.approx(hist.corr().iloc[0, 1], abs=0.03)


def test_weights_from_tree():
    assert mc.weights_from_tree({"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "TLT"}]}) == {"SPY": 0.6, "TLT": 0.4}
    assert mc.weights_from_tree({"if": "x", "then": {"asset": "SPY"}, "else": {"asset": "TLT"}}) is None
    assert mc.parse_weights("SPY 60, TLT 40") == {"SPY": 60, "TLT": 40}
    assert mc.parse_weights("60% SPY 40% TLT") == {"SPY": 60, "TLT": 40}
    assert mc.parse_weights("SPY TLT") == {"SPY": 1, "TLT": 1}


# ------------------------------------------------------------ optimiser

def test_round_weights_sum_to_exactly_100():
    w = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    assert sum(research.round_weights(w).values()) == 100
    w = {"A": 0.125, "B": 0.125, "C": 0.125, "D": 0.625}
    r = research.round_weights(w)
    assert sum(r.values()) == 100
    rng = np.random.default_rng(3)
    for _ in range(200):
        x = rng.dirichlet(np.ones(rng.integers(2, 9)))
        assert sum(research.round_weights({str(i): v for i, v in enumerate(x)}).values()) == 100


def test_parse_constraints():
    b, g = research.parse_constraints("SPY <= 50%; TLT >= 0.1, SPY+QQQ <= 70%; 20% <= TLT+GLD <= 60%", ["SPY", "QQQ", "TLT", "GLD"])
    assert b["SPY"][1] == 0.5 and b["TLT"][0] == pytest.approx(0.1)
    assert g[0]["tickers"] == ["SPY", "QQQ"] and g[0]["hi"] == pytest.approx(0.7) and g[0]["lo"] is None
    assert g[1]["lo"] == pytest.approx(0.2) and g[1]["hi"] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        research.parse_constraints("IWM <= 10%", ["SPY", "TLT"])
    with pytest.raises(ValueError):
        research.parse_constraints("SPY is small", ["SPY", "TLT"])


def _synthetic_opt(groups=(), bounds=None):
    rng = np.random.default_rng(5)
    C = np.array([[0.04, 0.01, 0.0, 0.0], [0.01, 0.09, 0.0, 0.0], [0.0, 0.0, 0.01, 0.002], [0.0, 0.0, 0.002, 0.02]]) / 12
    R = rng.multivariate_normal(np.array([0.08, 0.12, 0.03, 0.05]) / 12, C, size=240)
    return research._Opt(R, ["A", "B", "C", "D"], 0.02, bounds or {}, list(groups))


def test_optimiser_objectives_are_optimal_and_feasible():
    o = _synthetic_opt()
    ws = {m: o.weights(m) for m in ("max_sharpe", "min_variance", "max_sortino", "min_cvar", "risk_parity", "max_diversification", "inverse_vol", "equal")}
    for m, w in ws.items():
        assert w is not None and abs(w.sum() - 1) < 1e-9 and (w >= -1e-12).all(), m
    sharpe = lambda w: (w @ o.mu - o.rf) / o.vol(w)  # noqa: E731
    for m, w in ws.items():
        assert sharpe(ws["max_sharpe"]) >= sharpe(w) - 1e-6, m
        assert o.vol(ws["min_variance"]) <= o.vol(w) + 1e-6, m
        assert o.cvar(ws["min_cvar"]) <= o.cvar(w) + 1e-6, m
        assert (w @ o.sd) / o.vol(w) <= (ws["max_diversification"] @ o.sd) / o.vol(ws["max_diversification"]) + 1e-6, m
    rc = ws["risk_parity"] * (o.cov @ ws["risk_parity"]) / o.vol(ws["risk_parity"]) ** 2
    assert np.allclose(rc, 0.25, atol=1e-3)
    # targets
    w = o.weights("target_return", 0.07)
    assert w @ o.mu >= 0.07 - 1e-6
    assert o.vol(w) <= o.vol(o.weights("equal")) + 1e-9 or o.weights("equal") @ o.mu < 0.07
    w = o.weights("target_vol", 0.12)
    assert o.vol(w) <= 0.12 + 1e-6
    assert o.weights("target_vol", 0.0001) is None


def test_optimiser_respects_bounds_and_groups():
    o = _synthetic_opt(groups=[{"tickers": ["A", "B"], "lo": None, "hi": 0.5}], bounds={"D": [0.2, 0.3]})
    for m in ("max_sharpe", "min_variance", "max_sortino", "min_cvar", "risk_parity", "max_diversification", "inverse_vol", "equal"):
        w = o.weights(m)
        assert w[0] + w[1] <= 0.5 + 1e-6 and 0.2 - 1e-6 <= w[3] <= 0.3 + 1e-6, m
    with pytest.raises(ValueError):
        _synthetic_opt(bounds={"A": [0.6, 1], "B": [0.6, 1]})


@needs_data
def test_optimize_end_to_end_with_rolling_and_sentences():
    R = research.optimize(["SPY", "QQQ", "TLT", "GLD"], constraints=["SPY+QQQ <= 70%"], target_vol=0.1,
                          rolling_months=12, lookback_months=60)
    for name, p in R["portfolios"].items():
        assert sum(p["rounded"].values()) == 100
        assert p["weights"].get("SPY", 0) + p["weights"].get("QQQ", 0) <= 0.7 + 1e-6
        spec = parser.parse(p["sentence"])
        tot = {}
        for c, x in zip(spec.tree["children"], spec.tree["w"]):
            tot[c["asset"]] = round(x * 100)
        assert tot == p["rounded"], name
    ro = R["rolling"]
    assert set(ro["curves"]) >= {"Max Sharpe", "Max Sharpe (static)"} and len(ro["dates"]) == len(ro["curves"]["Max Sharpe"])
    assert len(R["frontier"]) > 5
    vols = [f["vol"] for f in R["frontier"]]
    assert all(b >= a - 1e-6 for a, b in zip(vols, vols[1:]))


# ------------------------------------------------------------ factors

def test_ols_matches_numpy():
    from backtester import factors
    rng = np.random.default_rng(2)
    X = np.column_stack([np.ones(500), rng.normal(size=(500, 2))])
    y = X @ np.array([0.01, 1.2, -0.3]) + rng.normal(scale=0.1, size=500)
    o = factors.ols(y, X)
    assert np.allclose(o["coef"], np.linalg.lstsq(X, y, rcond=None)[0])
    assert o["r2"] > 0.9 and abs(o["t"][1]) > 50


@needs_data
@pytest.mark.skipif(data.factors().empty, reason="factor data not downloaded")
def test_factor_models_on_the_market():
    from backtester import factors
    r, name = factors.returns_for("SPY")
    for model in [m for m in factors.MODELS if factors.REGION.get(m, "us") == "us"]:  # US market factors
        for freq in ("monthly", "daily"):
            R = factors.analyze(r, model, freq, start="2005-01-01", name=name)
            beta = R["coefficients"][1]["loading"]
            assert 0.9 < beta < 1.1 and R["r_squared"] > 0.9, (model, freq)
            assert [c["factor"] for c in R["coefficients"]][1:] == factors.MODELS[model][1]
    assert R["rolling"]["dates"] and len(R["rolling"]["loadings"]["Mkt-RF"]) == len(R["rolling"]["dates"])


# ------------------------------------------------------------ allocation reports

@needs_data
@pytest.mark.parametrize("reinvest", [True, False])
def test_pnl_attribution_adds_up(reinvest):
    from backtester import portfolio as pf
    p = pf.Portfolio(tree={"weights": "specified", "w": [0.5, 0.3, 0.2], "children": [{"asset": "SPY"}, {"asset": "TLT"}, {"asset": "GLD"}]},
                     rebalance="quarterly", contribution=300, withdrawal=250, withdrawal_freq="quarterly", inflation_adjust=True,
                     commission=1.0, commission_pct=0.001, slippage_bps=3, expense_ratio=0.002, start="2008-01-01",
                     reinvest_dividends=reinvest)
    r = pf.run(p)
    a = r.extras["attribution"]
    lhs = a["pnl"].sum() + r.interest - r.extras["fees"]
    rhs = r.equity.iloc[-1] - r.equity.iloc[0] - r.extras["flows"].sum()
    assert lhs == pytest.approx(rhs, abs=0.01)
    assert a["dividends"].sum() > 0
    # holding periods partition each ticker's history: they add up to the same P&L
    assert r.trades["pnl"].sum() == pytest.approx(a["pnl"].sum(), abs=0.01)
    by = r.trades.groupby("ticker")["pnl"].sum()
    for row in a.itertuples():
        assert by[row.ticker] == pytest.approx(row.pnl, abs=0.01)


@needs_data
def test_benchmark_starts_at_the_first_invested_close_and_gets_the_same_flows():
    spec = parser.parse("hold 60% SPY and 40% TLT, add $500 every month, rebalance quarterly, since 2010")
    res = runner.run(spec)
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    idx = res.equity.index
    b = A["benchmarks"]["SPY buy & hold"]
    # day 0 (2009-12-31): the portfolio is bought at that close, so the benchmark is too (the first day counts)
    assert res.extras["day0"] and idx[0] == pd.Timestamp("2009-12-31")
    assert b.index[0] == idx[0] and b.iloc[0] == pytest.approx(res.equity.iloc[0])
    adj = data.load("SPY")["adj_close"]
    assert b.iloc[5] / b.iloc[0] == pytest.approx(adj[idx[5]] / adj[idx[0]])
    # displayed dates skip the synthetic day before the first bar
    assert A["stats"]["start"] == idx[1].date()
    assert A["drawdowns"]["peak"].min() >= idx[1].date()
    # the benchmark with flows receives exactly the same contributions
    bw = A["benchmarks_with_flows"]["SPY buy & hold"]
    fl = res.extras["flows"]
    assert A["benchmark_cash"]["SPY buy & hold"]["total_contributions"] == pytest.approx(fl[fl > 0].sum())
    # and its time-weighted growth is the plain benchmark's
    r = (bw - fl).iloc[1:] / bw.shift().iloc[1:]
    assert np.allclose(r.to_numpy(), (b / b.shift()).iloc[1:].to_numpy(), rtol=1e-9)
    # payload fields for the template
    P = report.build_payload([A])
    assert "with_flows" in P["benchmarks"][0] and P["runs"][0]["equity_real"] is not None
    assert P["runs"][0]["attribution"]["check"] == pytest.approx(0.0, abs=0.01)


@needs_data
def test_withdrawal_rates_for_portfolios_with_withdrawals():
    spec = parser.parse("hold 60% SPY and 40% TLT, withdraw 4% per year adjusted for inflation, starting with $1,000,000, rebalance annually")
    A = report.analyze(runner.run(spec), sensitivity=False, mc=False)
    wr = A["withdrawal_rates"]
    assert 0 < wr["swr"] < 0.3 and 0 <= wr["pwr"] < 0.3 and 0 < wr["swr_mc95"] < 0.3


def test_monthly_returns_skip_a_lone_first_point():
    idx = pd.DatetimeIndex(["2020-02-29", "2020-03-02", "2020-03-31", "2020-04-30"])
    s = pd.Series([100.0, 100.0, 110.0, 121.0], index=idx)
    m = metrics.monthly_returns(s)
    assert list(m.round(6)) == [0.1, 0.1]


# ------------------------------------------------------------ long-history (SIM) series

@needs_data
def test_sim_series_load_and_run(tmp_path, monkeypatch):
    """A SPYSIM-shaped file (volume 0, open = high = low = close, no dividends) works in portfolios,
    the optimiser and the Monte Carlo tool."""
    import shutil
    for t in ("SPY", "TLT"):
        shutil.copy(data.PRICES / f"{t}.csv", tmp_path / f"{t}.csv")
    spy = pd.read_csv(data.PRICES / "SPY.csv", parse_dates=["date"], index_col="date")
    lvl = 100 * spy["adj_close"] / spy["adj_close"].iloc[0]
    for name, s in (("SPYSIM", lvl), ("TLTSIM", 100 * (1.0001 ** np.arange(len(lvl))))):
        df = pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 0, "dividend": 0.0, "adj_close": s}, index=lvl.index)
        df.index.name = "date"
        df.to_csv(tmp_path / f"{name}.csv")
    monkeypatch.setattr(data, "PRICES", tmp_path)
    data.load.cache_clear()
    monkeypatch.setenv("BACKTESTER_OFFLINE", "1")
    try:
        assert data.sims() == ["SPYSIM", "TLTSIM"]
        # (the sentence parser only reads tickers of up to 5 letters, so use a spec here)
        spec = runner.from_dict({"kind": "allocation", "rebalance": "quarterly", "tree": {
            "weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPYSIM"}, {"asset": "TLTSIM"}]}})
        r = runner.run(spec)
        A = report.analyze(r, sensitivity=False, mc=False)
        assert A["stats"]["cagr"] > 0
        assert r.equity.iloc[-1] > 0
        R = research.optimize(["SPYSIM", "TLTSIM"])
        assert R["portfolios"]
        M = mc.run(mc.Settings(weights={"SPYSIM": 60, "TLTSIM": 40}, years=10, sims=200))
        assert M["final"]["50"] > 0
    finally:
        data.load.cache_clear()
