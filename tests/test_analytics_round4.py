"""Round 4 analytics: bond and developed-market factors, the correlation tool, new optimiser objectives and
infeasible targets, allocation-report tables, tactical warm-up, Monte Carlo (partial months, stress tests,
horizons, contribution timing that matches the backtest), readable names and interpretations, holding
periods, chart payloads (stop levels, rule state, other tickers), dry-run validation, short financing in
portfolios, and period ends on the scheduled NYSE calendar."""
import json

import numpy as np
import pandas as pd
import pytest

from backtester import data, factors, metrics, montecarlo as mc, portfolio as pf, report, research, runner
from backtester.strategy import Strategy

HAVE = {"SPY", "QQQ", "TLT", "GLD", "LQD", "EFA", "TLTSIM", "BILSIM", "IEFSIM", "EFASIM", "SPYSIM"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")
needs_factors = pytest.mark.skipif(data.factors().empty or not (data.DATA / "factors" / "dev_ff3_daily.csv").exists(),
                                   reason="factor data not downloaded")


def frame(closes, start="2019-01-01", highs=None, lows=None):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    df = pd.DataFrame({"open": closes, "high": closes if highs is None else highs, "low": closes if lows is None else lows,
                       "close": closes}, index=idx)
    df["volume"] = 1e6
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    df["split"] = 0.0
    return df


def walk(seed, n=600, drift=0.0003, vol=0.012):
    r = np.random.default_rng(seed).normal(drift, vol, n)
    r[0] = 0.0
    return 100 * np.cumprod(1 + r)


@pytest.fixture
def fake(monkeypatch):
    frames = {}
    real = data.load

    def load(t):
        t = data.canonical(t)
        if t in frames:
            return frames[t]
        if t in ("SPY", "QQQ", "SPYSIM"):   # benchmarks are not part of these synthetic tests
            raise FileNotFoundError(t)
        return real(t)

    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): load(t) for t in ts})
    return frames


# ------------------------------------------------------------ 1. factor models

@needs_data
@needs_factors
def test_bond_and_developed_market_factors():
    r, _ = factors.returns_for("LQD")
    R = factors.analyze(r, "ff3+bonds", "monthly")
    load = {c["factor"]: c["loading"] for c in R["coefficients"]}
    assert R["factors"] == ["Mkt-RF", "SMB", "HML", "TERM", "DEF"]
    assert 0.2 < load["TERM"] < 0.8 and 0.5 < load["DEF"] < 1.3 and R["r_squared"] > 0.8
    tlt, _ = factors.returns_for("TLT")
    B = factors.analyze(tlt, "bonds", "daily", start="2003-01-01")
    assert B["coefficients"][2]["factor"] == "TERM" and B["coefficients"][2]["loading"] == pytest.approx(1.0, abs=0.05)
    efa, _ = factors.returns_for("EFA")
    for model in ("dev_ff3", "intl"):
        D = factors.analyze(efa, model, "monthly")
        assert D["model"] == "dev_ff3" and 0.9 < D["coefficients"][1]["loading"] < 1.15 and D["r_squared"] > 0.9
    # the developed-market model explains EFA better than the US one
    assert D["r_squared"] > factors.analyze(efa, "ff3", "monthly")["r_squared"]


@needs_data
@needs_factors
def test_def_is_left_out_with_a_note_before_the_corporate_series():
    r, _ = factors.returns_for("SPY")
    R = factors.analyze(r, "ff3+bonds", "monthly", start="1995-01-01")
    if factors.bond_factors().attrs.get("def_leg") == "LQDSIM":
        pytest.skip("LQDSIM covers the whole period")
    assert "DEF" not in R["factors"] and "TERM" in R["factors"]
    assert any("DEF (credit) left out" in n for n in R["notes"])
    assert "DEF" in factors.analyze(r, "ff3+bonds", "monthly", start="2004-01-01")["factors"]


def test_monthly_spread_factors_compound_each_leg():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    a = pd.Series(0.01, index=idx)
    b = pd.Series(0.002, index=idx)
    daily = pd.DataFrame({"RF": 0.0001, "TERM": a - b, "TERM|a": a, "TERM|b": b}, index=idx)
    m = factors.monthly_factors(daily)
    jan = idx[idx.month == 1]
    assert m.loc["2020-01-31", "TERM"] == pytest.approx(1.01 ** len(jan) - 1.002 ** len(jan))
    assert "TERM|a" not in m


def test_factor_models_are_listed_and_extensible(monkeypatch):
    keys = [m["key"] for m in factors.model_list()]
    assert {"dev_ff3", "bonds", "ff3+bonds"} <= set(keys)
    assert "intl" in next(m for m in factors.model_list() if m["key"] == "dev_ff3")["aliases"]
    with pytest.raises(ValueError):
        factors.resolve("qmj")
    monkeypatch.setitem(factors.MODELS, "test_x", ("Test", ["Mkt-RF", "XF"]))
    factors.register_factor("XF", "a test factor", lambda: pd.DataFrame({"XF": [0.0]}, index=[pd.Timestamp("2020-01-02")]))
    try:
        assert factors.resolve("test_x") == "test_x" and factors.DESCRIBE["XF"] == "a test factor"
    finally:
        factors._EXTRA.pop("XF", None)
        factors.factor_table.cache_clear()


# ------------------------------------------------------------ 2. correlations

def test_correlation_matrix_rolling_and_stats(fake):
    fake["A"] = frame(walk(1, 900), start="2018-01-01")
    fake["B"] = frame(walk(2, 900), start="2018-01-01")
    fake["C"] = frame(walk(3, 700), start="2018-10-01")
    from backtester import correlation
    R = correlation.analyze(["A", "B", "C"], "monthly", 12, pair=["A", "B"])
    M = np.array(R["matrix"])
    assert np.allclose(np.diag(M), 1) and np.allclose(M, M.T)
    px = pd.concat({t: fake[t]["adj_close"] for t in "ABC"}, axis=1).dropna()
    me = metrics.complete_months(px)
    assert M[0, 1] == pytest.approx(me.pct_change().iloc[1:].corr().loc["A", "B"], abs=1e-4)
    # the rolling pair uses the pair's own (longer) common history
    assert R["rolling"]["start"] < R["start"] and len(R["rolling"]["values"]) > 0
    assert [s["ticker"] for s in R["stats"]] == ["A", "B", "C"] and all(np.isfinite(s["cagr"]) for s in R["stats"])
    assert R["stats"][0]["data_from"] == "2018-01-01"
    with pytest.raises(ValueError):
        correlation.analyze(["A"], "monthly")
    with pytest.raises(ValueError):
        correlation.analyze(["A", "B"], "monthly", start="2020-01-01", end="2019-01-01")
    D = correlation.analyze(["A", "B"], "daily", 20)
    assert D["window"] == 20 and D["observations"] > 500


def test_complete_months_drop_the_month_in_progress():
    idx = pd.bdate_range("2021-01-04", "2021-03-17")   # March still in progress
    s = pd.Series(np.arange(len(idx), dtype=float) + 100, index=idx)
    me = metrics.complete_months(s)
    assert list(me.index) == [pd.Timestamp("2021-01-29"), pd.Timestamp("2021-02-26")]
    full = pd.bdate_range("2021-01-04", "2021-03-31")
    assert metrics.complete_months(pd.Series(1.0, index=full)).index[-1] == pd.Timestamp("2021-03-31")


@needs_data
def test_correlation_cli(capsys):
    from backtester.__main__ import main
    assert main(["correlation", "SPY", "TLT", "GLD", "EFASIM", "--window", "36", "--freq", "monthly"]) == 0
    out = capsys.readouterr().out
    assert "Rolling 36-month correlation SPY / TLT" in out and "EFASIM" in out and "CAGR" in out


# ------------------------------------------------------------ 3 / 7. optimiser

def _opt(seed=0, T=120, n=4, **kw):
    rng = np.random.default_rng(seed)
    R = rng.normal(0.006, 0.04, (T, n)) + np.linspace(-0.002, 0.004, n)
    return research._Opt(R, [f"T{i}" for i in range(n)], 0.02, {}, [], **kw)


@pytest.mark.parametrize("thr", [0.0, 0.05])
def test_max_omega_is_the_best_feasible_omega(thr):
    o = _opt(omega_threshold=thr)
    w = o.weights("omega")
    assert o.ok(w)
    rng = np.random.default_rng(1)
    best_random = max(o.omega(x) for x in rng.dirichlet(np.ones(o.n), 3000))
    assert o.omega(w) >= best_random - 1e-9
    corners = max(o.omega(np.eye(o.n)[j]) for j in range(o.n))
    assert o.omega(w) >= corners - 1e-9


def test_max_return_over_max_drawdown_beats_the_classic_portfolios():
    o = _opt(seed=4, n=3)
    w = o.weights("max_return_over_maxdd")
    score = lambda x: (lambda c, m: c / abs(m))(*o.path(x))  # noqa: E731
    for m in ("max_sharpe", "min_variance", "equal", "inverse_vol"):
        assert score(w) >= score(o.weights(m)) - 1e-9, m
    rng = np.random.default_rng(2)
    assert score(w) >= max(score(x) for x in rng.dirichlet(np.ones(3), 500)) - 1e-3


def test_infeasible_targets_report_the_reachable_range(fake):
    for t, s in zip("ABC", (1, 2, 3)):
        fake[t] = frame(walk(s, 900, vol=0.01 * s), start="2015-01-01")
    R = research.optimize(["A", "B", "C"], target_vol=0.001, target_return=5.0, methods=["target_vol", "target_return", "omega"])
    notes = " ".join(R["notes"])
    assert "minimum achievable volatility is" in notes and "maximum achievable expected return is" in notes
    assert R["infeasible"]["min_vol"] > 0.001 and list(R["portfolios"]) == ["Max Omega"]
    with pytest.raises(ValueError):
        research.optimize(["A", "B"], methods=["best_ever"])


# ------------------------------------------------------------ 4. allocation reports

def test_trailing_returns():
    idx = pd.bdate_range("2010-01-01", "2021-06-30")
    g = pd.Series(1.0001 ** np.arange(len(idx)), index=idx)
    T = metrics.trailing_returns(g)
    one = g.iloc[-1] / g[g.index <= idx[-1] - pd.DateOffset(years=1)].iloc[-1] - 1
    assert T["1Y"] == pytest.approx(one)
    ten = g.iloc[-1] / g[g.index <= idx[-1] - pd.DateOffset(years=10)].iloc[-1]
    assert T["10Y"] == pytest.approx(ten ** 0.1 - 1)
    assert T["YTD"] == pytest.approx(g.iloc[-1] / g[g.index <= "2020-12-31"].iloc[-1] - 1)
    short = metrics.trailing_returns(g[g.index >= "2019-01-01"])
    assert short["5Y"] is None and short["10Y"] is None and short["3Y"] is None and short["1Y"] is not None


def test_allocation_report_tables_and_holding_periods(fake, tmp_path):
    fake["A"] = frame(walk(1, 800), start="2018-01-01")
    fake["B"] = frame(walk(2, 800), start="2018-01-01")
    p = pf.Portfolio(tree={"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "A"}, {"asset": "B"}]},
                     rebalance="daily", cash_rate=None)
    res = pf.run(p)
    tr = res.trades
    # the average position value, not the sum of every day's purchases
    a = tr[tr.ticker == "A"].iloc[0]
    mv = res.holdings["A"] * res.equity.iloc[1:]
    assert a.position_value == pytest.approx(mv.mean(), rel=1e-6)
    assert a.bought > 2 * a.position_value and a.entry_value == pytest.approx(6_000, rel=1e-6)
    assert a["return"] == pytest.approx(fake["A"]["adj_close"].iloc[-1] / fake["A"]["adj_close"].iloc[0] - 1)
    # the attribution identity still holds
    at = res.extras["attribution"]
    assert at["pnl"].sum() + res.interest == pytest.approx(res.equity.iloc[-1] - res.equity.iloc[0], abs=0.01)
    A = report.analyze(res, sensitivity=False, mc=False)
    assert set(A["trailing"]) >= {"Strategy"} and A["trailing"]["Strategy"]["1Y"] is not None
    assert [s["ticker"] for s in A["asset_stats"]] == ["A", "B"] and A["asset_stats"][0]["avg_weight"] == pytest.approx(0.6, abs=0.02)
    P = report.build_payload([A])
    assert P["runs"][0]["trailing"] and P["runs"][0]["asset_stats"]
    report.write_outputs(A, tmp_path, excel=False)
    eq = pd.read_csv(tmp_path / "equity.csv", parse_dates=["date"])
    assert eq["date"].iloc[0] == res.equity.index[1]   # no synthetic day-before row
    assert "Trailing returns" in report.console_summary(A)


# ------------------------------------------------------------ 5. tactical warm-up

@pytest.mark.parametrize("mode", ["all", "first"])
def test_warmup_waits_for_every_ranked_asset(fake, mode):
    fake["A"] = frame(walk(1, 700), start="2018-01-01")
    fake["B"] = frame(walk(2, 400), start="2019-02-25")    # B starts later: the run starts with B
    tree = {"filter": {"select": "top", "n": 1, "by": "tret(tr, 100)"}, "universe": "children",
            "children": [{"asset": "A"}, {"asset": "B"}]}
    res = pf.run(pf.Portfolio(tree=tree, cash_rate=None, warmup=mode))
    warm, notes = report.warmup(res)
    b_ready = fake["B"].index[100]
    if mode == "all":
        assert warm == b_ready
        assert "all ranked assets have their full lookback" in notes[0] and "waited for B" in notes[0]
    else:
        assert warm is None
        assert any(n.startswith("Lookback: B") for n in notes)


def test_warmup_value_is_validated():
    with pytest.raises(ValueError):
        pf.Portfolio(tree={"cash": True}, warmup="some").validate()


# ------------------------------------------------------------ 6. Monte Carlo and contribution timing

def test_contributions_start_on_day_one_and_match_monte_carlo(fake):
    fake["A"] = frame([100.0] * 5400, start="2001-01-15")   # starts mid-month
    p = pf.Portfolio(tree={"asset": "A"}, rebalance="none", cash_rate=None, capital=10_000,
                     contribution=1_000, contribution_freq="monthly", contribution_end=20)
    res = pf.run(p)
    fl = res.extras["flows"]
    got = fl[fl > 0]
    assert len(got) == 240 and got.index[0] == fake["A"].index[0]
    assert res.equity.iloc[-1] == pytest.approx(10_000 + 240_000)
    cf = mc.flows_from_portfolio(p)
    assert cf[0].end_year == 20
    B = mc.simulate_balances(np.zeros((1, 25 * 12)), np.ones((1, 25 * 12 + 1)), 10_000, cf)
    assert B[0, -1] == pytest.approx(10_000 + 240_000)
    # withdrawals: the first one on day one too
    q = pf.Portfolio(tree={"asset": "A"}, rebalance="none", cash_rate=None, capital=100_000, withdrawal=4_000,
                     withdrawal_freq="yearly")
    w = pf.run(q).extras["flows"]
    assert w.iloc[1] == -4_000 and (w < 0).sum() == len(set(fake["A"].index.year))


def test_monte_carlo_history_drops_the_month_in_progress(fake):
    idx_end = "2023-06-14"
    fake["A"] = frame(walk(1, 1200), start="2018-11-01")
    fake["A"] = fake["A"][fake["A"].index <= idx_end]
    fake["B"] = frame(walk(2, 1200), start="2018-11-01")
    fake["B"] = fake["B"][fake["B"].index <= idx_end]
    R = mc.run(mc.Settings(weights={"A": 50, "B": 50}, sims=200, years=5))
    assert str(R["settings"]["history_end"]) == "2023-05-31"


def test_monte_carlo_contribution_only_hides_withdrawal_rates(fake):
    fake["A"] = frame(walk(1, 1500), start="2015-01-01")
    R = mc.run(mc.Settings(weights={"A": 1}, sims=200, years=10, flows=[mc.CashFlow(amount=500, freq="monthly")]))
    assert R["has_contributions"] and not R["show_withdrawal_rates"]
    assert "Safe withdrawal rate" not in mc.console(R)
    W = mc.run(mc.Settings(weights={"A": 1}, sims=200, years=10, flows=[mc.CashFlow(amount=-500)]))
    assert W["show_withdrawal_rates"]


def test_monte_carlo_stress_tests_and_age_horizon(fake):
    r = np.full(1500, 0.0004)
    r[600:700] = -0.004      # the worst stretch
    fake["A"] = frame(100 * np.cumprod(1 + r), start="2012-01-02")
    base = dict(weights={"A": 1}, sims=300, years=20, inflation=0.0)
    S = mc.run(mc.Settings(**base, stress="worst_sequence", stress_years=2))
    N = mc.run(mc.Settings(**base))
    assert S["stress"]["kind"] == "worst_sequence" and S["stress"]["return"] < 0
    assert S["final"]["50"] < N["final"]["50"]
    assert any("worst 2-year stretch" in n for n in S["notes"])
    K = mc.run(mc.Settings(**base, stress="shock", stress_shock=-0.3))
    assert K["max_drawdown"]["90"] <= -0.3 + 1e-9
    H = mc.run(mc.Settings(weights={"A": 1}, sims=200, age=60, until_age=95, inflation=0.0))
    assert H["settings"]["years"] == 35 and len(H["years"]) == 36
    with pytest.raises(ValueError):
        mc.run(mc.Settings(weights={"A": 1}, sims=200, age=70, until_age=60))


# ------------------------------------------------------------ 8 / 9. interpretation and names

def test_describe_keeps_groups_together_and_exact_weights():
    tree = {"weights": "inverse_vol", "children": [{"asset": "SPY"},
                                                   {"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "QQQ"}, {"asset": "BIL"}]},
                                                   {"weights": "specified", "w": [0.925, 0.075], "children": [{"asset": "TLT"}, {"weights": "equal", "children": [{"asset": "GLD"}, {"asset": "SPY"}]}]}]}
    lines = pf.describe(tree)
    assert lines[1] == "  SPY" and lines[2] == "  60% QQQ / 40% BIL"
    assert lines[3] == "  group:" and lines[4] == "    92.5% TLT" and lines[5].strip() == "7.5%:"
    flt = {"filter": {"select": "top", "n": 1, "by": "tret(tr, 20)"}, "universe": "children",
           "children": [{"asset": "SPY"}, {"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "QQQ"}, {"asset": "BIL"}]}]}
    assert "60% QQQ / 40% BIL" in pf.describe(flt)[2]
    assert pf.fmt_weight(1 / 3) == "33.33%" and pf.fmt_weight(0.6) == "60%"


def test_readable_default_names():
    t = {"if": "rsi(close, 10) > 79", "on": "TQQQ", "then": {"asset": "UVXY"},
         "else": {"if": 'sym("SPY").close > sma(sym("SPY").close, 200)', "on": "SPY", "then": {"asset": "TQQQ"}, "else": {"asset": "BIL"}}}
    assert pf.short_name(t, 200) == "If TQQQ RSI(10) > 79: UVXY, else If SPY price > SMA(200) of SPY price: TQQQ, else BIL"
    assert len(pf.short_name(t, 40)) == 40 and pf.short_name(t, 40).endswith("…")
    f = {"filter": {"select": "top", "n": 2, "by": "tret(tr, 126)"}, "universe": "children", "children": [{"asset": "SPY"}, {"asset": "TLT"}]}
    assert pf.short_name(f) == "Top 2 of SPY, TLT by 126-day return"
    from backtester import web
    spec = web._spec({"spec": {"kind": "allocation", "tree": {"if": "rsi(close, 10) > 79", "on": "SPY", "then": {"asset": "TLT"}, "else": {"asset": "SPY"}}}})
    assert spec.description == "If SPY RSI(10) > 79: TLT, else SPY"
    row = web._readable_label({"label": "Portfolio: if rsi(close, 10) > 79 (on SPY):", "spec": {"tree": spec.tree}})
    assert row["label"] == "If SPY RSI(10) > 79: TLT, else SPY"


def test_build_page_warns_about_units_and_leaves_hold_bars_blank():
    html = (report.ROOT / "backtester" / "webapp.html").read_text()
    assert 'id="s_hold_bars" type="number" min="0" placeholder="none"' in html and 'value="1"' not in html.split('id="s_hold_bars"')[1][:40]
    assert "Different units" in html and "function unitWarnings" in html
    assert 'href="#correlations"' in html and "/api/correlation" in html


# ------------------------------------------------------------ TradingView items: periods, exports, no trades, charts

def test_summary_row_and_stats_start_after_the_warm_up(fake):
    fake["X"] = frame(walk(5, 400), start="2019-01-01")
    s = Strategy(universe=["X"], entry="close > sma(close, 50)", hold_bars=2, cash_rate=None)
    res = runner.run(s)
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    from backtester import web
    row = web._summary_row("x", report._clean(A), res.kind, "x", s, res=res)
    assert row["start"] == str(fake["X"].index[49].date()) == str(A["stats"]["start"])


def test_no_trades_leads_with_no_trades_and_no_drawdown_dates(fake):
    fake["X"] = frame(walk(6, 300))
    s = Strategy(universe=["X"], entry="close < 0", hold_bars=1, cash_rate=0.02)
    A = report.analyze(runner.run(s), sensitivity=False, mc=False, detail=False)
    txt = report.console_summary(A)
    assert "No trades" in txt and "cash interest alone" in txt and "Max drawdown      none" in txt
    assert A["stats"]["max_dd_peak"] is None and report.run_payload(A, 0, A["result"].equity.index)["no_trades"]


def test_trades_carry_stop_and_target_levels_and_rule_state(fake):
    c = np.array([100, 100, 102, 104, 103, 101, 99, 97, 98, 100, 101, 102] * 5, dtype=float)
    fake["X"] = frame(c, highs=c * 1.01, lows=c * 0.99)
    s = Strategy(universe=["X"], entry="close > ref(close, 1)", exit_when="close < ref(close, 1)", stop_loss=0.05,
                 take_profit=0.10, trailing_stop=0.03, cash_rate=None)
    res = runner.run(s)
    t = res.trades.iloc[0]
    assert t.stop_level == pytest.approx(t.entry_price * 0.95) and t.target_level == pytest.approx(t.entry_price * 1.10)
    assert t.trail_level == pytest.approx(fake["X"].close[pd.Timestamp(t.entry_date)] * 0.97)
    P = report.ticker_chart(res, "X")
    assert len(P["rs"]) == len(P["dates"]) and P["rs_exit"]
    ent = (fake["X"].close > fake["X"].close.shift()).reindex(pd.to_datetime(P["dates"])).fillna(False).to_numpy()
    assert all((int(ch) & 1) == bool(e) for ch, e in zip(P["rs"], ent))


def test_other_ticker_filters_get_a_pane():
    panes = report.other_ticker_panes('rsi(close, 2) < 10 and sym("SPY").close > sma(sym("SPY").close, 200) and sym("VIX").rsi(14) < 70', "QQQ")
    names = [p["name"] for p in panes]
    assert names[0] == "SPY" and 'sma(sym("SPY").close, 200)' in panes[0]["calls"]
    assert any("VIX" in n and "RSI" in n for n in names)
    assert report._label('sma(sym("SPY").close, 200)') == "SPY sma(200)"


def test_dry_run_runs_the_same_validations(capsys):
    from backtester.__main__ import main
    assert main(["buy SPY when RSI(2) is below 10, hold 3 days", "--start", "2020-01-01", "--end", "2019-01-01", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "reversed" in err
    if not HAVE:
        return
    assert main(["hold 60% SPY and 40% TLT", "--start", "2030-01-01", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "No price data in the requested period" in err or "after the last date with data" in err
    assert main(["--tickers", "SPY", "--entry", "close > sma(close, 3000)", "--hold", "1", "--start", "1993-06-01",
                 "--end", "1995-06-01", "--dry-run"]) == 2
    assert "needs more history" in capsys.readouterr().err
    assert main(["buy SPY when RSI(2) is below 10, hold 3 days", "--dry-run"]) == 0


# ------------------------------------------------------------ QuantConnect items

def test_portfolio_shorts_pay_the_rebate_spread_and_borrow_fee(fake):
    fake["A"] = frame(walk(1, 500), start="2019-01-01")
    fake["B"] = frame(walk(2, 500), start="2019-01-01")
    tree = {"weights": "specified", "w": [1.3, -0.3], "children": [{"asset": "A"}, {"asset": "B"}]}
    kw = dict(tree=tree, rebalance="monthly", cash_rate=0.03, maintenance_margin=0.0)
    r0 = pf.run(pf.Portfolio(**kw, short_rebate_spread=0.0))
    r1 = pf.run(pf.Portfolio(**kw, short_rebate_spread=0.0025))
    r2 = pf.run(pf.Portfolio(**kw, short_rebate_spread=0.0025, borrow_fee=0.02))
    assert r1.interest < r0.interest and r2.equity.iloc[-1] < r1.equity.iloc[-1]
    for r in (r0, r1, r2):
        at = r.extras["attribution"]
        assert at["pnl"].sum() + r.interest - r.extras["fees"] == pytest.approx(r.equity.iloc[-1] - r.equity.iloc[0], abs=0.01)
        assert r.trades["pnl"].sum() == pytest.approx(at["pnl"].sum(), abs=0.01)
    b1 = r1.extras["attribution"].set_index("ticker").pnl["B"]
    b2 = r2.extras["attribution"].set_index("ticker").pnl["B"]
    assert b2 < b1   # the borrow fee is a cost of the short holding
    assert "short proceeds earn the cash rate less 0.25%/yr, 2.00%/yr borrow fee" in pf.Portfolio(**kw, borrow_fee=0.02).summary()


def test_volatility_drag_caveat_and_same_close_fill_note(fake):
    assert metrics.caveats({"cagr": -0.43, "sharpe": 0.53, "volatility": 1.2})
    assert not metrics.caveats({"cagr": 0.1, "sharpe": 0.5, "volatility": 0.2})
    fake["A"] = frame(walk(1, 400), start="2019-01-01")
    fake["B"] = frame(walk(2, 400), start="2019-01-01")
    tree = {"if": "close > sma(close, 20)", "on": "A", "then": {"asset": "A"}, "else": {"asset": "B"}}
    p = pf.Portfolio(tree=tree, rebalance="daily", cash_rate=None)
    report.analyze(pf.run(p), sensitivity=False, mc=False, detail=False)
    assert any("market-on-close" in n and "next open" in n for n in p.notes)
    q = pf.Portfolio(tree=tree, rebalance="daily", cash_rate=None, fill="next_open")
    report.analyze(pf.run(q), sensitivity=False, mc=False, detail=False)
    assert not any("market-on-close" in n for n in q.notes)


# ------------------------------------------------------------ period ends on the scheduled calendar

def test_schedule_uses_the_scheduled_period_end():
    cal = pd.DatetimeIndex(pd.bdate_range("2001-09-04", "2001-09-28")).drop(pd.bdate_range("2001-09-11", "2001-09-14"))
    s = pf._schedule(cal, "weekly")
    d = dict(zip(cal.date, s))
    assert not d[pd.Timestamp("2001-09-10").date()]          # nobody knew on 9/10 that the week was over
    assert d[pd.Timestamp("2001-09-17").date()]              # so the rebalance happens on the next bar
    assert d[pd.Timestamp("2001-09-07").date()] and d[pd.Timestamp("2001-09-21").date()]


@needs_data
def test_weekly_rebalance_is_truncation_invariant_around_9_11():
    tree = {"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "QQQ"}]}
    kw = dict(tree=tree, rebalance="weekly", start="2001-08-01", slippage_bps=10, commission=1.0)
    full = pf.run(pf.Portfolio(**kw, end="2001-10-31"))
    cut = pf.run(pf.Portfolio(**kw, end="2001-09-10"))
    common = cut.equity.index
    assert np.allclose(full.equity.reindex(common).to_numpy(), cut.equity.to_numpy(), rtol=0, atol=1e-9)
    days = set(pd.to_datetime(full.orders["date"]).dt.date)
    assert pd.Timestamp("2001-09-10").date() not in days and pd.Timestamp("2001-09-17").date() in days
