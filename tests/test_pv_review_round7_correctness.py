"""Round 7 (Portfolio Visualizer expert review): corporate-bond SIM gap, data-quality guard for holes in SIMs,
benchmarks and report sections after the money runs out, monthly-stepped series (statistics, correlations, factor
regressions), the compare page's common period, "schedule or band" rebalancing, CPI-U NSA inflation and withdrawal
rates measured from the start of withdrawals."""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from backtester import correlation, data, factors, metrics, parser, report, runner
from backtester.portfolio import Portfolio

ROOT = pathlib.Path(__file__).parents[1]
AVAIL = set(data.available_tickers())
needs = lambda *ts: pytest.mark.skipif(not set(ts) <= AVAIL, reason="price data not downloaded")  # noqa: E731


@pytest.fixture(scope="module")
def fd():
    spec = importlib.util.spec_from_file_location("fetch_data_r7", ROOT / "scripts" / "fetch_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def port(text) -> Portfolio:
    p = parser.parse(text)
    assert isinstance(p, Portfolio), text
    return p


# ------------------------------------------------------------------ 1. corporates: no 1983-85 hole

def test_corporate_yield_uses_monthly_until_both_daily_series_exist(fd, tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "FACTORS", tmp_path)          # the NYSE schedule alone
    months = pd.date_range("1980-01-01", "1987-12-01", freq="MS")
    aaa = pd.Series(np.linspace(11.0, 9.0, len(months)), index=months)
    baa = aaa + 1.5
    d_aaa = pd.Series(10.0, index=pd.bdate_range("1983-01-03", "1987-12-31"))   # DAAA from 1983
    d_baa = pd.Series(12.0, index=pd.bdate_range("1986-01-02", "1987-12-31"))   # DBAA only from 1986
    y = fd.corporate_yield(d_aaa, d_baa, aaa, baa, since="1980-01-01")
    assert fd.internal_gaps(y) == []
    # 1983-85: the monthly average (held from each month's last session), not NaN and not DAAA alone
    jun84 = (aaa + baa)[pd.Timestamp("1984-05-01")] / 2
    assert y[pd.Timestamp("1984-06-15")] == pytest.approx(jun84)
    assert y[(y.index >= "1983-01-01") & (y.index < "1986-01-01")].notna().all()
    assert len(y[(y.index >= "1983-02-01") & (y.index < "1985-12-01")]) > 700          # every session is there
    # from the first day both daily series exist: their daily average
    assert y[pd.Timestamp("1986-01-02")] == pytest.approx(11.0)
    assert y.index.is_monotonic_increasing and not y.index.duplicated().any()
    # the old construction (sum of the two daily series, cut at the first DAAA date) left the hole
    old = (d_aaa + d_baa) / 2
    assert old[old.index < "1986-01-02"].isna().all()


def test_internal_gaps_flag_holes_longer_than_ten_sessions(fd):
    idx = pd.bdate_range("1980-01-01", "1990-12-31")
    s = pd.Series(1.0, index=idx)
    assert fd.internal_gaps(s) == []
    holed = s[(s.index < "1983-01-31") | (s.index >= "1986-01-02")]
    g = fd.internal_gaps(holed)
    assert len(g) == 1 and g[0][0] == pd.Timestamp("1983-01-28") and g[0][1] == pd.Timestamp("1986-01-02") and g[0][2] > 700
    # a single holiday week is fine
    assert fd.internal_gaps(s.drop(pd.bdate_range("1985-12-23", "1985-12-31"))) == []


def _write_sim(dirpath, name, level: pd.Series):
    df = pd.DataFrame({"open": level, "high": level, "low": level, "close": level, "volume": 0, "dividend": 0.0,
                       "adj_close": level})
    df.index.name = "date"
    df.to_csv(dirpath / f"{name}.csv")


@pytest.fixture
def tmp_prices(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PRICES", tmp_path)
    for f in (data.load, data.stepped_ranges, data.data_gaps):
        f.cache_clear()
    yield tmp_path
    for f in (data.load, data.stepped_ranges, data.data_gaps):
        f.cache_clear()


def test_a_sim_with_a_hole_gets_a_runtime_note(tmp_prices):
    from backtester import calendar as nyse
    idx = pd.DatetimeIndex([d for d in pd.bdate_range("1980-01-02", "1990-12-31") if nyse.is_session(d)])
    lvl = pd.Series(100 * 1.0002 ** np.arange(len(idx)), index=idx)
    _write_sim(tmp_prices, "AAASIM", lvl)
    _write_sim(tmp_prices, "HOLESIM", lvl[(lvl.index < "1983-01-31") | (lvl.index >= "1986-01-02")])
    assert data.data_gaps("AAASIM") == () and len(data.data_gaps("HOLESIM")) == 1
    assert data.gap_notes(["HOLESIM"], "1987-01-01", "1990-01-01") == []       # outside the period: no note
    p = Portfolio(tree={"weights": "specified", "w": [0.5, 0.5], "children": [{"asset": "AAASIM"}, {"asset": "HOLESIM"}]},
                  rebalance="yearly", start="1981-01-01", cash_rate=None)
    runner.run(p)
    assert any(n.startswith("Data gap: HOLESIM") and "1986-01-02" in n for n in p.notes)


# ------------------------------------------------------------------ 5. monthly steps: detection

def test_stepped_ranges_of_synthetic_series():
    idx = pd.bdate_range("2000-01-03", "2004-12-31")
    rng = np.random.default_rng(3)
    daily = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))), index=idx)
    assert data.stepped_ranges_of(daily) == []
    # monthly steps (zero on every other day) until 2002, daily after
    lvl = daily.copy()
    early = lvl.index < "2002-01-01"
    me = lvl[early].groupby(lvl[early].index.to_period("M")).transform("last")
    lvl[early] = me.groupby(me.index.to_period("M")).transform(lambda x: x.iloc[-1])
    lvl[early] = lvl[early].groupby(lvl[early].index.to_period("M")).transform("last").shift(1).bfill()
    rng_ = data.stepped_ranges_of(lvl)
    assert len(rng_) == 1 and rng_[0][0] <= pd.Timestamp("2000-03-01") and pd.Timestamp("2001-11-01") <= rng_[0][1] < pd.Timestamp("2002-02-01")
    # a bond model that accrues daily and re-prices once a month is stepped; a constant accrual is not
    acc = pd.Series(1.0001 ** (idx - idx[0]).days.to_numpy(), index=idx)     # interest per calendar day
    assert data.stepped_ranges_of(acc) == []
    jumps = acc.copy()
    last = jumps.groupby(jumps.index.to_period("M")).tail(1).index
    fac = pd.Series(1.0, index=idx)
    fac[last] = 1 + rng.normal(0, 0.01, len(last))
    assert len(data.stepped_ranges_of(acc * fac.cumprod())) == 1


@needs("EEMSIM", "SPYSIM", "EFASIM", "VNQSIM")
def test_stepped_ranges_of_the_sims():
    assert data.stepped_ranges("SPYSIM") == ()
    (a, b), = data.stepped_ranges("EEMSIM")
    assert a <= pd.Timestamp("1990-01-01") and b >= pd.Timestamp("2002-12-01")
    found = data.stepped_in(["SPYSIM", "EFASIM", "VNQSIM"], "1976-01-01", "1989-12-31")
    assert set(found) == {"EFASIM", "VNQSIM"}


# ------------------------------------------------------------------ 5a. backtests holding a stepped series

@needs("EEMSIM", "SPYSIM")
def test_backtest_with_a_stepped_holding_uses_monthly_statistics():
    res = runner.run(port("hold 50% SPYSIM and 50% EEMSIM, rebalance monthly, since 1990, until 2002"))
    A = report.analyze(res, sensitivity=False, mc=False)
    st = A["stats"]
    assert A["return_basis"] == "monthly" and "EEMSIM" in A["stepped"]
    mr = metrics.monthly_returns(A["nav"])
    assert st["volatility"] == pytest.approx(mr.std() * np.sqrt(12))
    assert st["sharpe"] == pytest.approx(st["sharpe_monthly"])
    for k in ("best_day", "worst_day", "var_95_daily", "pct_positive_days"):
        assert not np.isfinite(st[k])
    assert st["kurtosis"] == pytest.approx(mr.kurt())
    assert any(w["code"] == "monthly_steps" for w in A["warnings"])
    assert A["relative"].get("freq") == "monthly"
    if A["factors"]:
        assert A["factors"]["freq"] == "monthly"
    assert "Monthly steps:" in report.console_summary(A)
    # a daily-only portfolio keeps daily statistics
    B = report.analyze(runner.run(port("hold 50% SPYSIM and 50% TLTSIM, rebalance monthly, since 1990, until 2002")),
                       sensitivity=False, mc=False)
    assert B["return_basis"] == "daily" and np.isfinite(B["stats"]["best_day"])


@needs("SPYSIM", "EFASIM", "VNQSIM")
def test_correlations_switch_to_monthly_when_a_series_is_stepped():
    R = correlation.analyze(["SPYSIM", "EFASIM", "VNQSIM"], "daily", None, "1976-01-01", "1989-12-31")
    assert R["freq"] == "monthly" and R["window"] == 36 and any("Monthly returns used" in n for n in R["notes"])
    assert R["observations"] < 200
    S = correlation.analyze(["SPYSIM", "TLTSIM"], "daily", None, "1976-01-01", "1989-12-31")
    assert S["freq"] == "daily"
    st = {s["ticker"]: s for s in R["stats"]}
    assert st["EFASIM"].get("return_basis") == "monthly" and "return_basis" not in st["SPYSIM"]


@needs("EEMSIM")
def test_factor_regression_on_a_stepped_series_is_monthly():
    r, name = factors.returns_for("EEMSIM")
    R = factors.analyze(r, "ff3", "daily", "1990-01-01", "2002-12-31", name=name)
    assert R["freq"] == "monthly" and any("Monthly regression" in n for n in R["notes"])
    assert R["observations"] < 200


# ------------------------------------------------------------------ 3 + 4. after the money runs out

RUIN = ("hold 60% SPYSIM and 40% IEFSIM, withdraw 4% per year adjusted for inflation, starting with $1,000,000, "
        "rebalance yearly, since 1966")


@needs("SPYSIM", "IEFSIM", "SPY", "QQQ")
def test_depleted_portfolio_report_stops_and_benchmarks_all_get_flows():
    p = port(RUIN)
    res = runner.run(p)
    dep = res.extras.get("depleted")
    assert dep is not None and dep < pd.Timestamp("2000-01-01")
    A = report.analyze(res, sensitivity=False, mc=True)
    # 3. no benchmark without the cash flows: QQQ starts after the money ran out and is left out, with a note
    assert "QQQ buy & hold" not in A["benchmarks"]
    assert set(A["benchmarks"]) == set(A["benchmarks_with_flows"])
    assert any(n.startswith("QQQ buy & hold is left out") for n in p.notes)
    assert A["benchmark_cash"]["SPY buy & hold"]["from"] == pd.Timestamp("1993-01-29").date()
    # 4. everything stops on the depletion day
    assert int(A["yearly"].index.astype(int).max()) == dep.year
    assert A["result"].equity.index[-1] == dep and A["result"].holdings.index[-1] <= dep
    assert all(b.index[-1] <= dep for b in A["benchmarks"].values())
    assert all(pd.Timestamp(a["to"]) <= dep for a in A["asset_stats"])
    assert A["exposure"]["time_in_market"] > 0.99
    assert A["monte_carlo"] and 0 <= A["monte_carlo"]["success_rate"] <= 1
    P = report.build_payload([A])
    assert P["dates"][-1] == str(dep.date())
    assert pd.Timestamp(P["runs"][0]["holdings"]["dates"][-1]) <= dep + pd.Timedelta(days=7)
    text = report.console_summary(A)
    assert "QQQ" not in text.split("Returns by year")[1].splitlines()[1]
    assert "the day the money ran out" in text


# ------------------------------------------------------------------ 6. compare: the common period is common

@needs("SPYSIM", "SPY", "QQQ", "TLTSIM", "GLDSIM")
def test_compare_common_period_lists_later_benchmarks_separately():
    runs = [report.analyze(runner.run(port(t)), sensitivity=False, mc=False)
            for t in ("hold 60% SPYSIM and 40% TLTSIM, rebalance yearly, since 1978",
                      "hold 50% SPYSIM and 50% GLDSIM, rebalance yearly, since 1978")]
    C = report.common_window_stats(runs)
    assert str(C["start"]).startswith("1978")
    for k, st in C["columns"].items():
        assert str(st["start"]) <= "1978-01-06", k            # everything under the heading covers the period
    assert {"SPY buy & hold", "QQQ buy & hold"} <= set(C["benchmarks_own"])
    assert C["benchmarks_own_from"]["SPY buy & hold"] == "1993-01-29"
    assert "SPYSIM buy & hold" in C["columns"]


# ------------------------------------------------------------------ 7. "schedule or band"

@needs("SPY", "TLT")
def test_schedule_or_band_rebalances_on_both():
    n = {}
    for k, t in {"both": "hold 70% SPY and 30% TLT, rebalance semi-annually or when any weight drifts more than 5%",
                 "sched": "hold 70% SPY and 30% TLT, rebalance semi-annually",
                 "band": "hold 70% SPY and 30% TLT, rebalance when any weight drifts more than 5%"}.items():
        res = runner.run(port(t))
        n[k] = res.extras["rebalances"]
    assert n["both"] >= n["sched"] and n["both"] > n["band"]
    assert n["both"] <= n["sched"] + n["band"]
    p = port("hold 60% SPY and 40% TLT, rebalance quarterly or when any weight drifts more than 5%")
    assert "and also whenever a holding drifts 5%" in p.summary()


# ------------------------------------------------------------------ 8. CPI-U NSA; SWR from the start of withdrawals

def test_inflation_column_uses_cpi_u_not_seasonally_adjusted():
    m = data.cpi_monthly()
    if m.empty or m.index[0] > pd.Timestamp("1966-12-01"):
        pytest.skip("no CPI data")
    assert metrics.calendar_inflation(pd.Timestamp("1967-01-03"), pd.Timestamp("1967-12-29")) == pytest.approx(0.0304, abs=0.0006)
    nsa = data.DATA / "macro" / "CPIAUCNS.csv"
    if nsa.exists():
        raw = pd.read_csv(nsa, parse_dates=["date"], index_col="date")["value"]
        assert m[pd.Timestamp("2000-06-01")] == pytest.approx(raw[pd.Timestamp("2000-06-01")])


@needs("SPYSIM", "IEFSIM")
def test_withdrawal_rates_are_measured_from_the_start_of_withdrawals():
    p = port("add $1,000 a month for 20 years, then withdraw $50,000 a year adjusted for inflation, hold 60% SPYSIM and "
             "40% IEFSIM, since 1980, starting with $10,000, rebalance yearly")
    A = report.analyze(runner.run(p), sensitivity=False, mc=False)
    wr = A["withdrawal_rates"]
    assert wr and wr["base"] == "withdrawal_start" and wr["base_date"].year == 2000 and wr["base_balance"] > 100_000
    assert str(wr["from"]).startswith("2000")
    assert "of the balance when withdrawals start" in report.console_summary(A)
    # a plain withdrawal plan keeps the starting balance as the base
    B = report.analyze(runner.run(port("hold 60% SPYSIM and 40% IEFSIM, withdraw 4% per year adjusted for inflation, "
                                       "starting with $1,000,000, rebalance yearly, since 1980")), sensitivity=False, mc=False)
    assert B["withdrawal_rates"] and "base" not in B["withdrawal_rates"]
