"""Round 6: official monthly Fama-French factors (US and international regions), regional models, AQR QMJ/BAB,
returns-based style analysis, Monte Carlo cash-flow phases, the SSA mortality horizon and withdrawals capped at
the balance. The file parsers are tested on synthetic samples laid out like the real downloads."""
import numpy as np
import pandas as pd
import pytest

from backtester import data, factors, lifetable, montecarlo as mc, sources, style

HAVE = {"SPY", "IEF", "BIL", "EFA"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")

# ------------------------------------------------------------------ file formats

FRENCH_MONTHLY = """This file was created by CMPT_ME_BEME_RETS using the 202507 CRSP database.
The 1-month TBill return is from Ibbotson and Associates, Inc.

,Mkt-RF,SMB,HML,RF
192607,    2.89,   -2.55,   -2.39,    0.22
192608,    2.64,   -1.14,    3.81,    0.25
192609,    0.38,   -1.36,    0.05,    0.23

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1927,   29.47,   -2.46,   -3.75,    3.12
1928,   35.39,    4.41,   -5.83,    3.56

Copyright 2025 Eugene F. Fama and Kenneth R. French
"""
FRENCH_REGION_MOM = """This file was created using the 202507 Bloomberg database.
Missing data are indicated by -99.99.

,WML
199011,   -99.99
199012,    1.25
199101,   -0.87
"""
FRENCH_DAILY = """This file was created by CMPT_ME_BEME_RETS_DAILY using the 202507 CRSP database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,   -0.67,    0.00,   -0.34,   -0.01,    0.15,    0.012
19630702,    0.79,   -0.26,    0.26,   -0.07,   -0.20,    0.012
"""


def test_parse_french_monthly_skips_the_annual_section():
    df = sources.parse_french_csv(FRENCH_MONTHLY)
    assert list(df.columns) == ["date", "Mkt-RF", "SMB", "HML", "RF"]
    assert list(df["date"]) == ["1926-07-31", "1926-08-31", "1926-09-30"]
    assert df["Mkt-RF"].iloc[0] == pytest.approx(0.0289) and df["RF"].iloc[1] == pytest.approx(0.0025)


def test_parse_french_regional_momentum_renames_wml_and_blanks_missing():
    df = sources.parse_french_csv(FRENCH_REGION_MOM)
    assert list(df.columns) == ["date", "Mom"]
    assert np.isnan(df["Mom"].iloc[0]) and df["Mom"].iloc[1] == pytest.approx(0.0125)


def test_parse_french_daily():
    df = sources.parse_french_csv(FRENCH_DAILY.replace("\n", "\r\n"))
    assert list(df["date"]) == ["1963-07-01", "1963-07-02"] and df["CMA"].iloc[0] == pytest.approx(0.0015)


def aqr_raw(percent_strings=True):
    """An AQR 'QMJ Factors' sheet as read with header=None: notes, the EQUITIES row, the header, data."""
    cols = ["DATE", "AUS", "JPN", "USA", "Global", "Global Ex USA", "Europe", "North America", "Pacific"]
    rows = [["AQR Capital Management, LLC — Quality Minus Junk: Factors, Monthly"] + [None] * 8,
            ["This file contains monthly self-financing excess returns"] + [None] * 8,
            [None] * 9, [None, "EQUITIES"] + [None] * 5 + ["Aggregate Equity Portfolios", None], cols]
    vals = [("07/31/1957", [None, None, 0.0112, None, None, None, None, None]),
            ("08/31/1957", [None, None, 0.0049, None, None, None, None, None]),
            ("01/31/1990", [0.01, -0.02, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008])]
    for d, v in vals:
        cells = [f"{x * 100:.2f}%" if (percent_strings and x is not None) else x for x in v]
        rows.append([d if percent_strings else pd.Timestamp(d)] + cells)
    rows.append(["Sources and Definitions"] + [None] * 8)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("strings", [True, False])
def test_parse_aqr_sheet(strings):
    df = sources.parse_aqr_sheet(aqr_raw(strings))
    assert list(df["date"]) == ["1957-07-31", "1957-08-31", "1990-01-31"]
    assert df["USA"].iloc[0] == pytest.approx(0.0112) and df["Europe"].iloc[2] == pytest.approx(0.006)
    assert np.isnan(df["Europe"].iloc[0]) and "AUS" in df and "JPN" in df


def test_fetch_list_has_official_monthly_and_regional_files():
    names = dict(sources.FRENCH_FILES)
    assert names["ff3_monthly"] == "F-F_Research_Data_Factors_CSV.zip"
    assert names["ff5_monthly"] == "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    assert names["mom_monthly"] == "F-F_Momentum_Factor_CSV.zip"
    for r in ("developed", "developed_ex_us", "europe", "japan", "asia_pacific_ex_japan", "north_america"):
        for k in ("ff3", "ff5", "mom"):
            assert f"{r}_{k}_monthly" in names and f"{r}_{k}_daily" in names
    assert names["asia_pacific_ex_japan_mom_monthly"] == "Asia_Pacific_ex_Japan_MOM_Factor_CSV.zip"
    assert names["emerging_mom_monthly"] == "Emerging_MOM_Factor_CSV.zip" and "em_ff5_monthly" in names
    assert {n for n, _, _ in sources.AQR_FILES} == {"aqr_qmj_monthly", "aqr_bab_monthly"}


# ------------------------------------------------------------------ models

def test_regional_models_and_add_ons_resolve():
    assert factors.resolve("europe_ff5") == "europe_ff5"
    assert factors.resolve("em_ff3") == "emerging_ff3" and factors.resolve("apxj_carhart") == "asia_pacific_ex_japan_carhart"
    assert factors.resolve("intl") == "dev_ff3" and factors.resolve("dev_ff5") == "developed_ex_us_ff5"
    assert factors.resolve("japan") == "japan_ff3" and factors.resolve("developed") == "developed_ff3"
    label, cols, region = factors.model_spec("ff5+qmj+bab")
    assert cols == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "QMJ", "BAB"] and region == "us" and "QMJ" in label
    assert factors.model_spec("Europe_FF3+MOM")[1:] == (["Mkt-RF", "SMB", "HML", "Mom"], "europe")
    with pytest.raises(ValueError):
        factors.resolve("qmj")
    with pytest.raises(ValueError, match="add-on"):
        factors.resolve("ff3+nonsense")
    keys = {m["key"]: m for m in factors.model_list()}
    assert keys["north_america_ff6"]["region"] == "north_america" and keys["dev_ff3"]["region"] == "developed_ex_us"
    # composites are not added to MODELS (the US-model tests loop over it)
    assert "ff5+qmj+bab" not in factors.MODELS


def test_auto_model_and_region_hints():
    assert factors.auto_model("EFA") == "dev_ff3" and factors.auto_model("VGK", "ff5") == "europe_ff5"
    assert factors.auto_model("EEM") == "emerging_ff3" and factors.auto_model("SPY") == "ff3"
    assert factors.auto_model("ZZZZ") == "ff3"
    assert "try --model dev_ff3" in factors.model_hint("EFA", "ff3")
    assert factors.model_hint("SPY", "europe_ff3") and factors.model_hint("SPY", "ff5+qmj") is None
    assert factors.model_hint("ZZZZ", "ff3") is None


@pytest.fixture
def fdir(tmp_path, monkeypatch):
    """An empty data directory for factor files; caches cleared before and after."""
    (tmp_path / "factors").mkdir()
    monkeypatch.setattr(data, "DATA", tmp_path)
    factors.factor_data.cache_clear()
    yield tmp_path / "factors"
    factors.factor_data.cache_clear()


def _synthetic_factors(n_months=60, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    return pd.DataFrame({"Mkt-RF": rng.normal(0.008, 0.04, n_months), "SMB": rng.normal(0, 0.02, n_months),
                         "HML": rng.normal(0, 0.02, n_months), "RF": 0.001}, index=idx)


def _write(df, path):
    out = df.copy()
    out.index.name = "date"
    out.to_csv(path)


def _month_end_returns(y):
    """'Daily' returns with one observation per month (on the month end), so monthly compounding gives y."""
    return pd.Series(y.to_numpy(), index=y.index)


def test_monthly_regression_uses_the_official_monthly_file(fdir):
    fm = _synthetic_factors()
    _write(fm, fdir / "ff3_monthly.csv")
    # a daily file whose compounded months differ completely: must not be used for monthly regressions
    days = pd.bdate_range("2010-01-01", "2014-12-31")
    daily = pd.DataFrame({"Mkt-RF": 0.0005, "SMB": 0.0, "HML": 0.0, "RF": 0.00004}, index=days)
    _write(daily, fdir / "ff3_daily.csv")
    y = fm["RF"] + 1.2 * fm["Mkt-RF"] + 0.3 * fm["SMB"] - 0.2 * fm["HML"]
    R = factors.analyze(_month_end_returns(y), "ff3", "monthly", name="TEST")
    load = {c["factor"]: c["loading"] for c in R["coefficients"]}
    assert load["Mkt-RF"] == pytest.approx(1.2, abs=1e-9) and load["SMB"] == pytest.approx(0.3, abs=1e-9)
    assert R["r_squared"] == pytest.approx(1.0) and any("official monthly" in n for n in R["notes"])


def test_monthly_falls_back_to_compounded_daily_with_a_note(fdir):
    days = pd.bdate_range("2010-01-01", "2014-12-31")
    rng = np.random.default_rng(3)
    daily = pd.DataFrame({"Mkt-RF": rng.normal(0.0004, 0.01, len(days)), "SMB": rng.normal(0, 0.005, len(days)),
                          "HML": rng.normal(0, 0.005, len(days)), "RF": 0.00004}, index=days)
    _write(daily, fdir / "ff3_daily.csv")
    f, notes = factors.factor_data("ff3", "monthly")
    jan = daily.loc["2010-01"]
    assert f.loc["2010-01-31", "Mkt-RF"] == pytest.approx(np.prod(1 + jan["Mkt-RF"] + jan["RF"]) - np.prod(1 + jan["RF"]))
    assert any("compounded from the daily file" in n for n in notes)
    # daily regressions keep using the daily file
    assert len(factors.factor_table("ff3", "daily")) == len(days)


def test_regional_models_read_regional_files(fdir):
    fm = _synthetic_factors(seed=5)
    mom = pd.DataFrame({"WML": np.random.default_rng(6).normal(0, 0.03, len(fm))}, index=fm.index)
    _write(fm, fdir / "europe_ff3_monthly.csv")
    _write(mom, fdir / "europe_mom_monthly.csv")
    y = fm["RF"] + 0.9 * fm["Mkt-RF"] + 0.5 * mom["WML"]
    R = factors.analyze(_month_end_returns(y), "europe_carhart", "monthly", name="VGK")
    load = {c["factor"]: c["loading"] for c in R["coefficients"]}
    assert R["region"] == "europe" and load["Mom"] == pytest.approx(0.5, abs=1e-9)
    with pytest.raises(ValueError, match="missing"):
        factors.factor_data("japan_ff3", "monthly")
    with pytest.raises(ValueError, match="monthly only"):
        factors.factor_data("emerging_ff3", "daily")


def test_emerging_three_factors_come_from_the_five_factor_file(fdir):
    fm = _synthetic_factors(seed=7)
    fm["RMW"], fm["CMA"] = np.nan, np.nan
    _write(fm, fdir / "em_ff5_monthly.csv")
    f, _ = factors.factor_data("emerging_ff3", "monthly")
    assert len(f) == len(fm) and list(f.columns) == ["Mkt-RF", "SMB", "HML", "RF"]


def test_aqr_qmj_and_bab_use_the_regions_column(fdir):
    fm = _synthetic_factors(seed=9)
    _write(fm, fdir / "ff3_monthly.csv")
    _write(fm, fdir / "europe_ff3_monthly.csv")
    rng = np.random.default_rng(10)
    q = pd.DataFrame({"USA": rng.normal(0, 0.02, len(fm)), "Europe": rng.normal(0, 0.02, len(fm))}, index=fm.index)
    b = pd.DataFrame({"USA": rng.normal(0, 0.03, len(fm)), "Europe": rng.normal(0, 0.03, len(fm))}, index=fm.index)
    _write(q, fdir / "aqr_qmj_monthly.csv")
    _write(b, fdir / "aqr_bab_monthly.csv")
    y = fm["RF"] + fm["Mkt-RF"] + 0.4 * q["USA"] - 0.1 * b["USA"]
    R = factors.analyze(_month_end_returns(y), "ff3+qmj+bab", "monthly", name="X")
    load = {c["factor"]: c["loading"] for c in R["coefficients"]}
    assert load["QMJ"] == pytest.approx(0.4, abs=1e-9) and load["BAB"] == pytest.approx(-0.1, abs=1e-9)
    ye = fm["RF"] + fm["Mkt-RF"] + 0.7 * q["Europe"]
    E = factors.analyze(_month_end_returns(ye), "europe_ff3+qmj", "monthly", name="X")
    assert {c["factor"]: c["loading"] for c in E["coefficients"]}["QMJ"] == pytest.approx(0.7, abs=1e-9)
    with pytest.raises(ValueError, match="monthly"):
        factors.factor_data(factors.resolve("ff3+qmj"), "daily")
    with pytest.raises(ValueError, match="no QMJ"):
        factors.factor_data(factors.resolve("emerging_ff3+qmj"), "monthly")


# ------------------------------------------------------------------ style analysis

def test_style_solver_recovers_constrained_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(0.006, 0.04, (200, 5))
    w = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    y = X @ w + 0.001 + rng.normal(0, 0.002, 200)       # a constant selection return does not bias the weights
    got = style.solve(y, X)
    assert got.sum() == pytest.approx(1.0) and (got >= 0).all()
    assert np.allclose(got, w, atol=0.02)
    # a fund outside the assets' span: weights stay non-negative and add up to one
    got2 = style.solve(-X[:, 0], X)
    assert got2.sum() == pytest.approx(1.0) and (got2 >= -1e-12).all()


def test_style_analysis_on_synthetic_assets(monkeypatch):
    idx = pd.bdate_range("2005-01-03", "2014-12-31")
    rng = np.random.default_rng(4)
    A = pd.DataFrame(rng.normal(0.0003, 0.01, (len(idx), 3)), index=idx, columns=["AA", "BB", "CC"])
    monkeypatch.setattr(style, "asset_monthly", lambda t: style.monthly_returns(A[t]))
    fund = 0.7 * A["AA"] + 0.3 * A["CC"]                  # rebalanced daily
    R = style.analyze(fund, ["AA", "BB", "CC"], window=36, name="FUND")
    assert R["weights"]["AA"] == pytest.approx(0.7, abs=0.02) and R["weights"]["CC"] == pytest.approx(0.3, abs=0.02)
    assert R["r_squared"] > 0.99 and len(R["rolling"]["dates"]) == R["observations"] - 35
    assert all(abs(sum(R["rolling"]["weights"][k][i] for k in R["weights"]) - 1) < 1e-3 for i in range(len(R["rolling"]["dates"])))
    assert "FUND" in style.console(R)


@needs_data
def test_style_analysis_of_a_60_40_portfolio():
    r, name = factors.returns_for({"SPY": 60, "IEF": 40})
    R = style.analyze(r, ["SPY", "IEF", "BIL", "EFA"], name=name)
    assert R["weights"]["SPY"] == pytest.approx(0.6, abs=0.05) and R["weights"]["IEF"] == pytest.approx(0.4, abs=0.05)
    assert R["r_squared"] > 0.98


@needs_data
def test_style_default_asset_classes():
    assets, notes = style.default_assets()
    assert len(assets) >= 5 and all(isinstance(t, str) for t in assets.values())


# ------------------------------------------------------------------ life table and Monte Carlo

@pytest.mark.parametrize("age", [0, 40, 65, 85, 100])
def test_life_table_reproduces_ssa_life_expectancy(age):
    for sex, le in (("male", lifetable.LE_MALE), ("female", lifetable.LE_FEMALE)):
        S = lifetable.survival(age, sex, 120)
        assert lifetable.life_expectancy(S) == pytest.approx(le[age], abs=0.01)
        assert (np.diff(S) <= 0).all() and S[0] == 1.0


def test_joint_survival_is_the_longer_life():
    m, f = lifetable.survival(65, "male", 40), lifetable.survival(63, "female", 40)
    j = lifetable.joint_survival(65, "male", 63, "female", 40)
    assert (j >= np.maximum(m, f) - 1e-12).all()
    assert j[10] == pytest.approx(1 - (1 - m[10]) * (1 - f[10]))
    assert len(lifetable.QX_MALE) == len(lifetable.QX_FEMALE) == 120


def test_withdrawals_are_capped_at_the_balance():
    P = np.zeros((1, 72))
    ci = np.ones((1, 73))
    out = mc.simulate(P, ci, 100.0, [mc.CashFlow(amount=-30, freq="yearly", inflation_adjusted=False)])
    B = out["B"][0, ::12]
    assert B.tolist() == [100, 70, 40, 10, 0, 0, 0]
    assert out["withdrawn"][0] == pytest.approx(100) and out["requested"][0] == pytest.approx(180)
    assert (out["B"] >= 0).all()
    # a contribution in the same period is available to the withdrawal
    out2 = mc.simulate(np.zeros((1, 24)), np.ones((1, 25)), 0.0,
                       [mc.CashFlow(amount=50, inflation_adjusted=False), mc.CashFlow(amount=-80, inflation_adjusted=False)])
    assert out2["withdrawn"][0] == pytest.approx(100) and out2["B"][0, -1] == 0


def test_survival_weighted_success():
    S = np.array([1.0, 0.5, 0.2, 0.0])
    assert mc.survival_weighted_success(np.ones(4), S) == pytest.approx(1.0)
    # money runs out during year 2: those who die in year 1 succeed (0.5), the rest fail
    assert mc.survival_weighted_success(np.array([1, 1, 0, 0.0]), S) == pytest.approx(0.5)
    assert mc.survival_weighted_success(np.array([1, 1, 1, 0.0]), S) == pytest.approx(0.8)


def _series(months=240, seed=2):
    idx = pd.date_range("1990-01-31", periods=months, freq="ME")
    return pd.Series(np.random.default_rng(seed).normal(0.006, 0.035, months), index=idx)


def test_mortality_horizon_weights_success_by_survival():
    s = mc.Settings(series=_series(), start_balance=1_000_000, flows=[mc.CashFlow(amount=-70_000)], inflation=0.02,
                    horizon="mortality", age=70, sex="female", sims=500, seed=1)
    R = mc.run(s)
    M = R["mortality"]
    assert R["settings"]["years"] == len(M["survival_by_year"]) - 1 and 30 <= R["settings"]["years"] <= 50
    assert R["prob_success"] == pytest.approx(mc.survival_weighted_success(np.array(R["success_by_year"]), np.array(M["survival_by_year"])))
    assert R["prob_success"] >= R["prob_success_to_horizon"]
    assert M["life_expectancy"] == pytest.approx(lifetable.LE_FEMALE[70], abs=0.01)
    assert "life table" in " ".join(R["notes"]) and "withdrawals" in R
    with pytest.raises(ValueError, match="age"):
        mc.run(mc.Settings(series=_series(), horizon="mortality", sims=200))
    J = mc.run(mc.Settings(series=_series(), flows=[mc.CashFlow(amount=-70_000)], inflation=0.02, horizon="mortality",
                           age=70, sex="joint", age2=68, sims=500, seed=1))
    assert J["settings"]["years"] > R["settings"]["years"] and J["mortality"]["age2"] == 68


def test_contribute_then_withdraw_phases():
    flows = [mc.CashFlow(amount=20_000, end_year=10, inflation_adjusted=False),
             mc.CashFlow(amount=-50_000, start_year=11, inflation_adjusted=False)]
    R = mc.run(mc.Settings(series=_series(), start_balance=100_000, years=30, flows=flows, inflation=0.0, sims=300))
    assert R["has_contributions"] and R["has_withdrawals"]
    assert R["withdrawals"]["total"]["90"] <= 20 * 50_000 + 1e-6
    assert all("years" in f for f in R["settings"]["flows"])


@needs_data
def test_web_montecarlo_phases_and_mortality():
    from backtester import web
    body = {"weights": "SPY 60, IEF 40", "balance": 500000, "sims": 300, "horizon": "mortality", "age": 60, "sex": "joint",
            "age2": 58, "inflation": 0.025,
            "flows": [{"type": "contribution", "amount": 10000, "freq": "yearly", "inflation_adjusted": True, "start_year": 1, "end_year": 5},
                      {"type": "withdrawal", "amount": 40000, "freq": "yearly", "inflation_adjusted": False, "start_year": 6}]}
    R = web.api_montecarlo(body)
    assert R["mortality"]["sex"] == "joint" and len(R["settings"]["flows"]) == 2
    assert "fixed dollars" in R["settings"]["flows"][1]
    with pytest.raises(web.ClientError):
        web.api_montecarlo({**body, "age": None})
    S = web.api_style({"ticker": "SPY", "assets": "SPY IEF BIL"})
    assert S["weights"]["SPY"] > 0.95
