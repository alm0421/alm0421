"""Long-history (SIM) series built by scripts/fetch_data.py: monthly steps on the last NYSE session of
each month, parsers for every source (checked against synthetic samples shaped like the real files),
the international bond model, splicing, and the SIM tables used by named portfolios."""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from backtester import data, parser

ROOT = pathlib.Path(__file__).parents[1]


@pytest.fixture(scope="module")
def fd():
    spec = importlib.util.spec_from_file_location("fetch_data_sims", ROOT / "scripts" / "fetch_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=["calendar", "french_dates"])
def cal(fd, tmp_path, monkeypatch, request):
    """Run with the NYSE schedule alone, and with a Fama-French daily file supplying the sessions."""
    monkeypatch.setattr(fd, "FACTORS", tmp_path)
    if request.param == "french_dates":
        days = pd.bdate_range("1974-01-01", "2024-12-31")
        from backtester import calendar as nyse
        days = [d for d in days if nyse.is_session(d)]
        pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in days], "RF": 0.0}).to_csv(tmp_path / "ff3_daily.csv", index=False)
    return fd


def test_monthly_steps_land_on_the_last_session_of_the_month(cal):
    fd = cal
    r = pd.Series([0.01, 0.02, 0.03], index=pd.to_datetime(["1975-04-30", "1975-05-31", "1975-06-30"]))
    out = fd._monthly_steps(r)
    steps = out[out != 0]
    # May 1975 ended on a Saturday: its return belongs to Friday 1975-05-30, not Monday 1975-06-02
    assert list(steps.index.strftime("%Y-%m-%d")) == ["1975-04-30", "1975-05-30", "1975-06-30"]
    assert steps.tolist() == [0.01, 0.02, 0.03]
    assert out.index[0] == pd.Timestamp("1975-04-01") and out.index[-1] == pd.Timestamp("1975-06-30")
    assert np.isclose((1 + out).prod(), 1.01 * 1.02 * 1.03)
    # months ending on a weekend or a holiday (Good Friday 2024-03-29); dated at month start or as periods
    r2 = pd.Series([0.05, -0.02], index=pd.PeriodIndex(["2020-05", "2024-03"], freq="M"))
    s2 = fd._monthly_steps(r2)
    assert list(s2[s2 != 0].index.strftime("%Y-%m-%d")) == ["2020-05-29", "2024-03-28"]
    r3 = pd.Series([0.05], index=pd.to_datetime(["2020-05-01"]))
    assert fd._monthly_steps(r3).idxmax() == pd.Timestamp("2020-05-29")
    # every date is an NYSE session the backtester keeps (it drops other dates from SIM files)
    from backtester import calendar as nyse
    assert all(nyse.is_session(d) for d in s2.index)


def test_monthly_steps_never_run_past_today(fd, tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "FACTORS", tmp_path)
    this_month = pd.Timestamp.today().to_period("M")
    out = fd._monthly_steps(pd.Series([0.01], index=pd.PeriodIndex([this_month])))
    assert out.index.max() <= pd.Timestamp.today().normalize()


def test_monthly_level_and_foreign_calendar_series(fd, tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "FACTORS", tmp_path)
    lvl = pd.Series([100.0, 110.0, 99.0], index=pd.to_datetime(["1975-04-30", "1975-05-31", "1975-06-30"]))
    s = fd._monthly_level_steps(lvl)
    assert list(s[s != 0].round(10).items()) == [(pd.Timestamp("1975-05-30"), 0.1), (pd.Timestamp("1975-06-30"), -0.1)]
    # London fixings: a UK-only day's price is picked up by the next NYSE session; US-only days are flat
    gold = pd.Series([100.0, 101.0, 102.0, 103.0], index=pd.to_datetime(["2024-05-24", "2024-05-27", "2024-05-28", "2024-05-29"]))
    d = fd._on_sessions(gold)
    assert pd.Timestamp("2024-05-27") not in d.index            # Memorial Day: NYSE closed
    assert np.isclose(d.loc["2024-05-28"], 102 / 100 - 1)


def test_parse_lbma_json(fd):
    rows = [{"is_cms_locked": 0, "d": "1968-04-01", "v": [37.7, 15.68, None]},
            {"is_cms_locked": 0, "d": "1968-04-02", "v": [37.3, 37.3, None]},
            {"is_cms_locked": 0, "d": "1968-04-03", "v": [None, 15.6, None]}]
    rows += [{"is_cms_locked": 0, "d": str(d.date()), "v": [40 + i / 100, 20.0, 30.0]}
             for i, d in enumerate(pd.bdate_range("1970-01-01", periods=1200))]
    s = fd.parse_lbma_json(json.dumps(rows))
    assert s.index[0] == pd.Timestamp("1968-04-01") and s.iloc[0] == 37.7
    assert pd.Timestamp("1968-04-03") not in s.index           # no USD price that day
    assert len(s) == 1202
    with pytest.raises(RuntimeError):
        fd.parse_lbma_json(json.dumps(rows[:10]))


def _aqr_sheet(values):
    top = [["AQR Capital Management, LLC — Commodities for the Long Run: Index Level Data, Monthly"] + [None] * 11,
           [None] * 12, ["See Definition and Data Sources tabs for details."] + [None] * 11] + [[None] * 12] * 7
    head = [[None, "Excess return of equal-weight commodities portfolio", "Excess spot return of equal-weight commodities portfolio",
             "Interest rate adjusted carry of equal-weight commodities portfolio", "Spot return of equal-weight commodities portfolio",
             "Carry of equal-weight commodities portfolio", "Excess return of long/short commodities portfolio",
             "Excess spot return of long/short commodities portfolio", "Interest rate adjusted carry of long/short commodities portfolio",
             "Aggregate backwardation/contango", "State of backwardation/contango", "State of inflation"]]
    dates = pd.date_range("1877-02-28", periods=len(values), freq="ME")
    rows = [[d, v, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, -0.01, "Contango", "Inflation Up"] for d, v in zip(dates, values)]
    return pd.DataFrame(top + head + rows + [[None] * 12])


def test_parse_aqr_commodities(fd):
    vals = [-0.0826, -0.0429, 0.2157] + [0.001] * 200
    s = fd.parse_aqr_commodities(_aqr_sheet(vals))
    assert s.index[0] == pd.Timestamp("1877-02-28") and np.isclose(s.iloc[0], -0.0826) and len(s) == 203
    # the same numbers as percent strings
    s2 = fd.parse_aqr_commodities(_aqr_sheet([f"{v * 100:.2f}%" for v in vals]))
    assert np.allclose(s2.to_numpy(), s.to_numpy())
    with pytest.raises(RuntimeError):
        fd.parse_aqr_commodities(pd.DataFrame([["nothing here"]] * 20))


ME_SAMPLE = """This file was created by CMPT_ME_RETS_DAILY using the 202608 CRSP database.  It contains
value- and equal-weighted returns for size portfolios.  Each record contains returns for:

Negative (not used)  30%  40%  30%     5 Quintiles    10 Deciles

Missing data are indicated by -99.99 or -999.


  Value Weighted Returns -- Daily
,<= 0,Lo 30,Med 40,Hi 30,Lo 20,Qnt 2,Qnt 3,Qnt 4,Hi 20,Lo 10,Dec 2,Dec 3,Dec 4,Dec 5,Dec 6,Dec 7,Dec 8,Dec 9,Hi 10
19260701,  -99.99,    0.39,   -0.13,    0.14,    0.04,    0.23,   -0.20,   -0.07,    0.16,    0.57,   -0.13,    0.68,   -0.06,   -0.38,   -0.07,   -0.08,   -0.06,    0.08,    0.18
19260702,  -99.99,   -0.11,    0.38,    0.48,   -0.43,    0.12,    0.33,    0.44,    0.49,   -0.53,   -0.40,    0.16,    0.10,    0.29,    0.36,    0.51,    0.41,    0.37,    0.52


  Equal Weighted Returns -- Daily
,<= 0,Lo 30,Med 40,Hi 30,Lo 20,Qnt 2,Qnt 3,Qnt 4,Hi 20,Lo 10,Dec 2,Dec 3,Dec 4,Dec 5,Dec 6,Dec 7,Dec 8,Dec 9,Hi 10
19260701,  -99.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99
"""

DEV6_SAMPLE = """This file was created using the 202608 Bloomberg database. It contains
value- and equal-weighted returns for the intersections of 2 ME portfolios
and 3 BE/ME portfolios in Developed_ex_US.

Missing data are indicated by -99.99.


  Average Value Weighted Returns -- Daily
,SMALL LoBM,ME1 BM2,SMALL HiBM,BIG LoBM,ME2 BM2,BIG HiBM
19900702  ,0.12  ,-0.07  ,-0.34  ,-0.78  ,-0.68  ,-0.81
19900703  ,0.22  ,0.07  ,0.34  ,0.78  ,0.68  ,0.81
"""


def test_parse_french_csv_first_table_only(fd):
    me = fd.parse_french_csv(ME_SAMPLE.splitlines())
    assert list(me["date"]) == ["1926-07-01", "1926-07-02"]
    assert me["<= 0"].isna().all() and np.isclose(me["Med 40"].iloc[0], -0.0013)
    assert fd._col(me.set_index("date"), "MED40") == "Med 40"
    dev = fd.parse_french_csv(DEV6_SAMPLE.splitlines())
    assert list(dev.columns) == ["date", "SMALL LoBM", "ME1 BM2", "SMALL HiBM", "BIG LoBM", "ME2 BM2", "BIG HiBM"]
    assert np.isclose(dev["BIG HiBM"].iloc[1], 0.0081)


def test_parse_french_dat_named_column(fd):
    text = """This file was created using the 202608 Bloomberg/MSCI database.
Missing data are indicated by -99.99 or -999.

                    Mkt       HiBM       LoBM       HiEP       LoEP
     197501       9.73      12.05       8.10      11.00       7.70
     197502       5.10    -99.99        4.20       5.00       4.00
     197503       1.00       2.00       0.50       1.50       0.20

                    Mkt       HiBM       LoBM       HiEP       LoEP
     1975         30.00      40.00      20.00      35.00      25.00
"""
    m = fd.parse_french_dat(text)
    assert list(m.round(4)) == [0.0973, 0.051, 0.01] and m.index[0] == pd.Timestamp("1975-01-31")
    v = fd.parse_french_dat(text, column=("HIBM",))
    assert list(v.round(4)) == [0.1205, 0.02]                # the missing month is skipped
    with pytest.raises(RuntimeError):
        fd.parse_french_dat(text, column=("NOPE",))


def test_intl_bond_model_carry_hedge_and_weights(fd):
    months = pd.date_range("1970-01-01", periods=36, freq="MS")
    flat = lambda v: pd.Series(v, index=months)  # noqa: E731
    long = {"DE": flat(6.0), "FR": flat(8.0), "GB": flat(10.0), "JP": flat(7.0)}
    short = {"DE": flat(4.0), "FR": flat(6.0), "GB": flat(8.0)}         # no short rate for JP: left out
    us = flat(5.0)
    r = fd.intl_bond_model(long, short, us, weights={"DE": 0.5, "FR": 0.25, "GB": 0.25, "JP": 1.0})
    # flat yields: a par bond earns its coupon; hedging swaps the local short rate for the US one
    exp = {"DE": 6 - 4, "FR": 8 - 6, "GB": 10 - 8}
    mix = sum(w * exp[c] for c, w in (("DE", 0.5), ("FR", 0.25), ("GB", 0.25)))
    assert len(r) == 35 and np.allclose(r.to_numpy(), (mix + 5.0) / 1200, atol=2e-4)
    # fewer than three countries: no index
    with pytest.raises(RuntimeError):
        fd.intl_bond_model({"DE": long["DE"]}, {"DE": short["DE"]}, us)
    # a yield rise loses money (duration)
    up = {c: s.copy() for c, s in long.items()}
    for s in up.values():
        s.iloc[20] += 1.0
    r2 = fd.intl_bond_model(up, short, us)
    assert r2.iloc[19] < -0.03


def test_splice_chains_funds(fd, tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "PRICES", tmp_path)
    days = pd.bdate_range("2000-01-03", periods=30)
    for name, start, step in (("AAA", 10, 1.01), ("BBB", 20, 1.02)):
        idx = days[start:]
        lvl = 50 * step ** np.arange(len(idx))
        pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "close": lvl, "adj_close": lvl}).to_csv(tmp_path / f"{name}.csv", index=False)
    sim = pd.Series(0.005, index=days)
    ret = fd._splice_returns(sim, "AAA", "MISSING", "BBB")
    assert np.isclose(ret.loc[days[5]], 0.005) and np.isclose(ret.loc[days[15]], 0.01) and np.isclose(ret.loc[days[25]], 0.02)
    assert fd._splice(sim, "AAA").iloc[0] == pytest.approx(100 * 1.005)


def test_sim_tables_are_consistent():
    """Every series a named portfolio may swap in is described in data.SIMS, and every fund in a named
    portfolio (except the leveraged ones) has a long-history series."""
    named = {sim for opts in parser.SIM_FOR.values() for sim, _ in opts} - {"AGGSIM"}
    assert named <= set(data.SIMS), named - set(data.SIMS)
    held = {etf for _, _, holdings, _ in parser.MODEL_PORTFOLIOS for _, etf in holdings} - {"UPRO", "TMF"}
    assert held <= set(parser.SIM_FOR), held - set(parser.SIM_FOR)


def test_sims_tolerate_missing_series():
    # the data job adds new series after this code is merged: nothing may assume they exist yet
    have = set(data.sims())
    assert have <= set(data.available_tickers())
    for t in ("MIDSIM", "BNDXSIM", "DBCSIM"):
        assert data.is_sim(t)
