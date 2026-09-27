"""SIM fee/cost haircut (scripts/fetch_data.py): each model segment is cut by a constant annual drag before the
real fund takes over - max(expense ratio, min(cap, the model's CAGR gap over the fund on their overlap)) - and
the fund segment is never touched. Synthetic series only; plus the data.py side (descriptions, backtest note)."""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from backtester import data

ROOT = pathlib.Path(__file__).parents[1]


@pytest.fixture(scope="module")
def fd():
    spec = importlib.util.spec_from_file_location("fetch_data_drag", ROOT / "scripts" / "fetch_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _const(annual: float, start="2000-01-03", end="2009-12-31") -> pd.Series:
    """Daily returns on business days compounding to `annual` a year (by calendar time)."""
    idx = pd.bdate_range(start, end)
    days = np.array(pd.Series(idx, index=idx).diff().dt.days.fillna(1), dtype=float)
    return pd.Series((1 + annual) ** (days / 365.25) - 1, index=idx)


def test_overlap_calibration_closes_the_cagr_gap(fd):
    model = _const(0.10, "1990-01-01", "2009-12-31")
    fund = _const(0.08, "2000-01-03", "2009-12-31")
    cal = fd.calibrate_drag(model, fund, expense_ratio=0.002)
    assert cal["gap"] == pytest.approx(1.10 / 1.08 - 1, abs=2e-4)
    assert cal["drag"] == pytest.approx(cal["gap"]) and cal["basis"] == "overlap gap"
    assert cal["months"] == 118 and cal["overlap"] == "2000-02..2009-11"   # full months only
    net = fd.apply_drag(model, cal["drag"])
    cm, ce, *_ = fd.overlap_cagrs(net, fund)
    assert cm == pytest.approx(ce, abs=2e-4)          # the drag makes the model earn what the fund earned


def test_drag_is_floored_at_the_expense_ratio_and_capped(fd):
    fund = _const(0.08)
    worse = _const(0.07, "1995-01-02")
    c = fd.calibrate_drag(worse, fund, expense_ratio=0.0035)
    assert c["gap"] == 0.0 and c["drag"] == 0.0035 and "expense ratio" in c["basis"]
    close = _const(0.081, "1995-01-02")               # beat the fund by less than the expense ratio
    assert fd.calibrate_drag(close, fund, 0.005)["drag"] == 0.005
    wild = _const(0.20, "1995-01-02")
    c = fd.calibrate_drag(wild, fund, 0.001)
    assert c["gap"] > 0.10 and c["drag"] == fd.SIM_DRAG_CAP == 0.03 and "capped" in c["basis"]
    # no overlap at all (or under a year): the expense ratio alone
    early = _const(0.10, "1980-01-01", "1989-12-29")
    c = fd.calibrate_drag(early, fund, 0.0017)
    assert c["drag"] == 0.0017 and c["gap"] is None and c["months"] == 0


def test_apply_drag_by_calendar_time_and_only_before_the_date(fd):
    r = _const(0.0, "2000-01-03", "2001-12-31")
    net = fd.apply_drag(r, 0.02)
    # a flat series loses 2% a year: 1 / 1.02 per 365.25 days (the first row counts 1/252 of a year)
    span = (r.index[-1] - r.index[0]).days / 365.25 + 1 / 252
    assert (1 + net).prod() == pytest.approx(1.02 ** -span, rel=1e-12)
    # monthly steps (zero on most sessions): the same drag per year as a daily series
    m = r.copy()
    m[:] = 0.0
    assert (1 + fd.apply_drag(m, 0.02)).prod() == pytest.approx((1 + net).prod(), rel=1e-12)
    cut = pd.Timestamp("2001-01-02")
    part = fd.apply_drag(_const(0.05, "2000-01-03", "2001-12-31"), 0.02, before=cut)
    orig = _const(0.05, "2000-01-03", "2001-12-31")
    assert (part[part.index >= cut] == orig[orig.index >= cut]).all()
    assert (part[part.index < cut] < orig[orig.index < cut]).all()
    assert fd.apply_drag(orig, 0.0).equals(orig.dropna().sort_index())


def test_haircut_applies_only_before_the_splice_date(fd, monkeypatch):
    model = _const(0.10, "1990-01-01", "2009-12-31")
    base = _const(0.08, "2000-01-03", "2009-12-31")
    fund = base + 0.0001 * np.sin(np.arange(len(base)))
    monkeypatch.setattr(fd, "_real_returns", lambda t: fund.copy() if t == "FUNDX" else pd.Series(dtype=float))
    monkeypatch.setitem(fd.SIM_EXPENSE_RATIOS, "FUNDX", 0.001)
    fd.SIM_DRAG.pop("TESTSIM", None)
    net = fd.haircut_model("TESTSIM", model, ("NOPE", "FUNDX"))    # the first fund with data takes over
    d = fd.SIM_DRAG["TESTSIM"]
    assert d["fund"] == "FUNDX" and d["model_until"] == "2000-01-03" and d["expense_ratio"] == 0.001
    assert 0.015 < d["drag"] < 0.022 and d["basis"] == "overlap gap"
    spliced = fd._splice_returns(net, "NOPE", "FUNDX")
    after = spliced[spliced.index >= fund.index[0]]
    assert np.array_equal(after.to_numpy(), fund.to_numpy())       # the fund itself, untouched
    before = spliced[spliced.index < fund.index[0]]
    g = (1 + before) / (1 + model[before.index])
    days = np.array(pd.Series(before.index, index=before.index).diff().dt.days.fillna(365.25 / 252), dtype=float)
    assert np.allclose(g.to_numpy(), (1 + d["drag"]) ** (-days / 365.25), rtol=1e-12, atol=0)
    assert any(n.startswith("drag TESTSIM:") and "FUNDX" in n for n in fd.SIM_NOTES)
    # no fund data at all: the ETF's expense ratio (the last fund named), no calibration
    fd.haircut_model("TEST2SIM", model, ("NOPE", "FUNDX@2050-01-01"))
    assert fd.SIM_DRAG["TEST2SIM"]["drag"] == 0.001 and fd.SIM_DRAG["TEST2SIM"]["model_until"] is None


def test_since_suffix_calibrates_from_the_index_fund_date(fd, monkeypatch):
    model = _const(0.10, "1980-01-01", "2009-12-31")
    fund = pd.concat([_const(0.02, "1985-01-01", "1994-12-30"), _const(0.09, "1995-01-02", "2009-12-31")])
    monkeypatch.setattr(fd, "_real_returns", lambda t: fund.copy() if t == "FUNDY" else pd.Series(dtype=float))
    fd.haircut_model("TEST3SIM", model, ("FUNDY@1995-01-01",))
    d = fd.SIM_DRAG["TEST3SIM"]
    assert d["model_until"] == "1995-01-02" and d["gap"] == pytest.approx(1.10 / 1.09 - 1, abs=3e-4)


def test_expense_table_covers_every_fund_a_model_hands_over_to(fd):
    src = (ROOT / "scripts" / "fetch_data.py").read_text()
    for t in ("SPY", "BIL", "VTSMX", "VIVAX", "VIGRX", "VISVX", "VISGX", "NAESX", "VOE", "VOT", "MDY", "EFA", "VGK",
              "SCZ", "AVDV", "EFV", "LQD", "VBMFX", "PFORX", "EEM", "VEIEX", "VGTSX", "VGSIX", "GLD", "DBC", "PCRIX",
              "VWESX", "TLT", "IEF", "SHY", "IEI", "EWJ", "EWH"):
        assert 0 < fd.SIM_EXPENSE_RATIOS[t] < 0.01, t
    # the real-fund-only series are built without a drag
    for t in ("TIPSIM", "HYGSIM", "MUBSIM", "EMBSIM"):
        i = src.index(f'build("{t}"')
        assert "model=False" in src[i:i + 400], t


def test_write_merges_old_entries(fd, tmp_path, monkeypatch):
    p = tmp_path / "sims_drag.json"
    p.write_text(json.dumps({"cap": 0.03, "series": {"OLDSIM": {"drag": 0.01}, "TESTSIM": {"drag": 0.5}}}))
    monkeypatch.setattr(fd, "SIM_DRAG", {"TESTSIM": {"drag": 0.02}})
    fd.write_sim_drag(p)
    j = json.loads(p.read_text())
    assert j["cap"] == 0.03 and j["series"] == {"OLDSIM": {"drag": 0.01}, "TESTSIM": {"drag": 0.02}}


def test_descriptions_and_backtest_note(tmp_path, monkeypatch):
    (tmp_path / "sims_drag.json").write_text(json.dumps({"cap": 0.03, "series": {
        "VOTSIM": {"drag": 0.0164, "expense_ratio": 0.0007, "fund": "VOT", "gap": 0.0164, "overlap": "2006-09..2026-07",
                   "months": 239, "basis": "overlap gap", "model_until": "2006-08-28"},
        "BILSIM": {"drag": 0.001356, "expense_ratio": 0.001356, "fund": "BIL", "gap": 0.0012, "overlap": "2007-06..2026-07",
                   "months": 230, "basis": "expense ratio", "model_until": "2007-05-31"}}}))
    monkeypatch.setattr(data, "DATA", tmp_path)
    a = data.sim_about("VOTSIM")
    assert a.startswith(data.SIMS["VOTSIM"]) and "net of an estimated 1.64%/yr fee/cost drag" in a and "VOT by 1.64%/yr" in a
    assert "BIL expense ratio 0.14%" in data.sim_about("BILSIM")
    assert data.sim_about("TIPSIM") == data.SIMS["TIPSIM"]            # no model: nothing added
    n = data.sim_drag_note(["VOTSIM", "BILSIM", "SPY"], "1990-01-01", "2020-12-31")
    assert "model periods are net of an estimated fee/cost drag" in n
    assert "VOTSIM 1.64%/yr until 2006-08-28" in n and "BILSIM 0.14%/yr until 2007-05-31" in n
    # a backtest that starts after the hand-over holds the real fund only: no note
    assert data.sim_drag_note(["VOTSIM"], "2010-01-01", "2020-12-31") is None
    monkeypatch.setattr(data, "DATA", tmp_path / "missing")
    assert data.sim_drag_note(["VOTSIM"], "1990-01-01") is None and data.sim_about("VOTSIM") == data.SIMS["VOTSIM"]


def test_runner_adds_the_note_for_held_sims(tmp_path, monkeypatch):
    from backtester import parser, runner
    if not {"SPYSIM", "TLTSIM"} <= set(data.available_tickers()):
        pytest.skip("no SIM data")
    (tmp_path / "sims_drag.json").write_text(json.dumps({"cap": 0.03, "series": {
        "SPYSIM": {"drag": 0.0015, "expense_ratio": 0.000945, "fund": "SPY", "gap": 0.0015, "overlap": "x",
                   "months": 400, "basis": "overlap gap", "model_until": "1993-02-01"}}}))
    real = data.DATA
    monkeypatch.setattr(data, "DATA", tmp_path)
    monkeypatch.setattr(data, "PRICES", real / "prices")
    spec = parser.parse("hold 60% SPYSIM and 40% TLTSIM, rebalance yearly, from 1985 to 1995")
    runner.run(spec)
    assert any("SPYSIM 0.15%/yr until 1993-02-01" in n for n in spec.notes)
