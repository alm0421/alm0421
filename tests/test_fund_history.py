"""Mutual funds whose free (Yahoo) daily history misses capital-gain distributions (backtester/fund_history.py):
quarters matched to the published total returns (data/fund_returns.json), or the unreliable early stretch cut with a
Warning and an opt-in; plus the SIMs that splice those funds in, the new asset-class series (VOOSIM, IWCSIM, VTSIM,
VPLSIM, VNQISIM) and the monthly-step detection's rounding fix."""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from backtester import data, fund_history, parser, runner
from backtester.portfolio import Portfolio

ROOT = pathlib.Path(__file__).parents[1]
HAVE = set(data.available_tickers())


def needs(*tickers):
    return pytest.mark.skipif(not set(tickers) <= HAVE, reason=f"needs {tickers}")


@pytest.fixture(scope="module")
def fd():
    spec = importlib.util.spec_from_file_location("fetch_data_fund_history", ROOT / "scripts" / "fetch_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raw(t: str) -> pd.DataFrame:
    df = pd.read_csv(data.price_path(t), parse_dates=["date"], index_col="date").sort_index()
    return df[~df.index.duplicated(keep="last")]


def _years(px: pd.Series) -> pd.Series:
    px = px.dropna()
    return px.groupby(px.index.year).last().pct_change()


def _pub_years(t: str) -> pd.Series:
    q = fund_history.published(t)
    y = (1 + q).groupby(q.index.year).prod() - 1
    whole = q.groupby(q.index.year).count() == 4
    return y[whole]


# ------------------------------------------------------------------ the reviewer's case: VFINX before 1987

@needs("VFINX")
def test_vfinx_1985_and_1986_match_the_published_returns():
    y = _years(data.load("VFINX")["adj_close"])
    assert y[1985] == pytest.approx(0.3123, abs=0.002)     # published 31.23% (Yahoo's raw history: about 22%)
    assert y[1986] == pytest.approx(0.1806, abs=0.002)     # published 18.06% (raw: about 10%)
    raw = _years(_raw("VFINX")["adj_close"])
    assert raw[1985] < 0.25 and raw[1986] < 0.12            # the understatement being fixed


@needs("VFINX", "SPYSIM")
def test_vfinx_1980_87_is_no_longer_far_behind_the_market():
    def cagr(px):
        px = px.loc["1979-12-31":"1987-12-31"]
        return (px.iloc[-1] / px.iloc[0]) ** (1 / 8) - 1
    f, m = cagr(data.load("VFINX")["adj_close"]), cagr(data.load("SPYSIM")["adj_close"])
    assert abs(f - m) < 0.01                               # was 11.8% vs 15.4%
    assert cagr(_raw("VFINX")["adj_close"]) < m - 0.03


@needs("FMAGX")
def test_fmagx_1984_quarter_repaired():
    y = _years(data.load("FMAGX")["adj_close"])
    assert y[1984] == pytest.approx(_pub_years("FMAGX")[1984], abs=0.003)
    ev = data.fund_repairs("FMAGX")
    assert "1984Q2" in set(ev["quarter"])


# ------------------------------------------------------------------ every repaired fund against its published returns

def _published_funds():
    return sorted(t for t in fund_history.doc().get("funds", {}) if t in HAVE)


@pytest.mark.parametrize("t", _published_funds())
def test_repaired_years_match_the_published_years(t):
    y = _years(data.load(t)["adj_close"])
    pub = _pub_years(t)
    first = data.load(t).index[0].year
    both = [yr for yr in pub.index if yr in y.index and yr > first and np.isfinite(y[yr])]
    assert both, t
    gaps = {yr: float(np.log1p(y[yr]) - np.log1p(pub[yr])) for yr in both}
    # a distribution the free data books on the other side of a year end than Morningstar shows as two opposite
    # gaps that cancel (the repair leaves those alone): such a year matches together with its neighbour
    bad = {yr: g for yr, g in gaps.items() if abs(g) >= 0.003
           and not any(abs(g + gaps.get(yr + d, np.nan)) < 0.003 for d in (-1, 1))}
    assert not bad, (t, bad)


# The lag test: a mutual fund's usable history must never trail its reference by more than LAG_MAX a year over any
# LAG_WINDOW-quarter window unless it carries a flag (repaired quarters or a cut). The reference is the fund's own
# published return where it is on file; the raw free history must fail the same test wherever it is flagged, so the
# test would catch the problem coming back.
LAG_WINDOW, LAG_MAX = 12, 0.0025


def _worst_lag(adj: pd.Series, pub: pd.Series) -> float:
    g = fund_history.quarter_gaps(adj, pub)["gap"]      # log(1 + published) - log(1 + series), per quarter
    if len(g) < LAG_WINDOW:
        return float(-g.sum() / max(len(g), 1) * 4)
    return float((-g.rolling(LAG_WINDOW).sum() / (LAG_WINDOW / 4)).min())


@pytest.mark.parametrize("t", _published_funds())
def test_no_fund_lags_its_published_returns_without_a_flag(t):
    pub = fund_history.published(t)
    usable = _worst_lag(data.load(t)["adj_close"], pub)
    assert usable > -LAG_MAX, (t, usable)
    raw = _worst_lag(pd.to_numeric(_raw(t)["adj_close"], errors="coerce").where(lambda s: s > 0).ffill(), pub)
    ev = data.fund_repairs(t)
    if raw < -LAG_MAX:
        assert ev is not None and len(ev), (t, raw)           # the raw history lags: the fund must be flagged


# Index funds against an independent index where published returns are not the only reference: VFINX against Ken
# French's largest-30%-of-NYSE portfolio (the S&P 500's universe; SPY tracks it within 1.3%/yr), VTSMX against his
# market return. Rolling 36 months, before the fund's own data could have fed the reference.
INDEX_PAIRS = (("VFINX", "Hi 30", 0.03), ("VTSMX", "market", 0.015))


@pytest.mark.parametrize("fund,ref,lag", INDEX_PAIRS)
def test_index_funds_track_an_independent_index(fund, ref, lag):
    if fund not in HAVE:
        pytest.skip(fund)
    if ref == "market":
        f = pd.read_csv(data.DATA / "factors" / "ff3_daily.csv", index_col=0, parse_dates=True)
        r = (f["Mkt-RF"] + f["RF"]).dropna()
    else:
        r = pd.read_csv(data.DATA / "factors" / "me_daily.csv", index_col=0, parse_dates=True)[ref].dropna()
    lvl = (1 + r).cumprod()

    def monthly(px):
        return np.log(px.groupby(px.index.to_period("M")).last()).diff().dropna()
    j = pd.concat({"f": monthly(data.load(fund)["adj_close"]), "r": monthly(lvl)}, axis=1, join="inner").dropna()
    g = (j["f"] - j["r"]).rolling(36).sum() / 3
    assert g.min() > -lag, (fund, g.idxmin(), g.min())
    jr = pd.concat({"f": monthly(_raw(fund)["adj_close"]), "r": monthly(lvl)}, axis=1, join="inner").dropna()
    if fund == "VFINX":                                       # the raw history fails it (the reviewer's finding)
        assert ((jr["f"] - jr["r"]).rolling(36).sum() / 3).min() < -lag


# ------------------------------------------------------------------ the repair itself (synthetic)

def _synthetic(miss_day="2001-06-15", miss=0.05):
    idx = pd.bdate_range("2000-01-03", "2002-12-31")
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0004, 0.008, len(idx)), index=idx)
    true = 100 * (1 + r).cumprod()
    shown = true.copy()
    shown[shown.index >= miss_day] *= 1 - miss                # the NAV drops by a distribution the data never pays
    df = pd.DataFrame({"open": shown, "high": shown, "low": shown, "close": shown, "volume": 0, "dividend": 0.0,
                       "adj_close": shown})
    pub = fund_history.quarter_returns(true)
    return df, pub, true


def test_repair_books_the_missed_distribution_on_its_day(monkeypatch):
    df, pub, true = _synthetic()
    monkeypatch.setattr(fund_history, "_market_log_returns", lambda: pd.Series(dtype=float))
    out, ev = fund_history.repair("TESTX", df, pub)
    assert list(ev["quarter"]) == ["2001Q2"] and ev.iloc[0]["placed"] == "2001-06-15"
    assert ev.iloc[0]["kind"] == "missed_distribution"
    assert out.loc["2001-06-15", "dividend"] == pytest.approx(true.loc["2001-06-15"] - df.loc["2001-06-15", "close"],
                                                               rel=1e-3)
    q = fund_history.quarter_returns(out["adj_close"])
    assert np.allclose(q.to_numpy(), pub.reindex(q.index).to_numpy(), atol=1e-9)
    assert (out["close"] == df["close"]).all()               # prices as traded; only the total return is fixed


def test_repair_leaves_matching_quarters_and_small_gaps_alone(monkeypatch):
    df, pub, true = _synthetic(miss=0.001)                    # 0.1%: within the tolerance
    monkeypatch.setattr(fund_history, "_market_log_returns", lambda: pd.Series(dtype=float))
    out, ev = fund_history.repair("TESTX", df, pub)
    assert ev.empty and out is df


def test_process_only_touches_mutual_funds():
    df, pub, _ = _synthetic()
    assert fund_history.is_mutual_fund("VFINX") and not fund_history.is_mutual_fund("SPY")
    assert not fund_history.is_mutual_fund("VFINXX") and not fund_history.is_mutual_fund("QQQX1")
    out, ev, cut = fund_history.process("SPY", df)
    assert out is df and ev.empty and cut is None


def test_detection_finds_a_yearly_missed_distribution_and_ignores_random_drops():
    idx = pd.bdate_range("1985-01-02", "1999-12-31")
    rng = np.random.default_rng(7)
    mkt = pd.Series(rng.normal(0.0004, 0.009, len(idx)), index=idx)
    fac = pd.DataFrame({"SPYSIM": mkt, "IEFSIM": rng.normal(0.0002, 0.003, len(idx))}, index=idx)
    fund = mkt * 1.0 + rng.normal(0, 0.002, len(idx))
    base = 100 * (1 + fund).cumprod()
    missed = base.copy()
    for y in range(1985, 1992):                               # a December distribution the data misses, 7 years
        d = idx[(idx >= f"{y}-12-10")][0]
        missed[missed.index >= d] *= 0.96
    f = fund_history.detect_unreliable(missed, fac)
    assert f is not None and f["days"] >= 5 and pd.Timestamp(f["until"]).year == 1991
    assert f["usable_from"] > f["until"]
    random = base.copy()
    for d in rng.choice(len(idx) - 50, 6, replace=False):     # a fund's own bad days, at random dates
        random.iloc[d:] *= 0.96
    assert fund_history.detect_unreliable(random, fac) is None


# ------------------------------------------------------------------ cut funds: Warning and opt-in

def _cut_fund():
    for t in sorted(HAVE):
        if fund_history.is_mutual_fund(t) and fund_history.published(t) is None:
            f = data.fund_unreliable(t)
            if f:
                return t, f
    pytest.skip("no fund with an unreliable stretch in the data")


def test_cut_fund_starts_at_its_first_reliable_date_unless_opted_in():
    t, f = _cut_fund()
    assert data.load(t).index[0] == pd.Timestamp(f["usable_from"])
    token = data.RAW_FUND_HISTORY.set(True)
    try:
        assert data.load(t).index[0] < pd.Timestamp(f["usable_from"])
    finally:
        data.RAW_FUND_HISTORY.reset(token)
    raw_first = _raw(t).index[0]
    lcut = data.LOAD_CUTOFF.set(pd.Timestamp(f["usable_from"]) + pd.Timedelta(days=40))
    try:
        cut = data.load(t)
    finally:
        data.LOAD_CUTOFF.reset(lcut)
    assert cut.index[0] == pd.Timestamp(f["usable_from"]) and raw_first < cut.index[0]


def test_cut_fund_warning_and_the_raw_history_phrase():
    t, f = _cut_fund()
    y0 = pd.Timestamp(f["first"]).year
    p = parser.parse(f"hold 100% {t} from {y0} to {y0 + 12}")
    res = runner.run(p)
    assert any(n.startswith("Warning: fund history cut") and t in n and "using raw fund history" in n for n in p.notes)
    assert res.equity.index[0] >= pd.Timestamp(f["usable_from"]) - pd.Timedelta(days=5)
    q = parser.parse(f"hold 100% {t} from {y0} to {y0 + 12}, using raw fund history")
    assert isinstance(q, Portfolio) and q.raw_fund_history
    res2 = runner.run(q)
    assert res2.equity.index[0] < pd.Timestamp(f["usable_from"])
    assert any(n.startswith("Warning: using the raw fund history") and f"{t} before" in n for n in q.notes)


@needs("VFINX")
def test_repaired_fund_note():
    p = parser.parse("hold 100% VFINX from 1980 to 1990")
    runner.run(p)
    n = next(n for n in p.notes if n.startswith("Fund history repaired"))
    assert "VFINX: 7 quarter(s) in 1980-1986" in n and "data/fund_returns.json" in n


def test_raw_fund_history_json_and_command_line(tmp_path):
    from backtester import __main__ as cli
    t, f = _cut_fund()
    text = f"hold 100% {t} from 1990 to 2000"
    a = cli.build_run_parser().parse_args([text, "--raw-fund-history"])
    assert cli.make_spec(a.text, a).raw_fund_history
    a = cli.build_run_parser().parse_args([text])
    assert not cli.make_spec(a.text, a).raw_fund_history
    p = parser.parse(f"{text}, with raw fund history")
    path = tmp_path / "spec.json"
    path.write_text(p.to_json())
    assert json.loads(path.read_text())["raw_fund_history"] is True
    assert runner.load(path).raw_fund_history


# ------------------------------------------------------------------ the SIMs that splice repaired funds in

@needs("VTISIM", "VTSMX")
def test_vtisim_matches_vtsmx_published_years():
    y = _years(data.load("VTISIM")["adj_close"])
    pub = _pub_years("VTSMX")
    for yr in (1993, 1994, 1995, 1996):
        assert y[yr] == pytest.approx(pub[yr], abs=0.002), yr


def test_sim_real_returns_use_the_repaired_fund(fd):
    if "VFINX" not in HAVE:
        pytest.skip("VFINX")
    r = fd._real_returns("VFINX")
    y = (1 + r).groupby(r.index.year).prod() - 1
    assert y[1985] == pytest.approx(0.3123, abs=0.002)


# ------------------------------------------------------------------ new asset classes

@pytest.mark.parametrize("phrase,want", [
    ("US large cap", "VOOSIM"), ("large caps", "VOOSIM"), ("US micro cap", "IWCSIM"), ("micro caps", "IWCSIM"),
    ("global stocks", "VTSIM"), ("world stocks", "VTSIM"), ("ACWI", "VTSIM"), ("total world stock market", "VTSIM"),
    ("pacific", "VPLSIM"), ("developed pacific stocks", "VPLSIM"), ("asia pacific", "VPLSIM"),
    ("international REITs", "VNQISIM"), ("ex-US real estate", "VNQISIM"), ("REITs", "VNQSIM"),
    ("global ex-US stocks", "VXUSSIM"),
])
def test_new_asset_class_words(phrase, want):
    got = parser._asset_class_ticker(phrase)
    if want not in HAVE:
        pytest.skip(want)
    assert got and got[0] == want, (phrase, got)


@needs("VTSIM", "VNQISIM", "IWCSIM")
def test_new_asset_classes_in_a_sentence():
    p = parser.parse("hold 50% global stocks, 25% international REITs, 25% micro caps since 2008")
    # the long series where the start is before the fund (VT 2008-06, VNQI 2010), the fund itself where not (IWC 2005)
    assert [k["asset"] for k in p.tree["children"]] == ["VTSIM", "VNQISIM", "IWC"]
    p = parser.parse("hold 50% global stocks, 50% micro caps since 1980")
    assert [k["asset"] for k in p.tree["children"]] == ["VTSIM", "IWCSIM"]


@needs("VOOSIM", "SPY")
def test_voosim_is_large_caps_not_the_whole_market():
    assert "largest 30% of NYSE" in data.SIMS["VOOSIM"] and "not only large caps" in data.SIMS["SPYSIM"]
    v, s = data.load("VOOSIM")["adj_close"], data.load("SPY")["adj_close"]
    j = pd.concat({"v": v, "s": s}, axis=1, join="inner").dropna()
    m = j.groupby(j.index.to_period("M")).last().pct_change().dropna()
    assert (m["v"] - m["s"]).std() * 12 ** 0.5 < 0.02      # VOOSIM is VFINX, then VOO, on SPY's whole life
    assert parser.SIM_FOR["VOO"][0][0] == "VOOSIM"


@pytest.mark.parametrize("t,first", [("VOOSIM", "1926-07"), ("IWCSIM", "1926-07"), ("VTSIM", "1975-01"),
                                     ("VPLSIM", "1975-01"), ("VNQISIM", "2006-12")])
def test_new_sims_exist_without_holes(t, first):
    if t not in HAVE:
        pytest.skip(t)
    df = data.load(t)
    assert str(df.index[0])[:7] == first and df.index[-1] > pd.Timestamp("2026-01-01")
    assert data.data_gaps(t) == ()
    assert t in data.SIMS and data.sim_about(t).startswith(data.SIMS[t])
    drag = data.sim_drags().get(t)
    assert (drag is None) if t == "VNQISIM" else drag["drag"] >= drag["expense_ratio"] - 1e-12


@needs("VTSIM")
def test_vtsim_and_vplsim_are_monthly_steps_before_their_daily_data():
    assert data.stepped_ranges("VTSIM") == ((pd.Timestamp("1975-01-01"), pd.Timestamp("1990-06-30")),)
    (a, b), = data.stepped_ranges("VPLSIM")
    assert a == pd.Timestamp("1975-01-01") and b < pd.Timestamp("1990-07-01")


def test_rebalanced_blend_drifts_and_resets(fd):
    idx = pd.bdate_range("2000-01-03", periods=6)
    a = pd.Series([0.10, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)
    b = pd.Series([0.0, 0.0, 0.0, 0.10, 0.0, 0.0], index=idx)
    w = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]}, index=[idx[0], idx[3]])
    out = fd.rebalanced_blend({"a": a, "b": b}, w)
    assert out.iloc[0] == pytest.approx(0.05)
    assert out.iloc[3] == pytest.approx(0.05)                 # reset to 50/50 on day 4, not the drifted 52.4/47.6
    w1 = pd.DataFrame({"a": [0.5], "b": [0.5]}, index=[idx[0]])
    out = fd.rebalanced_blend({"a": a, "b": b}, w1)
    assert out.iloc[3] == pytest.approx(0.10 * 0.5 / 1.05)     # drifted: b holds 0.5 of 1.05
    late = pd.Series([np.nan, np.nan, 0.02, 0.02, 0.02, 0.02], index=idx)
    out = fd.rebalanced_blend({"a": a, "c": late}, pd.DataFrame({"a": [0.5], "c": [0.5]}, index=[idx[0]]))
    assert out.iloc[1] == 0.0 and out.iloc[2] == pytest.approx(0.01)   # c enters on its first day at its weight


def test_world_bank_parser_and_weights(fd):
    doc = [{"page": 1}, [
        {"country": {"id": "US"}, "countryiso3code": "USA", "date": "1980", "value": 60.0},
        {"country": {"id": "1W"}, "countryiso3code": "WLD", "date": "1980", "value": 100.0},
        {"country": {"id": "XD"}, "countryiso3code": "", "date": "1980", "value": 98.0},
        {"country": {"id": "XO"}, "countryiso3code": "LMY", "date": "1980", "value": None}]]
    t = fd.parse_world_bank(doc)
    assert t.loc[1980, "USA"] == 60.0 and t.loc[1980, "HIC"] == 98.0 and "LMY" not in t
    w = fd.year_start_weights(pd.DataFrame({"us": [60.0], "dev": [38.0]}, index=[1980]))
    assert w.index[0] == pd.Timestamp("1981-01-01") and w.iloc[0]["us"] == pytest.approx(60 / 98)
    caps = pd.read_csv(ROOT / "data" / "factors" / "world_market_caps.csv", index_col="year")
    assert caps.loc[1989, "USA"] / caps.loc[1989, "WLD"] == pytest.approx(0.293, abs=0.005)   # Japan's bubble year


def test_fund_performance_payload_parser_and_fetch(fd, tmp_path):
    payload = {"quoteSummary": {"result": [{"fundPerformance": {"pastQuarterlyReturns": {"returns": [
        {"year": 1985, "q1": 0.0917, "q2": {"raw": 0.0718, "fmt": "7.18%"}, "q3": -0.0415, "q4": 0.1686},
        {"year": {"raw": 1986, "fmt": "1986"}, "q1": "14.02%", "q2": "5.84%", "q3": "-6.99%", "q4": "5.47%"},
        {"year": 2026, "q1": 0.01, "q2": None, "q3": None, "q4": None}]}}}]}}
    q = fd.parse_fund_performance(payload)
    assert q == {"1985": [9.17, 7.18, -4.15, 16.86], "1986": [14.02, 5.84, -6.99, 5.47]}
    path = tmp_path / "fund_returns.json"
    path.write_text(json.dumps({"units": "percent", "funds": {"VFINX": {"source": "hand", "quarterly": {"1985": [1, 2, 3, 4]}}}}))
    doc = fd.fetch_fund_returns(["VFINX", "SPY", "TESTX"], path=path, today="2026-09-27", get=lambda t: payload)
    assert doc["funds"]["VFINX"]["source"] == "hand"                          # never replaced
    assert doc["funds"]["TESTX"]["quarterly"]["1986"] == [14.02, 5.84, -6.99, 5.47]
    assert "SPY" not in doc["funds"] and "quoteSummary" in doc["funds"]["TESTX"]["source"]
    assert json.loads(path.read_text()) == doc


# ------------------------------------------------------------------ monthly-step detection: rounding is no move

def test_rounded_prices_do_not_break_a_monthly_stepped_stretch():
    assert data._price_quantum(pd.Series([143.662244, 143.66303, 1.5])) == pytest.approx(1e-6)
    assert data._price_quantum(pd.Series([1 / 3, 2 / 3])) == 0.0
    idx = pd.bdate_range("1975-01-01", "1976-12-31")
    level, px = 143.0, []
    rng = np.random.default_rng(3)
    for i, d in enumerate(idx):
        days = (d - idx[i - 1]).days if i else 1
        level *= (1 - 0.0000055) ** days                         # a small fee accrual per calendar day (apply_drag)
        if i + 1 == len(idx) or idx[i + 1].month != d.month:
            level *= 1 + rng.choice([-1, 1]) * rng.uniform(0.002, 0.02)   # the month's return on its last session
        px.append(round(level, 6))                               # a price file with 6 decimals
    rng_ = data.stepped_ranges_of(pd.Series(px, index=idx))
    assert rng_ == [(pd.Timestamp("1975-01-01"), pd.Timestamp("1976-12-31"))]


@needs("TIPSIM")
def test_tipsim_model_is_one_stepped_stretch():
    (a, b), = data.stepped_ranges("TIPSIM")
    assert a == pd.Timestamp("1972-01-01") and b >= pd.Timestamp("2000-05-31")
