"""Portfolio Visualizer review, round 10: longer asset-class histories (a TIPS model, a high-yield factor model, unhedged
international bonds), Shiller's CAPE in the rule language and a CAPE-based allocation, the VTSMX distribution repair,
the VBSIM validation window, and the fund research layer (metadata, screener, comparison). Synthetic data only,
except where a test says it reads the repository's price files."""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from backtester import data, expr, funds, parser, web

ROOT = pathlib.Path(__file__).parents[1]


@pytest.fixture(scope="module")
def fd():
    spec = importlib.util.spec_from_file_location("fetch_data_r10", ROOT / "scripts" / "fetch_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ Shiller data and CAPE

def _shiller_raw(months=240, start=1990):
    """A small 'Data' sheet laid out like ie_data.xls: disclaimer rows, a multi-row header, float dates
    (October written 1990.1), 'NA' before ten years of earnings, thousands separators in a total-return column."""
    head = [["Stock Market Data Used in \"Irrational Exuberance\"", None, None, None, None, None, None, None, None],
            [None, "S&P", None, None, "Consumer", None, "Long", None, "Cyclically"],
            [None, "Comp.", "Dividend", "Earnings", "Price", "Date", "Interest", "Real", "P/E10 or"],
            ["Date", "P", "D", "E", "CPI", "Fraction", "Rate GS10", "Price", "CAPE", None, "TR CAPE"]]
    rows = []
    for k in range(months):
        y, m = start + k // 12, k % 12 + 1
        date = float(f"{y}.{m:02d}") if m != 10 else y + 0.1
        cape = "NA" if k < 12 else round(10 + k / 10, 2)
        rows.append([date, 100 + k, 2.0, 5.0, 100 + k / 10, y + (m - 0.5) / 12, 5.0, "1,234.5", cape, None, cape])
    tail = [[None, "Sept price is Sept 1st close", None, None, None, None, None, None, None]]
    w = max(len(r) for r in head + rows)
    return pd.DataFrame([r + [None] * (w - len(r)) for r in head + rows + tail])


def test_parse_shiller_layout(fd):
    df = fd.parse_shiller(_shiller_raw())
    assert list(df.columns) == ["month", "price", "dividend", "earnings", "cpi", "gs10", "cape", "tr_cape"]
    assert df["month"].iloc[0] == "1990-01" and "1990-10" in set(df["month"]) and len(df) == 240
    assert df["cape"].iloc[:12].isna().all() and df.loc[df.month == "1991-01", "cape"].iloc[0] == pytest.approx(11.2)
    assert fd._shiller_month(1871.1) == "1871-10" and fd._shiller_month("1871.01") == "1871-01"
    assert fd._shiller_month("Sept price") is None
    bad = _shiller_raw()
    bad.iat[3, 8] = "Ratio"                              # no CAPE column
    with pytest.raises(RuntimeError, match="cape"):
        fd.parse_shiller(bad)


@pytest.fixture
def shiller_file(fd, tmp_path, monkeypatch):
    df = fd.parse_shiller(_shiller_raw(months=12 * 40, start=1980))
    p = tmp_path / "shiller.csv"
    df.to_csv(p, index=False)
    monkeypatch.setattr(data, "SHILLER_FILE", p)
    return df


def _bars(start="2000-01-03", end="2012-12-31"):
    idx = pd.bdate_range(start, end)
    c = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6, "adj_close": c})


def test_cape_is_lagged_and_point_in_time(shiller_file):
    known = data.shiller_known("cape")
    # month M is known from the 1st of month M + 1 + CAPE_LAG_MONTHS
    jan = shiller_file.set_index("month").loc["2005-01", "cape"]
    d = pd.Period("2005-01", "M") + 1 + data.CAPE_LAG_MONTHS
    assert known.loc[d.to_timestamp()] == pytest.approx(jan)
    df = _bars()
    ns = expr.Namespace(df, ticker="SPY")
    v = expr.evaluate_value("cape()", ns)
    day = d.to_timestamp()
    first_session = v.index[v.index >= day][0]
    assert v.loc[first_session] == pytest.approx(jan)
    assert v.loc[v.index[v.index < day][-1]] == pytest.approx(shiller_file.set_index("month").loc["2004-12", "cape"])
    assert expr.evaluate_value("earnings_yield()", ns).loc[first_session] == pytest.approx(1 / jan)
    # truncating the bars changes nothing before the cut
    cut = pd.Timestamp("2008-06-30")
    v2 = expr.evaluate_value("cape()", expr.Namespace(df[df.index <= cut], ticker="SPY"))
    assert v2.equals(v[v.index <= cut])
    assert expr.open_safe("cape() < 25 and cape_pct(10) > 0.5") and not expr.open_safe("treasury_10y() > 0.03")


def test_cape_percentile(shiller_file):
    p = data.cape_percentile(0)
    # the synthetic CAPE rises every month: each new value is the highest so far
    assert (p.dropna() == 1.0).all()
    k = data.cape_percentile(5)
    assert (k.dropna() == 1.0).all() and k.dropna().index[0] >= p.dropna().index[0]


def test_cape_phrases_and_model(shiller_file):
    s = parser.parse("buy SPY when the Shiller CAPE is below 25, sell when CAPE is above 30")
    assert "cape() < 25" in s.entry and "cape() > 30" in s.exit_when
    assert any("CAPE" in n for n in s.notes)
    pf = parser.parse("hold SPY when the CAPE percentile is below 70% otherwise hold IEF")
    assert pf.tree["if"] == "(cape_pct(0) < 0.7)"
    pf = parser.parse("hold SPY if the earnings yield is above the 10 year treasury yield otherwise hold IEF")
    assert pf.tree["if"] == "(earnings_yield() > treasury_10y())"
    pf = parser.parse("CAPE-based allocation between SPY and IEF since 2003")
    t = pf.tree
    assert t["if"] == "cape_pct(0) <= 0.3333" and t["then"]["w"] == [0.8, 0.2]
    assert t["else"]["if"] == "cape_pct(0) <= 0.6667" and t["else"]["then"]["w"] == [0.6, 0.4]
    assert t["else"]["else"]["w"] == [0.4, 0.6] and pf.rebalance == "monthly"
    assert any(n.startswith("CAPE-based allocation: SPY 80% / IEF 20%") for n in pf.notes)


# ------------------------------------------------------------------ TIPS model

def _monthly_ts(start, n, values):
    idx = pd.date_range(start, periods=n, freq="MS")
    return pd.Series(values, index=idx)


def test_tips_model_real_carry_plus_lagged_inflation(fd):
    n = 12 * 40
    gs10 = _monthly_ts("1960-01-01", n, 6.0)
    cpi = _monthly_ts("1960-01-01", n, 100 * 1.003 ** np.arange(n))          # 0.3% a month
    real = _monthly_ts("1982-01-01", 12 * 18, 2.5)
    m, info = fd.tips_model(gs10, cpi, real, maturity=8, start="1972-01")
    assert info["first_real_month"] == "1981-12"                  # the Cleveland value dated m = end of month m-1
    # a flat real yield: each month earns the real coupon for a month plus the lagged CPI change
    after = m[m.index >= pd.Period("1983-01", "M")]
    assert np.allclose(after, (1 + 0.025 / 12) * 1.003 - 1, atol=1e-6)
    # before 1982: GS10 minus trailing inflation (flat too: 6% - 3.66%)
    early = m[(m.index >= pd.Period("1975-01", "M")) & (m.index < pd.Period("1981-06", "M"))]
    infl = (1.003 ** 12 - 1) * 100
    assert np.allclose(early, (1 + (6 - infl) / 1200) * 1.003 - 1, atol=1e-6)
    assert m.index[0] == pd.Period("1972-01", "M")
    # a real-yield rise of 1 point loses about the bond's duration
    real2 = real.copy()
    real2.iloc[100:] = 3.5
    m2, _ = fd.tips_model(gs10, cpi, real2)
    jump = m2.loc[pd.Period(real2.index[100], "M") - 1]
    assert -0.08 < jump < -0.05


def test_trailing_inflation_uses_only_past_months(fd):
    cpi = _monthly_ts("2000-01-01", 48, 100 * 1.01 ** np.arange(48))
    ti = fd.trailing_inflation(cpi, 12)
    assert ti.loc[pd.Period("2002-01", "M")] == pytest.approx((1.01 ** 12 - 1) * 100)
    spike = cpi.copy()
    spike.iloc[30:] *= 1.5
    ti2 = fd.trailing_inflation(spike, 12)
    per = pd.Period(spike.index[30], "M")
    assert ti2.loc[per] == pytest.approx(ti.loc[per])              # the month's own CPI is not used yet
    assert ti2.loc[per + 1] > ti.loc[per + 1] + 10


# ------------------------------------------------------------------ unhedged international bonds

def test_unhedged_bond_model_passes_currency_moves_through(fd):
    idx = pd.date_range("1990-01-01", periods=60, freq="MS")
    long = {c: pd.Series(5.0, index=idx) for c in ("JP", "DE", "GB")}
    fx = {c: pd.Series(1.0, index=idx.to_period("M")) for c in long}
    fx["JP"] = pd.Series(1.01 ** np.arange(60), index=idx.to_period("M"))   # the yen gains 1% a month
    r = fd.unhedged_bond_model(long, fx, weights={"JP": 0.5, "DE": 0.25, "GB": 0.25})
    carry = 0.05 / 12
    assert np.allclose(r, 0.5 * ((1 + carry) * 1.01 - 1) + 0.5 * carry, atol=1e-9)
    # under three countries in any month: no value at all
    with pytest.raises(RuntimeError):
        fd.unhedged_bond_model({"JP": long["JP"]}, fx)


def test_fx_quotes_and_euro_conversion(fd, monkeypatch):
    months = pd.date_range("1997-01-01", "2000-12-01", freq="MS")
    dem = pd.Series(1.8, index=months[months < "1999-01-01"])            # DEM per USD
    eur = pd.Series(1.1, index=months[months >= "1999-01-01"])           # USD per EUR
    gbp_daily = pd.Series([1.60, 1.62], index=pd.to_datetime(["1998-03-02", "1998-03-31"]))
    series = {"EXGEUS": dem, "EXUSEU": eur, "DEXUSUK": gbp_daily, "EXUSUK": pd.Series(1.5, index=months)}

    def fake(sid):
        if sid not in series:
            raise FileNotFoundError(sid)
        return series[sid]
    monkeypatch.setattr(fd, "_fred", fake)
    fx = fd.fx_monthly_all()
    de = fx["DE"]
    assert de.loc[pd.Period("1998-12", "M")] == pytest.approx(1 / 1.8)
    assert de.loc[pd.Period("1999-01", "M")] == pytest.approx(1.1 / fd.EURO_RATES["DE"])
    gb = fx["GB"]
    assert gb.loc[pd.Period("1998-03", "M")] == pytest.approx(1.62)    # the month-end daily quote wins
    assert gb.loc[pd.Period("1998-04", "M")] == pytest.approx(1.5)     # monthly average where no daily


# ------------------------------------------------------------------ VTSMX distribution repair, validation window

def test_repair_missed_distributions(fd):
    idx = pd.bdate_range("1993-01-04", "1996-12-31")
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0.0004, 0.008, len(idx)), index=idx)
    fund = ref + rng.normal(0, 0.0003, len(idx))
    div = pd.Series(0.0, index=idx)
    short = idx[500]                       # the dividend understated: the fund shows a 0.7% loss the market didn't have
    fine = idx[700]                        # a correctly booked ex-date
    div[[short, fine]] = 0.1
    fund[short] = ref[short] - 0.007
    fixed, days = fd.repair_missed_distributions(fund, ref, div)
    assert [d for d, *_ in days] == [short]
    assert fixed[short] == ref[short] and fixed[fine] == fund[fine]
    assert (fixed.drop(short) == fund.drop(short)).all()
    _, none = fd.repair_missed_distributions(fund, ref, div, until=idx[400])
    assert none == []


def test_validate_since_whole_months(fd, monkeypatch):
    idx = pd.bdate_range("1985-01-02", "1995-12-29")
    model = pd.Series(0.0005, index=idx)
    fund = pd.Series(0.0004, index=idx)
    monkeypatch.setattr(fd, "_real_returns", lambda t: fund.copy())
    notes = []
    monkeypatch.setattr(fd, "_simnote", notes.append)
    fd._validate("VBSIM", model, "NAESX", since="1990-01-01")
    assert "(from 1990-01-01) 1990-01..1995-11" in notes[-1]


# ------------------------------------------------------------------ fund metadata (data job)

class _Funds:
    fund_overview = {"categoryName": "Large Blend", "family": "Vanguard", "legalType": "Exchange Traded Fund"}
    fund_operations = pd.DataFrame({"VTI": [0.0003, 0.02, 5e11], "Category Average": [0.008, 0.5, 1e9]},
                                   index=pd.Index(["Annual Report Expense Ratio", "Annual Holdings Turnover",
                                                   "Total Net Assets"], name="Attributes"))
    top_holdings = pd.DataFrame({"Name": ["Apple Inc", "Microsoft Corp"], "Holding Percent": [0.06, 0.055]},
                                index=pd.Index(["AAPL", "MSFT"], name="Symbol"))
    asset_classes = {"cashPosition": 0.001, "stockPosition": 0.999, "bondPosition": 0.0}
    sector_weightings = {"technology": 0.3, "realestate": 0.0}
    description = "Tracks the CRSP US Total Market Index."

    @property
    def equity_holdings(self):
        raise RuntimeError("not available")


def test_extract_fund_meta(fd):
    info = {"longName": "Vanguard Total Stock Market Index Fund ETF Shares", "quoteType": "ETF",
            "fundInceptionDate": 990662400, "totalAssets": 4.5e11, "yield": 0.0125, "netExpenseRatio": 0.03}
    m = fd.extract_fund_meta(info, _Funds())
    assert m["name"].startswith("Vanguard Total") and m["inception"] == "2001-05-24"
    assert m["expense_ratio"] == pytest.approx(0.0003)          # the fraction from funds_data, not 0.03 (percent)
    assert m["category"] == "Large Blend" and m["family"] == "Vanguard" and m["net_assets"] == 4.5e11
    assert m["top_holdings"][0] == {"symbol": "AAPL", "name": "Apple Inc", "weight": 0.06}
    assert m["asset_classes"]["stock"] == pytest.approx(0.999) and m["sectors"] == {"technology": 0.3}
    json.dumps(m)                                                  # plain JSON types only
    m2 = fd.extract_fund_meta({"netExpenseRatio": 0.2, "category": "Foreign Large Blend"}, None)
    assert m2 == {"category": "Foreign Large Blend", "expense_ratio": pytest.approx(0.002)}
    assert fd.extract_fund_meta({}, None) == {}


def test_funds_meta_batch_rotation(fd):
    entries = {"AAA": {"fetched": "2026-09-01"}, "BBB": {"fetched": "2026-06-01"}, "CCC": {"fetched": "2026-07-01"}}
    b = fd.funds_meta_batch(["AAA", "BBB", "CCC", "NEW", "BAD"], entries, "2026-09-27", budget=3,
                            failed={"BAD": "2026-09-20"})
    assert b == ["NEW", "BBB", "CCC"]                   # missing first, then the stalest; a recent failure waits


# ------------------------------------------------------------------ fund research: statistics, screener, API

def test_stats_from_prices():
    idx = pd.bdate_range("2010-01-01", "2020-12-31")
    yrs = (idx - idx[0]).days / 365.25
    s = pd.Series(100 * 1.08 ** yrs, index=idx)
    s[s.index >= "2015-03-02"] *= 0.7                     # a 30% fall that is never recovered in level terms
    st = funds.stats_from_prices(s)
    assert st["first"] == "2010-01-01" and st["last"] == "2020-12-31"
    assert st["r1y"] == pytest.approx(0.08, abs=2e-3) and st["r5y"] == pytest.approx(0.08, abs=2e-3)
    assert st["r10y"] == pytest.approx(1.08 * 0.7 ** 0.1 - 1, abs=3e-3)
    assert st["max_dd"] == pytest.approx(-0.3, abs=1e-3) and st["max_day"] == pytest.approx(-0.3, abs=1e-3)
    young = funds.stats_from_prices(s[s.index >= "2018-06-01"])
    assert young["r3y"] is None and young["r5y"] is None and young["r1y"] is not None
    assert funds.suspect("X", {"max_day": 2.9, "max_day_sigmas": 300}) and not funds.suspect("X", {"max_day": 0.4, "max_day_sigmas": 12})


def test_screen_filters():
    rows = [{"ticker": "AAA", "type": "ETF", "name": "Alpha Total Market", "category": "Large Blend",
             "expense_ratio": 0.0003, "years": 20, "net_assets": 5e10, "r5y": 0.1, "vol3y": 0.15},
            {"ticker": "BBBX", "type": "Mutual fund", "name": "Beta Bond", "category": "Intermediate Core Bond",
             "expense_ratio": 0.005, "years": 30, "net_assets": 2e9, "r5y": 0.02, "vol3y": 0.05},
            {"ticker": "CCC", "type": "ETF", "name": None, "category": None, "expense_ratio": None, "years": 3,
             "net_assets": None, "r5y": None, "vol3y": 0.3}]
    assert [r["ticker"] for r in funds.screen(rows, q="beta")] == ["BBBX"]
    assert [r["ticker"] for r in funds.screen(rows, kind="ETF")] == ["AAA", "CCC"]
    assert [r["ticker"] for r in funds.screen(rows, max_er=0.001)] == ["AAA"]      # no expense ratio: excluded
    assert [r["ticker"] for r in funds.screen(rows, min_years=10, max_vol=0.1)] == ["BBBX"]
    assert [r["ticker"] for r in funds.screen(rows, category="large blend")] == ["AAA"]


@pytest.fixture
def fund_env(tmp_path, monkeypatch):
    monkeypatch.setattr(funds, "WARM", False)
    monkeypatch.setattr(funds, "CACHE_FILE", tmp_path / "fund_stats.json")
    monkeypatch.setattr(funds, "_CACHE", {})
    meta = {"updated_utc": "2026-09-27T00:00:00+00:00", "source": "test",
            "funds": {"SPY": {"name": "SPDR S&P 500 ETF Trust", "category": "Large Blend", "expense_ratio": 0.000945,
                              "net_assets": 6e11, "fetched": "2026-09-27"}}}
    (tmp_path / "funds_meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(funds, "META_FILE", tmp_path / "funds_meta.json")
    return tmp_path


needs_prices = pytest.mark.skipif(not {"SPY", "TLT", "GLD"} <= set(data.available_tickers()),
                                  reason="reads the repository's price files")


@needs_prices
def test_api_funds_table_and_compare(fund_env):
    T = web.api_funds({"q": ["spy"]})
    spy = next(r for r in T["funds"] if r["ticker"] == "SPY")
    assert spy["name"] == "SPDR S&P 500 ETF Trust" and spy["has_meta"] and spy["r1y"] is not None
    assert spy["max_dd"] < -0.4                              # 2008 is in SPY's history
    # a fund with no metadata still gets its statistics (and the fund list's category)
    T2 = web.api_funds({"q": ["TLT"]})
    tlt = next(r for r in T2["funds"] if r["ticker"] == "TLT")
    assert not tlt["has_meta"] and tlt["r5y"] is not None and tlt["vol3y"] > 0
    assert any("Metadata for" in n for n in T2["notes"])
    with pytest.raises(web.ClientError):
        web.api_funds({"max_er": ["cheap"]})
    few = web.api_funds({"max_er": ["0.001"]})
    assert [r["ticker"] for r in few["funds"]] == ["SPY"]
    C = web.api_funds_compare({"tickers": "SPY TLT GLD"})
    assert C["tickers"] == ["SPY", "TLT", "GLD"] and len(C["correlation"]["matrix"]) == 3
    assert C["growth"]["values"]["SPY"][0] == 10_000 and len(C["annual"]["years"]) >= 10
    assert {s["ticker"] for s in C["stats"]} == {"SPY", "TLT", "GLD"} and C["meta"]["SPY"]["expense_ratio"] == 0.000945
    with pytest.raises(ValueError, match="2 to 6"):
        web.api_funds_compare({"tickers": ["SPY"]})
    D = web.api_fund_detail({"t": ["spy"]})
    assert D["ticker"] == "SPY" and D["meta"]["category"] == "Large Blend" and D["stats"]["checked"]


def test_funds_without_metadata_file(tmp_path, monkeypatch):
    monkeypatch.setattr(funds, "META_FILE", tmp_path / "missing.json")
    assert funds.meta_doc() == {} and funds.meta("SPY") == {}


# ------------------------------------------------------------------ SIM tables

def test_new_sims_are_described_and_mapped():
    for t in ("TIPSIM", "HYGSIM", "BWXSIM"):
        assert t in data.SIMS
    assert "MODEL" in data.SIMS["TIPSIM"] and "MODEL" in data.SIMS["HYGSIM"]
    assert parser.SIM_FOR["BWX"][0][0] == "BWXSIM" and parser.SIM_FOR["IGOV"][0][0] == "BWXSIM"
    if {"BWXSIM", "BWX"} & set(data.available_tickers()):
        assert parser._asset_class_ticker("unhedged international bonds")[0] in ("BWXSIM", "BWX")
        assert parser._asset_class_ticker("international bonds")[0] in ("BNDXSIM", "BNDX")
    assert "CAPE" in parser.NOT_TICKERS
