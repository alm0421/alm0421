"""Round 9 (Portfolio Visualizer expert review): day 0 (the starting balance is bought at the close of the session
before the first day, so the first day's return counts), calendar-month 12-1 momentum on month-end prices, the
allocation grid's common period, blended benchmarks of asset-class names, portfolio phrases, income and per-asset
yearly tables, cosmetics, and imported custom return series."""
import json

import numpy as np
import pandas as pd
import pytest

from backtester import data, metrics, parser, report, runner
from backtester import portfolio as pf

AVAIL = set(data.available_tickers())
needs = lambda *ts: pytest.mark.skipif(not set(ts) <= AVAIL, reason="price data not downloaded")  # noqa: E731


def port(text) -> pf.Portfolio:
    p = parser.parse(text)
    assert isinstance(p, pf.Portfolio), text
    return p


def _yr(adj: pd.Series, y: int) -> float:
    """Calendar-year total return from adjusted closes: last close of y-1 -> last close of y."""
    return float(adj[adj.index.year == y].iloc[-1] / adj[adj.index.year == y - 1].iloc[-1] - 1)


# ------------------------------------------------------------------ 1. day 0: the first day's return counts

@needs("SPY", "QQQ")
def test_spy_2010_is_the_whole_calendar_year():
    p = port("hold 100% SPY from 2010 to 2019, rebalance yearly")
    res = runner.run(p)
    adj = data.load("SPY")["adj_close"]
    want = _yr(adj, 2010)
    assert want == pytest.approx(0.1506, abs=5e-4)
    # day 0 = 2009-12-31 with the starting balance; bought at that close
    assert res.equity.index[0] == pd.Timestamp("2009-12-31") and res.equity.iloc[0] == 10_000
    assert res.extras["day0"] and str(res.orders.iloc[0]["date"]) == "2009-12-31"
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    # (the portfolio reinvests the cash dividends at the close; adj_close uses Yahoo's factors: equal to rounding)
    assert A["yearly"].loc[2010, "return"] == pytest.approx(want, abs=5e-5)
    assert not A["yearly"].loc[2010, "partial"]
    # CAGR over exactly ten calendar years
    ten = float(adj.loc["2019-12-31"] / adj.loc["2009-12-31"])
    assert A["stats"]["cagr"] == pytest.approx(ten ** (1 / 10) - 1, abs=2e-4)
    # the benchmark follows the same convention: SPY buy & hold equals the portfolio
    assert A["yearly"].loc[2010, "SPY buy & hold"] == pytest.approx(want, abs=1e-9)
    b = A["benchmarks"]["SPY buy & hold"]
    assert b.index[0] == res.equity.index[0] and b.iloc[-1] == pytest.approx(res.equity.iloc[-1], rel=1e-3)
    # P&L by holding still adds up to the cent
    at = A["attribution"]
    assert float(at["pnl"].sum()) + A["interest"] - A.get("fees", 0.0) == pytest.approx(res.equity.iloc[-1] - 10_000, abs=0.01)


@needs("SPY", "QQQ")
def test_equity_csv_keeps_day_0(tmp_path):
    res = runner.run(port("hold 100% SPY from 2010 to 2011, rebalance yearly"))
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    report.write_outputs(A, tmp_path, excel=False)
    eq = pd.read_csv(tmp_path / "equity.csv", index_col="date", parse_dates=True)
    assert eq.index[0] == pd.Timestamp("2009-12-31") and eq["equity"].iloc[0] == 10_000


@needs("SPYSIM")
def test_a_year_is_the_same_whichever_year_the_run_starts():
    a = runner.run(port("hold 100% SPYSIM since 1972, rebalance yearly, until 1975"))
    b = runner.run(port("hold 100% SPYSIM since 1973, rebalance yearly, until 1975"))
    ya = metrics.yearly_returns({"x": a.equity})["x"]
    yb = metrics.yearly_returns({"x": b.equity})["x"]
    assert ya.loc[1973] == pytest.approx(yb.loc[1973], abs=1e-12)
    assert ya.loc[1973] == pytest.approx(_yr(data.load("SPYSIM")["adj_close"], 1973), abs=1e-9)


@needs("SPY", "QQQ")
def test_next_open_fill_decides_on_day_0_and_buys_at_the_first_open():
    res = runner.run(port("hold 100% SPY from 2010 to 2010-06-30, trade at the next open"))
    o = res.orders.iloc[0]
    spy = data.load("SPY")
    assert str(o["date"]) == "2010-01-04" and o["price"] == pytest.approx(spy.loc["2010-01-04", "open"], rel=1e-9)
    assert res.equity.iloc[0] == 10_000 and res.equity.index[0] == pd.Timestamp("2009-12-31")


@needs("SPY", "TLT")
def test_day_0_decision_uses_data_up_to_that_close():
    """A timing rule is evaluated on day 0 (2009-12-31) with only the data up to that close."""
    p = port("if SPY is above its 200 day moving average hold SPY, otherwise hold TLT, rebalance monthly, since 2010")
    res = runner.run(p)
    first = res.orders.iloc[0]
    assert str(first["date"]) == "2009-12-31" and first["reason"] == "initial"
    spy = data.load("SPY")
    c = spy["adj_close"].loc[:"2009-12-31"]
    want = "SPY" if c.iloc[-1] > c.iloc[-200:].mean() else "TLT"
    assert first["ticker"] == want
    # truncating the data after day 0 does not change the day-0 decision (it only used data up to that close)
    p2 = port("if SPY is above its 200 day moving average hold SPY, otherwise hold TLT, rebalance monthly, "
              "from 2010 to 2010-01-15")
    assert runner.run(p2).orders.iloc[0]["ticker"] == want


@needs("SPY", "AGG")
def test_contributions_and_withdrawals_count_from_the_first_day():
    p = port("hold 60% SPY and 40% AGG, rebalance yearly, from 2010 to 2012, add $1,000 a month")
    res = runner.run(p)
    f = res.extras["flows"]
    assert f.iloc[0] == 0 and f.index[f > 0][0] == pd.Timestamp("2010-01-04") and (f > 0).sum() == 36


@needs("SPY", "QQQ")
def test_first_ever_bar_keeps_the_previous_convention_with_a_note():
    p = pf.Portfolio(tree={"asset": "QQQ"}, rebalance="yearly", cash_rate=None, end="2000-12-31")
    res = pf.run(p)
    assert not res.extras["day0"] and res.equity.index[1] == data.load("QQQ").index[0]
    assert any(n.startswith("First day: QQQ has no price before") for n in p.notes)


@needs("SPYSIM", "IEFSIM")
def test_monte_carlo_history_includes_the_first_month():
    from backtester import montecarlo as mc
    h = mc.monthly_asset_returns(["SPYSIM", "IEFSIM"], "1972-01-01")
    assert h.index[0] == pd.Timestamp("1972-01-31")
    adj = data.load("SPYSIM")["adj_close"]
    want = adj[adj.index <= "1972-01-31"].iloc[-1] / adj[adj.index <= "1971-12-31"].iloc[-1] - 1
    assert h["SPYSIM"].iloc[0] == pytest.approx(want, rel=1e-12)
    # a window starting mid-month starts with the next whole month
    assert mc.monthly_asset_returns(["SPYSIM"], "1972-01-15").index[0] == pd.Timestamp("1972-02-29")
    R = mc.run(mc.Settings(weights={"SPYSIM": 0.6, "IEFSIM": 0.4}, start="1972-01-01", sims=100, years=10))
    assert str(R["settings"]["history_start"]) == "1972-01-01"


@needs("SPY", "TLT")
def test_partial_years_are_labelled():
    res = runner.run(port("hold 60% SPY and 40% TLT, rebalance yearly, from 2010-03-03 to 2012-09-14"))
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    y = A["yearly"]
    assert y.loc[2010, "partial"] and str(y.loc[2010, "from"]) == "2010-03-03"
    assert y.loc[2012, "partial"] and str(y.loc[2012, "to"]) == "2012-09-14" and not y.loc[2011, "partial"]
    txt = report.console_summary(A)
    assert "* partial year: 2010 (from Mar 3), 2012 (to Sep 14)" in txt
    tpl = (report.Path(report.__file__).with_name("report_template.html")).read_text()
    assert "function yrLabel" in tpl and "'from ' + MON" in tpl


# ------------------------------------------------------------------ 2. calendar-month 12-1 momentum

@needs("SPY", "EFA", "EEM", "IEF", "TLT", "VNQ", "GLD", "DBC")
def test_calendar_skip_month_momentum_matches_month_end_prices():
    T = ["SPY", "EFA", "EEM", "IEF", "TLT", "VNQ", "GLD", "DBC"]
    p = port("hold the top 3 of SPY, EFA, EEM, IEF, TLT, VNQ, GLD, DBC by 12 month return skipping the last month, "
             "using calendar months, rebalance monthly, since 2008")
    assert any("12-1 month momentum on month-end prices" in n for n in p.notes)
    assert "12-1 month momentum on month-end prices" in p.summary()
    res = runner.run(p)
    h = res.holdings
    px = pd.concat({t: data.load(t)["adj_close"] for t in T}, axis=1)
    me = px.groupby(px.index.to_period("M")).last()
    mom = me.shift(1) / me.shift(12) - 1          # P(t-1m) / P(t-12m) - 1 on month-end prices
    per = h.index.to_period("M")
    ends = [d for k, d in enumerate(h.index[:-1]) if per[k] != per[k + 1]]
    ok = 0
    for d in ends:
        held = {c for c in h.columns if c != "cash" and h.loc[d, c] > 1e-6}
        want = set(mom.loc[d.to_period("M")].dropna().sort_values(ascending=False).index[:3])
        ok += held == want
    assert len(ends) >= 224 and ok == len(ends)


def test_calendar_skip_return_function():
    idx = pd.bdate_range("2019-12-02", "2021-03-31")
    x = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    from backtester import expr
    r = expr.calendar_month_return(x, 11, 1)
    me = x.groupby(x.index.to_period("M")).last()
    d = pd.Timestamp("2021-01-29")
    assert r.loc[d] == pytest.approx(me.loc["2020-12"] / me.loc["2020-01"] - 1)
    # mid-month: from the last completed month-end
    assert r.loc["2021-02-10"] == pytest.approx(me.loc["2020-12"] / me.loc["2020-01"] - 1)
    ns = expr.Namespace(pd.DataFrame({"open": x, "high": x, "low": x, "close": x, "volume": 0.0, "dividend": 0.0,
                                      "adj_close": x}), month_lookbacks="calendar")
    v = expr.evaluate_value("ref(tret(tr, 231), 21)", ns)
    assert v.loc[d] == pytest.approx(r.loc[d])


# ------------------------------------------------------------------ 3-4. the grid's common period, blended benchmark

GRID = {"rows": [{"asset": "US Stock Market", "w": [60, 40, 30]}, {"asset": "Total Bond Market", "w": [40, 40, None]},
                 {"asset": "VXUSSIM", "w": [None, 20, None]}, {"asset": "TLTSIM", "w": [None, None, 40]},
                 {"asset": "IEFSIM", "w": [None, None, 15]}, {"asset": "GLD", "w": [None, None, 15]}],
        "names": ["60/40", "Four-fund", "All weather"], "start": "1972",
        "flows": [{"kind": "contribute", "amount": 1000, "freq": "monthly", "start_year": 1, "end_year": 20},
                  {"kind": "withdraw", "amount": 60000, "freq": "annual", "start_year": 21}],
        "rebalance": "yearly", "benchmark": "60% US Stock Market 40% Total Bond Market"}


@needs("VTISIM", "BNDSIM", "VXUSSIM", "TLTSIM", "IEFSIM", "GLD")
def test_grid_columns_share_the_common_period(tmp_path, monkeypatch):
    from backtester import web
    monkeypatch.setattr(web, "RUNS", tmp_path)
    g = dict(GRID, rows=[r for r in GRID["rows"] if r["asset"] != "GLD"])
    g["rows"] = [dict(r, w=r["w"][:2]) for r in g["rows"]]
    specs, notes, problems = web.grid_specs(g)
    assert problems == []
    first = data.load("VXUSSIM").index
    common = str(first[1].date())
    assert all(s.start == common for s in specs)
    assert notes[0].startswith("Warning: The period is constrained") and "Four-fund holds VXUSSIM" in notes[0]
    assert all(s.notes[0] == notes[0] for s in specs)
    assert all(f"Period: {common}" in s.summary() for s in specs)
    assert all(s.benchmark == "60% VTISIM / 40% BNDSIM" for s in specs)
    j = web.api_grid(dict(g))
    assert j["portfolios"][0]["interpretation"].count(f"Period: {common}") == 1
    for s in specs:
        res = runner.run(s)
        assert res.equity.index[0] == first[0]
        f = res.extras["flows"]
        # years 1-20 of contributions from 1975, withdrawals from year 21 (1995)
        assert f[f > 0].index[0].year == first[0].year and f[f > 0].index[-1].year == first[0].year + 19
        assert f[f < 0].index[0].year == first[0].year + 20
    html = (tmp_path / j["id"] / "report.html").read_text()
    assert '"universe": ["VTISIM", "BNDSIM"]' in html or '"universe":["VTISIM","BNDSIM"]' in html


@needs("VTISIM", "BNDSIM", "SPY", "AGG")
def test_blended_benchmark_of_asset_class_names():
    from backtester import web
    assert metrics.benchmark_label("60% US Stock Market 40% Total Bond Market") == "60% VTISIM / 40% BNDSIM"
    assert metrics.benchmark_label("60% SPY and 40% AGG") == "60% SPY / 40% AGG"
    assert metrics.benchmark_label({"US Stock Market": 0.5, "AGG": 0.5}) == "50% VTISIM / 50% AGG"
    specs, _, problems = web.grid_specs({"rows": [{"asset": "SPY", "w": [100]}], "start": "2010",
                                         "benchmark": "60% US Stock Market 40% Total Bond Market"})
    assert problems == [] and specs[0].benchmark == "60% VTISIM / 40% BNDSIM"
    p = port("hold 60% SPY and 40% TLT, rebalance yearly, since 2005, benchmark 60% US Stock Market and 40% Total Bond Market")
    assert p.benchmark == "60 VTISIM 40 BNDSIM" or metrics.benchmark_label(p.benchmark) == "60% VTISIM / 40% BNDSIM"
    from backtester.__main__ import main
    assert main(["hold 60% SPY and 40% TLT, since 2015", "--benchmark", "60% US Stock Market 40% Total Bond Market",
                 "--dry-run"]) == 0


def test_report_template_hides_empty_cost_table_and_counts_all_tickers():
    tpl = (report.Path(report.__file__).with_name("report_template.html")).read_text()
    assert 'id="costCard"' in tpl and "$('costCard').classList.toggle('hide', !(r.sensitivity || []).length)" in tpl
    assert "new Set(D.runs.flatMap(x => x.universe || []))" in tpl


# ------------------------------------------------------------------ 5. phrases

@needs("SPY", "TLT", "VTI", "GLD", "SHY", "EFA")
@pytest.mark.parametrize("text, check", [
    ("hold 60% SPY and 40% TLT, rebalance yearly, start 2008", lambda p: p.start == "2008-01-01"),
    ("hold 60% SPY and 40% TLT, rebalance yearly, starting 2008", lambda p: p.start == "2008-01-01"),
    ("hold 60% SPY and 40% TLT, rebalance yearly, start in March 2008", lambda p: p.start == "2008-03-01"),
    ("hold 60% SPY and 40% TLT, rebalance yearly, starting balance $1,000,000", lambda p: p.capital == 1_000_000),
    ("hold 60% SPY and 40% TLT, initial investment of $250,000", lambda p: p.capital == 250_000),
    ("60% US stocks 40% bonds 1972-2020",
     lambda p: (p.start, p.end) == ("1972-01-01", "2020-12-31") and p.universe == ["VTISIM", "BNDSIM"]),
    ("hold 60% SPY and 40% TLT from 2000-2010, rebalance yearly", lambda p: (p.start, p.end) == ("2000-01-01", "2010-12-31")),
    ("hold 25% each of VTI, TLT, GLD and SHY", lambda p: p.tree["w"] == [0.25] * 4 and p.universe == ["VTI", "TLT", "GLD", "SHY"]),
    ("hold VTI, TLT, GLD and SHY at 25% each, rebalance quarterly", lambda p: p.tree["w"] == [0.25] * 4 and p.rebalance == "quarterly"),
    ("hold 60% SPY and 40% TLT, rebalance every year in June", lambda p: p.rebalance == "yearly_6"),
    ("hold 60% SPY and 40% TLT, rebalance annually in June", lambda p: p.rebalance == "yearly_6"),
    ("hold 60% SPY and 40% TLT, rebalance each September", lambda p: p.rebalance == "yearly_9"),
    ("hold 60% SPY and 40% TLT, annual rebalancing in March", lambda p: p.rebalance == "yearly_3"),
])
def test_portfolio_phrases(text, check):
    assert check(port(text)), text


@needs("SPY", "TLT", "VTI", "GLD", "SHY")
def test_each_weights_must_add_up():
    with pytest.raises(parser.ParseError, match="not 100%"):
        port("hold 20% each of VTI, TLT, GLD and SHY")


@needs("SPY", "EFA", "TLT")
def test_weighted_multi_period_momentum():
    p = port("hold the top 2 of SPY, EFA, TLT by 3 month return weighted 50%, 6 month weighted 30% and 12 month weighted 20%, "
             "rebalance monthly")
    assert p.tree["filter"]["by"] == "0.5 * tret(tr, 63) + 0.3 * tret(tr, 126) + 0.2 * tret(tr, 252)"
    assert any("weighted average of the total returns" in n for n in p.notes)
    with pytest.raises(parser.ParseError, match="add up to 90%"):
        port("hold the top 2 of SPY, EFA, TLT by 3 month return weighted 50%, 6 month weighted 20% and 12 month weighted 20%")
    # calendar mode: each lookback month-end to month-end
    pc = port("hold the top 1 of SPY, EFA, TLT by 3 month return weighted 50% and 12 month weighted 50%, using calendar "
              "months, rebalance monthly, since 2010")
    res = runner.run(pc)
    px = pd.concat({t: data.load(t)["adj_close"] for t in ("SPY", "EFA", "TLT")}, axis=1)
    me = px.groupby(px.index.to_period("M")).last()
    score = 0.5 * (me / me.shift(3) - 1) + 0.5 * (me / me.shift(12) - 1)
    h = res.holdings
    d = h.index[h.index.to_period("M") == pd.Period("2015-06", "M")][-1]
    assert h.loc[d].drop("cash").idxmax() == score.loc["2015-06"].idxmax()


@needs("SPY", "TLT")
def test_rebalance_in_a_chosen_month():
    res = runner.run(port("hold 60% SPY and 40% TLT, rebalance every year in June, from 2010 to 2013"))
    days = sorted({str(d) for d in res.orders["date"]})
    assert days == ["2009-12-31", "2010-06-30", "2011-06-30", "2012-06-29", "2013-06-28"]
    assert "at the end of June" in res.strategy.summary()
    with pytest.raises(ValueError):
        pf.Portfolio(tree={"asset": "SPY"}, rebalance="yearly_13").validate()


# ------------------------------------------------------------------ 6. income and per-asset yearly returns

@needs("SPY", "AGG")
def test_income_and_asset_year_tables(tmp_path):
    res = runner.run(port("hold 60% SPY and 40% AGG, rebalance yearly, from 2012 to 2015"))
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    inc = A["income_yearly"]
    assert list(inc.index) == [2012, 2013, 2014, 2015]
    divs = res.extras["income"]["dividends"]
    assert inc["dividends"].sum() == pytest.approx(divs.sum(), rel=1e-12) and inc["dividends"].min() > 0
    assert inc.loc[2012, "start_balance"] == 10_000
    assert inc.loc[2013, "yield"] == pytest.approx(inc.loc[2013, "income"] / res.equity.loc[:"2012-12-31"].iloc[-1])
    assert 0.01 < inc.loc[2013, "dividend_yield"] < 0.04
    ay = A["asset_yearly"]
    assert ay.loc[2013, "SPY"] == pytest.approx(_yr(data.load("SPY")["adj_close"], 2013), abs=1e-12)
    assert ay.loc[2012, "AGG"] == pytest.approx(_yr(data.load("AGG")["adj_close"], 2012), abs=1e-12)
    pay = report.run_payload(A, 0, res.equity.index)
    assert pay["income_yearly"][0]["year"] == 2012 and pay["asset_yearly"]["tickers"] == ["SPY", "AGG"]
    report.write_outputs(A, tmp_path, excel=True)
    x = pd.ExcelFile(tmp_path / "report.xlsx")
    assert "Income" in x.sheet_names and "Asset returns by year" in x.sheet_names


# ------------------------------------------------------------------ 7. cosmetics

@needs("SPY", "TLT")
def test_interpretation_texts():
    p = port("hold 60% SPY and 40% TLT, withdraw 5% a year taken quarterly, 0.2% expense ratio, 1.5x leverage")
    assert p.withdrawal_pct == pytest.approx(0.0125) and p.withdrawal_freq == "quarterly"
    s = p.summary()
    assert "withdraw 5%/yr of the balance (1.25% quarterly)" in s
    assert "charged daily on the gross invested assets" in s
    assert "gross invested" not in port("hold 60% SPY and 40% TLT, 0.2% expense ratio").summary()
    tpl = (report.Path(report.__file__).with_name("report_template.html")).read_text()
    assert "flat ? '0.0'" in tpl and "flat ? '·'" not in tpl


# ------------------------------------------------------------------ 8. custom series

@pytest.fixture
def custom(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CUSTOM", tmp_path / "custom")
    data.clear_caches()
    yield tmp_path / "custom"
    data.clear_caches()


def _spy_monthly_csv(pct=True, start="2005-01", end="2019-12"):
    a = data.load("SPY")["adj_close"]
    me = a.groupby(a.index.to_period("M")).last()
    r = me.pct_change().dropna().loc[start:end]
    return "date,return\n" + "\n".join(f"{p.end_time.date()},{v * 100:.8f}%" if pct else f"{p.end_time.date()},{v:.10f}"
                                       for p, v in r.items()), me


@needs("SPY", "TLT")
def test_import_monthly_returns_and_use_them_everywhere(custom):
    from backtester import custom_series as cs
    txt, me = _spy_monthly_csv()
    info = cs.import_series("MYFUND", txt, source="mine.csv")
    assert info["frequency"] == "monthly" and info["kind"] == "returns" and info["units"] == "percent"
    assert (custom / "MYFUND.csv").is_file() and json.loads((custom / "MYFUND.json").read_text())["source"] == "mine.csv"
    df = data.load("MYFUND")
    assert df.index[0] == pd.Timestamp("2004-12-31") and df["adj_close"].iloc[0] == 100
    # stepped on the last NYSE session of each month, flat in between
    assert df.loc["2010-05-03", "close"] == df.loc["2010-04-30", "close"] != df.loc["2010-04-29", "close"]
    assert data.stepped_ranges("MYFUND")
    # the month-end levels reproduce SPY's monthly total returns
    lv = df["adj_close"].groupby(df.index.to_period("M")).last()
    assert (lv.loc["2015-06"] / lv.loc["2014-06"]) == pytest.approx(me.loc["2015-06"] / me.loc["2014-06"], rel=1e-7)
    assert "MYFUND" in data.available_tickers()
    p = port("hold 60% MYFUND and 40% TLT, rebalance yearly, from 2008 to 2018")
    assert p.universe == ["MYFUND", "TLT"]
    res = runner.run(p)
    assert any(n.startswith("Custom series: MYFUND") for n in p.notes)
    solo = runner.run(port("hold 100% MYFUND, rebalance yearly, from 2008 to 2018"))
    assert solo.equity.iloc[-1] / solo.equity.iloc[0] == pytest.approx(me.loc["2018-12"] / me.loc["2007-12"], rel=1e-7)
    # as a benchmark, in the Monte Carlo and the optimiser inputs
    A = report.analyze(runner.run(pf.Portfolio(tree={"asset": "SPY"}, start="2010-01-01", end="2015-12-31",
                                               benchmark="MYFUND")), sensitivity=False, mc=False, detail=False)
    assert A["primary_benchmark"] == "MYFUND buy & hold"
    from backtester import montecarlo as mc
    h = mc.monthly_asset_returns(["MYFUND", "TLT"], "2006-01-01")
    assert h.index[0] == pd.Timestamp("2006-01-31")
    assert res.equity.iloc[-1] > 0


def test_import_validation(custom):
    from backtester import custom_series as cs
    with pytest.raises(ValueError, match="increasing order"):
        cs.import_series("ABC", "2020-01-31,1%\n2020-03-31,1%\n2020-02-29,1%\n")
    with pytest.raises(ValueError, match="every month"):
        cs.import_series("ABC", "2020-01-31,1%\n2020-02-29,1%\n2020-05-31,1%\n2020-06-30,1%\n")
    with pytest.raises(ValueError, match="gap of"):
        cs.import_series("ABC", "\n".join(f"{d.date()},{100 + k}" for k, d in enumerate(
            list(pd.bdate_range("2020-01-01", "2020-02-28")) + list(pd.bdate_range("2020-05-01", "2020-06-30")))),
                         kind="prices", monthly=False)
    with pytest.raises(ValueError, match="not plausible"):
        cs.import_series("ABC", "2020-01-31,1%\n2020-02-29,500%\n2020-03-31,1%\n")
    with pytest.raises(ValueError, match="capital letters"):
        cs.import_series("my fund", "2020-01-31,1%\n2020-02-29,1%\n2020-03-31,1%\n")
    if "SPY" in AVAIL:
        with pytest.raises(ValueError, match="already a ticker"):
            cs.import_series("SPY", "2020-01-31,1%\n2020-02-29,1%\n2020-03-31,1%\n")
    # daily prices, a weekend date moves to the next session; decimals and prices detected
    days = pd.bdate_range("2021-01-04", "2021-03-31")
    txt = "date,close\n" + "\n".join(f"{d.date()},{100 * 1.001 ** k:.6f}" for k, d in enumerate(days))
    info = cs.import_series("DAILY1", txt)
    assert info["kind"] == "prices" and info["frequency"] == "daily" and len(data.load("DAILY1")) >= 55
    info = cs.import_series("DEC", "2021-01-29,0.01\n2021-02-26,-0.02\n2021-03-31,0.015\n")
    assert info["units"] == "decimal" and data.load("DEC")["adj_close"].iloc[-1] == pytest.approx(100 * 1.01 * 0.98 * 1.015)
    cs.delete_series("DEC")
    assert "DEC" not in data.available_tickers()


@needs("SPY")
def test_import_series_cli_and_site(custom, tmp_path, capsys):
    from backtester import web
    from backtester.__main__ import main
    txt, _ = _spy_monthly_csv(pct=False, start="2012-01", end="2016-12")
    f = tmp_path / "r.csv"
    f.write_text(txt)
    assert main(["import-series", "CLI1", str(f), "--returns", "--monthly"]) == 0
    out = capsys.readouterr().out
    assert "Imported CLI1: a custom series (monthly returns (decimal), from r.csv)" in out
    j = web.api_series({"action": "import", "name": "site2", "csv": txt, "source": "r.csv", "frequency": "monthly"})
    assert j["series"]["name"] == "SITE2" and "SITE2" in j["example"]
    names = [x["name"] for x in web.api_series({"action": "list"})["series"]]
    assert names == ["CLI1", "SITE2"]
    with pytest.raises(web.ClientError, match="increasing order"):
        web.api_series({"action": "import", "name": "BAD", "csv": "2020-02-29,1%\n2020-01-31,1%\n2020-03-31,1%\n"})
    web.api_series({"action": "delete", "name": "SITE2"})
    assert [x["name"] for x in web.api_series({"action": "list"})["series"]] == ["CLI1"]
    tpl = web.APP.read_text()
    assert 'id="serFile"' in tpl and "/api/series" in tpl


# ------------------------------------------------------------------ the pages in headless Chromium (port 8850)

PORT = 8850


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    import glob
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    import shutil
    import subprocess
    import threading
    import time
    from http.server import ThreadingHTTPServer
    from backtester import web
    old = web.RUNS
    web.RUNS = tmp_path_factory.mktemp("r9_runs")
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), web.Handler)
    except OSError:
        if shutil.which("fuser"):
            subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True)
            time.sleep(1.0)
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{PORT}"
    srv.shutdown()
    srv.server_close()
    web.RUNS = old


@pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")
@needs("SPY", "AGG")
def test_report_shows_income_asset_years_and_partial_labels(tmp_path):
    from playwright.sync_api import sync_playwright
    res = runner.run(port("hold 60% SPY and 40% AGG, rebalance yearly, from 2012-03-05 to 2014-12-31"))
    A = report.analyze(res, sensitivity=False, mc=False)
    report.write_outputs(A, tmp_path, excel=False)
    errs: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto((tmp_path / "report.html").as_uri())
        pg.wait_for_selector("#incomeTbl tr")
        assert pg.is_visible("#incomeCard") and pg.is_visible("#assetYrCard") and pg.is_hidden("#costCard")
        assert "2012 (from Mar 5)" in pg.inner_text("#yrTbl") and "2012 (from Mar 5)" in pg.inner_text("#incomeTbl")
        head = pg.inner_text("#assetYrTbl")
        assert "SPY" in head and "AGG" in head and "2013" in head
        assert "·" not in pg.inner_text("#heat")
        b.close()
    assert not errs, errs


@pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")
@needs("SPY")
def test_data_page_imports_a_series(site, custom, tmp_path):
    from playwright.sync_api import sync_playwright
    txt, _ = _spy_monthly_csv(start="2010-01", end="2015-12")
    f = tmp_path / "fund.csv"
    f.write_text(txt)
    errs: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": 390, "height": 900})
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(site + "/#data")
        pg.wait_for_selector("#serBtn")
        pg.fill("#serName", "UIFUND")
        pg.set_input_files("#serFile", str(f))
        pg.click("#serBtn")
        pg.wait_for_function("document.querySelector('#serStatus').textContent.includes('Imported UIFUND')", timeout=20_000)
        pg.wait_for_function("document.querySelector('#serList').textContent.includes('UIFUND')", timeout=20_000)
        assert (custom / "UIFUND.csv").is_file()
        assert pg.evaluate("document.documentElement.scrollWidth") <= 390
        # a bad file says why
        bad = tmp_path / "bad.csv"
        bad.write_text("2020-02-29,1%\n2020-01-31,1%\n2020-03-31,1%\n")
        pg.fill("#serName", "BADONE")
        pg.set_input_files("#serFile", str(bad))
        pg.click("#serBtn")
        pg.wait_for_function("document.querySelector('#serStatus').textContent.includes('increasing order')", timeout=20_000)
        b.close()
    assert not errs, errs


@pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")
@needs("VTISIM", "BNDSIM", "VXUSSIM")
def test_grid_page_warns_about_the_common_period(site):
    from playwright.sync_api import sync_playwright
    from backtester import web
    g = {"rows": [{"asset": "US Stock Market", "w": [60, 40]}, {"asset": "Total Bond Market", "w": [40, 40]},
                  {"asset": "VXUSSIM", "w": [None, 20]}],
         "names": ["60/40", "Four-fund"], "start": "1972", "end": "1980", "rebalance": "yearly",
         "benchmark": "60% US Stock Market 40% Total Bond Market"}
    errs: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(site + "/#backtest?g=" + web.grid_share_token(g))
        pg.wait_for_selector("#gResult:not(.hide)", timeout=180_000)
        w = pg.locator("#gNotes .warn").first
        assert "every portfolio runs from 1975-01-03" in w.inner_text() and "Four-fund holds VXUSSIM" in w.inner_text()
        fr = pg.frame_locator("#gFrame")
        fr.locator("#subtitle").wait_for(timeout=60_000)
        sub = fr.locator("#subtitle").inner_text()
        assert "2 portfolios" in sub and "3 tickers" in sub
        assert "Period: 1975-01-03" in fr.locator("#interp").inner_text()
        b.close()
    assert not errs, errs
