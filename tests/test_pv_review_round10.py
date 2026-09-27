"""Portfolio Visualizer review, round 10: time-weighted returns with cash flows, asset-class names in the
analysis tools, custom series as benchmarks and ending early, partial benchmark years, grid labels and
proxies before a sleeve's inception."""
import numpy as np
import pandas as pd
import pytest

from backtester import data, metrics, report
from backtester import portfolio as pf

HAVE = {"SPY", "QQQ"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def frame(closes, start="2019-01-01"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes}, index=idx)
    df["volume"] = 1e6
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    df["split"] = 0.0
    return df


def walk(seed, n=700, drift=0.0003, vol=0.015):
    r = np.random.default_rng(seed).normal(drift, vol, n)
    r[0] = 0.0
    return 100 * np.cumprod(1 + r)


@pytest.fixture
def fake(monkeypatch):
    frames = {}

    def load(t):
        return frames[data.canonical(t)]

    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): load(t) for t in ts})
    return frames


# ------------------------------------------------------------ 1. TWR convention = flows at the close

FLOWS = [dict(contribution=500.0, contribution_freq="monthly"),
         dict(withdrawal=300.0, withdrawal_freq="monthly"),
         dict(withdrawal_pct=0.10, withdrawal_freq="yearly"),
         dict(contribution=1000.0, contribution_freq="quarterly", withdrawal=200.0, withdrawal_freq="monthly")]


@pytest.mark.parametrize("fill", ["close", "next_open"])
@pytest.mark.parametrize("fl", FLOWS)
def test_single_asset_twr_is_the_assets_own_return_whatever_the_flows(fake, fl, fill):
    """Invariant: a portfolio 100% in one asset has a time-weighted index identical (1e-9) to the asset's own
    total-return index, and to its buy-and-hold benchmark receiving the same flows."""
    fake["A"] = frame(walk(3))
    p = pf.Portfolio(tree={"asset": "A"}, rebalance="none", cash_rate=None, capital=10_000, fill=fill, **fl)
    res = pf.run(p)
    flows = res.extras["flows"]
    assert flows.abs().sum() > 0
    nv = metrics.nav(res.equity, flows)
    px = fake["A"]["close"].reindex(nv.index)
    # from the first full day in the market (a next-open start buys on the first day's open)
    k = 2 if fill == "next_open" else 1
    got = nv.iloc[k:] / nv.iloc[k]
    want = px.iloc[k:] / px.iloc[k]
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), rtol=1e-9)
    # the benchmark with the same flows (bought at the portfolio's first value) has the same index
    growth = fake["A"]["close"].reindex(res.equity.index).bfill()
    b = report.benchmarks_with_flows({"A": growth}, flows, res.equity)["A"]
    bn = metrics.nav(b, flows)
    np.testing.assert_allclose((bn.iloc[k:] / bn.iloc[k]).to_numpy(), want.to_numpy(), rtol=1e-9)


def test_withdrawal_that_empties_the_account_keeps_the_market_move(fake):
    fake["A"] = frame(walk(5, drift=-0.001))
    p = pf.Portfolio(tree={"asset": "A"}, rebalance="none", cash_rate=None, capital=10_000,
                     withdrawal=2_000, withdrawal_freq="quarterly")
    res = pf.run(p)
    dep = res.extras["depleted"]
    assert dep is not None
    nv = metrics.nav(res.equity, res.extras["flows"])
    px = fake["A"]["close"].reindex(nv.index)
    cut = nv.index <= dep
    np.testing.assert_allclose((nv[cut] / nv.iloc[1]).iloc[1:].to_numpy(), (px[cut] / px.iloc[1]).iloc[1:].to_numpy(),
                               rtol=1e-9)
    assert (nv[~cut] == nv[cut].iloc[-1]).all()      # nothing invested afterwards: no return


def test_twr_formula():
    idx = pd.bdate_range("2020-01-01", periods=4)
    eq = pd.Series([100.0, 160.0, 150.0, 0.0], index=idx)
    fl = pd.Series([0.0, 50.0, -20.0, -140.0], index=idx)
    r = metrics.twr_returns(eq, fl)
    assert r.tolist() == pytest.approx([110 / 100 - 1, 170 / 160 - 1, 140 / 150 - 1])


@needs_data
def test_qqq_with_contributions_has_the_benchmarks_cagr():
    from backtester import parser, runner
    res = runner.run(parser.parse("buy and hold QQQ, add $5,000 every month, starting with $1,000, since 2000"))
    A = report.analyze(res, sensitivity=False, mc=False)
    nv = A["nav"]
    c = data.load("QQQ")["adj_close"].reindex(nv.index).ffill()
    years = (nv.index[-1] - nv.index[0]).days / 365.25
    want = (c.iloc[-1] / c.iloc[0]) ** (1 / years) - 1
    assert A["stats"]["cagr"] == pytest.approx(want, abs=2e-4)


@needs_data
def test_qqq_withdrawals_cagr_until_depleted():
    from backtester import parser, runner
    res = runner.run(parser.parse("hold 100% QQQ, withdraw 10% per year adjusted for inflation, starting with "
                                  "$1,000,000, since 2000"))
    A = report.analyze(res, sensitivity=False, mc=False)
    assert A["stats"]["cagr"] == pytest.approx(-0.2054, abs=0.002)


# ------------------------------------------------------------ 2. asset-class names in the analysis tools

AC = {"VTISIM", "BNDSIM", "GLDSIM", "VBRSIM", "SPY", "TLT"}
needs_ac = pytest.mark.skipif(not AC <= set(data.available_tickers()), reason="long-history series not downloaded")


@needs_ac
def test_asset_class_names_resolve_like_the_parser():
    from backtester import parser
    out, notes = parser.resolve_asset_names("US Stock Market 60, Total Bond Market 40")
    assert out == "VTISIM 60, BNDSIM 40" and len(notes) == 2 and "VTISIM" in notes[0]
    assert parser.resolve_asset_names("SPY 60 TLT 40") == ("SPY 60 TLT 40", [])
    assert parser.resolve_asset_names("VTI, Gold")[0] == "VTI, GLDSIM"
    assert parser.resolve_asset_list(["US", "Stock", "Market", "TLT"])[0] == ["VTISIM", "TLT"]     # unquoted CLI words
    assert parser.resolve_asset_list(["US Stock Market", "Gold"])[0] == ["VTISIM", "GLDSIM"]
    from backtester.montecarlo import parse_weights
    nn = []
    assert parse_weights("US Stock Market 60, Total Bond Market 40", nn) == {"VTISIM": 60.0, "BNDSIM": 40.0} and len(nn) == 2


@needs_ac
def test_asset_class_names_in_correlation_factors_style_optimiser_and_monte_carlo():
    from backtester import correlation, factors, research, style, web
    R = correlation.analyze(["US Stock Market, Total Bond Market, Gold"], pair=["US Stock Market", "Gold"])
    assert R["tickers"] == ["VTISIM", "BNDSIM", "GLDSIM"] and R["rolling"]["pair"] == ["VTISIM", "GLDSIM"]
    assert any("read as VTISIM" in n for n in R["notes"])
    r, name = factors.returns_for("US Small Cap Value")
    assert name == "VBRSIM" and r.attrs["input_notes"]
    F = factors.analyze(r, "ff3", "monthly", "2000-01-01")
    assert any("read as VBRSIM" in n for n in F["notes"])
    r2, name2 = factors.returns_for({"US Stock Market": 60, "Total Bond Market": 40})
    assert name2 == "60% VTISIM / 40% BNDSIM"
    S = style.analyze(data.load("SPY")["adj_close"].pct_change().dropna(), "US Stock Market, Total Bond Market, Gold",
                      start="2005-01-01")
    assert set(S["weights"]) == {"VTISIM", "BNDSIM", "GLDSIM"} and any("read as BNDSIM" in n for n in S["notes"])
    O = research.optimize(["US Stock Market, Total Bond Market, Gold"], start="2000-01-01", methods=["max_sharpe"],
                          constraints=["Gold <= 20%"])
    w = O["portfolios"]["Max Sharpe"]["weights"]
    assert set(w) == {"VTISIM", "BNDSIM", "GLDSIM"} and w["GLDSIM"] <= 0.2 + 1e-6
    assert any("read as GLDSIM" in n for n in O["notes"])
    M = web.api_montecarlo({"weights": "US Stock Market 60, Total Bond Market 40", "years": 5, "sims": 200})
    assert M["settings"]["weights"] == {"VTISIM": 0.6, "BNDSIM": 0.4}
    assert any("read as VTISIM" in n for n in M["notes"])


# ------------------------------------------------------------ 3. custom series: benchmarks, and ending early

@pytest.fixture
def custom(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CUSTOM", tmp_path / "custom")
    data.clear_caches()
    yield tmp_path / "custom"
    data.clear_caches()


def _import_spy_monthly(name, start="1995-01", end="2019-12"):
    from backtester import custom_series as cs
    a = data.load("SPY")["adj_close"]
    me = a.groupby(a.index.to_period("M")).last()
    r = me.pct_change().dropna().loc[start:end]
    cs.import_series(name, "date,return\n" + "\n".join(f"{p.end_time.date()},{v * 100:.8f}%" for p, v in r.items()))


@pytest.mark.skipif(not {"SPY", "VBMFX"} <= set(data.available_tickers()), reason="price data not downloaded")
def test_custom_series_as_benchmark_and_ending_early(custom):
    from backtester import parser, runner
    _import_spy_monthly("MYFUNDX")
    p = parser.parse("hold 50% MYFUNDX and 50% VBMFX, rebalance yearly, since 1995, vs MYFUNDX")
    assert p.benchmark == "MYFUNDX"
    res = runner.run(p)
    # the custom series ends in 2019: the backtest ends there (not held in cash to today), with a Warning
    assert res.equity.index[-1] == pd.Timestamp("2019-12-31")
    assert any(n.startswith("Warning: the data of MYFUNDX (2019-12-31, custom series)") for n in p.notes)
    assert not (res.orders["reason"] == "delisted").any()
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    assert A["primary_benchmark"] == "MYFUNDX buy & hold"
    p2 = parser.parse("hold 50% SPY and 50% VBMFX, rebalance yearly, since 1995, vs 60% MYFUNDX 40% VBMFX")
    assert p2.benchmark == "60% MYFUNDX / 40% VBMFX"
    assert runner.run(p2).equity.index[-1] == pd.Timestamp("2019-12-31")     # the benchmark ends there too
    # a plain ticker blend without separators is read too
    assert parser.parse("hold 50% SPY and 50% VBMFX, since 1995, vs 60% SPY 40% VBMFX").benchmark == "60% SPY / 40% VBMFX"


def test_series_ending_early_ends_the_run_but_a_delisting_does_not(fake, monkeypatch):
    fake["A"] = frame(walk(1))
    fake["B"] = frame(walk(2))[:400]
    tree = {"weights": "equal", "children": [{"asset": "A"}, {"asset": "B"}]}
    monkeypatch.setattr(data, "delisted", lambda: {})
    monkeypatch.setattr(data, "nasdaq100_ever", lambda: [])
    p = pf.Portfolio(tree=tree, rebalance="monthly", cash_rate=None)
    r = pf.run(p)
    assert r.equity.index[-1] == fake["B"].index[-1] and p.notes[0].startswith("Warning: the data of B")
    monkeypatch.setattr(data, "delisted", lambda: {"B": {"last_date": str(fake["B"].index[-1].date())}})
    p2 = pf.Portfolio(tree=tree, rebalance="monthly", cash_rate=None)
    r2 = pf.run(p2)
    assert r2.equity.index[-1] == fake["A"].index[-1]
    assert any(n.startswith("Delisted: B") for n in p2.notes)


# ------------------------------------------------------------ 4. benchmark partial first years

@pytest.mark.skipif(not {"SPY", "QQQ", "SPYSIM", "TLTSIM"} <= set(data.available_tickers()), reason="data not downloaded")
def test_benchmark_partial_first_year_is_labelled():
    from backtester import parser, runner
    res = runner.run(parser.parse("hold 60% SPYSIM and 40% TLTSIM, rebalance yearly, since 1990"))
    A = report.analyze(res, sensitivity=False, mc=False)
    assert A["benchmark_partial"]["SPY buy & hold"] == {"year": 1993, "from": "1993-01-29"}
    assert A["benchmark_partial"]["QQQ buy & hold"]["year"] == 1999
    txt = report.console_summary(A)
    assert "*" in next(x for x in txt.splitlines() if x.startswith("1993 "))
    assert "SPY 1993 (from Jan 29)" in txt and "QQQ 1999 (from Mar 10)" in txt
    tpl = (report.Path(report.__file__).with_name("report_template.html")).read_text()
    assert "benchmark_partial" in tpl


# ------------------------------------------------------------ 5. grid report labels, frontier ticks

def test_head_to_head_uses_the_real_starting_amount(fake):
    fake["A"] = frame(walk(4))
    fake["SPY"] = fake["QQQ"] = frame(walk(9))
    p = pf.Portfolio(tree={"asset": "A"}, rebalance="none", cash_rate=None, capital=100_000, withdrawal=3_000,
                     withdrawal_freq="yearly")
    res = pf.run(p)
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    A["benchmarks"] = {}
    A["benchmarks_with_flows"] = {}
    C = report.common_window_stats([A])
    assert C["initial"] == 100_000 and C["has_flows"]
    col = C["columns"]["Strategy"]
    px = fake["A"]["close"]
    assert col["end_equity"] == pytest.approx(100_000 * px.iloc[-1] / px.iloc[0], rel=1e-9)
    assert col["final_balance"] == pytest.approx(float(res.equity.iloc[-1]))
    tpl = (report.Path(report.__file__).with_name("report_template.html")).read_text()
    assert "Final value of $10,000" not in tpl and "Growth of ${usd(C.initial" in tpl
    assert "Details for ${R().name}" in tpl
    rt = (report.Path(report.__file__).with_name("research_template.html")).read_text()
    assert "niceStep" in rt and "pct(vv, 0)" not in rt


# ------------------------------------------------------------ 6. proxies before inception (opt-in)

def test_splice_proxy_joins_on_the_first_day():
    a = frame(walk(7, n=300), start="2010-01-01")
    b = frame(walk(8, n=100), start="2010-10-01")
    b["close"] *= 3
    out = pf.splice_proxy(b, a)
    assert out.index[0] == a.index[0] and out.index[-1] == b.index[-1]
    r = out["close"].pct_change()
    ra = a["close"].pct_change()
    t0 = b.index[0]
    assert r[t0] == pytest.approx(ra[t0], rel=1e-12)                       # the join day has the proxy's return
    pre = out.index < t0
    np.testing.assert_allclose(r[pre].iloc[1:].to_numpy(), ra[a.index < t0].iloc[1:].to_numpy(), rtol=1e-9, atol=1e-15)
    assert (out.loc[out.index >= t0, "close"] == b["close"]).all()


@pytest.mark.skipif(not {"TIPSIM", "IEFSIM", "EEMSIM", "EFASIM", "VTISIM", "VNQSIM"} <= set(data.available_tickers()),
                    reason="long-history series not downloaded")
def test_named_portfolio_can_start_earlier_with_stated_proxies():
    from backtester import parser, runner
    p = parser.parse("Swensen portfolio, rebalance yearly, since 1980")
    assert not p.proxies
    runner.run(p)
    w = next(n for n in p.notes if n.startswith("Warning: You asked to start on 1980"))
    assert "with proxies before inception" in w and "IEFSIM for TIPSIM" in w
    q = parser.parse("Swensen portfolio, rebalance yearly, since 1980, with proxies before inception")
    assert q.proxies == {"EEMSIM": "EFASIM", "TIPSIM": "IEFSIM"}
    assert "Before inception (opt-in proxies)" in q.summary()
    res = runner.run(q)
    assert res.equity.index[1] < pd.Timestamp("1980-01-10")
    assert any(n.startswith("Proxy before inception (opt-in): IEFSIM (intermediate Treasuries) stands in for TIPSIM")
               for n in q.notes)
    assert res.holdings["TIPSIM"].loc["1985"].mean() > 0.1
    from backtester import runner as rn
    assert rn.from_dict(rn.to_dict(q)).proxies == q.proxies
