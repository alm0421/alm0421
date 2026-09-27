"""Round 5 (Composer, Portfolio Visualizer, QuantConnect and TradingView experts): indicator price basis
(total-return prices, built causally), Composer metadata, warm-up that waits for weighting lookbacks and starts
trading only then, one headline rule, blended benchmarks, the strategies' common period, volatility targeting,
rule-argument errors, crypto bars read from US tickers, Nasdaq-100 membership start, gross-exposure limits,
readable sweep labels and the Build page (automatic daily rebalancing, phone width, unit warnings)."""
import glob
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtester import composer_import as ci, data, engine, expr, metrics, portfolio as pf, report, research, web
from backtester.strategy import Strategy

AVAIL = set(data.available_tickers())
needs = lambda *ts: pytest.mark.skipif(not set(ts) <= AVAIL, reason="price data not downloaded")  # noqa: E731


def frame(closes, start="2019-01-01", dividends=None, freq="B"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes)) if freq == "B" else pd.date_range(start, periods=len(closes))
    df = pd.DataFrame({"open": closes * 0.995, "high": closes * 1.01, "low": closes * 0.98, "close": closes}, index=idx)
    df["volume"] = 1e6
    df["dividend"] = 0.0 if dividends is None else np.asarray(dividends, dtype=float)
    # a back-adjusted total-return close, as the data files carry it
    g = (df["close"] + df["dividend"]) / df["close"].shift(1)
    df["adj_close"] = df["close"].iloc[0] * g.fillna(1.0).cumprod()
    df["adj_close"] *= df["close"].iloc[-1] / df["adj_close"].iloc[-1]
    df["split"] = 0.0
    return df


def walk(seed, n=600, drift=0.0003, vol=0.012):
    r = np.random.default_rng(seed).normal(drift, vol, n)
    r[0] = 0.0
    return 100 * np.cumprod(1 + r)


def monthly_divs(n, amount=0.5, every=21):
    d = np.zeros(n)
    d[every::every] = amount
    return d


@pytest.fixture
def fake(monkeypatch):
    frames = {}
    real = data.load

    def load(t):
        t = data.canonical(t)
        if t in frames:
            return frames[t]
        if t in ("SPY", "QQQ", "SPYSIM"):
            raise FileNotFoundError(t)
        return real(t)

    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): load(t) for t in ts})
    return frames


def _truncated(monkeypatch, cut):
    real = data.load
    cache = {}

    def load(t):
        t = data.canonical(t)
        if t not in cache:
            df = real(t)
            cache[t] = df[df.index <= cut]
        return cache[t]
    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): load(t) for t in ts})


# ------------------------------------------------------------ 1. indicator price basis

def test_adjusted_frame_is_a_causal_total_return_index():
    df = frame([100, 101, 99, 100], dividends=[0, 0, 1.0, 0])
    adj = expr.adjusted_frame(df)
    want = [100, 101, 100, 100 * 100 / 99]
    assert adj["close"].to_numpy() == pytest.approx(want)
    f = adj["close"] / df["close"]
    for k in ("open", "high", "low"):          # the bars keep their shape
        assert (adj[k] / df[k]).to_numpy() == pytest.approx(f.to_numpy())
    assert (adj["dividend"] == 0).all() and (adj["volume"] == df["volume"]).all()
    assert (adj["adj_close"] == df["adj_close"]).all()     # `tr` is unchanged
    # causal: a value never changes when later bars (and dividends) arrive; applying it twice changes nothing
    long = frame(walk(1, 300), dividends=monthly_divs(300))
    a = expr.adjusted_frame(long)
    for cut in (50, 125, 299):
        pd.testing.assert_frame_equal(expr.adjusted_frame(long.iloc[:cut]), a.iloc[:cut])
    pd.testing.assert_frame_equal(expr.adjusted_frame(a), a)
    # daily returns are total returns
    tr = (long["close"] + long["dividend"]) / long["close"].shift(1) - 1
    assert a["close"].pct_change().iloc[1:].to_numpy() == pytest.approx(tr.iloc[1:].to_numpy())


def test_namespace_basis_applies_to_own_prices_and_to_sym(fake):
    fake["X"] = frame(walk(2, 300), dividends=monthly_divs(300, 1.5))
    fake["Y"] = frame(walk(3, 300))
    q = expr.Namespace(fake["Y"], ticker="Y")
    a = expr.Namespace(fake["Y"], ticker="Y", price_basis="adjusted")
    adj_x = expr.adjusted_frame(fake["X"])["close"]
    assert expr.evaluate_value('sym("X").close', a).to_numpy() == pytest.approx(adj_x.to_numpy())
    assert expr.evaluate_value('sym("X").close', q).to_numpy() == pytest.approx(fake["X"]["close"].to_numpy())
    ax = expr.Namespace(fake["X"], ticker="X", price_basis="adjusted")
    r_adj = expr.evaluate_value("rsi(close, 20)", ax)
    assert r_adj.dropna().to_numpy() == pytest.approx(expr.rsi_wilder(adj_x, 20).dropna().to_numpy())
    # a dividend payer's RSI is higher on total-return prices (the ex-dividend drops are not losses)
    r_q = expr.evaluate_value("rsi(close, 20)", expr.Namespace(fake["X"], ticker="X"))
    assert r_adj.mean() > r_q.mean() + 2
    with pytest.raises(ValueError):
        expr.Namespace(fake["Y"], price_basis="dividends")


@needs("BIL")
def test_bil_rsi_on_total_return_prices_matches_composer():
    df = data.load("BIL")
    q = expr.evaluate_value("rsi(close, 60)", expr.Namespace(df, ticker="BIL")).iloc[-250:]
    a = expr.evaluate_value("rsi(close, 60)", expr.Namespace(df, ticker="BIL", price_basis="adjusted")).iloc[-250:]
    assert 35 < q.median() < 65 and a.median() > 90    # T-bill fund: flat quote, steadily rising total return


def test_portfolio_basis_changes_rules_but_trades_at_quoted_prices(fake):
    fake["X"] = frame(walk(4, 400, drift=0.0), dividends=monthly_divs(400, 1.2))
    tree = {"if": "rsi(close, 14) > 55", "on": "X", "then": {"asset": "X"}, "else": {"cash": True}}
    runs = {}
    for basis in ("adjusted", "quoted"):
        p = pf.Portfolio(tree=tree, rebalance="daily", cash_rate=None, price_basis=basis)
        runs[basis] = (p, pf.run(p))
    assert pf.Portfolio(tree=tree).price_basis == "adjusted"               # Composer / PV semantics by default
    pa, ra = runs["adjusted"]
    pq, rq = runs["quoted"]
    assert not ra.holdings["X"].equals(rq.holdings["X"].reindex(ra.holdings.index))
    assert "total-return prices" in pa.summary() and "as quoted" in pq.summary()
    assert any(n.startswith("Indicator prices: total return") for n in pa.notes)
    # every order fills at the quoted close, and dividends are paid in cash
    od = ra.orders
    px = fake["X"]["close"].reindex(pd.to_datetime(od["date"])).to_numpy()
    assert od["price"].to_numpy() == pytest.approx(px)
    at = ra.extras["attribution"]
    assert at.loc[at.ticker == "X", "dividends"].iloc[0] > 0
    assert ra.equity.iloc[-1] == pytest.approx(10_000 + at["pnl"].sum() + ra.interest)


@needs("BIL", "IEF", "TLT")
def test_adjusted_basis_allocation_does_not_depend_on_future_data(monkeypatch):
    tree = {"if": 'rsi(close, 60) < rsi(sym("BIL").close, 60)', "on": "IEF", "then": {"asset": "TLT"},
            "else": {"asset": "BIL"}}
    kw = dict(tree=tree, rebalance="daily", start="2012-01-01", end="2018-12-31")
    full = pf.run(pf.Portfolio(**kw))
    cut = pd.Timestamp("2016-01-29")
    _truncated(monkeypatch, cut)
    past = pf.run(pf.Portfolio(**kw))
    a = full.equity[full.equity.index < cut - pd.Timedelta(days=3)]
    assert np.allclose(a.to_numpy(), past.equity.reindex(a.index).to_numpy(), rtol=1e-12)


def test_fixed_price_levels_get_a_basis_note(fake):
    assert pf.price_level_threshold("close > 400") and pf.price_level_threshold('sym("SPY").close < 300')
    assert pf.price_level_threshold("sma(close, 200) > 250")
    for r in ("rsi(close, 14) > 70", "close > sma(close, 200)", "tret(tr, 20) > 0.05", "sma(rsi(close, 2), 5) > 50"):
        assert not pf.price_level_threshold(r), r
    fake["X"] = frame(walk(5, 300), dividends=monthly_divs(300))
    p = pf.Portfolio(tree={"if": "close > 100", "on": "X", "then": {"asset": "X"}, "else": {"cash": True}}, cash_rate=None)
    pf.run(p)
    assert any("compares a price level with a fixed number" in n for n in p.notes)


# ------------------------------------------------------------ 2. Composer metadata

def _root(child, **kw):
    return {"step": "root", "name": "T", "rebalance": "daily", "children": [child], **kw}


def test_composer_accepts_benign_metadata_in_both_spellings():
    asset = {"step": "asset", "ticker": "SPY", "asset_class": "EQUITIES", "name": "SPDR", "exchange": "ARCX"}
    sym = _root(asset, **{"asset_class": "EQUITIES", "asset_classes": ["EQUITIES"], "description": "d", "color": "#fff",
                          "rebalance-corridor-width": 0.05, "version": "v2", "created-at": "2024-01-01",
                          "last-updated-at": "2024-02-01", "asset-class": "EQUITIES"})
    spec = ci.convert(sym)
    assert spec["tree"] == {"asset": "SPY"} and spec["rebalance"] == "daily"
    assert "drift_band" not in spec                         # the corridor only applies to threshold rebalancing
    assert any("corridor" in n for n in spec["notes"])
    assert spec["price_basis"] == "adjusted"                # Composer computes indicators on adjusted prices
    # threshold rebalancing still uses the corridor
    t = ci.convert(_root({"step": "asset", "ticker": "SPY"}, **{"rebalance": "none", "rebalance-corridor-width": 0.05}))
    assert t["drift_band"] == pytest.approx(0.05)


def test_composer_still_refuses_fields_that_could_change_meaning():
    for extra in ({"weight-override": 0.5}, {"leverage": 2}, {"rebalance_corridor_width": 0.05}):
        with pytest.raises(ci.ComposerImportError, match="unknown field"):
            ci.convert(_root({"step": "asset", "ticker": "SPY", **extra}))


# ------------------------------------------------------------ 3/4/5. warm-up and headline

def test_warm_up_waits_for_the_filter_weighting_lookback(fake):
    fake["A"] = frame(walk(6, 500), start="2018-01-01")
    fake["B"] = frame(walk(7, 300), start="2018-10-01")
    tree = {"filter": {"select": "top", "n": 2, "by": "tret(tr, 10)", "weights": "inverse_vol", "lookback": 40},
            "universe": "children", "children": [{"asset": "A"}, {"asset": "B"}]}
    p = pf.Portfolio(tree=tree, rebalance="monthly", cash_rate=None)
    r = pf.run(p)
    assert r.equity.index[1] == fake["B"].index[40]           # B's 40-day volatility, not just its 10-day return
    first = r.holdings.iloc[0]
    assert first["A"] > 0.1 and first["B"] > 0.1               # the first rebalance holds both, weighted
    assert r.equity.iloc[1] == pytest.approx(10_000)           # nothing was traded during the warm-up
    note = next(n for n in p.notes if n.startswith("Warm-up:"))
    assert "waited for B" in note


def test_headline_rule_is_the_same_everywhere(fake, tmp_path):
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 260)))
    fake["X"] = frame(c)
    r = engine.run(Strategy(universe=["X"], entry="close > sma(close, 100)", hold_bars=2))
    A = report.analyze(r, rf=0.0, sensitivity=False, mc=False, detail=False)
    h = report.headline(A)
    assert h["warmup"] and h["start_value"] == pytest.approx(A["stats"]["start_equity"]) and h["start_value"] > 10_000
    assert h["rule"].startswith(f"Stats from {A['stats']['start']} (value ${A['stats']['start_equity']:,.2f}) after the warm-up")
    assert h["rule"] in report.console_summary(A)
    report.write_outputs(A, tmp_path, excel=False)
    S = json.loads((tmp_path / "summary.json").read_text())
    assert S["headline"]["rule"] == h["rule"] and S["headline"]["final_value"] == pytest.approx(A["stats"]["end_equity"])
    page = (tmp_path / "report.html").read_text()
    assert "after warm-up" in page


# ------------------------------------------------------------ PV: common period, blended benchmarks, target vol

def test_common_period_is_the_strategies_not_a_young_benchmark():
    def nav(start, n):
        idx = pd.bdate_range(start, periods=n)
        return pd.Series(10_000 * np.cumprod(1 + np.random.default_rng(len(idx)).normal(0.0003, 0.01, n)), index=idx)
    runs = []
    for name, start in (("A", "2000-01-03"), ("B", "2001-01-02")):
        v = nav(start, 5000 - (0 if name == "A" else 260))
        res = SimpleNamespace(strategy=SimpleNamespace(name=name), equity=v)
        runs.append({"result": res, "nav": v, "first_bar": v.index[0], "benchmarks": {}})
    runs[0]["benchmarks"] = {"QQQ buy & hold": nav("2010-01-04", 2000)}
    C = report.common_window_stats(runs, 0.0)
    # everything under the common period covers it; a benchmark that starts later is listed on its own dates
    assert str(C["start"]).startswith("2001") and C["benchmarks_own_from"] == {"QQQ buy & hold": "2010-01-04"}
    assert set(C["columns"]) == {"A", "B"} and set(C["benchmarks_own"]) == {"QQQ buy & hold"}


def test_blended_benchmark_forms_and_monthly_rebalancing(fake):
    for form in ("60 SPY 40 AGG", "60% SPY / 40% AGG", "SPY 60 AGG 40", "SPY:0.6, AGG:0.4", "60/40 SPY/AGG",
                 {"SPY": 0.6, "AGG": 0.4}):
        assert metrics.benchmark_label(form) == "60% SPY / 40% AGG", form
    assert metrics.parse_blend("SPY") is None and metrics.benchmark_label("spy") == "SPY"
    with pytest.raises(ValueError):
        metrics.parse_blend("60 SPY 30 AGG")
    fake["P"] = frame(walk(8, 300), dividends=monthly_divs(300))
    fake["R"] = frame(walk(9, 300, vol=0.004))
    idx = fake["P"].index
    b = metrics.buy_and_hold("70 P 30 R", idx, 10_000)
    assert b.name == "70% P / 30% R"
    # independent: within each month the parts drift; at each month end the mix is reset to 70/30
    tp, tr = fake["P"]["adj_close"], fake["R"]["adj_close"]
    v, out = 10_000.0, []
    for _, g in pd.Series(range(len(idx)), index=idx).groupby(idx.to_period("M")):
        i0 = g.iloc[0] - 1 if g.iloc[0] > 0 else 0
        for i in g:
            out.append(v * (0.7 * tp.iloc[i] / tp.iloc[i0] + 0.3 * tr.iloc[i] / tr.iloc[i0]))
        v = out[-1]
    assert b.to_numpy() == pytest.approx(np.array(out), rel=1e-12)
    p = pf.Portfolio(tree={"asset": "P"}, benchmark={"P": 0.5, "R": 0.5}, cash_rate=None)
    p.validate()
    assert p.benchmark == "50% P / 50% R"
    A = report.analyze(pf.run(p), rf=0.0, sensitivity=False, mc=False, detail=False)
    assert A["primary_benchmark"] == "50% P / 50% R blend" and A["relative"]


def test_volatility_target_scales_the_mix(fake):
    fake["X"] = frame(walk(10, 700, vol=0.02))
    p = pf.Portfolio(tree={"asset": "X"}, rebalance="daily", cash_rate=None, target_vol=0.10)
    r = pf.run(p)
    ks = r.extras["vol_scale"]
    assert r.equity.index[1] == fake["X"].index[60]          # the 60-day lookback is part of the warm-up
    assert 0.3 < ks.median() < 0.7 and ks.max() <= 1.0 + 1e-12
    rets = fake["X"]["adj_close"].pct_change()
    d = ks.index[100]
    want = 0.10 / (rets.loc[:d].iloc[-60:].std() * np.sqrt(252))
    assert ks.loc[d] == pytest.approx(min(1.0, want))
    assert r.holdings["X"].max() <= 1.0 + 1e-9 and "Volatility target: 10.0%" in p.summary()
    lev = pf.Portfolio(tree={"asset": "X"}, rebalance="daily", cash_rate=None, target_vol=0.60, leverage=2,
                       maintenance_margin=0.25)
    r2 = pf.run(lev)
    assert r2.extras["vol_scale"].max() == pytest.approx(2.0) and "up to 2x" in lev.summary()
    with pytest.raises(ValueError):
        pf.Portfolio(tree={"asset": "X"}, target_vol=-0.1).validate()


@needs("SPYSIM", "EFASIM", "TLTSIM")
def test_tactical_model_starts_with_its_capital_after_warm_up():
    tree = {"if": 'tret(tr, 252) > tret(sym("TLTSIM").tr, 252)', "on": "SPYSIM",
            "then": {"asset": "SPYSIM"}, "else": {"asset": "TLTSIM"}}
    p = pf.Portfolio(tree=tree, start="1972-01-01", rebalance="monthly")
    r = pf.run(p)
    A = report.analyze(r, sensitivity=False, mc=False, detail=False)
    assert A["warmup_start"] is None and A["stats"]["start_equity"] == pytest.approx(10_000)
    assert report.headline(A)["start_value"] == pytest.approx(10_000)


# ------------------------------------------------------------ QC / TV

def test_series_in_a_lookback_position_is_an_error(fake):
    ns = expr.Namespace(frame(walk(11, 100)))
    for bad in ("sma(close, rsi(close, 2)) > 1", "ref(close, close) > 1", "ema(close, high, 5) > 1"):
        with pytest.raises(ValueError, match="lookback|offset"):
            expr.evaluate(bad, ns)
    assert expr.evaluate_value("sma(rsi(close, 2), 5)", ns).notna().any()
    assert expr.evaluate_value("sma(5, close)", ns).equals(expr.evaluate_value("sma(close, 5)", ns))


def test_crypto_read_from_a_us_ticker_uses_the_previous_crypto_day(fake):
    fake["X"] = frame(walk(12, 100))
    fake["BTC-USD"] = frame(walk(13, 150), freq="D")
    ns = expr.Namespace(fake["X"], ticker="X")
    v = expr.evaluate_value('sym("BTC-USD").close', ns)
    btc = fake["BTC-USD"]["close"]
    for d in fake["X"].index[5:20]:
        assert v[d] == btc.loc[d - pd.Timedelta(days=1)]
    assert expr.CRYPTO_LAG_NOTE in ns.notes
    own = expr.Namespace(fake["BTC-USD"], ticker="BTC-USD")
    fake["ETH-USD"] = frame(walk(14, 150), freq="D")
    w = expr.evaluate_value('sym("ETH-USD").close', own)
    assert w.to_numpy() == pytest.approx(fake["ETH-USD"]["close"].to_numpy()) and not own.notes
    p = pf.Portfolio(tree={"if": 'sym("BTC-USD").close > sma(sym("BTC-USD").close, 10)', "on": "X",
                           "then": {"asset": "X"}, "else": {"cash": True}}, cash_rate=None)
    pf.run(p)
    assert expr.CRYPTO_LAG_NOTE in p.notes


@pytest.mark.skipif(data.membership() is None or "SPY" not in AVAIL, reason="membership data not downloaded")
def test_ndx_filter_starts_when_membership_data_does():
    m0 = data.membership().index[0]
    tree = {"filter": {"select": "top", "n": 5, "by": "tret(tr, 126)"}, "universe": "NDX"}
    p = pf.Portfolio(tree=tree, start="1995-01-01", end=str((m0 + pd.Timedelta(days=400)).date()))
    r = pf.run(p)
    assert r.equity.index[1] >= m0 and r.holdings.drop(columns="cash").abs().sum(axis=1).iloc[5] > 0.9
    assert any(n.startswith(f"Start moved to {m0.date()}") and "you asked for 1995-01-01" in n for n in p.notes)


def test_gross_exposure_must_fit_the_maintenance_margin(fake):
    fake["U"] = frame(walk(15, 100))
    fake["B"] = frame(walk(16, 100))
    tree = {"weights": "specified", "w": [3.0, -2.0], "children": [{"asset": "U"}, {"asset": "B"}]}
    with pytest.raises(ValueError, match="gross exposure can reach 5x"):
        pf.Portfolio(tree=tree).validate()
    pf.Portfolio(tree=tree, maintenance_margin=0.15).validate()
    assert pf.max_gross({"if": "close > 1", "on": "U", "then": tree, "else": {"cash": True}}) == 5.0
    with pytest.raises(ValueError, match="gross exposure"):
        pf.Portfolio(tree={"weights": "specified", "w": [1.5, -0.5], "children": [{"asset": "U"}, {"asset": "B"}]},
                     leverage=2.5).validate()   # 2x gross weights times 2.5x = 5x


@needs("SPY")
def test_status_shows_the_last_bar_not_the_fetch_time():
    st = web.api_status()
    assert st["data"]["last_bar"] == str(data.load("SPY").index[-1].date())
    assert "last_bar" in (web.APP.read_text())


def test_sweep_labels_are_readable_for_portfolio_sweeps():
    labels, _, _ = research.expand("if QQQ {10,14} day RSI is greater than {75..85 step 5} hold UVXY otherwise hold QQQ")
    assert labels == ["QQQ {10,14} day RSI", "greater than {75..85 step 5}"]
    labels, _, _ = research.expand("hold the top {1,2} of SPY, QQQ, TLT by {3,6} month momentum")
    assert labels == ["top {1,2}", "{3,6} month momentum"]
    labels, _, _ = research.expand("if the {10,14}-day RSI of QQQ is above {75..85 step 5} hold UVXY, otherwise QQQ")
    assert labels == ["{10,14}-day RSI", "above {75..85 step 5}"]


# ------------------------------------------------------------ Build page in headless Chromium

def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = web.RUNS
    web.RUNS = tmp_path_factory.mktemp("runs")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    web.RUNS = old


NESTED = {"kind": "allocation", "rebalance": "daily", "tree": {"weights": "specified", "w": [0.5, 0.5], "children": [
    {"if": "rsi(close, 10) > 79", "on": "TQQQ", "then": {"asset": "UVXY"},
     "else": {"filter": {"select": "top", "n": 2, "by": "tret(tr, 20)", "weights": "inverse_vol", "lookback": 20},
              "universe": "children", "fallback": {"cash": True}, "children": [
                  {"asset": "SOXL"}, {"asset": "TQQQ"},
                  {"weights": "specified", "w": [0.6, 0.4], "children": [
                      {"asset": "SPY"}, {"if": 'rsi(sym("BIL").close, 60) > rsi(close, 60)', "on": "IEF",
                                         "then": {"asset": "TLT"}, "else": {"cash": True}}]}]}},
    {"asset": "BIL"}]}}


@pytest.mark.skipif(_chromium() is None or not {"SPY", "TLT"} <= AVAIL, reason="headless Chromium or data not available")
def test_build_page_rebalance_units_and_phone_width(site):
    from playwright.sync_api import sync_playwright
    SELS = "() => [...document.querySelectorAll('#tree > .node > .hd select')].map(s => s.value)"
    NOTE = "() => document.querySelector('#tree > .node > .nerr').textContent"
    root = "#tree > .node > .hd > select"
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(site + "/#build")
        pg.wait_for_function("document.querySelector('#tree .node')")
        reb = lambda: pg.eval_on_selector("#p_rebalance", "e => e.value")  # noqa: E731
        assert reb() == "monthly"                                  # fixed weights
        pg.select_option(root, "if")
        assert reb() == "daily" and "daily" in pg.inner_text("#rebalNote")
        pg.select_option(root, "weights")
        assert reb() == "monthly"
        pg.select_option(root, "filter")
        assert reb() == "daily"
        pg.select_option("#p_rebalance", "weekly")                 # the user's choice sticks
        pg.select_option(root, "weights")
        pg.select_option(root, "if")
        assert reb() == "weekly"
        # unit warnings follow every change of either side: price > SMA, then RSI on the left
        sels = pg.query_selector_all("#tree > .node > .hd select")
        assert pg.evaluate(SELS)[1:5] == ["close", ">", "ind", "sma"] and "Different units" not in pg.evaluate(NOTE)
        sels[1].select_option("rsi")
        # an oscillator on the left: the right side becomes a number (RSI against a price average means nothing)
        assert pg.evaluate(SELS)[1:4] == ["rsi", ">", "num"] and "Different units" not in pg.evaluate(NOTE)
        pg.query_selector_all("#tree > .node > .hd select")[3].select_option("ind")
        pg.query_selector_all("#tree > .node > .hd select")[4].select_option("sma")
        assert "Different units" in pg.evaluate(NOTE)
        pg.query_selector_all("#tree > .node > .hd select")[4].select_option("rsi")
        assert "Different units" not in pg.evaluate(NOTE)
        pg.query_selector_all("#tree > .node > .hd select")[4].select_option("tret")
        assert "Different units" in pg.evaluate(NOTE)
        assert errs == []
        pg.close()
        # phone width: nothing wider than the screen, even with a deeply nested tree
        pg = b.new_page(viewport={"width": 390, "height": 900})
        pg.goto(site + "/#build")
        pg.wait_for_function("document.querySelector('#tree .node')")
        assert pg.evaluate("document.documentElement.scrollWidth") <= 390
        pg.evaluate("s => { document.querySelector('#specJson').value = s; document.querySelector('#fromJson').click(); }",
                    json.dumps(NESTED))
        pg.wait_for_function("document.querySelectorAll('#tree .node').length > 10")
        assert pg.evaluate("document.documentElement.scrollWidth") <= 390
        assert reb() == "daily"                                    # a loaded spec keeps its own schedule
        for h in ("library", "backtest"):
            pg.goto(site + "/#" + h)
            pg.wait_for_timeout(500)
            assert pg.evaluate("document.documentElement.scrollWidth") <= 390, h
        b.close()
