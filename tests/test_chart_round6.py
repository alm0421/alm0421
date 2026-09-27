"""Round 6: the report's price chart charts every series a rule uses (no pane limit, weekly / monthly series as
steps with the engine's completed-period values), its TradingView-style tools (drawings, bar replay, candles /
line, log scale, pane fold / resize, legend values at the crosshair), and the site's Run / Share buttons not
losing the first click after a settings field is edited."""
import glob
import json
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr, report, web
from backtester.strategy import Strategy

AVAIL = set(data.available_tickers())

RICH = ('rsi(close, 2) < 30 and close > sma(close, 50) and weekly_rsi(14) > 20 and close > weekly_sma(4) - 1000 '
        'and ret(close, 5) < 0.2 and volatility(20) < 5 and obv() > -1e15 and atr(14) > 0 and adx(14) > 1 '
        'and stoch_k(14, 3) < 99 and cci(20) < 1000 and mfi(14) < 101 and willr(14) < 1 and close > bb_lower(20, 2) - 1000 '
        'and close > keltner_lower(20, 2) - 1000 and close > donchian_lower(20) - 1000 and close > supertrend(10, 3) - 1000 '
        'and close > sar() - 1000 and close > vwap(20) - 1000 and sym("YY").close > sma(sym("YY").close, 20) - 1000 '
        'and monthly_ret(1) > -1 and ibs < 1.1 and volume > 0')


def _frame(seed=1, n=420):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": rng.integers(5e5, 2e6, n).astype(float), "quote_close": c, "adj_close": c, "dividend": 0.0,
                         "split": 0.0}, index=idx)


@pytest.fixture
def fake(monkeypatch):
    frames = {"X": _frame(1), "YY": _frame(2)}
    def load(t):
        if data.canonical(t) not in frames:
            raise FileNotFoundError(t)
        return frames[data.canonical(t)]
    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {t: load(t) for t in ts})
    return frames


# ------------------------------------------------------------ layout from the rule's syntax tree

def test_chart_layout_covers_every_indicator_family():
    L = report.chart_layout([RICH, "rsi(close, 2) > 70 or bars_held >= 10"], "X")
    names = [p["name"] for p in L["panes"]]
    for want in ("RSI", "RSI (weekly)", "Return", "Volatility", "OBV", "ATR", "ADX / DMI", "Stochastic", "CCI", "MFI",
                 "Williams %R", "YY", "Return (monthly)", "IBS", "Volume"):
        assert want in names, want
    assert len(names) > 3                                          # no hard limit of 3 panes
    ov = L["overlays"]
    for want in ("sma(close, 50)", "weekly_sma(4)", "bb_lower(20, 2)", "bb_upper(20, 2)", "keltner_upper(20, 2)",
                 "donchian_upper(20)", "supertrend(10, 3)", "sar()", "vwap(20)"):
        assert want in ov, want
    assert {"weekly_rsi(14)", "weekly_sma(4)", "monthly_ret(1)"} <= L["step"]
    pane = {p["name"]: p for p in L["panes"]}
    assert pane["RSI"]["levels"] == [30, 70] and pane["RSI"]["range"] == [0, 100]
    assert pane["ADX / DMI"]["calls"] == ["adx(14)", "plus_di(14)", "minus_di(14)"]
    assert pane["YY"]["calls"] == ['sym("YY").close', 'sma(sym("YY").close, 20)']
    # the charted ticker's own sym() is its price: an overlay, not a pane
    own = report.chart_layout('sym("X").close > sma(sym("X").close, 20)', "X")
    assert own["panes"] == [] and own["overlays"] == ['sma(sym("X").close, 20)']
    # weekly(expr), averages of oscillators and another ticker's oscillator
    L2 = report.chart_layout('weekly(macd_hist()) > 0 and sma(rsi(close, 2), 5) < 20 and rsi(sym("YY").close, 14) < 70', "X")
    p2 = {p["name"]: p for p in L2["panes"]}
    assert "weekly(macd_hist())" in L2["step"] and p2["MACD (weekly)"]["levels"] == [0]
    assert p2["RSI"]["calls"] == ["sma(rsi(close, 2), 5)", "rsi(close, 2)"] and p2["RSI"]["levels"] == [20]
    assert p2["YY RSI"]["calls"] == ['rsi(sym("YY").close, 14)'] and p2["YY RSI"]["range"] == [0, 100]


def test_regex_pane_helpers_have_no_limit():
    rules = "rsi(2) < 10 and cci(20) < 0 and mfi(14) < 20 and willr(14) < -80 and atr(14) > 1 and zscore(close, 20) < -2"
    assert len(report.oscillator_panes(rules)) == 6
    panes = report.other_ticker_panes('sym("A").close > 1 and sym("B").close > 1 and sym("C").close > 1', "X")
    assert [p["name"] for p in panes] == ["A", "B", "C"]


def test_payload_values_are_the_engines_and_weekly_series_are_steps(fake):
    s = Strategy(cash_rate=None, universe=["X"], entry=RICH, exit_when="rsi(close, 2) > 70")
    res = engine.run(s)
    assert not res.trades.empty
    P = report.price_payload(res)["X"]
    idx = pd.to_datetime(P["dates"])
    ns = expr.Namespace(fake["X"], ticker="X")
    panes = {p["name"]: p for p in P["panes"]}
    assert len(panes) >= 15
    for call, got in [("weekly_sma(4)", P["overlays"]["weekly_sma(4)"]), ("weekly_rsi(14)", panes["RSI (weekly)"]["series"]["weekly_rsi(14)"]),
                      ("monthly_ret(1)", panes["Return (monthly)"]["series"]["monthly_ret(1)"]),
                      ("obv()", panes["OBV"]["series"]["obv()"])]:
        want = expr.evaluate_value(call, ns).reindex(idx).to_numpy()
        np.testing.assert_allclose(np.array(got, dtype=float), want, rtol=1e-6, atol=1e-3, equal_nan=True)
    assert {"weekly_sma(4)", "weekly_rsi(14)", "monthly_ret(1)"} <= set(P["step"])
    # held flat within a week: at most one change per week
    w = pd.Series(panes["RSI (weekly)"]["series"]["weekly_rsi(14)"], index=idx, dtype=float).dropna()
    changes = (w.diff().fillna(0) != 0).groupby(w.index.to_period("W-FRI")).sum()
    assert changes.max() <= 1
    # completed periods only: truncating the data does not change earlier values (no lookahead)
    cut = idx[len(idx) // 2]
    short = expr.Namespace(fake["X"][fake["X"].index <= cut], ticker="X")
    for call in ("weekly_rsi(14)", "weekly_sma(4)", "monthly_ret(1)"):
        a = report._values(call, ns, idx[idx <= cut])
        b = report._values(call, short, idx[idx <= cut])
        assert a == b, call
    assert panes["YY"]["series"]["YY close"] and panes["Volume"]["series"]["volume"]
    assert panes["IBS"]["range"] == [0, 1]


def test_report_html_carries_the_chart_tools(fake, tmp_path):
    res = engine.run(Strategy(cash_rate=None, universe=["X"], entry="rsi(close, 2) < 20", exit_when="rsi(close, 2) > 70"))
    report.write_outputs(report.analyze(res, rf=0.0, sensitivity=False, mc=False), tmp_path, excel=False)
    html = (tmp_path / "report.html").read_text()
    for needle in ('id="pxTools"', 'id="pxReplay"', 'id="rpPlay"', 'id="rpSpeed"', 'id="pxStyle"', 'id="pxScale"', 'data-v="hline"', 'data-v="trend"'):
        assert needle in html, needle


# ------------------------------------------------------------ headless Chromium

def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


def _ds(pg, k):
    return pg.evaluate("k => document.querySelector('#pxChart canvas').dataset[k]", k)


@pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")
def test_price_chart_tools_in_the_browser(fake, tmp_path):
    from playwright.sync_api import sync_playwright
    s = Strategy(cash_rate=None, universe=["X"], entry=RICH, exit_when="rsi(close, 2) > 70")
    report.write_outputs(report.analyze(engine.run(s), rf=0.0, sensitivity=False, mc=False), tmp_path, excel=False)
    url = (tmp_path / "report.html").as_uri()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url)
        pg.wait_for_function("document.querySelector('#pxChart canvas') && document.querySelector('#pxChart canvas').dataset.panes")
        assert int(_ds(pg, "panes")) >= 15
        pg.click("#pxRange button[data-v='0']")
        pg.evaluate("document.getElementById('pxChart').scrollIntoView({block: 'start'})")
        pg.wait_for_timeout(100)
        bx = lambda: pg.query_selector("#pxChart canvas.top").bounding_box()  # noqa: E731 (clicks on buttons scroll)
        box_x = bx()["x"]
        regions = json.loads(_ds(pg, "regions"))
        l, iw = json.loads(_ds(pg, "plot"))
        n_bars = int(_ds(pg, "lastBar")) + 1
        # legend: values follow the crosshair
        last = pg.inner_text("#pxLegend .ohlc")
        pg.mouse.move(box_x + l + iw * 0.3, bx()["y"] + regions[0][0] + 50)
        pg.wait_for_timeout(150)
        hovered = pg.inner_text("#pxLegend .ohlc")
        assert hovered != last and hovered[:10] < last[:10]
        assert "rsi(close, 2)" in pg.inner_text("#pxLegend")
        # fold the first pane by clicking its title, then unfold it
        y1 = regions[1][0]
        pg.mouse.click(box_x + l + 20, bx()["y"] + y1 + 10)
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.folded === '1'")
        assert json.loads(_ds(pg, "regions"))[1][1] == 20
        pg.mouse.click(box_x + l + 20, bx()["y"] + y1 + 10)
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.folded === '0'")
        # resize the price pane by dragging its bottom edge
        h0 = regions[0][1]
        edge = bx()["y"] + regions[0][0] + h0
        pg.mouse.move(box_x + l + 200, edge)
        pg.mouse.down()
        pg.mouse.move(box_x + l + 200, edge + 60, steps=4)
        pg.mouse.up()
        pg.wait_for_function("h => JSON.parse(document.querySelector('#pxChart canvas').dataset.regions)[0][1] === h", arg=h0 + 60)
        # drawing tools: a horizontal line, a trend line (drag), select + Delete key, Clear
        py = lambda: bx()["y"] + regions[0][0] + 80  # noqa: E731
        pg.click("#pxTools button[data-v='hline']")
        pg.mouse.click(box_x + l + 100, py())
        assert _ds(pg, "drawings") == "1"
        pg.click("#pxTools button[data-v='trend']")
        pg.mouse.move(box_x + l + 150, py() + 40)
        pg.mouse.down()
        pg.mouse.move(box_x + l + 300, py() + 120, steps=5)
        pg.mouse.up()
        assert _ds(pg, "drawings") == "2"
        pg.mouse.click(box_x + l + 225, py() + 80)         # on the trend line: selects it
        pg.keyboard.press("Delete")
        assert _ds(pg, "drawings") == "1"
        pg.click("#pxClear")
        assert _ds(pg, "drawings") == "0"
        # candles / line and log scale
        pg.click("#pxStyle button[data-v='line']")
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.style === 'line'")
        pg.click("#pxScale button[data-v='log']")
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.log === 'true'")
        # bar replay: fewer bars and trades, step, play / pause, speed, exit
        markers_all = int(_ds(pg, "markers"))
        pg.click("#pxReplay")
        pg.wait_for_function("+document.querySelector('#pxChart canvas').dataset.lastBar < %d" % (n_bars - 1))
        k0 = int(_ds(pg, "lastBar"))
        assert int(_ds(pg, "markers")) < markers_all
        assert pg.is_visible("#pxReplayBar")
        pg.click("#rpStep")
        pg.wait_for_function("+document.querySelector('#pxChart canvas').dataset.lastBar === %d" % (k0 + 1))
        pg.select_option("#rpSpeed", "60")
        pg.click("#rpPlay")
        pg.wait_for_function("+document.querySelector('#pxChart canvas').dataset.lastBar > %d" % (k0 + 10))
        pg.click("#rpPlay")                                   # pause
        # the canvas's lastBar is written on the next animation frame: let a frame already queued draw first
        pg.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        k1 = int(_ds(pg, "lastBar"))
        pg.wait_for_timeout(300)
        assert int(_ds(pg, "lastBar")) == k1
        assert "bar %d of %d" % (k1 + 1, n_bars) in pg.inner_text("#rpDate")
        pg.click("#rpExit")
        pg.wait_for_function("+document.querySelector('#pxChart canvas').dataset.lastBar === %d" % (n_bars - 1))
        assert not pg.is_visible("#pxReplayBar")
        assert errs == []
        pg.close()
        # phone width: no horizontal scroll, no errors, the chart fits
        pg = b.new_page(viewport={"width": 390, "height": 844})
        errs.clear()
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url)
        pg.wait_for_function("document.querySelector('#pxChart canvas') && document.querySelector('#pxChart canvas').dataset.panes")
        assert pg.evaluate("document.documentElement.scrollWidth") <= 390
        assert pg.evaluate("document.querySelector('#pxChart canvas').getBoundingClientRect().right") <= 390
        pg.click("#pxReplay")
        pg.click("#rpStep")
        assert pg.evaluate("document.documentElement.scrollWidth") <= 390
        assert errs == []
        b.close()


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = web.RUNS
    web.RUNS = tmp_path_factory.mktemp("runs")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    web.RUNS = old


@pytest.mark.skipif(_chromium() is None or "SPY" not in AVAIL, reason="headless Chromium or data not available")
def test_first_click_after_editing_a_setting_is_not_lost(site):
    """The settings field commits (change) on the mousedown that blurs it; re-interpreting then used to clear the
    notes at once, the button moved up and the mouseup missed it: no /api/run. The parse reply is held back here
    until the button is pressed, so the swap also lands mid-click."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        for width in (1280, 390):
            pg = b.new_page(viewport={"width": width, "height": 900})
            errs, runs = [], []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("request", lambda r: runs.append(r.url) if "/api/run" in r.url else None)
            pg.goto(site + "/#backtest")
            pg.fill("#text", "buy SPY when RSI(2) is below 10, hold 3 days")
            pg.wait_for_function("document.getElementById('interp').textContent.includes('SPY')")
            pg.wait_for_timeout(600)
            pg.click("#p-backtest details summary")
            for btn in ("#runBtn", "#shareBtn"):
                runs.clear()
                pg.fill("#o_capital", "25000" if btn == "#runBtn" else "30000")
                held = []
                pg.route("**/api/parse*", lambda route: held.append(route))
                pg.eval_on_selector(btn, "e => e.scrollIntoView({block: 'center'})")   # the field keeps the focus
                bb = pg.query_selector(btn).bounding_box()
                pg.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
                pg.mouse.down()                                   # blurs the field: change -> interpret()
                pg.wait_for_timeout(200)
                assert held, "the settings change re-interprets the sentence"
                for r in held:
                    r.continue_()
                pg.wait_for_timeout(400)                          # the reply arrives while the button is held
                pg.mouse.up()
                pg.wait_for_timeout(300)
                pg.unroute("**/api/parse*")
                assert runs, (width, btn)
                pg.wait_for_function("!document.getElementById('runBtn').disabled", timeout=60_000)
            assert errs == []
            pg.close()
        # Build page: editing a block's field and clicking Run straight away
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        runs = []
        pg.on("request", lambda r: runs.append(r.url) if "/api/run" in r.url else None)
        pg.goto(site + "/#build")
        pg.evaluate("s => { document.querySelector('#specJson').value = s; document.querySelector('#fromJson').click(); }",
                    json.dumps({"kind": "allocation", "tree": {"weights": "specified", "w": [0.6, 0.3],
                                                               "children": [{"asset": "SPY"}, {"asset": "SPY"}]}}))
        pg.wait_for_function("document.querySelectorAll('#tree input[type=number]').length >= 2")
        inp = pg.query_selector_all("#tree input[type=number]")[1]
        inp.fill("40")
        bb = pg.query_selector("#buildRun").bounding_box()
        pg.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        pg.wait_for_timeout(500)
        assert runs
        b.close()
