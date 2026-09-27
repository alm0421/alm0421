"""TradingView review round 11, the report's price chart: Fibonacci retracement, rectangle and ray drawings (kept per
ticker in the browser like the others), an "add indicator" menu (SMA / EMA / Bollinger Bands overlays, RSI / MACD
panes, computed in the browser for reading only), and a daily / weekly / monthly bar-period toggle with candles
resampled from the daily bars. Headless Chromium: no console errors, and a phone-width page does not scroll sideways."""
import glob
import json

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, report
from backtester.strategy import Strategy


def _frame(seed=1, n=420):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": rng.integers(5e5, 2e6, n).astype(float), "quote_close": c, "adj_close": c, "dividend": 0.0,
                         "split": 0.0}, index=idx)


@pytest.fixture
def fake(monkeypatch):
    frames = {"X": _frame(1)}

    def load(t):
        if data.canonical(t) not in frames:
            raise FileNotFoundError(t)
        return frames[data.canonical(t)]
    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {t: load(t) for t in ts})
    return frames


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


def _ds(pg, k):
    return pg.evaluate("k => document.querySelector('#pxChart canvas').dataset[k]", k)


def _report(tmp_path):
    s = Strategy(cash_rate=None, universe=["X"], entry="rsi(close, 2) < 20", exit_when="rsi(close, 2) > 70")
    report.write_outputs(report.analyze(engine.run(s), rf=0.0, sensitivity=False, mc=False), tmp_path, excel=False)
    return (tmp_path / "report.html")


def test_report_html_carries_the_new_chart_tools(fake, tmp_path):
    html = _report(tmp_path).read_text()
    for needle in ('data-v="fib"', 'data-v="rect"', 'data-v="ray"', 'id="pxIndBtn"', 'id="pxIndMenu"', 'id="pxView"',
                   'value="macd"', 'value="bb"'):
        assert needle in html, needle


@pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")
def test_drawings_indicators_and_bar_period_in_the_browser(fake, tmp_path):
    from playwright.sync_api import sync_playwright
    url = _report(tmp_path).as_uri()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url)
        pg.wait_for_function("document.querySelector('#pxChart canvas') && document.querySelector('#pxChart canvas').dataset.panes")
        pg.evaluate("localStorage.clear()")
        pg.click("#pxRange button[data-v='0']")
        pg.evaluate("document.getElementById('pxChart').scrollIntoView({block: 'start'})")
        pg.wait_for_timeout(100)
        bx = lambda: pg.query_selector("#pxChart canvas.top").bounding_box()  # noqa: E731
        regions = json.loads(_ds(pg, "regions"))
        l, iw = json.loads(_ds(pg, "plot"))
        py = lambda: bx()["y"] + regions[0][0] + 60  # noqa: E731

        def drag(tool, x0, y0, x1, y1):
            pg.click(f"#pxTools button[data-v='{tool}']")
            x, y = bx()["x"], py()
            pg.mouse.move(x + l + x0, y + y0)
            pg.mouse.down()
            pg.mouse.move(x + l + x1, y + y1, steps=5)
            pg.mouse.up()
        # three new two-point drawings
        drag("fib", 100, 0, 250, 150)
        assert _ds(pg, "drawings") == "1"
        drag("rect", 400, 10, 520, 90)
        assert _ds(pg, "drawings") == "2"
        drag("ray", 600, 20, 650, 60)
        assert _ds(pg, "drawings") == "3"
        saved = json.loads(pg.evaluate("localStorage.getItem('bt-draw|X')"))
        assert sorted(d["t"] for d in saved) == ["fib", "ray", "rect"]
        assert all(d["d1"] and d["d2"] for d in saved)        # stored by date, like trend lines
        # the ray goes past its second point to the edge: a click beyond it selects it; Delete removes it
        x, y = bx()["x"], py()
        pg.mouse.click(x + l + 700, y + 20 + (700 - 600) * 40 / 50)
        pg.keyboard.press("Delete")
        assert _ds(pg, "drawings") == "2"
        # a click on the rectangle's edge selects it
        pg.mouse.click(x + l + 460, y + 10)
        pg.keyboard.press("Delete")
        assert _ds(pg, "drawings") == "1"
        # the Fibonacci levels: a click on the 0.5 line selects the drawing
        pg.mouse.click(x + l + 175, y + 75)
        pg.keyboard.press("Delete")
        assert _ds(pg, "drawings") == "0"
        drag("fib", 100, 0, 250, 150)
        # drawings survive a reload (per ticker in this browser)
        pg.reload()
        pg.wait_for_function("document.querySelector('#pxChart canvas') && document.querySelector('#pxChart canvas').dataset.panes")
        assert _ds(pg, "drawings") == "1"
        panes0 = int(_ds(pg, "panes"))
        legend0 = pg.inner_text("#pxLegend")
        # add indicators: an SMA overlay, Bollinger Bands, an RSI pane and a MACD pane
        pg.click("#pxIndBtn")
        assert pg.is_visible("#pxIndMenu")
        pg.select_option("#pxIndType", "sma")
        pg.fill("#pxIndP0", "30")
        pg.click("#pxIndAdd")
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.inds === '1'")
        assert "SMA(30)" in pg.inner_text("#pxLegend") and "SMA(30)" not in legend0
        pg.select_option("#pxIndType", "bb")
        pg.click("#pxIndAdd")
        pg.select_option("#pxIndType", "rsi")
        pg.fill("#pxIndP0", "7")
        pg.click("#pxIndAdd")
        pg.select_option("#pxIndType", "macd")
        pg.click("#pxIndAdd")
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.inds === '4'")
        assert int(_ds(pg, "panes")) == panes0 + 2
        lg = pg.inner_text("#pxLegend")
        for k in ("BB(20, 2) upper", "RSI(7)", "MACD(12, 26)", "signal(9)"):
            assert k in lg, k
        # the values are the textbook ones: the SMA at the last bar
        closes = fake["X"]["close"].round(4)
        want = closes.iloc[-30:].mean()
        got = pg.evaluate("(() => { const s = [...document.querySelectorAll('#pxLegend span')].find(x => x.textContent.startsWith('SMA(30)'));"
                          " return s.querySelector('b').textContent; })()")
        assert abs(float(got.replace(",", "")) - want) < 0.01
        assert pg.evaluate("document.querySelectorAll('#pxIndList .indrow').length") == 4
        # remove one; the list and the chart follow; the choice survives a reload
        pg.click("#pxIndList .indrow:first-child button")
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.inds === '3'")
        pg.reload()
        pg.wait_for_function("document.querySelector('#pxChart canvas') && document.querySelector('#pxChart canvas').dataset.inds === '3'")
        # weekly and monthly candles: fewer bars, the drawings and trades stay
        bars_d = int(_ds(pg, "bars"))
        pg.click("#pxView button[data-v='W']")
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.view === 'W'")
        bars_w = int(_ds(pg, "bars"))
        assert bars_d / 6 < bars_w < bars_d / 4
        assert _ds(pg, "drawings") == "1"
        pg.click("#pxRange button[data-v='0']")
        pg.wait_for_timeout(100)
        assert int(_ds(pg, "markers")) > 0
        pg.click("#pxView button[data-v='M']")
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.view === 'M'")
        bars_m = int(_ds(pg, "bars"))
        assert bars_d / 25 < bars_m < bars_d / 18
        # the monthly bar (the legend shows the last one) holds the month's first open, high, low and last close
        import re
        head = pg.inner_text("#pxLegend .ohlc")
        f = fake["X"].round(4)
        last = f[f.index.to_period("M") == f.index[-1].to_period("M")]
        vals = dict(re.findall(r"\b([OHLC]) ([\d,.]+)", head))
        assert head.startswith(str(f.index[-1].date()))
        for k, v in (("O", last["open"].iloc[0]), ("H", last["high"].max()), ("L", last["low"].min()), ("C", last["close"].iloc[-1])):
            assert abs(float(vals[k].replace(",", "")) - v) < 0.01, (k, vals[k], v)
        pg.click("#pxView button[data-v='D']")
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.view === 'D'")
        assert int(_ds(pg, "bars")) == bars_d
        assert errs == []
        pg.close()
        # phone width: the menu and the toolbar fit, nothing scrolls sideways
        pg = b.new_page(viewport={"width": 390, "height": 844})
        errs.clear()
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url)
        pg.wait_for_function("document.querySelector('#pxChart canvas') && document.querySelector('#pxChart canvas').dataset.panes")
        pg.click("#pxIndBtn")
        assert pg.evaluate("document.documentElement.scrollWidth") <= 390
        assert pg.evaluate("document.getElementById('pxIndMenu').getBoundingClientRect().right") <= 390
        pg.click("#pxView button[data-v='W']")
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.view === 'W'")
        assert pg.evaluate("document.documentElement.scrollWidth") <= 390
        assert errs == []
        b.close()
