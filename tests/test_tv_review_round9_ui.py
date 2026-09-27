"""Round 9 UI: chart drawings are kept per ticker (they carry across reports and runs), and a Build run keeps the
loaded spec's notes and description."""
import glob
import json
import shutil
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, report, web
from backtester.strategy import Strategy

AVAIL = set(data.available_tickers())
PORT = 8851


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


browser = pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")


def _frame(seed=1, n=420):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0, 0.006, n))
    hi = np.maximum(o, c) * 1.004
    lo = np.minimum(o, c) * 0.996
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": 1e6, "adj_close": c,
                         "quote_close": c, "dividend": 0.0}, index=idx)


@pytest.fixture
def fake(monkeypatch):
    frames = {"X": _frame()}
    real = data.load
    monkeypatch.setattr(data, "load", lambda t: frames[t] if t in frames else real(t))
    monkeypatch.setattr(data, "load_many", lambda ts: {t: frames[t] for t in ts})
    return frames


def _ds(pg, k):
    return pg.evaluate("k => document.querySelector('#pxChart canvas').dataset[k]", k)


@browser
def test_drawings_carry_across_reports_of_the_same_ticker(fake, tmp_path):
    from playwright.sync_api import sync_playwright
    urls = []
    for n, rule in enumerate(("rsi(close, 2) < 15", "rsi(close, 3) < 20")):
        s = Strategy(cash_rate=None, universe=["X"], entry=rule, hold_bars=3)
        out = tmp_path / f"r{n}"
        report.write_outputs(report.analyze(engine.run(s), rf=0.0, sensitivity=False, mc=False), out, excel=False)
        urls.append((out / "report.html").as_uri())
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        ctx = b.new_context(viewport={"width": 1280, "height": 900})
        pg = ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        ready = "document.querySelector('#pxChart canvas') && document.querySelector('#pxChart canvas').dataset.drawings !== undefined"
        pg.goto(urls[0])
        pg.wait_for_function(ready)
        pg.evaluate("document.getElementById('pxChart').scrollIntoView({block: 'start'})")
        pg.wait_for_timeout(100)
        box = pg.query_selector("#pxChart canvas.top").bounding_box()
        regions = json.loads(_ds(pg, "regions"))
        l, _ = json.loads(_ds(pg, "plot"))
        pg.click("#pxTools button[data-v='hline']")
        box = pg.query_selector("#pxChart canvas.top").bounding_box()   # the click may scroll
        pg.mouse.click(box["x"] + l + 100, box["y"] + regions[0][0] + 80)
        assert _ds(pg, "drawings") == "1"
        stored = pg.evaluate("localStorage.getItem('bt-draw|X')")
        assert stored and json.loads(stored)[0]["t"] == "h"
        # a trend line is stored by date, so another run's chart places it on the same days
        pg.evaluate("""localStorage.setItem('bt-draw|X', JSON.stringify(JSON.parse(localStorage.getItem('bt-draw|X'))
            .concat([{t: 'tr', d1: '2019-06-03', v1: 100, d2: '2019-09-03', v2: 110}])))""")
        pg2 = ctx.new_page()
        pg2.on("pageerror", lambda e: errs.append(str(e)))
        pg2.goto(urls[1])
        pg2.wait_for_function(ready)
        assert _ds(pg2, "drawings") == "2"
        assert not errs, errs
        b.close()


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = web.RUNS
    web.RUNS = tmp_path_factory.mktemp("runs9")
    try:
        srv = web.ThreadingHTTPServer(("127.0.0.1", PORT), web.Handler)
    except OSError:
        if shutil.which("fuser"):
            subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True)
            time.sleep(1.0)
        srv = web.ThreadingHTTPServer(("127.0.0.1", PORT), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{PORT}"
    srv.shutdown()
    srv.server_close()
    web.RUNS = old


@browser
@pytest.mark.skipif(not {"SPY"} <= AVAIL, reason="price data not downloaded")
def test_build_run_keeps_notes_and_description(site):
    from playwright.sync_api import sync_playwright
    text = "buy SPY when RSI(2) is below 10, hold 3 days, since 2018"
    errs: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.set_default_timeout(180_000)
        pg.goto(site + "/#backtest")
        pg.fill("#text", text)
        pg.wait_for_function("document.querySelector('#interp').textContent.includes('rsi')")
        pg.click("#toBuild")
        pg.wait_for_selector("#bSignal:not(.hide)")
        time.sleep(1.0)
        with pg.expect_request(lambda r: "/api/run" in r.url) as rq:
            pg.click("#buildRun")
        body = json.loads(rq.value.post_data)
        assert body["spec"]["description"] == text
        assert any(n.startswith("Entry timing not stated") for n in body["spec"]["notes"])
        pg.wait_for_function("document.querySelector('#buildStatus').textContent.includes('CAGR')")
        # once a rule is edited, the sentence no longer describes it: not sent
        pg.fill("#s_entry", "rsi(close, 2) < 15")
        with pg.expect_request(lambda r: "/api/run" in r.url) as rq:
            pg.click("#buildRun")
        body = json.loads(rq.value.post_data)
        assert "description" not in body["spec"] and "notes" not in body["spec"]
        assert not errs, errs
        b.close()


@browser
@pytest.mark.skipif(not {"SPY"} <= AVAIL, reason="price data not downloaded")
def test_pasted_pine_script_runs_with_the_ticker_setting(site):
    from pathlib import Path
    from playwright.sync_api import sync_playwright
    script = (Path(__file__).parent / "fixtures" / "pine" / "ma_cross.pine").read_text()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.set_default_timeout(180_000)
        pg.goto(site + "/#backtest")
        pg.fill("#text", script)
        pg.wait_for_function("document.querySelector('#interp').textContent.includes('Which ticker')")
        pg.evaluate("document.querySelector('#interp').closest('section') && 0")
        pg.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
        pg.fill("#o_ticker", "SPY")
        pg.dispatch_event("#o_ticker", "change")
        pg.wait_for_function("document.querySelector('#interp').textContent.includes('crossover(sma(close, 10), sma(close, 30))')")
        assert "Pine import" in pg.text_content("#interpNotes")
        b.close()
