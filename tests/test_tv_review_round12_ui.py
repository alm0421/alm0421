"""Round 12 UI (headless Chromium): the Build page's "TradingView-compatible: on" applies the mode's defaults (next-open
fills, no interest on cash, no borrow fee, no margin calls), negative percentages are refused in the field's own unit,
the live parse never logs a console error while typing, and the report's ＋ Indicator menu closes on an outside click,
Escape or after adding an indicator (it no longer covers the Replay buttons)."""
import glob
import shutil
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, report, web
from backtester.strategy import Strategy

PORT = 8824


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


browser = pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")


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


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = web.RUNS
    web.RUNS = tmp_path_factory.mktemp("runs12")
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


def _page(p, errs, width=1280):
    b = p.chromium.launch(executable_path=_chromium())
    pg = b.new_page(viewport={"width": width, "height": 900})
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    return b, pg


@browser
def test_build_page_tradingview_mode_applies_its_defaults(site):
    from playwright.sync_api import sync_playwright
    errs: list[str] = []
    with sync_playwright() as p:
        b, pg = _page(p, errs)
        pg.set_default_timeout(60_000)
        pg.goto(site + "/#build")
        pg.click("#buildMode button[data-v='signal']")
        pg.wait_for_selector("#bSignal:not(.hide)")
        pg.fill("#s_universe", "SPY")
        pg.fill("#s_hold_bars", "3")
        assert pg.input_value("#s_cash_rate") == "tbill" and pg.input_value("#s_exit_when_fill") == "close"
        pg.select_option("#s_tv_compat", "true")
        assert pg.input_value("#s_cash_rate") == "0"
        assert pg.input_value("#s_exit_when_fill") == "next_open" and pg.input_value("#s_entry_fill") == "next_open"
        assert pg.input_value("#s_borrow_fee") == "0"
        note = pg.inner_text("#s_tvNote")
        assert "TradingView-compatible" in note and "cash interest: none" in note and "no margin calls" in note
        # the live interpretation is the TradingView-compatible one
        pg.wait_for_function("document.getElementById('buildInterp').textContent.includes('TradingView-compatible returns')")
        txt = pg.inner_text("#buildInterp")
        assert "Buy SPY at the next day's open" in txt and "cash earns nothing" in txt
        # back off: the default mode's fills and interest
        pg.select_option("#s_tv_compat", "false")
        assert pg.input_value("#s_cash_rate") == "tbill" and pg.input_value("#s_exit_when_fill") == "close"
        assert pg.input_value("#s_entry_fill") == "close"
        assert errs == [], errs
        b.close()


@browser
def test_build_page_typing_logs_no_console_errors_and_says_percent(site):
    from playwright.sync_api import sync_playwright
    errs: list[str] = []
    bad: list[str] = []
    with sync_playwright() as p:
        b, pg = _page(p, errs)
        pg.on("response", lambda r: bad.append(f"{r.status} {r.url}") if r.status >= 400 else None)
        pg.set_default_timeout(60_000)
        pg.goto(site + "/#build")
        pg.click("#buildMode button[data-v='signal']")
        pg.wait_for_selector("#bSignal:not(.hide)")
        pg.fill("#s_universe", "SPY")
        pg.fill("#s_hold_bars", "3")
        for t in ["rsi(", "close >", "__import__('os')", "foo(1) > 2", "close > open and open > close", "rsi(close, 2) < 10"]:
            pg.fill("#s_entry", t)
            pg.wait_for_timeout(700)
        pg.wait_for_function("document.getElementById('buildInterp').textContent.includes('rsi(close, 2) < 10')")
        pg.fill("#s_stop_loss", "-5")
        pg.wait_for_function("document.getElementById('buildInterp').textContent.includes('must be positive')")
        txt = pg.inner_text("#buildInterp")
        assert "Stop loss (%): the value must be positive, in percent (e.g. 5 for 5%)" in txt and "0.05" not in txt
        pg.click("#buildRun")
        assert "must be positive, in percent" in pg.inner_text("#buildStatus")
        assert errs == [] and bad == [], (errs, bad)
        b.close()


def _report(tmp_path):
    s = Strategy(cash_rate=None, universe=["X"], entry="rsi(close, 2) < 20", exit_when="rsi(close, 2) > 70")
    report.write_outputs(report.analyze(engine.run(s), rf=0.0, sensitivity=False, mc=False), tmp_path, excel=False)
    return tmp_path / "report.html"


@browser
def test_indicator_menu_closes_on_outside_click_escape_and_add(fake, tmp_path):
    from playwright.sync_api import sync_playwright
    url = _report(tmp_path).as_uri()
    errs: list[str] = []
    with sync_playwright() as p:
        b, pg = _page(p, errs)
        pg.goto(url)
        pg.wait_for_function("document.querySelector('#pxChart canvas') && document.querySelector('#pxChart canvas').dataset.panes")
        menu = "#pxIndMenu"
        # Escape
        pg.click("#pxIndBtn")
        assert pg.is_visible(menu)
        pg.keyboard.press("Escape")
        assert not pg.is_visible(menu) and pg.get_attribute("#pxIndBtn", "aria-expanded") == "false"
        # a click outside (on the Replay button, which the menu used to cover) closes it and reaches the button
        pg.click("#pxIndBtn")
        assert pg.is_visible(menu)
        pg.click("#pxReplay")
        assert not pg.is_visible(menu)
        assert pg.is_visible("#pxReplayBar")
        pg.click("#pxReplay")
        # clicks inside the menu keep it open; adding an indicator closes it
        pg.click("#pxIndBtn")
        pg.select_option("#pxIndType", "ema")
        pg.fill("#pxIndP0", "15")
        assert pg.is_visible(menu)
        pg.click("#pxIndAdd")
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.inds === '1'")
        assert not pg.is_visible(menu)
        # removing one from the list (inside the menu) keeps the menu open
        pg.click("#pxIndBtn")
        pg.click("#pxIndList .indrow:first-child button")
        pg.wait_for_function("document.querySelector('#pxChart canvas').dataset.inds === '0'")
        assert pg.is_visible(menu)
        assert errs == [], errs
        b.close()
