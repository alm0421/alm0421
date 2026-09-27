"""The Funds page (screener, detail, comparison) in headless Chromium at desktop and phone width (390px): no page
errors, no console errors, no failed requests, no horizontal page scroll.

The server listens on 127.0.0.1:8862; a stale one left on that port is stopped with `fuser -k 8862/tcp`."""
import glob
import shutil
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from backtester import data, funds, web

PORT = 8862


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


pytestmark = pytest.mark.skipif(_chromium() is None or not {"VTI", "BND", "VXUS"} <= set(data.available_tickers()),
                                reason="headless Chromium or price data not available")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = (web.RUNS, funds.WARM, funds.CACHE_FILE)
    tmp = tmp_path_factory.mktemp("funds_site")
    web.RUNS, funds.WARM, funds.CACHE_FILE = tmp / "runs", False, tmp / "fund_stats.json"
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
    web.RUNS, funds.WARM, funds.CACHE_FILE = old


@pytest.mark.parametrize("width", [1280, 390])
def test_funds_page(site, width):
    from playwright.sync_api import sync_playwright
    errs: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": width, "height": 900})
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg.on("console", lambda m: errs.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
        pg.on("response", lambda r: errs.append(f"HTTP {r.status} {r.request.method} {r.url}") if r.status >= 400 else None)
        pg.set_default_timeout(120_000)
        pg.goto(site + "/#funds")
        pg.wait_for_selector("#p-funds:not(.hide)")
        pg.wait_for_selector("#fTable tr:nth-child(3)")
        total = pg.inner_text("#fTitle")
        # the screener: a search, a numeric filter, sorting by a column
        pg.fill("#f_q", "VTI")
        pg.wait_for_function("document.querySelector('#fTitle').innerText !== " + repr(total))
        assert "VTI" in pg.inner_text("#fTable")
        pg.fill("#f_q", "")
        pg.fill("#f_years", "15")
        pg.click("#fTable th:nth-child(11)")          # 5y
        assert "▼" in pg.inner_text("#fTable tr:first-child") or "▲" in pg.inner_text("#fTable tr:first-child")
        pg.click("#fReset")
        # tick funds to compare, then compare them
        pg.fill("#f_q", "VTI")
        pg.check("#fTable input[aria-label='compare VTI']")
        pg.fill("#f_q", "")
        pg.fill("#f_cmp", pg.input_value("#f_cmp") + " BND VXUS")
        pg.click("#fCmpRun")
        pg.wait_for_selector("#fCmpOut:not(.hide)")
        pg.wait_for_selector("#fGrowth svg")
        assert pg.locator("#fAnnual tr").count() > 5 and pg.locator("#fCorr tr").count() == 4
        # a fund's detail card
        pg.fill("#f_q", "BND")
        pg.click("#fTable a:text-is('BND')")
        pg.wait_for_selector("#fDetail h2")
        assert "BND" in pg.inner_text("#fDetail h2")
        # too few funds: a message, no request error
        pg.fill("#f_cmp", "VTI")
        pg.click("#fCmpRun")
        assert "2 to 6" in pg.inner_text("#fCmpStatus")
        assert pg.evaluate("document.documentElement.scrollWidth") <= width + 1
        b.close()
    assert not errs, errs
