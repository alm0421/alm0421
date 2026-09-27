"""The Funds page's unknown-expense-ratio handling and the Data page's ticker directory in headless Chromium, at desktop
and phone width: no page errors, no failed requests, no horizontal page scroll.

The server listens on 127.0.0.1:8871; a stale one left on that port is stopped with `fuser -k 8871/tcp`."""
import glob
import shutil
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from backtester import data, funds, web

PORT = 8871


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


pytestmark = pytest.mark.skipif(_chromium() is None or not {"VTI", "BND", "VFINX"} <= set(data.available_tickers()),
                                reason="headless Chromium or price data not available")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = (web.RUNS, funds.WARM, funds.CACHE_FILE)
    tmp = tmp_path_factory.mktemp("funds_dir_site")
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
def test_funds_unknowns_and_directory(site, width):
    from playwright.sync_api import sync_playwright
    errs: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": width, "height": 900})
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg.on("console", lambda m: errs.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
        pg.on("response", lambda r: errs.append(f"HTTP {r.status} {r.request.method} {r.url}") if r.status >= 400 else None)
        pg.set_default_timeout(120_000)
        t0 = time.time()
        pg.goto(site + "/#funds")
        pg.wait_for_selector("#fTable tr:nth-child(3)")
        first_paint = time.time() - t0
        # mutual funds under 0.2%: some pass, the ones without an expense ratio are counted, and can be included
        pg.select_option("#f_kind", "Mutual fund")
        pg.fill("#f_er", "0.2")
        pg.wait_for_function("document.querySelector('#fUnknown').innerText.includes('no expense ratio yet')")
        n = int(pg.inner_text("#fTitle").split(":")[1].split(" of ")[0].replace(",", ""))
        assert n > 10
        assert "VFINX" in pg.inner_text("#fTable") or "left out" in pg.inner_text("#fUnknown")
        pg.check("#f_unk")
        n2 = int(pg.inner_text("#fTitle").split(":")[1].split(" of ")[0].replace(",", ""))
        assert n2 > n and "shown" in pg.inner_text("#fUnknown")
        pg.click("#fReset")
        assert not pg.is_checked("#f_unk")
        # the directory on the Data page
        pg.click("a[href='#data']")
        pg.wait_for_selector("#p-data:not(.hide)")
        pg.fill("#tkSearch", "berkshire")
        pg.wait_for_function("document.querySelector('#tkList').innerText.includes('BRK-B')")
        assert "S&P 500" in pg.inner_text("#tkList")
        pg.fill("#tkSearch", "vanguard wellington")
        pg.wait_for_function("document.querySelector('#tkList').innerText.includes('VWELX')")
        pg.select_option("#tkKind", "ETF")
        pg.wait_for_function("!document.querySelector('#tkList').innerText.includes('VWELX')")
        assert pg.evaluate("document.documentElement.scrollWidth") <= width + 1
        b.close()
    assert not errs, errs
    assert first_paint < 30, first_paint
