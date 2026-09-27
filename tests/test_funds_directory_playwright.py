"""The Funds page's unknown-expense-ratio handling and the Data page's ticker directory in headless Chromium, at desktop
and phone width: no page errors, no failed requests, no horizontal page scroll.

The server listens on 127.0.0.1:8871; a stale one left on that port is stopped with `fuser -k 8871/tcp`.

The data job fills in expense ratios in batches, so whether any real mutual fund still lacks one depends on the day's
data (after one refresh none did, and the unknown-ER path went untested; after the next, one did). The site therefore
serves the real fund table with the expense ratio of a few mutual funds (UNKNOWN_ER) blanked, so that path is always
exercised, and the test counts the unknowns it served (SERVED) instead of assuming the real table has none."""
import glob
import shutil
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from backtester import data, funds, web

PORT = 8871
UNKNOWN_ER = 3      # mutual funds served without an expense ratio (the ones with the lowest ratios, after VFINX)
SERVED = {}         # "unknown": mutual funds the site serves without an expense ratio (the blanked ones + the real ones)


def _with_unknown_er(real):
    """api_funds with the expense ratio of UNKNOWN_ER low-cost mutual funds removed (a copy: the real table is kept)."""
    def api_funds(query):
        r = dict(real(query))
        funds_ = [dict(f) for f in r["funds"]]
        mf = sorted((f for f in funds_ if f.get("type") == "Mutual fund" and f.get("ticker") != "VFINX"
                     and isinstance(f.get("expense_ratio"), (int, float))), key=lambda f: (f["expense_ratio"], f["ticker"]))
        for f in mf[:UNKNOWN_ER]:
            f["expense_ratio"] = None
            f.pop("er_source", None)
        # a real fund may still lack one on the day's data (the data job fills them in batches): counted, not assumed
        SERVED["unknown"] = sum(1 for f in funds_ if f.get("type") == "Mutual fund" and f.get("expense_ratio") is None)
        r["funds"] = funds_
        return r
    return api_funds


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
    old = (web.RUNS, funds.WARM, funds.CACHE_FILE, web.api_funds)
    tmp = tmp_path_factory.mktemp("funds_dir_site")
    web.RUNS, funds.WARM, funds.CACHE_FILE = tmp / "runs", False, tmp / "fund_stats.json"
    web.api_funds = _with_unknown_er(web.api_funds)
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
    web.RUNS, funds.WARM, funds.CACHE_FILE, web.api_funds = old


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
        # (the per-filter count, "N funds have no expense ratio yet: left out ...", not the page-wide fallback line)
        pg.wait_for_function("document.querySelector('#fUnknown').innerText.includes('left out by your filters')")
        unknown = SERVED["unknown"]
        assert unknown >= UNKNOWN_ER
        assert pg.inner_text("#fUnknown").startswith(f"{unknown} funds have no expense ratio yet")
        n = int(pg.inner_text("#fTitle").split(":")[1].split(" of ")[0].replace(",", ""))
        assert n > 10
        assert "VFINX" in pg.inner_text("#fTable")                          # 0.14%: passes the filter
        pg.check("#f_unk")
        n2 = int(pg.inner_text("#fTitle").split(":")[1].split(" of ")[0].replace(",", ""))
        assert n2 == n + unknown and "shown" in pg.inner_text("#fUnknown")
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
