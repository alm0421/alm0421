"""Every page of the site in headless Chromium: visits each tab, exercises the main controls (including the Monte
Carlo "Forecast returns" mode with weight changes, which once failed with "afterPress is not defined" because the
helper lived in another script block) and fails on any uncaught page error or console error.

The server listens on 127.0.0.1:8830; a stale one left on that port is stopped with `fuser -k 8830/tcp`."""
import glob
import shutil
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from backtester import data, web

PORT = 8830
AVAIL = set(data.available_tickers())


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


pytestmark = pytest.mark.skipif(_chromium() is None or not {"SPY", "TLT", "QQQ", "GLD", "SPYSIM", "TLTSIM", "EFASIM"} <= AVAIL,
                                reason="headless Chromium or price data not available")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = web.RUNS
    web.RUNS = tmp_path_factory.mktemp("runs")
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


# Chromium's own background calls (component updates, Google services) can hang behind a proxy and keep
# "networkidle" from arriving; the site itself makes no outside calls
_QUIET = ["--disable-background-networking", "--disable-component-update", "--disable-sync", "--disable-default-apps",
          "--no-first-run", "--disable-domain-reliability", "--disable-features=OptimizationHints,Translate"]


PAGES = ["backtest", "library", "gallery", "community", "build", "compare", "research", "montecarlo", "factors",
         "correlations", "signals", "history", "data"]


def test_every_page_and_main_controls_without_errors(site):
    from playwright.sync_api import sync_playwright
    errs: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium(), args=_QUIET)
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg.on("console", lambda m: errs.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
        pg.on("response", lambda r: errs.append(f"HTTP {r.status} {r.request.method} {r.url}") if r.status >= 400 else None)
        pg.set_default_timeout(120_000)
        pg.goto(site + "/")
        pg.wait_for_load_state("networkidle")

        # every page (tab) of the site
        for name in PAGES:
            pg.evaluate(f"location.hash = '#{name}'")
            pg.wait_for_selector(f"#p-{name}:not(.hide)")
            pg.wait_for_load_state("networkidle")
        assert not errs, errs

        # Monte Carlo: the forecast mode shows one return and one volatility box per ticker, following weight edits
        pg.evaluate("location.hash = '#montecarlo'")
        pg.wait_for_selector("#p-montecarlo:not(.hide)")
        for v in ("text", "run", "weights"):
            pg.click(f"#mcSrc button[data-v='{v}']")
        pg.select_option("#mc_model", "forecast")
        time.sleep(0.2)
        assert not errs, errs          # once: "afterPress is not defined"
        pg.wait_for_selector("#mcForecast:not(.hide)", timeout=10_000)
        assert pg.eval_on_selector_all("#mcForecastGrid input", "xs => xs.length") == 4        # SPY, TLT
        pg.fill("#mc_weights", "SPY 50, TLT 30, GLD 20")
        pg.dispatch_event("#mc_weights", "change")
        pg.wait_for_function("document.querySelectorAll('#mcForecastGrid input').length === 6")
        pg.fill("#mcf_SPY_ret", "5")
        pg.fill("#mc_weights", "SPY 70, GLD 30")
        pg.dispatch_event("#mc_weights", "change")
        pg.wait_for_function("document.querySelectorAll('#mcForecastGrid input').length === 4")
        assert pg.input_value("#mcf_SPY_ret") == "5"                                          # kept across edits
        pg.fill("#mc_sims", "200")
        pg.fill("#mc_years", "10")
        pg.click("#mcRun")
        pg.wait_for_function("!document.querySelector('#mcRun').disabled")
        pg.select_option("#mc_model", "historical")
        pg.wait_for_selector("#mcForecast.hide", state="attached")
        assert not errs, errs

        # Backtest: interpretation as you type, then a run with its report
        pg.evaluate("location.hash = '#backtest'")
        pg.fill("#text", "hold 60% SPY and 40% TLT, rebalance quarterly, since 2015")
        pg.wait_for_function("document.querySelector('#interp').textContent.includes('SPY')")
        pg.click("#runBtn")
        pg.wait_for_selector("#result:not(.hide)")
        pg.wait_for_function("document.querySelector('#frame').src && document.querySelector('#frame').src.includes('report')")
        pg.wait_for_load_state("networkidle")
        # an allocation portfolio's report has no trade-return distribution or excursion charts
        rep = pg.frame_locator("#frame")
        rep.locator("#distCard.hide").wait_for(state="attached")
        time.sleep(1.0)
        assert not errs, errs

        # Build: both editor modes
        pg.evaluate("location.hash = '#build'")
        pg.wait_for_selector("#p-build:not(.hide)")
        pg.click("#buildMode button[data-v='signal']")
        pg.click("#buildMode button[data-v='portfolio']")
        pg.wait_for_load_state("networkidle")

        # Research: all three modes
        pg.evaluate("location.hash = '#research'")
        for v in ("walkforward", "optimize", "sweep"):
            pg.click(f"#resMode button[data-v='{v}']")

        # Factors: every source tab, then a regression on a ticker
        pg.evaluate("location.hash = '#factors'")
        for v in ("weights", "text", "run", "ticker"):
            pg.click(f"#fxSrc button[data-v='{v}']")
        pg.fill("#fx_ticker", "QQQ")
        pg.click("#fxRun")
        pg.wait_for_function("!/…|\\.\\.\\./.test(document.querySelector('#fxStatus').textContent)")
        pg.wait_for_load_state("networkidle")

        # Correlations: daily returns with a monthly-stepped series switch to monthly (with a note)
        pg.evaluate("location.hash = '#correlations'")
        pg.fill("#k_tickers", "SPYSIM TLTSIM EFASIM")
        pg.select_option("#k_freq", "daily")
        pg.fill("#k_start", "1985-01-01")
        pg.fill("#k_end", "1989-12-31")
        pg.click("#kRun")
        pg.wait_for_function("document.querySelector('#p-correlations').innerText.includes('Monthly returns used')")

        # Compare: add a row
        pg.evaluate("location.hash = '#compare'")
        pg.click("#cmpAdd")
        pg.wait_for_load_state("networkidle")
        b.close()
    assert not errs, errs


@pytest.mark.skipif(not {"EEMSIM", "IEFSIM"} <= AVAIL, reason="price data not downloaded")
def test_reports_with_steps_depletion_and_late_benchmarks_render_without_errors(tmp_path):
    from playwright.sync_api import sync_playwright

    from backtester import parser, report, runner
    texts = ["hold 50% SPYSIM and 50% EEMSIM, rebalance monthly, since 1990, until 2002",
             "hold 60% SPYSIM and 40% IEFSIM, withdraw 4% per year adjusted for inflation, starting with $1,000,000, "
             "rebalance yearly, since 1966"]
    A = [report.analyze(runner.run(parser.parse(t)), sensitivity=False, mc=False) for t in texts]
    paths = [report.write_outputs(A[0], tmp_path / "steps", excel=False),
             report.write_outputs(A[1], tmp_path / "ruin", excel=False),
             report.write_outputs(A, tmp_path / "both", excel=False)]
    errs: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium(), args=_QUIET)
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg.on("console", lambda m: errs.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
        for path in paths:
            pg.goto(path.resolve().as_uri())
            pg.wait_for_timeout(800)
        # the stepped run says so in its risk table
        pg.goto(paths[0].resolve().as_uri())
        pg.wait_for_timeout(800)
        assert "Monthly-stepped data" in pg.inner_text("#riskTbl")
        # the comparison's common period lists the later benchmarks separately
        pg.goto(paths[2].resolve().as_uri())
        pg.wait_for_timeout(800)
        if pg.is_visible("#cmpSeg"):
            pg.click("#cmpSeg button[data-v='common']")
            assert pg.is_visible("#cmpOwn") and "SPY (1993-01-29" in pg.inner_text("#cmpOwnTbl")
        b.close()
    assert not errs, errs
