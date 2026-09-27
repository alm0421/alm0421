"""The Backtest page's asset allocation grid (Portfolio Visualizer "Backtest Portfolio"): rows of tickers or
asset-class names x up to 3 portfolios, shared settings and cash-flow phases -> one Portfolio spec per column,
run together as a Compare report (POST /api/grid). API checks, then the page in headless Chromium at desktop and
phone width (the server listens on 127.0.0.1:8841; a stale one is stopped with `fuser -k 8841/tcp`)."""
import glob
import shutil
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from backtester import data, web

AVAIL = set(data.available_tickers())
pytestmark = pytest.mark.skipif(not {"SPY", "AGG", "VTI", "BND", "VXUSSIM", "BNDSIM"} <= AVAIL, reason="price data not available")

GRID = {"rows": [{"asset": "SPY", "w": [60, None, None]}, {"asset": "AGG", "w": [40, None, None]},
                 {"asset": "VTI", "w": [None, 40, None]}, {"asset": "International Stocks", "w": [None, 20, None]},
                 {"asset": "Total Bond Market", "w": [None, 40, None]}],
        "names": ["60/40", "Three-fund", ""], "start": "2010", "end": "2020-06", "capital": 100000,
        "flows": [{"kind": "contribute", "mode": "amount", "amount": 1000, "freq": "monthly", "inflation": True,
                   "start_year": 1, "end_year": 5},
                  {"kind": "withdraw", "mode": "pct", "amount": 4, "freq": "annual", "start_year": 6}],
        "rebalance": "bands", "band": 5, "benchmark": "SPY", "expense_ratio": 0.1, "leverage": 1}


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "RUNS", tmp_path)
    return tmp_path


def test_grid_builds_one_portfolio_per_column():
    specs, notes, problems = web.grid_specs(GRID)
    assert problems == [] and [s.name for s in specs] == ["60/40", "Three-fund"]
    a, b = specs
    assert a.tree == {"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "AGG"}]}
    assert b.universe == ["VTI", "VXUSSIM", "BNDSIM"] and "'International Stocks' is read as VXUSSIM." in notes
    for s in specs:
        assert (s.start, s.end, s.capital) == ("2010-01-01", "2020-06-30", 100000.0)
        assert s.rebalance == "none" and s.drift_band == 0.05 and s.expense_ratio == pytest.approx(0.001)
        assert (s.contribution, s.contribution_freq, s.contribution_inflation, s.contribution_start, s.contribution_end) == \
            (1000, "monthly", True, 1, 5)
        assert s.withdrawal_pct == pytest.approx(0.04) and s.withdrawal_freq == "yearly" and s.withdrawal_start == 6
        assert s.benchmark == "SPY"
    # quarterly % withdrawals are a share per year split over the periods; relative bands; calendar years
    g = dict(GRID, rebalance="bands", band_mode="relative", band=25,
             flows=[{"kind": "withdraw", "mode": "pct", "amount": 4, "freq": "quarterly", "start_year": 2015, "end_year": 2019},
                    {"kind": "withdraw", "mode": "amount", "amount": 500, "freq": "quarterly", "start_year": 2015,
                     "end_year": 2019, "inflation": False}])
    s = web.grid_specs(g)[0][0]
    assert s.drift_band_relative == 0.25 and s.drift_band is None and s.withdrawal_pct == pytest.approx(0.01)
    assert (s.withdrawal, s.withdrawal_start, s.withdrawal_end, s.withdrawal_inflation) == (500, 2015, "2019-12-31", False)


@pytest.mark.parametrize("change, msg", [
    ({"rows": [{"asset": "SPY", "w": [60]}, {"asset": "AGG", "w": [30]}]}, "Portfolio 1: the weights add up to 90%, not 100%."),
    ({"rows": [{"asset": "SPY", "w": [60]}, {"asset": "XQZZQ", "w": [40]}]}, "Unknown ticker or asset class 'XQZZQ'"),
    ({"rows": [{"asset": "SPY", "w": [60]}, {"asset": "Moon rocks", "w": [40]}]}, "Unknown ticker or asset class 'Moon rocks'"),
    ({"rows": [{"asset": "SPY", "w": [60]}, {"asset": "spy", "w": [40]}]}, "use one row per asset"),
    ({"rows": [{"asset": "SPY", "w": [None]}]}, "Enter weights (%) for at least one portfolio."),
    ({"rows": [{"asset": "SPY", "w": [110]}, {"asset": "AGG", "w": [-10]}]}, "weight of AGG is negative"),
    ({"start": "May 1990"}, "Dates are a year"),
    ({"rebalance": "fortnightly"}, "Rebalancing must be one of"),
    ({"benchmark": "NOSUCHX"}, "Benchmark: Unknown ticker"),
    ({"flows": [{"kind": "contribute", "amount": 100}, {"kind": "contribute", "amount": 200}]}, "One contribution phase"),
    ({"flows": [{"kind": "contribute", "mode": "pct", "amount": 2}]}, "contributions are a $ amount"),
    ({"flows": [{"kind": "withdraw", "mode": "pct", "amount": 4, "freq": "yearly"},
                {"kind": "withdraw", "amount": 100, "freq": "monthly"}]}, "must share the frequency"),
])
def test_validation_messages(change, msg):
    specs, _, problems = web.grid_specs({**GRID, **change})
    assert specs == [] and any(msg in p for p in problems), problems


def test_run_share_and_load(runs):
    j = web.api_grid(dict(GRID, action="check"))
    assert j["problems"] == [] and [p["name"] for p in j["portfolios"]] == ["60/40", "Three-fund"] and "url" not in j
    j = web.api_grid(dict(GRID))
    assert j["url"].endswith("/report.html") and (runs / j["id"] / "report.html").is_file()
    html = (runs / j["id"] / "report.html").read_text()
    assert "60/40" in html and "Three-fund" in html
    assert web._index()[0]["kind"] == "compare" and web._index()[0]["label"] == "60/40 vs Three-fund"
    back = web.api_grid({"action": "load", "share": j["share"]})["grid"]
    assert back["rows"] == GRID["rows"] and back["flows"] == GRID["flows"] and back["names"] == GRID["names"]
    with pytest.raises(web.ClientError, match="weights add up to 90%"):
        web.api_grid({**GRID, "rows": [{"asset": "SPY", "w": [90]}]})
    with pytest.raises(web.ClientError, match="damaged"):
        web.api_grid({"action": "load", "share": "nonsense"})
    # one column is a plain one-portfolio report
    one = web.api_grid({**GRID, "rows": GRID["rows"][:2]})
    assert (runs / one["id"] / "report.html").is_file() and one["common"] is None
    assert "US Stock Market" in web.api_tickers()["asset_classes"]


# ------------------------------------------------------------------ the page in headless Chromium

PORT = 8841


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = web.RUNS
    web.RUNS = tmp_path_factory.mktemp("grid_runs")
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


def _fill_grid(pg):
    pg.wait_for_selector("#gTbl tr.grow")
    rows = [("SPY", ["60", ""]), ("AGG", ["40", ""]), ("VTI", ["", "40"])]
    for k, (a, w) in enumerate(rows):
        r = pg.locator("#gTbl tr.grow").nth(k)
        r.locator("input.gasset").fill(a)
        for c, x in enumerate(w):
            if x:
                r.locator("input.gw").nth(c).fill(x)
    pg.click("#gAddRow")
    pg.click("#gAddRow")
    r = pg.locator("#gTbl tr.grow").nth(3)
    r.locator("input.gasset").fill("International Stocks")
    r.locator("input.gw").nth(1).fill("20")
    r = pg.locator("#gTbl tr.grow").nth(4)
    r.locator("input.gasset").fill("Total Bond Market")
    r.locator("input.gw").nth(1).fill("30")
    pg.locator("#gTbl tr.grow").nth(2).locator("input.gasset").press("Escape")
    # live validation: the second column adds up to 90%
    pg.wait_for_function("document.querySelector('#gMsgs').textContent.includes('add up to 90%')", timeout=5000)
    assert "neg" in pg.get_attribute("#gTot1", "class")
    r.locator("input.gw").nth(1).fill("40")
    pg.wait_for_function("!document.querySelector('#gMsgs').textContent.includes('add up to')", timeout=5000)
    assert pg.inner_text("#gTot0") == "100%" and pg.inner_text("#gTot1") == "100%"
    pg.locator("input.gname").nth(0).fill("60/40")
    pg.locator("input.gname").nth(1).fill("Three-fund")
    pg.fill("#g_start", "2010")
    pg.fill("#g_end", "2020")
    pg.fill("#g_capital", "100000")
    pg.click("#gAddFlow")
    pg.click("#gAddFlow")
    f0 = pg.locator("#gFlows .gflow").nth(0)
    assert f0.locator("select[aria-label=kind]").input_value() == "contribute"
    f0.locator("input[aria-label=amount]").fill("1000")
    f0.locator("input[aria-label=start_year]").fill("1")
    f0.locator("input[aria-label=end_year]").fill("5")
    f1 = pg.locator("#gFlows .gflow").nth(1)
    assert f1.locator("select[aria-label=kind]").input_value() == "withdraw"
    f1.locator("select[aria-label=mode]").select_option("pct")
    f1 = pg.locator("#gFlows .gflow").nth(1)
    f1.locator("input[aria-label=amount]").fill("4")
    f1.locator("input[aria-label=start_year]").fill("6")


@pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")
@pytest.mark.parametrize("width", [1280, 390])
def test_grid_page_runs_a_comparison(site, width):
    from playwright.sync_api import sync_playwright
    errs: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": width, "height": 900})
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(site + "/#backtest")
        _fill_grid(pg)
        # an unknown ticker is flagged before running
        pg.locator("#gTbl tr.grow").nth(1).locator("input.gasset").fill("XQZZQ")
        pg.wait_for_function("document.querySelector('#gMsgs').textContent.includes('Unknown ticker: XQZZQ')", timeout=5000)
        pg.click("#gRun")
        assert "Unknown ticker: XQZZQ" in pg.inner_text("#gMsgs") and pg.is_hidden("#gResult")
        pg.locator("#gTbl tr.grow").nth(1).locator("input.gasset").fill("AGG")
        pg.locator("#gTbl tr.grow").nth(1).locator("input.gasset").press("Escape")
        pg.click("#gRun")
        pg.wait_for_selector("#gResult:not(.hide)", timeout=180_000)
        fr = pg.frame_locator("#gFrame")
        fr.locator("text=Comparison: 60/40 vs Three-fund").wait_for(timeout=60_000)
        txt = fr.locator("body").inner_text()
        assert "40% VTI" in txt and "20% VXUSSIM" in txt and "withdraw 4.0% of the balance yearly from year 6" in txt
        assert "#backtest?g=" in pg.url
        assert pg.evaluate("document.documentElement.scrollWidth") <= width     # no horizontal page scroll
        pg.click("#gShare")
        pg.wait_for_selector("#gShareLink")
        link = pg.get_attribute("#gShareLink", "href")
        # the share link rebuilds the form in a fresh page and runs it
        pg2 = b.new_page(viewport={"width": width, "height": 900})
        pg2.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg2.on("pageerror", lambda e: errs.append(str(e)))
        pg2.goto(link)
        pg2.wait_for_selector("#gResult:not(.hide)", timeout=180_000)
        assert pg2.locator("input.gname").nth(1).input_value() == "Three-fund"
        assert pg2.locator("#gTbl tr.grow").nth(3).locator("input.gasset").input_value() == "International Stocks"
        assert pg2.locator("#gFlows .gflow").count() == 2 and pg2.input_value("#g_start") == "2010"
        pg2.frame_locator("#gFrame").locator("text=Comparison: 60/40 vs Three-fund").wait_for(timeout=60_000)
        b.close()
    assert not errs, errs
