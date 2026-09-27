"""Round 10 UI (Composer expert review): the Build page's block editor refuses bad windows (red field, message, no
run) instead of rewriting them, applies the parser's unit and range checks, offers MACD / PPO / Bollinger / standard
deviation of price, and runs a loaded filter strategy unchanged ("Edit as blocks", gallery Fork, share links) - it no
longer adds an inert "if nothing qualifies: cash" fallback."""
import glob
import json
import shutil
import subprocess
import threading
import time

import pytest

from backtester import data, parser, web

AVAIL = set(data.available_tickers())
PORT = 8863


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


browser = pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = web.RUNS
    web.RUNS = tmp_path_factory.mktemp("runs10")
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


class Page:
    def __init__(self, p, site):
        self.b = p.chromium.launch(executable_path=_chromium())
        self.pg = self.b.new_page(viewport={"width": 1280, "height": 900})
        self.errs: list[str] = []
        self.runs: list[dict] = []
        self.pg.on("pageerror", lambda e: self.errs.append(str(e)))
        self.pg.on("request", lambda r: self.runs.append(json.loads(r.post_data)) if "/api/run" in r.url else None)
        self.pg.set_default_timeout(120_000)
        self.site = site

    def load_json(self, spec: dict):
        pg = self.pg
        pg.goto(self.site + "/#build")
        pg.wait_for_selector("#bPortfolio:not(.hide)")
        pg.evaluate("s => { document.querySelector('#specJson').value = s; document.querySelector('#fromJson').click(); }",
                    json.dumps(spec))
        time.sleep(0.8)

    def tree_json(self) -> dict:
        return self.pg.evaluate("document.getElementById('toJson').click(), JSON.parse(document.getElementById('specJson').value)")

    def run_and_wait(self):
        """Click Run; True when a run request was sent (and finished), False when the editor refused it."""
        n = len(self.runs)
        self.pg.click("#buildRun")
        for _ in range(100):      # the editor's checks (and the guard's server round trip) come first
            time.sleep(0.15)
            if len(self.runs) > n or self.pg.locator("#buildStatus .err").count():
                break
        if len(self.runs) == n:
            return False
        self.pg.wait_for_function("!document.querySelector('#buildStatus .spin')", timeout=300_000)
        return True


def _if(rule, on="SPY"):
    return {"kind": "allocation", "rebalance": "daily",
            "tree": {"if": rule, "on": on, "then": {"asset": "QQQ"}, "else": {"asset": "TLT"}}}


@browser
@pytest.mark.skipif(not {"SPY", "QQQ", "TLT"} <= AVAIL, reason="price data not downloaded")
@pytest.mark.parametrize("value", ["-5", "0", "2.5"])
def test_bad_window_is_marked_and_refused_never_rewritten(site, value):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        P = Page(p, site)
        pg = P.pg
        P.load_json(_if("tret(tr, 10) > 0"))
        win = pg.locator("#tree input.win").first
        win.fill(value)
        win.press("Enter")
        time.sleep(0.6)
        win = pg.locator("#tree input.win").first
        assert win.input_value() == value
        assert "bad-in" in (win.get_attribute("class") or "")
        assert "positive whole number of days" in pg.inner_text("#tree .nerr")
        assert pg.inner_text("#treeErrors").strip()
        rule = P.tree_json()["tree"]["if"]
        assert rule.replace(" ", "") == f"tret(tr,{value})>0"      # as typed, never RSI(1) / RSI(14) / RSI(3)
        assert not P.run_and_wait()
        assert "fix" in pg.inner_text("#buildStatus")
        # a good window clears it and runs
        win.fill("5")
        win.press("Enter")
        time.sleep(0.6)
        assert "bad-in" not in (pg.locator("#tree input.win").first.get_attribute("class") or "")
        assert not P.errs, P.errs
        P.b.close()


@browser
@pytest.mark.skipif(not {"SPY", "QQQ", "TLT"} <= AVAIL, reason="price data not downloaded")
def test_unit_and_range_checks_in_the_editor(site):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        P = Page(p, site)
        pg = P.pg
        # an RSI threshold it can never reach: an error, not a note
        P.load_json(_if("rsi(close, 10) > 120"))
        assert "0 to 100" in pg.inner_text("#tree .nerr .err")
        assert not P.run_and_wait()
        # a signed max drawdown: read as a fall shallower than 10%, said in the block and in the interpretation
        P.load_json(_if("max_drawdown(tr, 10) > -0.1"))
        assert "shallower than 10%" in pg.inner_text("#tree .nerr")
        pg.wait_for_function("document.querySelector('#buildInterp').textContent.includes('positive size')")
        assert P.run_and_wait()
        assert "max_drawdown(tr, 10) < 0.1" in P.runs[-1]["spec"]["tree"]["if"] or \
            "max_drawdown(tr, 10) > -0.1" in P.runs[-1]["spec"]["tree"]["if"]
        assert "CAGR" in pg.inner_text("#buildStatus") or "Not run" not in pg.inner_text("#buildStatus")
        assert not P.errs, P.errs
        P.b.close()


@browser
@pytest.mark.skipif(not {"SPY", "QQQ", "TLT"} <= AVAIL, reason="price data not downloaded")
def test_macd_bollinger_ppo_and_stdev_in_the_pickers(site):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        P = Page(p, site)
        pg = P.pg
        P.load_json(_if("macd(12, 26) > macd_signal(12, 26, 9)"))
        sels = pg.locator("#tree select").evaluate_all("ss => ss.map(s => s.value)")
        assert "macd" in sels and "macd_signal" in sels
        assert pg.locator("#tree input.win").count() == 5
        opts = pg.locator("#tree select").evaluate_all("ss => [...ss[1].options].map(o => o.value)")
        for k in ("macd", "macd_signal", "ppo", "ppo_signal", "bb_upper", "bb_lower", "stdev"):
            assert k in opts, opts
        # switch the left side to the upper Bollinger band of QQQ
        P.load_json(_if("close > sma(close, 200)"))
        pg.locator("#tree select").nth(1).select_option("bb_upper")
        time.sleep(0.6)
        rule = P.tree_json()["tree"]["if"]
        assert rule.startswith("bb_upper(20, 2")
        # a filter ranked by PPO
        P.load_json({"kind": "allocation", "rebalance": "daily", "tree": {
            "filter": {"select": "top", "n": 1, "by": "ppo(12, 26)", "weights": "equal"}, "universe": "children",
            "children": [{"asset": "QQQ"}, {"asset": "SPY"}, {"asset": "TLT"}]}})
        assert pg.locator("#tree input.win").count() == 2
        assert P.run_and_wait()
        assert P.runs[-1]["spec"]["tree"]["filter"]["by"] == "ppo(12, 26)"
        assert not P.errs, P.errs
        P.b.close()


FILTER_TEXTS = [
    "hold the top 1 of TQQQ, SOXL and TECL by 10 day cumulative return",
    "hold the top 2 of TQQQ, SOXL and TECL by 10 day cumulative return, inverse volatility weighted",
    "hold the top 1 of TQQQ, SOXL and TECL by 10 day cumulative return only if their 20 day RSI is above 50",
]


@browser
@pytest.mark.skipif(not {"TQQQ", "SOXL", "TECL"} <= AVAIL, reason="price data not downloaded")
@pytest.mark.parametrize("text", FILTER_TEXTS)
def test_edit_as_blocks_then_run_a_filter_strategy(site, text):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        P = Page(p, site)
        pg = P.pg
        pg.goto(site + "/#backtest")
        pg.fill("#text", text)
        pg.wait_for_function("document.querySelector('#interp').textContent.includes('tret')")
        pg.click("#toBuild")
        pg.wait_for_selector("#bPortfolio:not(.hide)")
        time.sleep(1.5)     # the loaded spec is normalised by the server (the guard's reference)
        assert P.run_and_wait(), pg.inner_text("#buildStatus")
        assert "Not run" not in pg.inner_text("#buildStatus"), pg.inner_text("#buildStatus")
        assert "fallback" not in P.runs[-1]["spec"]["tree"] or "only if" in text
        assert not P.errs, P.errs
        P.b.close()


@browser
@pytest.mark.skipif(not set("XLK XLE XLF XLV XLY XLP XLI XLU XLB".split()) <= AVAIL, reason="price data not downloaded")
def test_gallery_fork_of_a_filter_strategy_runs(site):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        P = Page(p, site)
        pg = P.pg
        pg.goto(site + "/#gallery")
        row = pg.locator("#galLib tr", has_text="Sector momentum top 2 (3 months)")
        row.get_by_role("button", name="Fork into editor").click()
        pg.wait_for_selector("#bPortfolio:not(.hide)")
        time.sleep(1.5)
        assert P.run_and_wait(), pg.inner_text("#buildStatus")
        assert "Not run" not in pg.inner_text("#buildStatus"), pg.inner_text("#buildStatus")
        assert not P.errs, P.errs
        P.b.close()


@browser
@pytest.mark.skipif(not {"TQQQ", "SOXL", "TECL"} <= AVAIL, reason="price data not downloaded")
def test_share_link_of_a_filter_strategy_edit_as_blocks_runs(site):
    from playwright.sync_api import sync_playwright
    spec = parser.parse(FILTER_TEXTS[0] + ", since 2020")
    tok = web.share_token(spec)
    with sync_playwright() as p:
        P = Page(p, site)
        pg = P.pg
        pg.goto(site + f"/#backtest?s={tok}")
        pg.wait_for_function("document.querySelector('#interp').textContent.includes('tret')")
        time.sleep(1.0)
        pg.click("#toBuild")
        pg.wait_for_selector("#bPortfolio:not(.hide)")
        time.sleep(1.5)
        assert P.run_and_wait(), pg.inner_text("#buildStatus")
        assert "Not run" not in pg.inner_text("#buildStatus"), pg.inner_text("#buildStatus")
        assert not P.errs, P.errs
        P.b.close()
