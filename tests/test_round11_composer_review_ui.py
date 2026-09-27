"""Round 11 UI (Composer expert review): the Build page never turns an empty or non-numeric threshold into 0 (the
field is red, the rule keeps no number and Run is refused, for if/else conditions and "only if" requirements alike),
and a threshold-rebalanced symphony (rebalance none + a corridor) is shown as it runs: "evaluated daily · threshold
5%" and Rebalance = threshold, with the band kept through the editor."""
import glob
import json
import shutil
import subprocess
import threading
import time

import pytest

from backtester import data, web

AVAIL = set(data.available_tickers())
PORT = 8872


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


browser = pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")
needs = pytest.mark.skipif(not {"SPY", "QQQ", "TLT", "TQQQ", "BIL"} <= AVAIL, reason="price data not downloaded")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = web.RUNS
    web.RUNS = tmp_path_factory.mktemp("runs11")
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

    def spec(self) -> dict:
        return self.pg.evaluate("document.getElementById('toJson').click(), JSON.parse(document.getElementById('specJson').value)")

    def run_refused(self) -> bool:
        n = len(self.runs)
        self.pg.click("#buildRun")
        for _ in range(40):
            time.sleep(0.15)
            if len(self.runs) > n or self.pg.locator("#buildStatus .err").count():
                break
        return len(self.runs) == n and self.pg.locator("#buildStatus .err").count() > 0


def _if(rule, on="SPY", **kw):
    return {"kind": "allocation", "rebalance": "daily", **kw,
            "tree": {"if": rule, "on": on, "then": {"asset": "QQQ"}, "else": {"asset": "TLT"}}}


@browser
@needs
@pytest.mark.parametrize("rule, typed", [("tret(tr, 63) > 0.02", ""), ("tret(tr, 63) > 0.02", "abc"),
                                         ("rsi(close, 10) > 79", ""), ("rsi(close, 10) < 30", "abc")])
def test_empty_or_non_numeric_threshold_is_red_and_refused(site, rule, typed):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        P = Page(p, site)
        pg = P.pg
        P.load_json(_if(rule))
        thr = pg.locator("#tree input.thr").first
        assert thr.input_value() in ("2", "79", "30")
        thr.fill("")
        if typed:
            thr.press_sequentially(typed)     # a number box takes no letters: it reads as empty
        thr.press("Tab")
        time.sleep(0.8)
        thr = pg.locator("#tree input.thr").first
        assert thr.input_value() == ""
        assert "bad-in" in (thr.get_attribute("class") or "")
        assert "enter the number" in pg.inner_text("#tree .nerr").lower()
        assert pg.inner_text("#treeErrors").strip()
        r = P.spec()["tree"]["if"]
        assert not r.rstrip().endswith(("0", "0.0")) and not r.rstrip()[-1].isdigit(), r     # never a silent 0
        assert P.run_refused()
        # a number clears it
        thr.fill("5")
        thr.press("Tab")
        time.sleep(0.8)
        assert "bad-in" not in (pg.locator("#tree input.thr").first.get_attribute("class") or "")
        assert not pg.inner_text("#treeErrors").strip()
        assert P.spec()["tree"]["if"].replace(" ", "").endswith(("5", "0.05"))
        assert not P.errs, P.errs
        P.b.close()


@browser
@needs
def test_empty_requirement_threshold_is_refused(site):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        P = Page(p, site)
        pg = P.pg
        P.load_json({"kind": "allocation", "rebalance": "daily", "tree": {
            "filter": {"select": "top", "n": 1, "by": "tret(tr, 21)", "weights": "equal", "require": "tret(tr, 126) > 0"},
            "universe": "children", "children": [{"asset": "QQQ"}, {"asset": "SPY"}, {"asset": "TLT"}],
            "fallback": {"cash": True}}})
        thr = pg.locator("#tree input.thr").first
        thr.fill("")
        thr.press("Tab")
        time.sleep(0.8)
        assert "bad-in" in (pg.locator("#tree input.thr").first.get_attribute("class") or "")
        req = P.spec()["tree"]["filter"]["require"]
        assert req.replace(" ", "") == "tret(tr,126)>", req
        assert P.run_refused()
        assert not P.errs, P.errs
        P.b.close()


def test_server_refuses_a_missing_threshold():
    from backtester import portfolio as pf
    for rule in ("tret(tr, 63) >", "rsi(close, 10) > and close > 1", "> 5"):
        with pytest.raises(ValueError, match="missing a number"):
            pf.Portfolio(tree={"if": rule, "on": "SPY", "then": {"asset": "SPY"}, "else": {"cash": True}}).validate()


@browser
@needs
def test_threshold_symphony_badge_and_round_trip(site):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        P = Page(p, site)
        pg = P.pg
        # a Composer symphony with threshold rebalancing (rebalance none + a 5% corridor)
        sym = {"step": "root", "name": "T", "rebalance": "none", "rebalance-corridor-width": 0.05, "children": [
            {"step": "if", "children": [
                {"step": "if-child", "is-else-condition?": False, "lhs-fn": "current-price", "lhs-val": "SPY",
                 "comparator": "gt", "rhs-val": "400", "rhs-fixed-value?": True, "children": [{"step": "asset", "ticker": "TQQQ"}]},
                {"step": "if-child", "is-else-condition?": True, "children": [{"step": "asset", "ticker": "BIL"}]}]}]}
        pg.goto(site + "/#build")
        pg.wait_for_selector("#bPortfolio:not(.hide)")
        pg.evaluate("document.getElementById('composerBox').open = true")
        pg.fill("#composerJson", json.dumps(sym))
        pg.click("#composerLoad")
        pg.wait_for_function("document.querySelector('#composerStatus').textContent.includes('Loaded')")
        time.sleep(0.8)
        badge = pg.locator("#tree .badge").first
        assert badge.inner_text() == "evaluated daily · threshold 5%"
        assert "evaluated once" not in pg.inner_text("#tree")
        assert "first day" not in (badge.get_attribute("title") or "")
        assert pg.input_value("#p_rebalance") == "threshold"
        assert "threshold" in pg.evaluate("document.querySelector('#p_rebalance').selectedOptions[0].textContent")
        s = P.spec()
        assert s["rebalance"] == "none" and abs(s["drift_band"] - 0.05) < 1e-12
        # the band round-trips through an edit, and a threshold without a band is refused
        pg.fill("#p_drift_band", "7.5")
        pg.press("#p_drift_band", "Tab")
        time.sleep(0.6)
        assert pg.locator("#tree .badge").first.inner_text() == "evaluated daily · threshold 7.5%"
        assert abs(P.spec()["drift_band"] - 0.075) < 1e-12
        pg.fill("#p_drift_band", "")
        pg.press("#p_drift_band", "Tab")
        time.sleep(0.6)
        assert "needs a drift band" in pg.inner_text("#treeErrors")
        assert P.run_refused()
        # no schedule and no band: a tree with an if/else is still evaluated every day (never "once")
        pg.select_option("#p_rebalance", "none")
        time.sleep(0.6)
        assert pg.locator("#tree .badge").first.inner_text() == "evaluated daily"
        assert "rule switches" in pg.evaluate("document.querySelector('#p_rebalance').selectedOptions[0].textContent")
        assert not P.errs, P.errs
        P.b.close()


@browser
@needs
def test_market_cap_weights_over_etfs_are_marked(site):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        P = Page(p, site)
        pg = P.pg
        P.load_json({"kind": "allocation", "rebalance": "monthly",
                     "tree": {"weights": "market_cap", "children": [{"asset": "SPY"}, {"asset": "QQQ"}, {"asset": "TLT"}]}})
        pg.wait_for_function("[...document.querySelectorAll('#tree .nerr')].some(x => x.textContent.includes('no market cap'))")
        assert P.run_refused()
        assert not P.errs, P.errs
        P.b.close()
