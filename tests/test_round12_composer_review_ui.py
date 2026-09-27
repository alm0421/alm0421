"""Round 12 UI (Composer expert review): the Build page marks the branch of each if/else that is active on the latest
close ("active today"), a new filter starts with no placeholder tickers (a hint says how to add them), and the
rebalance day is a control."""
import glob
import json
import shutil
import subprocess
import threading
import time

import pytest

from backtester import data, orders, web
from backtester.portfolio import Portfolio

AVAIL = set(data.available_tickers())
PORT = 8826


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


class Page:
    def __init__(self, p, site):
        self.b = p.chromium.launch(executable_path=_chromium())
        self.pg = self.b.new_page(viewport={"width": 1280, "height": 900})
        self.errs: list[str] = []
        self.pg.on("pageerror", lambda e: self.errs.append(str(e)))
        self.pg.on("console", lambda m: self.errs.append(m.text) if m.type == "error" else None)
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


NESTED = {"kind": "allocation", "rebalance": "daily",
          "tree": {"if": "close > sma(close, 200)", "on": "SPY",
                   "then": {"if": "rsi(close, 10) > 79", "on": "QQQ", "then": {"asset": "TLT"}, "else": {"asset": "QQQ"}},
                   "else": {"weights": "equal", "children": [
                       {"asset": "TLT"},
                       {"if": "close < sma(close, 20)", "on": "QQQ", "then": {"asset": "GLD"}, "else": {"cash": True}}]}}}


@pytest.mark.skipif(not {"SPY", "QQQ", "TLT", "GLD"} <= AVAIL, reason="price data not downloaded")
def test_active_branches_follow_the_latest_close():
    p = Portfolio.from_dict({k: v for k, v in NESTED.items() if k != "kind"})
    r = orders.active_branches(p)
    assert len(r["branches"]) == 3 and set(r["branches"]) <= {"then", "else"}
    # the same answer as the rules evaluated by hand on the latest bar
    spy = data.load("SPY")
    assert r["as_of"] == str(spy.index[-1].date())
    ans = web.api_branches({"spec": NESTED})
    assert ans == r


@browser
@pytest.mark.skipif(not {"SPY", "QQQ", "TLT", "GLD"} <= AVAIL, reason="price data not downloaded")
def test_active_today_badges_and_empty_new_filter(site):
    from playwright.sync_api import sync_playwright
    want = orders.active_branches(Portfolio.from_dict({k: v for k, v in NESTED.items() if k != "kind"}))["branches"]
    with sync_playwright() as p:
        P = Page(p, site)
        pg = P.pg
        P.load_json(NESTED)
        pg.wait_for_selector("#tree .badge.active:not(.hide)")
        pg.wait_for_function(f"document.querySelectorAll('#tree .badge.active:not(.hide)').length === {len(want)}")
        got = pg.evaluate("[...document.querySelectorAll('#tree .badge.active')].filter(b => !b.classList.contains('hide'))"
                          ".map(b => b.dataset.branch)")
        assert got == want
        # one badge per if/else, on the branch the server says; the title names the date
        title = pg.locator("#tree .badge.active:not(.hide)").first.get_attribute("title")
        assert "latest close" in title and "branch is held today" in title
        # a new filter: turn the cash block of the second if's otherwise branch into a filter -> no placeholder tickers
        cash_sel = pg.locator("#tree .node").filter(has=pg.locator("select >> nth=0")).last.locator("select").first
        cash_sel.select_option("filter")
        time.sleep(0.8)
        tree = P.tree_json()["tree"]
        f = tree["else"]["children"][1]["else"]
        assert "filter" in f and f["children"] == []
        assert "add the assets to choose from" in pg.inner_text("#tree")
        # the rebalance day control
        pg.select_option("#p_rebalance", "monthly")
        pg.select_option("#p_rebalance_day", "start")
        time.sleep(0.5)
        assert P.tree_json()["rebalance_day"] == "start"
        assert "period start" in pg.inner_text("#tree .badge >> nth=0")
        assert not P.errs, P.errs
        P.b.close()


@browser
@pytest.mark.skipif(not {"SPY", "TLT"} <= AVAIL, reason="price data not downloaded")
def test_rebalance_day_from_a_loaded_spec_round_trips(site):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        P = Page(p, site)
        P.load_json({"kind": "allocation", "rebalance": "weekly", "rebalance_day": "monday",
                     "tree": {"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "TLT"}]}})
        assert P.pg.input_value("#p_rebalance_day") == "monday"
        spec = P.tree_json()
        assert spec["rebalance"] == "weekly" and spec["rebalance_day"] == "monday"
        # daily: no rebalance day is sent
        P.pg.select_option("#p_rebalance", "daily")
        time.sleep(0.3)
        assert "rebalance_day" not in P.tree_json()
        assert not P.errs, P.errs
        P.b.close()
