"""Round 13 UI (Composer expert review): the Build editor never loses keystrokes. A change in one box (a ticker
committed by the click into the next box) re-renders the tree; the box being typed in keeps its focus, text and caret,
so "79" typed at human speed into the threshold stays "79" (it used to become "7")."""
import glob
import shutil
import subprocess
import threading
import time

import pytest

from backtester import data, web

AVAIL = set(data.available_tickers())
PORT = 8834


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


pytestmark = [pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available"),
              pytest.mark.skipif(not {"SPY", "TQQQ", "UVXY", "QQQ", "TLT"} <= AVAIL, reason="price data not downloaded")]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = web.RUNS
    web.RUNS = tmp_path_factory.mktemp("runs13")
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


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        yield b
        b.close()


def _page(browser, site):
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    errs: list[str] = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.set_default_timeout(60_000)
    pg.goto(site + "/#build")
    pg.wait_for_selector("#bPortfolio:not(.hide)")
    pg.wait_for_timeout(800)
    return pg, errs


def _if_rsi(pg):
    """Build -> root If / else -> indicator RSI; returns the root node and its header."""
    root = pg.locator("#tree > .node")
    root.locator(":scope > .hd > select").first.select_option("if")
    pg.wait_for_timeout(300)
    hd = root.locator(":scope > .hd")
    hd.locator("select").nth(1).select_option("rsi")
    pg.wait_for_timeout(300)
    return root, hd


def _spec(pg):
    return pg.evaluate("document.getElementById('toJson').click(), JSON.parse(document.getElementById('specJson').value)")


def _typer(pg, mode, gap):
    def typ(loc, txt):
        if mode == "fill":
            loc.fill(txt)
        else:
            loc.click()
            pg.keyboard.press("Control+a")
            pg.keyboard.type(txt, delay=60 if mode == "slow" else 5)
        pg.wait_for_timeout(gap)
    return typ


def _then_ticker(root):
    return root.locator(":scope > .kids > .node").nth(0).locator(":scope > .hd input[placeholder=ticker]")


@pytest.mark.parametrize("mode,gap", [("slow", 1200), ("slow", 150), ("fast", 0), ("fill", 0), ("fill", 600)])
def test_ticker_comparator_threshold_keeps_every_keystroke(site, browser, mode, gap):
    pg, errs = _page(browser, site)
    root, hd = _if_rsi(pg)
    typ = _typer(pg, mode, gap)
    typ(hd.locator("input.win").first, "10")
    typ(hd.locator("input[placeholder=ticker]").first, "TQQQ")
    pg.keyboard.press("Escape")
    hd.locator("select").nth(2).select_option(">")
    pg.wait_for_timeout(gap)
    typ(hd.locator("input.thr"), "79")
    typ(_then_ticker(root), "UVXY")
    pg.keyboard.press("Escape")
    pg.locator("#p-build h3").first.click()
    pg.wait_for_timeout(1200)
    assert hd.locator("input.thr").input_value() == "79"
    tree = _spec(pg)["tree"]
    assert tree["if"] == "rsi(close, 10) > 79" and tree["on"] == "TQQQ", tree
    assert tree["then"] == {"asset": "UVXY"}
    pg.wait_for_function("document.getElementById('buildInterp').textContent.includes('79')")
    assert "> 7 " not in pg.inner_text("#buildInterp") + " "
    assert not errs
    pg.close()


@pytest.mark.parametrize("mode", ["slow", "fill"])
def test_threshold_then_ticker_and_window_sequences(site, browser, mode):
    """Other orders: the threshold first and then the ticker; the window right after the ticker; the then-branch
    ticker right after the threshold (each click commits the previous box and re-renders the tree)."""
    pg, errs = _page(browser, site)
    root, hd = _if_rsi(pg)
    typ = _typer(pg, mode, 0)
    typ(hd.locator("input.thr"), "31")
    typ(hd.locator("input[placeholder=ticker]").first, "qqq")
    typ(hd.locator("input.win").first, "14")
    typ(hd.locator("input.thr"), "28.5")
    typ(_then_ticker(root), "TQQQ")
    typ(hd.locator("input[placeholder=ticker]").first, "SPY")
    pg.locator("#p-build h3").first.click()
    pg.wait_for_timeout(1000)
    tree = _spec(pg)["tree"]
    assert tree["on"] == "SPY" and tree["if"] == "rsi(close, 14) > 28.5", tree
    assert tree["then"] == {"asset": "TQQQ"}
    assert not errs
    pg.close()


def test_group_boxes_keep_keystrokes(site, browser):
    """A weighted group's ticker and % boxes, typed in turn at human speed."""
    pg, errs = _page(browser, site)
    root = pg.locator("#tree > .node")
    root.locator(":scope > .hd > select").first.select_option("weights")
    pg.wait_for_timeout(300)
    root.locator(":scope > .hd select").nth(1).select_option("specified")
    pg.wait_for_timeout(300)
    tick = root.locator(":scope > .kids input[placeholder=ticker]")
    pct = root.locator(":scope > .kids > .row > label input[type=number]")
    assert tick.count() >= 2 and pct.count() >= 2
    for k, (t, w) in enumerate([("QQQ", "60"), ("TLT", "40")]):
        tick.nth(k).click(); pg.keyboard.press("Control+a"); pg.keyboard.type(t, delay=60)
        pct.nth(k).click(); pg.keyboard.press("Control+a"); pg.keyboard.type(w, delay=60)
    pg.locator("#p-build h3").first.click()
    pg.wait_for_timeout(800)
    tree = _spec(pg)["tree"]
    assert [c["asset"] for c in tree["children"][:2]] == ["QQQ", "TLT"], tree
    assert [round(x, 6) for x in tree["w"][:2]] == [0.6, 0.4], tree
    assert not errs
    pg.close()


def test_focus_survives_a_rerender_with_caret(site, browser):
    """A re-render while a box has the focus: the same box in the new tree has it, with the text and the caret."""
    pg, errs = _page(browser, site)
    root, hd = _if_rsi(pg)
    thr = hd.locator("input.thr")
    thr.click(); pg.keyboard.press("Control+a"); pg.keyboard.type("65", delay=30)
    tk = hd.locator("input[placeholder=ticker]").first
    tk.click(); pg.keyboard.press("Control+a"); pg.keyboard.type("tq", delay=30)
    pg.keyboard.press("ArrowLeft")
    # something else re-renders the tree (as a change elsewhere would)
    pg.evaluate("document.getElementById('p_rebalance').dispatchEvent(new Event('change'))")
    pg.wait_for_timeout(100)
    st = pg.evaluate("(() => { const a = document.activeElement; return [a.placeholder, a.value, a.selectionStart, a.isConnected]; })()")
    assert st == ["ticker", "tq", 1, True]
    pg.keyboard.type("Q", delay=30)
    pg.keyboard.press("End")
    pg.keyboard.type("Q", delay=30)
    pg.locator("#p-build h3").first.click()
    pg.wait_for_timeout(800)
    assert _spec(pg)["tree"]["on"] == "TQQQ"
    assert hd.locator("input.thr").input_value() == "65"
    assert not errs
    pg.close()
