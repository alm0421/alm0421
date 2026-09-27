"""Round 13 UI (headless Chromium): typing the Pine script's ticker in Settings refreshes the live interpretation
(no need to edit the script or leave the field), the report's trade list labels its prices as traded and shows the
split-adjusted chart price beside a trade from before a later split (the chart tooltip gives both), and the rule
strip's 'entry' / 'exit' labels each keep to their own row."""
import glob
import json
import shutil
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, report, web
from backtester.strategy import Strategy

PORT = 8831


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


browser = pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")


def _frame(seed=1, n=420, split_at=None):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame({"open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c,
                       "volume": rng.integers(5e5, 2e6, n).astype(float), "quote_close": c, "adj_close": c, "dividend": 0.0,
                       "split": 0.0}, index=idx)
    if split_at is not None:
        df.iloc[split_at, df.columns.get_loc("split")] = 10.0     # a 10:1 split: earlier prices were 10x as traded
    return df


@pytest.fixture
def fake(monkeypatch):
    frames = {"XR": _frame(1), "XS": _frame(2, split_at=300), "SPY": _frame(3)}

    def load(t):
        if data.canonical(t) not in frames:
            raise FileNotFoundError(t)
        return frames[data.canonical(t)]
    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {t: load(t) for t in ts})
    monkeypatch.setattr(data, "available_tickers", lambda: sorted(frames))
    return frames


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


def _page(p, errs, width=1280):
    b = p.chromium.launch(executable_path=_chromium())
    pg = b.new_page(viewport={"width": width, "height": 900})
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    return b, pg


PINE = ('//@version=5\nstrategy("RSI2", overlay=true, initial_capital=100000)\nr = ta.rsi(close, 2)\n'
        'if r < 10\n    strategy.entry("L", strategy.long)\nif r > 70\n    strategy.close("L")\n')


@browser
def test_typing_the_pine_ticker_refreshes_the_interpretation(site, fake):
    from playwright.sync_api import sync_playwright
    errs: list[str] = []
    with sync_playwright() as p:
        b, pg = _page(p, errs)
        pg.set_default_timeout(60_000)
        pg.goto(site + "/#backtest")
        pg.fill("#text", PINE)
        pg.wait_for_function("document.getElementById('interp').textContent.includes('Which ticker')")
        pg.click("text=Settings (override anything in the sentence)")
        # typing only (no change event: the field keeps the focus)
        pg.locator("#o_ticker").press_sequentially("XR", delay=30)
        pg.wait_for_function("!document.getElementById('interp').textContent.includes('Which ticker') && "
                             "document.getElementById('interp').textContent.includes('XR')", timeout=15_000)
        assert pg.evaluate("document.activeElement.id") == "o_ticker"
        assert "rsi(close, 2) < 10" in pg.inner_text("#interp")
        # clearing it asks again
        pg.fill("#o_ticker", "")
        pg.wait_for_function("document.getElementById('interp').textContent.includes('Which ticker')", timeout=15_000)
        assert errs == [], errs
        b.close()


def _report(tmp_path):
    s = Strategy(cash_rate=None, universe=["XS"], entry="rsi(close, 2) < 20", exit_when="rsi(close, 2) > 70",
                 stop_loss=0.08)
    res = engine.run(s)
    assert (res.trades["entry_split_factor"] == 10).any() and (res.trades["entry_split_factor"] == 1).any()
    report.write_outputs(report.analyze(res, rf=0.0, sensitivity=False, mc=False), tmp_path, excel=False)
    return tmp_path / "report.html", res.trades


@browser
def test_trade_list_is_labelled_as_traded_with_the_chart_price(fake, tmp_path):
    from playwright.sync_api import sync_playwright
    path, trades = _report(tmp_path)
    before = trades[trades["entry_split_factor"] == 10].iloc[0]
    after = trades[trades["entry_split_factor"] == 1].iloc[0]
    errs: list[str] = []
    with sync_playwright() as p:
        b, pg = _page(p, errs, width=1400)
        pg.goto(path.as_uri())
        pg.wait_for_function("document.querySelector('#pxChart canvas') && document.querySelector('#pxChart canvas').dataset.panes")
        head = pg.inner_text("#trades thead")
        assert "Entry (as traded)" in head and "Exit (as traded)" in head
        rows = pg.locator("#trades tbody tr")
        txt = {int(rows.nth(i).inner_text().split("\t")[0]): rows.nth(i).inner_text() for i in range(rows.count())}
        k_before = int(trades.index.get_loc(before.name)) + 1
        k_after = int(trades.index.get_loc(after.name)) + 1
        assert f"{before.entry_price:,.2f} (chart {before.entry_price / 10:,.2f})" in txt[k_before]
        assert "(chart" not in txt[k_after].split("\t")[4]
        assert "as traded on the day" in pg.inner_text("#tradeNote")
        # the chart's tooltip over that entry gives the chart price and the price as traded
        rows.nth(k_before - 1).click()
        pg.locator("#pxChart").scroll_into_view_if_needed()
        pg.wait_for_timeout(1200)
        box = pg.locator("#pxChart").bounding_box()
        l, iw = json.loads(pg.get_attribute("#pxChart canvas", "data-plot"))
        found = None
        for x in range(int(l) + 1, int(l + iw), 2):
            pg.mouse.move(box["x"] + x, box["y"] + 120)
            t = pg.inner_text("#pxChart")
            if f"#{k_before} long entry" in t:
                found = t
                break
        assert found is not None
        assert f"{before.entry_price / 10:,.2f} (as traded {before.entry_price:,.2f})" in found
        # the rule strip's labels sit in their own rows: 'exit' at least one font height below 'entry'
        lab = json.loads(pg.get_attribute("#pxChart canvas", "data-rs-labels"))
        assert lab["exit"] - lab["entry"] >= lab["fs"] + 2
        assert errs == [], errs
        b.close()
