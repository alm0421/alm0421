"""Round 6 (Composer expert review): direction words in portfolio holdings (short = negative weight, sell /
exit / cover in a branch = cash, refused when ambiguous), identical branches and self-comparisons, one sign
convention for ranking by drawdown, fixed price levels read on quoted prices (quoted()), new phrasings
("the 2 of ... with the highest ...", "the best performing of ...", "beats BIL's", when / unless, "less than
10% below its 52 week high"), requested tickers (data/extra_tickers.txt), the Build page's oscillator default
and the community gallery API."""
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import composer_import as ci, data, expr, parser, portfolio as pf, web
from backtester.parser import ParseError

AVAIL = set(data.available_tickers())
needs = lambda *ts: pytest.mark.skipif(not set(ts) <= AVAIL, reason="price data not downloaded")  # noqa: E731
SHORT = lambda t: {"weights": "specified", "w": [-1.0, 2.0], "children": [{"asset": t}, {"cash": True}]}  # noqa: E731


# ------------------------------------------------------------ 1. direction words in holdings

@needs("SPY", "QQQ", "TQQQ", "SQQQ", "SOXL", "TMF", "UVXY")
@pytest.mark.parametrize("text, then, other", [
    ("if SPY is above its 200 day moving average then buy TQQQ else sell TQQQ", {"asset": "TQQQ"}, {"cash": True}),
    ("if QQQ RSI(10) is above 79 then short TQQQ else hold TQQQ", SHORT("TQQQ"), {"asset": "TQQQ"}),
    ("if QQQ RSI(10) is above 79 then exit TQQQ else hold TQQQ", {"cash": True}, {"asset": "TQQQ"}),
    ("if QQQ RSI(10) is above 79 then hold TQQQ else cover TQQQ", {"asset": "TQQQ"}, {"cash": True}),
    ("if QQQ RSI(10) is above 79 then hold UVXY else hold TQQQ short", {"asset": "UVXY"}, SHORT("TQQQ")),
    ("if QQQ RSI(10) is above 79 then go short SQQQ else sell everything", SHORT("SQQQ"), {"cash": True}),
    ("if QQQ RSI(10) is above 79 then go long TQQQ else exit", {"asset": "TQQQ"}, {"cash": True}),
    ("if QQQ RSI(10) is above 79 then sell short TQQQ else hold TQQQ", SHORT("TQQQ"), {"asset": "TQQQ"}),
])
def test_direction_words_in_branches(text, then, other):
    p = parser.parse(text)
    assert p.tree["then"] == then and p.tree["else"] == other
    if SHORT("TQQQ") in (then, other) or SHORT("SQQQ") in (then, other):
        assert any("short position" in n and "rebate" in n for n in p.notes)
    if {"cash": True} in (then, other):
        assert any("holding cash" in n for n in p.notes)


@needs("TQQQ", "SOXL", "SQQQ", "TMF", "SPY", "TLT", "SHY")
def test_direction_words_in_lists():
    p = parser.parse("equal weight TQQQ, SOXL and short SQQQ")
    assert p.tree == {"weights": "equal", "children": [{"asset": "TQQQ"}, {"asset": "SOXL"}, SHORT("SQQQ")]}
    with pytest.raises(ParseError, match="ambiguous"):
        parser.parse("hold TQQQ and sell TMF")
    with pytest.raises(ParseError, match="'sell'"):
        parser.parse("if SPY is above its 200 day moving average then hold TQQQ else hold TQQQ sell")
    p = parser.parse("hold 50% SPY and 50% short TLT")
    assert p.tree["children"][1] == SHORT("TLT")
    # names that start with long / short are not directions
    assert parser.parse("hold 60% SPY and 40% short-term treasuries").tree["children"][1] == {"asset": "SHY"}
    assert parser.parse("hold 60% SPY and 40% long treasuries").tree["children"][1] == {"asset": "TLT"}


@needs("SPY")
def test_short_holding_returns_minus_the_asset():
    p = pf.Portfolio(tree=SHORT("SPY"), rebalance="daily", cash_rate=None, short_rebate_spread=0.0, start="2015-01-01",
                     end="2016-12-31", maintenance_margin=0.0)
    r = pf.run(p)
    eq = r.equity
    df = data.load("SPY").reindex(eq.index)
    got = eq.pct_change().iloc[2:]
    want = -((df["close"] + df["dividend"]) / df["close"].shift(1) - 1).iloc[2:]   # a short pays the dividends
    assert np.allclose(got.to_numpy(), want.to_numpy(), atol=1e-9)


@needs("SPY", "TQQQ", "BIL")
def test_identical_branches_and_self_comparisons():
    p = parser.parse("if SPY is above its 200 day moving average then hold TQQQ else hold TQQQ")
    assert any("Both branches" in n and "changes nothing" in n for n in p.notes)
    with pytest.raises(ParseError, match="compares a value with itself"):
        parser.parse("if SPY price is above SPY price then hold TQQQ else hold BIL")
    with pytest.raises(ValueError, match="compares a value with itself"):
        pf.Portfolio(tree={"if": 'close > sym("SPY").close', "on": "SPY", "then": {"asset": "TQQQ"},
                           "else": {"asset": "BIL"}}).validate()
    assert pf._self_comparison("rsi(close, 10) > rsi(close, 20)", "SPY") is None


# ------------------------------------------------------------ 2. ranking by drawdown

@needs("TQQQ", "SOXL", "TECL", "SPY", "TLT")
def test_drawdown_rankings_share_one_sign_convention():
    a = parser.parse("hold the top 1 of TQQQ, SOXL and TECL by 10 day max drawdown").tree["filter"]
    b = parser.parse("hold the top 1 of TQQQ, SOXL and TECL by 10 day drawdown")
    assert a["by"] == "max_drawdown(tr, 10)" and a["select"] == "top"
    assert b.tree["filter"]["by"] == "-drawdown(close, 10)" and b.tree["filter"]["select"] == "top"
    assert any("largest drawdown" in n and "most drawn down" in n for n in b.notes)
    c = parser.parse("hold the top 1 of TQQQ, SOXL and TECL by smallest 10 day drawdown")
    assert c.tree["filter"]["select"] == "bottom" and any("smallest drawdown" in n for n in c.notes)
    d = parser.parse("whichever of SPY and TLT has the lowest 20 day drawdown")
    assert d.tree["filter"] == {"select": "bottom", "n": 1, "by": "-drawdown(close, 20)", "weights": "equal"}


def test_drawdown_magnitudes_are_positive():
    idx = pd.bdate_range("2020-01-01", periods=40)
    px = pd.Series(np.r_[np.linspace(100, 120, 20), np.linspace(120, 90, 20)], index=idx)
    df = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1e6, "dividend": 0.0, "adj_close": px})
    ns = expr.Namespace(df)
    assert expr.evaluate_value("-drawdown(close, 10)", ns).iloc[-1] > 0
    assert expr.evaluate_value("max_drawdown(tr, 10)", ns).iloc[-1] > 0


# ------------------------------------------------------------ 3. fixed price levels on quoted prices

def _div_frame(n=400):
    idx = pd.bdate_range("2018-01-01", periods=n)
    c = pd.Series(100 + np.sin(np.arange(n) / 9) * 5, index=idx)
    d = pd.Series(0.0, index=idx)
    d.iloc[21::21] = 1.0
    g = (c + d) / c.shift(1)
    adj = c.iloc[0] * g.fillna(1.0).cumprod()
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6, "dividend": d, "adj_close": adj})


def test_quoted_reads_the_quoted_prices_on_an_adjusted_basis():
    df = _div_frame()
    adj = expr.Namespace(df, price_basis="adjusted")
    q = expr.Namespace(df, price_basis="quoted")
    assert expr.evaluate_value("close", adj).iloc[-1] > df["close"].iloc[-1] + 5    # total-return level grew
    assert np.allclose(expr.evaluate_value("quoted(close)", adj), df["close"])
    assert np.allclose(expr.evaluate_value("quoted(sma(close, 20))", adj).dropna(), expr.evaluate_value("sma(close, 20)", q).dropna())
    assert expr.open_safe("quoted(ref(close, 1)) > 100") and not expr.open_safe("quoted(close) > 100")
    with pytest.raises(ValueError, match="one expression"):
        expr.compile_expr("quoted(close, 2) > 1")


@pytest.mark.parametrize("rule, want", [
    ("close > 400", "quoted(close) > 400"),
    ("(close > 400)", "(quoted(close) > 400)"),
    ("sma(close, 200) < 350 and close > sma(close, 200)", "quoted(sma(close, 200)) < 350 and close > sma(close, 200)"),
    ('sym("SPY").close < 300', 'quoted(sym("SPY").close) < 300'),
    ("400 < close < 500", "400 < quoted(close) < 500"),
    ("stdev(close, 20) > 5", "quoted(stdev(close, 20)) > 5"),
    ("bb_upper(20, 2) < 450", "quoted(bb_upper(20, 2)) < 450"),
    ("rsi(close, 10) > 79", "rsi(close, 10) > 79"),
    ("close > sma(close, 200)", "close > sma(close, 200)"),
    ("tret(tr, 20) > 0.05", "tret(tr, 20) > 0.05"),
])
def test_quote_levels(rule, want):
    assert pf.quote_levels(rule) == want


@needs("SPY", "TQQQ", "BIL")
def test_price_level_conditions_use_quoted_prices():
    p = parser.parse("if SPY price is above 400 then hold TQQQ else hold BIL")
    assert p.tree["if"] == "(quoted(close) > 400)"
    assert any("compares a price level with a fixed number" in n and "quoted" in n for n in p.notes)
    q = parser.parse("if SPY price is above 400 then hold TQQQ else hold BIL, rebalance daily")
    assert q.tree["if"] == "(quoted(close) > 400)"
    # a quoted price basis needs no rewrite
    r = pf.Portfolio(tree={"if": "close > 400", "on": "SPY", "then": {"asset": "TQQQ"}, "else": {"asset": "BIL"}},
                     price_basis="quoted")
    r.validate()
    assert r.tree["if"] == "close > 400"
    # the rule really is the quoted level: held on days SPY closed above 400
    spec = pf.Portfolio(tree={"if": "close > 400", "on": "SPY", "then": {"asset": "TQQQ"}, "else": {"asset": "BIL"}},
                        rebalance="daily", start="2021-01-01", end="2023-12-31")
    res = pf.run(spec)
    assert spec.tree["if"] == "quoted(close) > 400" and res.equity.iloc[-1] > 0


def test_composer_price_levels_are_quoted():
    root = {"step": "root", "name": "T", "rebalance": "daily", "children": [{"step": "if", "children": [
        {"step": "if-child", "is-else-condition?": False, "lhs-fn": "current-price", "lhs-val": "SPY", "comparator": "gt",
         "rhs-val": "400", "rhs-fixed-value?": True, "children": [{"step": "asset", "ticker": "TQQQ"}]},
        {"step": "if-child", "is-else-condition?": True, "children": [{"step": "asset", "ticker": "BIL"}]}]}]}
    s = ci.convert(root)
    assert s["tree"]["if"] == "quoted(close) > 400"
    assert any("quoted prices" in n for n in s["notes"])


# ------------------------------------------------------------ 4. phrasings

@needs("SPY", "QQQ", "IWM", "BIL", "TLT")
@pytest.mark.parametrize("text, check", [
    ("hold the 2 of SPY, QQQ, IWM with the highest 10 day return",
     lambda t: t["filter"] == {"select": "top", "n": 2, "by": "tret(tr, 10)", "weights": "equal"}),
    ("hold the 1 of SPY, QQQ and IWM with the lowest 20 day volatility",
     lambda t: t["filter"]["select"] == "bottom" and t["filter"]["by"] == "volatility(20)"),
    ("hold the best performing of SPY, QQQ and IWM over 10 days",
     lambda t: t["filter"] == {"select": "top", "n": 1, "by": "tret(tr, 10)", "weights": "equal"}),
    ("hold the worst performing of SPY, QQQ and IWM over 10 days",
     lambda t: t["filter"]["select"] == "bottom" and t["filter"]["n"] == 1),
    ("hold the top 2 of SPY, QQQ and IWM by 60 day return, only if their 60 day return beats BIL's",
     lambda t: t["filter"]["require"] == 'tret(tr, 60) > tret(sym("BIL").tr, 60)' and t["fallback"] == {"cash": True}),
    ("hold the top 2 of SPY, QQQ and IWM by 60 day return, only if their 60 day return beats BIL's, otherwise TLT",
     lambda t: t["filter"]["require"] == 'tret(tr, 60) > tret(sym("BIL").tr, 60)' and t["fallback"] == {"asset": "TLT"}),
    ("when SPY is above its 200 day moving average hold QQQ else hold TLT",
     lambda t: t == {"if": "(close > sma(close, 200))", "on": "SPY", "then": {"asset": "QQQ"}, "else": {"asset": "TLT"}}),
    ("unless SPY is below its 200 day moving average hold QQQ, otherwise TLT",
     lambda t: t == {"if": "(close < sma(close, 200))", "on": "SPY", "then": {"asset": "TLT"}, "else": {"asset": "QQQ"}}),
    ("if SPY is less than 10% below its 52 week high then hold QQQ else hold TLT",
     lambda t: t["if"] == "(drawdown(close, 252) >= -0.1)"),
])
def test_new_phrasings(text, check):
    assert check(parser.parse(text).tree)


@needs("SPY")
def test_less_than_below_high_in_signal_rules():
    s = parser.parse("buy SPY when it is less than 10% below its 52 week high, sell when it is more than 20% below its 52 week high")
    assert s.entry == "(drawdown(close, 252) >= -0.1)" and s.exit_when == "(drawdown(close, 252) <= -0.2)"


# ------------------------------------------------------------ 5. requested tickers

def test_unknown_ticker_messages_point_to_extra_tickers():
    assert "data/extra_tickers.txt" in data.unknown_ticker_message("ZZZQX")
    with pytest.raises(ParseError, match=r"extra_tickers\.txt"):
        parser.parse("hold 50% SPY and 50% ZZZQX")


def test_fetch_script_reads_extra_tickers(tmp_path):
    root = Path(__file__).resolve().parent.parent
    pytest.importorskip("yfinance")
    pytest.importorskip("requests")
    spec = importlib.util.spec_from_file_location("fetch_data_t", root / "scripts" / "fetch_data.py")
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    f = tmp_path / "extra.txt"
    f.write_text("# comment\nboil, KOLD  # leveraged gas\n$mstr\nBRK-B ^VIX\nnot a ticker!\nBOIL\n")
    assert fd.requested_tickers(f) == ["BOIL", "KOLD", "MSTR", "BRK-B", "^VIX", "NOT", "A"]
    assert fd.requested_tickers(tmp_path / "missing.txt") == []
    for t in ("BOIL", "KOLD", "UTSL", "WEBL", "BULZ", "DPST", "JNK", "IGV", "FXE", "SPHB", "NAIL", "GUSH", "DRIP"):
        assert t in fd.ETFS, t
    assert "MSTR" in fd.STOCKS and "MSTR" not in fd.ETFS and "COIN" in data.LARGE_STOCKS


# ------------------------------------------------------------ 6. Build page

def test_build_page_defaults_oscillators_to_a_number_and_warns_on_levels():
    html = web.APP.read_text()
    assert "UNIT[st.lhs.k] === 'osc' && st.rhsMode === 'ind'" in html   # RSI on the left: the right side becomes a number
    assert "Different units" in html and "read on quoted prices" in html
    assert "unquote(" in html                                           # quoted(...) rules still open in the picker


# ------------------------------------------------------------ 7. community gallery API

@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(web, "COMMUNITY", tmp_path / "community.json")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(base, path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@needs("SPY", "TLT", "QQQ")
def test_community_publish_list_search_sort(server, tmp_path):
    s, j = call(server, "/api/community")
    assert s == 200 and j == {"strategies": [], "count": 0}
    tree = {"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "TLT"}]}
    s, a = call(server, "/api/community/publish", {"name": "Classic 60/40", "author": "ann", "description": "balanced",
                                                  "spec": {"kind": "allocation", "tree": tree, "start": "2010-01-01"}})
    assert s == 200, a
    assert a["entry"]["name"] == "Classic 60/40" and a["entry"]["author"] == "ann" and a["entry"]["spec"]["name"] == "Classic 60/40"
    assert a["entry"]["stats"]["cagr"] is not None and a["entry"]["run_id"] and a["url"].endswith("report.html")
    s, b = call(server, "/api/community/publish", {"name": "QQQ trend", "author": "",
                                                  "text": "if SPY is above its 200 day moving average then hold QQQ else sell QQQ, since 2010"})
    assert s == 200, b
    assert b["entry"]["author"] == "anonymous" and b["entry"]["text"].startswith("if SPY")
    assert b["entry"]["spec"]["tree"]["else"] == {"cash": True}
    s, j = call(server, "/api/community")
    assert j["count"] == 2 and [x["name"] for x in j["strategies"]] == ["QQQ trend", "Classic 60/40"]
    assert all("hay" not in x for x in j["strategies"])
    for sort, key, desc in (("cagr", "cagr", True), ("sharpe", "sharpe", True), ("max_drawdown", "max_drawdown", True)):
        vals = [x["stats"][key] for x in call(server, f"/api/community?sort={sort}")[1]["strategies"]]
        assert vals == sorted(vals, reverse=desc), sort
    assert [x["name"] for x in call(server, "/api/community?q=tlt")[1]["strategies"]] == ["Classic 60/40"]
    assert [x["name"] for x in call(server, "/api/community?q=ann+balanced")[1]["strategies"]] == ["Classic 60/40"]
    assert call(server, "/api/community?q=nothing-matches")[1]["count"] == 0
    assert call(server, "/api/community?sort=bogus")[0] == 400
    # forking = loading the stored spec; running it again gives the same numbers
    spec = j["strategies"][1]["spec"]
    s, r = call(server, "/api/run", {"spec": spec})
    assert s == 200 and abs(r["summary"]["cagr"] - a["entry"]["stats"]["cagr"]) < 1e-9
    assert json.loads((tmp_path / "community.json").read_text())[0]["name"] == "QQQ trend"


@needs("SPY")
@pytest.mark.parametrize("body, needle", [
    ({"name": "", "text": "buy and hold SPY"}, "name"),
    ({"name": "x"}, "Nothing to publish"),
    ({"name": "x" * 81, "text": "buy and hold SPY"}, "too long"),
    ({"name": "x", "text": "hold TQQQ and sell TMF"}, "ambiguous"),
])
def test_community_publish_errors(server, body, needle):
    s, j = call(server, "/api/community/publish", body)
    assert s == 400 and needle in j["error"]


# ------------------------------------------------------------ the Community page in headless Chromium

def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    import glob
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


@pytest.mark.skipif(_chromium() is None or not {"SPY", "TLT", "QQQ"} <= AVAIL, reason="headless Chromium or data not available")
def test_community_page_publish_search_sort_fork_run(server):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        # publish the Build page's default 60/40, then a sentence from the Backtest page
        pg.goto(server + "/#build")
        pg.wait_for_selector("#tree .node")
        pg.click("#buildPublish summary")
        box = pg.locator("#buildPublish")
        box.locator("input").nth(0).fill("Plain 60/40")
        box.locator("button", has_text="Publish").click()
        pg.wait_for_function("document.getElementById('buildPublish').textContent.includes('Published')", timeout=120000)
        pg.goto(server + "/#backtest")
        pg.fill("#text", "if SPY is above its 200 day moving average then hold QQQ else sell QQQ, since 2010")
        pg.click("#btPublish summary")
        box = pg.locator("#btPublish")
        box.locator("input").nth(0).fill("SPY trend QQQ")
        box.locator("input").nth(1).fill("tester")
        box.locator("button", has_text="Publish").click()
        pg.wait_for_function("document.getElementById('btPublish').textContent.includes('Published')", timeout=120000)
        # list, sort, search
        pg.goto(server + "/#community")
        pg.wait_for_selector("#comList tr[data-id]")
        names = lambda: pg.locator("#comList .com-name").all_inner_texts()  # noqa: E731
        assert names() == ["SPY trend QQQ", "Plain 60/40"]
        by_cagr = [x["name"] for x in web.api_community({"sort": ["cagr"]})["strategies"]]
        pg.select_option("#comSort", "cagr")
        pg.wait_for_function(f"[...document.querySelectorAll('#comList .com-name')].map(e => e.textContent).join('|') === {json.dumps('|'.join(by_cagr))}")
        pg.fill("#comSearch", "tester")
        pg.wait_for_function("document.querySelectorAll('#comList .com-name').length === 1")
        assert names() == ["SPY trend QQQ"]
        # fork opens it in the Build editor
        pg.locator("#comList .com-fork").first.click()
        pg.wait_for_function("location.hash.startsWith('#build')")
        assert "Forked" in pg.inner_text("#buildStatus")
        assert pg.evaluate("document.getElementById('toJson').click(), JSON.parse(document.getElementById('specJson').value).tree.else") == {"cash": True}
        # run from the list
        pg.goto(server + "/#community")
        pg.wait_for_selector("#comList .com-run")
        pg.locator("#comList .com-run").first.click()
        pg.wait_for_function("!document.getElementById('result').classList.contains('hide')", timeout=120000)
        assert "CAGR" in pg.inner_text("#resultMeta")
        assert errs == []
        b.close()
