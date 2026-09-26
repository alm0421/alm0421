"""API-level tests for the site (backtester/web.py): a real HTTP server on a free port."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from backtester import data, web

HAVE = {"MSFT", "QQQ", "SPY", "TLT", "GLD"} <= set(data.available_tickers())
pytestmark = pytest.mark.skipif(not HAVE, reason="price data not downloaded")

# The Build page's default payloads (keep in sync with webapp.html: s_* inputs and the default tree).
SIGNAL_FORM_DEFAULT = {
    "side": "long", "entry_fill": "close", "entry_order": "market", "entry": "down_days >= 5", "hold_bars": 1,
    "hold_exit_fill": "close", "max_positions": 1, "sizing": "percent", "leverage": 1, "pyramiding": 1,
    "slippage_bps": 0, "commission": 0, "commission_pct": 0, "borrow_fee": 0, "capital": 10000,
    "universe": ["MSFT"], "short_entry": None, "exit_when": None, "entry_level": None,
    # blank inputs the page may send as empty strings / nulls
    "position_size": None, "risk_per_trade": "", "target_vol": None, "fixed_amount": None, "stop_loss": None,
}
PORTFOLIO_SETTINGS_DEFAULT = {
    "kind": "allocation", "rebalance": "monthly", "fill": "close", "capital": 10000, "contribution": 0,
    "contribution_freq": "monthly", "withdrawal": 0, "withdrawal_pct": 0, "withdrawal_freq": "yearly",
    "inflation_adjust": False, "slippage_bps": 0, "commission_pct": 0, "leverage": 1, "margin_rate": 0,
    "expense_ratio": 0, "drift_band": "",
}
DEFAULT_TREE = {"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "TLT"}]}
BLOCK_DEFAULTS = [  # every block type as the editor creates it
    {"asset": "SPY"},
    {"cash": True},
    {"weights": "equal", "children": [{"asset": "SPY"}, {"asset": "TLT"}]},
    {"weights": "inverse_vol", "lookback": 20, "children": [{"asset": "SPY"}, {"asset": "TLT"}]},
    {"weights": "risk_parity", "lookback": 60, "children": [{"asset": "SPY"}, {"asset": "TLT"}]},
    {"if": "close > sma(close, 200)", "on": "SPY", "then": {"asset": "QQQ"}, "else": {"asset": "TLT"}},
    {"if": 'tret(tr, 63) > tret(sym("TLT").tr, 63)', "on": "SPY", "then": {"asset": "SPY"}, "else": {"asset": "TLT"}},
    {"filter": {"select": "top", "n": 2, "by": "tret(tr, 126)", "weights": "equal"}, "universe": "children",
     "children": [{"asset": "QQQ"}, {"asset": "SPY"}, {"asset": "TLT"}, {"asset": "GLD"}], "fallback": {"cash": True}},
]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    old = web.RUNS
    web.RUNS = runs
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    web.RUNS = old


def call(base, path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_build_form_defaults_run(server):
    code, j = call(server, "/api/run", {"spec": SIGNAL_FORM_DEFAULT, "sensitivity": False})
    assert code == 200, j
    assert "100% of equity each" in j["interpretation"]
    code, j = call(server, "/api/run", {"spec": {**PORTFOLIO_SETTINGS_DEFAULT, "tree": DEFAULT_TREE}, "sensitivity": False})
    assert code == 200, j
    assert j["summary"]["start"] != "None" and j["summary"]["rebalances"] >= 1
    # the header shows the first real trading day, not the synthetic starting-capital point
    first = str(data.load("TLT").index[0].date() if data.load("TLT").index[0] > data.load("SPY").index[0]
                else data.load("SPY").index[0].date())
    assert j["summary"]["start"] == first


@pytest.mark.parametrize("node", BLOCK_DEFAULTS)
def test_every_block_type_parses(server, node):
    code, j = call(server, "/api/parse", {"spec": {**PORTFOLIO_SETTINGS_DEFAULT, "tree": node}})
    assert code == 200, j


@pytest.mark.parametrize("body, needle", [
    ({"spec": {**SIGNAL_FORM_DEFAULT, "entry": "foo > 1"}}, "'foo'"),
    ({"spec": {**SIGNAL_FORM_DEFAULT, "entry": "rsi(close, 2 < 10"}}, "brackets"),
    ({"spec": {**SIGNAL_FORM_DEFAULT, "universe": ["APPL"]}}, "AAPL"),
    ({"spec": {**SIGNAL_FORM_DEFAULT, "hold_bars": "abc"}}, "not a number"),
    ({"spec": {"universe": ["MSFT"]}}, "missing"),
    ({"spec": {**PORTFOLIO_SETTINGS_DEFAULT, "tree": {"filter": {"select": "top"}, "children": []}}}, "by"),
    ({"spec": {**PORTFOLIO_SETTINGS_DEFAULT, "tree": {"asset": "SPYY"}}}, "SPY"),
    ({"text": "buy APPL when RSI(2) is below 10, hold 1 day"}, "AAPL"),
    ({"spec": "nonsense"}, "object"),
    ({"share": "garbage!!"}, "share link"),
])
def test_user_errors_are_readable_400s(server, body, needle):
    code, j = call(server, "/api/parse", body)
    assert code == 400, j
    msg = j["error"]
    assert needle in msg
    assert "Error" not in msg and "Traceback" not in msg
    assert len(msg) < 400  # no dump of every ticker


def test_share_link_reproduces_the_run(server):
    body = {"text": "buy QQQ at the close when RSI(2) is below 10, hold 3 days",
            "options": {"capital": 25000, "start": "2012-01-01", "end": "2020-12-31", "slippage_bps": 3,
                        "commission": 1, "benchmark": "SPY"}, "rf": 0, "sensitivity": False}
    code, a = call(server, "/api/run", body)
    assert code == 200, a
    code, dec = call(server, "/api/share", {"share": a["share"]})
    assert code == 200, dec
    s = dec["spec"]
    assert (s["capital"], s["start"], s["end"], s["slippage_bps"], s["commission"], s["benchmark"]) == \
        (25000, "2012-01-01", "2020-12-31", 3, 1, "SPY")
    assert dec["rf"] == 0
    code, b = call(server, "/api/run", {"share": a["share"], "sensitivity": False})
    assert code == 200, b
    for k in ("cagr", "sharpe", "max_drawdown", "final", "start", "end", "trades"):
        assert a["summary"][k] == b["summary"][k], k
    assert b["share"] == a["share"]


def test_share_link_for_a_built_portfolio(server):
    spec = {**PORTFOLIO_SETTINGS_DEFAULT, "tree": BLOCK_DEFAULTS[-1], "leverage": 1.5, "expense_ratio": 0.001,
            "start": "2010-01-01"}
    code, a = call(server, "/api/run", {"spec": spec, "sensitivity": False})
    assert code == 200, a
    code, b = call(server, "/api/run", {"share": a["share"], "sensitivity": False})
    assert code == 200, b
    for k in ("cagr", "sharpe", "max_drawdown", "final", "start", "end"):
        assert a["summary"][k] == b["summary"][k], k


def test_todays_orders(server):
    code, j = call(server, "/api/orders", {"text": "hold 60% SPY and 40% TLT", "account_value": "100,000",
                                            "holdings": "symbol,qty\nSPY,10\nAAPL,5"})
    assert code == 200, j
    o = {r["ticker"]: r for r in j["orders"]}
    assert o["AAPL"]["side"] == "SELL" and o["AAPL"]["shares"] == 5
    assert o["SPY"]["target_shares"] == int(60000 / o["SPY"]["est_price"])
    assert o["TLT"]["side"] == "BUY"
    assert j["ib_csv"].splitlines()[0].startswith("Action,Quantity,Symbol")
    assert "SELL,5,AAPL,STK,SMART,USD" in j["ib_csv"]
    code, j = call(server, "/api/orders", {"spec": SIGNAL_FORM_DEFAULT, "account_value": 50000})
    assert code == 200, j
    code, j = call(server, "/api/orders", {"text": "hold 60% SPY and 40% TLT", "account_value": ""})
    assert code == 400


def test_gallery(server):
    code, j = call(server, "/api/gallery")
    assert code == 200 and len(j["library"]) > 10
    code, s = call(server, "/api/gallery/stats", {"text": j["library"][0]["text"]})
    assert code == 200, s
    assert s["stats"]["cagr"] is not None
    code, j = call(server, "/api/gallery")
    assert j["library"][0]["stats"]["cagr"] == s["stats"]["cagr"]
