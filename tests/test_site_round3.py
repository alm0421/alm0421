"""Round 3 of the site: Composer symphony import, library/gallery additions, run reuse and the favicon."""
import copy
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from backtester import composer_import as ci
from backtester import data, library, web
from backtester import portfolio as pf
from backtester.portfolio import Portfolio

FIXTURE = Path(__file__).with_name("fixtures") / "composer_tqqq_ftlt.json"
AVAILABLE = set(data.available_tickers())
needs = lambda *t: pytest.mark.skipif(not set(t) <= AVAILABLE, reason="price data not downloaded")  # noqa: E731


def _conv(obj):
    """ci.convert with the Composer block ids left out of the tree (these tests check its structure)."""
    d = ci.convert(obj)
    d["tree"] = pf.without_ids(d["tree"])
    return d


def asset(t, **kw):
    return {"id": f"id-{t}", "step": "asset", "ticker": t, "name": f"{t} fund", **kw}


def root(*children, **kw):
    return {"id": "root", "step": "root", "name": "Test", "description": "", "rebalance": "daily",
            "children": list(children), **kw}


def cond(lhs_fn, lhs, cmp, rhs, *children, window=None, rhs_fn=None, rhs_window=None):
    c = {"step": "if-child", "is-else-condition?": False, "lhs-fn": lhs_fn, "lhs-val": lhs, "comparator": cmp,
         "rhs-val": rhs, "rhs-fixed-value?": rhs_fn is None, "children": list(children)}
    if window is not None:
        c["lhs-window-days"] = str(window)
    if rhs_fn:
        c["rhs-fn"] = rhs_fn
    if rhs_window is not None:
        c["rhs-window-days"] = str(rhs_window)
    return c


def other(*children):
    return {"step": "if-child", "is-else-condition?": True, "children": list(children)}


# ------------------------------------------------------------------ Composer import: node types

def test_asset_and_ticker_formats():
    assert _conv(root(asset("SPY")))["tree"] == {"asset": "SPY"}
    assert _conv(root(asset("EQUITIES::TQQQ//USD")))["tree"] == {"asset": "TQQQ"}
    assert _conv(root(asset("CRYPTO::BTC//USD")))["tree"] == {"asset": "BTC-USD"}
    assert _conv(root(asset("BRK.B")))["tree"] == {"asset": "BRK-B"}


def test_equal_weights_and_group():
    s = _conv(root({"step": "wt-cash-equal", "children": [
        asset("SPY"), {"step": "group", "name": "Bonds", "children": [
            {"step": "wt-cash-equal", "children": [asset("TLT"), asset("IEF")]}]}]}))
    assert s["tree"] == {"weights": "equal", "children": [
        {"asset": "SPY"}, {"weights": "equal", "children": [{"asset": "TLT"}, {"asset": "IEF"}], "name": "Bonds"}]}
    # (the group's name is kept, for the export back to Composer)
    # a single child needs no wrapper
    assert _conv(root({"step": "wt-cash-equal", "children": [asset("QQQ")]}))["tree"] == {"asset": "QQQ"}


def test_specified_weights():
    s = _conv(root({"step": "wt-cash-specified", "children": [
        asset("UPRO", weight={"num": 55, "den": 100}), asset("TMF", weight={"num": "45", "den": "100"})]}))
    assert s["tree"] == {"weights": "specified", "w": [0.55, 0.45], "children": [{"asset": "UPRO"}, {"asset": "TMF"}]}
    # weights that don't add up are scaled, with a note
    s = _conv(root({"step": "wt-cash-specified", "children": [
        asset("SPY", weight={"num": 30, "den": 100}), asset("TLT", weight={"num": 30, "den": 100})]}))
    assert s["tree"]["w"] == [0.5, 0.5] and any("scaled" in n for n in s["notes"])
    with pytest.raises(ci.ComposerImportError, match="weight"):
        _conv(root({"step": "wt-cash-specified", "children": [asset("SPY")]}))


def test_inverse_vol():
    s = _conv(root({"step": "wt-inverse-vol", "window-days": "30", "children": [asset("SPY"), asset("TLT")]}))
    assert s["tree"] == {"weights": "inverse_vol", "lookback": 30, "children": [{"asset": "SPY"}, {"asset": "TLT"}]}


def test_if_fixed_value_and_percent_functions():
    s = _conv(root({"step": "if", "children": [
        cond("relative-strength-index", "TQQQ", "gt", "79", asset("UVXY"), window=10), other(asset("TQQQ"))]}))
    assert s["tree"] == {"if": "rsi(close, 10) > 79", "on": "TQQQ", "then": {"asset": "UVXY"}, "else": {"asset": "TQQQ"}}
    # cumulative return thresholds are in percent in Composer: -12 means -12%
    s = _conv(root({"step": "if", "children": [
        cond("cumulative-return", "QQQ", "lte", "-12", asset("TQQQ"), window=5), other(asset("BIL"))]}))
    assert s["tree"]["if"] == "tret(tr, 5) <= -0.12"


@pytest.mark.parametrize("fn, expect, rhs, rhs_expect", [
    ("relative-strength-index", "rsi(close, 14)", "30", "30"),
    ("cumulative-return", "tret(tr, 14)", "5", "0.05"),
    ("moving-average-price", "quoted(sma(close, 14))", "400", "400"),       # price levels vs a fixed number: quoted
    ("exponential-moving-average-price", "quoted(ema(close, 14))", "400", "400"),
    ("moving-average-return", "ma_return(tr, 14)", "0.5", "0.005"),
    ("standard-deviation-return", "stdev_return(tr, 14)", "3", "0.03"),
    ("standard-deviation-price", "quoted(stdev(close, 14))", "12.5", "12.5"),
    ("max-drawdown", "max_drawdown(tr, 14)", "10", "0.1"),
])
def test_function_map(fn, expect, rhs, rhs_expect):
    s = _conv(root({"step": "if", "children": [cond(fn, "SPY", "gte", rhs, asset("SPY"), window=14),
                                                    other(asset("BIL"))]}))
    assert s["tree"]["if"] == f"{expect} >= {rhs_expect}"


def test_two_ticker_comparisons_use_sym():
    s = _conv(root({"step": "if", "children": [
        cond("cumulative-return", "SPY", "gt", "TLT", asset("SPY"), window=63, rhs_fn="cumulative-return", rhs_window=63),
        other(asset("TLT"))]}))
    assert s["tree"]["if"] == 'tret(tr, 63) > tret(sym("TLT").tr, 63)' and s["tree"]["on"] == "SPY"
    s = _conv(root({"step": "if", "children": [
        cond("current-price", "SPY", "lt", "SPY", asset("BIL"), rhs_fn="moving-average-price", rhs_window=200),
        other(asset("SPY"))]}))
    assert s["tree"]["if"] == "close < sma(close, 200)"
    s = _conv(root({"step": "if", "children": [
        cond("current-price", "QQQ", "gt", "SPY", asset("QQQ"), rhs_fn="current-price"), other(asset("SPY"))]}))
    # price levels of two tickers are only comparable as quoted (each total-return level starts at its own first close)
    assert s["tree"]["if"] == 'quoted(close) > quoted(sym("SPY").close)'
    # the window may also come as lhs-fn-params (newer exports)
    c = cond("relative-strength-index", "SPY", "lt", "30", asset("UPRO"))
    c["lhs-fn-params"] = {"window": 10}
    assert _conv(root({"step": "if", "children": [c, other(asset("SPY"))]}))["tree"]["if"] == "rsi(close, 10) < 30"


def test_else_if_chain_and_missing_else():
    s = _conv(root({"step": "if", "children": [
        cond("relative-strength-index", "SPY", "gt", "80", asset("UVXY"), window=10),
        cond("relative-strength-index", "SPY", "lt", "30", asset("UPRO"), window=10),
        other(asset("SPY"))]}))
    t = s["tree"]
    assert t["then"] == {"asset": "UVXY"} and t["else"]["if"] == "rsi(close, 10) < 30"
    assert t["else"]["then"] == {"asset": "UPRO"} and t["else"]["else"] == {"asset": "SPY"}
    s = _conv(root({"step": "if", "children": [cond("relative-strength-index", "SPY", "gt", "80", asset("UVXY"), window=10)]}))
    assert s["tree"]["else"] == {"cash": True} and any("no else" in n for n in s["notes"])


def test_filter():
    f = {"step": "filter", "sort-by-fn": "relative-strength-index", "sort-by-window-days": "10", "select-fn": "bottom",
         "select-n": "2", "children": [asset("SOXL"), asset("TECL"), asset("TQQQ")]}
    s = _conv(root(f))
    assert s["tree"] == {"filter": {"select": "bottom", "n": 2, "by": "rsi(close, 10)", "weights": "equal"},
                         "universe": "children", "children": [{"asset": "SOXL"}, {"asset": "TECL"}, {"asset": "TQQQ"}],
                         "fallback": {"cash": True}}
    # groups inside a filter are kept as they are (ranked by their own NAV by the simulator)
    f2 = {**f, "children": [{"step": "group", "name": "A", "children": [asset("SPY")]},
                            {"step": "wt-cash-equal", "children": [asset("TLT"), asset("GLD")]}]}
    assert _conv(root(f2))["tree"]["children"][1] == {"weights": "equal", "children": [{"asset": "TLT"}, {"asset": "GLD"}]}


def test_rebalance_settings():
    assert _conv(root(asset("SPY"), rebalance="monthly"))["rebalance"] == "monthly"
    s = _conv(root(asset("SPY"), **{"rebalance": "none", "rebalance-corridor-width": 0.05}))
    assert s["rebalance"] == "none" and s["drift_band"] == 0.05
    s = _conv(root(asset("SPY"), **{"rebalance": "none", "rebalance-corridor-width": "10"}))
    assert s["drift_band"] == 0.1
    with pytest.raises(ci.ComposerImportError, match="rebalance"):
        _conv(root(asset("SPY"), rebalance="hourly"))


@pytest.mark.parametrize("sym, needle", [
    (root(asset("SPY", **{"mystery-field": 1})), "mystery-field"),
    (root({"step": "wt-magic", "children": [asset("SPY")]}), "wt-magic"),
    (root({"step": "if", "children": [cond("hurst-exponent", "SPY", "gt", "0.5", asset("SPY"), window=10), other(asset("BIL"))]}), "hurst-exponent"),
    (root({"step": "if", "children": [cond("relative-strength-index", "SPY", "eq", "50", asset("SPY"), window=10), other(asset("BIL"))]}), "comparator"),
    (root({"step": "if", "children": [cond("relative-strength-index", "SPY", "gt", "50", asset("SPY")), other(asset("BIL"))]}), "window"),
    (root({"step": "wt-cash-equal", "children": []}), "empty"),
    ({"nothing": "here"}, "Composer symphony"),
    ("{not json", "Not valid JSON"),
])
def test_unknown_or_bad_input_is_a_clear_error(sym, needle):
    with pytest.raises(ci.ComposerImportError) as e:
        _conv(sym)
    assert needle in str(e.value)


def test_wrappers_and_text_input():
    sym = root(asset("SPY"))
    assert _conv(json.dumps(sym))["tree"] == {"asset": "SPY"}
    assert _conv({"symphony": sym})["tree"] == {"asset": "SPY"}
    assert _conv({"fields": {"score": json.dumps(sym)}})["tree"] == {"asset": "SPY"}


@needs("SPY", "TQQQ", "UVXY", "TECL", "QQQ", "TLT", "GLD", "SHY", "BTAL", "SHV")
def test_fixture_imports_validates_and_runs():
    from backtester import runner
    d = ci.load(FIXTURE)
    assert d["name"] == "TQQQ For The Long Term (sample)" and d["rebalance"] == "daily"
    t = pf.without_ids(d["tree"])
    assert (t["if"], t["on"]) == ("close > sma(close, 200)", "SPY")
    assert t["then"]["if"] == "rsi(close, 10) > 79" and t["else"]["then"] == {"asset": "TECL"}
    hedge = t["else"]["else"]["else"]
    assert hedge["weights"] == "specified" and hedge["w"] == [0.6, 0.4]
    assert hedge["children"][0]["filter"]["by"] == "tret(tr, 20)"
    assert hedge["children"][1] == {"weights": "inverse_vol", "lookback": 30, "children": [{"asset": "BTAL"}, {"asset": "SHV"}]}
    p = Portfolio.from_dict(copy.deepcopy(d))
    p.validate()
    res = runner.run(p)
    assert res.equity.iloc[-1] > 0 and {"TQQQ", "UVXY"} <= set(res.holdings.columns)


@needs("SPY", "TQQQ", "UVXY", "TECL", "QQQ", "TLT", "GLD", "SHY", "BTAL", "SHV")
def test_cli_import_composer(tmp_path, capsys):
    from backtester.__main__ import main
    out = tmp_path / "spec.json"
    assert main(["import-composer", str(FIXTURE), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "rsi(close, 10) > 79 (on TQQQ)" in printed
    spec = json.loads(out.read_text())
    assert spec["kind"] == "allocation" and spec["tree"]["on"] == "SPY"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(root(asset("SPY", oops=True))))
    assert main(["import-composer", str(bad)]) == 2
    assert "oops" in capsys.readouterr().err


# ------------------------------------------------------------------ library

def test_library_entries_all_parse_and_validate():
    ids = [x["id"] for x in library.LIBRARY]
    assert len(ids) == len(set(ids))
    for x in library.LIBRARY:
        assert x["tags"] and set(x["tags"]) <= set(library.TAGS), x["name"]
        assert ("text" in x) != ("spec" in x), x["name"]
        missing = [t for t in _tickers(x) if t not in AVAILABLE]
        if missing:
            continue
        spec = library.entry_spec(x)
        spec.validate()
    assert len(library.LIBRARY) >= 25 + 15  # 25 originals + the Composer-style additions
    for tag in library.TAGS:
        assert any(tag in x["tags"] for x in library.LIBRARY), tag
    assert sum("leveraged" in x["tags"] for x in library.LIBRARY) >= 10
    assert any("spec" in x for x in library.LIBRARY)


def _tickers(x):
    if "spec" in x:
        return Portfolio.from_dict(copy.deepcopy(x["spec"])).universe
    return []


@needs("SPY", "TQQQ", "UVXY", "TECL", "UPRO", "SQQQ", "TLT", "EFA", "AGG", "QQQ", "BIL")
def test_library_json_trees_run():
    from backtester import runner
    for x in library.LIBRARY:
        if "spec" in x:
            res = runner.run(library.entry_spec(x))
            assert res.equity.iloc[-1] > 0, x["name"]
    assert library.find("tqqq-for-the-long-term-ftlt")["name"] == "TQQQ For The Long Term (FTLT)"


# ------------------------------------------------------------------ site

HAVE = {"SPY", "TLT", "QQQ", "GLD"} <= AVAILABLE


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs3")
    old = web.RUNS
    web.RUNS = runs
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    web.RUNS = old


def call(base, path, body=None, raw=False):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            b = r.read()
            return r.status, (b, r.headers.get("Content-Type")) if raw else json.loads(b)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_favicon(server):
    code, (body, ctype) = call(server, "/favicon.ico", raw=True)
    assert code == 200 and ctype == "image/x-icon"
    assert body[:4] == b"\x00\x00\x01\x00" and b"PNG" in body[:40]
    code, (body, ctype) = call(server, "/favicon.svg", raw=True)
    assert code == 200 and ctype == "image/svg+xml" and body.startswith(b"<svg")
    assert 'href="/favicon.ico"' in web.APP.read_text()


@pytest.mark.skipif(not HAVE, reason="price data not downloaded")
def test_identical_runs_are_reused_not_duplicated(server):
    body = {"text": "hold 60% SPY and 40% TLT, rebalance quarterly", "options": {"start": "2015-01-01"}, "sensitivity": False}
    code, a = call(server, "/api/run", body)
    assert code == 200, a
    assert not a.get("reused")
    # the share link re-runs the same spec: it reopens the saved run instead of adding another to History
    code, b = call(server, "/api/run", {"share": a["share"], "sensitivity": False})
    assert code == 200 and b["reused"] and b["id"] == a["id"] and b["summary"]["cagr"] == a["summary"]["cagr"]
    assert b["url"] == a["url"] and b["interpretation"]
    code, runs = call(server, "/api/runs")
    assert [r["id"] for r in runs].count(a["id"]) == 1 and len(runs) == 1
    # different settings -> a new run; force -> a new run
    code, c = call(server, "/api/run", {**body, "options": {"start": "2016-01-01"}})
    assert code == 200 and c["id"] != a["id"] and not c.get("reused")
    code, d = call(server, "/api/run", {"share": a["share"], "sensitivity": False, "force": True})
    assert code == 200 and d["id"] != a["id"]


@needs("SPY", "TQQQ", "UVXY", "TECL", "QQQ", "TLT", "GLD", "SHY", "BTAL", "SHV")
def test_composer_import_endpoint(server):
    code, j = call(server, "/api/import/composer", {"json": FIXTURE.read_text()})
    assert code == 200, j
    assert j["spec"]["tree"]["on"] == "SPY" and j["problems"] == [] and "rsi(close, 10) > 79" in j["interpretation"]
    # it runs as a spec from the Build page
    code, r = call(server, "/api/run", {"spec": j["spec"], "sensitivity": False})
    assert code == 200, r
    assert r["summary"]["label"] == "TQQQ For The Long Term (sample)"
    # a ticker without data still loads the tree (for the editor to mark), with the problem listed
    code, j = call(server, "/api/import/composer", {"json": root(asset("NOSUCHX"))})
    assert code == 200 and pf.without_ids(j["spec"]["tree"]) == {"asset": "NOSUCHX"} and j["problems"]
    code, j = call(server, "/api/import/composer", {"json": root(asset("SPY", surprise=1))})
    assert code == 400 and "surprise" in j["error"]


@pytest.mark.skipif(not HAVE, reason="price data not downloaded")
def test_gallery_tags_and_json_entries(server):
    code, j = call(server, "/api/gallery")
    assert code == 200 and set(j["tags"]) == set(library.TAGS)
    tree_entry = next(x for x in j["library"] if x["id"] == "global-equities-momentum-gem")
    assert tree_entry["spec"]["tree"]["if"] and "leveraged" not in tree_entry["tags"]
    code, s = call(server, "/api/gallery/stats", {"id": tree_entry["id"]})
    assert code == 200 and s["stats"]["cagr"] is not None
    code, j = call(server, "/api/gallery")
    assert next(x for x in j["library"] if x["id"] == tree_entry["id"])["stats"]["cagr"] == s["stats"]["cagr"]
    code, _ = call(server, "/api/gallery/stats", {"id": "no-such-strategy"})
    assert code == 400


def test_build_editor_allows_groups_in_filters_and_defaults_lookbacks():
    page = web.APP.read_text()
    assert "A filter chooses among single assets" not in page
    assert "weighting needs single assets" not in page
    assert "function withDefaults" in page and "evaluated daily" in page
    assert "Import Composer symphony" in page and "/api/import/composer" in page
