"""Portfolio Visualizer review, round 11: fund types and the issuer table, unknown expense ratios in the screener,
precomputed fund statistics, the S&P 1500 / ADR stock universe, the metadata job's priorities and rate-limit backoff,
on-demand downloads and the ticker directory."""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from backtester import data, fund_lists, funds, web

ROOT = pathlib.Path(__file__).parents[1]


@pytest.fixture(scope="module")
def fd():
    spec = importlib.util.spec_from_file_location("fetch_data_round11", ROOT / "scripts" / "fetch_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """Funds module state isolated from the repository's caches (the repository's data files are still read)."""
    monkeypatch.setattr(funds, "WARM", False)
    monkeypatch.setattr(funds, "CACHE_FILE", tmp_path / "fund_stats_cache.json")
    monkeypatch.setattr(funds, "_CACHE", {})
    monkeypatch.setattr(funds, "_LOADED", {"done": False})
    return tmp_path


# ------------------------------------------------------------------ lists

def test_fund_lists_families_and_other_stocks():
    mf = set(fund_lists.MUTUAL_FUNDS)
    assert len(mf) >= 500
    for t in ("VTSAX", "VBTLX", "VWELX", "DFSVX", "DFQTX", "SWPPX", "SWTSX", "AGTHX", "EMEEX", "DODGX", "DODWX",
              "FXAIX", "PRWCX", "PTTRX"):
        assert t in mf, t
    assert len(fund_lists.OTHER_STOCKS) >= 280
    for t in ("TSM", "ASML", "NVO", "SAP", "TM", "BABA", "SHEL", "RY"):
        assert t in fund_lists.OTHER_STOCKS, t
    assert not set(fund_lists.OTHER_STOCKS) & fund_lists.ALL_FUNDS
    # stocks, never funds: is_fund stays False for them once the data job lists them
    assert not data.is_fund("TSM")


def test_index_constituents_seed_file():
    doc = data.index_constituents()
    assert 480 <= len(doc["sp500"]) <= 520 and 380 <= len(doc["sp400"]) <= 420 and 560 <= len(doc["sp600"]) <= 640
    assert doc["sp500"]["BRK-B"].startswith("Berkshire") and "MSFT" in doc["sp500"]
    assert "BRK-B" in data.builtin_symbols() and "VTSAX" in data.builtin_symbols()


# ------------------------------------------------------------------ the data job

HTML = """<html><body><table class="wikitable" id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
{rows}</table><table id="changes"><tr><th>Date</th></tr><tr><td>x</td></tr></table></body></html>"""


def _page(n, extra=()):
    sym = lambda i: "".join(chr(65 + (i // 26 ** k) % 26) for k in range(3))   # AAA, BAA, ... (letters only)
    rows = [f"<tr><td>Q{sym(i)}</td><td>Company {i}</td><td>X</td></tr>" for i in range(n)]
    rows += [f"<tr><td>{s}</td><td>{c}</td><td>X</td></tr>" for s, c in extra]
    return HTML.format(rows="".join(rows))


def test_parse_constituents_and_keep_saved_lists(fd, tmp_path):
    got = fd.parse_constituents(_page(3, [("BRK.B", "Berkshire Hathaway"), ("BF.B", "Brown-Forman")]))
    assert got["BRK-B"] == "Berkshire Hathaway" and got["BF-B"] == "Brown-Forman" and len(got) == 5
    path = tmp_path / "index_constituents.json"
    path.write_text(json.dumps({"sp500": {"OLD": "Old Co"}, "sp400": {"MID": "Mid Co"}, "sp600": {"SML": "Small"}}))
    pages = {"500": _page(500), "400": _page(10), "600": _page(600)}   # the S&P 400 page looks broken: 10 rows

    def download(url):
        if "S%26P_400" in url:
            return pages["400"]
        if "S%26P_600" in url:
            raise OSError("offline")
        return pages["500"]
    doc = fd.index_constituents(path, download)
    assert len(doc["sp500"]) == 500 and doc["sp400"] == {"MID": "Mid Co"} and doc["sp600"] == {"SML": "Small"}
    assert json.loads(path.read_text())["sp500"] == doc["sp500"]
    lst = fd.broad_stock_list({"sp500": {"AAPL": "", "JPM": "", "SPY": ""}, "sp400": {"MID": ""}, "sp600": {"JPM": ""}},
                              exclude={"AAPL"})
    assert lst[:2] == ["JPM", "MID"] and "SPY" not in lst and "AAPL" not in lst and lst.count("JPM") == 1
    assert "TSM" in lst                                    # the other large stocks follow the index members


def test_meta_batch_priorities(fd):
    uni = ["ZZZZ", "VFINX", "SPY", "TQQQ", "QQQ"]
    got = fd.funds_meta_batch(uni, {}, "2026-09-27", budget=5)
    assert got[:2] == ["SPY", "QQQ"] and got[2] == "VFINX"          # core funds, then mutual funds
    assert fd.FUNDS_META_PER_RUN >= 900


class RateLimited(Exception):
    pass


RateLimited.__name__ = "YFRateLimitError"


def test_meta_job_backs_off_and_stops_on_rate_limits(fd, tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(fd, "META_SLEEP", lambda s: slept.append(s))
    monkeypatch.setattr(fd, "FUNDS_META_MAX_429", 3)

    class Tk:
        def __init__(self, t):
            self.t = t

        @property
        def info(self):
            if self.t == "OKAY":
                return {"longName": "Okay Fund", "quoteType": "ETF", "netExpenseRatio": 0.05}
            raise RateLimited("Too Many Requests. Rate limited. Try after a while.")

        funds_data = None
    monkeypatch.setattr(fd.yf, "Ticker", Tk)
    path = tmp_path / "funds_meta.json"
    doc = fd.fetch_funds_meta(["OKAY", "LIM1", "LIM2", "LIM3", "LIM4", "LIM5"], path=path, today="2026-09-27")
    assert doc["funds"]["OKAY"]["expense_ratio"] == pytest.approx(0.0005) and doc["funds"]["OKAY"]["quote_type"] == "ETF"
    assert not doc["failed"]                  # rate-limited symbols are not failures: they are looked up next run
    assert slept and set(slept) <= set(fd.FUNDS_META_BACKOFF)
    assert fd.rate_limited(RateLimited("x")) and not fd.rate_limited(ValueError("404 Not Found"))


def test_history_meta_and_ticker_info(fd, tmp_path, monkeypatch):
    class Tk:
        history_metadata = {"longName": "Vanguard Wellington Inv", "instrumentType": "MUTUALFUND", "currency": "USD",
                            "fullExchangeName": "Nasdaq", "firstTradeDate": 315671400}
    assert fd.history_meta(Tk()) == {"name": "Vanguard Wellington Inv", "type": "MUTUALFUND", "exchange": "Nasdaq",
                                      "currency": "USD"}
    path = tmp_path / "ticker_info.json"
    path.write_text(json.dumps({"tickers": {"OLD": {"name": "Old", "type": "EQUITY", "seen": "2026-01-01"}}}))
    monkeypatch.setattr(fd, "TICKER_INFO", {"VWELX": fd.history_meta(Tk())})
    doc = fd.write_ticker_info(path, today="2026-09-27")
    assert doc["tickers"]["VWELX"]["type"] == "MUTUALFUND" and doc["tickers"]["OLD"]["name"] == "Old"


# ------------------------------------------------------------------ fund types and the issuer table

def test_fund_type_prefers_yahoo_then_issuer_then_lists(tmp_path, monkeypatch):
    meta = {"funds": {"VFINX": {"quote_type": "MUTUALFUND", "name": "Vanguard 500 Index Investor"},
                      "ABCDX": {"quote_type": "ETF"}}}
    (tmp_path / "m.json").write_text(json.dumps(meta))
    (tmp_path / "r.json").write_text(json.dumps({"as_of": "2026-09-27", "funds": {
        "QQQ": {"name": "Invesco QQQ Trust, Series 1", "type": "ETF", "family": "Invesco", "expense_ratio": 0.0018,
                "inception": "1999-03-10"},
        "SWPPX": {"type": "Mutual fund", "family": "Schwab", "expense_ratio": 0.0002}}}))
    monkeypatch.setattr(funds, "META_FILE", tmp_path / "m.json")
    monkeypatch.setattr(funds, "REFERENCE_FILE", tmp_path / "r.json")
    monkeypatch.setattr(funds, "INFO_FILE", tmp_path / "none.json")
    assert funds.fund_type("VFINX") == "Mutual fund"      # in the core ETF download list, but Yahoo says MUTUALFUND
    assert funds.fund_type("ABCDX") == "ETF"              # Yahoo's word beats the 5-letter-X rule
    assert funds.fund_type("SWPPX") == "Mutual fund" and funds.fund_type("QQQ") == "ETF"
    assert funds.fund_type("DODGX") == "Mutual fund" and funds.fund_type("VTI") == "ETF"
    f = funds.facts("QQQ")
    assert f["expense_ratio"] == 0.0018 and f["sources"]["expense_ratio"].startswith("issuer (Invesco, as of 2026-09-27")
    assert f["name"] == "Invesco QQQ Trust, Series 1"


def test_reference_table_covers_the_core_funds():
    doc = json.loads((ROOT / "data" / "fund_reference.json").read_text())
    assert doc["as_of"] and doc["sources"] and len(doc["funds"]) >= 300
    for t in ("SPY", "QQQ", "VTI", "VOO", "IVV", "BND", "AGG", "SCHD", "VFIAX", "VTSAX", "SWPPX", "DFSVX", "RSP"):
        e = doc["funds"][t]
        assert 0 < e["expense_ratio"] < 0.02 and e["name"] and e["inception"][:2] in ("19", "20"), t
    assert doc["funds"]["QQQ"]["inception"] == "1999-03-10" and doc["funds"]["VTSAX"]["type"] == "Mutual fund"


needs_prices = pytest.mark.skipif(not {"SPY", "VFINX", "VTSAX", "BND"} <= set(data.available_tickers()),
                                  reason="reads the repository's price files")


@needs_prices
def test_screener_mutual_funds_and_unknown_expense_ratios(fresh):
    T = web.api_funds({"kind": ["Mutual fund"], "max_er": ["0.002"]})
    assert T["count"] > 20 and all(r["type"] == "Mutual fund" and r["expense_ratio"] <= 0.002 for r in T["funds"])
    assert "VFINX" in {r["ticker"] for r in T["funds"]} or T["unknown"]["unknown_er"] > 0
    n_unknown = T["unknown"]["unknown_er"]
    T2 = web.api_funds({"kind": ["Mutual fund"], "max_er": ["0.002"], "include_unknown": ["1"]})
    assert T2["count"] == T["count"] + n_unknown
    kinds = {r["ticker"]: r["type"] for r in web.api_funds({})["funds"]}
    assert kinds["VFINX"] == "Mutual fund" and kinds["SPY"] == "ETF" and kinds["VTSAX"] == "Mutual fund"
    spy = next(r for r in web.api_funds({"q": ["SPY"]})["funds"] if r["ticker"] == "SPY")
    assert spy["expense_ratio"] and spy["name"]


def test_screen_counts_unknowns():
    rows = [{"ticker": "AAA", "type": "ETF", "expense_ratio": 0.0003, "net_assets": 1e9},
            {"ticker": "BBB", "type": "ETF", "expense_ratio": None, "net_assets": None},
            {"ticker": "CCC", "type": "ETF", "expense_ratio": 0.01, "net_assets": 5e9},
            {"ticker": "DDD", "type": "ETF", "expense_ratio": 0.0001, "pending": True}]
    c = {}
    assert [r["ticker"] for r in funds.screen(rows, max_er=0.001, counts=c)] == ["AAA", "DDD"]
    assert c["unknown_er"] == 1
    c = {}
    assert [r["ticker"] for r in funds.screen(rows, max_er=0.001, include_unknown=True, counts=c)] == ["AAA", "BBB", "DDD"]
    c = {}
    assert [r["ticker"] for r in funds.screen(rows, min_years=5, counts=c)] == [] and c["unknown_stats"] == 1


@needs_prices
def test_precomputed_stats_answer_at_once(fresh, monkeypatch):
    sample = ["SPY", "BND", "VFINX"]
    path = fresh / "fund_stats.json"
    assert funds.precompute(path, sample) == 3
    doc = json.loads(path.read_text())
    assert doc["version"] == funds.STATS_VERSION and doc["stats"]["SPY"]["checked"]
    assert doc["stats"]["SPY"]["key"] == funds._key("SPY")          # content key: size + crc of the file's tail
    monkeypatch.setattr(funds, "STATS_FILE", path)
    monkeypatch.setattr(funds, "_CACHE", {})
    monkeypatch.setattr(funds, "_LOADED", {"done": False})
    monkeypatch.setattr(funds, "universe", lambda: [(t, funds.fund_type(t)) for t in sample])
    T = funds.table(budget=0)
    assert T["pending"] == 0 and all(r.get("checked") for r in T["funds"])
    # without the file and no time budget: rows come back pending and the page asks again
    monkeypatch.setattr(funds, "STATS_FILE", fresh / "missing.json")
    monkeypatch.setattr(funds, "_CACHE", {})
    monkeypatch.setattr(funds, "_LOADED", {"done": False})
    T = funds.table(budget=0)
    assert T["pending"] == 3 and all(r.get("pending") and r.get("r1y") is None for r in T["funds"])
    T = funds.table(budget=30)
    assert T["pending"] == 0 and all(r.get("r1y") is not None for r in T["funds"])


def test_repository_fund_stats_file_is_valid():
    p = ROOT / "data" / "fund_stats.json"
    if not p.exists():
        pytest.skip("no precomputed statistics")
    doc = json.loads(p.read_text())
    assert doc["version"] == funds.STATS_VERSION and len(doc["stats"]) > 500


# ------------------------------------------------------------------ on-demand downloads

def _history(n=300, start="2020-01-02"):
    idx = pd.bdate_range(start, periods=n)
    px = 50 * np.cumprod(1 + np.random.default_rng(1).normal(0.0003, 0.01, n))
    return pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "adj_close": px,
                         "volume": 1e6, "dividend": 0.0, "split": 0.0}, index=pd.DatetimeIndex(idx, name="date"))


def test_fetch_on_demand_saves_validates_and_queues(tmp_path, monkeypatch):
    prices = tmp_path / "prices"
    prices.mkdir()
    for t in ("SPY",):                      # the integrity gate's reference
        src = data.PRICES / f"{t}.csv"
        if src.exists():
            (prices / f"{t}.csv").write_text(src.read_text())
    monkeypatch.setattr(data, "PRICES", prices)
    monkeypatch.setattr(data, "EXTRA_TICKERS_FILE", tmp_path / "extra_tickers.txt")
    monkeypatch.setattr(data, "ON_DEMAND", {})
    monkeypatch.setattr(data, "_ON_DEMAND_FAILED", {})
    monkeypatch.setenv("BACKTESTER_OFFLINE", "1")
    assert not data.fetch_on_demand("QZXQ")                          # offline, no downloader: nothing happens
    calls = []

    def dl(t):
        calls.append(t)
        return _history()
    assert data.fetch_on_demand("QZXQ", download=dl) and calls == ["QZXQ"]
    assert (prices / "QZXQ.csv").exists() and not list(prices.glob(".*.part"))
    assert "downloaded from Yahoo Finance" in data.ON_DEMAND["QZXQ"] and "price-integrity gate" in data.ON_DEMAND["QZXQ"]
    from backtester import coverage
    assert "QZXQ" in coverage.read_queue()                    # queued: the data job adds it to its refresh
    assert data.on_demand_notes(["qzxq", "SPY"]) == [data.ON_DEMAND["QZXQ"]]
    assert len(data.load("QZXQ")) == 300
    # too short to use: not saved, and not retried at once
    assert not data.fetch_on_demand("QZXR", download=lambda t: _history(5))
    assert not (prices / "QZXR.csv").exists() and "QZXR" in data._ON_DEMAND_FAILED
    assert data.validate_download(None) == "no data" and data.validate_download(_history(5)).startswith("only 5")
    data.load.cache_clear()


def test_site_fetch_and_fund_detail_use_on_demand(tmp_path, monkeypatch):
    prices = tmp_path / "prices"
    prices.mkdir()
    monkeypatch.setattr(data, "PRICES", prices)
    monkeypatch.setattr(data, "EXTRA_TICKERS_FILE", tmp_path / "extra_tickers.txt")
    monkeypatch.setattr(data, "ON_DEMAND", {})
    real = data.fetch_on_demand
    monkeypatch.setattr(data, "fetch_on_demand", lambda t, download=None: real(t, download=lambda s: _history()))
    out = web.api_fetch({"ticker": "QZXS"})
    assert out["status"] == "downloaded" and "QZXS" in out["note"]
    monkeypatch.setattr(funds, "WARM", False)
    D = web.api_fund_detail({"t": ["QZXT"]})               # not on disk: fetched, then described
    assert D["ticker"] == "QZXT" and D["stats"]["r1y"] is not None and D["notes"]
    data.load.cache_clear()


# ------------------------------------------------------------------ the ticker directory

def test_directory_search():
    D = web.api_directory({"q": ["spy"]})
    assert D["results"][0]["ticker"] == "SPY" and D["results"][0]["type"] == "ETF"
    D = web.api_directory({"q": ["berkshire"]})
    assert "BRK-B" in {r["ticker"] for r in D["results"]}
    brk = next(r for r in D["results"] if r["ticker"] == "BRK-B")
    assert brk["type"] == "Stock" and "S&P 500" in brk["indexes"]
    D = web.api_directory({"q": ["taiwan semiconductor"]})
    assert "TSM" in {r["ticker"] for r in D["results"]}
    D = web.api_directory({"q": ["vanguard"], "kind": ["Mutual fund"], "limit": ["20"]})
    assert D["total"] > 50 and len(D["results"]) == 20 and all(r["type"] == "Mutual fund" for r in D["results"])
    D = web.api_directory({"q": [""], "has_data": ["1"], "limit": ["5"]})
    assert D["total"] == D["with_data"] and all(r["has_data"] for r in D["results"])
    with pytest.raises(web.ClientError):
        web.api_directory({"limit": ["many"]})
