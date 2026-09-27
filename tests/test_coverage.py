"""Ticker coverage: the US listing (NASDAQ Trader symbol directory), the curated mutual fund list, the request queue
(data/requested_tickers.txt) and the data job's rate-limit-aware rotating batches. Offline: every download is mocked."""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from backtester import coverage, data, fund_lists, parser, web
from backtester.parser import ParseError

ROOT = pathlib.Path(__file__).parents[1]
SAMPLE = json.loads((ROOT / "tests" / "fixtures" / "symbol_directory_sample.json").read_text())


@pytest.fixture(scope="module")
def fd():
    spec = importlib.util.spec_from_file_location("fetch_data_coverage", ROOT / "scripts" / "fetch_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def listing(tmp_path, monkeypatch):
    """A small listing with made-up symbols (so no real price file exists for them)."""
    doc = {"updated": "2026-09-25", "symbols": {
        "QZZF": {"name": "Quizzical Motor Company", "exchange": "NYSE", "type": "Stock", "cap": 49351},
        "QZZE": {"name": "Quizzical Total Market ETF", "exchange": "NYSE Arca", "type": "ETF"},
        "QZZS": {"name": "Quizzical Micro Corp", "exchange": "Nasdaq", "type": "Stock", "cap": 12}},
        "mutual_funds": ["QZZMX"]}
    f = tmp_path / "listed_symbols.json"
    f.write_text(coverage.dumps_listing(doc))
    monkeypatch.setattr(coverage, "LISTED_FILE", f)
    return doc


def _history(n=300):
    idx = pd.bdate_range("2020-01-01", periods=n)
    px = 100 * np.exp(np.cumsum(np.full(n, 0.0004)))
    return pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "adj_close": px,
                         "volume": 1e6, "dividend": 0.0, "split": 0.0}, index=pd.DatetimeIndex(idx, name="date"))


# ------------------------------------------------------------------ the symbol directory

def test_symbol_directory_keeps_common_stocks_etfs_and_etns_only():
    nas = coverage.parse_nasdaq_listed(SAMPLE["nasdaqlisted"])
    oth = coverage.parse_other_listed(SAMPLE["otherlisted"])
    assert nas["AAPL"] == {"name": "Apple Inc.", "exchange": "Nasdaq", "type": "Stock"}
    assert nas["QQQ"]["type"] == nas["TQQQ"]["type"] == nas["IBIT"]["type"] == "ETF"
    assert "AACG" in nas and "ACTU" in nas                        # an ADR; a deficient (but listed) company
    for t in ("AACIU", "AACIW", "AACPR"):                         # SPAC units, warrants, rights
        assert t not in nas
    assert oth["F"] == {"name": "Ford Motor Company", "exchange": "NYSE", "type": "Stock"}
    assert oth["BRK-B"]["name"] == "Berkshire Hathaway Inc." and "BF-A" in oth      # class shares, Yahoo spelling
    assert oth["SPY"]["type"] == oth["GLD"]["type"] == "ETF"
    assert oth["VXX"]["type"] == oth["AMUB"]["type"] == "ETN"
    assert "AB" in oth and "BSBR" in oth                          # partnership units, an ADS "representing one unit"
    for t in ("AAC-U", "AAC-W", "ABR$D", "ABR-D", "AFGB", "AHL$D", "ZZZTX"):   # units, warrants, preferreds, debt, test
        assert t not in oth


def test_build_listing_keeps_saved_parts_when_a_download_fails():
    caps = coverage.parse_screener_caps({"data": {"rows": [{"symbol": "F", "marketCap": "49351000000.00"},
                                                           {"symbol": "BRK/B", "marketCap": "1114584378402.00"},
                                                           {"symbol": "AAPL", "marketCap": ""}]}})
    assert caps == {"F": 49351000000.0, "BRK-B": 1114584378402.0}
    mf = coverage.parse_sec_mutual_funds({"fields": ["cik", "seriesId", "classId", "symbol"],
                                          "data": [[1, "S1", "C1", "VFIAX"], [2, "S2", "C2", "llpfx"], [3, "S", "C", ""]]})
    assert mf == ["LLPFX", "VFIAX"]
    doc = coverage.build_listing(SAMPLE["nasdaqlisted"], SAMPLE["otherlisted"], caps, mf, {}, "2026-09-25", min_rows=1)
    assert doc["symbols"]["F"]["cap"] == 49351 and "cap" not in doc["symbols"]["SPY"]
    assert doc["counts"]["mutual_funds"] == 2 and doc["updated"] == "2026-09-25"
    assert json.loads(coverage.dumps_listing(doc)) == doc
    # the next day every download failed: the saved symbols, caps and fund list stay
    again = coverage.build_listing("", "", None, None, doc, "2026-09-26", min_rows=1)
    assert again["symbols"] == doc["symbols"] and again["mutual_funds"] == mf
    assert again["updated"] == "2026-09-25" and again["caps_updated"] == "2026-09-25"
    # a truncated directory file (fewer rows than plausible) is not trusted either
    assert coverage.build_listing(SAMPLE["nasdaqlisted"], SAMPLE["otherlisted"], caps, mf, doc, "2026-09-27")["symbols"] \
        == doc["symbols"]


def test_listed_tier_all_funds_and_stocks_above_the_cap_largest_first(listing):
    doc = dict(listing)
    tier = coverage.listed_tier(doc, {"QZZE": 5e9}, exclude=(), min_cap=250e6)
    assert tier == ["QZZF", "QZZE"]                     # $49B stock, then the $5B ETF; the $12M stock is left out
    assert coverage.listed_tier(doc, {}, exclude={"QZZF"}, min_cap=1e6) == ["QZZS", "QZZE"]


def test_committed_listing_covers_the_reviewers_examples():
    doc = json.loads((ROOT / "data" / "listed_symbols.json").read_text())
    s = doc["symbols"]
    assert s["F"]["type"] == "Stock" and s["F"]["cap"] > 1000 and s["BRK-B"]["type"] == "Stock"
    assert doc["counts"]["Stock"] > 4000 and doc["counts"]["ETF"] > 3000 and "LLPFX" in doc["mutual_funds"]
    assert not any("$" in t or t.endswith(("-U", "-W", "-WS")) for t in s)
    # F is an S&P 500 member (the stock batch), so the listing batch doesn't download it twice
    tier = coverage.listed_tier(doc, {}, exclude=set(data.index_constituents().get("sp500") or {}))
    assert "F" not in tier and len(tier) > 5000


# ------------------------------------------------------------------ the curated mutual fund list

def test_curated_mutual_funds_merge_into_the_fund_lists():
    fams = fund_lists.read_curated_funds()
    assert "LLPFX" in fams["Longleaf (Southeastern)"].split() and len(fams) >= 40
    n = sum(len(v.split()) for v in fams.values())
    assert n >= 700
    for t in ("LLPFX", "OAKMX", "DODGX", "PTTRX", "FCNTX", "AGTHX", "PRWCX", "VFIAX", "DFSVX", "SWPPX"):
        assert t in fund_lists.MUTUAL_FUNDS and t in fund_lists.ALL_FUNDS, t
    assert len(fund_lists.MUTUAL_FUNDS) >= 1000
    assert not set(fund_lists.MUTUAL_FUNDS) & set(fund_lists.OTHER_STOCKS)
    from backtester import funds
    funds._list_categories.cache_clear()
    assert funds._list_categories()["LLPFX"] == "Longleaf mutual fund"
    assert coverage.lookup("LLPFX")["type"] == "Mutual fund"


def test_read_curated_funds_format(tmp_path):
    f = tmp_path / "mf.txt"
    f.write_text("# header\nABCDX  # stray before a family\n## Acme (Family)\nacmex ACMEY, bad! # comment\n\n## other\nZZZZX\n")
    assert fund_lists.read_curated_funds(f) == {"other": "ABCDX ZZZZX", "Acme (Family)": "ACMEX"}
    assert fund_lists.read_curated_funds(tmp_path / "missing.txt") == {}


# ------------------------------------------------------------------ the request queue, site and CLI

def test_queue_read_enqueue_rewrite(tmp_path):
    q = tmp_path / "q.txt"
    assert coverage.read_queue(q) == []
    assert coverage.enqueue("f", "why", q) == "added" and coverage.enqueue("F", path=q) == "already requested"
    assert coverage.enqueue("llpfx", path=q) == "added"
    assert q.read_text().startswith("# Tickers the site") and coverage.read_queue(q) == ["F", "LLPFX"]
    with pytest.raises(ValueError):
        coverage.enqueue("no way!", path=q)
    coverage.write_queue(["LLPFX"], {"LLPFX": "no data yet (1 run)"}, q)
    assert coverage.read_queue(q) == ["LLPFX"] and "no data yet" in q.read_text()


def test_offline_valid_ticker_is_queued_with_a_clear_message(listing, monkeypatch):
    monkeypatch.setenv("BACKTESTER_OFFLINE", "1")
    with pytest.raises(data.DataError) as e:
        data.load("QZZF")
    msg = str(e.value)
    assert "QZZF (Quizzical Motor Company, stock, NYSE) is a valid symbol" in msg
    assert "isn't downloaded yet" in msg and "no internet access" in msg and "next data refresh" in msg
    assert coverage.read_queue() == ["QZZF"]
    # a mutual fund known only from the SEC's list is recognised too
    assert "Mutual fund" in data.unknown_ticker_message("QZZMX") and coverage.read_queue() == ["QZZF", "QZZMX"]
    # a symbol no directory knows is not queued by itself: the message says how to request it
    msg = data.unknown_ticker_message("QZZQX")
    assert "isn't a US-listed" in msg and "extra_tickers.txt" in msg and "QZZQX" not in coverage.read_queue()


def test_cli_sentence_with_a_valid_but_missing_ticker(listing, monkeypatch):
    monkeypatch.setenv("BACKTESTER_OFFLINE", "1")
    with pytest.raises(ParseError) as e:
        parser.parse("hold 50% SPY and 50% QZZE")
    msg = str(e.value)
    assert "QZZE (Quizzical Total Market ETF, ETF, NYSE Arca) is a valid symbol not downloaded yet" in msg
    assert "requested_tickers.txt" in msg and "next data refresh" in msg
    assert "QZZE" in coverage.read_queue()
    with pytest.raises(ParseError, match=r"extra_tickers\.txt"):
        parser.parse("hold 50% SPY and 50% ZZZQX")


def test_site_fetch_queues_a_listed_symbol(listing, monkeypatch):
    monkeypatch.setattr(data, "fetch_on_demand", lambda t: False)
    out = web.api_fetch({"ticker": "qzzs"})
    assert out["queued"] == "added" and "Quizzical Micro Corp" in out["status"] and "requested_tickers.txt" in out["status"]
    again = web.api_fetch({"ticker": "QZZS"})
    assert again["queued"] == "already requested" and "already queued" in again["status"]


def test_directory_lists_the_whole_listing_with_coverage_counts(monkeypatch):
    monkeypatch.setenv("BACKTESTER_QUEUE_FILE", str(ROOT / "data" / "requested_tickers.txt"))
    D = web.api_directory({"q": ["ford motor"]})
    row = next(r for r in D["results"] if r["ticker"] == "F")
    assert row["type"] == "Stock" and row["name"].startswith("Ford Motor")
    C = D["coverage"]
    assert C["stocks"]["listed"] > 4000 and C["etfs"]["listed"] > 3000 and C["etfs"]["with_data"] > 300
    assert C["stocks"]["in_scope"] < C["stocks"]["listed"] and C["mutual_funds"]["curated"] >= 1000
    assert C["listing_date"] and C["min_market_cap"] == coverage.LISTED_MIN_MARKET_CAP
    assert D["size"] > 10000


# ------------------------------------------------------------------ the data job

def test_queue_is_downloaded_first_and_graduates(fd, tmp_path):
    saved = {}

    def save(t, df):
        saved[t] = len(df)
        return {"rows": len(df)}
    answers = {"GOOD": (_history(), "ok"), "NOPE": (None, "no data"), "SLOW": (None, "rate limited"),
               "GONE": (None, "no data")}
    out = fd.process_queue(["GOOD", "NOPE", "SLOW", "GONE"], {"GONE": 2}, "2026-09-28", save, lambda t: answers[t])
    assert out["ok"] == {"GOOD": {"rows": 300}} and saved == {"GOOD": 300}
    assert out["waiting"] == ["NOPE", "SLOW"] and out["attempts"] == {"NOPE": 1}      # a rate limit is not an attempt
    assert out["dropped"] == {"GONE": "2026-09-28"}                                    # 3 runs without data
    extra = tmp_path / "extra.txt"
    extra.write_text("# mine\nABC")
    assert fd.graduate(["GOOD", "F", "ABC"], refreshed={"F"}, path=extra) == ["GOOD"]
    assert extra.read_text() == "# mine\nABC\nGOOD  # from data/requested_tickers.txt\n"
    assert fd.graduate(["GOOD"], refreshed=set(), path=extra) == []


def test_rate_limits_pause_then_stop_and_are_not_failures(fd, monkeypatch):
    class YFRateLimitError(Exception):
        pass
    sleeps = []
    monkeypatch.setattr(fd, "SLEEP", lambda s: sleeps.append(round(s)))
    monkeypatch.setattr(fd, "RATE", {"pauses": 0, "until": 0.0, "stop": False, "limited": 0, "t0": fd.time.time()})
    monkeypatch.setattr(fd, "RATE_PAUSES", (0, 0))

    def fake_fetch(t):
        if t.startswith("RL"):
            raise YFRateLimitError("Too Many Requests. Rate limited. Try after a while.")
        if t == "UNKNOWN":
            raise RuntimeError("no data")
        return _history(50)
    monkeypatch.setattr(fd, "fetch", fake_fetch)
    assert fd.fetch_one("OK1")[1] == "ok"
    assert fd.fetch_one("UNKNOWN", tries=2) == (None, "no data")
    assert fd.fetch_one("RL1", tries=2) == (None, "rate limited")
    assert fd.RATE["pauses"] == 2 and not fd.RATE["stop"]
    assert fd.fetch_one("RL2", tries=1) == (None, "rate limited") and fd.RATE["stop"]
    # a stopped run leaves the rest of a rotating batch for the next run: nothing is recorded as failed
    failed = {}
    got = fd.rotate("test", ["OK2", "RL3"], failed, "2026-09-28", lambda t, df: {"rows": len(df)})
    assert got == {} and failed == {}
    monkeypatch.setattr(fd, "RATE", {"pauses": 0, "until": 0.0, "stop": False, "limited": 0, "t0": fd.time.time()})
    got = fd.rotate("test", ["OK2", "UNKNOWN"], failed, "2026-09-28", lambda t, df: {"rows": len(df)})
    assert got == {"OK2": {"rows": 50}} and failed == {"UNKNOWN": {"date": "2026-09-28", "why": "no data"}}


def test_legacy_failures_are_retried(fd):
    # on 2026-09-27 a rate-limited run parked 415 S&P members (F among them) for 30 days as bare dates
    prev = {"F": "2026-09-27", "ZZZZ": {"date": "2026-09-20", "why": "no data"}, "OLD": {"date": "2026-09-01"}}
    got = fd.failures(prev, keep={"F", "ZZZZ"})
    assert got == {"ZZZZ": {"date": "2026-09-20", "why": "no data"}}
    assert fd.broad_batch(["F", "ZZZZ"], {}, got, "2026-09-28", on_disk=lambda t: False) == ["F"]


def test_refresh_listing_with_mocked_downloads(fd, tmp_path):
    def rows(n, fmt):
        return "\n".join(fmt(i) for i in range(n))
    letters = [a + b + c for a in "ABCDEFGHIJ" for b in "ABCDEFGHIJ" for c in "ABCDEFGHIJKL"]
    nas = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n" + rows(
        1100, lambda i: f"N{letters[i]}|Company {i} - Common Stock|Q|N|N|100|N|N") + "\nFile Creation Time: x|||||||"
    oth = "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n" + rows(
        1100, lambda i: f"O{letters[i]}|Fund {i} ETF|P|O{letters[i]}|Y|100|N|O{letters[i]}") + "\nF|Ford Motor Company Common Stock|N|F|N|100|N|F"
    screener = json.dumps({"data": {"rows": [{"symbol": "F", "marketCap": "49000000000"}]}})
    sec = json.dumps({"fields": ["cik", "seriesId", "classId", "symbol"],
                      "data": [[1, "S", "C", f"M{letters[i]}X"] for i in range(1100)] + [[2, "S", "C", "llpfx"]]})
    pages = {fd.NASDAQ_LISTED_URL: nas, fd.OTHER_LISTED_URL: oth, fd.NASDAQ_SCREENER_URL: screener, fd.SEC_MF_URL: sec}
    calls = []

    def download(url, headers):
        calls.append(url)
        return pages[url]
    path = tmp_path / "listed_symbols.json"
    doc = fd.refresh_listing(path, download, "2026-09-28")
    assert set(calls) == set(pages)
    assert doc["counts"] == {"Stock": 1101, "ETF": 1100, "ETN": 0, "mutual_funds": 1101}
    assert doc["symbols"]["F"] == {"name": "Ford Motor Company", "exchange": "NYSE", "type": "Stock", "cap": 49000}
    assert json.loads(path.read_text()) == doc and "LLPFX" in doc["mutual_funds"]

    def broken(url, headers):
        raise OSError("blocked")
    assert fd.refresh_listing(path, broken, "2026-09-29")["symbols"] == doc["symbols"]      # the saved copy stays


def test_full_history_is_requested(fd, monkeypatch):
    """Histories start at Yahoo's first date: period="max" (99 years back in yfinance). VFINX starts 1980-01-02
    because Yahoo's mutual fund data starts there (every pre-1980 fund in data/prices does), not because of the fetch."""
    seen = {}

    class T:
        history_metadata = {"longName": "Vanguard 500 Index Investor", "instrumentType": "MUTUALFUND"}

        def __init__(self, t):
            seen["ticker"] = t

        def history(self, **kw):
            seen.update(kw)
            h = _history(10)
            h.index = h.index.tz_localize("America/New_York")
            return h.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close",
                                     "adj_close": "Adj Close", "volume": "Volume", "dividend": "Dividends",
                                     "split": "Stock Splits"})
    monkeypatch.setattr(fd.yf, "Ticker", T)
    df = fd.fetch("VFINX")
    assert seen["period"] == "max" and seen["interval"] == "1d" and seen["auto_adjust"] is False and len(df) == 10
    first = {}
    for t in ("VFINX", "VWELX", "FMAGX", "DODGX"):
        p = data.PRICES / f"{t}.csv"
        if p.exists():
            with p.open() as f:
                f.readline()
                first[t] = f.readline()[:10]
    assert all(d >= "1980-01-02" for d in first.values())
    if (data.PRICES / "IBM.csv").exists():
        with (data.PRICES / "IBM.csv").open() as f:
            f.readline()
            assert f.readline()[:4] == "1962"          # stocks do go back further on Yahoo
