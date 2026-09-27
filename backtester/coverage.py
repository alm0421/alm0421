"""Ticker coverage: the whole US listing, the request queue and the coverage counts.

data/listed_symbols.json   every US-listed common stock, ETF and ETN trading today, derived by the data job
                           (scripts/fetch_data.py, refresh_listing) from the NASDAQ Trader symbol directory
                           (nasdaqlisted.txt: Nasdaq; otherlisted.txt: NYSE, NYSE American, NYSE Arca, Cboe BZX, IEX),
                           with each stock's market cap from Nasdaq's stock screener and the SEC's list of mutual fund
                           share-class symbols (company_tickers_mf.json). Test issues, warrants, rights, SPAC units,
                           preferred shares and exchange-listed debt are left out.
data/requested_tickers.txt the queue: a symbol the site or the CLI was asked for but has no price file for (offline,
                           or downloaded on demand on someone's computer) is appended here; the data job downloads the
                           queue FIRST, then moves each symbol that worked into its regular refresh (the rotating lists
                           it belongs to, else data/extra_tickers.txt) and drops a symbol Yahoo doesn't know after
                           QUEUE_MAX_ATTEMPTS runs (listed under requested_failed in data/universe.json).

Coverage policy (what the data job downloads by itself, in rotating batches; LISTED_MIN_MARKET_CAP and
LISTED_PER_RUN in scripts/fetch_data.py): every ETF and ETN in the listing, every listed stock with a market cap of at
least LISTED_MIN_MARKET_CAP, the S&P 500/400/600 members, every current or former Nasdaq-100 member and the curated
mutual fund lists (backtester/fund_lists.py and data/mutual_funds.txt). Anything else - a micro-cap, an OTC symbol, a
mutual fund outside the lists - is downloaded on request (the queue, or on demand when online).

Pure functions (no network): the data job, the site and the tests share them.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LISTED_FILE = DATA / "listed_symbols.json"
QUEUE_FILE = DATA / "requested_tickers.txt"
QUEUE_MAX_ATTEMPTS = 3                   # data-job runs a queued symbol is tried before it is dropped
LISTED_MIN_MARKET_CAP = 250e6            # stocks below this are downloaded on request only (see the module docstring)

EXCHANGES = {"Q": "Nasdaq", "N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX",
             "M": "NYSE Texas", "F": "Texas Stock Exchange"}
NASDAQ_TIERS = {"Q": "Global Select", "G": "Global Market", "S": "Capital Market"}
SYMBOL_RE = r"\^?[A-Z0-9][A-Z0-9.\-]{0,14}"

# Securities that are not common stock (the listing flags ETFs and test issues, nothing else)
_NOT_COMMON = re.compile(
    r"\bwarrants?\b|\brights?\b|\bunits?,? each\b|\bpreferred\b|\bpreference\b|\bpfd\b|\bdebentures?\b"
    r"|\bnotes?\b(?: due)?|\bbonds? due\b|\bdue \d{4}\b|\bcontingent value\b|\bwhen[- ]issued\b|\bwhen[- ]distributed\b",
    re.I)
_NASDAQ_KIND_NOT_COMMON = re.compile(r"^(units?|warrants?|rights?)\b|preferred|notes?\b|debentures?", re.I)
_ETN = re.compile(r"\bETNs?\b|exchange[- ]traded notes?", re.I)
# otherlisted.txt suffixes that are never common stock: units, warrants, rights, when-issued / when-distributed
_OTHER_BAD_SUFFIX = re.compile(r"\.(U|W|WS|WD|WI|R|RT)$")


def yahoo_symbol(sym: str) -> str:
    """The listing's spelling -> Yahoo's: class shares use '-' (BRK.B -> BRK-B)."""
    return sym.strip().upper().replace(".", "-")


def _rows(text: str) -> list[list[str]]:
    lines = [ln for ln in str(text or "").splitlines() if ln.strip()]
    if not lines:
        return []
    head = [h.strip() for h in lines[0].split("|")]
    out = []
    for ln in lines[1:]:
        if ln.startswith("File Creation Time"):
            continue
        cells = [c.strip() for c in ln.split("|")]
        if len(cells) >= len(head):
            out.append(dict(zip(head, cells)))
    return out


def _kind(name: str, etf_flag: str) -> str | None:
    """'ETF', 'ETN', 'Stock' or None (not common stock: warrant, right, unit, preferred, listed debt)."""
    if etf_flag == "Y":
        return "ETF"
    if _ETN.search(name):
        return "ETN"
    if _NOT_COMMON.search(name):
        return None
    return "Stock"


def parse_nasdaq_listed(text: str) -> dict[str, dict]:
    """{yahoo symbol: {"name", "exchange", "type"}} from nasdaqlisted.txt (Symbol|Security Name|Market Category|
    Test Issue|Financial Status|Round Lot Size|ETF|NextShares). The security type is the part of the name after the
    last ' - ' ("Apple Inc. - Common Stock", "... - Units", "... - Warrant")."""
    out = {}
    for r in _rows(text):
        sym, name = r.get("Symbol", ""), r.get("Security Name", "")
        if r.get("Test Issue") == "Y" or r.get("NextShares") == "Y" or not re.fullmatch(r"[A-Z]{1,5}", sym):
            continue
        base, _, sec = name.rpartition(" - ")
        kind = _kind(name, r.get("ETF", "N"))
        if kind == "Stock" and sec and _NASDAQ_KIND_NOT_COMMON.search(sec):
            kind = None
        if kind is None:
            continue
        out[sym] = {"name": (base if base and kind == "Stock" else name).strip(), "exchange": "Nasdaq", "type": kind}
    return out


def parse_other_listed(text: str) -> dict[str, dict]:
    """{yahoo symbol: {"name", "exchange", "type"}} from otherlisted.txt (ACT Symbol|Security Name|Exchange|CQS Symbol|
    ETF|Round Lot Size|Test Issue|NASDAQ Symbol). Preferred shares ('$' in the symbol), units, warrants and rights
    (.U / .W / .WS / .R suffixes) are left out; class shares keep their letter (BRK.B -> BRK-B)."""
    out = {}
    for r in _rows(text):
        sym, name = r.get("ACT Symbol", ""), r.get("Security Name", "")
        if r.get("Test Issue") == "Y" or not re.fullmatch(r"[A-Z]{1,5}(\.[A-Z]{1,2})?", sym) or _OTHER_BAD_SUFFIX.search(sym):
            continue
        kind = _kind(name, r.get("ETF", "N"))
        if kind is None:
            continue
        ex = r.get("Exchange", "")
        nm = " ".join(name.split())
        if kind == "Stock":
            nm = re.sub(r"\s+(New\s+)?(Common Stock|Common Shares|Ordinary Shares)$", "", nm, flags=re.I) or nm
        out[yahoo_symbol(sym)] = {"name": nm, "exchange": EXCHANGES.get(ex, ex), "type": kind}
    return out


def parse_screener_caps(doc: dict) -> dict[str, float]:
    """{yahoo symbol: market cap in USD} from Nasdaq's stock screener JSON (api.nasdaq.com/api/screener/stocks?
    download=true: {"data": {"rows": [{"symbol", "marketCap", ...}]}}); symbols with no cap are left out."""
    rows = ((doc or {}).get("data") or {}).get("rows") or []
    out = {}
    for r in rows:
        sym = yahoo_symbol(str(r.get("symbol") or "").replace("/", "."))
        try:
            cap = float(str(r.get("marketCap") or "").replace(",", ""))
        except ValueError:
            continue
        if sym and cap > 0:
            out[sym] = cap
    return out


def parse_sec_mutual_funds(doc: dict) -> list[str]:
    """Every mutual fund / ETF share-class symbol in the SEC's company_tickers_mf.json ({"fields": [..., "symbol"],
    "data": [[cik, seriesId, classId, symbol], ...]}), sorted."""
    fields = list((doc or {}).get("fields") or [])
    i = fields.index("symbol") if "symbol" in fields else 3
    out = set()
    for row in (doc or {}).get("data") or []:
        try:
            s = str(row[i]).strip().upper()
        except (IndexError, TypeError):
            continue
        if re.fullmatch(r"[A-Z]{1,6}", s):
            out.add(s)
    return sorted(out)


def build_listing(nasdaq_text: str, other_text: str, caps: dict[str, float] | None = None,
                  mutual_funds: list[str] | None = None, old: dict | None = None, today: str = "",
                  min_rows: int = 1000) -> dict:
    """The data/listed_symbols.json document. A part that failed to download (None / empty) keeps the saved copy:
    the symbols when either directory file is missing or implausibly short, the market caps when the screener failed
    (caps are kept per symbol, with the date they were read), the mutual fund list when the SEC file failed."""
    old = old or {}
    syms = {**parse_other_listed(other_text), **parse_nasdaq_listed(nasdaq_text)}
    n_nas = len(parse_nasdaq_listed(nasdaq_text))
    fresh = n_nas >= min_rows and len(syms) - n_nas >= min_rows
    if not fresh:
        syms = {t: {k: v for k, v in r.items() if k != "cap"} for t, r in (old.get("symbols") or {}).items()}
    old_caps = {t: r["cap"] for t, r in (old.get("symbols") or {}).items() if r.get("cap")}
    cap_src = {}
    if caps:
        cap_src = {t: round(c / 1e6) for t, c in caps.items()}
    for t, r in syms.items():
        if r.get("type") != "Stock":
            continue
        c = cap_src.get(t) if cap_src else old_caps.get(t)
        if c:
            r["cap"] = int(c)
    counts = {k: sum(1 for r in syms.values() if r["type"] == k) for k in ("Stock", "ETF", "ETN")}
    mf_ok = bool(mutual_funds) and len(mutual_funds) >= min_rows
    mf = sorted(set(mutual_funds)) if mf_ok else list(old.get("mutual_funds") or [])
    return {
        "updated": today if fresh else old.get("updated", today),
        "caps_updated": today if cap_src else old.get("caps_updated"),
        "mutual_funds_updated": today if mf_ok else old.get("mutual_funds_updated"),
        "source": "NASDAQ Trader symbol directory (nasdaqlisted.txt, otherlisted.txt; test issues, warrants, rights, "
                  "units, preferreds and listed debt left out; '.' written '-' as Yahoo does); cap = market cap in $M "
                  "from Nasdaq's stock screener; mutual_funds = the SEC's company_tickers_mf.json share-class symbols",
        "counts": {**counts, "mutual_funds": len(mf)},
        "symbols": dict(sorted(syms.items())),
        "mutual_funds": mf,
    }


def dumps_listing(doc: dict) -> str:
    """One symbol per line (small diffs from one day to the next); the mutual fund list 25 symbols per line."""
    head = {k: v for k, v in doc.items() if k not in ("symbols", "mutual_funds")}
    lines = ["{"] + [f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in head.items()]
    lines.append('"symbols": {')
    items = list((doc.get("symbols") or {}).items())
    lines += [f"{json.dumps(t)}: {json.dumps(r, ensure_ascii=False, separators=(',', ':'))}"
              + ("," if i < len(items) - 1 else "") for i, (t, r) in enumerate(items)]
    lines.append("},")
    mf = list(doc.get("mutual_funds") or [])
    chunks = [mf[i:i + 25] for i in range(0, len(mf), 25)]
    lines.append('"mutual_funds": [')
    lines += [", ".join(json.dumps(s) for s in c) + ("," if i < len(chunks) - 1 else "") for i, c in enumerate(chunks)]
    lines.append("]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def listed_tier(doc: dict, fund_assets: dict | None = None, exclude=(), min_cap: float = LISTED_MIN_MARKET_CAP) -> list[str]:
    """The listing symbols the data job downloads by itself, largest first: every ETF / ETN (ordered by net assets
    when Yahoo's fund metadata has them, `fund_assets` {symbol: USD}) and every stock with a market cap of at least
    `min_cap` (by cap), minus `exclude` (the symbols another list already downloads). Stocks with no known cap are
    left out (they can still be requested)."""
    ex = set(exclude)
    fund_assets = fund_assets or {}
    ranked = []
    for t, r in (doc.get("symbols") or {}).items():
        if t in ex:
            continue
        if r.get("type") in ("ETF", "ETN"):
            size = float(fund_assets.get(t) or 0)
        else:
            cap = float(r.get("cap") or 0) * 1e6
            if cap < min_cap:
                continue
            size = cap
        ranked.append((-size, t))
    return [t for _, t in sorted(ranked)]


# ------------------------------------------------------------------ the queue (data/requested_tickers.txt)

QUEUE_HEADER = ("# Tickers the site or the CLI was asked for that had no price data yet (one per line; '#' starts a "
                "comment).\n# The 'Fetch price data' workflow downloads these FIRST, then moves each into its regular "
                "refresh.\n# Push this file (or run the workflow) to fetch them now; otherwise the next daily run "
                "does.\n")


def queue_file() -> Path:
    """data/requested_tickers.txt, or $BACKTESTER_QUEUE_FILE (the tests point it at a temporary file)."""
    import os
    env = os.environ.get("BACKTESTER_QUEUE_FILE")
    return Path(env) if env else QUEUE_FILE


def read_queue(path: Path | None = None) -> list[str]:
    """The queued symbols, in order (upper-cased, de-duplicated; comments and anything not ticker-shaped skipped)."""
    path = path or queue_file()
    try:
        text = path.read_text()
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        for tok in line.split("#", 1)[0].replace(",", " ").split():
            t = tok.strip().upper().lstrip("$")
            if re.fullmatch(SYMBOL_RE, t):
                out.append(t)
    return list(dict.fromkeys(out))


def enqueue(ticker: str, note: str = "", path: Path | None = None) -> str:
    """Append a symbol to the queue: 'added' or 'already requested'. Raises ValueError for a non-symbol."""
    path = path or queue_file()
    t = str(ticker or "").strip().upper().lstrip("$")
    if not re.fullmatch(SYMBOL_RE, t):
        raise ValueError(f"{ticker!r} is not a ticker symbol.")
    if t in read_queue(path):
        return "already requested"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text() if path.exists() else ""
    with path.open("a") as f:
        if not text:
            f.write(QUEUE_HEADER)
        elif not text.endswith("\n"):
            f.write("\n")
        f.write(t + (f"  # {note}" if note else "") + "\n")
    return "added"


def write_queue(symbols: list[str], notes: dict | None = None, path: Path | None = None) -> None:
    """Rewrite the queue with `symbols` (the data job, after a run: what is still waiting)."""
    path = path or queue_file()
    notes = notes or {}
    path.write_text(QUEUE_HEADER + "".join(t + (f"  # {notes[t]}" if notes.get(t) else "") + "\n" for t in symbols))


# ------------------------------------------------------------------ lookups for the site / CLI

def _mtime(p: Path):
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return None


@lru_cache(maxsize=2)
def _listing(path: str, _mt) -> dict:
    try:
        doc = json.loads(Path(path).read_text())
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def listing() -> dict:
    """data/listed_symbols.json (empty when the data job hasn't written it yet)."""
    return _listing(str(LISTED_FILE), _mtime(LISTED_FILE))


@lru_cache(maxsize=2)
def _mf_set(path: str, _mt) -> frozenset:
    return frozenset(_listing(path, _mt).get("mutual_funds") or [])


def lookup(ticker: str) -> dict | None:
    """What the directories say a symbol is: {"name", "exchange", "type"} for a listed stock / ETF / ETN,
    {"type": "Mutual fund", ...} for a symbol in the curated fund lists or the SEC's mutual fund list, else None."""
    t = str(ticker or "").strip().upper()
    r = (listing().get("symbols") or {}).get(t)
    if r:
        return dict(r)
    from . import fund_lists
    if t in fund_lists.MUTUAL_FUNDS:
        return {"type": "Mutual fund", "source": "the curated mutual fund list"}
    if t in fund_lists.ALL_FUNDS:
        return {"type": "ETF", "source": "the curated ETF list"}
    if t in _mf_set(str(LISTED_FILE), _mtime(LISTED_FILE)):
        return {"type": "Mutual fund", "source": "the SEC's mutual fund symbol list"}
    return None


def describe(t: str, info: dict) -> str:
    bits = [info.get("name"), info.get("type") if info.get("type") != "Stock" else "stock", info.get("exchange")]
    return f"{t} ({', '.join(b for b in bits if b)})"


def missing_message(ticker: str, offline: bool) -> str | None:
    """For a symbol with no price file: when a directory knows it (listed stock / ETF, or a known mutual fund), queue it
    (data/requested_tickers.txt) and say that the next data refresh adds it; None for a symbol no directory knows."""
    t = str(ticker or "").strip().upper()
    info = lookup(t)
    if not info:
        return None
    try:
        how = enqueue(t, "requested from the site / CLI")
    except (OSError, ValueError):
        how = "not queued"
    where = {"added": "It has been added to data/requested_tickers.txt",
             "already requested": "It is already in data/requested_tickers.txt"}.get(how, "Add it to data/requested_tickers.txt")
    why = ("there is no internet access here (BACKTESTER_OFFLINE)" if offline
           else "the download from Yahoo Finance just failed (try again in a few minutes)")
    return (f"{describe(t, info)} is a valid symbol but its price history isn't downloaded yet, and {why}. {where}: "
            "the next data refresh (the daily 'Fetch price data' workflow; pushing that file starts it now) downloads "
            "the queue first - then pull the new data.")


def missing_hint(symbols, offline: bool) -> str:
    """The advice after "no price data for X, Y" (the parser's unknown-ticker error): valid symbols are queued and the
    next data refresh adds them; for the others, how to request them by hand."""
    syms = [str(s).strip().upper() for s in symbols if str(s).strip()]
    known = {s: lookup(s) for s in syms}
    valid = [s for s in syms if known[s]]
    rest = [s for s in syms if not known[s]]
    parts = []
    if valid:
        for s in valid:
            try:
                enqueue(s, "requested from the site / CLI")
            except (OSError, ValueError):
                pass
        why = "there is no internet access here" if offline else "the download from Yahoo Finance failed"
        parts.append(f"{', '.join(describe(s, known[s]) for s in valid)} {'is a valid symbol' if len(valid) == 1 else 'are valid symbols'} "
                     f"not downloaded yet ({why}): queued in data/requested_tickers.txt, which the next data refresh "
                     "(the daily 'Fetch price data' workflow; pushing the file starts it now) downloads first - then pull.")
    if rest:
        parts.append(f"If {', '.join(rest)} {'is' if len(rest) == 1 else 'are'} right, add "
                     f"{'it' if len(rest) == 1 else 'them'} to data/requested_tickers.txt or data/extra_tickers.txt and "
                     "run the 'Fetch price data' workflow (GitHub Actions), then pull.")
    return " ".join(parts)


def counts(available: set[str] | None = None) -> dict:
    """Coverage counts for the ticker directory: how much of the US listing and the fund lists has price data."""
    from . import fund_lists
    if available is None:
        from . import data
        available = set(data.available_tickers())
    doc = listing()
    syms = doc.get("symbols") or {}
    out: dict = {"listing_date": doc.get("updated"), "queued": len(read_queue())}
    for label, kinds in (("stocks", ("Stock",)), ("etfs", ("ETF", "ETN"))):
        ts = [t for t, r in syms.items() if r.get("type") in kinds]
        out[label] = {"listed": len(ts), "with_data": sum(1 for t in ts if t in available)}
        if label == "stocks":
            big = [t for t in ts if float(syms[t].get("cap") or 0) * 1e6 >= LISTED_MIN_MARKET_CAP]
            out[label]["in_scope"] = len(big)
            out[label]["in_scope_with_data"] = sum(1 for t in big if t in available)
    mf = list(fund_lists.MUTUAL_FUNDS)
    out["mutual_funds"] = {"curated": len(mf), "with_data": sum(1 for t in mf if t in available),
                           "sec_symbols": len(doc.get("mutual_funds") or [])}
    out["min_market_cap"] = LISTED_MIN_MARKET_CAP
    return out
