"""Fund research: a screener table of every ETF and mutual fund with data, a detail view, a comparison and a ticker
directory.

Fund facts come from three places, best first:
  data/funds_meta.json      Yahoo (Ticker.info / funds_data), filled by the data job in rotating batches
                            (scripts/fetch_data.py fetch_funds_meta): name, category, family, expense ratio,
                            inception, net assets, yield, holdings, and quote_type (ETF / MUTUALFUND)
  data/fund_reference.json  the issuers' own fund catalogs (Vanguard, iShares, SPDR, Invesco, Schwab, Dimensional),
                            collected on its "as_of" date: name, family, category, expense ratio, inception. Used for
                            any field Yahoo has not supplied yet, so the screener works before the metadata arrives.
  data/ticker_info.json     the name and instrument type Yahoo sent with each price download
A fund's type (ETF or mutual fund) is Yahoo's quote_type / instrument type when known, else the issuer table's, else
the built-in fund lists' (backtester/fund_lists.py), else a 5-letter symbol ending in X is a mutual fund.

Performance never depends on the metadata: trailing returns, volatility and drawdown are computed from our own price
files (total return: adj_close, dividends reinvested), so a fund with no metadata still gets a full row.

Statistics per fund (as of its last price):
  r1y            total return over the last 12 months
  r3y r5y r10y   annualised total return over 3 / 5 / 10 years (None when the fund is younger)
  cagr           annualised total return since the first price on file
  vol3y          annualised standard deviation of monthly returns over the last 36 months (at least 12)
  max_dd         the largest peak-to-trough fall of the daily total-return index over the whole history
They come from data/fund_stats.json (precomputed by the data job after data.load's integrity repairs) when its entry
matches the price file (same size and last bytes), else from the price file directly (quick) and then, in a background
pass, after the integrity repairs; results are cached in reports/fund_stats.json. The table answers within a time
budget (SYNC_BUDGET seconds): funds still being computed come back as "pending" and the page asks again.
"""
from __future__ import annotations

import json
import threading
import time
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, fund_lists

META_FILE = data.DATA / "funds_meta.json"
REFERENCE_FILE = data.DATA / "fund_reference.json"
INFO_FILE = data.DATA / "ticker_info.json"
STATS_FILE = data.DATA / "fund_stats.json"        # precomputed by the data job (shipped with the data)
CACHE_FILE = data.ROOT / "reports" / "fund_stats.json"
MAX_COMPARE = 6
STATS_VERSION = 4             # bump when stats_from_prices changes: cached statistics are recomputed
SYNC_BUDGET = 2.5             # seconds /api/funds spends computing missing statistics before answering

_LOCK = threading.Lock()
_CACHE: dict[str, dict] = {}          # ticker -> {"key": file key, "stats": {...}, "checked": bool}
_WARM = {"running": False, "done": 0, "total": 0}
WARM = True                           # False: no background pass (tests)


# ---------------------------------------------------------------- metadata

def _mtime(p: Path):
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return None


@lru_cache(maxsize=8)
def _json_doc(path: str, _mt) -> dict:
    try:
        doc = json.loads(Path(path).read_text())
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def meta_doc() -> dict:
    return _json_doc(str(META_FILE), _mtime(META_FILE))


def reference_doc() -> dict:
    return _json_doc(str(REFERENCE_FILE), _mtime(REFERENCE_FILE))


def info_doc() -> dict:
    return _json_doc(str(INFO_FILE), _mtime(INFO_FILE))


def meta(ticker: str) -> dict:
    """Yahoo's fund metadata (data/funds_meta.json) for one fund."""
    return dict((meta_doc().get("funds") or {}).get(data.canonical(ticker)) or {})


def reference(ticker: str) -> dict:
    """The issuer's fund facts (data/fund_reference.json) for one fund."""
    return dict((reference_doc().get("funds") or {}).get(data.canonical(ticker)) or {})


def ticker_info(ticker: str) -> dict:
    return dict((info_doc().get("tickers") or {}).get(data.canonical(ticker)) or {})


@lru_cache(maxsize=1)
def _list_categories() -> dict[str, str]:
    """Ticker -> the built-in list's group ('US small cap', 'Vanguard mutual fund', ...), a fallback category."""
    out = {}
    for group, txt in fund_lists.BROAD_ETFS_BY_CLASS.items():
        for t in txt.split():
            out.setdefault(t, group)
    for fam, txt in getattr(fund_lists, "MUTUAL_FUNDS_BY_FAMILY", {}).items():
        fam = fam.split(" (")[0]
        for t in (txt.split() if isinstance(txt, str) else txt):
            out.setdefault(t, "mutual fund" if fam == "other" else f"{fam} mutual fund")
    for t in fund_lists.LEVERAGED_ETFS.split():
        out.setdefault(t, "leveraged / inverse")
    return out


_TYPES = {"MUTUALFUND": "Mutual fund", "ETF": "ETF", "EQUITY": "Stock", "INDEX": "Index",
          "CRYPTOCURRENCY": "Crypto", "MONEYMARKET": "Money market"}


def fund_type(t: str, m: dict | None = None) -> str:
    """'ETF' or 'Mutual fund' (see the module docstring for the order of the sources)."""
    m = meta(t) if m is None else m
    for qt in (m.get("quote_type"), ticker_info(t).get("type")):
        if qt and str(qt).upper() in ("MUTUALFUND", "ETF"):
            return _TYPES[str(qt).upper()]
    ref = reference(t).get("type")
    if ref in ("ETF", "Mutual fund"):
        return ref
    if t in _MUTUAL_FUND_SET:
        return "Mutual fund"
    if t in _ETF_SET:
        return "ETF"
    return "Mutual fund" if len(t) == 5 and t.endswith("X") and t.isalpha() else "ETF"


_MUTUAL_FUND_SET = frozenset(fund_lists.MUTUAL_FUNDS)
_ETF_SET = frozenset(fund_lists.BROAD_ETFS)


def universe() -> list[tuple[str, str]]:
    """[(ticker, 'ETF' | 'Mutual fund')] of the funds with a price file: the data job's ETF and fund lists, the
    built-in fund lists, the issuer table and anything Yahoo calls an ETF or mutual fund (never a stock, index or
    simulated series)."""
    have = set(data.available_tickers())
    um = data.universe_meta()
    stocks = set(um.get("stocks", [])) | set(um.get("broad_stocks", [])) | set(um.get("nasdaq100", []))
    cands = list(dict.fromkeys(data.etfs() + data.funds() + sorted(fund_lists.ALL_FUNDS)
                               + sorted(reference_doc().get("funds") or {})
                               + sorted(t for t, m in (meta_doc().get("funds") or {}).items()
                                        if str(m.get("quote_type") or "").upper() in ("ETF", "MUTUALFUND"))
                               + sorted(t for t, m in (info_doc().get("tickers") or {}).items()
                                        if str(m.get("type") or "").upper() in ("ETF", "MUTUALFUND"))
                               + sorted(t for t in have if len(t) == 5 and t.endswith("X") and t.isalpha())))
    out = []
    for t in cands:
        if t not in have or t.startswith("^") or "-" in t or t.endswith("SIM") or data.is_custom(t):
            continue
        if t in stocks and t not in fund_lists.ALL_FUNDS:
            continue
        if str(ticker_info(t).get("type") or "").upper() in ("EQUITY", "INDEX", "CRYPTOCURRENCY"):
            continue
        out.append((t, fund_type(t)))
    return out


# ---------------------------------------------------------------- statistics

def stats_from_prices(tr: pd.Series) -> dict:
    """Trailing statistics of a total-return price series (see the module docstring)."""
    s = pd.to_numeric(tr, errors="coerce").dropna()
    s = s[s > 0]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s) < 2:
        return {}
    last_d, last = s.index[-1], float(s.iloc[-1])
    first_d = s.index[0]
    out = {"first": str(first_d.date()), "last": str(last_d.date()),
           "years": round((last_d - first_d).days / 365.25, 2)}

    def trailing(years: int):
        cut = last_d - pd.DateOffset(years=years)
        if first_d > cut + pd.Timedelta(days=7):
            return None
        base = s[s.index <= cut]
        if base.empty:
            base = s.iloc[:1]
        r = last / float(base.iloc[-1])
        return float(r - 1) if years == 1 else float(r ** (1 / years) - 1)
    for y in (1, 3, 5, 10):
        out[f"r{y}y"] = trailing(y)
    yrs = (last_d - first_d).days / 365.25
    out["cagr"] = float((last / float(s.iloc[0])) ** (1 / yrs) - 1) if yrs >= 1 else None
    me = s.groupby(s.index.to_period("M")).last()
    mr = me.pct_change().dropna().iloc[-36:]
    out["vol3y"] = float(mr.std() * np.sqrt(12)) if len(mr) >= 12 else None
    out["max_dd"] = float((s / s.cummax() - 1).min())
    d = s.pct_change().dropna()
    if len(d):
        big = d.abs().idxmax()
        out["max_day"] = float(d.loc[big])
        out["max_day_date"] = str(big.date())
        typical = float(1.4826 * d.abs().median())          # a robust daily standard deviation
        out["max_day_sigmas"] = float(abs(d.loc[big]) / typical) if typical > 0 else None
    return out


SUSPECT_DAY = 0.30            # a one-day total return beyond +-30% ...
SUSPECT_SIGMAS = 30           # ... and over 30 typical daily moves: leveraged and volatility funds move 10-20


def suspect(t: str, st: dict) -> str | None:
    """A warning when the fund's history has a one-day move far outside its own range (Yahoo sometimes books a
    large distribution or a reverse split wrongly: PCRAX +294% on 2008-12-10), else None."""
    v, k = st.get("max_day"), st.get("max_day_sigmas")
    if v is None or k is None or abs(v) <= SUSPECT_DAY or k <= SUSPECT_SIGMAS:
        return None
    return (f"a one-day total return of {v:+.0%} on {st.get('max_day_date')}, {k:.0f} times this fund's typical daily "
            "move: check for a data error in the price file (a distribution or split booked wrongly) before relying "
            "on its statistics")


def _key(t: str) -> str | None:
    """The price file's identity: size and a checksum of its last 256 bytes (the newest rows). Unlike a modification
    time it survives a git checkout, so data/fund_stats.json computed by the data job is valid on every machine."""
    try:
        p = data.price_path(t)
        size = p.stat().st_size
        with p.open("rb") as f:
            f.seek(max(0, size - 256))
            tail = f.read()
        return f"{STATS_VERSION}:{size}:{zlib.crc32(tail):08x}"
    except OSError:
        return None


def quick_stats(t: str) -> dict:
    """Statistics straight from the price file's adj_close (no integrity pass)."""
    p = data.price_path(t)
    df = pd.read_csv(p, usecols=lambda c: c in ("date", "adj_close", "close"))
    col = "adj_close" if "adj_close" in df else "close"
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(), index=pd.to_datetime(df["date"]))
    return stats_from_prices(s)


def checked_stats(t: str) -> dict:
    """Statistics after data.load's integrity repairs."""
    return stats_from_prices(data.load(t)["adj_close"])


_LOADED = {"done": False}


def _load_cache() -> None:
    """The precomputed file first (data/fund_stats.json), then the local cache on top."""
    if _LOADED["done"]:
        return
    _LOADED["done"] = True
    for f in (STATS_FILE, CACHE_FILE):
        try:
            doc = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            body = doc.get("stats") if isinstance(doc.get("stats"), dict) else doc
            for k, v in body.items():
                if isinstance(v, dict) and "key" in v and not (k in _CACHE and _CACHE[k].get("checked") and not v.get("checked")):
                    _CACHE[k] = v


def reset_cache() -> None:
    """Forget the in-memory statistics (tests; after new price files arrive)."""
    with _LOCK:
        _CACHE.clear()
        _LOADED["done"] = False


def _save_cache() -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            snap = dict(_CACHE)
        tmp = CACHE_FILE.with_suffix(".part")
        tmp.write_text(json.dumps(snap))
        tmp.replace(CACHE_FILE)
    except OSError:
        pass


def cached_stats(t: str, key: str | None = None) -> dict | None:
    """The cached statistics of `t` if they match its price file, else None (never computes)."""
    k = key if key is not None else _key(t)
    with _LOCK:
        _load_cache()
        hit = _CACHE.get(t)
    if hit and hit.get("key") == k:
        return dict(hit.get("stats") or {}, checked=bool(hit.get("checked")))
    return None


def stats(t: str, checked: bool = False) -> dict:
    """Cached statistics of one fund (checked=True forces the integrity-checked version)."""
    k = _key(t)
    with _LOCK:
        _load_cache()
        hit = _CACHE.get(t)
    if hit and hit.get("key") == k and (hit.get("checked") or not checked):
        return dict(hit.get("stats") or {}, checked=bool(hit.get("checked")))
    try:
        st = checked_stats(t) if checked else quick_stats(t)
    except Exception:  # noqa: BLE001 - an unreadable file: no statistics
        st = {}
    with _LOCK:
        _CACHE[t] = {"key": k, "stats": st, "checked": checked}
    return dict(st, checked=checked)


def _warm(tickers: list[str]) -> None:
    try:
        for i, t in enumerate(tickers):
            stats(t, checked=True)
            _WARM["done"] = i + 1
            if i % 100 == 99:
                _save_cache()
        _save_cache()
    finally:
        _WARM["running"] = False


def start_warm(tickers: list[str]) -> None:
    """Compute the integrity-checked statistics of `tickers` in a background thread (once at a time), in the order
    given (the table puts the funds still pending first)."""
    if not WARM:
        return
    todo = []
    with _LOCK:
        _load_cache()
        for t in tickers:
            hit = _CACHE.get(t)
            if not (hit and hit.get("checked") and hit.get("key") == _key(t)):
                todo.append(t)
        if not todo or _WARM["running"]:
            return
        _WARM.update(running=True, done=0, total=len(todo))
    threading.Thread(target=_warm, args=(todo,), daemon=True).start()


def precompute(path: Path | None = None, tickers: list[str] | None = None) -> int:
    """Write data/fund_stats.json: the integrity-checked statistics of every fund (the data job runs this after the
    downloads; entries whose price file is unchanged are reused). Returns the number of funds."""
    path = path or STATS_FILE
    try:
        old = json.loads(path.read_text()).get("stats") or {}
    except (OSError, ValueError, AttributeError):
        old = {}
    names = tickers if tickers is not None else [t for t, _ in universe()]
    out = {}
    for t in names:
        k = _key(t)
        if k is None:
            continue
        o = old.get(t)
        if o and o.get("key") == k and o.get("checked"):
            out[t] = o
            continue
        try:
            st = checked_stats(t)
        except Exception:  # noqa: BLE001 - an unreadable file: no statistics
            st = {}
        out[t] = {"key": k, "stats": st, "checked": True}
    doc = {"version": STATS_VERSION, "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
           "about": "Funds page statistics (backtester/funds.py stats_from_prices after data.load's integrity repairs); "
                    "key = version:file size:crc32 of the file's last 256 bytes", "stats": dict(sorted(out.items()))}
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(doc, separators=(",", ":")))
    tmp.replace(path)
    return len(out)


# ---------------------------------------------------------------- the table and the screener

META_FIELDS = ("name", "category", "family", "expense_ratio", "inception", "net_assets", "yield", "legal_type")
REF_FIELDS = ("name", "category", "family", "expense_ratio", "inception")
STAT_FILTERS = ("min_years", "min_r5y", "max_vol")


def facts(t: str) -> dict:
    """The fund's facts: Yahoo's metadata, with any missing field filled from the issuer table, then from Yahoo's
    download metadata (name) and the built-in lists (category). `sources` says where each filled field came from."""
    m, ref = meta(t), reference(t)
    r = {k: m.get(k) for k in META_FIELDS}
    src = {k: "Yahoo" for k in META_FIELDS if m.get(k) not in (None, "")}
    as_of = reference_doc().get("as_of")
    for k in REF_FIELDS:
        if r.get(k) in (None, "") and ref.get(k) not in (None, ""):
            r[k] = ref[k]
            src[k] = f"issuer ({ref.get('family') or 'fund company'}, as of {as_of})"
    if not r.get("name"):
        n = ticker_info(t).get("name")
        if n:
            r["name"], src["name"] = n, "Yahoo (price download)"
    if not r.get("category"):
        cat = _list_categories().get(t)
        if cat:
            r["category"], src["category"] = cat, "fund list"
    r["sources"] = src
    return r


def row(t: str, kind: str | None = None, checked: bool = False, compute: bool = True) -> dict:
    """One fund's table row. compute=False: only cached statistics (the row is "pending" when there are none)."""
    m = meta(t)
    f = facts(t)
    r = {"ticker": t, "type": kind or fund_type(t, m), **{k: f.get(k) for k in META_FIELDS}, "has_meta": bool(m),
         "meta_fetched": m.get("fetched"), "sources": f["sources"]}
    if f["sources"].get("category") == "fund list":
        r["category_source"] = "fund list"
    if f["sources"].get("expense_ratio", "").startswith("issuer"):
        r["er_source"] = f["sources"]["expense_ratio"]
    st = cached_stats(t) if not compute else stats(t, checked=checked)
    if st is None:
        r["pending"] = True
    else:
        r.update(st)
        w = suspect(t, r)
        if w:
            r["warning"] = w
    return r


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def screen(rows: list[dict], q: str | None = None, kind: str | None = None, category: str | None = None,
           max_er: float | None = None, min_years: float | None = None, min_aum: float | None = None,
           min_r5y: float | None = None, max_vol: float | None = None, include_unknown: bool = False,
           counts: dict | None = None) -> list[dict]:
    """The rows passing every given filter; q matches the ticker, the name, the category or the family
    (case-insensitive substring). A filter on a field a fund lacks (no expense ratio on file yet, statistics still
    being computed) excludes the fund unless include_unknown is set, and is counted in `counts`
    ({"unknown_er", "unknown_aum", "unknown_stats"}) so the page can say how many were left out."""
    out = []
    ql = (q or "").strip().lower()
    c = counts if counts is not None else {}
    for k in ("unknown_er", "unknown_aum", "unknown_stats"):
        c.setdefault(k, 0)
    which = {"expense_ratio": "unknown_er", "net_assets": "unknown_aum"}
    for r in rows:
        if ql and not any(ql in str(r.get(f) or "").lower() for f in ("ticker", "name", "category", "family")):
            continue
        if kind and kind.lower() not in ("all", "") and str(r.get("type") or "").lower() != kind.lower():
            continue
        if category and str(r.get("category") or "").lower() != category.lower():
            continue
        checks = ((max_er, "expense_ratio", lambda v, x: v <= x), (min_years, "years", lambda v, x: v >= x),
                  (min_aum, "net_assets", lambda v, x: v >= x), (min_r5y, "r5y", lambda v, x: v >= x),
                  (max_vol, "vol3y", lambda v, x: v <= x))
        ok, unknown = True, None
        for lim, field, cmp in checks:
            if lim is None:
                continue
            v = _num(r.get(field))
            if v is None:
                if r.get("pending") and field in ("years", "r5y", "vol3y"):
                    unknown = unknown or "unknown_stats"
                elif field in which:
                    unknown = unknown or which[field]
                if not include_unknown:
                    ok = False
                    break
                continue
            if not cmp(v, float(lim)):
                ok = False
                unknown = None
                break
        if unknown and not ok:
            c[unknown] += 1
        if ok:
            out.append(r)
    return out


def table(filters: dict | None = None, budget: float | None = None) -> dict:
    """Every fund's row, filtered. Statistics come from the caches; missing ones are computed for up to `budget`
    seconds (SYNC_BUDGET) and the rest are marked pending while a background pass computes them."""
    f = dict(filters or {})
    budget = SYNC_BUDGET if budget is None else budget
    uni = universe()
    t0 = time.time()
    rows, pending = [], []
    q_filters = {k: f.get(k) for k in ("q", "kind", "category")}
    for t, kind in uni:
        r = row(t, kind, compute=False)
        if r.get("pending") and (not any(q_filters.values()) or screen([r], **q_filters)):
            if time.time() - t0 < budget:
                r = row(t, kind, compute=True)
            else:
                pending.append(t)
        rows.append(r)
    start_warm(pending + [t for t, _ in uni if t not in set(pending)])
    counts: dict = {}
    rows = screen(rows, **{k: f.get(k) for k in ("q", "kind", "category", "max_er", "min_years", "min_aum",
                                                 "min_r5y", "max_vol")},
                  include_unknown=str(f.get("include_unknown") or "").lower() in ("1", "true", "yes", "on"),
                  counts=counts)
    doc = meta_doc()
    cats = sorted({str(r.get("category")) for r in rows if r.get("category")})
    n_meta = sum(1 for r in rows if r["has_meta"])
    no_er = sum(1 for r in rows if _num(r.get("expense_ratio")) is None)
    notes = []
    if not doc:
        notes.append("No Yahoo fund metadata yet (data/funds_meta.json is filled by the data job in batches); names, "
                     "categories and expense ratios come from the issuers' fund lists (data/fund_reference.json) "
                     "where they have them. Returns and risk come from the price files.")
    elif n_meta < len(rows):
        notes.append(f"Metadata for {n_meta} of {len(rows)} funds from Yahoo so far (the data job adds a batch each run); "
                     "for the others, names and expense ratios come from the issuers' fund lists where available and "
                     "the statistics from the price files.")
    ref = reference_doc()
    if ref.get("as_of"):
        notes.append(f"Expense ratios marked 'issuer' are from the fund companies' own lists as of {ref['as_of']} "
                     f"({', '.join(sorted({v.get('family') for v in (ref.get('funds') or {}).values() if v.get('family')}))}).")
    return {"funds": rows, "categories": cats, "count": len(rows), "universe": len(uni),
            "pending": len(pending), "unknown": counts, "no_expense_ratio": no_er,
            "meta_updated": doc.get("updated_utc"), "meta_source": doc.get("source"),
            "reference_as_of": ref.get("as_of"), "checking": dict(_WARM), "notes": notes}


def detail(t: str) -> dict:
    t = data.canonical(t)
    if not data.ensure(t):
        raise ValueError(data.unknown_ticker_message(t))
    kinds = dict(universe())
    f = facts(t)
    m = {**meta(t), **{k: v for k, v in f.items() if k != "sources" and v not in (None, "")}}
    return {"ticker": t, "type": kinds.get(t) or fund_type(t), "meta": m, "sources": f["sources"],
            "stats": stats(t, checked=True), "notes": data.on_demand_notes([t])}


# ---------------------------------------------------------------- the ticker directory

def _directory_key() -> tuple:
    from . import coverage
    return tuple(_mtime(p) for p in (META_FILE, REFERENCE_FILE, INFO_FILE, data.DATA / "index_constituents.json",
                                     data.UNIVERSE_FILE, data.PRICES, coverage.LISTED_FILE))


def _directory_entries() -> list[dict]:
    return list(_directory(_directory_key()))


@lru_cache(maxsize=2)
def _coverage(_key) -> dict:
    from . import coverage
    return coverage.counts(set(data.available_tickers()))


@lru_cache(maxsize=2)
def _directory(_key) -> tuple:
    have = set(data.available_tickers())
    um = data.universe_meta()
    idx = data.index_constituents()
    names: dict[str, str] = {}
    kinds: dict[str, str] = {}
    cats: dict[str, str] = {}
    fams: dict[str, str] = {}
    member: dict[str, list[str]] = {}
    for t, m in (info_doc().get("tickers") or {}).items():
        if m.get("name"):
            names[t] = m["name"]
        if m.get("type"):
            kinds[t] = _TYPES.get(str(m["type"]).upper(), str(m["type"]).title())
    for label, key in (("S&P 500", "sp500"), ("S&P 400", "sp400"), ("S&P 600", "sp600")):
        for t, n in (idx.get(key) or {}).items():
            names.setdefault(t, n)
            kinds.setdefault(t, "Stock")
            member.setdefault(t, []).append(label)
    for t, n in ((idx.get("other_large") or {}).get("names") or {}).items():
        names.setdefault(t, n)
        kinds.setdefault(t, "Stock")
    # every US-listed stock, ETF and ETN (data/listed_symbols.json), with or without price data yet
    from . import coverage
    listed = coverage.listing().get("symbols") or {}
    for t, r in listed.items():
        if r.get("name"):
            names.setdefault(t, r["name"])
        kinds.setdefault(t, "ETF" if r.get("type") in ("ETF", "ETN") else "Stock")
    for t in um.get("nasdaq100", []):
        kinds.setdefault(t, "Stock")
        member.setdefault(t, []).insert(0, "Nasdaq-100")
    for t in list(um.get("stocks", [])) + list(um.get("broad_stocks", [])) + list(um.get("former_members", [])):
        kinds.setdefault(t, "Stock")
    for t in um.get("former_members", []):
        if t not in um.get("nasdaq100", []):
            member.setdefault(t, []).append("former Nasdaq-100")
    for src in (reference_doc().get("funds") or {}, meta_doc().get("funds") or {}):
        for t, m in src.items():
            if m.get("name"):
                names[t] = m["name"]
            if m.get("category"):
                cats[t] = m["category"]
            if m.get("family"):
                fams[t] = m["family"]
    for t in set(fund_lists.ALL_FUNDS) | set(um.get("etfs", [])) | set(um.get("funds", [])) | set(reference_doc().get("funds") or {}):
        if t in have or t in names or t in fund_lists.ALL_FUNDS:
            kinds[t] = fund_type(t)
    lc = _list_categories()
    for t, c in lc.items():
        cats.setdefault(t, c)
    for t in um.get("indexes", []):
        kinds[t] = "Index"
    for t, about in data.SIMS.items():
        names.setdefault(t, about)
        kinds[t] = "Simulated"
    out = []
    for t in sorted(have | set(names) | set(fund_lists.ALL_FUNDS) | set(member) | set(listed)):
        if t.startswith("."):
            continue
        kind = kinds.get(t) or ("Index" if t.startswith("^") else "Simulated" if t.endswith("SIM") else
                                "Custom" if data.is_custom(t) else "Crypto" if t.endswith("-USD") else "Stock")
        out.append({"ticker": t, "name": names.get(t) or "", "type": kind, "category": cats.get(t) or "",
                    "family": fams.get(t) or "", "indexes": member.get(t, []), "has_data": t in have})
    return tuple(out)


def directory(q: str = "", kind: str | None = None, limit: int = 100, has_data: bool | None = None) -> dict:
    """Search every ticker with data or metadata by ticker, name, category, family, type or index ('S&P 500'):
    every word of `q` must match somewhere. Exact tickers first, then tickers starting with the query, then the rest
    (those with price data before those still to be downloaded)."""
    words = [w for w in str(q or "").lower().split() if w]
    ql = " ".join(words)
    rows = _directory_entries()
    hits = []
    for r in rows:
        if kind and kind.lower() not in ("", "all") and r["type"].lower() != kind.lower():
            continue
        if has_data is not None and r["has_data"] != has_data:
            continue
        if words:
            hay = " ".join([r["ticker"], r["name"], r["category"], r["family"], r["type"], " ".join(r["indexes"])]).lower()
            if not all(w in hay for w in words):
                continue
        t = r["ticker"].lower()
        rank = (0 if t == ql else 1 if ql and t.startswith(ql) else 2, not r["has_data"], len(r["ticker"]), r["ticker"])
        hits.append((rank, r))
    hits.sort(key=lambda x: x[0])
    counts: dict = {}
    for r in rows:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    from . import coverage
    queued = set(coverage.read_queue())
    results = [dict(r, queued=True) if r["ticker"] in queued and not r["has_data"] else r
               for _, r in hits[:max(1, int(limit))]]
    return {"q": q, "total": len(hits), "results": results, "counts": counts,
            "size": len(rows), "with_data": sum(1 for r in rows if r["has_data"]),
            "coverage": _coverage(_directory_key() + (_mtime(coverage.queue_file()),))}


# ---------------------------------------------------------------- comparison

def compare(tickers: list[str], start=None, end=None) -> dict:
    """2-6 funds over their common period: growth of $10,000 (month-end values), calendar-year returns, the
    correlation matrix of monthly returns and each fund's statistics (correlation.analyze), plus metadata."""
    from . import correlation
    tickers = list(dict.fromkeys(data.canonical(t) for t in tickers if str(t).strip()))
    if not 2 <= len(tickers) <= MAX_COMPARE:
        raise ValueError(f"Compare 2 to {MAX_COMPARE} funds (got {len(tickers)}).")
    missing = [t for t in tickers if not data.ensure(t)]
    if missing:
        raise ValueError(f"No price data for {', '.join(missing)}.")
    R = correlation.analyze(tickers, "monthly", None, start, end)
    px = pd.concat({t: data.load(t)["adj_close"] for t in tickers}, axis=1, sort=True).dropna()
    if start:
        px = px[px.index >= pd.Timestamp(start)]
    if end:
        px = px[px.index <= pd.Timestamp(end)]
    # month-end values (the latest month to the last price), from the first common day
    per = px.index.to_period("M")
    me = px.groupby(per).last()
    me.index = pd.DatetimeIndex(px.index.to_series().groupby(per).max().to_numpy())
    me = pd.concat([px.iloc[:1], me[me.index > px.index[0]]])
    growth = me / px.iloc[0] * 10_000
    yr = px.groupby(px.index.year).last()
    base = pd.concat([px.iloc[:1].set_axis([px.index[0].year - 1]), yr])
    annual = (base / base.shift(1) - 1).iloc[1:]
    partial = {int(px.index[0].year): "from " + str(px.index[0].date())} if px.index[0].month > 1 or px.index[0].day > 7 else {}
    if px.index[-1] < pd.Timestamp(px.index[-1].year, 12, 24):
        partial[int(px.index[-1].year)] = "to " + str(px.index[-1].date())
    return {"tickers": tickers, "start": str(px.index[0].date()), "end": str(px.index[-1].date()),
            "growth": {"dates": [d.strftime("%Y-%m-%d") for d in growth.index],
                       "values": {t: [round(float(v), 2) for v in growth[t]] for t in tickers}},
            "annual": {"years": [int(y) for y in annual.index],
                       "values": {t: [round(float(v), 6) for v in annual[t]] for t in tickers}, "partial": partial},
            "correlation": {"matrix": R["matrix"], "observations": R["observations"]},
            "stats": R["stats"], "notes": list(R["notes"]) + data.on_demand_notes(tickers),
            "meta": {t: {k: facts(t).get(k) for k in META_FIELDS} for t in tickers}}
