"""Fund research: a screener table of every ETF and mutual fund with data, a detail view and a comparison.

Metadata (name, category, family, expense ratio, inception, net assets, yield, top holdings) comes from
data/funds_meta.json, which the data job fills from Yahoo in rotating batches (scripts/fetch_data.py,
fetch_funds_meta). Performance never depends on it: trailing returns, volatility and drawdown are computed from
our own price files (total return: adj_close, dividends reinvested), so a fund with no metadata yet still gets a
full row, with its category taken from the built-in fund lists when Yahoo's is missing.

Statistics per fund (as of its last price):
  r1y            total return over the last 12 months
  r3y r5y r10y   annualised total return over 3 / 5 / 10 years (None when the fund is younger)
  cagr           annualised total return since the first price on file
  vol3y          annualised standard deviation of monthly returns over the last 36 months (at least 12)
  max_dd         the largest peak-to-trough fall of the daily total-return index over the whole history
The table computes them from the price file directly (fast); `checked` rows come from data.load, i.e. after the
price-integrity repairs (missed splits, bad ticks), and replace the quick ones once the background pass reaches
them. Results are cached by file modification time in reports/fund_stats.json.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, fund_lists

META_FILE = data.DATA / "funds_meta.json"
CACHE_FILE = data.ROOT / "reports" / "fund_stats.json"
MAX_COMPARE = 6
STATS_VERSION = 3             # bump when stats_from_prices changes: cached statistics are recomputed

_LOCK = threading.Lock()
_CACHE: dict[str, dict] = {}          # ticker -> {"key": file key, "stats": {...}, "checked": bool}
_WARM = {"running": False, "done": 0, "total": 0}
WARM = True                           # False: no background pass (tests)


# ---------------------------------------------------------------- metadata

def meta_doc() -> dict:
    return _meta_doc(str(META_FILE), _mtime(META_FILE))


@lru_cache(maxsize=4)
def _meta_doc(path: str, _mt) -> dict:
    try:
        doc = json.loads(Path(path).read_text())
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def _mtime(p: Path):
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return None


def meta(ticker: str) -> dict:
    return dict((meta_doc().get("funds") or {}).get(data.canonical(ticker)) or {})


@lru_cache(maxsize=1)
def _list_categories() -> dict[str, str]:
    """Ticker -> the built-in list's group ('US small cap', 'Vanguard', ...), a fallback category."""
    out = {}
    for group, txt in fund_lists.BROAD_ETFS_BY_CLASS.items():
        for t in txt.split():
            out.setdefault(t, group)
    for fam, txt in getattr(fund_lists, "MUTUAL_FUNDS_BY_FAMILY", {}).items():
        for t in (txt.split() if isinstance(txt, str) else txt):
            out.setdefault(t, f"{fam} mutual fund")
    return out


def universe() -> list[tuple[str, str]]:
    """[(ticker, 'ETF' | 'Mutual fund')] of the funds with a price file."""
    have = set(data.available_tickers())
    out, seen = [], set()
    for kind, lst in (("ETF", data.etfs()), ("Mutual fund", data.funds())):
        for t in lst:
            if t in have and t not in seen and not t.startswith("^") and "-" not in t:
                seen.add(t)
                out.append((t, kind))
    # mutual funds in the core download list (VFINX, VTSMX, ...): 5 letters ending in X
    for t in sorted(have - seen):
        if len(t) == 5 and t.endswith("X") and t.isalpha() and not t.endswith("SIM"):
            out.append((t, "Mutual fund"))
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
    try:
        st = data.price_path(t).stat()
        return f"{STATS_VERSION}:{st.st_mtime_ns}:{st.st_size}"
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


def _load_cache() -> None:
    if _CACHE:
        return
    try:
        doc = json.loads(CACHE_FILE.read_text())
        if isinstance(doc, dict):
            _CACHE.update({k: v for k, v in doc.items() if isinstance(v, dict)})
    except (OSError, ValueError):
        pass


def _save_cache() -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            snap = dict(_CACHE)
        CACHE_FILE.write_text(json.dumps(snap))
    except OSError:
        pass


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
    """Compute the integrity-checked statistics of `tickers` in a background thread (once at a time)."""
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


# ---------------------------------------------------------------- the table and the screener

META_FIELDS = ("name", "category", "family", "expense_ratio", "inception", "net_assets", "yield", "legal_type")


def row(t: str, kind: str, checked: bool = False) -> dict:
    m = meta(t)
    r = {"ticker": t, "type": kind, **{k: m.get(k) for k in META_FIELDS}, "has_meta": bool(m),
         "meta_fetched": m.get("fetched")}
    if not r["category"]:
        cat = _list_categories().get(t)
        if cat:
            r["category"] = cat
            r["category_source"] = "fund list"
    r.update(stats(t, checked=checked))
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
           min_r5y: float | None = None, max_vol: float | None = None) -> list[dict]:
    """The rows passing every given filter. A filter on a field a fund lacks (no expense ratio on file) excludes
    it; q matches the ticker or the name (case-insensitive substring)."""
    out = []
    ql = (q or "").strip().lower()
    for r in rows:
        if ql and ql not in r["ticker"].lower() and ql not in str(r.get("name") or "").lower():
            continue
        if kind and kind.lower() not in ("all", "") and r["type"].lower() != kind.lower():
            continue
        if category and str(r.get("category") or "").lower() != category.lower():
            continue
        checks = ((max_er, "expense_ratio", lambda v, x: v <= x), (min_years, "years", lambda v, x: v >= x),
                  (min_aum, "net_assets", lambda v, x: v >= x), (min_r5y, "r5y", lambda v, x: v >= x),
                  (max_vol, "vol3y", lambda v, x: v <= x))
        ok = True
        for lim, field, cmp in checks:
            if lim is None:
                continue
            v = _num(r.get(field))
            if v is None or not cmp(v, float(lim)):
                ok = False
                break
        if ok:
            out.append(r)
    return out


def table(filters: dict | None = None) -> dict:
    """Every fund's row (quick statistics until the background pass has checked them), filtered."""
    uni = universe()
    rows = [row(t, kind) for t, kind in uni]
    start_warm([t for t, _ in uni])
    f = filters or {}
    rows = screen(rows, **{k: f.get(k) for k in ("q", "kind", "category", "max_er", "min_years", "min_aum", "min_r5y",
                                                 "max_vol")})
    doc = meta_doc()
    cats = sorted({str(r.get("category")) for r in rows if r.get("category")})
    n_meta = sum(1 for r in rows if r["has_meta"])
    notes = []
    if not doc:
        notes.append("No fund metadata yet (data/funds_meta.json is filled by the data job in batches): names, "
                     "categories and expense ratios appear as it runs; returns and risk come from the price files.")
    elif n_meta < len(rows):
        notes.append(f"Metadata for {n_meta} of {len(rows)} funds so far (the data job adds a batch each run); the "
                     "others show statistics from the price files and the built-in list's category.")
    return {"funds": rows, "categories": cats, "count": len(rows), "universe": len(uni),
            "meta_updated": doc.get("updated_utc"), "meta_source": doc.get("source"),
            "checking": dict(_WARM), "notes": notes}


def detail(t: str) -> dict:
    t = data.canonical(t)
    kinds = dict(universe())
    if t not in set(data.available_tickers()):
        raise ValueError(f"No price data for {t}.")
    m = meta(t)
    return {"ticker": t, "type": kinds.get(t, "ETF" if t in data.etfs() else "Fund"), "meta": m,
            "stats": stats(t, checked=True)}


# ---------------------------------------------------------------- comparison

def compare(tickers: list[str], start=None, end=None) -> dict:
    """2-6 funds over their common period: growth of $10,000 (month-end values), calendar-year returns, the
    correlation matrix of monthly returns and each fund's statistics (correlation.analyze), plus metadata."""
    from . import correlation
    tickers = list(dict.fromkeys(data.canonical(t) for t in tickers if str(t).strip()))
    if not 2 <= len(tickers) <= MAX_COMPARE:
        raise ValueError(f"Compare 2 to {MAX_COMPARE} funds (got {len(tickers)}).")
    missing = [t for t in tickers if t not in set(data.available_tickers())]
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
            "stats": R["stats"], "notes": R["notes"],
            "meta": {t: {k: meta(t).get(k) for k in META_FIELDS} for t in tickers}}
