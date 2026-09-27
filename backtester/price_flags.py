"""Price flags: the days in the price files that nothing explains, recorded so they are never used silently.

The data job keeps adding price files (the whole US listing, in rotating batches), and free price data has errors no
curated list can keep up with. Every price file is therefore scanned (after data.load's repairs) for anomalies:

  - "move": a one-day total return, (close + payout) / previous close, beyond +60% / -60% (MOVE_UP / MOVE_DOWN);
  - "total_return": that total return disagrees with the adjusted close's by more than data.CA_TOLERANCE.

Each anomaly is either explained -
  - "security_break": a bankruptcy re-listing splice (integrity.find_breaks; the engines close positions at it),
  - "whitelist": a curated entry with its evidence (backtester/known_moves.py, data.CA_WHITELIST),
  - "fund_distribution": a mutual fund's misreported distribution (warned about by data.distribution_note),
- or it is FLAGGED: listed in data/price_flags.json, and any backtest that holds the ticker across the day gets a
"Warning: unexplained price moves" note naming it (flag_note, via runner). Missed splits, phantom splits, bad ticks
and inconsistent payouts that the evidence decides are repaired on load (integrity.check, data.reconcile_actions)
and no longer show up as anomalies at all. A universe filter (the Nasdaq-100 and so on) can still select a flagged
ticker: the warning is about the data, not a reason to drop the company.

data/price_flags.json holds, for every price file, a fingerprint of its content (so a refresh that forgets to regenerate
the flags is caught by the tests), the flagged days, and the security breaks found. The data job regenerates the entries
of the files it wrote (write(), after each refresh); `python -m backtester.price_flags [--all] [TICKER ...]` does it by
hand. A backtest reads the file, or scans the ticker itself when its entry is missing or stale (a download on demand).
"""
from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, integrity, known_moves

VERSION = 1
MOVE_UP = 1.6
MOVE_DOWN = 0.4
FILE_NAME = "price_flags.json"

# the core universe, where every anomaly must be explained (tests/test_price_flags.py): today's and past Nasdaq-100
# members, SPY / QQQ, the reference ETFs of the integrity gate and the core ETFs of the data job
CORE_ETFS = set(integrity.REFERENCES) | set(data.BENCHMARKS) | set("""
QQQ SPY DIA IWM MDY VTI VOO QLD TQQQ PSQ SQQQ SSO UPRO SPXL SH SDS SPXU SOXL SOXS TECL
TLT IEF SHY BIL SGOV TMF TMV TBT AGG BND LQD HYG TIP GLD SLV DBC USO UUP
VIXY UVXY SVXY EFA EEM VEA VWO VNQ XLK XLF XLE XLV XLY XLP XLI XLU XLB XLRE XLC SMH SOXX IBB ARKK
""".split())


def flags_path() -> Path:
    """data/price_flags.json next to the price folder in use (a test pointing data.PRICES elsewhere writes there)."""
    return data.PRICES.parent / FILE_NAME


def core_tickers() -> set[str]:
    """Nasdaq-100 members past and present (with their former-listing files) and the core ETFs."""
    out = set(CORE_ETFS) | set(data.nasdaq100())
    mem = data.membership()
    if mem is not None:
        out |= set(mem.columns)
    out |= {alias for alias, _ in data.FORMER_LISTINGS.values()}
    return out


# ------------------------------------------------------------------ fingerprints

_STAMPS: dict = {}


def file_stamp(ticker: str) -> str | None:
    """A fingerprint of the ticker's price file content (None when there is no file): what the flags were computed
    from. Content, not modification time, so a fresh checkout of the same data matches."""
    path = data.price_path(data.canonical(ticker))
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _STAMPS.get(key)
    if hit is None:
        hit = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
        if len(_STAMPS) > 50_000:
            _STAMPS.clear()
        _STAMPS[key] = hit
    return hit


# ------------------------------------------------------------------ the scan

def explain(t: str, day: pd.Timestamp, kinds: list[str], breaks: set) -> tuple[str | None, str]:
    """(explanation code, why) for an anomaly of ticker t on `day`; (None, "") when nothing explains it."""
    ds = str(pd.Timestamp(day).date())
    if pd.Timestamp(day) in breaks:
        return "security_break", "a new security spliced into the series (positions are closed at the old last price)"
    if t in known_moves.WHITELIST_TICKERS:
        return "whitelist", known_moves.WHITELIST_TICKERS[t]
    if (t, ds) in known_moves.WHITELIST:
        return "whitelist", known_moves.WHITELIST[(t, ds)]
    if t in known_moves.WHITELIST_BEFORE and ds < known_moves.WHITELIST_BEFORE[t]:
        return "whitelist", f"quotes before {known_moves.WHITELIST_BEFORE[t]} (tick-era / thin-trading series)"
    if (t, ds) in data.CA_WHITELIST:
        return "whitelist", data.CA_WHITELIST[(t, ds)]
    if kinds == ["total_return"]:
        from .fund_lists import MUTUAL_FUNDS
        if t in MUTUAL_FUNDS:
            return "fund_distribution", "a mutual fund's distribution misreported by the source (data.distribution_note)"
    return None, ""


def anomalies(ticker: str, df: pd.DataFrame | None = None) -> list[dict]:
    """Every anomaly of the ticker's loaded bars, each with its explanation ("explained_by" None: flagged):
    {"date", "kinds", "move", "gap", "explained_by", "why"}."""
    t = data.canonical(ticker)
    if t.startswith("^") or data.is_sim(t):
        return []
    if df is None:
        df = data.load(t)
    if len(df) < 2:
        return []
    c, d = df["close"].astype(float), df["dividend"].astype(float)
    a = df["adj_close"].astype(float) if "adj_close" in df else c
    tr = (c + d) / c.shift(1)
    gap = (tr - a / a.shift(1)).abs()
    huge = ((tr > MOVE_UP) | (tr < MOVE_DOWN)).fillna(False).to_numpy()
    off = (gap > data.CA_TOLERANCE).fillna(False).to_numpy()
    rows = np.flatnonzero(huge | off)
    if not len(rows):
        return []
    breaks = set(data.security_breaks(t, df))
    out = []
    for i in rows.tolist():
        day = df.index[i]
        kinds = (["move"] if huge[i] else []) + (["total_return"] if off[i] else [])
        code, why = explain(t, day, kinds, breaks)
        out.append({"date": day, "kinds": kinds, "move": float(tr.iloc[i]) - 1, "gap": float(gap.iloc[i]),
                    "explained_by": code, "why": why})
    return out


def _flag_row(a: dict) -> dict:
    return {"date": str(pd.Timestamp(a["date"]).date()), "kinds": a["kinds"], "move": round(a["move"], 4),
            "gap": round(a["gap"], 4)}


def scan(ticker: str) -> dict:
    """The flags-file entry of one ticker: {"stamp", "flags": [...], "breaks": [...]} (empty lists left out)."""
    t = data.canonical(ticker)
    entry: dict = {"stamp": file_stamp(t)}
    try:
        df = data.load(t)
    except Exception as e:  # noqa: BLE001 - an unreadable file is itself worth a flag
        entry["error"] = str(e)[:200]
        return entry
    flags = [_flag_row(a) for a in anomalies(t, df) if a["explained_by"] is None]
    if flags:
        entry["flags"] = flags
    br = data.break_events(t, df)
    if len(br):
        entry["breaks"] = [{"date": str(pd.Timestamp(r["date"]).date()), "source": r["source"], "why": r["why"]}
                           for _, r in br.iterrows()]
    return entry


def _price_files() -> list[str]:
    return sorted(p.stem for p in data.PRICES.glob("*.csv") if not p.name.startswith("."))


def read(path: Path | None = None) -> dict:
    path = path or flags_path()
    try:
        st = path.stat()
    except OSError:
        return {}
    return _read(str(path), st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=4)
def _read(path: str, _mt, _size) -> dict:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def stale(doc: dict | None = None) -> tuple[list[str], list[str]]:
    """(price files whose entry is missing, from another code version or computed from other content; entries whose
    price file is gone)."""
    doc = read() if doc is None else doc
    files = doc.get("files") or {}
    names = _price_files()
    if doc.get("version") != VERSION:
        return names, sorted(set(files) - set(names))
    out = [t for t in names if (files.get(t) or {}).get("stamp") != file_stamp(t)]
    return out, sorted(set(files) - set(names))


def write(tickers: list[str] | None = None, everything: bool = False, path: Path | None = None,
          quiet: bool = False) -> dict:
    """Regenerate data/price_flags.json: the entries of `tickers`, else (everything=False) of every price file whose
    entry is stale (see stale), else of every price file; entries of deleted files are dropped."""
    path = path or flags_path()
    doc = read(path) if not everything else {}
    if doc.get("version") != VERSION:
        doc = {}
    files = dict(doc.get("files") or {})
    if tickers is not None:
        todo = [data.canonical(t) for t in tickers]
    elif everything or not files:
        todo = _price_files()
    else:
        todo, _ = stale({**doc, "files": files})
    have = set(_price_files())
    for gone in set(files) - have:
        files.pop(gone, None)
    for n, t in enumerate(todo):
        if t not in have:
            files.pop(t, None)
            continue
        files[t] = scan(t)
        if not quiet and n and n % 500 == 0:
            print(f"price flags: {n}/{len(todo)}", file=sys.stderr)
    out = {"_about": ("Days in the price files that no repair, corporate action, security break or curated entry "
                      "explains (backtester/price_flags.py): 'move' = one-day total return beyond +60%/-60%, "
                      "'total_return' = (close + payout) / previous close differs from the adjusted close's return by "
                      "more than 2%. Backtests holding a ticker across a flagged day are warned. 'stamp' fingerprints "
                      "the price file the entry was computed from; 'breaks' are the security breaks found (positions "
                      "are closed at the old security's last price). Regenerated by the data job for the files it "
                      "writes, or by hand: python -m backtester.price_flags [--all] [TICKER ...]"),
           "version": VERSION, "files": dict(sorted(files.items()))}
    # one line per price file, so a refresh's diff shows exactly the files it touched
    lines = [f"{json.dumps(t)}:{json.dumps(e, separators=(',', ':'), default=str)}" for t, e in out["files"].items()]
    path.write_text("{\n" + f'"_about":{json.dumps(out["_about"])},\n"version":{VERSION},\n"files":{{\n'
                    + ",\n".join(lines) + "\n}}\n")
    _read.cache_clear()
    if not quiet:
        nf = sum(len(e.get("flags", [])) for e in files.values())
        print(f"price flags: {nf} flagged days in {sum(1 for e in files.values() if e.get('flags'))} of {len(files)} "
              f"files, {sum(len(e.get('breaks', [])) for e in files.values())} security breaks ({path.name}; "
              f"{len(todo)} rescanned)", file=sys.stderr)
    return out


@lru_cache(maxsize=1)
def all_anomalies(_key: str = "") -> dict[str, list[dict]]:
    """{ticker: anomalies} of every price file (one pass, remembered: both data tests read it)."""
    out = {}
    for t in data.available_tickers():
        if t.startswith("^"):
            continue
        try:
            out[t] = anomalies(t)
        except data.DataError:
            continue
    return out


def unrecorded(kind: str | None = None) -> tuple[list[tuple], list[tuple]]:
    """The scalable data check: (anomalies that are neither explained nor in data/price_flags.json, unexplained
    anomalies in the core universe - which must be explained even when flagged), as (ticker, date, kinds, move, gap)
    rows, of one kind ("move" / "total_return") or all."""
    doc = read()
    files = (doc.get("files") or {}) if doc.get("version") == VERSION else {}
    core = core_tickers()
    missing, core_bad = [], []
    for t, rows in all_anomalies(str(data.PRICES)).items():
        recorded = {(f["date"], k) for f in (files.get(t) or {}).get("flags", []) for k in f["kinds"]}
        for a in rows:
            if a["explained_by"] is not None:
                continue
            ks = [k for k in a["kinds"] if kind is None or k == kind]
            if not ks:
                continue
            row = (t, str(pd.Timestamp(a["date"]).date()), ks, round(a["move"], 4), round(a["gap"], 4))
            if t in core:
                core_bad.append(row)
            if any((row[1], k) not in recorded for k in ks):
                missing.append(row)
    return missing, core_bad


# ------------------------------------------------------------------ the warning

def flags(ticker: str) -> list[dict]:
    """The flagged days of `ticker` ({"date": Timestamp, "kinds", "move", "gap"}): from data/price_flags.json when its
    entry matches the file, else scanned now (a file downloaded on demand, or a stale entry)."""
    t = data.canonical(ticker)
    if t.startswith("^") or data.is_sim(t) or data.is_custom(t):
        return []
    stamp = file_stamp(t)
    if stamp is None:
        return []                  # no price file (a cash column, a synthetic series): nothing to scan
    doc = read()
    entry = (doc.get("files") or {}).get(t) if doc.get("version") == VERSION else None
    if entry is None or entry.get("stamp") != stamp:
        entry = _live(t, stamp)
    return [{**f, "date": pd.Timestamp(f["date"])} for f in entry.get("flags", [])]


@lru_cache(maxsize=512)
def _live(t: str, _stamp) -> dict:
    return scan(t)


def flag_note(holdings: pd.DataFrame, max_named: int = 6) -> str | None:
    """'Warning: unexplained price moves ...' naming each flagged day a backtest held a ticker across: the ticker is
    held at the close before the day (holdings: date x ticker, non-zero = held)."""
    if holdings is None or holdings.empty:
        return None
    idx = holdings.index
    found = []
    for t in holdings.columns:
        col = holdings[t].to_numpy()
        if not (col != 0).any():
            continue
        try:
            fl = flags(t)
        except Exception:  # noqa: BLE001 - a note must never break a backtest
            continue
        for f in fl:
            k = int(idx.searchsorted(f["date"])) - 1      # the last close before the day
            if k < 0 or f["date"] > idx[-1] or col[k] == 0 or not np.isfinite(col[k]):
                continue
            what = []
            if "move" in f["kinds"]:
                what.append(f"{f['move']:+.0%} in one day")
            if "total_return" in f["kinds"]:
                what.append(f"its total return is {f['gap']:.0%} off the adjusted close's")
            found.append(f"{t} {f['date'].date()} ({', '.join(what)})")
    if not found:
        return None
    more = f" and {len(found) - max_named} more" if len(found) > max_named else ""
    return ("Warning: unexplained price moves: the run held " + "; ".join(found[:max_named]) + more + ". No split, "
            "corporate action or known event explains these days in the source data (data/price_flags.json), so "
            "they may be data errors; returns across them are as reported.")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    everything = "--all" in argv
    names = [a for a in argv if not a.startswith("--")]
    write(names or None, everything=everything)


if __name__ == "__main__":
    main()
