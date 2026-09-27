"""Custom return or price series imported by the user (Portfolio Visualizer's "import your own returns").

A CSV of `date,value` rows - daily or monthly returns (in % or as decimals) or prices / index levels - is stored as
a named ticker under data/custom/NAME.csv (the price-file format: open = high = low = close = adj_close = the level,
no dividends) plus data/custom/NAME.json (what was imported). data.load reads it like any other ticker, so the name
works anywhere a ticker does: portfolios, benchmarks, Monte Carlo, the optimiser, factor regressions.

Daily series sit on NYSE sessions (a value dated on a non-session day, e.g. a Saturday, counts from the next
session; sessions without a value carry the last level). Monthly series step on the last NYSE session of each month
(the level is flat in between), as the long-history simulated series built from monthly data do, so the reports'
stepped-series handling applies: monthly statistics, and daily ones flagged.

The files are meant to be committed with the repository (data/custom/ is not git-ignored).
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import calendar as _cal
from . import data

NAME_RX = re.compile(r"[A-Z][A-Z0-9]{0,9}")
MAX_DAILY_GAP_DAYS = 40         # a daily series may not skip more than this many calendar days
MAX_DAILY_RETURN = 0.75         # |daily return| above this is not a plausible daily return
MAX_MONTHLY_RETURN = 3.0        # |monthly return| above this (300%) is not a plausible monthly return


def custom_dir() -> Path:
    return data.CUSTOM


def _parse_csv(text: str) -> tuple[pd.DatetimeIndex, list[str]]:
    """Rows of 'date,value' (a header row, blank lines and a thousands separator inside quotes are allowed) ->
    (dates, raw value strings)."""
    rows = []
    for k, line in enumerate(io.StringIO(text.replace("\r\n", "\n").replace("\r", "\n")), 1):
        line = line.strip().lstrip("﻿")
        if not line or line.startswith("#"):
            continue
        parts = [p.strip().strip('"').strip() for p in re.split(r"[,;\t]", line) if p.strip()]
        if len(parts) < 2:
            raise ValueError(f"Line {k}: expected 'date,value', got {line[:60]!r}.")
        if len(parts) > 2:
            # "2020-01-31,1,234.5": a thousands separator without quotes
            parts = [parts[0], "".join(parts[1:])]
        rows.append((k, parts[0], parts[1]))
    if rows and not re.search(r"\d", rows[0][2]) and not re.match(r"\d", rows[0][1]):
        rows = rows[1:]                       # a header row ("date,return")
    if len(rows) < 3:
        raise ValueError("The file needs at least 3 rows of 'date,value'.")
    dates = []
    for k, d, _ in rows:
        try:
            dates.append(pd.Timestamp(d))
        except (ValueError, TypeError):
            raise ValueError(f"Line {k}: {d!r} is not a date (use e.g. 2020-01-31).") from None
    return pd.DatetimeIndex(dates).normalize(), [v for _, _, v in rows]


def _values(raw: list[str]) -> tuple[np.ndarray, bool]:
    """Numbers from the raw strings, and whether any was written with a % sign."""
    pct = any(v.endswith("%") for v in raw)
    out = []
    for v in raw:
        s = v.replace("%", "").replace("$", "").replace(",", "").replace(" ", "")
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        try:
            out.append(float(s))
        except ValueError:
            raise ValueError(f"{v!r} is not a number.") from None
    a = np.asarray(out, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("Every value must be a finite number.")
    return a, pct


def _month_end_sessions(months: pd.PeriodIndex) -> pd.DatetimeIndex:
    """The last NYSE session of each month."""
    out = []
    for m in months:
        d = m.end_time.normalize()
        for _ in range(12):
            if _cal.is_session(d):
                break
            d -= pd.Timedelta(days=1)
        out.append(d)
    return pd.DatetimeIndex(out)


def _sessions(first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    days = pd.bdate_range(first, last)
    return pd.DatetimeIndex([d for d in days if _cal.is_session(d)])


def build(text: str, kind: str = "auto", monthly: bool | None = None, units: str = "auto") -> tuple[pd.Series, dict]:
    """The CSV text -> (a total-return level on NYSE sessions, a description). kind: "returns", "prices" or "auto";
    monthly: None = detect from the spacing of the dates; units (returns): "percent", "decimal" or "auto" (a % sign,
    or any value above 1 in size, means percent)."""
    dates, raw = _parse_csv(text)
    vals, pct_sign = _values(raw)
    if (dates[1:] <= dates[:-1]).any():
        k = int(np.argmax(dates[1:] <= dates[:-1]))
        raise ValueError(f"The dates must be in increasing order without repeats: {dates[k].date()} is followed by "
                         f"{dates[k + 1].date()}.")
    gaps = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    if monthly is None:
        monthly = float(np.median(gaps)) > 20
    if kind not in ("auto", "returns", "prices"):
        raise ValueError("kind must be 'returns', 'prices' or 'auto'")
    if kind == "auto":
        kind = "returns" if (pct_sign or (vals <= 0).any() or np.abs(vals).max() < 0.5) else "prices"
    info: dict = {"kind": kind, "frequency": "monthly" if monthly else "daily", "rows": int(len(vals))}
    if monthly:
        per = dates.to_period("M")
        if (per[1:] == per[:-1]).any():
            k = int(np.argmax(per[1:] == per[:-1]))
            raise ValueError(f"Monthly data has two rows in {per[k]}: one value per month.")
        miss = [str(p) for a, b in zip(per[:-1], per[1:]) if (b - a).n > 1 for p in [a + 1]]
        if miss:
            raise ValueError(f"Monthly data must have every month: {miss[0]} is missing"
                             + (f" (and {len(miss) - 1} more gaps)" if len(miss) > 1 else "") + ".")
    else:
        if (gaps > MAX_DAILY_GAP_DAYS).any():
            k = int(np.argmax(gaps > MAX_DAILY_GAP_DAYS))
            raise ValueError(f"Daily data has a gap of {gaps[k]} days ({dates[k].date()} to {dates[k + 1].date()}); at most "
                             f"{MAX_DAILY_GAP_DAYS} are allowed. For monthly data say so (--monthly).")
    if kind == "returns":
        if units == "auto":
            units = "percent" if (pct_sign or np.abs(vals).max() > 1.0) else "decimal"
        if units not in ("percent", "decimal"):
            raise ValueError("units must be 'percent', 'decimal' or 'auto'")
        r = vals / 100 if units == "percent" else vals
        lim = MAX_MONTHLY_RETURN if monthly else MAX_DAILY_RETURN
        bad = np.flatnonzero((r <= -1) | (np.abs(r) > lim))
        if len(bad):
            k = int(bad[0])
            raise ValueError(f"{dates[k].date()}: a {'monthly' if monthly else 'daily'} return of {r[k]:.2%} is not "
                             f"plausible (returns are read as {units}; above -100% and at most {lim:.0%} in size). "
                             "Check the units (--percent / --decimal) or whether the file holds prices (--prices).")
        info["units"] = units
        level = 100.0 * np.cumprod(1 + r)
        # the first return runs from the period before the first date: that point is the base (100)
        if monthly:
            ends = _month_end_sessions(dates.to_period("M"))
            base = _month_end_sessions(pd.PeriodIndex([dates.to_period("M")[0] - 1]))[0]
        else:
            ends = dates
            base = _cal.anchor_day(dates[0])
        lv = pd.Series(np.r_[100.0, level], index=pd.DatetimeIndex([base]).append(ends))
    else:
        if (vals <= 0).any():
            k = int(np.argmax(vals <= 0))
            raise ValueError(f"{dates[k].date()}: prices must be above 0 (got {vals[k]:g}); for returns use --returns.")
        rr = vals[1:] / vals[:-1] - 1
        lim = MAX_MONTHLY_RETURN if monthly else MAX_DAILY_RETURN
        bad = np.flatnonzero(np.abs(rr) > lim)
        if len(bad):
            k = int(bad[0])
            raise ValueError(f"{dates[k + 1].date()}: the price moves {rr[k]:+.0%} from the row before, not plausible for "
                             f"{'monthly' if monthly else 'daily'} data. Check the file (or use --returns for returns).")
        ends = _month_end_sessions(dates.to_period("M")) if monthly else dates
        lv = pd.Series(vals, index=ends)
    # onto NYSE sessions: a non-session date counts from the next session; days without a value carry the level
    sess = _sessions(lv.index[0], lv.index[-1])
    first = lv.index[0]
    if not _cal.is_session(first):
        sess = sess[sess > first]
    lv = lv.groupby(lv.index).last()
    out = lv.reindex(sess.union(lv.index)).ffill()
    # values dated on non-session days move to the next session
    extra = out.index.difference(sess)
    out = out.drop(extra).reindex(sess).ffill().dropna()
    if len(out) < 3:
        raise ValueError("Too few NYSE sessions in the data.")
    info.update({"first": str(out.index[0].date()), "last": str(out.index[-1].date())})
    return out, info


def import_series(name: str, text: str, kind: str = "auto", monthly: bool | None = None, units: str = "auto",
                  source: str = "", overwrite: bool = True) -> dict:
    """Validate and store a custom series as ticker `name`; returns its description."""
    n = str(name or "").strip().upper()
    if not NAME_RX.fullmatch(n):
        raise ValueError("The name is 1 to 10 capital letters or digits, starting with a letter (e.g. MYFUND).")
    if data.canonical(n) != n:
        raise ValueError(f"{n} is an alias of {data.canonical(n)}; choose another name.")
    if (data.PRICES / f"{n}.csv").exists():
        raise ValueError(f"{n} is already a ticker with market data; choose another name for your series.")
    path = data.CUSTOM / f"{n}.csv"
    if path.exists() and not overwrite:
        raise ValueError(f"A custom series {n} exists already.")
    level, info = build(text, kind, monthly, units)
    df = pd.DataFrame({"open": level, "high": level, "low": level, "close": level, "volume": 0.0, "dividend": 0.0,
                       "adj_close": level}, index=level.index)
    df.index.name = "date"
    data.CUSTOM.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, float_format="%.10g")
    info.update({"name": n, "source": str(source or ""), "imported": datetime.now().isoformat(timespec="seconds"),
                 "sessions": int(len(df))})
    (data.CUSTOM / f"{n}.json").write_text(json.dumps(info, indent=2) + "\n")
    data.clear_caches()
    return info


def delete_series(name: str) -> None:
    n = str(name or "").strip().upper()
    for ext in ("csv", "json"):
        p = data.CUSTOM / f"{n}.{ext}"
        if p.exists():
            p.unlink()
    data.clear_caches()


def list_series() -> list[dict]:
    out = []
    for p in sorted(data.CUSTOM.glob("*.csv")) if data.CUSTOM.exists() else []:
        meta = p.with_suffix(".json")
        try:
            info = json.loads(meta.read_text()) if meta.exists() else {"name": p.stem}
        except (OSError, ValueError):
            info = {"name": p.stem}
        info.setdefault("name", p.stem)
        out.append(info)
    return out


def describe(info: dict) -> str:
    what = ("monthly" if info.get("frequency") == "monthly" else "daily") + " " + (
        f"returns ({info.get('units', '')})" if info.get("kind") == "returns" else "prices")
    return (f"{info.get('name')}: a custom series ({what}"
            + (f", from {info['source']}" if info.get("source") else "") + f"), {info.get('first')} to {info.get('last')}"
            + ("; monthly values step on the last trading day of each month, so its daily statistics are not "
               "meaningful" if info.get("frequency") == "monthly" else "")
            + ". It has no dividends or opening prices (trades fill at the close).")
