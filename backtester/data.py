"""Load daily price history and reference data from data/."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRICES = DATA / "prices"
UNIVERSE_FILE = DATA / "universe.json"
MEMBERSHIP_FILE = DATA / "ndx_membership.csv"

BENCHMARKS = ["SPY", "QQQ"]
ALIASES = {"VIX": "^VIX", "NDX100": "^NDX", "SPX": "^GSPC", "GSPC": "^GSPC", "IRX": "^IRX", "TNX": "^TNX",
           "RUT": "^RUT", "DJI": "^DJI", "NASDAQ100": "^NDX"}


class DataError(ValueError):
    pass


def canonical(ticker: str) -> str:
    t = ticker.strip().upper().lstrip("$")
    return ALIASES.get(t, t)


def available_tickers() -> list[str]:
    return sorted(p.stem for p in PRICES.glob("*.csv"))


@lru_cache(maxsize=1)
def universe_meta() -> dict:
    return json.loads(UNIVERSE_FILE.read_text()) if UNIVERSE_FILE.exists() else {}


def nasdaq100() -> list[str]:
    """Today's Nasdaq-100 members."""
    m = universe_meta()
    if m.get("nasdaq100"):
        return list(m["nasdaq100"])
    return [t for t in available_tickers() if t not in BENCHMARKS and not t.startswith("^")]


def etfs() -> list[str]:
    return list(universe_meta().get("etfs", []))


@lru_cache(maxsize=1)
def membership() -> pd.DataFrame | None:
    """Monthly point-in-time Nasdaq-100 membership (rows: month start; columns: tickers; bool)."""
    if not MEMBERSHIP_FILE.exists():
        return None
    raw = pd.read_csv(MEMBERSHIP_FILE, dtype=str)
    if raw.empty:
        return None
    rows = {pd.Period(m, "M").to_timestamp(): set(t.split()) for m, t in zip(raw["month"], raw["tickers"])}
    names = sorted(set().union(*rows.values()))
    df = pd.DataFrame(False, index=sorted(rows), columns=names)
    for d, s in rows.items():
        df.loc[d, list(s)] = True
    return df


def nasdaq100_ever() -> list[str]:
    """Every ticker that has been a Nasdaq-100 member (in the membership history) and has price data."""
    have = set(available_tickers())
    mem = membership()
    names = set(nasdaq100())
    if mem is not None:
        names |= set(mem.columns)
    return sorted(t for t in names if t in have)


def member_mask(tickers: list[str], index: pd.DatetimeIndex) -> tuple[np.ndarray, pd.Timestamp | None]:
    """(T x N) bool: was ticker a Nasdaq-100 member on each date? Before the first snapshot every
    currently-known member is allowed (and the caller is told when point-in-time data begins)."""
    mem = membership()
    cur = set(nasdaq100())
    out = np.zeros((len(index), len(tickers)), bool)
    if mem is None:
        out[:, [j for j, t in enumerate(tickers) if t in cur]] = True
        return out, None
    snap = mem.reindex(columns=tickers, fill_value=False)
    first = snap.index[0]
    # each day uses the latest snapshot at or before it; the current list covers the months after the last snapshot
    aligned = snap.reindex(index.union(snap.index)).ffill().reindex(index).fillna(False).astype(bool).to_numpy()
    out[:] = aligned
    before = index < first
    if before.any():
        # no point-in-time data this early: fall back to the earliest snapshot
        out[before] = snap.iloc[0].to_numpy()
    last = snap.index[-1]
    after = index >= last + pd.offsets.MonthBegin(1)
    if after.any():
        out[after] = np.array([t in cur for t in tickers])
    return out, first


@lru_cache(maxsize=None)
def load(ticker: str) -> pd.DataFrame:
    """Daily bars for `ticker`.

    open/high/low/close/volume are split-adjusted but NOT dividend-adjusted (prices as quoted, like a
    chart). `dividend` is the cash dividend per share on its ex-date; `adj_close` is the
    total-return (dividend-reinvested) close; `tr` is the total-return index (adj_close / first close).
    """
    t = canonical(ticker)
    path = PRICES / f"{t}.csv"
    if not path.exists() and not fetch_on_demand(t):
        raise DataError(f"No price data for {t}. Available: {', '.join(available_tickers())}")
    raw = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw[(raw["close"] > 0) & raw["close"].notna()]
    df = pd.DataFrame(index=raw.index)
    opn = raw["open"].where(raw["open"] > 0, raw["close"]).fillna(raw["close"])
    low = raw["low"].where(raw["low"] > 0)
    df["open"] = opn
    df["high"] = pd.concat([raw["high"], opn, raw["close"]], axis=1).max(axis=1)
    df["low"] = pd.concat([low, opn, raw["close"]], axis=1).min(axis=1)
    df["close"] = raw["close"]
    df["volume"] = raw["volume"].fillna(0) if "volume" in raw else 0.0
    df["dividend"] = raw["dividend"].fillna(0.0) if "dividend" in raw else 0.0
    adj = raw["adj_close"] if "adj_close" in raw else raw["close"]
    df["adj_close"] = adj.where(adj > 0, raw["close"]).ffill()
    df["quote_close"] = df["close"]
    return df


def fetch_on_demand(ticker: str) -> bool:
    """Download a missing ticker's full daily history with yfinance (needs internet; used on your own
    machine - the cloud sandbox relies on the GitHub Action instead). Returns True on success."""
    import os
    if os.environ.get("BACKTESTER_OFFLINE"):
        return False
    try:
        import yfinance as yf
    except ImportError:
        return False
    try:
        df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False, actions=True)
    except Exception:  # noqa: BLE001 - no internet, unknown symbol, rate limit...
        return False
    if df is None or len(df) < 20:
        return False
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close",
                            "Volume": "volume", "Dividends": "dividend", "Stock Splits": "split"})
    cols = [c for c in ("open", "high", "low", "close", "adj_close", "volume", "dividend", "split") if c in df]
    df = df[cols][~df.index.duplicated(keep="last")].dropna(subset=["close"])
    df.index.name = "date"
    PRICES.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(PRICES / f"{ticker}.csv", float_format="%.6g")
    return True


def load_many(tickers: list[str]) -> dict[str, pd.DataFrame]:
    return {canonical(t): load(t) for t in tickers}


def total_return_close(ticker: str) -> pd.Series:
    return load(ticker)["adj_close"]


@lru_cache(maxsize=1)
def tbill_rate() -> pd.Series:
    """Annualised 3-month T-bill yield (decimal), daily, forward-filled. FRED DTB3, else Yahoo ^IRX."""
    f = DATA / "macro" / "DTB3.csv"
    s = None
    if f.exists():
        d = pd.read_csv(f, parse_dates=["date"], index_col="date")["value"]
        s = pd.to_numeric(d, errors="coerce").dropna() / 100.0
    irx = PRICES / "^IRX.csv"
    if irx.exists():
        y = pd.read_csv(irx, parse_dates=["date"], index_col="date")["close"] / 100.0
        s = y if s is None else s.combine_first(y)
    if s is None:
        return pd.Series(dtype=float)
    return s.sort_index()


@lru_cache(maxsize=1)
def cpi() -> pd.Series:
    f = DATA / "macro" / "CPIAUCSL.csv"
    if not f.exists():
        return pd.Series(dtype=float)
    d = pd.read_csv(f, parse_dates=["date"], index_col="date")["value"]
    return pd.to_numeric(d, errors="coerce").dropna()


@lru_cache(maxsize=1)
def factors() -> pd.DataFrame:
    """Daily Fama-French 5 factors + momentum (decimal returns), if downloaded."""
    base = DATA / "factors"
    parts = []
    for fn in ("ff5_daily.csv", "mom_daily.csv"):
        p = base / fn
        if p.exists():
            parts.append(pd.read_csv(p, parse_dates=["date"], index_col="date"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, axis=1)
    df.columns = [c.strip() for c in df.columns]
    return df


@lru_cache(maxsize=None)
def shares_outstanding(ticker: str) -> pd.Series:
    p = DATA / "shares" / f"{canonical(ticker)}.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    return pd.read_csv(p, parse_dates=["date"], index_col="date")["shares"].sort_index()


def data_status() -> dict:
    m = universe_meta()
    mem = membership()
    return {
        "updated_utc": m.get("updated_utc"),
        "tickers": len(available_tickers()),
        "nasdaq100_current": len(nasdaq100()),
        "former_members_with_data": len(m.get("former_members", [])),
        "former_members_missing": len(m.get("former_members_missing_data", [])),
        "membership_from": str(mem.index[0].date()) if mem is not None else None,
        "has_tbill": not tbill_rate().empty,
        "has_cpi": not cpi().empty,
        "has_factors": not factors().empty,
    }
