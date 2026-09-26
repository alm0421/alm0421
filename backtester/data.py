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


def suggest(ticker: str, n: int = 5) -> list[str]:
    """The `n` available tickers closest in spelling to `ticker` (for "did you mean" messages)."""
    from difflib import SequenceMatcher
    t = canonical(ticker).lstrip("^")

    def score(h: str) -> float:
        b = h.lstrip("^")
        return (SequenceMatcher(None, t, b).ratio() + 0.3 * (sorted(t) == sorted(b))
                + 0.1 * (b[:1] == t[:1]) - 0.05 * abs(len(b) - len(t)))
    ranked = sorted(available_tickers(), key=lambda h: -score(h))
    return [h for h in ranked[:n] if score(h) > 0.3]


def unknown_ticker_message(ticker: str) -> str:
    t = canonical(ticker)
    s = suggest(t)
    return (f"No price data for {t}." + (f" Did you mean {', '.join(s)}?" if s else "")
            + " (The Data page lists every ticker; on your own computer new tickers are downloaded automatically.)")


@lru_cache(maxsize=1)
def universe_meta() -> dict:
    return json.loads(UNIVERSE_FILE.read_text()) if UNIVERSE_FILE.exists() else {}


def nasdaq100() -> list[str]:
    """Today's Nasdaq-100 members."""
    m = universe_meta()
    if m.get("nasdaq100"):
        return list(m["nasdaq100"])
    return [t for t in available_tickers() if t not in BENCHMARKS and not t.startswith("^") and not is_sim(t)]


def etfs() -> list[str]:
    return list(universe_meta().get("etfs", []))


# Long-history simulated series built by scripts/fetch_data.py (build_sims): a total-return index
# (close = adj_close, no dividends, volume 0, open = high = low = close) that uses a model before
# the fund existed and the real fund's total return after.
SIMS = {"SPYSIM": "US stock market (Fama-French market return) spliced into SPY",
        "TLTSIM": "20-year Treasuries priced from FRED yields, spliced into TLT",
        "IEFSIM": "~9-year Treasuries from the 10-year yield, spliced into IEF",
        "SHYSIM": "2-year Treasuries from the 2-year yield, spliced into SHY",
        "BILSIM": "1-month T-bills (Fama-French RF), spliced into BIL"}


def is_sim(ticker: str) -> bool:
    return canonical(ticker).endswith("SIM")


def sims() -> list[str]:
    """The simulated long-history series that exist in data/prices."""
    have = set(available_tickers())
    return [t for t in SIMS if t in have] + sorted(t for t in have if t.endswith("SIM") and t not in SIMS)


# symbols that show up in old revisions of the Wikipedia article but were never index members
NOT_MEMBERS = {"NDX", "QQQ", "QQQQ", "TQQQ", "SQQQ", "QLD", "QID", "PSQ", "ONEQ", "NASDAQ", "ETF", "NQ", "ND"}

# a member's series must look like a large Nasdaq stock: this catches recycled tickers (a small
# company that later took over a former member's symbol) and junk series with no trading
MIN_DOLLAR_VOLUME = 2_000_000.0


@lru_cache(maxsize=None)
def quality(ticker: str) -> pd.Series:
    """Per-bar bool: does the series look like a real, liquid listing on that day?

    Rules (20-bar window): median dollar volume >= $2M and at most half the bars with zero volume.
    (No price floor: prices are split-adjusted, so early AAPL or NVDA trade below $1.)"""
    try:
        df = load(ticker)
    except DataError:
        return pd.Series(dtype=bool)
    dv = (df["close"] * df["volume"]).rolling(20, min_periods=5).median()
    zero = (df["volume"] <= 0).astype(float).rolling(20, min_periods=5).mean()
    ok = (dv >= MIN_DOLLAR_VOLUME) & (zero <= 0.5)
    return ok.fillna(False)


def quality_report(tickers: list[str] | None = None) -> pd.DataFrame:
    """Member-months whose price series fails the quality rules, per ticker."""
    mem = membership()
    if mem is None:
        return pd.DataFrame(columns=["ticker", "member_months", "failed_months"])
    have = set(available_tickers())
    rows = []
    for t in tickers or [c for c in mem.columns if c in have]:
        months = mem.index[mem[t].to_numpy()] if t in mem else []
        if len(months) == 0:
            continue
        q = quality(t)
        if q.empty:
            continue
        per = q.groupby(q.index.to_period("M")).mean()
        failed = [m for m in months if per.get(m.to_period("M"), 0.0) < 0.5]
        if failed:
            rows.append({"ticker": t, "member_months": len(months), "failed_months": len(failed),
                         "first_failed": str(failed[0].date())[:7], "last_failed": str(failed[-1].date())[:7]})
    return pd.DataFrame(rows)


@lru_cache(maxsize=64)
def coverage(start=None, end=None) -> pd.DataFrame:
    """Per year: average number of index members, how many have usable price data, and the share."""
    mem = membership()
    if mem is None:
        return pd.DataFrame()
    have = set(available_tickers())
    m = mem
    if start is not None:
        m = m[m.index >= pd.Timestamp(start).to_period("M").to_timestamp()]
    if end is not None:
        m = m[m.index <= pd.Timestamp(end)]
    rows = []
    for y, g in m.groupby(m.index.year):
        tot = g.sum(axis=1).mean()
        cols = [c for c in g.columns if c in have]
        ok = 0.0
        for d, row in g[cols].iterrows():
            names = [c for c in cols if row[c]]
            ok += sum(1 for c in names if _quality_month(c, d))
        rows.append({"year": int(y), "members": round(float(tot), 1), "with_data": round(ok / len(g), 1),
                     "coverage": ok / len(g) / tot if tot else 0.0})
    return pd.DataFrame(rows)


def _quality_month(t: str, month_start: pd.Timestamp) -> bool:
    q = quality(t)
    if q.empty:
        return False
    sl = q[(q.index >= month_start) & (q.index < month_start + pd.offsets.MonthBegin(1))]
    return bool(len(sl)) and sl.mean() >= 0.5


def coverage_note(start, end) -> str | None:
    c = coverage(start, end)
    if c.empty:
        return None
    tot = (c["members"]).sum()
    got = (c["with_data"]).sum()
    worst = c.loc[c["coverage"].idxmin()]
    return (f"Survivorship: {got / tot:.0%} of index member-months in this period have usable price data "
            f"(lowest {worst['coverage']:.0%} in {int(worst['year'])}). The rest are mostly companies that were acquired "
            f"or went bankrupt; free data sources no longer carry them, so results lean optimistic.")


@lru_cache(maxsize=1)
def membership() -> pd.DataFrame | None:
    """Monthly point-in-time Nasdaq-100 membership (rows: month start; columns: tickers; bool)."""
    if not MEMBERSHIP_FILE.exists():
        return None
    raw = pd.read_csv(MEMBERSHIP_FILE, dtype=str)
    if raw.empty:
        return None
    bad = NOT_MEMBERS | set(etfs())
    rows = {pd.Period(m, "M").to_timestamp(): set(t.split()) - bad for m, t in zip(raw["month"], raw["tickers"])}
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
    tickers = list(tickers)
    if mem is None:
        out[:, [j for j, t in enumerate(tickers) if t in cur]] = True
        return out, None
    snap = mem.reindex(columns=tickers, fill_value=False)
    first = snap.index[0]
    # each day uses the latest snapshot at or before it; the current list covers the months after the last snapshot
    aligned = snap.reindex(index.union(snap.index)).ffill().reindex(index).fillna(False).astype(bool).to_numpy()
    out[:] = aligned
    # never treat a junk or recycled-ticker series as the index member
    for j, t in enumerate(tickers):
        q = quality(t)
        out[:, j] &= q.reindex(index).fillna(False).to_numpy(dtype=bool) if not q.empty else False
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
        raise DataError(unknown_ticker_message(t))
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


def _membership_gaps(mem) -> list[str]:
    if mem is None or len(mem) < 2:
        return []
    d = pd.Series(mem.index)
    g = d.diff().dt.days
    return [f"{d[i - 1].date()} .. {d[i].date()}" for i in range(1, len(d)) if g[i] > 62]


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
        "membership_to": str(mem.index[-1].date()) if mem is not None else None,
        "membership_gaps": _membership_gaps(mem),
        "has_tbill": not tbill_rate().empty,
        "has_cpi": not cpi().empty,
        "has_factors": not factors().empty,
    }
