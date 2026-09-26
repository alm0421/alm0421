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
            + f" To add {t}, put it on its own line in data/extra_tickers.txt and run the 'Fetch price data' workflow "
            "(GitHub Actions; pushing the file also starts it), then pull the new data. (The Data page lists every "
            "ticker; on your own computer new tickers are downloaded automatically.)")


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
SIMS = {"SPYSIM": "US stock market (Fama-French market return) from 1926, spliced into SPY",
        "TLTSIM": "20-year Treasuries priced from FRED yields from 1962, spliced into TLT",
        "IEFSIM": "~9-year Treasuries from the 10-year yield from 1962, spliced into IEF",
        "SHYSIM": "2-year Treasuries from the 2-year yield from 1962, spliced into SHY",
        "BILSIM": "1-month T-bills (Fama-French RF) from 1926, spliced into BIL",
        "IEISIM": "5-year Treasuries from the 5-year yield from 1962, spliced into IEI",
        "BNDSIM": "US aggregate bonds: 70% 5-year Treasury / 30% IG corporate model from 1962, the Vanguard Total Bond "
                  "Market Index fund (VBMFX) from Dec 1986, then BND",
        "LQDSIM": "Investment-grade corporates priced off Moody's Aaa/Baa yields from 1953, spliced into LQD",
        "HYGSIM": "US high-yield bonds: the Vanguard High-Yield Corporate fund (VWEHX) from 1985, then HYG (no model before)",
        "TIPSIM": "US TIPS: the Vanguard Inflation-Protected Securities fund (VIPSX) from mid-2000, then TIP (no model before)",
        "BNDXSIM": "International government bonds hedged to USD: a par-bond model on OECD 10-year yields of up to 12 "
                   "developed markets (monthly steps) from 1970, PIMCO International Bond USD-hedged (PFORX) from 1993, then BNDX",
        "VBSIM": "US small caps (Fama-French small portfolios) from 1926, spliced into VB",
        "VBRSIM": "US small-cap value (Fama-French small/high B/M) from 1926, spliced into VBR",
        "VBKSIM": "US small-cap growth (Fama-French small/low B/M) from 1926, spliced into VBK",
        "MIDSIM": "US mid caps (Fama-French 30th-70th NYSE size percentiles) from 1926, spliced into MDY (S&P 400)",
        "VTVSIM": "US large-cap value (Fama-French big/high B/M) from 1926, spliced into VTV",
        "VUGSIM": "US large-cap growth (Fama-French big/low B/M) from 1926, spliced into VUG",
        "EFASIM": "Developed ex-US stocks: Fama-French EAFE index (monthly steps) from 1975, daily from 1990, spliced into EFA",
        "EFVSIM": "Developed ex-US value: Fama-French EAFE high-B/M index (monthly steps) from 1975, daily big/high B/M "
                  "from 1990, spliced into EFV",
        "SCZSIM": "Developed ex-US small caps (Fama-French, daily) from 1990, spliced into SCZ",
        "AVDVSIM": "Developed ex-US small-cap value (Fama-French small/high B/M, daily) from 1990, spliced into AVDV",
        "VGKSIM": "European stocks: Fama-French Europe index (monthly steps) from 1975, daily from 1990, spliced into VGK",
        "EEMSIM": "Emerging markets (Fama-French, monthly steps) from 1989, spliced into EEM",
        "VNQSIM": "US REITs: FTSE Nareit All Equity REITs total return (monthly steps) from 1972, spliced into VNQ",
        "GLDSIM": "Gold: LBMA PM fixing (daily) from April 1968 (World Bank monthly average price 1960-68), spliced into GLD",
        "DBCSIM": "Commodity futures: AQR equal-weight commodity index excess return + T-bills (monthly steps) from 1960, "
                  "spliced into DBC",
        }


def is_sim(ticker: str) -> bool:
    return canonical(ticker).endswith("SIM")


def sims() -> list[str]:
    """The simulated long-history series that exist in data/prices."""
    have = set(available_tickers())
    return [t for t in SIMS if t in have] + sorted(t for t in have if t.endswith("SIM") and t not in SIMS)


# symbols that show up in old revisions of the Wikipedia article but were never index members
LARGE_STOCKS = set("""JPM XOM BRK-B JNJ UNH V MA HD PG CVX LLY ABBV MRK KO BAC WFC DIS MCD NKE ORCL CRM IBM GE CAT
BA GS MS C T VZ PFE TMO DHR ABT NEE DUK SO LMT RTX UPS UNP MMM MSTR COIN""".split())
NOT_MEMBERS = {"NDX", "QQQ", "QQQQ", "TQQQ", "SQQQ", "QLD", "QID", "PSQ", "ONEQ", "NASDAQ", "ETF", "NQ", "ND",
               "NXP"}  # NXP is a Nuveen municipal fund (a typo for NXPI in one stretch of revisions)

# Symbols reused by a different company. The price file holds the current company, so it only stands
# for the index member from this date on (earlier member-months count as missing data).
IDENTITY_FROM = {
    "MNST": "2012-01-09",   # Monster Worldwide until 2011; the file is Monster Beverage (ex-HANS)
}
# the same company listed under two symbols in some revisions: keep the second
DUPLICATES = {"KLA": "KLAC", "WFMI": "WFM", "ERTS": "EA"}

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
    dv = (df["close"] * df["volume"]).rolling(20, min_periods=5).median().shift(1)
    zero = (df["volume"] <= 0).astype(float).rolling(20, min_periods=5).mean().shift(1)
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


def corporate_action_days(ticker: str, threshold: float = 0.15) -> pd.DatetimeIndex:
    """Days where the quoted price jumps by more than `threshold` and the total-return series does not
    agree with price + cash dividend - usually a spin-off, special dividend or split that the free data
    did not fully adjust (e.g. EXPE on 2011-12-21, the TripAdvisor spin-off)."""
    try:
        df = load(ticker)
    except Exception:  # noqa: BLE001 - unknown or synthetic ticker
        return pd.DatetimeIndex([])
    if not {"close", "dividend", "adj_close"} <= set(df.columns):
        return pd.DatetimeIndex([])
    c, d, a = df["close"], df["dividend"], df["adj_close"]
    price_tr = (c + d) / c.shift(1) - 1
    adj_tr = a / a.shift(1) - 1
    big = price_tr.abs() > threshold
    mismatch = (price_tr - adj_tr).abs() > 0.05
    return df.index[(big & mismatch).fillna(False).to_numpy()]


def coverage_note(start, end) -> str | None:
    c = coverage(start, end)
    if c.empty:
        return None
    tot = (c["members"]).sum()
    got = (c["with_data"]).sum()
    worst = c.loc[c["coverage"].idxmin()]
    note = (f"Survivorship: {got / tot:.0%} of index member-months in this period have usable price data "
            f"(lowest {worst['coverage']:.0%} in {int(worst['year'])}). The rest are mostly companies that were acquired "
            f"or went bankrupt; free data sources no longer carry them, so the former members that are included "
            f"are almost all companies still trading today and results lean optimistic.")
    gaps = [g for g in _membership_gaps(membership())
            if pd.Timestamp(g.split(" .. ")[1]) >= pd.Timestamp(start) and pd.Timestamp(g.split(" .. ")[0]) <= pd.Timestamp(end)]
    if gaps:
        note += (" Membership snapshots are missing for " + ", ".join(gaps)
                 + "; the last known list is carried forward across those gaps.")
    return note


@lru_cache(maxsize=1)
def membership() -> pd.DataFrame | None:
    """Monthly point-in-time Nasdaq-100 membership (rows: month start; columns: tickers; bool)."""
    if not MEMBERSHIP_FILE.exists():
        return None
    raw = pd.read_csv(MEMBERSHIP_FILE, dtype=str)
    if raw.empty:
        return None
    # strip funds, never stocks (older universe files listed some large stocks among the ETFs)
    stocks = set(universe_meta().get("stocks", [])) | LARGE_STOCKS
    bad = NOT_MEMBERS | (set(etfs()) - stocks)
    rows = {pd.Period(m, "M").to_timestamp(): set(t.split()) - bad for m, t in zip(raw["month"], raw["tickers"])}
    for d, syms in rows.items():
        for old, new in DUPLICATES.items():
            if old in syms and new in syms:
                syms.discard(old)
    names = sorted(set().union(*rows.values()))
    df = pd.DataFrame(False, index=sorted(rows), columns=names)
    for d, syms in rows.items():
        df.loc[d, list(syms)] = True
    # a name missing from one snapshot but present before and after is a transcription slip, not a
    # removal and re-addition
    v = df.to_numpy(copy=True)
    blip = ~v[1:-1] & v[:-2] & v[2:]
    v[1:-1] |= blip
    return pd.DataFrame(v, index=df.index, columns=df.columns)


def nasdaq100_ever() -> list[str]:
    """Every ticker that has been a Nasdaq-100 member (in the membership history) and has price data."""
    have = set(available_tickers())
    mem = membership()
    names = set(nasdaq100())
    if mem is not None:
        names |= set(mem.columns)
    return sorted(t for t in names if t in have)


def member_mask(tickers: list[str], index: pd.DatetimeIndex) -> tuple[np.ndarray, pd.Timestamp | None]:
    """(T x N) bool: was ticker a Nasdaq-100 member on each date?

    Only dates covered by point-in-time snapshots can have members: before the first snapshot nobody
    is treated as a member (using a later list there would be pure survivorship bias). The quality
    and company-identity checks apply on every date."""
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
    out[index < first] = False
    last = snap.index[-1]
    after = index >= last + pd.offsets.MonthBegin(1)
    if after.any():
        out[after] = np.array([t in cur for t in tickers])
    # never treat a junk or recycled-ticker series as the index member
    for j, t in enumerate(tickers):
        if t in IDENTITY_FROM:
            out[:, j] &= np.asarray(index >= pd.Timestamp(IDENTITY_FROM[t]))
        q = quality(t)
        out[:, j] &= q.reindex(index).fillna(False).to_numpy(dtype=bool) if not q.empty else False
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
    if t.endswith("SIM") and len(df):
        # simulated series are built from sources with other calendars (Fama-French, World Bank,
        # FRED): keep only NYSE sessions so month-ends line up with every real ticker. Levels are
        # total-return indexes, so a dropped day's return simply rolls into the next session.
        from . import calendar as _cal
        # (the modern holiday rules only from 1972: earlier NYSE calendars differed - Saturday sessions,
        # fixed-date holidays - so older rows are kept as they are)
        keep = np.array([d.year < 1972 or _cal.is_session(d) for d in df.index])
        df = df[keep]
    # opening prices that were never quoted: missing, or a flat bar (open = high = low = close), as
    # for mutual funds, simulated series and very old index data. Such an "open" is really the close.
    flat = (raw["open"] == raw["close"]) & (raw["high"] == raw["low"]) & (raw["high"] == raw["close"])
    df["open_ok"] = (raw["open"] > 0).fillna(False) & ~flat.fillna(False)
    # old records often carry the previous close as the "open": where that happens on most days of a
    # rolling quarter, the open was never really quoted
    stale = (raw["open"] - raw["close"].shift(1)).abs() <= 1e-9 * raw["close"].abs().clip(lower=1)
    # causal: judged on the quarter up to the day before (so it never depends on later bars)
    df["open_ok"] &= ~(stale.astype(float).rolling(63, min_periods=20).mean().shift(1).fillna(0) > 0.5)
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
    """US CPI, indexed by the date each month's figure was published (about the 15th of the following
    month), so inflation-indexed cash flows and real returns only use CPI that was known at the time.
    Seasonally adjusted (CPIAUCSL) from 1947, not-seasonally-adjusted CPIAUCNS back to 1913 before that."""
    s = cpi_monthly()
    if s.empty:
        return s
    s = s.copy()
    s.index = s.index + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
    return s


@lru_cache(maxsize=1)
def cpi_monthly() -> pd.Series:
    """US CPI by the month it measures (dated the 1st of that month, not lagged): for reporting inflation over
    calendar periods (December to December), not for decisions. Same splice as cpi()."""
    parts = []
    for sid in ("CPIAUCSL", "CPIAUCNS"):
        f = DATA / "macro" / f"{sid}.csv"
        if f.exists():
            d = pd.read_csv(f, parse_dates=["date"], index_col="date")["value"]
            parts.append(pd.to_numeric(d, errors="coerce").dropna())
    if not parts:
        return pd.Series(dtype=float)
    s = parts[0]
    if len(parts) > 1 and len(parts[1]) and parts[1].index[0] < s.index[0]:
        older = parts[1][parts[1].index < s.index[0]]
        if len(older):
            # splice: scale the older series to meet the newer one at the join
            join = parts[1].reindex([s.index[0]]).iloc[0] if s.index[0] in parts[1].index else older.iloc[-1]
            s = pd.concat([older * (s.iloc[0] / join), s])
    return s


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
