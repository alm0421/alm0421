"""Mutual-fund histories whose free (Yahoo) data misses distributions: repaired from published returns, or cut.

Yahoo's daily history of many mutual funds is wrong in its early years: capital-gain distributions (and some income
distributions) are missing, so the NAV drops on the ex-date and nothing in the dividend column or the adjusted close
makes up for it. The fund's total return is understated, often by 5-20% a year: Vanguard 500 Index (VFINX) returns
22.6% in 1985 in Yahoo's data against 31.2% published, Fidelity Magellan (FMAGX) -14.7% in the second quarter of
1984 against -4.6%. How long this lasts depends on the fund family: Vanguard, Fidelity and Dodge & Cox histories are right from
1987 on, American Funds only from 1996, some small funds miss distributions into the 2000s. A backtest over those
years would silently understate the fund by several percent a year.

Two remedies, in this order:

1. Repair from published returns. data/fund_returns.json holds each fund's published quarterly total returns
   (Morningstar's, as Yahoo Finance's performance page shows them: NAV-based, distributions reinvested, net of fees,
   before sales loads), with their source and retrieval date. Every quarter where the daily series' total return
   differs from the published one by more than QUARTER_TOL (log) is matched to it: the missing growth goes on the
   day the price shows the distribution (the quarter's largest unexplained move against the US market, when that
   move accounts for at least PLACE_SHARE of the gap - for a gap under PLACE_UNPAID only on a day the data shows a
   distribution paid) and is booked as a distribution in the dividend column, so price + distributions and the
   adjusted close agree; otherwise it is spread evenly over the quarter's sessions (adjusted close only). A
   distribution whose NAV drop the data books in one quarter and its reinvestment in the next shows as two opposite
   gaps: both quarters are matched (the growth goes on the ex-date and the late reinvestment is taken back). A fund
   with published returns is used over its whole history.

2. Otherwise, detect and cut. For a mutual fund with no published returns, missed distributions are found from the
   daily data itself: days (in two-day windows, which absorb NAV dates that are off by one) where the fund falls by
   more than max(DETECT_K robust deviations, DETECT_MIN) beyond what its rolling betas to US stocks and Treasuries
   (SPYSIM, IEFSIM) explain, the next DETECT_REVERSAL sessions do not give it back, and the market itself did not
   move by more than STRESS_MOVE around it. A missed distribution recurs at the same time of year, so a day only
   counts when such days fall within DETECT_SEASON_DAYS of its date in at least DETECT_SEASON_YEARS years. The
   unreliable period is the first cluster of those days (no gap of more than DETECT_GAP_YEARS between them), starting
   within DETECT_START_YEARS of the fund's first price and before DETECT_UNTIL, when it holds at least
   DETECT_MIN_DAYS days worth DETECT_MIN_SUM in all. (Checked against the funds with published returns: it finds most
   of the funds that need a repair and flags few that do not.) The fund's usable history then starts on the session
   after the cluster's last day; the backtest says so in a Warning and how to opt in to the raw history ("using raw
   fund history", or "raw_fund_history": true in a JSON spec, or --raw-fund-history).

data.load applies both (its cache keys on this module's code and on data/fund_returns.json). The data job fills
data/fund_returns.json from Yahoo's quoteSummary fundPerformance module (scripts/fetch_data.py fetch_fund_returns);
the entries collected by hand from Yahoo's performance pages carry that page as their source.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

FILE_NAME = "fund_returns.json"
QUARTER_TOL = 0.0015         # a quarter whose log total return is off the published one by more than this is repaired
#                              (where the free data is right the two agree to about 0.01%)
PLACE_SHARE = 0.5            # the day's unexplained move must account for this share of the gap to carry it
PLACE_UNPAID = 0.01          # a gap below this is booked on a day only when the data shows a distribution paid that day
DETECT_K = 3.0               # detection: robust deviations of the market-adjusted two-day return ...
DETECT_MIN = 0.006           # ... and at least this drop
DETECT_REVERSAL = 10         # sessions after the drop that must not give back more than half of it
DETECT_GAP_YEARS = 4.0       # flagged days further apart than this end the cluster
DETECT_MIN_DAYS = 3          # a cluster needs this many flagged days ...
DETECT_MIN_SUM = 0.05        # ... worth this much (sum of the drops)
DETECT_START_YEARS = 5.0     # ... and starting within this many years of the fund's first price (an early-data problem)
DETECT_SEASON_DAYS = 21      # a missed distribution recurs: a flagged day counts only when other years have one within
DETECT_SEASON_YEARS = 4      # this many calendar days of the same date, in at least this many different years in all
STRESS_MOVE = 0.03           # sessions within STRESS_WINDOW of a two-day US market move beyond this are never flagged
STRESS_WINDOW = 3            # (crashes hit funds unevenly: 1987-10-19, 1997-10-27, 1998-08-31 are not distributions)
DETECT_UNTIL = pd.Timestamp("2000-01-01")   # the free data's missing distributions are an early-history problem
MUTUAL_FUND_RE = re.compile(r"[A-Z]{4}X")   # Nasdaq symbology: a fifth letter X is a mutual fund
EVENT_COLUMNS = ["date", "quarter", "kind", "yahoo", "published", "factor", "placed"]


def path() -> Path:
    from . import data
    return data.DATA / FILE_NAME


def is_mutual_fund(ticker: str) -> bool:
    """A US mutual fund by its symbol (five letters ending in X), not a SIM or index."""
    return bool(MUTUAL_FUND_RE.fullmatch(ticker or ""))


@lru_cache(maxsize=4)
def _doc(p: str, _stamp) -> dict:
    try:
        d = json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def doc() -> dict:
    p = path()
    try:
        st = p.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    return _doc(str(p), stamp)


def published(ticker: str) -> pd.Series | None:
    """The fund's published quarterly total returns (fractions, PeriodIndex 'Q'), or None."""
    e = (doc().get("funds") or {}).get(ticker)
    if not isinstance(e, dict):
        return None
    out = {}
    for y, row in (e.get("quarterly") or {}).items():
        if not isinstance(row, (list, tuple)):
            continue
        for i, v in enumerate(row[:4]):
            if v is None:
                continue
            try:
                out[pd.Period(f"{int(y)}Q{i + 1}", "Q")] = float(v) / 100.0
            except (TypeError, ValueError):
                continue
    if not out:
        return None
    return pd.Series(out).sort_index()


def source(ticker: str) -> str:
    e = (doc().get("funds") or {}).get(ticker) or {}
    return str(e.get("source") or "published returns")


# ------------------------------------------------------------------ comparison with the published returns

def quarter_returns(adj: pd.Series) -> pd.Series:
    """Total return of each whole calendar quarter of a daily total-return level: last level of the quarter over the
    last level of the one before (so a quarter needs the previous quarter's end in the data)."""
    adj = pd.to_numeric(adj, errors="coerce").dropna()
    adj = adj[adj > 0]
    if adj.empty:
        return pd.Series(dtype=float)
    per = pd.DatetimeIndex(adj.index).to_period("Q")
    last = adj.groupby(per).last()
    q = last / last.shift(1) - 1
    return q.iloc[1:].dropna()


def quarter_gaps(adj: pd.Series, pub: pd.Series) -> pd.DataFrame:
    """Per quarter covered by both: the series' return, the published one and the log gap log((1+pub)/(1+series))."""
    q = quarter_returns(adj)
    both = pd.concat({"yahoo": q, "published": pub}, axis=1).dropna()
    both["gap"] = np.log1p(both["published"]) - np.log1p(both["yahoo"])
    return both


def _quarters_to_fix(g: pd.DataFrame, tol: float = QUARTER_TOL) -> list:
    """The quarters whose gap exceeds tol. Neighbours whose gaps cancel (the free data books a distribution's NAV drop
    in one quarter and its reinvestment in the next, e.g. DFSCX 2001Q2/Q3 by 3.8%) are repaired too: each quarter then
    has its published return, the missing growth on the ex-date and the late reinvestment taken back."""
    return [q for q in g.index if abs(g.at[q, "gap"]) > tol]


def _market_log_returns() -> pd.Series:
    from . import data
    r = data._ref_returns("SPYSIM")
    return r if r is not None else pd.Series(dtype=float)


def _residuals(r_log: pd.Series, mkt: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """The fund's log returns less beta x the market's on `days`, beta from the surrounding half year."""
    lo, hi = days[0] - pd.Timedelta(days=95), days[-1] + pd.Timedelta(days=95)
    f = r_log.loc[lo:hi]
    m = mkt.reindex(f.index).fillna(0.0)
    beta = 1.0
    if len(f) > 30 and float(m.var()) > 0:
        beta = float(np.clip(np.cov(f.to_numpy(), m.to_numpy())[0, 1] / m.var(), -0.5, 2.0))
    return (f - beta * m).reindex(days)


def repair(t: str, raw: pd.DataFrame, pub: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each quarter of the bars to the fund's published quarterly return (see the module docstring).
    (repaired bars, events)."""
    pub = published(t) if pub is None else pub
    empty = pd.DataFrame(columns=EVENT_COLUMNS)
    if pub is None or raw.empty or "adj_close" not in raw:
        return raw, empty
    adj = pd.to_numeric(raw["adj_close"], errors="coerce")
    adj = adj.where(adj > 0).ffill()
    if adj.isna().all():
        return raw, empty
    g = quarter_gaps(adj, pub)
    fix = _quarters_to_fix(g)
    if not fix:
        return raw, empty
    out = raw.copy()
    ratio = np.array((adj / adj.shift(1)).to_numpy(dtype=float), copy=True)
    ratio[0] = 1.0
    r_log = np.log(pd.Series(ratio, index=adj.index))
    mkt = _market_log_returns()
    per = pd.DatetimeIndex(adj.index).to_period("Q")
    close = np.array(pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float), copy=True)
    div = (np.array(pd.to_numeric(out["dividend"], errors="coerce").fillna(0.0).to_numpy(dtype=float), copy=True)
           if "dividend" in out else np.zeros(len(out)))
    events = []
    for q in fix:
        pos = np.nonzero(per == q)[0]
        pos = pos[pos > 0]
        if not len(pos):
            continue
        gap = float(g.at[q, "gap"])
        days = adj.index[pos]
        e = _residuals(r_log, mkt, days)
        k = None
        ev = e.to_numpy(dtype=float) * (1.0 if gap > 0 else -1.0)      # the move the gap should explain, as a drop
        shows = np.isfinite(ev) & (ev <= -PLACE_SHARE * abs(gap))
        if shows.any():
            # prefer an ex-date the data does book (a distribution paid that day, often only its income part)
            paid_days = shows & (div[pos] > 0)
            pick = paid_days if paid_days.any() else (shows if abs(gap) >= PLACE_UNPAID else None)
            if pick is not None:
                j = int(np.nanargmin(np.where(pick, ev, np.nan)))
                k = int(pos[j])
        if k is not None:
            ratio[k] *= float(np.exp(gap))
            if np.isfinite(close[k]) and np.isfinite(close[k - 1]) and close[k - 1] > 0:
                # the day's distribution as the prices then imply it: prev close x (1 + total return) - close
                paid = close[k - 1] * ratio[k] - close[k]
                div[k] = max(0.0, float(paid))
            placed = str(adj.index[k].date())
        else:
            ratio[pos] *= float(np.exp(gap / len(pos)))
            placed = "spread"
        events.append({"date": adj.index[k] if k is not None else days[-1], "quarter": str(q),
                       "kind": "missed_distribution" if gap > 0 else "overstated_return",
                       "yahoo": float(g.at[q, "yahoo"]), "published": float(g.at[q, "published"]),
                       "factor": float(np.exp(gap)), "placed": placed})
    new_adj = float(adj.iloc[0]) * np.cumprod(ratio)
    out["adj_close"] = new_adj
    if "dividend" in out:
        out["dividend"] = div
    return out, pd.DataFrame(events, columns=EVENT_COLUMNS)


# ------------------------------------------------------------------ detection without published returns

def _factor_returns() -> pd.DataFrame:
    """Daily simple returns of the detection's reference factors: the US stock market (SPYSIM) and ~9-year Treasuries
    (IEFSIM), so bond and balanced funds are judged against rates too."""
    from . import data
    cols = {}
    for name in ("SPYSIM", "IEFSIM"):
        r = data._ref_returns(name)
        if r is not None:
            cols[name] = np.expm1(r)
    return pd.DataFrame(cols)


def flagged_drops(adj: pd.Series, factors: pd.DataFrame | None = None) -> pd.Series:
    """Two-day factor-adjusted drops the fund never gives back (see the module docstring): {day: drop (negative)}.
    Uses the whole series (it is a data-quality screen, not a trading signal)."""
    adj = pd.to_numeric(adj, errors="coerce").dropna()
    adj = adj[adj > 0]
    if len(adj) < 120:
        return pd.Series(dtype=float)
    r = adj.pct_change().dropna()
    fac = _factor_returns() if factors is None else factors
    X = fac.reindex(r.index).fillna(0.0)
    w = dict(window=250, min_periods=60, center=True)
    # rolling least squares of the fund on the factors (centred year): beta = Cov(X)^-1 Cov(X, r)
    mx = X.rolling(**w).mean()
    mr = r.rolling(**w).mean()
    k = X.shape[1]
    cxx = np.empty((len(r), k, k))
    cxy = np.empty((len(r), k))
    for i, a in enumerate(X.columns):
        cxy[:, i] = ((X[a] * r).rolling(**w).mean() - mx[a] * mr).to_numpy()
        for j, b in enumerate(X.columns):
            cxx[:, i, j] = ((X[a] * X[b]).rolling(**w).mean() - mx[a] * mx[b]).to_numpy()
    beta = np.zeros((len(r), k))
    ok_rows = np.isfinite(cxx).all(axis=(1, 2)) & np.isfinite(cxy).all(axis=1)
    if ok_rows.any():
        reg = cxx[ok_rows] + np.eye(k) * 1e-10
        beta[ok_rows] = np.linalg.solve(reg, cxy[ok_rows][..., None])[..., 0]
    beta = np.clip(beta, -1.0, 3.0)
    e = r - pd.Series((beta * X.to_numpy()).sum(axis=1), index=r.index)
    e2 = e + e.shift(-1).fillna(0.0)
    sd = (e - e.rolling(**w).median()).abs().rolling(**w).median() * 1.4826
    thr = np.maximum(DETECT_K * sd * np.sqrt(2.0), DETECT_MIN)
    fwd = e.shift(-2).rolling(DETECT_REVERSAL).sum().shift(-(DETECT_REVERSAL - 1)).fillna(0.0)
    ok = (e2 < -thr) & (fwd < -e2 * 0.5) & (e2 == e2.rolling(5, center=True, min_periods=1).min())
    stress = pd.Series(False, index=r.index)
    for col, lim in (("SPYSIM", STRESS_MOVE), ("IEFSIM", STRESS_MOVE / 2)):
        if col in X:
            m2 = (X[col] + X[col].shift(-1).fillna(0.0)).abs()
            stress |= (m2 > lim).rolling(2 * STRESS_WINDOW + 1, center=True, min_periods=1).max().astype(bool)
    ok &= ~stress
    return e2[ok.fillna(False)]


def recurring(fl: pd.Series, days: int = DETECT_SEASON_DAYS, years: int = DETECT_SEASON_YEARS) -> pd.Series:
    """The flagged drops that recur at the same time of year: funds pay their distributions on about the same date
    every year, so a distribution the data misses leaves a drop each year at that date, while a fund's own bad days
    (a sector crash, a stock's collapse) fall at random dates."""
    if fl.empty:
        return fl
    idx = pd.DatetimeIndex(fl.index)
    doy = idx.dayofyear.to_numpy()
    yr = idx.year.to_numpy()
    keep = np.zeros(len(idx), dtype=bool)
    for i in range(len(idx)):
        d = np.abs(doy - doy[i])
        d = np.minimum(d, 365 - d)
        keep[i] = len(set(yr[d <= days])) >= years
    return fl[keep]


def detect_unreliable(adj: pd.Series, factors: pd.DataFrame | None = None) -> dict | None:
    """{"until": last day of the unreliable stretch, "usable_from": the next session, "days": n, "sum": total drop,
    "first": first flagged day} for a series whose early history shows the missed-distribution signature, else None."""
    fl = flagged_drops(adj, factors)
    fl = fl[fl.index < DETECT_UNTIL]
    fl = recurring(fl)
    if fl.empty:
        return None
    first_price = pd.DatetimeIndex(pd.to_numeric(adj, errors="coerce").dropna().index)[0]
    if (fl.index[0] - first_price).days > DETECT_START_YEARS * 365.25:
        return None
    d = fl.index
    last, n = d[0], 1
    for x in d[1:]:
        if (x - last).days > DETECT_GAP_YEARS * 365.25:
            break
        last, n = x, n + 1
    cl = fl.loc[:last]
    if len(cl) < DETECT_MIN_DAYS or float(cl.sum()) > -DETECT_MIN_SUM:
        return None
    idx = pd.DatetimeIndex(pd.to_numeric(adj, errors="coerce").dropna().index)
    after = idx[idx > last]
    if not len(after):
        return None
    return {"until": last, "usable_from": after[0], "days": int(len(cl)), "sum": float(cl.sum()), "first": d[0]}


def process(t: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    """data.load's step for mutual funds: (bars, repair events, unreliable-history finding or None)."""
    if not is_mutual_fund(t):
        return raw, pd.DataFrame(columns=EVENT_COLUMNS), None
    try:
        pub = published(t)
        if pub is not None:
            out, ev = repair(t, raw, pub)
            return out, ev, None
        adj = pd.to_numeric(raw["adj_close"], errors="coerce") if "adj_close" in raw else raw["close"]
        if adj.dropna().empty or adj.dropna().index[0] >= DETECT_UNTIL:
            return raw, pd.DataFrame(columns=EVENT_COLUMNS), None
        return raw, pd.DataFrame(columns=EVENT_COLUMNS), detect_unreliable(adj)
    except Exception:  # noqa: BLE001 - a data check must never make a price file unreadable
        return raw, pd.DataFrame(columns=EVENT_COLUMNS), None


# ------------------------------------------------------------------ notes

RAW_HINT = ("say 'using raw fund history' (JSON: \"raw_fund_history\": true; command line: --raw-fund-history) to "
            "use the raw history anyway")


def repair_note(tickers, start=None, end=None) -> str | None:
    """'Fund history repaired: ...' for the held funds whose repaired quarters fall inside [start, end]."""
    from . import data
    s = pd.Timestamp(start) if start is not None else pd.Timestamp.min
    e = pd.Timestamp(end) if end is not None else pd.Timestamp.max
    parts = []
    for t in dict.fromkeys(data.canonical(x) for x in tickers):
        ev = data.fund_repairs(t)
        if ev is None or ev.empty:
            continue
        ev = ev[(pd.to_datetime(ev["date"]) >= s) & (pd.to_datetime(ev["date"]) <= e)]
        if ev.empty:
            continue
        worst = ev.iloc[int(np.argmax(np.abs(np.log(ev["factor"].to_numpy(dtype=float)))))]
        years = pd.to_datetime(ev["date"]).dt.year
        span = f"{years.min()}" if years.min() == years.max() else f"{years.min()}-{years.max()}"
        parts.append(f"{t}: {len(ev)} quarter(s) in {span} (largest {worst['quarter']}: {worst['yahoo']:+.1%} in the "
                     f"free data, {worst['published']:+.1%} published)")
    if not parts:
        return None
    return ("Fund history repaired: the free (Yahoo) daily history misses or misstates distributions in some quarters, "
            "which were matched to the funds' published quarterly total returns (data/fund_returns.json; the missing "
            "growth is booked on the distribution's ex-date where the prices show it, else spread over the quarter): "
            + "; ".join(parts) + ".")


def cut_note(tickers, start=None, raw: bool = False) -> str | None:
    """The Warning for funds whose usable history starts later than the free data (or, with raw=True, that the raw
    history is in use)."""
    from . import data
    s = pd.Timestamp(start) if start is not None else None
    parts = []
    for t in dict.fromkeys(data.canonical(x) for x in tickers):
        f = data.fund_unreliable(t)
        if not f:
            continue
        if s is not None and s >= pd.Timestamp(f["usable_from"]):
            continue
        why = (f"its data from {pd.Timestamp(f['first']).year} shows {f['days']} unexplained drops worth "
               f"{f['sum']:.0%} in all up to {pd.Timestamp(f['until']).date()}")
        if raw:
            parts.append(f"{t} before {pd.Timestamp(f['usable_from']).date()} ({why})")
        else:
            parts.append(f"{t} from {pd.Timestamp(f['usable_from']).date()} ({why})")
    if not parts:
        return None
    if raw:
        return ("Warning: using the raw fund history of " + "; ".join(parts) + ": in those years the free (Yahoo) data "
                "misses capital-gain distributions, so the fund's returns are understated.")
    return ("Warning: fund history cut: " + "; ".join(parts) + ". In those early years the free (Yahoo) data misses "
            "capital-gain distributions (the NAV falls on the ex-date with nothing paid), which understates the fund's "
            "return by several percent a year, and there are no published returns on file to repair it "
            "(data/fund_returns.json), so the backtest uses the fund only from that date; " + RAW_HINT + ".")


def main(argv: list[str] | None = None) -> None:
    """python -m backtester.fund_history [TICKER ...]: each fund's repairs or unreliable stretch."""
    import sys
    from . import data
    args = list(sys.argv[1:] if argv is None else argv)
    tickers = args or sorted(t for t in data.available_tickers() if is_mutual_fund(t))
    for t in tickers:
        try:
            data.load(t)
        except Exception as e:  # noqa: BLE001
            print(f"{t}: {e}")
            continue
        ev, f = data.fund_repairs(t), data.fund_unreliable(t)
        if ev is not None and len(ev):
            print(f"{t}: {len(ev)} quarter(s) repaired from published returns: "
                  + ", ".join(f"{r.quarter} {r.yahoo:+.2%}->{r.published:+.2%} ({r.placed})" for r in ev.itertuples()))
        elif f:
            print(f"{t}: usable from {pd.Timestamp(f['usable_from']).date()} ({f['days']} drops, {f['sum']:.1%})")
        elif args:
            print(f"{t}: no repair, no cut")


if __name__ == "__main__":
    main()
