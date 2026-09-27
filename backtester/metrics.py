"""Performance statistics.

All return-based statistics use time-weighted returns (external cash flows removed), so
contributions and withdrawals do not distort CAGR, volatility, Sharpe or drawdowns. Sharpe, Sortino
and alpha are measured against the 3-month T-bill rate unless a fixed rate is given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

TRADING_DAYS = 252


def periods_per_year(index) -> float:
    """Bars per year of the calendar a series actually uses, for annualising its statistics: 252 for a stock-market
    calendar (weekdays less holidays), 365 for a seven-day calendar (crypto, or a portfolio that holds a crypto asset:
    it has weekend bars), 52 for weekly bars, 12 for monthly bars; otherwise the observed count of bars per year.
    A daily calendar is told apart by its share of weekend bars (a stock calendar has none), so a short or gappy
    series is not mistaken for another frequency."""
    idx = pd.DatetimeIndex(index) if not isinstance(index, pd.DatetimeIndex) else index
    if len(idx) < 3:
        return TRADING_DAYS
    ns = idx.as_unit("ns").asi8 if hasattr(idx, "as_unit") else idx.asi8
    gaps = np.diff(ns) / 86_400_000_000_000
    med = float(np.median(gaps)) if len(gaps) else 1.0
    if med <= 4:
        weekend = float((idx.dayofweek >= 5).mean())
        return 365 if weekend > 0.2 else TRADING_DAYS
    if med <= 10:
        return 52
    if 25 <= med <= 35:
        return 12
    if 85 <= med <= 95:
        return 4
    days = (ns[-1] - ns[0]) / 86_400_000_000_000
    return (len(idx) - 1) / (days / 365.25) if days > 0 else TRADING_DAYS
NO_DRAWDOWN = 5e-5   # a drawdown smaller than 0.005% is reported as none

# S&P 500 peak-to-trough dates (closing prices). The pre-1987 events show only for long-history runs (the SIM series,
# e.g. SPYSIM from 1926): an event is listed when a series has data on its first day. Before the 1950s several of the
# long-history series move in monthly steps, so their figures there are month-end to month-end.
CRISES = [
    ("1929 crash and Great Depression", "1929-09-16", "1932-06-01"),
    ("1937-38 recession", "1937-03-06", "1938-03-31"),
    ("1939-42 wartime bear market", "1939-10-25", "1942-04-28"),
    ("1946-49 post-war bear market", "1946-05-29", "1949-06-13"),
    ("1957 recession", "1957-07-15", "1957-10-22"),
    ("1962 flash crash (Kennedy slide)", "1961-12-12", "1962-06-26"),
    ("1966 credit crunch", "1966-02-09", "1966-10-07"),
    ("1968-70 bear market", "1968-11-29", "1970-05-26"),
    ("1973-74 bear market", "1973-01-11", "1974-10-03"),
    ("1980 recession (Volcker)", "1980-02-13", "1980-03-27"),
    ("1981-82 bear market", "1980-11-28", "1982-08-12"),
    ("1987 crash", "1987-10-01", "1987-12-04"),
    ("1990 recession", "1990-07-16", "1990-10-11"),
    ("1998 LTCM", "1998-07-17", "1998-10-08"),
    ("Dot-com bust", "2000-03-24", "2002-10-09"),
    ("Global financial crisis", "2007-10-09", "2009-03-09"),
    ("2010 flash crash", "2010-04-23", "2010-07-02"),
    ("2011 euro crisis", "2011-04-29", "2011-10-03"),
    ("2015-16 selloff", "2015-07-20", "2016-02-11"),
    ("Q4 2018", "2018-09-20", "2018-12-24"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
    ("2025 tariff shock", "2025-02-19", "2025-04-08"),
]


# ------------------------------------------------------------------ return series

def twr_returns(equity: pd.Series, flows: pd.Series | None = None) -> pd.Series:
    """Daily time-weighted returns. Cash flows are made at the close, after the day's return, as both engines
    and the buy-and-hold benchmarks make them (a contribution is invested, and a withdrawal sold, at that day's
    close): r_t = (E_t - cf_t) / E_(t-1) - 1. So a portfolio that is fully invested in one asset has exactly
    that asset's return every day, whatever the flows. On the day a withdrawal empties the account the return is
    the holdings' own move (what was paid out over the previous balance). Days that start with nothing invested
    (a $0 start before the first contribution, or after the money ran out) have no return (0)."""
    prev = equity.shift(1)
    f = flows.reindex(equity.index).fillna(0.0) if flows is not None else 0.0
    r = ((equity - f) / prev - 1).where(prev > 0)
    return r.iloc[1:].fillna(0.0)


def nav(equity: pd.Series, flows: pd.Series | None = None) -> pd.Series:
    """Growth of the starting capital with cash flows removed (time-weighted index). An account that starts
    with $0 (funded by contributions) is indexed to its first funded balance: the returns are time-weighted
    from the first funded day."""
    r = twr_returns(equity, flows)
    scale = float(equity.iloc[0])
    if scale <= 0 and flows is not None:
        funded = equity[equity > 0]
        scale = float(funded.iloc[0]) if len(funded) else 0.0
    out = (1 + r).cumprod() * scale
    return pd.concat([pd.Series([scale], index=equity.index[:1], name=equity.name), out])


def floor_at_zero(nav_: pd.Series) -> pd.Series:
    """A growth index that stays at zero from the first day the account is wiped out (equity at or below
    zero), so returns are -100% on that day and nothing after it; the dollar equity keeps its real value."""
    if not len(nav_) or not (nav_ <= 0).any():
        return nav_
    dead = (nav_ <= 0).cummax()
    return nav_.where(~dead, 0.0)


_RF_MEMO: dict = {}     # the T-bill rate on a set of dates: a report asks for the same dates many times


def rf_daily(index: pd.DatetimeIndex, rf, ppy: float = TRADING_DAYS) -> pd.Series:
    """The risk-free return per bar: the annual rate over `ppy` bars a year (252 trading days by default)."""
    if rf == "tbill":
        s = data.tbill_rate()
        if s.empty:
            return pd.Series(0.0, index=index)
        key = None
        if isinstance(index, pd.DatetimeIndex) and index.tz is None:
            key = (len(index), hash(index.to_numpy(dtype="datetime64[ns]").tobytes()), ppy)
            hit = _RF_MEMO.get(key)
            if hit is not None and hit[0] is s and hit[1].index.equals(index):
                return pd.Series(hit[1].to_numpy(copy=True), index=index, name=hit[1].name)
        out = (s.reindex(index.union(s.index)).ffill().reindex(index).fillna(0.0) / ppy)
        if key is not None:
            if len(_RF_MEMO) > 256:
                _RF_MEMO.clear()
            _RF_MEMO[key] = (s, out.copy())
        return out
    return pd.Series(float(rf or 0.0) / ppy, index=index)


def _anchor_year(series: pd.Series):
    """The first calendar year when it holds only the curve's starting point (the anchor row on the session
    before the first bar, e.g. Dec 31 for a run starting Jan 2): it has no return of its own. Else None."""
    if len(series) > 1 and series.index[0].year != series.index[1].year:
        return series.index[0].year
    return None


def monthly_max_drawdown(nav_: pd.Series) -> float:
    """Max drawdown measured on month-end values only (plus the starting value), as Portfolio Visualizer reports
    it: shallower than the daily one when the trough or the peak fell inside a month."""
    s = nav_.dropna()
    if len(s) < 2:
        return float("nan")
    me = pd.concat([s.iloc[:1], s.resample("ME").last().dropna()])
    return float((me / me.cummax() - 1).min())


def monthly_returns(nav_: pd.Series) -> pd.Series:
    me = nav_.resample("ME").last()
    first = nav_.iloc[0]
    prev = me.shift(1)
    prev.iloc[0] = first
    out = (me / prev - 1).dropna()
    # a month holding only the starting point (e.g. the day before the first bar) has no return
    if len(out) > 1 and (nav_.index.to_period("M") == nav_.index[0].to_period("M")).sum() == 1:
        out = out.iloc[1:]
    return out


def apply_flows(growth: pd.Series, flows: pd.Series | None, start: float | None = None) -> tuple[pd.Series, pd.Series]:
    """(dollar value, flows actually made) of a buy-and-hold curve that receives the given cash flows:
    contributions are invested and withdrawals sold at that day's close. A withdrawal is capped at the balance:
    when it is more than the account holds, what is left is paid out and the account stays at zero from then on
    (later withdrawals are not made). `start`: the starting balance (default: the curve's first value; 0 for an
    account funded only by contributions)."""
    f = flows.reindex(growth.index).fillna(0.0).to_numpy() if flows is not None else np.zeros(len(growth))
    r = growth.pct_change().fillna(0.0).to_numpy()
    out = np.empty(len(growth))
    paid = np.zeros(len(growth))
    v = float(growth.iloc[0]) if start is None else float(start)
    dead = False
    for t in range(len(growth)):
        if t:
            v = v * (1 + r[t])
        if dead:
            out[t] = 0.0
            continue
        amt = f[t]
        if amt < 0 and -amt >= v:
            amt, dead = -v, v > 0
        v = max(v + amt, 0.0)
        paid[t] = amt
        out[t] = v
    return (pd.Series(out, index=growth.index, name=growth.name),
            pd.Series(paid, index=growth.index, name="flows"))


def with_flows(growth: pd.Series, flows: pd.Series | None, start: float | None = None) -> pd.Series:
    """Dollar value of a buy-and-hold curve that receives the same cash flows as a portfolio (see apply_flows)."""
    if flows is None or not len(growth):
        return growth
    return apply_flows(growth, flows, start)[0]


def display_date(ts, first_bar=None):
    """Dates shown to people: the synthetic point one day before the first bar shows as the first bar."""
    if first_bar is not None and ts is not None and not pd.isna(ts) and pd.Timestamp(ts) < pd.Timestamp(first_bar):
        return pd.Timestamp(first_bar)
    return ts


def parse_blend(b) -> list[tuple[str, float]] | None:
    """A blended benchmark as [(ticker, weight)], or None for a single ticker. Accepts {"SPY": 0.6, "AGG": 0.4},
    "60 SPY 40 AGG", "60% SPY / 40% AGG", "SPY 60 AGG 40", "SPY:0.6, AGG:0.4" and "60/40 SPY/AGG". Weights may be
    percentages or fractions; they must add up to 100%."""
    import re
    if isinstance(b, dict):
        pairs = [(_asset_class_or(str(t)), float(w)) for t, w in b.items()]
    else:
        s = _blend_asset_classes(str(b or "").strip())
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s+([\w.^-]+)\s*/\s*([\w.^-]+)", s)
        if m:
            pairs = [(m.group(3), float(m.group(1))), (m.group(4), float(m.group(2)))]
        else:
            toks = [t for t in re.split(r"[\s,/+:;=]+|(?<=\d)%", s) if t and t != "%" and t.lower() not in ("and", "&")]
            if len(toks) < 4 and not any(re.fullmatch(r"\d+(\.\d+)?", t) for t in toks):
                return None
            num = lambda t: re.fullmatch(r"\d+(\.\d+)?", t) is not None  # noqa: E731
            pairs = []
            if toks and num(toks[0]):          # 60 SPY 40 AGG
                if len(toks) % 2:
                    raise ValueError(f"Can't read the benchmark {b!r}: write e.g. \"60 SPY 40 AGG\"")
                for i in range(0, len(toks), 2):
                    if not num(toks[i]) or num(toks[i + 1]):
                        raise ValueError(f"Can't read the benchmark {b!r}: write e.g. \"60 SPY 40 AGG\"")
                    pairs.append((toks[i + 1], float(toks[i])))
            else:                               # SPY 60 AGG 40
                if len(toks) % 2:
                    raise ValueError(f"Can't read the benchmark {b!r}: write e.g. \"60 SPY 40 AGG\"")
                for i in range(0, len(toks), 2):
                    if num(toks[i]) or not num(toks[i + 1]):
                        raise ValueError(f"Can't read the benchmark {b!r}: write e.g. \"60 SPY 40 AGG\"")
                    pairs.append((toks[i], float(toks[i + 1])))
    if len(pairs) == 1 and not isinstance(b, dict):
        return None
    tot = sum(w for _, w in pairs)
    if tot > 1.5:                               # percentages
        pairs = [(t, w / 100) for t, w in pairs]
        tot /= 100
    if not pairs or any(w <= 0 for _, w in pairs) or abs(tot - 1) > 1e-6:
        raise ValueError(f"Benchmark weights must be positive and add up to 100% (got {tot:.1%} in {b!r})")
    out: dict[str, float] = {}
    for t, w in pairs:
        c = data.canonical(t.upper())
        out[c] = out.get(c, 0.0) + w
    return list(out.items())


def _asset_class_or(name: str) -> str:
    """A Portfolio Visualizer asset-class name ("US Stock Market") -> its series (VTISIM); anything else as is."""
    n = " ".join(name.split())
    if " " not in n and n == n.upper():
        return n
    from . import parser as _parser        # (parser imports this module)
    hit = _parser._asset_class_ticker(n)
    return hit[0] if hit else n


def _blend_asset_classes(s: str) -> str:
    """Asset-class names after the weights of a blend -> their series: "60% US Stock Market 40% Total Bond Market"
    -> "60% VTISIM 40% BNDSIM" (tickers are left alone)."""
    import re
    if not re.search(r"[a-z]", s):
        return s

    def fix(m):
        hit = _asset_class_or(m.group(2))
        return f"{m.group(1)} {hit}" if hit != " ".join(m.group(2).split()) else m.group(0)
    return re.sub(r"(\d+(?:\.\d+)?\s*%?)\s+(?:in |of )?(?:the )?([A-Za-z][A-Za-z .&'-]*?[A-Za-z])"
                  r"(?=\s*(?:[,/+;]|\band\b)?\s*(?:\d|$))", fix, s)


def benchmark_label(b) -> str:
    """The benchmark as shown and stored: a ticker ("SPY") or a blend ("60% SPY / 40% AGG")."""
    parts = parse_blend(b)
    if parts is None:
        return data.canonical(_asset_class_or(str(b).strip()))
    return " / ".join(f"{round(w * 100, 2):g}% {t}" for t, w in parts)


def benchmark_first_date(b):
    """The first date the benchmark (every part of a blend) has data, or None."""
    parts = parse_blend(b) or [(b, 1.0)]
    try:
        return max(data.load(t).index[0] for t, _ in parts)
    except (FileNotFoundError, data.DataError, KeyError):
        return None


def check_benchmark(b) -> None:
    """Raise (with did-you-mean suggestions) for an unknown ticker or an unreadable blend."""
    for t, _ in parse_blend(b) or [(b, 1.0)]:
        data.load(t)


BLEND_REBALANCE = ("daily", "weekly", "monthly", "quarterly", "semiannual", "yearly", "none")


def blend_growth(parts: list[tuple[str, float]], index: pd.DatetimeIndex, rebalance: str = "monthly") -> pd.Series | None:
    """Growth of 1 in a blend of total returns rebalanced to its weights at the end of every month (or on another
    schedule: `rebalance` as a portfolio's, e.g. "yearly", "quarterly", "yearly_6", or "none" for buy and hold), from
    the first date of `index` on which every part has data."""
    try:
        cs = [data.load(t)["adj_close"] for t, _ in parts]
    except (FileNotFoundError, data.DataError):
        return None
    df = pd.concat([c.reindex(index.union(c.index)).ffill().reindex(index) for c in cs], axis=1).dropna()
    if len(df) < 2:
        return None
    r = df.pct_change().to_numpy()[1:]
    w0 = np.array([w for _, w in parts])
    if rebalance in (None, "", "monthly"):
        month = df.index.to_period("M")
        ends = np.r_[month[1:] != month[:-1], False]
    else:
        from . import portfolio as _pf     # (portfolio imports this module)
        ends = _pf._schedule(df.index, rebalance).copy()
        ends[0] = False
    val = np.empty(len(df))
    val[0] = 1.0
    hold = w0.copy()                    # dollar value in each part, per 1 of portfolio
    for i in range(1, len(df)):
        hold = hold * (1 + r[i - 1])
        val[i] = hold.sum()
        if i + 1 < len(df) and ends[i]:
            hold = w0 * val[i]          # rebalance at the period's last close
    return pd.Series(val, index=df.index)


def buy_and_hold(ticker: str, index: pd.DatetimeIndex, capital: float, rebalance: str = "monthly") -> pd.Series | None:
    """Growth of `capital` in a ticker (total return), or in a blend such as "60% SPY / 40% AGG" rebalanced
    monthly (or on `rebalance`), from the first date of `index` it has data."""
    try:
        parts = parse_blend(ticker)
    except ValueError:
        return None
    if parts is not None:
        g = blend_growth(parts, index, rebalance)
        return None if g is None or g.empty else (capital * g).rename(benchmark_label(ticker))
    try:
        c = data.load(ticker)["adj_close"]
    except (FileNotFoundError, data.DataError):
        return None
    c = c.reindex(index.union(c.index)).ffill().reindex(index).dropna()
    if c.empty:
        return None
    return (capital * c / c.iloc[0]).rename(ticker)


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def drawdown_table(equity: pd.Series, top: int = 5, first_bar=None) -> pd.DataFrame:
    """The `top` deepest drawdowns with peak / trough / recovery dates."""
    dd = drawdown(equity)
    rows = []
    in_dd, peak_i = False, None
    vals = dd.to_numpy()
    idx = dd.index
    for i, v in enumerate(vals):
        if v < 0 and not in_dd:
            in_dd, peak_i = True, max(i - 1, 0)
        elif v >= 0 and in_dd:
            seg = vals[peak_i:i]
            t = peak_i + int(np.argmin(seg))
            rows.append((idx[peak_i], idx[t], idx[i], vals[t]))
            in_dd = False
    if in_dd:
        seg = vals[peak_i:]
        t = peak_i + int(np.argmin(seg))
        rows.append((idx[peak_i], idx[t], None, vals[t]))
    out = pd.DataFrame(rows, columns=["peak", "trough", "recovery", "depth"])
    if out.empty:
        return out
    out["days_to_trough"] = [(t - p).days for p, t in zip(out.peak, out.trough)]
    out["days_to_recover"] = [(r - t).days if r is not None and not pd.isna(r) else None for t, r in zip(out.trough, out.recovery)]
    out["total_days"] = [((r if r is not None and not pd.isna(r) else idx[-1]) - p).days for p, r in zip(out.peak, out.recovery)]
    out["recovery"] = [r.date() if r is not None and not pd.isna(r) else None for r in out["recovery"]]
    out["peak"] = [display_date(p, first_bar).date() for p in out["peak"]]
    out["trough"] = [t.date() for t in out["trough"]]
    return out.sort_values("depth").head(top).reset_index(drop=True)


def xirr(dates: list, amounts: list) -> float:
    """Money-weighted annual return for cash flows (negative = invested, positive = received)."""
    if len(amounts) < 2 or not any(a > 0 for a in amounts) or not any(a < 0 for a in amounts):
        return np.nan
    t0 = dates[0]
    years = np.array([(d - t0).days / 365.25 for d in dates])
    a = np.array(amounts, dtype=float)

    def npv(r):
        return np.sum(a / (1 + r) ** years)
    lo, hi = -0.99, 10.0
    flo, fhi = npv(lo), npv(hi)
    if np.sign(flo) == np.sign(fhi):
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = npv(mid)
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------------------------------ statistics

def annualise(growth, years):
    """Annualised rate from a growth multiple; a multiple at or below zero (the account was wiped out) is -100%
    rather than NaN, without numpy's warnings about fractional powers of negative numbers."""
    if isinstance(growth, (pd.Series, pd.DataFrame)):
        g = growth.where(growth > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = g ** (1 / years) - 1
        return out.where(~(growth <= 0), -1.0)
    if growth is None or not np.isfinite(growth) or not years or years <= 0:
        return np.nan
    return -1.0 if growth <= 0 else float(growth) ** (1 / years) - 1


def equity_stats(equity: pd.Series, rf="tbill", flows: pd.Series | None = None, first_bar=None,
                 years_from_first_bar: bool = False) -> dict:
    """Statistics for an equity curve; time-weighted if flows are given. `first_bar`: the first real
    bar when the series starts with the synthetic day-before point (changes the dates shown; with
    years_from_first_bar - a signal strategy, in cash until the first bar's close - also the year count of
    the CAGR, which then runs from the first bar, not from the day before it)."""
    nv = nav(equity, flows) if flows is not None and flows.abs().sum() > 0 else equity
    wiped_out = bool((nv <= 0).any())
    nv = floor_at_zero(nv)
    r = nv.pct_change().iloc[1:].fillna(0.0)
    t0 = pd.Timestamp(first_bar) if (years_from_first_bar and first_bar is not None
                                     and nv.index[0] < pd.Timestamp(first_bar) < nv.index[-1]) else nv.index[0]
    years = (nv.index[-1] - t0).days / 365.25
    total = nv.iloc[-1] / nv.iloc[0] - 1
    cagr = annualise(nv.iloc[-1] / nv.iloc[0], years) if years > 0 else np.nan
    ppy = periods_per_year(nv.index)      # bars a year of the calendar used (365 with weekend bars)
    rfd = rf_daily(r.index, rf, ppy)
    ex = r - rfd
    sd = r.std()
    vol = sd * np.sqrt(ppy)
    sharpe = ex.mean() / sd * np.sqrt(ppy) if sd > 0 else np.nan
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(ppy)
    sortino = ex.mean() * ppy / downside if downside > 0 else np.nan
    mr = monthly_returns(nv)
    rfm = rf_daily(mr.index, rf) * 21
    mex = mr - rfm
    sharpe_m = mex.mean() / mr.std() * np.sqrt(12) if len(mr) > 2 and mr.std() > 0 else np.nan
    mdown = np.sqrt((np.minimum(mex, 0) ** 2).mean()) * np.sqrt(12) if len(mr) > 2 else np.nan
    sortino_m = mex.mean() * 12 / mdown if mdown and mdown > 0 else np.nan
    dd = drawdown(nv)
    mdd = dd.min()
    trough = dd.idxmin()
    peak = nv[:trough].idxmax()
    rec = nv[trough:][nv[trough:] >= nv[peak]]
    recovery = rec.index[0] if len(rec) else None
    under = (dd < 0).to_numpy()
    longest, cur = 0, None
    ns = nv.index.to_numpy(dtype="datetime64[ns]").view("i8").tolist()   # (Timestamp - Timestamp).days, without Timestamps
    for i, u in enumerate(under.tolist()):
        if u and cur is None:
            cur = i
        if (not u or i == len(under) - 1) and cur is not None:
            length = (ns[i] - ns[max(cur - 1, 0)]) // 86_400_000_000_000
            longest = max(longest, length)
            cur = None
    ulcer = np.sqrt((dd.pow(2)).mean()) * 100
    yr = nv.groupby(nv.index.year).last()
    yprev = yr.shift(1)
    yprev.iloc[0] = nv.iloc[0]
    yret = yr / yprev - 1
    if _anchor_year(nv) is not None:
        yret = yret.iloc[1:]
    var95 = -np.percentile(r, 5) if len(r) > 20 else np.nan
    cvar95 = -r[r <= np.percentile(r, 5)].mean() if len(r) > 20 else np.nan
    mvar95 = -np.percentile(mr, 5) if len(mr) > 12 else np.nan
    mcvar95 = -mr[mr <= np.percentile(mr, 5)].mean() if len(mr) > 12 else np.nan
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    # days in the market: moving by more than interest could (4x a T-bill day covers a long weekend; >= 0.001%)
    active = (r.abs() > np.maximum(4 * rf_daily(r.index, "tbill", ppy).abs().to_numpy(), 1e-5)) if len(r) else r.astype(bool)
    real = np.nan
    c = data.cpi()
    if not c.empty and years > 0:
        ci = c.reindex(nv.index.union(c.index)).ffill().reindex(nv.index)
        if ci.notna().iloc[0] and ci.notna().iloc[-1]:
            real = annualise((nv.iloc[-1] / nv.iloc[0]) / (ci.iloc[-1] / ci.iloc[0]), years)
    return {
        "start": display_date(nv.index[0], first_bar).date(),
        "end": nv.index[-1].date(),
        "years": years,
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "total_return": total,
        "cagr": cagr,
        "wiped_out": wiped_out,
        "real_cagr": real,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "sharpe_monthly": sharpe_m,
        "sortino_monthly": sortino_m,
        "max_drawdown": mdd,
        # month-end values only (Portfolio Visualizer's figure): shallower when the trough fell inside a month
        "max_drawdown_monthly": min(monthly_max_drawdown(nv), 0.0) if len(nv) > 1 else np.nan,
        # no drawdown to speak of (e.g. only cash interest, which can dip by a hair when T-bill yields turn
        # negative): no peak / trough dates to show
        "max_dd_peak": display_date(peak, first_bar).date() if mdd < -NO_DRAWDOWN else None,
        "max_dd_trough": trough.date() if mdd < -NO_DRAWDOWN else None,
        "max_dd_recovery": recovery.date() if recovery is not None and mdd < -NO_DRAWDOWN else None,
        "longest_underwater_days": longest,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "ulcer_index": ulcer,
        "ulcer_performance": (cagr - float(rfd.mean() * ppy)) / (ulcer / 100) if ulcer > 0 else np.nan,
        "periods_per_year": ppy,
        "best_day": r.max() if len(r) else np.nan,
        "worst_day": r.min() if len(r) else np.nan,
        "best_month": mr.max() if len(mr) else np.nan,
        "worst_month": mr.min() if len(mr) else np.nan,
        "best_year": yret.max() if len(yret) else np.nan,
        "worst_year": yret.min() if len(yret) else np.nan,
        # share of up days among the days the equity moved by more than cash interest could explain (a strategy
        # mostly in cash earns interest every day: counting those days would read ~99% positive)
        "pct_positive_days": (r[active] > 0).mean() if active.any() else np.nan,
        "active_days": float(active.mean()) if len(r) else np.nan,
        "pct_positive_months": (mr > 0).mean() if len(mr) else np.nan,
        "var_95_daily": var95,
        "cvar_95_daily": cvar95,
        "var_95_monthly": mvar95,
        "cvar_95_monthly": mcvar95,
        "skew": float(r.skew()) if len(r) > 3 else np.nan,
        "kurtosis": float(r.kurt()) if len(r) > 3 else np.nan,
        "gain_pain": gains / losses if losses > 0 else np.nan,
        "tail_ratio": (np.percentile(r, 95) / -np.percentile(r, 5)) if len(r) > 20 and np.percentile(r, 5) < 0 else np.nan,
    }


DAILY_ONLY = ("best_day", "worst_day", "pct_positive_days", "var_95_daily", "cvar_95_daily", "tail_ratio", "gain_pain")


def monthly_basis(stats: dict, nv: pd.Series, rf="tbill") -> dict:
    """The statistics with every daily-return figure replaced by its monthly equivalent, for a run that holds a
    series moving in monthly steps (a monthly source spread over daily sessions): volatility, Sharpe, Sortino,
    skew and kurtosis from monthly returns; best / worst day, positive days, daily VaR / CVaR, tail and gain/pain
    ratios blank (they would describe the steps, not the asset)."""
    out = dict(stats)
    mr = monthly_returns(nv)
    if len(mr) < 3:
        return out
    out["volatility"] = float(mr.std() * np.sqrt(12))
    out["sharpe"] = stats.get("sharpe_monthly", np.nan)
    out["sortino"] = stats.get("sortino_monthly", np.nan)
    out["skew"] = float(mr.skew()) if len(mr) > 3 else np.nan
    out["kurtosis"] = float(mr.kurt()) if len(mr) > 3 else np.nan
    for k in DAILY_ONLY:
        out[k] = np.nan
    out["return_basis"] = "monthly"
    return out


def cashflow_stats(equity: pd.Series, flows: pd.Series | None) -> dict:
    if flows is None or flows.abs().sum() == 0:
        return {}
    f = flows[flows != 0]
    contributed = float(f[f > 0].sum())
    withdrawn = float(-f[f < 0].sum()) + 0.0      # + 0.0: no "-0" when nothing was withdrawn
    dates = [equity.index[0]] + list(f.index) + [equity.index[-1]]
    amounts = [-float(equity.iloc[0])] + [-float(x) for x in f] + [float(equity.iloc[-1])]
    return {
        "starting_balance": float(equity.iloc[0]),
        "total_contributions": contributed,
        "total_withdrawals": withdrawn,
        "ending_balance": float(equity.iloc[-1]),
        "net_gain": float(equity.iloc[-1]) - float(equity.iloc[0]) - contributed + withdrawn,
        "money_weighted_return": xirr(dates, amounts),
        "ran_out": depleted_on(equity, flows) is not None,
        "depleted_on": depleted_on(equity, flows),
    }


def depleted_on(equity: pd.Series, flows: pd.Series | None = None):
    """The date a funded account first fell to zero (money ran out), or None."""
    e = equity.iloc[1:]
    funded = (equity.shift(1).fillna(0.0) + (flows.reindex(equity.index).fillna(0.0) if flows is not None else 0.0)).iloc[1:] > 0
    hit = e[(e <= 0) & funded.cummax()]
    return hit.index[0].date() if len(hit) else None


OPEN_REASONS = ("open at end", "still held")


def split_open(trades: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(closed trades, trades still open at the end of the test, marked at the last close)."""
    if trades is None or trades.empty or "exit_reason" not in trades:
        empty = trades.iloc[:0] if trades is not None else pd.DataFrame()
        return (trades if trades is not None else empty), empty
    is_open = trades["exit_reason"].isin(OPEN_REASONS)
    return trades[~is_open], trades[is_open]


def trade_stats(trades: pd.DataFrame, years: float) -> dict:
    """Trade statistics on CLOSED trades. A position still open at the end is not a result yet: its
    mark-to-market P&L is reported separately as open_pnl (open_trades positions)."""
    trades, opened = split_open(trades)
    extra = {"open_trades": int(len(opened)), "open_pnl": float(opened["pnl"].sum()) if len(opened) else 0.0,
             "open_value": float(opened["position_value"].sum()) if len(opened) and "position_value" in opened else 0.0}
    if trades is None or trades.empty:
        return {"trades": 0, **extra}
    r = trades["return"]
    pnl = trades["pnl"]
    wins, losses = trades[pnl > 0], trades[pnl <= 0]
    gross_win, gross_loss = wins["pnl"].sum(), -losses["pnl"].sum()

    def max_run(mask: pd.Series) -> int:
        best = cur = 0
        for m in mask:
            cur = cur + 1 if m else 0
            best = max(best, cur)
        return best

    n = len(trades)
    t_stat = r.mean() / (r.std() / np.sqrt(n)) if n > 1 and r.std() > 0 else np.nan
    has_mae = "mae" in trades and trades["mae"].notna().any()
    return {
        "trades": n,
        "trades_per_year": n / years if years > 0 else np.nan,
        "win_rate": len(wins) / n,
        "avg_return": r.mean(),
        "median_return": r.median(),
        "avg_win": wins["return"].mean() if len(wins) else np.nan,
        "avg_loss": losses["return"].mean() if len(losses) else np.nan,
        "payoff_ratio": (wins["return"].mean() / -losses["return"].mean()) if len(wins) and len(losses) and losses["return"].mean() < 0 else np.nan,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else np.inf,
        "expectancy_usd": pnl.mean(),
        "total_pnl": pnl.sum(),
        "largest_win": r.max(),
        "largest_loss": r.min(),
        "avg_bars_held": trades["bars_held"].mean(),
        "max_consecutive_wins": max_run(pnl > 0),
        "max_consecutive_losses": max_run(pnl <= 0),
        "avg_mae": trades["mae"].mean() if has_mae else np.nan,
        "avg_mfe": trades["mfe"].mean() if has_mae else np.nan,
        "t_stat": t_stat,
        "total_commission": trades["commission"].sum(),
        "long_trades": int((trades["side"] == "long").sum()),
        "short_trades": int((trades["side"] == "short").sum()),
        "long_win_rate": (trades.loc[trades.side == "long", "pnl"] > 0).mean() if (trades.side == "long").any() else np.nan,
        "short_win_rate": (trades.loc[trades.side == "short", "pnl"] > 0).mean() if (trades.side == "short").any() else np.nan,
        **extra,
    }


def relative_stats(nav_: pd.Series, bench: pd.Series | None, rf="tbill", freq: str = "daily") -> dict:
    """Regression and capture statistics vs a benchmark over the overlapping period. freq "monthly": beta, alpha,
    correlation and tracking error from monthly returns (for series that move in monthly steps)."""
    if bench is None or len(bench.dropna()) < 30:
        return {}
    df = pd.concat([nav_.pct_change(), bench.pct_change()], axis=1, join="inner").dropna()
    if len(df) < 30:
        return {}
    daily = df
    ppy = periods_per_year(df.index)
    k = ppy
    if freq == "monthly":
        both = pd.concat([nav_, bench], axis=1, join="inner").dropna()
        df = monthly_returns_frame(both).dropna()
        if len(df) < 12:
            return {}
        k = 12
        rfd = rf_daily(df.index, rf) * 21
    else:
        rfd = rf_daily(df.index, rf, ppy)
    s, b = df.iloc[:, 0] - rfd, df.iloc[:, 1] - rfd
    beta = np.cov(s, b)[0, 1] / b.var()
    alpha = (s.mean() - beta * b.mean()) * k
    active = df.iloc[:, 0] - df.iloc[:, 1]
    te = active.std() * np.sqrt(k)
    ir = active.mean() * k / te if te > 0 else np.nan
    corr = s.corr(b)
    df = daily
    rfd = rf_daily(df.index, rf, ppy)
    # capture ratios on monthly returns
    m = pd.concat([monthly_returns((1 + df.iloc[:, 0]).cumprod()), monthly_returns((1 + df.iloc[:, 1]).cumprod())], axis=1).dropna()
    up, dn = m[m.iloc[:, 1] > 0], m[m.iloc[:, 1] < 0]

    def geo(x):
        return (1 + x).prod() ** (1 / len(x)) - 1 if len(x) else np.nan
    upc = geo(up.iloc[:, 0]) / geo(up.iloc[:, 1]) if len(up) else np.nan
    dnc = geo(dn.iloc[:, 0]) / geo(dn.iloc[:, 1]) if len(dn) else np.nan
    ann = (1 + df.iloc[:, 0]).prod() ** (ppy / len(df)) - 1
    rf_ann = float(rfd.mean() * ppy)
    return {"beta": beta, "alpha_annual": alpha, "correlation": corr, "r_squared": corr ** 2,
            "tracking_error": te, "information_ratio": ir,
            "treynor": (ann - rf_ann) / beta if beta else np.nan,
            "up_capture": upc, "down_capture": dnc, "period_start": df.index[0].date(), "freq": freq}


def rolling_series(nav_: pd.Series, bench: pd.Series | None, rf="tbill") -> dict:
    """Rolling 12-month and 36-month returns, 6-month Sharpe and beta, 3-month volatility. The windows and the
    annualisation use the series' own bars a year (periods_per_year: 252, or 365 on a seven-day calendar)."""
    r = nav_.pct_change()
    ppy = periods_per_year(nav_.index)
    y1, h6, q3 = int(round(ppy)), int(round(ppy / 2)), int(round(ppy / 4))
    out = {}
    out["return_12m"] = nav_ / nav_.shift(y1) - 1
    out["return_36m_ann"] = annualise(nav_ / nav_.shift(3 * y1), 3)
    ex = r - rf_daily(r.index, rf, ppy)
    out["sharpe_6m"] = ex.rolling(h6).mean() / r.rolling(h6).std() * np.sqrt(ppy)
    out["vol_3m"] = r.rolling(q3).std() * np.sqrt(ppy)
    if bench is not None:
        b = bench.reindex(nav_.index).pct_change()
        out["beta_6m"] = r.rolling(h6).cov(b) / b.rolling(h6).var()
    return out


def rolling_summary(nav_: pd.Series) -> dict:
    """Best / worst / average rolling-period annualised returns (Portfolio Visualizer style)."""
    out = {}
    ppy = periods_per_year(nav_.index)
    for yrs in (1, 3, 5, 10):
        n = int(round(yrs * ppy))
        if len(nav_) <= n + 5:
            continue
        rr = annualise(nav_ / nav_.shift(n), yrs)
        rr = rr.dropna()
        out[f"{yrs}y"] = {"avg": rr.mean(), "best": rr.max(), "worst": rr.min(), "pct_positive": (rr > 0).mean()}
    return out


def crisis_table(series: dict[str, pd.Series]) -> list[dict]:
    rows = []
    for name, a, b in CRISES:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        row = {"event": name, "start": a.date(), "end": b.date()}
        any_val = False
        for k, s in series.items():
            if s is None:
                continue
            seg = s[(s.index >= a - pd.Timedelta(days=5)) & (s.index <= b)]
            seg0 = s[s.index <= a]
            if len(seg0) == 0 or s.index[0] > a:
                row[k] = None
                continue
            end = s[s.index <= b]
            row[k] = float(end.iloc[-1] / seg0.iloc[-1] - 1) if len(end) and seg0.iloc[-1] > 0 else None
            any_val = any_val or row[k] is not None
        if any_val:
            rows.append(row)
    return rows


def factor_regression(nav_: pd.Series, rf="tbill", freq: str = "daily") -> dict:
    """Fama-French 5 factors + momentum regression on daily excess returns (OLS with t-stats). freq "monthly":
    on complete calendar months (the daily factors compounded), for series that move in monthly steps."""
    f = data.factors()
    if f.empty:
        return {}
    if freq == "monthly":
        m = monthly_returns_frame(nav_.to_frame("r")).dropna()
        per = f.index.to_period("M")
        fm = (1 + f).groupby(per).prod() - 1
        m.index = m.index.to_period("M")
        df = pd.concat([m, fm], axis=1, join="inner").dropna()
        if len(df) < 24 or "RF" not in df:
            return {}
        k = 12
    else:
        nv_ = nav_
        if periods_per_year(nav_.index) != TRADING_DAYS and len(nav_):
            # a seven-day series (weekend bars): measured on the factors' own trading days, so Monday's return
            # includes the weekend's move rather than dropping it
            fi = f.index[(f.index >= nav_.index[0]) & (f.index <= nav_.index[-1])]
            nv_ = nav_.reindex(nav_.index.union(fi)).ffill().reindex(fi)
        r = nv_.pct_change().dropna()
        df = pd.concat([r.rename("r"), f], axis=1, join="inner").dropna()
        if len(df) < 120 or "RF" not in df:
            return {}
        k = TRADING_DAYS       # the factors' own (stock-market) calendar
    y = df["r"] - df["RF"]
    cols = [c for c in ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom") if c in df]
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
    coef, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    resid = y.to_numpy() - X @ coef
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tstat = coef / se
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    names = ["alpha"] + cols
    d0, d1 = df.index[0], df.index[-1]
    if freq == "monthly":
        d0, d1 = d0.start_time, d1.end_time
    return {"period_start": d0.date(), "period_end": d1.date(), "r_squared": r2,
            "observations": len(df), "freq": freq,
            "coefficients": [{"factor": n, "loading": float(c * (k if n == "alpha" else 1)),
                              "t_stat": float(t)} for n, c, t in zip(names, coef, tstat)]}


def correlation_matrix(series: dict[str, pd.Series]) -> dict:
    df = pd.concat({k: v for k, v in series.items() if v is not None}, axis=1).dropna()
    if len(df) < 60:
        return {}
    m = monthly_returns_frame(df).corr()
    return {"names": list(m.columns), "matrix": m.round(4).to_numpy().tolist()}


def complete_months(px):
    """Month-end values labelled with each month's last trading day in the data (not a calendar month end that
    may lie in the future). The last month is dropped when the data stops before that month's last scheduled
    NYSE session: a month still in progress has no monthly return yet."""
    from . import calendar as _cal
    if not len(px):
        return px
    per = px.index.to_period("M")
    me = px.groupby(per).last()
    me.index = pd.DatetimeIndex(px.index.to_series().groupby(per).max().to_numpy())
    last = px.index[-1]
    if len(me) > 1 and _cal.next_sessions(last)[0].to_period("M") == last.to_period("M"):
        me = me.iloc[:-1]
    return me


def monthly_returns_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly returns of complete months, labelled by each month's last trading day."""
    return complete_months(df).pct_change().dropna()


def yearly_returns(series: dict[str, pd.Series]) -> pd.DataFrame:
    cols = {}
    for name, eq in series.items():
        if eq is None:
            continue
        ye = eq.groupby(eq.index.year).last()
        prev = ye.shift(1)
        prev.iloc[0] = eq.iloc[0]
        r = ye / prev - 1
        cols[name] = r.iloc[1:] if _anchor_year(eq) is not None else r
    return pd.DataFrame(cols)


class _Years:
    """A series' rows by calendar year: before(y) is s[s.index.year < y] and of(y) is s[s.index.year == y], as
    positional slices when the dates are in order (the usual case: no boolean mask over the whole series per year)."""

    def __init__(self, s: pd.Series):
        self.s = s
        self.years = s.index.year.to_numpy()        # (computed once: .year is recomputed on every access)
        self.ordered = bool(s.index.is_monotonic_increasing)

    def before(self, y) -> pd.Series:
        if self.ordered:
            return self.s.iloc[:int(np.searchsorted(self.years, y, "left"))]
        return self.s[self.years < y]

    def of(self, y) -> pd.Series:
        if self.ordered:
            return self.s.iloc[int(np.searchsorted(self.years, y, "left")):int(np.searchsorted(self.years, y, "right"))]
        return self.s[self.years == y]


def yearly_detail(nav_: pd.Series, trades: pd.DataFrame, exposure: pd.Series, first_bar=None) -> pd.DataFrame:
    rows = {}
    skip = _anchor_year(nav_)
    nav_y, exp_y = _Years(nav_), _Years(exposure)
    for y, eq in nav_.groupby(nav_y.years):
        if y == skip:
            continue
        prev = nav_y.before(y)
        base = prev.iloc[-1] if len(prev) else eq.iloc[0]
        path = pd.concat([pd.Series([base]), eq]).reset_index(drop=True)
        first, last = display_date(eq.index[0], first_bar), eq.index[-1]
        opening = not len(prev) or (skip is not None and len(prev) == 1)   # the first year (after the anchor)
        partial = opening and (first.month > 1 or first.day > 7) or (last.month < 12 or last.day < 24)
        rows[y] = {
            "return": eq.iloc[-1] / base - 1 if base > 0 else np.nan,   # nothing left to earn a return on
            "max_drawdown": (path / path.cummax() - 1).min(),
            "end_equity": eq.iloc[-1],
            "exposure": exp_y.of(y).mean(),
            "partial": bool(partial),
            "from": first.date() if partial else None,
            "to": last.date() if partial else None,
        }
    out = pd.DataFrame(rows).T
    trades = split_open(trades)[0] if trades is not None else trades
    if trades is not None and not trades.empty:
        ty = pd.to_datetime(trades["exit_date"]).dt.year
        out["trades"] = trades.groupby(ty).size().reindex(out.index).fillna(0).astype(int)
        out["win_rate"] = trades.groupby(ty)["pnl"].apply(lambda p: (p > 0).mean()).reindex(out.index)
        out["pnl"] = trades.groupby(ty)["pnl"].sum().reindex(out.index).fillna(0)
    else:
        out["trades"], out["win_rate"], out["pnl"] = 0, np.nan, 0.0
    out.index.name = "year"
    return out


_CPI_PERIODS: list = [None, None]


def _cpi_periods(m: pd.Series) -> pd.PeriodIndex:
    """m.index.to_period("M"), kept for the CPI series it was computed from (a report asks once per year)."""
    if _CPI_PERIODS[0] is not m:
        _CPI_PERIODS[:] = [m, m.index.to_period("M")]
    return _CPI_PERIODS[1]


def calendar_inflation(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """CPI inflation between two dates for REPORTING, on calendar months (no publication lag): from the CPI of the
    month before `start` (a year starting in January uses the previous December) to the CPI of `end`'s month, or
    the latest month published if that is not out yet. A full calendar year is December to December."""
    m = data.cpi_monthly()
    if m.empty:
        return np.nan
    b = pd.Timestamp(start).to_period("M") - 1
    e = pd.Timestamp(end).to_period("M")
    per = _cpi_periods(m)
    mb = m[per <= b]
    me = m[per <= e]
    if not len(mb) or not len(me) or mb.index[-1].to_period("M") != b or me.index[-1].to_period("M") <= b:
        return np.nan
    return float(me.iloc[-1] / mb.iloc[-1] - 1)


def yearly_balances(equity: pd.Series, nav_: pd.Series, flows: pd.Series | None = None) -> pd.DataFrame:
    """Per calendar year, for the account itself (Portfolio Visualizer's annual table): start and end balance,
    contributions, withdrawals, inflation (CPI, December to December: see calendar_inflation) and the time-weighted
    return after inflation. Years after the money ran out have no return (the account held nothing), and the year
    it ran out has its return up to that day."""
    f = flows.reindex(equity.index).fillna(0.0) if flows is not None else pd.Series(0.0, index=equity.index)
    dep = depleted_on(equity, flows) if flows is not None else None
    rows = {}
    first_bar = equity.index[1] if len(equity) > 1 else equity.index[0]
    skip = _anchor_year(equity)
    eq_y, nav_y, f_y = _Years(equity), _Years(nav_), _Years(f)
    for y, eq in equity.groupby(eq_y.years):
        if y == skip:
            continue
        prev = eq_y.before(y)
        nprev, ny = nav_y.before(y), nav_y.of(y)
        fy = f_y.of(y)
        dead = dep is not None and y > dep.year
        ret = np.nan
        if len(ny) and not dead:
            base = nprev.iloc[-1] if len(nprev) else ny.iloc[0]
            ret = ny.iloc[-1] / base - 1 if base > 0 else np.nan
        days = eq.index[eq.index >= first_bar]
        infl = calendar_inflation(days[0], days[-1]) if len(days) else np.nan
        rows[y] = {"start_balance": float(prev.iloc[-1]) if len(prev) else float(eq.iloc[0]),
                   "contributions": float(fy[fy > 0].sum()), "withdrawals": float(-fy[fy < 0].sum()) + 0.0,
                   "end_balance": float(eq.iloc[-1]), "inflation": infl,
                   "real_return": (1 + ret) / (1 + infl) - 1 if _finite(ret) and _finite(infl) else np.nan}
    out = pd.DataFrame(rows).T
    out.index.name = "year"
    return out


def income_yearly(equity: pd.Series, income: pd.DataFrame | None) -> pd.DataFrame:
    """Per calendar year: the dividends (and other distributions) and the cash interest received, their total, the
    balance at the start of the year (the previous year-end, or the starting balance) and the yield on it (Portfolio
    Visualizer's annual income table). Interest paid on borrowing shows as negative interest."""
    if income is None or not len(income) or not len(equity):
        return pd.DataFrame()
    inc = income.reindex(equity.index).fillna(0.0)
    skip = _anchor_year(equity)
    rows = {}
    eq_y = _Years(equity)
    inc_years = inc.index.year.to_numpy()
    for y, eq in equity.groupby(eq_y.years):
        if y == skip:
            continue
        prev = eq_y.before(y)
        start = float(prev.iloc[-1]) if len(prev) else float(eq.iloc[0])
        iy = inc[inc_years == y]
        d, i_ = float(iy["dividends"].sum()), float(iy["interest"].sum())
        rows[y] = {"start_balance": start, "dividends": d, "interest": i_, "income": d + i_,
                   "dividend_yield": d / start if start > 0 else np.nan,
                   "yield": (d + i_) / start if start > 0 else np.nan}
    out = pd.DataFrame(rows).T
    out.index.name = "year"
    return out


def asset_yearly_returns(prices: dict, tickers: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Each holding's calendar-year total return (adjusted closes, dividends reinvested) over the run's dates: year-end
    to year-end, the first year from the run's first day (or from the holding's first price when it starts later)."""
    cols = {}
    for t in tickers:
        df = prices.get(t)
        if df is None or not len(df):
            continue
        c = (df["adj_close"] if "adj_close" in df else df["close"]).dropna()
        c = c[c.index <= index[-1]]
        s = c.reindex(index.union(c.index)).ffill().reindex(index).dropna()
        if len(s) > 1:
            cols[t] = s
    if not cols:
        return pd.DataFrame()
    out = yearly_returns(cols)
    out.index.name = "year"
    return out


def monthly_table(nav_: pd.Series) -> pd.DataFrame:
    m = monthly_returns(nav_)
    tbl = pd.DataFrame({"year": m.index.year, "month": m.index.month, "r": m.values})
    out = tbl.pivot(index="year", columns="month", values="r").reindex(columns=range(1, 13))
    out.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return out


def monte_carlo(equity: pd.Series, flows: pd.Series | None = None, sims: int = 1000, block: int = 20,
                seed: int = 7, schedule: pd.Series | None = None, until=None) -> dict:
    """Block bootstrap of daily time-weighted returns. With cash flows, the flow schedule is replayed on each
    resampled path, giving a probability that the money lasts. `schedule`: the flows as scheduled (before any
    cap at the balance; default `flows`); `until`: the last day the account was funded (returns after it are
    not resampled, but the paths still run over the whole period)."""
    if until is not None:
        cut = equity.index <= pd.Timestamp(until)
        r = twr_returns(equity[cut], flows[flows.index <= pd.Timestamp(until)] if flows is not None else None).to_numpy()
    else:
        r = twr_returns(equity, flows).to_numpy()
    n = len(equity) - 1
    m = len(r)
    if m < block * 3:
        return {}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    years = n / periods_per_year(equity.index)
    cagr, mdd, final = np.empty(sims), np.empty(sims), np.empty(sims)
    sched = schedule if schedule is not None else flows
    fl = sched.reindex(equity.index).fillna(0.0).to_numpy()[1:] if sched is not None else np.zeros(n)
    has_flows = np.abs(fl).sum() > 0
    ruined = 0
    # every path's block starts, drawn in the same order as one path at a time
    S = np.array([rng.integers(0, m - block, nb) for _ in range(sims)]).reshape(sims, nb)
    e0 = float(equity.iloc[0])
    blocks = np.lib.stride_tricks.sliding_window_view(1 + r, block)   # blocks[a] = 1 + r[a:a + block]
    def chunk(c0: int) -> tuple:
        c1 = min(c0 + 64, sims)
        G = np.ascontiguousarray(blocks[S[c0:c1]].reshape(c1 - c0, -1)[:, :n])      # 1 + each path's returns
        np.cumprod(G, axis=1, out=G)
        M = np.maximum.accumulate(G, axis=1)
        np.divide(G, M, out=M)
        # min(x - 1) is min(x) - 1 (subtracting 1 is monotonic, also after rounding)
        return c0, c1, G[:, -1].copy(), M.min(axis=1) - 1
    # 64 paths at a time (each row is one path, as computed alone), the chunks on a few threads: numpy releases the
    # GIL for these large array operations, and every chunk writes only its own paths
    starts = list(range(0, sims, 64))
    if len(starts) > 2 and n * 64 > 200_000:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(starts))) as pool:
            parts = list(pool.map(chunk, starts))
    else:
        parts = [chunk(c0) for c0 in starts]
    for c0, c1, g_last, dd in parts:
        for j, s in enumerate(range(c0, c1)):
            g_end = g_last[j]
            cagr[s] = g_end ** (1 / years) - 1 if g_end > 0 else -1.0
            mdd[s] = dd[j]
            if not has_flows:
                final[s] = e0 * g_end
    if has_flows:
        # the flow schedule replayed on every path at once, day by day (the same arithmetic as one path at a time):
        # a balance that reaches zero is ruined and stays at 0. Between two flow days nothing can ruin a path (its
        # daily growth factors are positive), so such a stretch is one cumulative product per path - the same
        # sequence of multiplications as the day-by-day loop, so the same numbers - and only flow days (and any day
        # with a non-positive growth factor) are stepped one at a time
        v = np.full(sims, float(equity.iloc[0]))
        alive = np.ones(sims, dtype=bool)
        fdays = np.flatnonzero(fl != 0)
        WIN = block * max(1, 2048 // block)       # a window of whole blocks: gathered block by block (fast)
        t0 = 0
        while t0 < n and alive.any():
            L = min(WIN, n - t0)
            b0 = t0 // block
            Wx = np.empty((sims, L + 1))           # column 0 (and each spent column) holds the balance a stretch starts from
            # day t0 + k of each path: 1 + r[S[path, (t0 + k) // block] + (t0 + k) % block], as the day-by-day loop reads it
            Wx[:, 1:] = blocks[S[:, b0:b0 + -(-L // block)]].reshape(sims, -1)[:, :L]
            k = 0
            while k < L:
                j = int(np.searchsorted(fdays, t0 + k))
                stop = min(int(fdays[j]) - t0 if j < len(fdays) else L, L)     # the next flow day (window-local)
                if stop > k:
                    if (Wx[:, k + 1:stop + 1] > 0).all():
                        Wx[:, k] = v
                        v = np.cumprod(np.ascontiguousarray(Wx[:, k:stop + 1]), axis=1)[:, -1].copy()
                        k = stop
                    else:
                        stop += 1 if stop < L else 0     # step the whole stretch (and the flow day) one day at a time
                while k < min(stop + (1 if stop < L and k == stop else 0), L):
                    nxt = v * Wx[:, k + 1] + fl[t0 + k]      # flows at the close, as in the backtest
                    v = np.where(alive, nxt, v)
                    dead = alive & (v <= 0)
                    if dead.any():
                        v[dead] = 0.0
                        alive &= ~dead
                    k += 1
            t0 += L
        final[:] = v
        ruined = int((~alive).sum())
    q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    out = {
        "sims": sims, "block_days": block,
        "cagr_p5": q(cagr, 5), "cagr_p50": q(cagr, 50), "cagr_p95": q(cagr, 95),
        "mdd_p5": q(mdd, 5), "mdd_p50": q(mdd, 50), "mdd_p95": q(mdd, 95),
        "final_p5": q(final, 5), "final_p50": q(final, 50), "final_p95": q(final, 95),
        "prob_loss": float((cagr < 0).mean()),
    }
    if has_flows:
        out["success_rate"] = 1 - ruined / sims
    return out


def exposure_stats(exposure: pd.Series, positions: pd.Series, in_market: pd.Series | None = None) -> dict:
    im = in_market if in_market is not None else positions > 0
    return {
        "time_in_market": float(im.iloc[1:].mean()) if len(im) > 1 else 0.0,
        "avg_exposure": float(exposure.iloc[1:].mean()) if len(exposure) > 1 else 0.0,
        "max_positions_held": int(positions.max()),
        "avg_positions_when_invested": float(positions[positions > 0].mean()) if (positions > 0).any() else 0.0,
    }


TRAILING = (("3M", 3, False), ("YTD", None, False), ("1Y", 12, False), ("3Y", 36, True), ("5Y", 60, True),
            ("10Y", 120, True))


def trailing_returns(series: pd.Series, first_bar=None, end=None) -> dict:
    """Portfolio Visualizer's trailing returns, as of the series' last date (or `end`): 3 months, year to date
    and 1 year (not annualised), 3, 5 and 10 years annualised, and the full period annualised. A period longer
    than the series' history is None. `series` is a growth index (no cash flows)."""
    s = series.dropna()
    s = s[s > 0] if (s > 0).any() else s
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    if len(s) < 2:
        return {}
    last_d, last_v = s.index[-1], float(s.iloc[-1])
    first_d = pd.Timestamp(first_bar) if first_bar is not None else s.index[0]
    out = {"as_of": last_d.date()}

    def value_on(d):
        x = s[s.index <= d]
        return float(x.iloc[-1]) if len(x) else None
    for key, months, ann in TRAILING:
        if months is None:
            d0 = pd.Timestamp(year=last_d.year, month=1, day=1) - pd.Timedelta(days=1)
            if d0 < s.index[0]:
                out[key] = None
                continue
        else:
            d0 = last_d - pd.DateOffset(months=months)
            if d0 < s.index[0] - pd.Timedelta(days=3) or d0 < first_d - pd.Timedelta(days=5):
                out[key] = None
                continue
        v0 = value_on(d0)
        if v0 is None or v0 <= 0:
            out[key] = None
            continue
        g = last_v / v0
        out[key] = annualise(g, months / 12) if ann else g - 1
    years = (last_d - s.index[0]).days / 365.25
    out["Full"] = annualise(last_v / float(s.iloc[0]), years) if years > 0 else None
    out["full_ann"] = years >= 1
    if not out["full_ann"] and years > 0:
        out["Full"] = last_v / float(s.iloc[0]) - 1
    return out


def caveats(stats: dict) -> list[str]:
    """Notes that help read the headline numbers correctly."""
    out = []
    cagr, sharpe, vol = stats.get("cagr"), stats.get("sharpe"), stats.get("volatility")
    if _finite(cagr) and _finite(sharpe) and cagr < 0 < sharpe:
        drag = f" (volatility {vol * 100:.0f}% a year costs about {vol * vol / 2 * 100:.0f} percentage points a year)" if _finite(vol) else ""
        out.append(f"CAGR is negative ({cagr * 100:.1f}%) while the Sharpe ratio is positive ({sharpe:.2f}): Sharpe uses the "
                   "average daily return, CAGR the compounded one. With high volatility the compounded return is roughly "
                   f"the average minus half the variance{drag}, so a strategy can gain on an average day and still lose "
                   "money over time (volatility drag). Judge it by CAGR and drawdown.")
    return out


# ------------------------------------------------------------------ sanity checks

# Ratios that are meaningless when a signal strategy never traded (the curve is just cash interest).
DEGENERATE_RATIOS = ("sharpe", "sortino", "sharpe_monthly", "sortino_monthly", "calmar", "ulcer_performance",
                     "gain_pain", "tail_ratio")
FEW_TRADES = 10


def _finite(x) -> bool:
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def suppress_degenerate(stats: dict) -> dict:
    """Copy of equity stats with the risk-adjusted ratios blanked (for runs that never traded)."""
    out = dict(stats)
    for k in DEGENERATE_RATIOS:
        if k in out:
            out[k] = np.nan
    return out


def result_warnings(kind: str, stats: dict, tstats: dict, interest: float | None = None,
                    has_flows: bool = False) -> list[dict]:
    """Warnings about degenerate or implausible results.

    Each warning is {"code", "level" ("error" | "warn" | "info"), "message", "detail", ...}. Trade-count checks
    apply to signal strategies only: an allocation portfolio's "trades" are holding periods, so a buy-and-hold
    portfolio legitimately has very few.
    """
    W: list[dict] = []
    n = int(tstats.get("trades") or 0)
    n_open = int(tstats.get("open_trades") or 0)
    signal = kind == "signal"
    if signal and n == 0 and n_open:
        W.append({"code": "few_trades", "level": "warn", "message": "No closed trades yet",
                  "detail": f"The only position{'s are' if n_open > 1 else ' is'} still open at the end, so there are no "
                            "closed-trade statistics; its unrealised P&L is shown as Open P&L."})
    elif signal and n == 0:
        W.append({"code": "no_trades", "level": "error", "message": "No trades",
                  "detail": "The entry condition never triggered, so there is nothing to evaluate. "
                            "Sharpe, Sortino, Calmar and the trade statistics are not shown."})
    elif signal and n < FEW_TRADES:
        W.append({"code": "few_trades", "level": "warn", "message": "Too few trades for the statistics to mean much",
                  "detail": f"Only {n} trade{'s' if n != 1 else ''}. Win rate, profit factor and Sharpe from fewer than "
                            f"{FEW_TRADES} trades are mostly luck."})
    alarms = []
    sharpe, cagr, mdd = stats.get("sharpe"), stats.get("cagr"), stats.get("max_drawdown")
    if not (signal and n == 0 and not n_open):
        if _finite(sharpe) and sharpe > 3:
            alarms.append(f"Sharpe ratio {sharpe:.2f} is above 3")
        if _finite(cagr) and cagr > 1:
            alarms.append(f"CAGR {cagr * 100:.0f}% is above 100% a year")
    if signal and n > 0:
        wr, pf = tstats.get("win_rate"), tstats.get("profit_factor")
        if n > 100 and _finite(wr) and wr > 0.9:
            alarms.append(f"win rate {wr * 100:.1f}% over {n} trades is above 90%")
        if _finite(mdd) and abs(mdd) < 1e-12:
            alarms.append("the equity curve never had a drawdown despite trading")
        if n > 50 and pf is not None and (not _finite(pf) or pf > 5):
            alarms.append(f"profit factor {'infinite (no losing trades)' if not _finite(pf) else f'{pf:.1f}'} "
                          f"over {n} trades is above 5")
    if alarms:
        W.append({"code": "too_good", "level": "error",
                  "message": "Results look too good — check for lookahead or data errors",
                  "detail": "Triggered by: " + "; ".join(alarms) + ".", "rules": alarms})
    if stats.get("wiped_out"):
        W.append({"code": "wiped_out", "level": "error", "message": "The account was wiped out",
                  "detail": f"Equity fell to zero or below (it ends at ${stats.get('end_equity', 0):,.0f}), so everything "
                            "was lost: CAGR is shown as -100% and the growth index stays at zero from that day."})
    if signal and n > 0 and not has_flows and _finite(interest) and interest > 0:
        start, end, years = stats.get("start_equity"), stats.get("end_equity"), stats.get("years")
        profit = (end or 0) - (start or 0)
        if profit > 0 and interest > 0.25 * profit and start and years and years > 0:
            ex_end = end - interest
            cagr_ex = (ex_end / start) ** (1 / years) - 1 if ex_end > 0 else np.nan
            ex_txt = f"{cagr_ex * 100:.2f}%" if _finite(cagr_ex) else "n/a (the trading alone lost everything)"
            W.append({"code": "interest_share", "level": "info",
                      "message": f"Cash interest is {interest / profit * 100:.0f}% of the total profit",
                      "detail": f"Idle cash earned ${interest:,.0f} of T-bill interest out of ${profit:,.0f} profit. "
                                f"Without it the CAGR would be about {ex_txt} instead of {pct_txt(cagr)} "
                                "(approximation: cumulative interest subtracted from the final balance).",
                      "cagr_ex_interest": cagr_ex, "interest_share": interest / profit})
    return W


def pct_txt(x) -> str:
    return f"{x * 100:.2f}%" if _finite(x) else "n/a"


# ------------------------------------------------------------------ multiple testing

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sd: float) -> float:
    """Expected maximum of n_trials Sharpe ratios under the null (true Sharpe 0) whose estimates have spread sd.

    Bailey & López de Prado (2014): E[max] ~ sd * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e))), g = Euler's
    constant. For large N this is close to sd * sqrt(2 ln N).
    """
    from statistics import NormalDist
    if n_trials < 2 or not _finite(sd) or sd <= 0:
        return 0.0
    z = NormalDist().inv_cdf
    return sd * ((1 - EULER_GAMMA) * z(1 - 1 / n_trials) + EULER_GAMMA * z(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(best_sr: float, trial_srs: list, n_obs: int, skew: float = 0.0, excess_kurt: float = 0.0,
                    periods: int = TRADING_DAYS) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Sharpe ratios are given annualised and converted to per-period values. SR0 is the expected best Sharpe of
    N = len(trial_srs) strategies with no skill; DSR is the probabilistic Sharpe ratio of best_sr against SR0:

        DSR = Phi((SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2))

    with T the number of return observations and kurt the raw (non-excess) kurtosis of the chosen strategy's
    returns. It is the probability that the true Sharpe is above zero once the selection among N trials is
    accounted for. The trials are treated as independent, so correlated combinations make it conservative.
    """
    from statistics import NormalDist
    srs = np.array([s for s in trial_srs if _finite(s)], dtype=float) / np.sqrt(periods)
    N = len(srs)
    out = {"n_trials": N, "best_sharpe": best_sr, "n_obs": n_obs,
           "sd_sharpe": float(srs.std(ddof=1) * np.sqrt(periods)) if N > 1 else np.nan,
           "expected_max_sharpe": np.nan, "expected_max_sharpe_simple": np.nan, "dsr": np.nan}
    if N < 2 or not _finite(best_sr) or not n_obs or n_obs < 3:
        return out
    sd = float(srs.std(ddof=1))
    sr0 = expected_max_sharpe(N, sd)
    out["expected_max_sharpe"] = sr0 * np.sqrt(periods)
    out["expected_max_sharpe_simple"] = sd * np.sqrt(2 * np.log(N)) * np.sqrt(periods)
    sr = best_sr / np.sqrt(periods)
    g3 = float(skew) if _finite(skew) else 0.0
    g4 = (float(excess_kurt) if _finite(excess_kurt) else 0.0) + 3.0
    var = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if var <= 0:
        return out
    out["dsr"] = NormalDist().cdf((sr - sr0) * np.sqrt(n_obs - 1) / np.sqrt(var))
    return out
