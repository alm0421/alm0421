"""A small, safe expression language for entry/exit rules.

Expressions are Python-like and evaluated per ticker on daily bars, e.g.

    down_days >= 5
    rsi(2) < 10 and close > sma(close, 200)
    ret(1) <= -0.03 and sym("SPY").close > sma(sym("SPY").close, 200)

`and`, `or`, `not` and chained comparisons (0.1 < ibs < 0.3) work on whole
series.  All prices are split- and dividend-adjusted.  See VARIABLES and
FUNCTIONS below (or `python -m backtester --help-expr`) for the vocabulary.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import re
from typing import Any, Callable

import numpy as np

from . import calendar as _cal
import pandas as pd

from . import data

# ---------------------------------------------------------------- indicators


def _s(x: Any, like: pd.Series) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x, index=like.index, dtype=float)


def streak(close: pd.Series, direction: int) -> pd.Series:
    """Number of consecutive closes strictly down (direction=-1) or up (+1)."""
    chg = np.sign(close.diff()).fillna(0).to_numpy()
    hit = chg == direction
    out = np.zeros(len(hit), dtype=float)
    run = 0
    for i, h in enumerate(hit):
        run = run + 1 if h else 0
        out[i] = run
    return pd.Series(out, index=close.index)


def wilder(x: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average (RMA) as TradingView computes it: seeded with the simple average of the
    first n values, then x_t / n + prev * (n - 1) / n."""
    n = int(n)
    v = x.to_numpy(dtype=float)
    ok = np.isfinite(v)
    run = np.convolve(ok.astype(int), np.ones(n, int), "full")[: len(v)] if len(v) else ok
    first = np.flatnonzero(run >= n)
    if not len(first):
        return pd.Series(np.nan, index=x.index)
    i = int(first[0])
    y = pd.Series(np.nan, index=x.index)
    seed = pd.Series(v[i:].copy(), index=x.index[i:])
    seed.iloc[0] = v[i - n + 1: i + 1].mean()
    y.iloc[i:] = seed.ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    return y


def ema_tv(x: pd.Series, n: int) -> pd.Series:
    """EMA as TradingView's ta.ema: seeded with the simple average of the first n values."""
    n = int(n)
    v = x.to_numpy(dtype=float)
    ok = np.isfinite(v)
    run = np.convolve(ok.astype(int), np.ones(n, int), "full")[: len(v)] if len(v) else ok
    first = np.flatnonzero(run >= n)
    y = pd.Series(np.nan, index=x.index)
    if not len(first):
        return y
    i = int(first[0])
    seed = pd.Series(v[i:].copy(), index=x.index[i:])
    seed.iloc[0] = v[i - n + 1: i + 1].mean()
    y.iloc[i:] = seed.ewm(alpha=2 / (n + 1), adjust=False).mean().to_numpy()
    return y


def rsi_wilder(x: pd.Series, n: int) -> pd.Series:
    # numpy for the element-wise steps (the same values as Series.diff / clip / where, a lot faster)
    v = x.to_numpy(dtype=float)
    d = np.full(len(v), np.nan)
    if len(v) > 1:
        d[1:] = v[1:] - v[:-1]
    with np.errstate(invalid="ignore"):
        up = wilder(pd.Series(np.where(np.isnan(d), np.nan, np.maximum(d, 0.0)), index=x.index), n).to_numpy()
        dn = wilder(pd.Series(np.where(np.isnan(d), np.nan, -np.minimum(d, 0.0)), index=x.index), n).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + up / dn)
    out = np.where(dn != 0, out, 100.0)
    out[np.isnan(d)] = np.nan
    return pd.Series(out, index=x.index)


def wma_tv(x: pd.Series, n: int) -> pd.Series:
    """Linearly weighted moving average (TradingView's ta.wma): weights 1..n, the latest bar weighted n."""
    n = int(n)
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        w = np.arange(1, n + 1, dtype=float)
        out[n - 1:] = np.lib.stride_tricks.sliding_window_view(v, n) @ w / w.sum()
    return pd.Series(out, index=x.index)


def hma_tv(x: pd.Series, n: int) -> pd.Series:
    """Hull moving average as TradingView's ta.hma: wma(2 * wma(x, n / 2) - wma(x, n), floor(sqrt(n))), with the
    half length rounded down."""
    n = int(n)
    return wma_tv(2 * wma_tv(x, max(n // 2, 1)) - wma_tv(x, n), max(int(np.floor(np.sqrt(n))), 1))


def linreg_tv(x: pd.Series, n: int, offset: int = 0) -> pd.Series:
    """TradingView's ta.linreg(x, n, offset): the least-squares line through the last n values, evaluated
    `offset` bars before the latest (intercept + slope * (n - 1 - offset)). offset 0 is the least-squares
    moving average. Only the last n values are used, so it is causal for any offset."""
    n = int(n)
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        w = np.lib.stride_tricks.sliding_window_view(v, n)
        t = np.arange(n, dtype=float)
        tm = t.mean()
        ym = w.mean(axis=1)
        den = ((t - tm) ** 2).sum()
        slope = ((w - ym[:, None]) @ (t - tm)) / den if den > 0 else np.zeros(len(w))
        intercept = ym - slope * tm
        out[n - 1:] = intercept + slope * (n - 1 - offset)
    return pd.Series(out, index=x.index)


def alma_tv(x: pd.Series, n: int = 9, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """Arnaud Legoux moving average as TradingView's ta.alma: Gaussian weights exp(-(i - m)^2 / (2 s^2)) over the
    window (i = 0 the oldest bar), m = offset * (n - 1), s = n / sigma."""
    n = int(n)
    m = offset * (n - 1)
    sd = n / sigma
    w = np.exp(-((np.arange(n) - m) ** 2) / (2 * sd * sd))
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        out[n - 1:] = np.lib.stride_tricks.sliding_window_view(v, n) @ w / w.sum()
    return pd.Series(out, index=x.index)


def kama_tv(x: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman's adaptive moving average, as TradingView's built-in KAMA script: efficiency ratio
    er = |x - x[n]| / sum(|x - x[1]|, n) (0 when the sum is 0), alpha = (er * (2/(fast+1) - 2/(slow+1)) +
    2/(slow+1))^2, kama = alpha * x + (1 - alpha) * kama[1], starting from x on the first bar with a value."""
    n = int(n)
    v = x.to_numpy(dtype=float)
    mom = np.abs(x - x.shift(n)).to_numpy()
    vol = x.diff().abs().rolling(n, min_periods=n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(vol != 0, mom / vol, 0.0)
    er[np.isnan(mom) | np.isnan(vol)] = np.nan
    fa, sa = 2 / (fast + 1), 2 / (slow + 1)
    alpha = (er * (fa - sa) + sa) ** 2
    out = np.full(len(v), np.nan)
    prev = np.nan
    for i in range(len(v)):
        if np.isnan(alpha[i]) or np.isnan(v[i]):
            prev = np.nan if np.isnan(alpha[i]) else prev
            continue
        base = v[i] if np.isnan(prev) else prev
        prev = alpha[i] * v[i] + (1 - alpha[i]) * base
        out[i] = prev
    return pd.Series(out, index=x.index)


def pivot_tv(x: pd.Series, left: int, right: int, high: bool) -> pd.Series:
    """TradingView's ta.pivothigh / ta.pivotlow: on bar t, the value of bar t - right if that bar is strictly
    above (below) the `left` bars before it and the `right` bars after it, else NaN. A pivot is only known once
    its `right` bars have closed, so the value appears `right` bars after the pivot bar (never earlier)."""
    left, right = int(left), int(right)
    v = x.to_numpy(dtype=float)
    n = left + right + 1
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        w = np.lib.stride_tricks.sliding_window_view(v, n)      # oldest .. newest; the candidate at index left
        mid = w[:, left]
        others = np.delete(w, left, axis=1)
        ok = (mid[:, None] > others).all(axis=1) if high else (mid[:, None] < others).all(axis=1)
        ok &= np.isfinite(mid) & np.isfinite(others).all(axis=1)
        out[n - 1:] = np.where(ok, mid, np.nan)
    return pd.Series(out, index=x.index)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin Ashi bars as TradingView draws them: ha_close = (o + h + l + c) / 4, ha_open = (previous ha_open +
    previous ha_close) / 2 (the first bar: (open + close) / 2), ha_high = max(high, ha_open, ha_close),
    ha_low = min(low, ha_open, ha_close)."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    hc = (o + h + l + c) / 4
    ho = np.full(len(o), np.nan)
    for i in range(len(o)):
        if i == 0 or np.isnan(ho[i - 1]) or np.isnan(hc[i - 1]):
            ho[i] = (o[i] + c[i]) / 2
        else:
            ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    hh = np.fmax(h, np.fmax(ho, hc))
    hl = np.fmin(l, np.fmin(ho, hc))
    return pd.DataFrame({"ha_open": ho, "ha_high": hh, "ha_low": hl, "ha_close": hc}, index=df.index)


def is_crypto(ticker: str | None) -> bool:
    """A crypto pair (BTC-USD): its daily bar closes at 00:00 UTC (8pm New York), after the US close."""
    return bool(ticker) and str(ticker).upper().endswith("-USD")


CRYPTO_LAG_NOTE = ("A rule reads a crypto pair from a US-market ticker: a crypto day closes at 00:00 UTC (8pm New "
                   "York), about 4 hours after the US close, so the rule uses the crypto bar of the previous day "
                   "(the latest one complete at the US close).")


# Cboe volatility indexes are calculated to 4:15pm New York: their "close" is not known at the 4:00pm stock close
LATE_CLOSE = {"^VIX", "^VIX3M", "^VIX9D", "^VIX6M", "^VIX1D", "^VVIX", "^VXN", "^VXD", "^RVX", "^OVX", "^GVZ", "^VXEEM",
              "^SKEW"}
LATE_CLOSE_NOTE = ("{t} closes at 4:15pm New York, after the 4:00pm close this rule trades at: a same-day close fill "
                   "uses {t}'s previous close (the latest known at 4:00pm). Fill at the next open to use that day's value.")


def is_late_close(ticker: str | None) -> bool:
    return bool(ticker) and str(ticker).upper() in LATE_CLOSE


def adjusted_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The bars on a total-return (dividend-adjusted) basis, built causally.

    A total-return index starts at the first close and grows by (close_t + dividend_t) / close_(t-1) each bar;
    open/high/low/close are all scaled by index/close, so each bar keeps its shape. Volume is unchanged, the
    dividend column becomes 0 (it is already in the prices) and adj_close (`tr`) is kept. Unlike a back-adjusted
    series (Yahoo's adjusted close), which rescales the whole history after each new dividend, the value on a
    date never changes when later data arrives, so indicators on it cannot see future dividends. Levels start at
    the first bar's quoted close; returns, RSI, moving-average crossings and other ratios match Composer's and
    Portfolio Visualizer's adjusted-price indicators. Applying it twice changes nothing."""
    if "close" not in df or "dividend" not in df or not len(df):
        return df
    c = df["close"].astype(float)
    d = df["dividend"].astype(float).fillna(0.0)
    if not (d != 0).any():
        return df
    prev = c.ffill().shift(1)
    g = ((c + d) / prev).where(c.notna() & prev.notna() & (prev > 0), 1.0)
    first = c.first_valid_index()
    if first is None:
        return df
    tri = g.cumprod() / g.loc[first] * float(c.loc[first])
    f = (tri / c).where(c.notna())
    out = df.copy()
    for k in ("open", "high", "low", "close"):
        if k in out:
            out[k] = df[k] * f
    out["dividend"] = 0.0
    return out


class Bars:
    """Price fields of another ticker aligned to the current ticker's dates.

    `lag` (a crypto pair read from a US-market ticker) uses each date's previous bar of the other ticker."""

    def __init__(self, df: pd.DataFrame, index: pd.Index, lag: bool = False):
        if lag:
            df = df.shift(1).iloc[1:]
        a = df.reindex(index.union(df.index)).ffill().reindex(index)
        self.open, self.high, self.low, self.close = a["open"], a["high"], a["low"], a["close"]
        if "open_ok" in df:  # an open that was never quoted is unknown, not the close
            ok = df["open_ok"].reindex(index).fillna(False).astype(bool)
            self.open = self.open.where(ok)
        self.volume = a["volume"]
        self.tr = a["adj_close"] if "adj_close" in a else a["close"]


MONTH_BARS = 21     # a month of trading days: "12 month return" is ret(252)


def month_end_flags(idx: pd.DatetimeIndex) -> np.ndarray:
    """True on each date that is the last session of its month: the next date of `idx` (after the last one, the next
    NYSE session) falls in another month. Calendar knowledge only, no prices, so it is causal."""
    if not len(idx):
        return np.zeros(0, bool)
    nxt = list(idx[1:]) + [_cal.next_sessions(idx[-1])[0]]
    per = idx.to_period("M")
    return np.asarray(per != pd.DatetimeIndex(nxt).to_period("M"))


def calendar_month_return(x: pd.Series, months: int, skip: int = 0) -> pd.Series:
    """Return over `months` calendar months, month-end to month-end, as Portfolio Visualizer and Antonacci measure
    it: on every day, the change from the month-end `months` months before the last completed month-end (that day
    itself when it is a month-end) to that month-end. Causal: a month's value is known on its last session.
    skip: end `skip` month-ends before the last completed one ("12 month return skipping the last month" in
    calendar months = calendar_month_return(x, 11, 1): the month-end price 1 month ago / 12 months ago - 1)."""
    v = x.to_numpy(dtype=float)
    pos = np.flatnonzero(month_end_flags(x.index))
    out = np.full(len(v), np.nan)
    if not len(pos):
        return pd.Series(out, index=x.index)
    per = x.index[pos].to_period("M")
    me = pd.Series(v[pos], index=per)
    me = me[~me.index.duplicated(keep="last")]
    prev = me.reindex(per - (months + skip)).to_numpy()
    r = (me.reindex(per - skip).to_numpy() if skip else v[pos]) / prev - 1
    # each day carries its last completed month-end's value (NaN there stays NaN, it is not filled from before)
    last = np.full(len(v), -1)
    last[pos] = np.arange(len(pos))
    last = np.maximum.accumulate(last)
    ok = last >= 0
    out[ok] = r[last[ok]]
    return pd.Series(out, index=x.index)


class Namespace(dict):
    """Evaluation namespace for one ticker; derived variables are lazy."""

    def __init__(self, df: pd.DataFrame, extra: dict[str, Any] | None = None, ticker: str | None = None,
                 price_basis: str = "quoted", month_lookbacks: str = "trading", close_fill: bool = False):
        """price_basis "quoted": prices as quoted (split-adjusted; TradingView). "adjusted": open/high/low/close on
        a total-return basis (dividends reinvested, see adjusted_frame; Composer, Portfolio Visualizer), for this
        ticker and for sym() of any other. month_lookbacks "calendar": ret / tret / tbill_ret over a whole number of
        months (a multiple of 21 bars: 21, 63, 126, 252) are measured month-end to month-end over calendar months
        (calendar_month_return) instead of over that many trading days."""
        super().__init__()
        if price_basis not in ("quoted", "adjusted"):
            raise ValueError("price_basis must be 'quoted' or 'adjusted'")
        if month_lookbacks not in ("trading", "calendar"):
            raise ValueError("month_lookbacks must be 'trading' or 'calendar'")
        self.price_basis = price_basis
        self.month_lookbacks = month_lookbacks
        self.quoted_df = df        # the bars as quoted, for quoted(<expr>) on an adjusted basis
        if price_basis == "adjusted":
            df = adjusted_frame(df)
        self.df = df
        self.ticker = ticker
        self.close_fill = close_fill   # rules acted on at this bar's own close: late-closing series (VIX) lag a day
        self.notes: list[str] = []
        c = df["close"]
        self.update({
            "open": df["open"].where(df["open_ok"]) if "open_ok" in df else df["open"],
            "high": df["high"], "low": df["low"], "close": c,
            "volume": df["volume"], "price": c,
            "tr": df["adj_close"] if "adj_close" in df else c,
            "True": True, "False": False,
        })
        self.update(self._functions())
        if extra:
            self.update(extra)

    # lazily computed variables
    def __missing__(self, key: str) -> Any:
        df, c = self.df, self.df["close"]
        idx = df.index
        lazy: dict[str, Callable[[], Any]] = {
            "down_days": lambda: streak(c, -1),
            "up_days": lambda: streak(c, +1),
            "ibs": lambda: ((c - df["low"]) / (df["high"] - df["low"])).where(df["high"] > df["low"], 0.5),
            "gap": lambda: self["open"] / c.shift(1) - 1,
            "range": lambda: df["high"] / df["low"] - 1,
            "change": lambda: c.pct_change(fill_method=None),
            "dow": lambda: pd.Series(idx.dayofweek, index=idx),
            "day": lambda: pd.Series(idx.day, index=idx),
            "month": lambda: pd.Series(idx.month, index=idx),
            "year": lambda: pd.Series(idx.year, index=idx),
            "dollar_volume": lambda: c * df["volume"],
            "hl2": lambda: (df["high"] + df["low"]) / 2,
            "hlc3": lambda: (df["high"] + df["low"] + c) / 3,
            "ohlc4": lambda: (df["open"] + df["high"] + df["low"] + c) / 4,
            "hlcc4": lambda: (df["high"] + df["low"] + 2 * c) / 4,
            "true_range": lambda: pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(),
                                             (df["low"] - c.shift()).abs()], axis=1).max(axis=1),
            "ha_open": lambda: self._ha()["ha_open"], "ha_high": lambda: self._ha()["ha_high"],
            "ha_low": lambda: self._ha()["ha_low"], "ha_close": lambda: self._ha()["ha_close"],
            "market_cap": lambda: self._market_cap(),
            "trading_day_of_month": lambda: pd.Series(idx.to_period("M"), index=idx).groupby(idx.to_period("M")).cumcount() + 1,
            # scheduled NYSE sessions left this month AFTER today (0 on the month's last session), on the schedule
            # as published that day (calendar.py: no hindsight about unscheduled closures); known at the open
            "trading_days_left_in_month": lambda: pd.Series(_cal.scheduled_sessions_left(idx, "M"), index=idx),
        }
        if key in lazy:
            v = lazy[key]()
            self[key] = v
            return v
        raise NameError(f"unknown name {key!r} in expression (see --help-expr)")

    def _ha(self) -> pd.DataFrame:
        ha = getattr(self, "_ha_frame", None)
        if ha is None:
            ha = self._ha_frame = heikin_ashi(self.df)
        return ha

    def _market_cap(self) -> pd.Series:
        # quoted close x point-in-time shares (data.market_cap), never the total-return price basis
        idx = self.df.index
        mc = data.market_cap(self.ticker) if self.ticker else pd.Series(dtype=float)
        if data.MCAP_NOTE not in self.notes:
            self.notes.append(data.MCAP_NOTE)
        if mc.empty:
            return pd.Series(np.nan, index=idx)
        return mc.reindex(idx)

    def _functions(self) -> dict[str, Callable]:
        df = self.df
        c, hi, lo, vol = df["close"], df["high"], df["low"], df["volume"]

        def pick(args, default_x, default_n):
            """Accept f(n), f(x, n), f(n, x) or f(). A series where the lookback belongs is an error, not a swap.

            The lookback must be a plain Python number (a literal in the rule), never a computed scalar such as
            abs(1) or a series: with two arguments exactly one must be a series. So the static open-time check
            (open_safe), which reads the same shapes off the rule's text, cannot disagree with what runs."""
            x, n = default_x, default_n
            series = [a for a in args if isinstance(a, pd.Series)]
            if len(series) > 1 or len(args) > 2:
                raise ValueError("an indicator takes one series and one lookback, e.g. sma(close, 20); the lookback "
                                 "must be a fixed whole number, not a series (got more than one series or argument)")
            if len(args) == 2 and not series:
                raise ValueError("an indicator with two arguments needs a series and a lookback, e.g. sma(close, 20) "
                                 "(got two numbers)")
            for a in args:
                if isinstance(a, pd.Series):
                    x = a
                else:
                    n = a
            n = lookback_number(n)
            if n < 1:
                raise ValueError("lookback periods must be at least 1")
            return x, n

        def lookback_number(n):
            if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, float)) or isinstance(n, (np.integer, np.floating)):
                raise ValueError(f"lookback periods must be whole numbers written in the rule, e.g. sma(close, 20) "
                                 f"(got {type(n).__name__})")
            if isinstance(n, float) and not float(n).is_integer():
                raise ValueError(f"lookback periods must be whole numbers (got {n})")
            return int(n)

        def series_arg(x, f):
            if not isinstance(x, pd.Series):
                raise ValueError(f"{f}() needs a series such as close or rsi(close, 2), not a single number")
            return x

        def nonneg(n):
            if isinstance(n, pd.Series) or isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, float)) \
                    or isinstance(n, (np.integer, np.floating)):
                raise ValueError("offsets must be fixed whole numbers written in the rule, e.g. ref(close, 5), not a "
                                 "series or a calculation")
            if isinstance(n, float) and not float(n).is_integer():
                raise ValueError(f"offsets must be whole numbers (got {n})")
            n = int(n)
            if n < 0:
                raise ValueError("negative offsets would look into the future and are not allowed")
            return n

        def sma(*a):
            x, n = pick(a, c, 20)
            return x.rolling(n, min_periods=n).mean()

        def ema(*a):
            x, n = pick(a, c, 20)
            return ema_tv(x, n)

        def rma(*a):  # Wilder's moving average
            x, n = pick(a, c, 14)
            return wilder(x, n)

        def wma(*a):
            x, n = pick(a, c, 20)
            w = np.arange(1, n + 1, dtype=float)
            return x.rolling(n, min_periods=n).apply(lambda v: np.dot(v, w) / w.sum(), raw=True)

        def highest(*a):
            x, n = pick(a, hi, 20)
            return x.rolling(n, min_periods=n).max()

        def lowest(*a):
            x, n = pick(a, lo, 20)
            return x.rolling(n, min_periods=n).min()

        def stdev(*a):
            x, n = pick(a, c, 20)
            return x.rolling(n, min_periods=n).std(ddof=0)  # population, as TradingView's ta.stdev

        def zscore(*a):
            x, n = pick(a, c, 20)
            return (x - x.rolling(n).mean()) / x.rolling(n).std(ddof=0)

        def ref(x, n=1):
            n0 = n
            if isinstance(x, pd.Series) and id(x) in cal_made and cal_made[id(x)][0] is x:
                # a calendar-month return lagged by whole months (12-1 momentum): month-ends, not sessions
                nn = nonneg(n0)
                if nn % MONTH_BARS == 0:
                    _, src, months, skip = cal_made[id(x)]
                    return calendar_month_return(src, months, skip + nn // MONTH_BARS)
            x = _s(x, c)
            n = nonneg(n)
            if x.dtype == bool:
                return x.shift(n, fill_value=False)
            return x.shift(n)

        cal_months = self.month_lookbacks == "calendar"
        cal_made: dict = {}     # id -> (series, source, months, skip) of the calendar-month returns made here

        def cal_ret(x, months):
            out = calendar_month_return(x, months)
            cal_made[id(out)] = (out, x, months, 0)
            return out

        def ret(*a):
            x, n = pick(a, c, 1)
            if cal_months and n % MONTH_BARS == 0:
                return cal_ret(x, n // MONTH_BARS)
            return x / x.shift(n) - 1

        rsi_memo: dict = {}
        c_ns = self.get("close")      # the namespace's `close` (the same bars as c)

        def rsi(*a):
            x, n = pick(a, c, 14)
            if x is not c and x is not c_ns:
                return rsi_wilder(x, n)
            if n not in rsi_memo:      # of the close: computed once per ticker (entry and exit rules often share it)
                rsi_memo[n] = rsi_wilder(x, n)
            return rsi_memo[n].copy()

        trs = df["adj_close"] if "adj_close" in df else c

        def tret(*a):
            """Total return (dividends reinvested) over n bars."""
            x, n = pick(a, trs, 1)
            if cal_months and n % MONTH_BARS == 0:
                return cal_ret(x, n // MONTH_BARS)
            return x / x.shift(n) - 1

        def tbill_ret(n=252):
            """Compounded 3-month T-bill return over the last n bars (from 1954)."""
            r = data.tbill_rate()
            if r.empty:
                return pd.Series(0.0, index=c.index)
            rr = r.reindex(c.index.union(r.index)).ffill().reindex(c.index).fillna(0.0)
            idx = (1 + rr / 252).cumprod()
            if cal_months and int(n) % MONTH_BARS == 0:
                return cal_ret(idx, int(n) // MONTH_BARS)
            return idx / idx.shift(int(n)) - 1

        def _known(s: pd.Series) -> pd.Series:
            """A macro series dated by when it became known, held forward onto this ticker's bars (NaN before)."""
            if s is None or s.empty:
                return pd.Series(np.nan, index=c.index)
            return s.reindex(c.index.union(s.index)).ffill().reindex(c.index)

        def cape():
            """Shiller's cyclically adjusted P/E (P/E10), lagged data.CAPE_LAG_MONTHS months to be point in time."""
            return _known(data.shiller_known("cape"))

        def earnings_yield():
            """The cyclically adjusted earnings yield 1 / CAPE (0.04 = 4%), point in time like cape()."""
            return 1.0 / cape()

        def cape_pct(years=0):
            """CAPE percentile (0..1) among the values known so far: all history (years=0) or the last N years."""
            return _known(data.cape_percentile(int(years)))

        def treasury_10y():
            """The 10-year Treasury yield (decimal) as of each day's close (FRED DGS10; monthly GS10 before 1962)."""
            return _known(data.treasury_10y())

        def max_drawdown(*a):
            """Largest peak-to-trough fall within the last n bars, as a positive fraction."""
            x, n = pick(a, trs, 63)
            v = x.to_numpy(dtype=float)
            out = np.full(len(v), np.nan)
            if len(v) >= n:
                w = np.lib.stride_tricks.sliding_window_view(v, n)
                peak = np.maximum.accumulate(w, axis=1)
                out[n - 1:] = -(w / peak - 1).min(axis=1)
            return pd.Series(out, index=x.index)

        def ma_return(*a):
            """Average daily (total) return over n bars."""
            x, n = pick(a, trs, 20)
            return x.pct_change(fill_method=None).rolling(n, min_periods=n).mean()

        def stdev_return(*a):
            """Standard deviation of daily (total) returns over n bars (not annualised)."""
            x, n = pick(a, trs, 20)
            return x.pct_change(fill_method=None).rolling(n, min_periods=n).std()

        def true_range():
            h_, l_, c_ = hi.to_numpy(dtype=float), lo.to_numpy(dtype=float), c.to_numpy(dtype=float)
            pc = np.full(len(c_), np.nan)
            pc[1:] = c_[:-1]
            with np.errstate(invalid="ignore"):   # the largest of the three, ignoring missing ones (as DataFrame.max)
                tr = np.fmax(np.fmax(h_ - l_, np.abs(h_ - pc)), np.abs(l_ - pc))
            return pd.Series(tr, index=c.index)

        def atr(n=14):
            n = int(n)
            return wilder(true_range(), n)

        def natr(n=14):
            return atr(n) / c

        def volatility(n=20, x=None):
            base = c if x is None else _s(x, c)
            if isinstance(n, pd.Series):
                base, n = n, (20 if x is None else x)
            return base.pct_change(fill_method=None).rolling(int(n)).std() * np.sqrt(252)

        def bb_upper(n=20, k=2.0, x=None):
            base = c if x is None else series_arg(x, "bb_upper")
            return sma(base, n) + k * stdev(base, n)

        def bb_lower(n=20, k=2.0, x=None):
            base = c if x is None else series_arg(x, "bb_lower")
            return sma(base, n) - k * stdev(base, n)

        def pct_rank(*a):
            x, n = pick(a, c, 252)
            return x.rolling(n, min_periods=n).rank(pct=True)

        def drawdown(*a):
            """x / highest(x, n) - 1 (<= 0); n defaults to all history."""
            x, n = c, None
            if len(a) > 2 or sum(isinstance(v, pd.Series) for v in a) > 1 or (len(a) == 2 and not any(isinstance(v, pd.Series) for v in a)):
                raise ValueError("drawdown takes a series and an optional lookback, e.g. drawdown(close, 252)")
            for v in a:
                if isinstance(v, pd.Series):
                    x = v
                else:
                    n = lookback_number(v)
            peak = x.cummax() if n is None else x.rolling(n, min_periods=1).max()
            return x / peak - 1

        def macd(fast=12, slow=26, x=None):
            base = c if x is None else _s(x, c)
            return ema_tv(base, int(fast)) - ema_tv(base, int(slow))

        def macd_signal(fast=12, slow=26, sig=9, x=None):
            return ema_tv(macd(fast, slow, x), int(sig))

        def macd_hist(fast=12, slow=26, sig=9, x=None):
            return macd(fast, slow, x) - macd_signal(fast, slow, sig, x)

        def ppo(fast=12, slow=26, x=None):
            """Percentage price oscillator, in percent: (EMA(fast) - EMA(slow)) / EMA(slow) * 100 (TradingView's
            EMAs, as macd)."""
            base = c if x is None else _s(x, c)
            s_ = ema_tv(base, int(slow))
            return (ema_tv(base, int(fast)) - s_) / s_ * 100

        def ppo_signal(fast=12, slow=26, sig=9, x=None):
            return ema_tv(ppo(fast, slow, x), int(sig))

        def stoch_k(n=14, smooth=3):
            ll, hh = lo.rolling(int(n)).min(), hi.rolling(int(n)).max()
            k = 100 * (c - ll) / (hh - ll)
            return k.rolling(int(smooth)).mean() if int(smooth) > 1 else k

        def stoch_d(n=14, smooth=3, d=3):
            return stoch_k(n, smooth).rolling(int(d)).mean()

        def stoch(x, h_, l_, n=14):
            """TradingView's ta.stoch(source, high, low, length): 100 * (x - lowest(l, n)) / (highest(h, n) - lowest(l, n))."""
            x, h_, l_ = (series_arg(v, "stoch") for v in (x, h_, l_))
            n = lookback_number(n)
            ll, hh = l_.rolling(n).min(), h_.rolling(n).max()
            with np.errstate(divide="ignore", invalid="ignore"):
                return 100 * (x - ll) / (hh - ll)

        def stoch_rsi_k(smooth_k=3, rsi_n=14, stoch_n=14, x=None):
            """TradingView's Stochastic RSI %K: sma(ta.stoch(r, r, r, stoch_n), smooth_k) with r = rsi(x, rsi_n)."""
            r = rsi_wilder(c if x is None else series_arg(x, "stoch_rsi_k"), lookback_number(rsi_n))
            k = stoch(r, r, r, stoch_n)
            sk = lookback_number(smooth_k)
            return k.rolling(sk).mean() if sk > 1 else k

        def stoch_rsi_d(smooth_k=3, smooth_d=3, rsi_n=14, stoch_n=14, x=None):
            """TradingView's Stochastic RSI %D: the smooth_d-bar SMA of %K (defaults 3, 3, 14, 14 as the built-in)."""
            return stoch_rsi_k(smooth_k, rsi_n, stoch_n, x).rolling(lookback_number(smooth_d)).mean()

        def _dm(n):
            n = int(n)
            up, dn = hi.diff(), -lo.diff()
            pdm = up.where((up > dn) & (up > 0), 0.0)
            mdm = dn.where((dn > up) & (dn > 0), 0.0)
            trn = wilder(true_range(), n)
            pdi = 100 * wilder(pdm, n) / trn
            mdi = 100 * wilder(mdm, n) / trn
            return pdi, mdi

        def plus_di(n=14):
            return _dm(n)[0]

        def minus_di(n=14):
            return _dm(n)[1]

        def adx(n=14):
            pdi, mdi = _dm(n)
            dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
            return wilder(dx, int(n))

        def cci(*a):
            """TradingView's ta.cci: (x - sma(x, n)) / (0.015 * mean absolute deviation); x defaults to hlc3."""
            x, n = pick(a, (hi + lo + c) / 3, 20)
            v = x.to_numpy(dtype=float)
            md = np.full(len(v), np.nan)
            if len(v) >= n:
                w = np.lib.stride_tricks.sliding_window_view(v, n)
                md[n - 1:] = np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)
            m = x.rolling(n, min_periods=n).mean()
            return (x - m) / (0.015 * pd.Series(md, index=x.index))

        def willr(n=14):
            hh, ll = hi.rolling(int(n)).max(), lo.rolling(int(n)).min()
            return -100 * (hh - c) / (hh - ll)

        def obv():
            return (np.sign(c.diff()).fillna(0) * vol).cumsum()

        def mfi(*a):
            """TradingView's ta.mfi: money flow index of x (default hlc3) and volume over n bars."""
            tp, n = pick(a, (hi + lo + c) / 3, 14)
            mf = tp * vol
            pos = mf.where(tp.diff() > 0, 0.0).rolling(n).sum()
            neg = mf.where(tp.diff() < 0, 0.0).rolling(n).sum()
            return 100 - 100 / (1 + pos / neg)

        def vwap(n=20):
            """Rolling volume-weighted average of the typical price (daily-bar VWAP proxy)."""
            tp = (hi + lo + c) / 3
            return (tp * vol).rolling(int(n)).sum() / vol.rolling(int(n)).sum()

        def donchian_upper(n=20):
            return hi.rolling(int(n)).max()

        def donchian_lower(n=20):
            return lo.rolling(int(n)).min()

        # TradingView's ta.kc: EMA of the close +/- k x EMA of the true range
        def keltner_upper(n=20, k=2.0):
            return ema(c, n) + k * ema(true_range(), n)

        def keltner_lower(n=20, k=2.0):
            return ema(c, n) - k * ema(true_range(), n)

        def _supertrend(n=10, k=3.0):
            """(line, direction) bar for bar as TradingView's ta.supertrend(k, n): direction +1 = downtrend (the
            line is the upper band), -1 = uptrend (the lower band). The first bar with an ATR starts in a downtrend,
            as Pine's reference implementation does (direction := 1 while atr[1] is na)."""
            a = atr(n).to_numpy()
            mid = ((hi + lo) / 2).to_numpy()
            cl = c.to_numpy()
            ub, lb = mid + k * a, mid - k * a
            st = np.full(len(cl), np.nan)
            dr = np.full(len(cl), np.nan)
            up = False
            fub, flb = np.nan, np.nan
            first = True
            for i in range(len(cl)):
                if np.isnan(a[i]):
                    continue
                fub = ub[i] if np.isnan(fub) or ub[i] < fub or cl[i - 1] > fub else fub
                flb = lb[i] if np.isnan(flb) or lb[i] > flb or cl[i - 1] < flb else flb
                if first:
                    up, first = False, False
                elif up and cl[i] < flb:
                    up = False
                elif not up and cl[i] > fub:
                    up = True
                st[i] = flb if up else fub
                dr[i] = -1.0 if up else 1.0
            return pd.Series(st, index=c.index), pd.Series(dr, index=c.index)

        def supertrend(n=10, k=3.0):
            """Supertrend line (below price in an uptrend, above in a downtrend)."""
            return _supertrend(n, k)[0]

        def supertrend_dir(n=10, k=3.0):
            """Supertrend direction as TradingView returns it: -1 in an uptrend, +1 in a downtrend."""
            return _supertrend(n, k)[1]

        def sar(step=0.02, max_step=0.2, inc=None):
            """Parabolic SAR, bar for bar as TradingView's ta.sar(start, inc, max) reference implementation: the
            acceleration starts at `step` and grows by `inc` (default: step) up to max_step."""
            inc = step if inc is None else inc
            h, l, cl = hi.to_numpy(), lo.to_numpy(), c.to_numpy()
            n = len(h)
            out = np.full(n, np.nan)
            if n < 2:
                return pd.Series(out, index=c.index)
            res = mm = acc = np.nan
            below = False
            for i in range(1, n):
                first_bar = False
                if i == 1:
                    if cl[1] > cl[0]:
                        below, mm, res = True, h[1], l[0]
                    else:
                        below, mm, res = False, l[1], h[0]
                    first_bar, acc = True, step
                res = res + acc * (mm - res)
                if below:
                    if res > l[i]:
                        first_bar, below = True, False
                        res, mm, acc = max(h[i], mm), l[i], step
                else:
                    if res < h[i]:
                        first_bar, below = True, True
                        res, mm, acc = min(l[i], mm), h[i], step
                if not first_bar:
                    if below and h[i] > mm:
                        mm, acc = h[i], min(acc + inc, max_step)
                    elif not below and l[i] < mm:
                        mm, acc = l[i], min(acc + inc, max_step)
                if below:
                    res = min(res, l[i - 1]) if i < 2 else min(res, l[i - 1], l[i - 2])
                else:
                    res = max(res, h[i - 1]) if i < 2 else max(res, h[i - 1], h[i - 2])
                out[i] = res
            return pd.Series(out, index=c.index)

        def params(a, f, default_x, defaults):
            """f(x, p1, p2, ...) or f(p1, p2, ...): at most one series, first; the rest numbers written in the rule."""
            a = list(a)
            x = default_x
            if a and isinstance(a[0], pd.Series):
                x = a.pop(0)
            if any(isinstance(v, pd.Series) for v in a):
                raise ValueError(f"{f}() takes one series first, then numbers, e.g. {f}(close, "
                                 f"{', '.join(str(d) for d in defaults)})")
            if len(a) > len(defaults):
                raise ValueError(f"{f}() takes at most {len(defaults)} numbers after the series")
            vals = []
            for i, d in enumerate(defaults):
                v = a[i] if i < len(a) else d
                if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float)):
                    raise ValueError(f"{f}(): parameters must be numbers written in the rule (got {v!r})")
                vals.append(v)
            return x, vals

        def length(v, f):
            if not float(v).is_integer() or int(v) < 1:
                raise ValueError(f"{f}(): lengths must be whole numbers of at least 1 (got {v})")
            return int(v)

        def hma(*a):
            x, n = pick(a, c, 9)
            return hma_tv(x, n)

        def vwma(*a):
            x, n = pick(a, c, 20)
            return (x * vol).rolling(n, min_periods=n).sum() / vol.rolling(n, min_periods=n).sum()

        def linreg(*a):
            x, (n, off) = params(a, "linreg", c, (20, 0))
            if not float(off).is_integer():
                raise ValueError("linreg(): the offset must be a whole number")
            return linreg_tv(x, length(n, "linreg"), int(off))

        def alma(*a):
            x, (n, off, sig) = params(a, "alma", c, (9, 0.85, 6))
            if sig <= 0:
                raise ValueError("alma(): sigma must be positive")
            return alma_tv(x, length(n, "alma"), float(off), float(sig))

        def kama(*a):
            x, (n, fast, slow) = params(a, "kama", c, (10, 2, 30))
            return kama_tv(x, length(n, "kama"), length(fast, "kama"), length(slow, "kama"))

        def _mid(n):
            n = int(n)
            return (hi.rolling(n, min_periods=n).max() + lo.rolling(n, min_periods=n).min()) / 2

        def tenkan(n=9):
            """Ichimoku conversion line: the midpoint of the n-bar high/low (TradingView default 9)."""
            return _mid(length(n, "tenkan"))

        def kijun(n=26):
            """Ichimoku base line: the midpoint of the n-bar high/low (default 26)."""
            return _mid(length(n, "kijun"))

        def senkou_a(conv=9, base=26, disp=26):
            """Ichimoku leading span A as TradingView draws it on the current bar: (tenkan + kijun) / 2 computed
            disp - 1 bars ago (the built-in plots it with offset = displacement - 1). Nothing from the future."""
            return ((tenkan(conv) + kijun(base)) / 2).shift(length(disp, "senkou_a") - 1)

        def senkou_b(n=52, disp=26):
            """Ichimoku leading span B on the current bar: the n-bar high/low midpoint computed disp - 1 bars ago."""
            return _mid(length(n, "senkou_b")).shift(length(disp, "senkou_b") - 1)

        def _aroon(n, up):
            n = length(n, "aroon")
            v = (hi if up else lo).to_numpy(dtype=float)
            out = np.full(len(v), np.nan)
            if len(v) >= n + 1:
                w = np.lib.stride_tricks.sliding_window_view(v, n + 1)[:, ::-1]    # newest first
                ago = np.argmax(w, axis=1) if up else np.argmin(w, axis=1)         # the most recent extreme
                val = 100.0 * (n - ago) / n
                val[~np.isfinite(w).all(axis=1)] = np.nan
                out[n:] = val
            return pd.Series(out, index=c.index)

        def aroon_up(n=14):
            """TradingView's Aroon: 100 * (n - bars since the highest high of the last n + 1 bars) / n."""
            return _aroon(n, True)

        def aroon_down(n=14):
            return _aroon(n, False)

        def aroon_osc(n=14):
            return _aroon(n, True) - _aroon(n, False)

        def cmf(n=20):
            """Chaikin money flow as TradingView: sum(money flow volume, n) / sum(volume, n), where money flow
            volume = ((2 close - low - high) / (high - low)) x volume (0 when high = low)."""
            n = length(n, "cmf")
            ad = (((2 * c - lo - hi) / (hi - lo)) * vol).where(hi != lo, 0.0)
            return ad.rolling(n, min_periods=n).sum() / vol.rolling(n, min_periods=n).sum()

        def pivothigh(*a):
            x, (left, right) = params(a, "pivothigh", hi, (5, 5))
            return pivot_tv(x, length(left, "pivothigh"), length(right, "pivothigh"), True)

        def pivotlow(*a):
            x, (left, right) = params(a, "pivotlow", lo, (5, 5))
            return pivot_tv(x, length(left, "pivotlow"), length(right, "pivotlow"), False)

        def avwap(anchor, x=None):
            """Anchored VWAP: sum(x * volume) / sum(volume) from the anchor date (x defaults to hlc3); NaN before."""
            if not isinstance(anchor, str):
                raise ValueError('avwap() takes the anchor date in quotes, e.g. avwap("2020-03-23")')
            try:
                d = pd.Timestamp(anchor)
            except (ValueError, TypeError):
                raise ValueError(f"avwap(): {anchor!r} is not a date (write e.g. avwap(\"2020-03-23\"))") from None
            src = (hi + lo + c) / 3 if x is None else series_arg(x, "avwap")
            on = pd.Series(c.index >= d, index=c.index)
            pv = (src * vol).where(on, 0.0).cumsum()
            vv = vol.where(on, 0.0).cumsum()
            return (pv / vv).where(on & (vv > 0))

        def na(x):
            """True where x has no value (TradingView's na())."""
            return _s(x, c).isna()

        def nz(x, y=0):
            """x with missing values replaced by y (TradingView's nz())."""
            return _s(x, c).fillna(y)

        def cross(a, b):
            """a crosses b in either direction (TradingView's ta.cross)."""
            return crossover(a, b) | crossunder(a, b)

        def crossover(a, b):
            a, b = _s(a, c), _s(b, c)
            return (a > b) & (a.shift() <= b.shift())

        def crossunder(a, b):
            a, b = _s(a, c), _s(b, c)
            return (a < b) & (a.shift() >= b.shift())

        def count(cond, n):
            return _s(cond, c).astype(float).rolling(int(n), min_periods=1).sum()

        def bars_since(cond):
            s = _s(cond, c).astype(bool).to_numpy()
            out = np.full(len(s), np.nan)
            last = None
            for i, v in enumerate(s):
                if v:
                    last = i
                out[i] = np.nan if last is None else i - last
            return pd.Series(out, index=c.index)

        def valuewhen(cond, x, n=0):
            """The value of x on the most recent bar where cond was true (n = 1: the one before that...),
            as TradingView's ta.valuewhen. Causal: each bar only sees conditions up to and including itself."""
            n = nonneg(n)
            cs = _s(cond, c)
            cs = (cs.fillna(0) if cs.dtype != bool else cs).astype(bool).to_numpy()
            xv = _s(x, c).astype(float).to_numpy()
            occ = np.flatnonzero(cs)
            k = np.cumsum(cs) - 1 - n            # index of the wanted occurrence, as of each bar
            out = np.full(len(cs), np.nan)
            ok = k >= 0
            out[ok] = xv[occ[k[ok]]]
            return pd.Series(out, index=c.index)

        def diff(*a):
            """x - x n bars ago (TradingView's ta.change / ta.mom): diff(), diff(n), diff(x), diff(x, n) or diff(n, x)."""
            series = [v for v in a if isinstance(v, pd.Series)]
            if len(a) > 2 or len(series) > 1 or (len(a) == 2 and not series):
                raise ValueError("diff takes a series and an offset, e.g. diff(close, 14); the offset must be a whole "
                                 "number written in the rule")
            base = series[0] if series else c
            nums = [v for v in a if not isinstance(v, pd.Series)]
            n = nonneg(nums[0]) if nums else 1
            return base - base.shift(n)

        def down_streak(x=None):
            return streak(c if x is None else series_arg(x, "down_streak"), -1)

        def up_streak(x=None):
            return streak(c if x is None else series_arg(x, "up_streak"), +1)

        def _period_close(freq):
            per = c.index.to_period(freq)
            last = pd.Series(c.index, index=c.index).groupby(per).transform("max")
            return per, last

        def _ends(base, freq):
            """Each complete period's closing value, indexed by the day it became known: the period's
            last bar if that was its last scheduled session, otherwise (an unscheduled closure, or a
            gap in the data) the first bar of the next period. A period still in progress at the data
            edge is left out."""
            if not len(base):
                return base
            per = base.index.to_period(freq)
            last_pos = pd.Series(np.arange(len(base)), index=base.index).groupby(per).max().to_numpy()
            sched = _cal.scheduled_period_end(base.index, freq)
            vals, when = [], []
            for p_ in last_pos:
                if sched[p_]:
                    k = p_
                elif p_ + 1 < len(base):
                    k = p_ + 1
                else:
                    continue
                vals.append(base.iloc[p_])
                when.append(base.index[k])
            out = pd.Series(vals, index=pd.DatetimeIndex(when), dtype=float)
            return out[~out.index.duplicated(keep="last")]

        def _periodic_sma(freq, n, x=None):
            base = c if x is None else _s(x, c)
            ends = _ends(base, freq)                         # value at each period's last bar
            m = ends.rolling(int(n), min_periods=int(n)).mean()
            # known from the period's last bar onward
            return m.reindex(base.index).ffill()

        def _periodic(freq, fn, n, x=None):
            """An indicator computed on completed weekly/monthly bars (each period's last close) and held
            flat until the next period ends: a period's value is known at its last bar's close."""
            base = c if x is None else _s(x, c)
            if isinstance(n, pd.Series):  # f(x, n) order
                base, n = n, (x if x is not None else 14)
            if isinstance(n, float) and not float(n).is_integer():
                raise ValueError(f"lookback periods must be whole numbers (got {n})")
            if int(n) < 1:
                raise ValueError("lookback periods must be at least 1")
            ends = _ends(base, freq)
            return fn(ends, int(n)).reindex(base.index).ffill()

        def _periodic_close(freq, x=None):
            base = c if x is None else _s(x, c)
            return _ends(base, freq).reindex(base.index).ffill()

        def is_period_end(freq):
            # the last scheduled session of the period, as known that day (on 2001-09-10 the week was
            # not over: 9/11 was a scheduled session)
            return pd.Series(_cal.scheduled_period_end(c.index, freq), index=c.index)

        def _tf(freq: str, src: str):
            """weekly(expr) / monthly(expr): evaluate `expr` on completed weekly/monthly bars built from
            the daily ones; each value is known from its period's last trading day onward."""
            per = df.index.to_period(freq)
            g = df.groupby(per)
            agg = pd.DataFrame({
                "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                "close": g["close"].last(), "volume": g["volume"].sum(),
                "dividend": g["dividend"].sum() if "dividend" in df else 0.0,
                "adj_close": g["adj_close"].last() if "adj_close" in df else g["close"].last(),
            })
            agg["open_ok"] = True
            # each period's bar becomes known on its last scheduled session, or - after an
            # unscheduled closure - on the next bar; the period in progress at the data edge is left out
            last_pos = pd.Series(np.arange(len(df)), index=df.index).groupby(per).max()
            sched = _cal.scheduled_period_end(df.index, freq)
            known = []
            for p_ in last_pos.reindex(agg.index).to_numpy():
                p_ = int(p_)
                known.append(df.index[p_] if sched[p_] else (df.index[p_ + 1] if p_ + 1 < len(df) else pd.NaT))
            agg.index = pd.DatetimeIndex(known)
            agg = agg[agg.index.notna()]
            agg = agg[~agg.index.duplicated(keep="last")]
            sub = Namespace(agg, ticker=self.ticker, price_basis=self.price_basis, close_fill=self.close_fill)
            val = evaluate_value(src, sub) if len(agg) else pd.Series(dtype=float)
            self.notes.extend(n for n in sub.notes if n not in self.notes)
            return val.reindex(df.index).ffill()

        def _q(src: str):
            """quoted(expr): `expr` evaluated on prices as quoted, whatever this namespace's price basis (a fixed
            price level such as `quoted(close) > 400` means the quoted price)."""
            if self.price_basis == "quoted":
                return evaluate_value(src, self)
            sub = getattr(self, "_quoted_ns", None)
            if sub is None:
                sub = self._quoted_ns = Namespace(self.quoted_df, ticker=self.ticker, price_basis="quoted",
                                                        close_fill=self.close_fill)
            val = evaluate_value(src, sub)
            self.notes.extend(n for n in sub.notes if n not in self.notes)
            return val

        def sym(ticker: str) -> Bars:
            other = _SYM_OVERRIDE.get(data.canonical(ticker))
            other = other if other is not None else data.load(ticker)
            if self.price_basis == "adjusted":
                other = adjusted_frame(other)
            lag = is_crypto(data.canonical(ticker)) and not is_crypto(self.ticker)
            if lag and CRYPTO_LAG_NOTE not in self.notes:
                self.notes.append(CRYPTO_LAG_NOTE)
            if (not lag and self.close_fill and is_late_close(data.canonical(ticker))
                    and not is_late_close(self.ticker) and not is_crypto(self.ticker)):
                lag = True
                note = LATE_CLOSE_NOTE.format(t=data.canonical(ticker))
                if note not in self.notes:
                    self.notes.append(note)
            return Bars(other, df.index, lag=lag)

        return {
            "sma": sma, "ma": sma, "ema": ema, "rma": rma, "wma": wma, "highest": highest, "lowest": lowest,
            "stdev": stdev, "zscore": zscore, "ref": ref, "ret": ret, "roc": ret,
            "rsi": rsi, "tret": tret, "tbill_ret": tbill_ret, "max_drawdown": max_drawdown,
            "cape": cape, "earnings_yield": earnings_yield, "cape_pct": cape_pct, "treasury_10y": treasury_10y,
            "ma_return": ma_return, "stdev_return": stdev_return, "atr": atr, "natr": natr, "volatility": volatility, "drawdown": drawdown,
            "bb_upper": bb_upper, "bb_lower": bb_lower, "pct_rank": pct_rank,
            "macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist, "ppo": ppo, "ppo_signal": ppo_signal,
            "stoch_k": stoch_k, "stoch_d": stoch_d, "stoch": stoch, "stoch_rsi_k": stoch_rsi_k,
            "stoch_rsi_d": stoch_rsi_d, "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
            "cci": cci, "willr": willr, "obv": obv, "mfi": mfi, "vwap": vwap,
            "donchian_upper": donchian_upper, "donchian_lower": donchian_lower,
            "keltner_upper": keltner_upper, "keltner_lower": keltner_lower,
            "supertrend": supertrend, "supertrend_dir": supertrend_dir, "sar": sar,
            "hma": hma, "vwma": vwma, "linreg": linreg, "alma": alma, "kama": kama,
            "tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b,
            "aroon_up": aroon_up, "aroon_down": aroon_down, "aroon_osc": aroon_osc, "cmf": cmf,
            "pivothigh": pivothigh, "pivotlow": pivotlow, "avwap": avwap, "na": na, "nz": nz, "cross": cross,
            "crossover": crossover, "crossunder": crossunder, "count": count, "bars_since": bars_since,
            "valuewhen": valuewhen, "diff": diff,
            "down_streak": down_streak, "up_streak": up_streak,
            "cummax": lambda x: series_arg(x, "cummax").cummax(), "cummin": lambda x: series_arg(x, "cummin").cummin(),
            "weekly_sma": lambda n, x=None: _periodic_sma("W-FRI", n, x),
            "monthly_sma": lambda n, x=None: _periodic_sma("M", n, x),
            "weekly_rsi": lambda n=14, x=None: _periodic("W-FRI", rsi_wilder, n, x),
            "monthly_rsi": lambda n=14, x=None: _periodic("M", rsi_wilder, n, x),
            # seeded with the simple average of the first n periods, like ema() and TradingView's ta.ema
            "weekly_ema": lambda n, x=None: _periodic("W-FRI", ema_tv, n, x),
            "monthly_ema": lambda n, x=None: _periodic("M", ema_tv, n, x),
            "weekly_ret": lambda n=1, x=None: _periodic("W-FRI", lambda e, k: e / e.shift(k) - 1, n, x),
            "monthly_ret": lambda n=1, x=None: _periodic("M", lambda e, k: e / e.shift(k) - 1, n, x),
            "weekly_close": lambda x=None: _periodic_close("W-FRI", x),
            "monthly_close": lambda x=None: _periodic_close("M", x),
            "is_week_end": lambda: is_period_end("W-FRI"), "is_month_end": lambda: is_period_end("M"),
            "is_quarter_end": lambda: is_period_end("Q"), "is_year_end": lambda: is_period_end("Y"),
            "_tf": _tf, "_q": _q, "sym": sym, "abs": np.abs, "maximum": np.maximum, "minimum": np.minimum,
            "log": np.log, "sqrt": np.sqrt,
        }


HELP = """
Variables (per bar; prices are split-adjusted, as quoted):
  open high low close volume        OHLCV
  change                            1-day % change of close (0.01 = 1%)
  down_days / up_days               consecutive down / up closes ending today
  ibs                               internal bar strength (close-low)/(high-low)
  gap                               open / previous close - 1
  range                             high/low - 1
  dow month day year                calendar (dow: 0=Mon .. 4=Fri)
  trading_day_of_month               1 on the month's first session
  trading_days_left_in_month        scheduled sessions left this month after today (0 = the last one)
  dollar_volume                     close * volume
  hl2 hlc3 ohlc4 hlcc4              price averages (hlc3 = typical price)   true_range
  ha_open ha_high ha_low ha_close   Heikin Ashi bars (as TradingView draws them)
  na(x) nz(x,y)                     x is missing / x with missing values replaced by y
Position variables (exit rules only):
  bars_held  entry_price  pnl (open trade return, 0.05 = +5%)
  highest_since_entry  lowest_since_entry
Functions (x defaults to close; n = lookback in bars, a number written in the rule - not a calculation):
  averages     sma(x,n) ema(x,n) rma(x,n) wma(x,n) hma(x,n) vwma(x,n) alma(x,n,offset,sigma) kama(x,n,fast,slow)
               linreg(x,n,offset)  least-squares moving average (TradingView's ta.linreg; offset 0 = latest)
               vwap(n)  rolling n-bar VWAP of the typical price   avwap("2020-03-23")  anchored VWAP from a date
  ranges       highest(x,n) lowest(x,n) donchian_upper(n) donchian_lower(n) atr(n) natr(n)
  bands        bb_upper(n,k) bb_lower(n,k) keltner_upper(n,k) keltner_lower(n,k)   (bb_*: x optional last argument)
  momentum     ret(x,n) rsi(x,n) macd(fast,slow) macd_signal(f,s,sig) macd_hist(f,s,sig)   (x optional last argument)
               ppo(fast,slow) ppo_signal(f,s,sig)  percentage price oscillator in percent: (EMA f - EMA s) / EMA s * 100
               stoch_k(n,smooth) stoch_d(n,smooth,d) cci(n) willr(n) mfi(n) obv()
               stoch_rsi_k(k,rsi_n,stoch_n) stoch_rsi_d(k,d,rsi_n,stoch_n)  Stochastic RSI (TradingView's 3, 3, 14, 14)
               stoch(x,high,low,n)  TradingView's ta.stoch of any series, e.g. stoch(rsi(14), rsi(14), rsi(14), 14)
  trend        adx(n) plus_di(n) minus_di(n) supertrend(n,k) supertrend_dir(n,k) (-1 up, +1 down, as TradingView)
               sar(start,max,inc) aroon_up(n) aroon_down(n) aroon_osc(n) cmf(n)
  ichimoku     tenkan(9) kijun(26) senkou_a(9,26,26) senkou_b(52,26): the cloud as drawn on the current bar
                 (computed displacement-1 bars ago, as TradingView plots it; no future data)
  pivots       pivothigh(x,left,right) pivotlow(x,left,right): the pivot's value on the bar it is confirmed
                 (right bars after the pivot), NaN otherwise, as ta.pivothigh; e.g.
                 valuewhen(pivothigh(5,5) > 0, pivothigh(5,5), 0) is the latest confirmed pivot high
  statistics   stdev(x,n) zscore(x,n) volatility(n) pct_rank(x,n) drawdown(x,n) max_drawdown(x,n)
               stdev_return(x,n) ma_return(x,n)
  total return tr (dividend-reinvested price)  tret(n) total return over n bars
               tbill_ret(n) compounded T-bill return over n bars   market_cap
  valuation    cape()  Shiller's cyclically adjusted P/E (US market, monthly from 1881), lagged 4 months so it
                 is point in time (month M's value is used from the 1st of month M+5: its earnings are
                 reported a quarter or two late)   earnings_yield()  1 / cape()
               cape_pct(years)  its percentile (0..1) among the values known so far (years=0: all history)
               treasury_10y()  the 10-year Treasury yield (0.04 = 4%) as of the close, e.g.
                 earnings_yield() - treasury_10y() > 0  (the excess CAPE yield is positive)
  timing       ref(x,n) (n >= 0; also written x[n]) crossover(a,b) crossunder(a,b) cross(a,b) count(cond,n)
               bars_since(cond)  bars since cond was last true (0 on a bar where it is true)
               valuewhen(cond,x,k)  x on the k-th most recent bar where cond was true (k=0: the latest)
               diff(x,n)  x - x n bars ago   down_streak(x) up_streak(x) cummax(x) cummin(x)
  timeframes   weekly_sma(n) weekly_ema(n) weekly_rsi(n) weekly_ret(n)      (x optional last argument)
               monthly_sma(n) monthly_ema(n) monthly_rsi(n) monthly_ret(n)
                 computed on completed weekly (Friday) / monthly bars, then held until the next
                 period closes: a week's value is known at the close of its last trading day
               weekly_close() monthly_close()   the last completed week's / month's close, as a
                 daily series. Daily indicators of it (rsi(weekly_close(), 14)) are refused: they
                 would run over repeated daily values; use weekly_rsi(14) etc. instead
               is_week_end() is_month_end() is_quarter_end() is_year_end()   the last scheduled session of
                 the week/month/quarter/year, from the published NYSE schedule (known at the open)
  other ticker sym("SPY").close, sym("^VIX").close  -> another ticker aligned to this one
  price basis  quoted(x)  x computed on prices as quoted (not dividend-adjusted), e.g.
                 quoted(close) > 400 in a portfolio whose indicators use total-return prices
Operators: + - * / < <= > >= == != and or not, e.g. 0.1 < ibs < 0.3
TradingView (Pine) spellings are accepted and translated: close[1] -> ref(close, 1) (literal offsets >= 0 only),
  ta.sma ta.ema ta.rma ta.wma ta.hma ta.vwma ta.alma ta.linreg ta.rsi ta.atr ta.highest ta.lowest ta.stdev
  ta.crossover ta.crossunder ta.cross ta.cci ta.mfi ta.wpr (-> willr) ta.obv ta.sar(start, inc, max) ta.tr
  ta.pivothigh ta.pivotlow ta.change (-> diff) ta.mom (-> diff) ta.roc (-> 100 * ret) ta.barssince ta.valuewhen
  ta.stoch(close, high, low, n) (-> stoch_k(n, 1), the raw %K; other sources -> stoch(x, h, l, n)) ta.vwap (-> hlc3: on daily bars the
  session VWAP is the bar's own typical price; use vwap(n) or avwap("date") for longer ones)
  ta.macd(src, fast, slow, signal) (-> the MACD line only; use macd_signal / macd_hist for the others)
  ta.bb / ta.supertrend / ta.dmi return several values and are refused with the equivalent:
  bb_upper(n, k) bb_lower(n, k) sma(close, n); supertrend(n, k) supertrend_dir(n, k); adx plus_di minus_di
  request.security(syminfo.tickerid, "W" / "M" / "D", x) -> weekly(x) / monthly(x) / x
  request.security("SPY", "D", close) -> sym("SPY").close (x may use SPY's open/high/low/close/volume
  with indicators that take the series explicitly); lookahead = barmerge.lookahead_on is refused
  math.abs math.max math.min math.log math.sqrt, na() nz() hl2 hlc3 ohlc4, true / false
"""


# ---------------------------------------------------------------- evaluator

_ALLOWED = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name,
    ast.Load, ast.Constant, ast.Attribute, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.BitAnd, ast.BitOr, ast.Invert,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.keyword,
)


_ATTRS = {"open", "high", "low", "close", "volume", "tr"}


class _Vectorize(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        out = node.values[0]
        for v in node.values[1:]:
            out = ast.BinOp(left=out, op=op, right=v)
        return out

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.UnaryOp(op=ast.Invert(), operand=node.operand)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        parts, left = [], node.left
        for op, right in zip(node.ops, node.comparators):
            parts.append(ast.Compare(left=left, ops=[op], comparators=[right]))
            left = right
        out = parts[0]
        for p in parts[1:]:
            out = ast.BinOp(left=out, op=ast.BitAnd(), right=p)
        return out


# daily-bar indicators: applied to weekly_close()/monthly_close() they would run over repeated values
_DAILY_INDICATORS = {"sma", "ma", "ema", "rma", "wma", "rsi", "stdev", "zscore", "ret", "roc", "tret", "highest", "lowest",
                     "pct_rank", "volatility", "ma_return", "stdev_return", "max_drawdown", "drawdown", "macd",
                     "down_streak", "up_streak"}


def _check_timeframes(tree) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _DAILY_INDICATORS:
            for a in node.args:
                for sub in ast.walk(a):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in ("weekly_close", "monthly_close"):
                        per = sub.func.id.split("_")[0]
                        f = node.func.id
                        hint = f"{per}_{f}(n)" if f in ("rsi", "ema", "sma", "ret") else f"{per}_sma(n), {per}_ema(n), {per}_rsi(n) or {per}_ret(n)"
                        raise ValueError(f"{f}({sub.func.id}(), ...) would compute a daily {f} over a {per} close repeated on every "
                                         f"day, not a {per} indicator. Use {hint}, which works on completed {per} bars.")


class _Timeframes(ast.NodeTransformer):
    """weekly(<expr>) -> _tf("W-FRI", "<expr>"), monthly(<expr>) -> _tf("M", "<expr>"), quoted(<expr>) -> _q("<expr>")."""
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "quoted":
            if len(node.args) != 1 or node.keywords:
                raise ValueError("quoted() takes one expression, e.g. quoted(sma(close, 200)) > 400")
            return ast.Call(func=ast.Name(id="_q", ctx=ast.Load()), args=[ast.Constant(ast.unparse(node.args[0]))], keywords=[])
        if isinstance(node.func, ast.Name) and node.func.id in ("weekly", "monthly"):
            if len(node.args) != 1 or node.keywords:
                raise ValueError(f"{node.func.id}() takes one expression, e.g. {node.func.id}(macd_hist())")
            freq = "W-FRI" if node.func.id == "weekly" else "M"
            return ast.Call(func=ast.Name(id="_tf", ctx=ast.Load()),
                            args=[ast.Constant(freq), ast.Constant(ast.unparse(node.args[0]))], keywords=[])
        self.generic_visit(node)
        return node


# TradingView (Pine Script v5) names -> the rule language
_PINE_TA = {"sma": "sma", "ema": "ema", "rma": "rma", "wma": "wma", "rsi": "rsi", "atr": "atr", "highest": "highest",
            "lowest": "lowest", "stdev": "stdev", "crossover": "crossover", "crossunder": "crossunder",
            "change": "diff", "mom": "diff", "barssince": "bars_since", "valuewhen": "valuewhen", "macd": "macd",
            "roc": "roc", "cci": "cci", "stoch": "stoch", "vwma": "vwma", "hma": "hma", "linreg": "linreg",
            "mfi": "mfi", "wpr": "willr", "obv": "obv", "vwap": "vwap", "sar": "sar", "cross": "cross",
            "alma": "alma", "pivothigh": "pivothigh", "pivotlow": "pivotlow", "cum": None, "bb": None,
            "supertrend": None, "dmi": None, "kc": None}
_PINE_MATH = {"abs": "abs", "max": "maximum", "min": "minimum", "log": "log", "sqrt": "sqrt"}
# ta.* functions that return several values (a tuple): not expressible in one rule, refused with the equivalent
_PINE_TUPLES = {
    "bb": "ta.bb returns [middle, upper, lower]; write sma(close, 20), bb_upper(20, 2) or bb_lower(20, 2)",
    "supertrend": "ta.supertrend(factor, atrPeriod) returns [line, direction]; write supertrend(atrPeriod, factor) for "
                  "the line or supertrend_dir(atrPeriod, factor) for the direction (-1 up, +1 down)",
    "dmi": "ta.dmi returns [+DI, -DI, ADX]; write plus_di(14), minus_di(14) or adx(14)",
    "kc": "ta.kc returns [middle, upper, lower]; write ema(close, 20), keltner_upper(20, 2) or keltner_lower(20, 2)",
    "cum": "ta.cum (a running total from the first bar) is not supported: it depends on where the data starts",
}
# ta.* built-in variables (no call)
_PINE_VARS = {"obv": "obv()", "vwap": "hlc3", "tr": "true_range"}
_TF = {"D": None, "1D": None, "W": "weekly", "1W": "weekly", "M": "monthly", "1M": "monthly"}
_OTHER_OK = {"open", "high", "low", "close", "volume"}


def _is_name(n, *ids) -> bool:
    return isinstance(n, ast.Name) and n.id in ids


def _security(node) -> ast.AST:
    """request.security(symbol, timeframe, expr) -> expr / weekly(expr) / monthly(expr), on this ticker or another."""
    for k in node.keywords:
        if k.arg == "lookahead":
            if not (isinstance(k.value, ast.Attribute) and _is_name(k.value.value, "barmerge")
                    and k.value.attr == "lookahead_off"):
                raise ValueError("request.security(..., lookahead = barmerge.lookahead_on) reads a higher timeframe bar "
                                 "before it closes (lookahead) and is refused; leave lookahead off.")
        elif k.arg not in ("gaps", "ignore_invalid_symbol", "symbol", "timeframe", "expression"):
            raise ValueError(f"request.security(): the argument {k.arg} is not supported")
    kw = {k.arg: k.value for k in node.keywords}
    args = list(node.args) + [kw[k] for k in ("symbol", "timeframe", "expression") if k in kw]
    if len(args) < 3:
        raise ValueError('request.security takes (symbol, timeframe, expression), e.g. '
                         'request.security(syminfo.tickerid, "W", close)')
    if len(args) > 3:
        if not (isinstance(args[3], ast.Attribute) and _is_name(args[3].value, "barmerge")):
            raise ValueError("request.security(): only (symbol, timeframe, expression[, gaps, lookahead]) is supported")
        for extra in args[3:]:
            if isinstance(extra, ast.Attribute) and extra.attr == "lookahead_on":
                raise ValueError("request.security(..., barmerge.lookahead_on) reads a higher timeframe bar before it "
                                 "closes (lookahead) and is refused; leave lookahead off.")
    sym_, tf, x = args[:3]
    if not (isinstance(tf, ast.Constant) and isinstance(tf.value, str) and tf.value.upper() in _TF):
        raise ValueError('request.security(): the timeframe must be "D", "W" or "M" (daily bars only; intraday '
                         'timeframes are not available)')
    per = _TF[tf.value.upper()]
    own = (isinstance(sym_, ast.Attribute) and _is_name(sym_.value, "syminfo") and sym_.attr in ("tickerid", "ticker"))
    if not own:
        if not (isinstance(sym_, ast.Constant) and isinstance(sym_.value, str) and sym_.value.strip()):
            raise ValueError('request.security(): the symbol must be syminfo.tickerid or a ticker in quotes, e.g. "SPY"')
        t = sym_.value.split(":")[-1].strip()     # "AMEX:SPY" -> SPY

        class _Other(ast.NodeTransformer):
            def visit_Name(self, n):
                if n.id in _OTHER_OK:
                    return ast.Attribute(value=ast.Call(func=ast.Name(id="sym", ctx=ast.Load()), args=[ast.Constant(t)],
                                                        keywords=[]), attr=n.id, ctx=ast.Load())
                if n.id in ("True", "False"):
                    return n
                raise ValueError(f"request.security({t!r}, ...): {n.id} of another ticker is not available; use its "
                                 "open/high/low/close/volume, e.g. request.security(\"SPY\", \"D\", ta.sma(close, 200))")

            def visit_Call(self, n):
                f = n.func.id if isinstance(n.func, ast.Name) else ""
                if f in _ALWAYS_CLOSE or f == "sym" or (f in _DEFAULTS_TO_CLOSE and not any(
                        isinstance(a, (ast.Name, ast.Attribute, ast.Call, ast.BinOp, ast.UnaryOp)) and _literal(a) is None
                        for a in n.args)):
                    raise ValueError(f"request.security({t!r}, ...): {f}() would read this chart's own bars, not "
                                     f"{t}'s; write the series explicitly, e.g. ta.sma(close, 200) or ta.rsi(close, 14)")
                n.args = [self.visit(a) for a in n.args]
                return n
        x = _Other().visit(x)
    if per is None:
        return x
    return ast.Call(func=ast.Name(id=per, ctx=ast.Load()), args=[x], keywords=[])


class _Pine(ast.NodeTransformer):
    """close[1] -> ref(close, 1); ta.sma(...) -> sma(...); math.abs -> abs; true/false -> True/False."""

    def visit_Subscript(self, node):
        self.generic_visit(node)
        sl = node.slice
        v = None
        if isinstance(sl, ast.Constant) and isinstance(sl.value, int) and not isinstance(sl.value, bool):
            v = sl.value
        elif (isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.USub) and isinstance(sl.operand, ast.Constant)
              and isinstance(sl.operand.value, int) and not isinstance(sl.operand.value, bool)):
            v = -sl.operand.value
        if v is None:
            raise ValueError(f"'{ast.unparse(node)}': the offset in [] must be a fixed whole number, e.g. close[1]")
        if v < 0:
            raise ValueError(f"'{ast.unparse(node)}': negative offsets would look into the future and are not allowed")
        if v == 0:
            return node.value
        return ast.Call(func=ast.Name(id="ref", ctx=ast.Load()), args=[node.value, ast.Constant(v)], keywords=[])

    def visit_Name(self, node):
        if node.id in ("true", "false"):
            return ast.Name(id=node.id.capitalize(), ctx=ast.Load())
        return node

    def visit_Attribute(self, node):
        # ta.obv / ta.vwap / ta.tr used as variables (no call)
        if _is_name(node.value, "ta"):
            if node.attr in _PINE_VARS:
                return ast.parse(_PINE_VARS[node.attr], mode="eval").body
            raise ValueError(f"ta.{node.attr} is not supported; see --help-expr for the functions available")
        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and _is_name(f.value, "request") and f.attr == "security":
            node.args = [self.visit(a) if i == 2 else a for i, a in enumerate(node.args)]
            for k in node.keywords:
                if k.arg == "expression":
                    k.value = self.visit(k.value)
            return _security(node)
        if isinstance(f, ast.Attribute) and _is_name(f.value, "ta"):
            node.args = [self.visit(a) for a in node.args]
            node.keywords = [ast.keyword(arg=k.arg, value=self.visit(k.value)) for k in node.keywords]
        else:
            self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("ta", "math"):
            if f.value.id == "ta" and f.attr in _PINE_TUPLES:
                raise ValueError(_PINE_TUPLES[f.attr])
            table = _PINE_TA if f.value.id == "ta" else _PINE_MATH
            name = table.get(f.attr)
            if name is None:
                raise ValueError(f"{f.value.id}.{f.attr}() is not supported; see --help-expr for the functions available")
            if node.keywords:
                raise ValueError(f"{f.value.id}.{f.attr}(): write the arguments in order, without names")
            args = list(node.args)
            if name == "stoch":
                # ta.stoch(source, high, low, length): the raw (unsmoothed) %K
                if len(args) != 4:
                    raise ValueError("ta.stoch takes (source, high, low, length), e.g. ta.stoch(close, high, low, 14)")
                if _is_name(args[0], "close") and _is_name(args[1], "high") and _is_name(args[2], "low"):
                    return ast.Call(func=ast.Name(id="stoch_k", ctx=ast.Load()), args=[args[3], ast.Constant(1)], keywords=[])
                return ast.Call(func=ast.Name(id="stoch", ctx=ast.Load()), args=args, keywords=[])
            if name == "cci" and len(args) == 2 and _is_name(args[0], "hlc3"):
                args = [args[1]]
            if name == "mfi" and len(args) == 2 and _is_name(args[0], "hlc3"):
                args = [args[1]]
            if name == "vwap":
                # session VWAP on daily bars: the bar's own source value
                if len(args) > 1:
                    raise ValueError("ta.vwap(source) with anchors or bands is not supported; use avwap(\"date\") or "
                                     "vwap(n)")
                return args[0] if args else ast.parse("hlc3", mode="eval").body
            if name == "willr" and len(args) == 1:
                return ast.Call(func=ast.Name(id="willr", ctx=ast.Load()), args=args, keywords=[])
            if name == "sar":
                # ta.sar(start, inc, max) -> sar(start, max, inc)
                if len(args) != 3:
                    raise ValueError("ta.sar takes (start, inc, max), e.g. ta.sar(0.02, 0.02, 0.2)")
                return ast.Call(func=ast.Name(id="sar", ctx=ast.Load()), args=[args[0], args[2], args[1]], keywords=[])
            if name == "obv" and args:
                raise ValueError("ta.obv is a variable: write ta.obv, not ta.obv(...)")
            if name == "macd":
                # ta.macd(source, fast, slow, signal) -> the MACD line
                if len(args) not in (3, 4):
                    raise ValueError("ta.macd takes (source, fast, slow, signal), e.g. ta.macd(close, 12, 26, 9)")
                src, fast, slow = args[0], args[1], args[2]
                call_args = [fast, slow] if (isinstance(src, ast.Name) and src.id == "close") else [fast, slow, src]
                return ast.Call(func=ast.Name(id="macd", ctx=ast.Load()), args=call_args, keywords=[])
            if name == "roc":
                inner = ast.Call(func=ast.Name(id="ret", ctx=ast.Load()), args=args, keywords=[])
                return ast.BinOp(left=inner, op=ast.Mult(), right=ast.Constant(100))
            return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[])
        return node


def pine_to_rule(text):
    """Translate TradingView (Pine) spellings to the rule language; other text is returned unchanged."""
    if not isinstance(text, str) or not re.search(r"\[|\bta\.|\bmath\.|\btrue\b|\bfalse\b|\brequest\.", text):
        return text
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return text
    tree = ast.fix_missing_locations(_Pine().visit(tree))
    return ast.unparse(tree)


# ---------------------------------------------------------------- static argument shapes
# Every argument of an indicator is either a series (a price, a variable, another indicator, arithmetic on
# them) or a lookback/length/offset/parameter, which must be a number written in the rule. A computed scalar
# (abs(1), 1+0, 2 and 1) where a lookback belongs is refused outright, so the static open-time check below and
# the runtime can never read the same argument differently.

_VALUE_FUNCS = {"abs", "maximum", "minimum", "log", "sqrt"}          # element-wise maths on values
_FREE_ARG_FUNCS = _VALUE_FUNCS | {"crossover", "crossunder", "cross", "sym", "weekly", "monthly", "_tf", "avwap",
                                  "nz", "na"}
_SERIES_ONLY_FUNCS = {"cummax", "cummin", "down_streak", "up_streak", "bars_since"}
_SCALAR_NAMES = {"entry_price"}                                       # position variable that is a number


def _literal(node):
    """A plain number literal (optionally with a sign), else None. Nothing computed counts."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _literal(node.operand)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    return None


def _kind(node) -> str:
    """Static shape of an expression: 'lit' (a number literal), 'const' (a computed scalar), 'str',
    'bars' (sym(...)) or 'series'."""
    if _literal(node) is not None:
        return "lit"
    if isinstance(node, ast.Constant):
        return "str" if isinstance(node.value, str) else "const"
    if isinstance(node, ast.Name):
        return "const" if node.id in _SCALAR_NAMES or node.id in ("True", "False") else "series"
    if isinstance(node, ast.Attribute):
        return "series"
    if isinstance(node, ast.Call):
        f = node.func.id if isinstance(node.func, ast.Name) else ""
        if f == "sym":
            return "bars"
        if f in _VALUE_FUNCS:
            kids = list(node.args) + [k.value for k in node.keywords]
            return "series" if any(_kind(a) == "series" for a in kids) else "const"
        return "series"
    kids = [k for k in ast.iter_child_nodes(node) if isinstance(k, ast.expr)]
    return "series" if any(_kind(k) == "series" for k in kids) else "const"


def _check_arguments(tree) -> None:
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        f = node.func.id
        if f == "sym":
            if len(node.args) != 1 or node.keywords or _kind(node.args[0]) != "str":
                raise ValueError('sym() takes one ticker in quotes, e.g. sym("SPY").close')
            continue
        if f in _FREE_ARG_FUNCS:
            continue
        for a in list(node.args) + [k.value for k in node.keywords]:
            k = _kind(a)
            if f in _SERIES_ONLY_FUNCS and k != "series":
                raise ValueError(f"{f}() needs a series such as close or rsi(close, 2), not {ast.unparse(a)!r}")
            if k in ("series", "lit"):
                continue
            raise ValueError(f"{f}(): {ast.unparse(a)!r} is not allowed where a lookback, length or parameter belongs; "
                             f"write the number itself (e.g. {f}(close, 20)): computed values there are refused")


# ---------------------------------------------------------------- value checks shared by every front end
# (the English parser, the Build page's block editor, JSON specs and the API all pass through check_rule)

# indicators whose number arguments are all windows: whole numbers of bars, at least 1
WINDOW_FUNCS = {
    "sma", "ema", "rma", "wma", "highest", "lowest", "stdev", "zscore", "ret", "rsi", "tret", "tbill_ret",
    "max_drawdown", "ma_return", "stdev_return", "pct_rank", "drawdown", "atr", "natr", "volatility", "macd",
    "macd_signal", "macd_hist", "ppo", "ppo_signal", "stoch_k", "stoch_d", "plus_di", "minus_di", "adx", "cci",
    "willr", "mfi", "vwap", "donchian_upper", "donchian_lower", "hma", "vwma", "aroon_up", "aroon_down",
    "aroon_osc", "cmf", "count", "weekly_sma", "monthly_sma", "weekly_ema", "monthly_ema", "weekly_rsi",
    "monthly_rsi", "weekly_ret", "monthly_ret",
}
# the first number is the window, the next ones multipliers (standard deviations, ATRs)
FIRST_WINDOW_FUNCS = {"bb_upper", "bb_lower", "keltner_upper", "keltner_lower", "supertrend", "supertrend_dir"}
# indicators with a fixed range: (name, low, high)
OSC_RANGES = {"rsi": ("RSI", 0, 100), "weekly_rsi": ("the weekly RSI", 0, 100), "monthly_rsi": ("the monthly RSI", 0, 100),
              "stoch_k": ("the stochastic %K", 0, 100), "stoch_d": ("the stochastic %D", 0, 100), "mfi": ("MFI", 0, 100),
              "adx": ("ADX", 0, 100), "willr": ("Williams %R", -100, 0)}
_CMP_TXT = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=", ast.Eq: "==", ast.NotEq: "!="}
_FLIP_TXT = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "==": "==", "!=": "!="}


def check_windows(rule: str, what: str = "") -> None:
    """Refuse a window that is not a whole number of days, at least 1 (tret(tr, -5), rsi(close, 2.5), sma(close, 0)):
    never round it or replace it with another value."""
    try:
        tree = ast.parse(pine_to_rule(rule).strip(), mode="eval")
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        f = node.func.id
        if f not in WINDOW_FUNCS and f not in FIRST_WINDOW_FUNCS:
            continue
        nums = [a for a in list(node.args) + [k.value for k in node.keywords if k.arg in (None, "n", "fast", "slow", "sig")]
                if _literal(a) is not None]
        if f in FIRST_WINDOW_FUNCS:
            nums = nums[:1]
        for a in nums:
            v = _literal(a)
            if v < 1 or not float(v).is_integer():
                raise ValueError(f"{what + ': ' if what else ''}`{ast.unparse(node)}` has a window of {v:g}: a window must be "
                                 "a positive whole number of days (1 or more).")


def check_osc_ranges(rule: str, phrase: str | None = None) -> None:
    """Refuse a threshold an oscillator can never reach ('RSI above 120', 'Williams %R below 80') or one written as a
    fraction ('RSI above 0.7' - did you mean 70?): the condition would always be true or always false."""
    try:
        tree = ast.parse(pine_to_rule(rule).strip(), mode="eval")
    except SyntaxError:
        return

    def pairs():
        for n in ast.walk(tree):
            if isinstance(n, ast.Compare):
                items = [n.left, *n.comparators]
                for a, b in zip(items, items[1:]):
                    yield a, b
                    yield b, a
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("crossover", "crossunder") and len(n.args) == 2:
                yield n.args[0], n.args[1]
                yield n.args[1], n.args[0]

    phrase = (phrase or rule).strip()
    for a, b in pairs():
        if not (isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id in OSC_RANGES):
            continue
        v = _literal(b)
        if v is None:
            continue
        v = float(v)
        name, lo, hi = OSC_RANGES[a.func.id]
        if lo <= v <= hi and not (0 < abs(v) < 1):
            continue
        if 0 < abs(v) < 1:
            raise ValueError(f"'{phrase}': {name} runs from {lo} to {hi}, so {v:g} looks like a fraction - did you mean "
                             f"{v * 100:g}?")
        if a.func.id == "willr" and 0 < v <= 100:
            raise ValueError(f"'{phrase}': Williams %R runs from -100 to 0 (-80 and below is oversold) - did you mean {-v:g}?")
        raise ValueError(f"'{phrase}': {name} runs from {lo} to {hi}, so it can never be compared with {v:g} usefully "
                         f"(always true or always false). Use a threshold between {lo} and {hi}.")


def check_rule(rule: str, what: str = "") -> tuple[str, list[str]]:
    """The value checks every condition gets, however it was written (sentence, Build page, JSON, API):

      - windows are whole numbers of days, at least 1 (check_windows)
      - oscillators are compared with a number they can reach (check_osc_ranges)
      - max_drawdown() is a positive size (0.1 = a 10% fall): compared with a negative number it is the same size
        written as a loss, so 'max_drawdown(tr, 10) > -0.1' is read as a fall shallower than 10% (max_drawdown(tr,
        10) < 0.1), with a note; compared with more than 1 (100%) it is refused
      - a return compared with -100% or less is refused (a return cannot fall below -100%)

    Returns (the rule, rewritten only for a signed max drawdown, and the notes)."""
    check_windows(rule, what)
    check_osc_ranges(rule)
    try:
        tree = ast.parse(rule, mode="eval")
    except SyntaxError:
        return rule, []
    notes: list[str] = []
    edits: list[tuple[int, int, str]] = []
    lines = rule.split("\n")
    if len(lines) > 1:     # offsets below are for a one-line rule
        return rule, notes
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP_TXT):
            continue
        a, b, op = node.left, node.comparators[0], _CMP_TXT[type(node.ops[0])]
        if _literal(a) is not None and _literal(b) is None:
            a, b, op = b, a, _FLIP_TXT[op]
        v = _literal(b)
        if v is None or not (isinstance(a, ast.Call) and isinstance(a.func, ast.Name)):
            continue
        f, v = a.func.id, float(v)
        src = ast.get_source_segment(rule, a) or ast.unparse(a)
        whole = (ast.get_source_segment(rule, node) or "").strip() == rule.strip()
        label = f"{what}: " if what and not whole else ""
        if f == "max_drawdown":
            if abs(v) > 1:
                raise ValueError(f"{label}`{ast.unparse(node)}`: the max drawdown is a fall between 0 and 100% (0.2 = 20%), "
                                 f"so comparing it with {v * 100:g}% is always true or always false.")
            if v < 0 and op in ("<", "<=", ">", ">="):
                new = f"{src} {_FLIP_TXT[op]} {-v:g}"
                edits.append((node.col_offset, node.end_col_offset, new))
                notes.append(f"{label}`{ast.get_source_segment(rule, node)}`: the max drawdown is measured as a positive size "
                             f"here (0.2 = a 20% fall), so {v * 100:g}% was read as a fall "
                             f"{'deeper' if op[0] == '<' else 'shallower'} than {-v * 100:g}%: {new}.")
        elif f in ("ret", "tret", "weekly_ret", "monthly_ret") and v <= -1 and op in ("<", "<=", ">", ">="):
            raise ValueError(f"{label}`{ast.unparse(node)}`: a return cannot fall below -100%, so comparing it with "
                             f"{v * 100:g}% is always true or always false.")
    for s, e, new in sorted(edits, reverse=True):
        rule = rule[:s] + new + rule[e:]
    return rule, notes


def compile_expr(text: str):
    return _compile_expr(pine_to_rule(text).strip())


@lru_cache(maxsize=4096)
def _compile_expr(text: str):
    tree = ast.parse(text, mode="eval")
    _check_timeframes(tree)
    _check_arguments(tree)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"not allowed in expression: {type(node).__name__} in {text!r}")
        if isinstance(node, ast.Attribute) and node.attr not in _ATTRS:
            raise ValueError(f"only {sorted(_ATTRS)} can follow a '.', e.g. sym(\"SPY\").close")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("private names are not allowed")
    tree = _Timeframes().visit(tree)
    tree = ast.fix_missing_locations(_Vectorize().visit(tree))
    return compile(tree, "<rule>", "eval")


POSITION_VARS = {"bars_held", "entry_price", "pnl"}


def names_in(text) -> set[str]:
    if callable(text):
        return set()
    return {n.id for n in ast.walk(ast.parse(pine_to_rule(text).strip(), mode="eval")) if isinstance(n, ast.Name)}


def evaluate(text, ns: Namespace) -> pd.Series:
    """Evaluate a rule to a boolean Series (NaN -> False). `text` may also be a Python callable
    f(df, ns) -> Series (the Python API): it is evaluated bar by bar on the data up to each bar (stream_callable),
    so it cannot look ahead."""
    if callable(text):
        out = stream_callable(text, ns, "bool")
    else:
        code = compile_expr(text)
        out = eval(code, {"__builtins__": {}}, ns)  # noqa: S307 - AST is whitelisted above
    idx = ns.df.index
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=idx)
    if out.dtype != bool:
        out = out.fillna(0).astype(bool)
    return out.reindex(idx, fill_value=False)


def evaluate_value(text, ns: Namespace) -> pd.Series:
    """Evaluate a numeric expression (used for ranking); callables f(df, ns) are allowed."""
    out = stream_callable(text, ns, "value") if callable(text) else eval(compile_expr(text), {"__builtins__": {}}, ns)  # noqa: S307
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=ns.df.index, dtype=float)
    return out.astype(float)


# ---------------------------------------------------------------- lookahead guard

OPEN_SAFE_NAMES = {"gap", "dow", "month", "day", "year", "trading_day_of_month",
                   "trading_days_left_in_month", "open", "True", "False"}
# functions whose series argument defaults to today's close/high/low when omitted
_DEFAULTS_TO_CLOSE = {"sma", "ma", "ema", "rma", "wma", "highest", "lowest", "stdev", "zscore", "ret", "roc", "diff",
                      "hma", "vwma", "linreg", "alma", "kama", "cci", "mfi",
                      "rsi", "pct_rank", "down_streak", "up_streak", "drawdown", "cummax", "cummin", "tret",
                      "max_drawdown", "ma_return", "stdev_return"}
# functions that always read today's close/high/low
_ALWAYS_CLOSE = {"atr", "natr", "volatility", "bb_upper", "bb_lower", "macd", "macd_signal", "macd_hist", "ppo", "ppo_signal",
                 "stoch_k", "stoch_d", "stoch_rsi_k", "stoch_rsi_d", "adx", "plus_di", "minus_di", "cci", "willr",
                 "obv", "mfi", "vwap", "donchian_upper", "donchian_lower", "keltner_upper", "keltner_lower", "supertrend", "sar",
                 "weekly_sma", "monthly_sma", "weekly_close", "monthly_close", "is_week_end",
                 "weekly_rsi", "monthly_rsi", "weekly_ema", "monthly_ema", "weekly_ret", "monthly_ret",
                 "is_month_end", "is_quarter_end", "is_year_end", "bars_since", "count", "weekly", "monthly",
                 "valuewhen", "vwma", "tenkan", "kijun", "senkou_a", "senkou_b", "aroon_up", "aroon_down",
                 "aroon_osc", "cmf", "pivothigh", "pivotlow", "avwap", "supertrend_dir"}


def first_defined(rule, ns) -> pd.Timestamp | None:
    """First date on which every indicator call in `rule` has a value (NaN during its look-back), or None."""
    if not isinstance(rule, str) or not rule.strip() or ns is None:
        return None
    text = pine_to_rule(rule).strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    first = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        seg = ast.get_source_segment(text, node)
        try:
            v = evaluate_value(seg, ns)
        except Exception:  # noqa: BLE001 - e.g. sym("SPY") alone is not a series
            continue
        if not isinstance(v, pd.Series) or v.dtype == bool or not len(v):
            continue
        fv = v.first_valid_index()
        if fv is not None:
            first = fv if first is None else max(first, fv)
    return first


# f(x, n): reads only its series argument x (close/high/low when x is left out)
_OPEN_SERIES_FUNCS = {"sma", "ma", "ema", "rma", "wma", "highest", "lowest", "stdev", "zscore", "ret", "roc", "rsi",
                      "tret", "pct_rank", "max_drawdown", "ma_return", "stdev_return", "drawdown", "diff", "hma",
                      "linreg", "alma", "kama"}
_OPEN_ONE_SERIES = {"cummax", "cummin", "down_streak", "up_streak", "quoted"}   # quoted(x): x on the quoted basis
_OPEN_ELEMENTWISE = _VALUE_FUNCS | {"crossover", "crossunder", "cross", "nz", "na"}
# zero-argument calendar functions computed from the published NYSE schedule (backtester/calendar.py), not from
# prices: known before the open (an unscheduled closure is never in the schedule, so none of them ever uses one)
_OPEN_CALENDAR_CALLS = {"is_week_end", "is_month_end", "is_quarter_end", "is_year_end"}
# monthly valuation data used months after the fact (data.CAPE_LAG_MONTHS): known before any open
_OPEN_LAGGED_CALLS = {"cape", "earnings_yield", "cape_pct"}


def _is_sym(node) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sym"
            and len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str))


def never_defined(rule, ns) -> list[tuple[str, int | None]]:
    """Indicator calls in `rule` that never get a value on this data (their look-back is longer than the history):
    [(call text, the longest look-back literal in it)]. Empty when every indicator warms up."""
    if not isinstance(rule, str) or not rule.strip() or ns is None:
        return []
    text = pine_to_rule(rule).strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        seg = ast.get_source_segment(text, node)
        try:
            v = evaluate_value(seg, ns)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(v, pd.Series) or v.dtype == bool or not len(v) or v.first_valid_index() is not None:
            continue
        nums = [int(n.value) for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                and not isinstance(n.value, bool) and float(n.value).is_integer() and n.value > 0]
        out = [(s, k) for s, k in out if seg not in s]    # the innermost call that never warms up is the cause
        if not any(s in seg for s, _ in out):
            out.append((seg, max(nums) if nums else None))
    return out


_NO_SERIES_NAMES = {"True", "False", "dow", "month", "day", "year", "trading_day_of_month", "trading_days_left_in_month"}


def reads_own_series(rule) -> bool:
    """Does `rule` read the prices of the ticker it is evaluated on (close, rsi(2), sma(close, 200), bb_lower(20, 2)),
    rather than only other tickers' (sym("QQQ").close, rsi(sym("QQQ").close, 10)) or the calendar?"""
    try:
        tree = ast.parse(pine_to_rule(str(rule)).strip(), mode="eval")
    except (SyntaxError, ValueError):
        return True     # not a rule we can read: assume it does

    def has_sym(n) -> bool:
        return any(_is_sym(x) for x in ast.walk(n))

    def own(n) -> bool:
        if isinstance(n, ast.Name):
            return n.id not in _NO_SERIES_NAMES
        if isinstance(n, ast.Attribute):
            return False if _is_sym(n.value) else own(n.value)
        if isinstance(n, ast.Call):
            if _is_sym(n):
                return False
            name = n.func.id if isinstance(n.func, ast.Name) else None
            if name in _DEFAULTS_TO_CLOSE | _ALWAYS_CLOSE and not has_sym(n):
                return True
            return any(own(a) for a in list(n.args) + [k.value for k in n.keywords])
        return any(own(c) for c in ast.iter_child_nodes(n))
    return own(tree.body)


def open_safe(rule) -> bool:
    """True if `rule` can be evaluated at the bar's open, i.e. uses no data from later in the bar.

    A whitelist: every node must be shown to be known at the open. Names: the open, gap and calendar
    variables. Calls: ref(x, n) with a literal n >= 1 (anything, shifted a bar), or n = 0 of an open-safe x;
    the one-series indicators (sma, rsi, highest, ...) given exactly one open-safe series and otherwise only
    number literals (left out, the series is today's close/high/low: not safe); element-wise maths and
    crossovers of open-safe arguments; sym("X").open. Everything else - other indicators, keyword arguments,
    computed lookbacks, today's close/high/low/volume - is not open-safe."""
    if callable(rule):
        return bool(getattr(rule, "open_safe", False))
    if not isinstance(rule, str) or not rule.strip():
        return False
    try:
        compile_expr(rule)
        tree = ast.parse(pine_to_rule(rule).strip(), mode="eval")
    except (ValueError, SyntaxError):
        return False

    def ok(node) -> bool:
        if isinstance(node, ast.Expression):
            return ok(node.body)
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float, bool))
        if isinstance(node, ast.Name):
            return node.id in OPEN_SAFE_NAMES
        if isinstance(node, ast.Attribute):
            return node.attr == "open" and _is_sym(node.value)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                return False
            f, args = node.func.id, node.args
            if f == "ref":
                if not 1 <= len(args) <= 2 or _kind(args[0]) != "series":
                    return False
                n = _literal(args[1]) if len(args) == 2 else 1
                if n is None or n < 0 or not float(n).is_integer():
                    return False
                return True if n >= 1 else ok(args[0])
            if f in _OPEN_SERIES_FUNCS:
                if len(args) > 2:
                    return False
                series = [a for a in args if _kind(a) == "series"]
                if len(series) != 1 or any(_literal(a) is None for a in args if a is not series[0]):
                    return False
                return ok(series[0])
            if f in _OPEN_ONE_SERIES:
                return len(args) == 1 and _kind(args[0]) == "series" and ok(args[0])
            if f in _OPEN_ELEMENTWISE:
                return bool(args) and all(ok(a) for a in args)
            if f in _OPEN_CALENDAR_CALLS:
                return not args
            if f in _OPEN_LAGGED_CALLS:
                return all(_literal(a) is not None for a in args)
            return False
        if isinstance(node, ast.BinOp):
            return ok(node.left) and ok(node.right)
        if isinstance(node, ast.UnaryOp):
            return ok(node.operand)
        if isinstance(node, ast.BoolOp):
            return all(ok(v) for v in node.values)
        if isinstance(node, ast.Compare):
            return ok(node.left) and all(ok(x) for x in node.comparators)
        return False

    return ok(tree)


_SYM_OVERRIDE: dict = {}   # lookahead probe: perturbed copies of other tickers' data


PROBE_SIZES = (1e-4, 2e-3, 0.01, 0.04, 0.12)
PROBE_WINDOW = 2520   # bars of history behind each probed day (10 years)


def _perturb_bar(frame: pd.DataFrame, day, rng, size: float, sign: int) -> pd.DataFrame:
    """A copy of `frame` with day's close/high/low/volume (and total-return close) replaced by other values
    consistent with its open: close = open * (1 +/- ~size), high >= max(open, close), low <= min(open, close),
    wicks of random length. No fixed pattern, so a rule cannot recognise the perturbed bar."""
    frame = frame.copy()
    if day not in frame.index:
        return frame
    o = float(frame.at[day, "open"])
    if not np.isfinite(o) or o <= 0:
        return frame
    old_c = float(frame.at[day, "close"])
    new_c = max(o * (1 + sign * size * rng.uniform(0.3, 1.7)), o * 0.05)
    wick = size * rng.uniform(0.0, 1.5, size=2)
    frame.at[day, "close"] = new_c
    frame.at[day, "high"] = max(o, new_c) * (1 + wick[0])
    frame.at[day, "low"] = min(o, new_c) * (1 - min(wick[1], 0.9))
    if "volume" in frame:
        frame["volume"] = frame["volume"].astype(float)
        frame.at[day, "volume"] = float(frame.at[day, "volume"]) * float(np.exp(rng.normal(0, 1.0))) + rng.uniform(0, 1e3)
    if "adj_close" in frame and np.isfinite(old_c) and old_c > 0:
        frame.at[day, "adj_close"] = float(frame.at[day, "adj_close"]) * new_c / old_c
    if "quote_close" in frame:
        frame.at[day, "quote_close"] = new_c
    return frame


def open_time_probe(rule, df: pd.DataFrame, ticker: str | None = None, samples: int = 32, seed: int = 0) -> str | None:
    """Empirical lookahead check for rules acted on at the open (defence in depth behind open_safe).

    On dates sampled across the WHOLE history (half of them days the rule fires), the data is cut at that bar
    and the bar's close/high/low/volume/total-return close are replaced by other values consistent with its
    open (sizes from 0.01% to ~20%, random wicks), for this ticker and every sym() ticker, together and
    separately. A rule knowable at the open gives the same answer on every variant, and the same answer on
    the cut data as on the full data. Returns a description of the first violation, or None."""
    if df is None or len(df) < 60:
        return None
    base = evaluate(rule, Namespace(df, ticker=ticker)).reindex(df.index, fill_value=False).to_numpy()
    idx = df.index
    rng = np.random.default_rng(seed)
    pos_true = np.flatnonzero(base[1:]) + 1
    pos_false = np.flatnonzero(~base[1:]) + 1
    half = samples // 2
    picks: list[int] = []
    for pool, k in ((pos_true, half), (pos_false, samples - half)):
        if len(pool):
            # stratified over the whole history: one random day from each of k equal slices of the pool
            for part in np.array_split(pool, min(k, len(pool))):
                if len(part):
                    picks.append(int(rng.choice(part)))
    picks.append(len(idx) - 1)
    import re as _re
    others = sorted({data.canonical(t) for t in _re.findall(r"""sym\(\s*["']([^"']+)["']\s*\)""", str(rule))})
    other_df = {}
    for t in others:
        try:
            other_df[t] = data.load(t)
        except Exception:  # noqa: BLE001 - the run itself reports an unknown ticker
            pass
    try:
        for n_pick, i in enumerate(sorted(set(picks))):
            cut = df.iloc[: i + 1]
            day = cut.index[-1]
            for t, od in other_df.items():
                _SYM_OVERRIDE[t] = od.loc[:day]
            # truncation (on every 4th day: a whole-history evaluation is the slow part)
            if n_pick % 4 == 0 and i < len(idx) - 1 and bool(evaluate(rule, Namespace(cut, ticker=ticker)).iloc[-1]) != bool(base[i]):
                return f"on {day.date()} the rule's answer depends on later bars"
            # the perturbed copies are compared with the unperturbed one on the same recent window (enough
            # history for any practical lookback; only the answer on `day` matters, so it is like for like)
            win = cut.iloc[-PROBE_WINDOW:]
            owin = {t: od.loc[:day].loc[win.index[0]:] for t, od in other_df.items()}
            for t, od in owin.items():
                _SYM_OVERRIDE[t] = od
            want = bool(evaluate(rule, Namespace(win, ticker=ticker)).iloc[-1])
            # the smallest and largest sizes both ways, the others one way at random
            trials = [(size, sign, "both") for size in PROBE_SIZES
                      for sign in ((1, -1) if size in (PROBE_SIZES[0], PROBE_SIZES[-1]) else (int(rng.choice([1, -1])),))]
            if other_df:   # each side on its own as well: one leak must not mask the other
                trials += [(float(rng.choice(PROBE_SIZES)), int(rng.choice([1, -1])), w) for w in ("self", "others")]
            for size, sign, which in trials:
                pert = _perturb_bar(win, day, rng, size, sign) if which != "others" else win
                for t, od in owin.items():
                    osign = sign if rng.random() < 0.5 else -sign
                    _SYM_OVERRIDE[t] = _perturb_bar(od, day, rng, size, osign) if which != "self" else od
                got = bool(evaluate(rule, Namespace(pert, ticker=ticker)).iloc[-1])
                if got != want:
                    return f"on {day.date()} the rule's answer changes when that day's close/high/low change"
    finally:
        _SYM_OVERRIDE.clear()
    return None


# ---------------------------------------------------------------- Python-function rules

_CALLABLE_PROBES: dict = {}     # (function, ticker, kind, id(bars)) -> (bars, verdict): each function is probed once


def callable_lookahead_probe(fn, df: pd.DataFrame, ticker: str | None = None, kind: str = "bool",
                             samples: int = 20, seed: int = 0, window=None, budget: float = 6.0,
                             max_evals: int = 1200, fire_cap: int = 300, stream: int = 40,
                             back: int = 5) -> str | None:
    """Empirical lookahead check for a rule written as a Python function f(df, ns) (the Python API).

    A text rule is checked statically (the whitelist, no negative offsets); a function can do anything, e.g.
    df.close.shift(-1) or a centred rolling window. A causal function's value on day D computed from the data up
    to D equals its value on D computed from the whole history; one that reads later rows does not (a
    `shift(-1)` becomes NaN on the last row of the cut data). So the function is run once on the whole history and
    then on the data cut at many days D, comparing its output on every day up to D (reporting a difference
    within `back` days of D first).

    A leak shows only on the days where the later rows would have changed the answer, which a handful of random
    cuts misses (e.g. `(rsi(2) < 10) & ~(close.shift(-1) < close)` differs only on oversold days followed by a
    down day), so the cut days are chosen adversarially, in this order, within a time budget (`budget` seconds,
    at most `max_evals` cuts):
      1. the last `stream` days (an incremental, streaming replay of the most recent bars),
      2. for a yes/no rule, every day it fires and the 3 days before each (up to `fire_cap` fire days, sampled
         uniformly when there are more), and every day its answer changes and the day before,
      3. a dense random grid over the rest.
    Days inside `window` (the run's (start, end)) come first, and every day of a short window is replayed. `kind` is "bool" (entry/exit
    rules) or "value" (ranking, order level). Returns a description of the first difference, or None. Cached per
    function and data. (`samples` is kept for compatibility: the minimum number of random cuts.)"""
    import time as _time
    if not callable(fn) or df is None or len(df) < 30:
        return None
    try:
        key = (fn, ticker, kind, id(df))
        hit = _CALLABLE_PROBES.get(key)
    except TypeError:      # an unhashable callable object: not cached
        key, hit = None, None
    if hit is not None and hit[0] is df:
        return hit[1]

    def run(frame: pd.DataFrame) -> np.ndarray:
        ns = Namespace(frame, ticker=ticker)
        if kind == "bool":
            return _as_bool(call_guarded(fn, frame, ns), frame.index).to_numpy(dtype=bool)
        return _as_value(call_guarded(fn, frame, ns), frame.index).to_numpy(dtype=float)

    def same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if kind == "bool":
            return a == b
        both_nan = np.isnan(a) & np.isnan(b)
        with np.errstate(invalid="ignore"):
            close = np.abs(a - b) <= 1e-9 * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
        return both_nan | np.nan_to_num(close, nan=0.0).astype(bool)

    t_start = _time.perf_counter()
    full = run(df)
    n = len(df)
    lo = min(max(20, n // 50), n - 2)
    rng = np.random.default_rng(seed)
    cand = np.arange(lo, n - 1)             # cut after day i (the last day, n - 1, has nothing after it)
    in_win = np.ones(n, bool)
    if window is not None:
        w0, w1 = (pd.Timestamp(x) if x is not None else None for x in window)
        in_win = np.asarray((df.index >= (w0 or df.index[0])) & (df.index <= (w1 or df.index[-1])))

    tiers: list[list[int]] = [list(range(n - 2, max(lo, n - 2 - stream) - 1, -1))]
    hot: list[int] = []
    if kind == "bool":
        fire = cand[full[cand]]
        if len(fire) > fire_cap:          # sampled uniformly, the run window's own fire days first
            fw, fo = fire[in_win[fire]], fire[~in_win[fire]]
            fw = rng.choice(fw, min(len(fw), fire_cap), replace=False)
            fire = np.r_[fw, rng.choice(fo, min(len(fo), fire_cap - len(fw)), replace=False)].astype(int)
        near = {int(f) - k for f in fire for k in (0, 1, 2, 3)}
        chg = np.flatnonzero(full[1:] != full[:-1]) + 1
        hot = [int(i) for i in rng.permutation(np.array(sorted(near | {int(c) - k for c in chg for k in (0, 1)}),
                                                         dtype=int))]
    rest = [int(i) for i in rng.permutation(cand)]
    small = int(in_win[cand].sum()) <= max_evals // 2     # a short run window: replay every day of it
    tiers += [[i for i in hot if in_win[i]], [i for i in rest if in_win[i]] if small else [],
              [i for i in hot if not in_win[i]], [i for i in rest if in_win[i]], [i for i in rest if not in_win[i]]]
    order: list[int] = []
    seen: set[int] = set()
    for tier in tiers:
        for i in tier:
            if i not in seen and lo <= i < n - 1:
                seen.add(i)
                order.append(i)
    verdict = None
    done = 0
    for i in order:
        if done >= max(samples, 1) and (done >= max_evals or _time.perf_counter() - t_start > budget):
            break
        cut = df.iloc[: i + 1]
        done += 1
        try:
            got = run(cut)
        except Exception:  # noqa: BLE001 - e.g. a function that needs more rows than the cut has
            continue
        if len(got) != i + 1:
            continue
        ok = same(got, full[: i + 1])      # the cut day and the days before it (cheap: one array compare)
        if not ok.all():
            j = int(np.flatnonzero(~ok)[-1] if not ok[max(0, i + 1 - back):].all() else np.flatnonzero(~ok)[0])
            d, cut_day = df.index[j].date(), df.index[i].date()
            verdict = f"its result on {d} changes when the data after {cut_day} is removed"
            break
    if key is not None:
        if len(_CALLABLE_PROBES) > 256:
            _CALLABLE_PROBES.clear()
        _CALLABLE_PROBES[key] = (df, verdict)
    return verdict


# ---------------------------------------------------------------- Python-function rules: causal by construction
#
# A rule written as a Python function f(df, ns) can do anything: df.close.shift(-1), a centred window, keep the
# largest frame it has seen in a global, or read data/prices/QQQ.csv itself. So it is never shown the future:
# stream_callable calls it on the data up to bar i (a copy, with a namespace - sym() and data.load included - cut at
# the same bar) for every bar i in increasing order and keeps only the last value of each call as bar i's answer.
# A function that caches what it has seen has, at bar i, seen nothing after bar i. File and network access is
# blocked while it runs (CallableIOError). A function marked `fn.vectorized_causal = True` is instead called once on
# the whole history (fast), guarded only by the empirical lookahead probe (callable_lookahead_probe), with a note.
# (Python cannot be sandboxed completely - a function could still dig through the interpreter's memory - but no
# ordinary way of writing a rule reaches later data.)

import threading as _threading  # noqa: E402
from contextlib import contextmanager as _contextmanager  # noqa: E402

_IO = _threading.local()                 # .blocked: a Python rule is running on this thread
_IO_INSTALLED = [False]
_IO_LOCK = _threading.Lock()


class CallableIOError(PermissionError):
    pass


def _io_blocked() -> bool:
    return bool(getattr(_IO, "blocked", 0)) and not getattr(_IO, "allowed", 0)


def _refuse(what: str):
    raise CallableIOError(f"A Python rule tried to {what} while it was being evaluated. Rules only get the bars up to "
                          "each day (df and ns, including ns['sym']('SPY') for other tickers); reading files or the "
                          "network could reach later data, so it is not allowed. Load what you need through "
                          "ns['sym'] instead.")


def _install_io_guard() -> None:
    """Wrap the file / network entry points once; the wrappers only refuse on a thread running a Python rule."""
    with _IO_LOCK:
        if _IO_INSTALLED[0]:
            return
        import builtins
        import io
        import os
        import socket

        def wrap(obj, name, what):
            orig = getattr(obj, name)

            def guarded(*a, **k):
                if _io_blocked():
                    _refuse(what)
                return orig(*a, **k)
            guarded.__wrapped__ = orig
            guarded.__name__ = getattr(orig, "__name__", name)
            guarded.__doc__ = getattr(orig, "__doc__", None)
            setattr(obj, name, guarded)

        wrap(builtins, "open", "open a file")
        wrap(io, "open", "open a file")
        wrap(os, "open", "open a file")
        wrap(socket.socket, "connect", "open a network connection")
        for name in ("read_csv", "read_table", "read_parquet", "read_json", "read_excel", "read_pickle", "read_feather",
                     "read_html", "read_sql", "read_fwf", "read_hdf", "read_orc", "read_xml", "read_stata", "read_sas"):
            if hasattr(pd, name):
                wrap(pd, name, f"read data with pandas.{name}")
        for name in ("load", "loadtxt", "genfromtxt", "fromfile"):
            if hasattr(np, name):
                wrap(np, name, f"read data with numpy.{name}")
        _IO_INSTALLED[0] = True


@_contextmanager
def io_blocked():
    _install_io_guard()
    _IO.blocked = getattr(_IO, "blocked", 0) + 1
    try:
        yield
    finally:
        _IO.blocked -= 1


@_contextmanager
def io_allowed():
    """The backtester's own data access (sym(), data.load) inside a Python rule."""
    _IO.allowed = getattr(_IO, "allowed", 0) + 1
    try:
        yield
    finally:
        _IO.allowed -= 1


def call_guarded(fn, df: pd.DataFrame, ns):
    """fn(df, ns) with file and network access blocked."""
    with io_blocked():
        return fn(df, ns)


def _as_bool(out, idx) -> pd.Series:
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=idx)
    if out.dtype != bool:
        out = out.fillna(0).astype(bool)
    return out.reindex(idx, fill_value=False)


def _as_value(out, idx) -> pd.Series:
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=idx, dtype=float)
    return out.astype(float).reindex(idx)


def _last(out, kind: str):
    """The answer for the last bar of a call on the data up to that bar."""
    if isinstance(out, pd.DataFrame):
        out = out.iloc[:, 0]
    if isinstance(out, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        if len(out) == 0:
            return False if kind == "bool" else np.nan
        v = out.iloc[-1] if isinstance(out, pd.Series) else out[-1]
    else:
        v = out
    if kind == "bool":
        try:
            return False if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v)
        except (TypeError, ValueError):
            return False
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _sub_namespace(ns, k: int):
    """ns cut after its k-th row (a copy of the bars, so nothing of the later rows is reachable through it)."""
    base = ns.quoted_df.iloc[:k].copy()
    extra = {}
    for key in POSITION_VARS | {"highest_since_entry", "lowest_since_entry"}:
        v = dict.get(ns, key)
        if isinstance(v, pd.Series):
            extra[key] = v.iloc[:k].copy()
    return Namespace(base, extra or None, ticker=ns.ticker, price_basis=ns.price_basis,
                     month_lookbacks=ns.month_lookbacks, close_fill=getattr(ns, "close_fill", False))


_STREAMED: dict = {}      # (fn, ticker, id(bars), basis, months, kind) -> (bars, Series, seconds)
STREAM_NOTE = ("Python rule {name}: evaluated bar by bar on the data up to each bar ({n:,} calls, {s:.1f}s), so it "
               "cannot see later data; mark the function `vectorized_causal = True` to run it once on the whole history "
               "instead (faster, but then only an empirical lookahead check guards it).")
VECTOR_NOTE = ("Warning: Python rule {name} is marked vectorized_causal, so it ran once on the whole history and only "
               "the empirical lookahead probe (the data cut at many dates) checks it - a weaker guarantee than the "
               "default bar-by-bar evaluation.")


def _fn_name(fn) -> str:
    nm = getattr(fn, "__name__", "")
    return f"{nm}()" if nm and not nm.startswith("<") else "(function)"


def stream_callable(fn, ns, kind: str = "bool") -> pd.Series:
    """Evaluate a Python-function rule causally (see the comment above): bar i's answer is the last value of
    fn(df.iloc[:i+1], ns cut at i). Cached per function and data. A `vectorized_causal` function is called once."""
    import time as _time
    df = ns.df
    idx = df.index
    if getattr(fn, "vectorized_causal", False):
        out = call_guarded(fn, df, ns)
        note = VECTOR_NOTE.format(name=_fn_name(fn))
        if note not in ns.notes:
            ns.notes.append(note)
        return _as_bool(out, idx) if kind == "bool" else _as_value(out, idx)
    positional = any(isinstance(dict.get(ns, k), pd.Series) for k in POSITION_VARS)
    try:
        key = None if positional else (fn, ns.ticker, id(ns.quoted_df), ns.price_basis, ns.month_lookbacks, kind,
                                       getattr(ns, "close_fill", False))
        hit = _STREAMED.get(key) if key is not None else None
    except TypeError:          # an unhashable callable object: not cached
        key, hit = None, None
    if hit is not None and hit[0] is ns.quoted_df:
        res, secs = hit[1], hit[2]
    else:
        t0 = _time.perf_counter()
        vals = np.zeros(len(idx), bool) if kind == "bool" else np.full(len(idx), np.nan)
        tok = data.LOAD_CUTOFF.set(None)
        try:
            with io_blocked():
                for i in range(len(idx)):
                    data.LOAD_CUTOFF.set(idx[i])
                    sub = _sub_namespace(ns, i + 1)
                    vals[i] = _last(fn(sub.df, sub), kind)
        finally:
            data.LOAD_CUTOFF.reset(tok)
        secs = _time.perf_counter() - t0
        res = pd.Series(vals, index=idx)
        if key is not None:
            if len(_STREAMED) > 512:
                _STREAMED.clear()
            _STREAMED[key] = (ns.quoted_df, res, secs)
    head = f"Python rule {_fn_name(fn)}: evaluated bar by bar"
    if not any(n.startswith(head) for n in ns.notes):
        ns.notes.append(STREAM_NOTE.format(name=_fn_name(fn), n=len(idx), s=secs))
    return res.copy()


def callable_check(fn, df: pd.DataFrame, ticker: str | None = None, kind: str = "bool", window=None) -> str | None:
    """Lookahead check of a Python-function rule before a run. A streamed function (the default) cannot use later
    data, but one that tries to - shift(-1), a centred window, a whole-series statistic, a cache of the largest frame
    it was given - answers differently when run once on the whole history; such a function is refused rather than
    silently run on logic other than what it says. A `vectorized_causal` one gets the empirical probe
    (callable_lookahead_probe). Returns a description of the problem, or None."""
    if getattr(fn, "vectorized_causal", False):
        return callable_lookahead_probe(fn, df, ticker, kind, window=window)
    ns = Namespace(df, ticker=ticker)
    streamed = (evaluate(fn, ns) if kind == "bool" else evaluate_value(fn, ns)).to_numpy()
    try:
        full = call_guarded(fn, df, Namespace(df, ticker=ticker))
    except CallableIOError:
        raise
    except Exception:  # noqa: BLE001 - the streamed values stand
        return None
    full = (_as_bool(full, df.index) if kind == "bool" else _as_value(full, df.index)).to_numpy()
    if kind == "bool":
        ok = streamed == full
    else:
        a, b = streamed.astype(float), full.astype(float)
        with np.errstate(invalid="ignore"):
            close = np.abs(a - b) <= 1e-9 * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
        ok = (np.isnan(a) & np.isnan(b)) | np.nan_to_num(close, nan=0.0).astype(bool)
    bad = np.flatnonzero(~ok)
    if window is not None and len(bad):
        w0, w1 = (pd.Timestamp(x) if x is not None else None for x in window)
        inside = np.asarray((df.index >= (w0 or df.index[0])) & (df.index <= (w1 or df.index[-1])))
        bad = np.flatnonzero(~ok & inside) if (~ok & inside).any() else bad
    if not len(bad):
        return None
    d = df.index[int(bad[0])].date()
    return (f"its result on {d} changes when the data after {d} is removed (run on the data up to each day it answers "
            "differently than on the whole history)")
