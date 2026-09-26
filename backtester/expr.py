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
    d = x.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder(-d.clip(upper=0), n)
    rs = up / dn
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(d.notna())


def is_crypto(ticker: str | None) -> bool:
    """A crypto pair (BTC-USD): its daily bar closes at 00:00 UTC (8pm New York), after the US close."""
    return bool(ticker) and str(ticker).upper().endswith("-USD")


CRYPTO_LAG_NOTE = ("A rule reads a crypto pair from a US-market ticker: a crypto day closes at 00:00 UTC (8pm New "
                   "York), about 4 hours after the US close, so the rule uses the crypto bar of the previous day "
                   "(the latest one complete at the US close).")


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


class Namespace(dict):
    """Evaluation namespace for one ticker; derived variables are lazy."""

    def __init__(self, df: pd.DataFrame, extra: dict[str, Any] | None = None, ticker: str | None = None,
                 price_basis: str = "quoted"):
        """price_basis "quoted": prices as quoted (split-adjusted; TradingView). "adjusted": open/high/low/close on
        a total-return basis (dividends reinvested, see adjusted_frame; Composer, Portfolio Visualizer), for this
        ticker and for sym() of any other."""
        super().__init__()
        if price_basis not in ("quoted", "adjusted"):
            raise ValueError("price_basis must be 'quoted' or 'adjusted'")
        self.price_basis = price_basis
        self.quoted_df = df        # the bars as quoted, for quoted(<expr>) on an adjusted basis
        if price_basis == "adjusted":
            df = adjusted_frame(df)
        self.df = df
        self.ticker = ticker
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
            "market_cap": lambda: self._market_cap(),
            "trading_day_of_month": lambda: pd.Series(idx.to_period("M"), index=idx).groupby(idx.to_period("M")).cumcount() + 1,
            # counted on the NYSE calendar, so the latest bar knows the sessions still to come this month
            # scheduled NYSE sessions left this month, as known on each day (no hindsight about closures)
            "trading_days_left_in_month": lambda: pd.Series(_cal.scheduled_sessions_left(idx, "M"), index=idx),
        }
        if key in lazy:
            v = lazy[key]()
            self[key] = v
            return v
        raise NameError(f"unknown name {key!r} in expression (see --help-expr)")

    def _market_cap(self) -> pd.Series:
        sh = data.shares_outstanding(self.ticker) if self.ticker else pd.Series(dtype=float)
        idx = self.df.index
        if sh.empty:
            return pd.Series(np.nan, index=idx)
        s = sh.reindex(idx.union(sh.index)).ffill().reindex(idx)
        return s * self.df["close"]

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
            x = _s(x, c)
            n = nonneg(n)
            if x.dtype == bool:
                return x.shift(n, fill_value=False)
            return x.shift(n)

        def ret(*a):
            x, n = pick(a, c, 1)
            return x / x.shift(n) - 1

        def rsi(*a):
            x, n = pick(a, c, 14)
            return rsi_wilder(x, n)

        trs = df["adj_close"] if "adj_close" in df else c

        def tret(*a):
            """Total return (dividends reinvested) over n bars."""
            x, n = pick(a, trs, 1)
            return x / x.shift(n) - 1

        def tbill_ret(n=252):
            """Compounded 3-month T-bill return over the last n bars (from 1954)."""
            r = data.tbill_rate()
            if r.empty:
                return pd.Series(0.0, index=c.index)
            rr = r.reindex(c.index.union(r.index)).ffill().reindex(c.index).fillna(0.0)
            idx = (1 + rr / 252).cumprod()
            return idx / idx.shift(int(n)) - 1

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
            return pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)

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

        def bb_upper(n=20, k=2.0):
            return sma(c, n) + k * stdev(c, n)

        def bb_lower(n=20, k=2.0):
            return sma(c, n) - k * stdev(c, n)

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

        def macd_signal(fast=12, slow=26, sig=9):
            return ema_tv(macd(fast, slow), int(sig))

        def macd_hist(fast=12, slow=26, sig=9):
            return macd(fast, slow) - macd_signal(fast, slow, sig)

        def stoch_k(n=14, smooth=3):
            ll, hh = lo.rolling(int(n)).min(), hi.rolling(int(n)).max()
            k = 100 * (c - ll) / (hh - ll)
            return k.rolling(int(smooth)).mean() if int(smooth) > 1 else k

        def stoch_d(n=14, smooth=3, d=3):
            return stoch_k(n, smooth).rolling(int(d)).mean()

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

        def cci(n=20):
            tp = (hi + lo + c) / 3
            m = tp.rolling(int(n)).mean()
            md = tp.rolling(int(n)).apply(lambda v: np.mean(np.abs(v - v.mean())), raw=True)
            return (tp - m) / (0.015 * md)

        def willr(n=14):
            hh, ll = hi.rolling(int(n)).max(), lo.rolling(int(n)).min()
            return -100 * (hh - c) / (hh - ll)

        def obv():
            return (np.sign(c.diff()).fillna(0) * vol).cumsum()

        def mfi(n=14):
            tp = (hi + lo + c) / 3
            mf = tp * vol
            pos = mf.where(tp.diff() > 0, 0.0).rolling(int(n)).sum()
            neg = mf.where(tp.diff() < 0, 0.0).rolling(int(n)).sum()
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

        def supertrend(n=10, k=3.0):
            """Supertrend line (below price in an uptrend, above in a downtrend)."""
            a = atr(n).to_numpy()
            mid = ((hi + lo) / 2).to_numpy()
            cl = c.to_numpy()
            ub, lb = mid + k * a, mid - k * a
            st = np.full(len(cl), np.nan)
            up = True
            fub, flb = np.nan, np.nan
            for i in range(len(cl)):
                if np.isnan(a[i]):
                    continue
                fub = ub[i] if np.isnan(fub) or ub[i] < fub or cl[i - 1] > fub else fub
                flb = lb[i] if np.isnan(flb) or lb[i] > flb or cl[i - 1] < flb else flb
                if up and cl[i] < flb:
                    up = False
                elif not up and cl[i] > fub:
                    up = True
                st[i] = flb if up else fub
            return pd.Series(st, index=c.index)

        def sar(step=0.02, max_step=0.2):
            """Parabolic SAR, bar for bar as TradingView's ta.sar reference implementation."""
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
                        mm, acc = h[i], min(acc + step, max_step)
                    elif not below and l[i] < mm:
                        mm, acc = l[i], min(acc + step, max_step)
                if below:
                    res = min(res, l[i - 1]) if i < 2 else min(res, l[i - 1], l[i - 2])
                else:
                    res = max(res, h[i - 1]) if i < 2 else max(res, h[i - 1], h[i - 2])
                out[i] = res
            return pd.Series(out, index=c.index)

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
            sub = Namespace(agg, ticker=self.ticker, price_basis=self.price_basis)
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
                sub = self._quoted_ns = Namespace(self.quoted_df, ticker=self.ticker, price_basis="quoted")
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
            return Bars(other, df.index, lag=lag)

        return {
            "sma": sma, "ma": sma, "ema": ema, "rma": rma, "wma": wma, "highest": highest, "lowest": lowest,
            "stdev": stdev, "zscore": zscore, "ref": ref, "ret": ret, "roc": ret,
            "rsi": rsi, "tret": tret, "tbill_ret": tbill_ret, "max_drawdown": max_drawdown,
            "ma_return": ma_return, "stdev_return": stdev_return, "atr": atr, "natr": natr, "volatility": volatility, "drawdown": drawdown,
            "bb_upper": bb_upper, "bb_lower": bb_lower, "pct_rank": pct_rank,
            "macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist,
            "stoch_k": stoch_k, "stoch_d": stoch_d, "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
            "cci": cci, "willr": willr, "obv": obv, "mfi": mfi, "vwap": vwap,
            "donchian_upper": donchian_upper, "donchian_lower": donchian_lower,
            "keltner_upper": keltner_upper, "keltner_lower": keltner_lower,
            "supertrend": supertrend, "sar": sar,
            "crossover": crossover, "crossunder": crossunder, "count": count, "bars_since": bars_since,
            "down_streak": down_streak, "up_streak": up_streak,
            "cummax": lambda x: series_arg(x, "cummax").cummax(), "cummin": lambda x: series_arg(x, "cummin").cummin(),
            "weekly_sma": lambda n, x=None: _periodic_sma("W-FRI", n, x),
            "monthly_sma": lambda n, x=None: _periodic_sma("M", n, x),
            "weekly_rsi": lambda n=14, x=None: _periodic("W-FRI", rsi_wilder, n, x),
            "monthly_rsi": lambda n=14, x=None: _periodic("M", rsi_wilder, n, x),
            "weekly_ema": lambda n, x=None: _periodic("W-FRI", lambda e, k: e.ewm(span=k, adjust=False, min_periods=k).mean(), n, x),
            "monthly_ema": lambda n, x=None: _periodic("M", lambda e, k: e.ewm(span=k, adjust=False, min_periods=k).mean(), n, x),
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
  trading_day_of_month, trading_days_left_in_month
  dollar_volume                     close * volume
Position variables (exit rules only):
  bars_held  entry_price  pnl (open trade return, 0.05 = +5%)
  highest_since_entry  lowest_since_entry
Functions (x defaults to close; n = lookback in bars, a number written in the rule - not a calculation):
  averages     sma(x,n) ema(x,n) rma(x,n) wma(x,n) vwap(n)
  ranges       highest(x,n) lowest(x,n) donchian_upper(n) donchian_lower(n) atr(n) natr(n)
  bands        bb_upper(n,k) bb_lower(n,k) keltner_upper(n,k) keltner_lower(n,k)
  momentum     ret(x,n) rsi(x,n) macd(fast,slow) macd_signal(f,s,sig) macd_hist(f,s,sig)
               stoch_k(n,smooth) stoch_d(n,smooth,d) cci(n) willr(n) mfi(n) obv()
  trend        adx(n) plus_di(n) minus_di(n) supertrend(n,k) sar(step,max)
  statistics   stdev(x,n) zscore(x,n) volatility(n) pct_rank(x,n) drawdown(x,n) max_drawdown(x,n)
               stdev_return(x,n) ma_return(x,n)
  total return tr (dividend-reinvested price)  tret(n) total return over n bars
               tbill_ret(n) compounded T-bill return over n bars   market_cap
  timing       ref(x,n) (n >= 0) crossover(a,b) crossunder(a,b) count(cond,n) bars_since(cond)
               down_streak(x) up_streak(x) cummax(x) cummin(x)
  timeframes   weekly_sma(n) weekly_ema(n) weekly_rsi(n) weekly_ret(n)      (x optional last argument)
               monthly_sma(n) monthly_ema(n) monthly_rsi(n) monthly_ret(n)
                 computed on completed weekly (Friday) / monthly bars, then held until the next
                 period closes: a week's value is known at the close of its last trading day
               weekly_close() monthly_close()   the last completed week's / month's close, as a
                 daily series. Daily indicators of it (rsi(weekly_close(), 14)) are refused: they
                 would run over repeated daily values; use weekly_rsi(14) etc. instead
               is_week_end() is_month_end() is_quarter_end() is_year_end()
  other ticker sym("SPY").close, sym("^VIX").close  -> another ticker aligned to this one
  price basis  quoted(x)  x computed on prices as quoted (not dividend-adjusted), e.g.
                 quoted(close) > 400 in a portfolio whose indicators use total-return prices
Operators: + - * / < <= > >= == != and or not, e.g. 0.1 < ibs < 0.3
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


# ---------------------------------------------------------------- static argument shapes
# Every argument of an indicator is either a series (a price, a variable, another indicator, arithmetic on
# them) or a lookback/length/offset/parameter, which must be a number written in the rule. A computed scalar
# (abs(1), 1+0, 2 and 1) where a lookback belongs is refused outright, so the static open-time check below and
# the runtime can never read the same argument differently.

_VALUE_FUNCS = {"abs", "maximum", "minimum", "log", "sqrt"}          # element-wise maths on values
_FREE_ARG_FUNCS = _VALUE_FUNCS | {"crossover", "crossunder", "sym", "weekly", "monthly", "_tf"}
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


def compile_expr(text: str):
    return _compile_expr(text.strip())


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
    return {n.id for n in ast.walk(ast.parse(text.strip(), mode="eval")) if isinstance(n, ast.Name)}


def evaluate(text, ns: Namespace) -> pd.Series:
    """Evaluate a rule to a boolean Series (NaN -> False). `text` may also be a Python callable
    f(df, ns) -> Series (the Python API); it is responsible for not looking ahead."""
    if callable(text):
        out = text(ns.df, ns)
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
    out = text(ns.df, ns) if callable(text) else eval(compile_expr(text), {"__builtins__": {}}, ns)  # noqa: S307
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=ns.df.index, dtype=float)
    return out.astype(float)


# ---------------------------------------------------------------- lookahead guard

OPEN_SAFE_NAMES = {"gap", "dow", "month", "day", "year", "trading_day_of_month",
                   "trading_days_left_in_month", "open", "True", "False"}


def first_defined(rule, ns) -> pd.Timestamp | None:
    """First date on which every indicator call in `rule` has a value (NaN during its look-back), or None."""
    if not isinstance(rule, str) or not rule.strip() or ns is None:
        return None
    text = rule.strip()
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
                      "tret", "pct_rank", "max_drawdown", "ma_return", "stdev_return", "drawdown"}
_OPEN_ONE_SERIES = {"cummax", "cummin", "down_streak", "up_streak"}
_OPEN_ELEMENTWISE = _VALUE_FUNCS | {"crossover", "crossunder"}


def _is_sym(node) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sym"
            and len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str))


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
        tree = ast.parse(rule.strip(), mode="eval")
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
    if callable(rule) or df is None or len(df) < 60:
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
