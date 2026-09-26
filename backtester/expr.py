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


class Bars:
    """Price fields of another ticker aligned to the current ticker's dates."""

    def __init__(self, df: pd.DataFrame, index: pd.Index):
        a = df.reindex(index.union(df.index)).ffill().reindex(index)
        self.open, self.high, self.low, self.close = a["open"], a["high"], a["low"], a["close"]
        if "open_ok" in df:  # an open that was never quoted is unknown, not the close
            ok = df["open_ok"].reindex(index).fillna(False).astype(bool)
            self.open = self.open.where(ok)
        self.volume = a["volume"]
        self.tr = a["adj_close"] if "adj_close" in a else a["close"]


class Namespace(dict):
    """Evaluation namespace for one ticker; derived variables are lazy."""

    def __init__(self, df: pd.DataFrame, extra: dict[str, Any] | None = None, ticker: str | None = None):
        super().__init__()
        self.df = df
        self.ticker = ticker
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
            "trading_days_left_in_month": lambda: (lambda ext: pd.Series(1, index=ext).groupby(ext.to_period("M")).transform(
                lambda s: np.arange(len(s), 0, -1)).reindex(idx))(_cal.extend(idx, 25)),
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
            """Accept f(n), f(x, n), f(n, x) or f()."""
            x, n = default_x, default_n
            for a in args:
                if isinstance(a, pd.Series):
                    x = a
                else:
                    n = a
            if isinstance(n, float) and not float(n).is_integer():
                raise ValueError(f"lookback periods must be whole numbers (got {n})")
            n = int(n)
            if n < 1:
                raise ValueError("lookback periods must be at least 1")
            return x, n

        def nonneg(n):
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
            for v in a:
                if isinstance(v, pd.Series):
                    x = v
                else:
                    n = int(v)
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
            return streak(c if x is None else _s(x, c), -1)

        def up_streak(x=None):
            return streak(c if x is None else _s(x, c), +1)

        def _period_close(freq):
            per = c.index.to_period(freq)
            last = pd.Series(c.index, index=c.index).groupby(per).transform("max")
            return per, last

        def _ends(base, freq):
            """Each complete period's last bar (a period still in progress at the data edge is left out)."""
            per = base.index.to_period(freq)
            ends = base.groupby(per).tail(1)
            if len(base) and _cal.next_sessions(base.index[-1])[0].to_period(freq) == per[-1]:
                ends = ends.iloc[:-1]
            return ends

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
            per = c.index.to_period(freq)
            s = pd.Series(False, index=c.index)
            s.loc[pd.Series(c.index, index=c.index).groupby(per).max().to_numpy()] = True
            if len(s) and _cal.next_sessions(c.index[-1])[0].to_period(freq) == per[-1]:
                s.iloc[-1] = False  # the period is not over yet: more sessions follow on the exchange calendar
            return s

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
            ends = pd.Series(df.index, index=df.index).groupby(per).max()
            agg.index = pd.DatetimeIndex(ends.reindex(agg.index).to_numpy())
            if len(df) and _cal.next_sessions(df.index[-1])[0].to_period(freq) == per[-1]:
                agg = agg.iloc[:-1]  # the period in progress at the data edge is not complete
            val = evaluate_value(src, Namespace(agg, ticker=self.ticker)) if len(agg) else pd.Series(dtype=float)
            return val.reindex(df.index).ffill()

        def sym(ticker: str) -> Bars:
            return Bars(data.load(ticker), df.index)

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
            "cummax": lambda x: _s(x, c).cummax(), "cummin": lambda x: _s(x, c).cummin(),
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
            "_tf": _tf, "sym": sym, "abs": np.abs, "maximum": np.maximum, "minimum": np.minimum,
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
Functions (x defaults to close; n = lookback in bars):
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
    """weekly(<expr>) -> _tf("W-FRI", "<expr>"), monthly(<expr>) -> _tf("M", "<expr>")."""
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in ("weekly", "monthly"):
            if len(node.args) != 1 or node.keywords:
                raise ValueError(f"{node.func.id}() takes one expression, e.g. {node.func.id}(macd_hist())")
            freq = "W-FRI" if node.func.id == "weekly" else "M"
            return ast.Call(func=ast.Name(id="_tf", ctx=ast.Load()),
                            args=[ast.Constant(freq), ast.Constant(ast.unparse(node.args[0]))], keywords=[])
        self.generic_visit(node)
        return node


def compile_expr(text: str):
    tree = ast.parse(text.strip(), mode="eval")
    _check_timeframes(tree)
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
# functions whose series argument defaults to today's close/high/low when omitted
_DEFAULTS_TO_CLOSE = {"sma", "ma", "ema", "rma", "wma", "highest", "lowest", "stdev", "zscore", "ret", "roc",
                      "rsi", "pct_rank", "down_streak", "up_streak", "drawdown", "cummax", "cummin", "tret",
                      "max_drawdown", "ma_return", "stdev_return"}
# functions that always read today's close/high/low
_ALWAYS_CLOSE = {"atr", "natr", "volatility", "bb_upper", "bb_lower", "macd", "macd_signal", "macd_hist",
                 "stoch_k", "stoch_d", "adx", "plus_di", "minus_di", "cci", "willr", "obv", "mfi", "vwap",
                 "donchian_upper", "donchian_lower", "keltner_upper", "keltner_lower", "supertrend", "sar",
                 "weekly_sma", "monthly_sma", "weekly_close", "monthly_close", "is_week_end",
                 "weekly_rsi", "monthly_rsi", "weekly_ema", "monthly_ema", "weekly_ret", "monthly_ret",
                 "is_month_end", "is_quarter_end", "is_year_end", "bars_since", "count", "weekly", "monthly"}


def open_safe(rule) -> bool:
    """True if `rule` can be evaluated at the bar's open, i.e. uses no data from later in the bar."""
    if callable(rule):
        return bool(getattr(rule, "open_safe", False))
    def const(node):
        """Value of a constant-foldable numeric expression (e.g. `1+0`, `-(-5)`), else None."""
        if any(isinstance(ch, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)) for ch in ast.walk(node)):
            return None
        try:
            v = eval(compile(ast.Expression(node), "<const>", "eval"), {"__builtins__": {}}, {})
        except Exception:
            return None
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    def ok(node) -> bool:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            f = node.func.id
            if node.keywords:
                return False  # keyword arguments can re-order series/lookback: treat as not knowable at the open
            if f == "ref":
                n = node.args[1] if len(node.args) > 1 else None
                if n is None:
                    return True
                v = const(n)
                return v is not None and v >= 1 and float(v).is_integer()
            if f in _ALWAYS_CLOSE:
                return False
            if f in _DEFAULTS_TO_CLOSE:
                series_args = [a for a in node.args if const(a) is None]
                if not series_args:
                    return False
            if f == "sym":
                return True
            return all(ok(a) for a in node.args)
        if isinstance(node, ast.Attribute):
            return node.attr == "open" and ok(node.value)
        if isinstance(node, ast.Name):
            return node.id in OPEN_SAFE_NAMES
        return all(ok(ch) for ch in ast.iter_child_nodes(node))

    return ok(ast.parse(rule.strip(), mode="eval"))


def open_time_probe(rule, df: pd.DataFrame, ticker: str | None = None, samples: int = 16) -> str | None:
    """Empirical lookahead check for rules acted on at the open (defence in depth behind open_safe).

    On sampled dates the data is cut at that bar and the bar's high/low/close/volume are replaced by
    very different values. A rule knowable at the open gives the same answer either way. Returns a
    description of the first violation, or None."""
    if callable(rule) or df is None or len(df) < 60:
        return None
    base = evaluate(rule, Namespace(df, ticker=ticker))
    idx = df.index
    pos_true = np.flatnonzero(base.reindex(idx, fill_value=False).to_numpy())
    rng = np.random.default_rng(0)
    cand = np.arange(max(30, len(idx) - 2000), len(idx))
    picks = list(rng.choice(pos_true[pos_true >= 30], size=min(samples // 2, int((pos_true >= 30).sum())), replace=False)) if len(pos_true) else []
    picks += list(rng.choice(cand, size=min(samples - len(picks), len(cand)), replace=False))
    for i in sorted(set(int(x) for x in picks)):
        cut = df.iloc[: i + 1]
        want = bool(evaluate(rule, Namespace(cut, ticker=ticker)).iloc[-1])
        for f in (1.09, 0.91):
            pert = cut.copy()
            o = float(pert["open"].iloc[-1])
            new_c = o * f
            j = pert.index[-1]
            pert.loc[j, "close"] = new_c
            pert.loc[j, "high"] = max(o, new_c) * 1.01
            pert.loc[j, "low"] = min(o, new_c) * 0.99
            pert.loc[j, "volume"] = float(pert["volume"].iloc[-1]) * (3 if f > 1 else 0.3)
            if "adj_close" in pert:
                pert.loc[j, "adj_close"] = float(pert["adj_close"].iloc[-1]) * new_c / max(float(cut["close"].iloc[-1]), 1e-12)
            got = bool(evaluate(rule, Namespace(pert, ticker=ticker)).iloc[-1])
            if got != want:
                return f"on {j.date()} the rule's answer changes when that day's close/high/low change"
    return None
