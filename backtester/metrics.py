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

CRISES = [
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
    """Daily time-weighted returns: flows arrive at the start of the day."""
    prev = equity.shift(1)
    f = flows.reindex(equity.index).fillna(0.0) if flows is not None else 0.0
    base = prev + f
    r = (equity / base - 1).where(base > 0)
    return r.iloc[1:].fillna(0.0)


def nav(equity: pd.Series, flows: pd.Series | None = None) -> pd.Series:
    """Growth of the starting capital with cash flows removed (time-weighted index)."""
    r = twr_returns(equity, flows)
    out = (1 + r).cumprod() * equity.iloc[0]
    return pd.concat([equity.iloc[:1], out])


def rf_daily(index: pd.DatetimeIndex, rf) -> pd.Series:
    if rf == "tbill":
        s = data.tbill_rate()
        if s.empty:
            return pd.Series(0.0, index=index)
        return (s.reindex(index.union(s.index)).ffill().reindex(index).fillna(0.0) / TRADING_DAYS)
    return pd.Series(float(rf or 0.0) / TRADING_DAYS, index=index)


def monthly_returns(nav_: pd.Series) -> pd.Series:
    me = nav_.resample("ME").last()
    first = nav_.iloc[0]
    prev = me.shift(1)
    prev.iloc[0] = first
    return (me / prev - 1).dropna()


def buy_and_hold(ticker: str, index: pd.DatetimeIndex, capital: float) -> pd.Series | None:
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


def drawdown_table(equity: pd.Series, top: int = 5) -> pd.DataFrame:
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
    out["peak"] = [p.date() for p in out["peak"]]
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

def equity_stats(equity: pd.Series, rf="tbill", flows: pd.Series | None = None) -> dict:
    """Statistics for an equity curve; time-weighted if flows are given."""
    nv = nav(equity, flows) if flows is not None and flows.abs().sum() > 0 else equity
    r = nv.pct_change().iloc[1:].fillna(0.0)
    years = (nv.index[-1] - nv.index[0]).days / 365.25
    total = nv.iloc[-1] / nv.iloc[0] - 1
    cagr = (nv.iloc[-1] / nv.iloc[0]) ** (1 / years) - 1 if years > 0 and nv.iloc[-1] > 0 else np.nan
    rfd = rf_daily(r.index, rf)
    ex = r - rfd
    sd = r.std()
    vol = sd * np.sqrt(TRADING_DAYS)
    sharpe = ex.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = ex.mean() * TRADING_DAYS / downside if downside > 0 else np.nan
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
    for i, u in enumerate(under):
        if u and cur is None:
            cur = i
        if (not u or i == len(under) - 1) and cur is not None:
            length = (nv.index[i] - nv.index[max(cur - 1, 0)]).days
            longest = max(longest, length)
            cur = None
    ulcer = np.sqrt((dd.pow(2)).mean()) * 100
    yr = nv.groupby(nv.index.year).last()
    yprev = yr.shift(1)
    yprev.iloc[0] = nv.iloc[0]
    yret = yr / yprev - 1
    var95 = -np.percentile(r, 5) if len(r) > 20 else np.nan
    cvar95 = -r[r <= np.percentile(r, 5)].mean() if len(r) > 20 else np.nan
    mvar95 = -np.percentile(mr, 5) if len(mr) > 12 else np.nan
    mcvar95 = -mr[mr <= np.percentile(mr, 5)].mean() if len(mr) > 12 else np.nan
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    real = np.nan
    c = data.cpi()
    if not c.empty and years > 0:
        ci = c.reindex(nv.index.union(c.index)).ffill().reindex(nv.index)
        if ci.notna().iloc[0] and ci.notna().iloc[-1]:
            real = ((nv.iloc[-1] / nv.iloc[0]) / (ci.iloc[-1] / ci.iloc[0])) ** (1 / years) - 1
    return {
        "start": nv.index[0].date(),
        "end": nv.index[-1].date(),
        "years": years,
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "total_return": total,
        "cagr": cagr,
        "real_cagr": real,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "sharpe_monthly": sharpe_m,
        "sortino_monthly": sortino_m,
        "max_drawdown": mdd,
        "max_dd_peak": peak.date(),
        "max_dd_trough": trough.date(),
        "max_dd_recovery": recovery.date() if recovery is not None else None,
        "longest_underwater_days": longest,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "ulcer_index": ulcer,
        "ulcer_performance": (cagr - float(rfd.mean() * TRADING_DAYS)) / (ulcer / 100) if ulcer > 0 else np.nan,
        "best_day": r.max() if len(r) else np.nan,
        "worst_day": r.min() if len(r) else np.nan,
        "best_month": mr.max() if len(mr) else np.nan,
        "worst_month": mr.min() if len(mr) else np.nan,
        "best_year": yret.max() if len(yret) else np.nan,
        "worst_year": yret.min() if len(yret) else np.nan,
        "pct_positive_days": (r > 0).mean() if len(r) else np.nan,
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


def cashflow_stats(equity: pd.Series, flows: pd.Series | None) -> dict:
    if flows is None or flows.abs().sum() == 0:
        return {}
    f = flows[flows != 0]
    contributed = float(f[f > 0].sum())
    withdrawn = float(-f[f < 0].sum())
    dates = [equity.index[0]] + list(f.index) + [equity.index[-1]]
    amounts = [-float(equity.iloc[0])] + [-float(x) for x in f] + [float(equity.iloc[-1])]
    return {
        "starting_balance": float(equity.iloc[0]),
        "total_contributions": contributed,
        "total_withdrawals": withdrawn,
        "ending_balance": float(equity.iloc[-1]),
        "net_gain": float(equity.iloc[-1]) - float(equity.iloc[0]) - contributed + withdrawn,
        "money_weighted_return": xirr(dates, amounts),
        "ran_out": bool((equity.iloc[1:] <= 0).any()),
    }


def trade_stats(trades: pd.DataFrame, years: float) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0}
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
    }


def relative_stats(nav_: pd.Series, bench: pd.Series | None, rf="tbill") -> dict:
    """Regression and capture statistics vs a benchmark over the overlapping period."""
    if bench is None or len(bench.dropna()) < 30:
        return {}
    df = pd.concat([nav_.pct_change(), bench.pct_change()], axis=1, join="inner").dropna()
    if len(df) < 30:
        return {}
    rfd = rf_daily(df.index, rf)
    s, b = df.iloc[:, 0] - rfd, df.iloc[:, 1] - rfd
    beta = np.cov(s, b)[0, 1] / b.var()
    alpha = (s.mean() - beta * b.mean()) * TRADING_DAYS
    active = df.iloc[:, 0] - df.iloc[:, 1]
    te = active.std() * np.sqrt(TRADING_DAYS)
    ir = active.mean() * TRADING_DAYS / te if te > 0 else np.nan
    corr = s.corr(b)
    # capture ratios on monthly returns
    m = pd.concat([monthly_returns((1 + df.iloc[:, 0]).cumprod()), monthly_returns((1 + df.iloc[:, 1]).cumprod())], axis=1).dropna()
    up, dn = m[m.iloc[:, 1] > 0], m[m.iloc[:, 1] < 0]

    def geo(x):
        return (1 + x).prod() ** (1 / len(x)) - 1 if len(x) else np.nan
    upc = geo(up.iloc[:, 0]) / geo(up.iloc[:, 1]) if len(up) else np.nan
    dnc = geo(dn.iloc[:, 0]) / geo(dn.iloc[:, 1]) if len(dn) else np.nan
    ann = (1 + df.iloc[:, 0]).prod() ** (TRADING_DAYS / len(df)) - 1
    rf_ann = float(rfd.mean() * TRADING_DAYS)
    return {"beta": beta, "alpha_annual": alpha, "correlation": corr, "r_squared": corr ** 2,
            "tracking_error": te, "information_ratio": ir,
            "treynor": (ann - rf_ann) / beta if beta else np.nan,
            "up_capture": upc, "down_capture": dnc, "period_start": df.index[0].date()}


def rolling_series(nav_: pd.Series, bench: pd.Series | None, rf="tbill") -> dict:
    r = nav_.pct_change()
    out = {}
    out["return_12m"] = nav_ / nav_.shift(TRADING_DAYS) - 1
    out["return_36m_ann"] = (nav_ / nav_.shift(3 * TRADING_DAYS)) ** (1 / 3) - 1
    ex = r - rf_daily(r.index, rf)
    out["sharpe_6m"] = ex.rolling(126).mean() / r.rolling(126).std() * np.sqrt(TRADING_DAYS)
    out["vol_3m"] = r.rolling(63).std() * np.sqrt(TRADING_DAYS)
    if bench is not None:
        b = bench.reindex(nav_.index).pct_change()
        out["beta_6m"] = r.rolling(126).cov(b) / b.rolling(126).var()
    return out


def rolling_summary(nav_: pd.Series) -> dict:
    """Best / worst / average rolling-period annualised returns (Portfolio Visualizer style)."""
    out = {}
    for yrs in (1, 3, 5, 10):
        n = yrs * TRADING_DAYS
        if len(nav_) <= n + 5:
            continue
        rr = (nav_ / nav_.shift(n)) ** (1 / yrs) - 1
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
            row[k] = float(end.iloc[-1] / seg0.iloc[-1] - 1) if len(end) else None
            any_val = any_val or row[k] is not None
        if any_val:
            rows.append(row)
    return rows


def factor_regression(nav_: pd.Series, rf="tbill") -> dict:
    """Fama-French 5 factors + momentum regression on daily excess returns (OLS with t-stats)."""
    f = data.factors()
    if f.empty:
        return {}
    r = nav_.pct_change().dropna()
    df = pd.concat([r.rename("r"), f], axis=1, join="inner").dropna()
    if len(df) < 120 or "RF" not in df:
        return {}
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
    return {"period_start": df.index[0].date(), "period_end": df.index[-1].date(), "r_squared": r2,
            "observations": len(df),
            "coefficients": [{"factor": n, "loading": float(c * (TRADING_DAYS if n == "alpha" else 1)),
                              "t_stat": float(t)} for n, c, t in zip(names, coef, tstat)]}


def correlation_matrix(series: dict[str, pd.Series]) -> dict:
    df = pd.concat({k: v for k, v in series.items() if v is not None}, axis=1).dropna()
    if len(df) < 60:
        return {}
    m = monthly_returns_frame(df).corr()
    return {"names": list(m.columns), "matrix": m.round(4).to_numpy().tolist()}


def monthly_returns_frame(df: pd.DataFrame) -> pd.DataFrame:
    me = df.resample("ME").last()
    return me.pct_change().dropna()


def yearly_returns(series: dict[str, pd.Series]) -> pd.DataFrame:
    cols = {}
    for name, eq in series.items():
        if eq is None:
            continue
        ye = eq.groupby(eq.index.year).last()
        prev = ye.shift(1)
        prev.iloc[0] = eq.iloc[0]
        cols[name] = ye / prev - 1
    return pd.DataFrame(cols)


def yearly_detail(nav_: pd.Series, trades: pd.DataFrame, exposure: pd.Series) -> pd.DataFrame:
    rows = {}
    for y, eq in nav_.groupby(nav_.index.year):
        prev = nav_[nav_.index.year < y]
        base = prev.iloc[-1] if len(prev) else eq.iloc[0]
        path = pd.concat([pd.Series([base]), eq]).reset_index(drop=True)
        first, last = eq.index[0], eq.index[-1]
        partial = (not len(prev)) and (first.month > 1 or first.day > 7) or (last.month < 12 or last.day < 24)
        rows[y] = {
            "return": eq.iloc[-1] / base - 1,
            "max_drawdown": (path / path.cummax() - 1).min(),
            "end_equity": eq.iloc[-1],
            "exposure": exposure[exposure.index.year == y].mean(),
            "partial": bool(partial),
            "from": first.date() if partial else None,
            "to": last.date() if partial else None,
        }
    out = pd.DataFrame(rows).T
    if trades is not None and not trades.empty:
        ty = pd.to_datetime(trades["exit_date"]).dt.year
        out["trades"] = trades.groupby(ty).size().reindex(out.index).fillna(0).astype(int)
        out["win_rate"] = trades.groupby(ty)["pnl"].apply(lambda p: (p > 0).mean()).reindex(out.index)
        out["pnl"] = trades.groupby(ty)["pnl"].sum().reindex(out.index).fillna(0)
    else:
        out["trades"], out["win_rate"], out["pnl"] = 0, np.nan, 0.0
    out.index.name = "year"
    return out


def monthly_table(nav_: pd.Series) -> pd.DataFrame:
    m = monthly_returns(nav_)
    tbl = pd.DataFrame({"year": m.index.year, "month": m.index.month, "r": m.values})
    out = tbl.pivot(index="year", columns="month", values="r").reindex(columns=range(1, 13))
    out.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return out


def monte_carlo(equity: pd.Series, flows: pd.Series | None = None, sims: int = 1000, block: int = 20,
                seed: int = 7) -> dict:
    """Block bootstrap of daily time-weighted returns. With cash flows, the same flow schedule is
    replayed on each resampled path, giving a probability that the money lasts."""
    r = twr_returns(equity, flows).to_numpy()
    n = len(r)
    if n < block * 3:
        return {}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    years = n / TRADING_DAYS
    cagr, mdd, final = np.empty(sims), np.empty(sims), np.empty(sims)
    fl = flows.reindex(equity.index).fillna(0.0).to_numpy()[1:] if flows is not None else np.zeros(n)
    has_flows = np.abs(fl).sum() > 0
    ruined = 0
    for s in range(sims):
        starts = rng.integers(0, n - block, nb)
        path = np.concatenate([r[a:a + block] for a in starts])[:n]
        g = np.cumprod(1 + path)
        cagr[s] = g[-1] ** (1 / years) - 1 if g[-1] > 0 else -1.0
        mdd[s] = (g / np.maximum.accumulate(g) - 1).min()
        if has_flows:
            v = float(equity.iloc[0])
            dead = False
            for t in range(n):
                v = (v + fl[t]) * (1 + path[t])
                if v <= 0:
                    dead = True
                    v = 0.0
                    break
            final[s] = v
            ruined += dead
        else:
            final[s] = float(equity.iloc[0]) * g[-1]
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
    signal = kind == "signal"
    if signal and n == 0:
        W.append({"code": "no_trades", "level": "error", "message": "No trades",
                  "detail": "The entry condition never triggered, so there is nothing to evaluate. "
                            "Sharpe, Sortino, Calmar and the trade statistics are not shown."})
    elif signal and n < FEW_TRADES:
        W.append({"code": "few_trades", "level": "warn", "message": "Too few trades for the statistics to mean much",
                  "detail": f"Only {n} trade{'s' if n != 1 else ''}. Win rate, profit factor and Sharpe from fewer than "
                            f"{FEW_TRADES} trades are mostly luck."})
    alarms = []
    sharpe, cagr, mdd = stats.get("sharpe"), stats.get("cagr"), stats.get("max_drawdown")
    if not (signal and n == 0):
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
