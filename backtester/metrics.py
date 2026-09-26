"""Performance statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

TRADING_DAYS = 252


def buy_and_hold(ticker: str, index: pd.DatetimeIndex, capital: float) -> pd.Series | None:
    try:
        c = data.load(ticker)["close"]
    except FileNotFoundError:
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
            in_dd, peak_i = True, i - 1 if i > 0 else 0
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
    out["days_to_recover"] = [(r - t).days if r is not None else None for t, r in zip(out.trough, out.recovery)]
    out["total_days"] = [((r if r is not None else idx[-1]) - p).days for p, r in zip(out.peak, out.recovery)]
    return out.sort_values("depth").head(top).reset_index(drop=True)


def equity_stats(equity: pd.Series, rf: float = 0.0) -> dict:
    r = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else np.nan
    ex = r - rf / TRADING_DAYS
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = ex.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = ex.mean() * TRADING_DAYS / downside if downside > 0 else np.nan
    dd = drawdown(equity)
    mdd = dd.min()
    trough = dd.idxmin()
    peak = equity[:trough].idxmax()
    rec = equity[trough:][equity[trough:] >= equity[peak]]
    recovery = rec.index[0] if len(rec) else None
    # longest time under water
    under = (dd < 0).astype(int).to_numpy()
    longest, run_start, cur = 0, None, None
    for i, u in enumerate(under):
        if u and cur is None:
            cur = i
        if (not u or i == len(under) - 1) and cur is not None:
            end = i
            length = (equity.index[end] - equity.index[max(cur - 1, 0)]).days
            if length > longest:
                longest, run_start = length, cur
            cur = None
    ulcer = np.sqrt((dd.pow(2)).mean()) * 100
    return {
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
        "years": years,
        "start_equity": equity.iloc[0],
        "end_equity": equity.iloc[-1],
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "max_dd_peak": peak.date(),
        "max_dd_trough": trough.date(),
        "max_dd_recovery": recovery.date() if recovery is not None else None,
        "longest_underwater_days": longest,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "ulcer_index": ulcer,
        "best_day": r.max() if len(r) else np.nan,
        "worst_day": r.min() if len(r) else np.nan,
        "pct_positive_days": (r > 0).mean() if len(r) else np.nan,
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
        "avg_mae": trades["mae"].mean(),
        "avg_mfe": trades["mfe"].mean(),
        "t_stat": t_stat,
        "total_commission": trades["commission"].sum(),
    }


def relative_stats(equity: pd.Series, bench: pd.Series | None, rf: float = 0.0) -> dict:
    if bench is None or len(bench) < 30:
        return {}
    df = pd.concat([equity.pct_change(), bench.pct_change()], axis=1, join="inner").dropna()
    if len(df) < 30:
        return {}
    s, b = df.iloc[:, 0] - rf / TRADING_DAYS, df.iloc[:, 1] - rf / TRADING_DAYS
    beta = np.cov(s, b)[0, 1] / b.var()
    alpha = (s.mean() - beta * b.mean()) * TRADING_DAYS
    active = s - b
    ir = active.mean() / active.std() * np.sqrt(TRADING_DAYS) if active.std() > 0 else np.nan
    return {"beta": beta, "alpha_annual": alpha, "correlation": s.corr(b), "information_ratio": ir}


def yearly_returns(series: dict[str, pd.Series]) -> pd.DataFrame:
    cols = {}
    for name, eq in series.items():
        if eq is None:
            continue
        ye = eq.groupby(eq.index.year).last()
        first = eq.iloc[0]
        prev = ye.shift(1)
        prev.iloc[0] = first
        cols[name] = ye / prev - 1
    return pd.DataFrame(cols)


def yearly_detail(equity: pd.Series, trades: pd.DataFrame, exposure: pd.Series) -> pd.DataFrame:
    g = equity.groupby(equity.index.year)
    rows = {}
    for y, eq in g:
        prev = equity[equity.index.year < y]
        base = prev.iloc[-1] if len(prev) else eq.iloc[0]
        path = pd.concat([pd.Series([base]), eq]).reset_index(drop=True)
        rows[y] = {
            "return": eq.iloc[-1] / base - 1,
            "max_drawdown": (path / path.cummax() - 1).min(),
            "end_equity": eq.iloc[-1],
            "exposure": exposure[exposure.index.year == y].mean(),
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


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    me = equity.resample("ME").last()
    prev = me.shift(1)
    prev.iloc[0] = equity.iloc[0]
    m = me / prev - 1
    tbl = pd.DataFrame({"year": m.index.year, "month": m.index.month, "r": m.values})
    out = tbl.pivot(index="year", columns="month", values="r")
    out.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][: len(out.columns)] if len(out.columns) == 12 else [pd.Timestamp(2000, c, 1).strftime("%b") for c in out.columns]
    return out


def monte_carlo(equity: pd.Series, sims: int = 1000, block: int = 20, seed: int = 7) -> dict:
    """Block bootstrap of daily returns: distribution of CAGR and max drawdown."""
    r = equity.pct_change().dropna().to_numpy()
    n = len(r)
    if n < block * 3:
        return {}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    cagr, mdd = np.empty(sims), np.empty(sims)
    years = n / TRADING_DAYS
    for s in range(sims):
        starts = rng.integers(0, n - block, nb)
        path = np.concatenate([r[a:a + block] for a in starts])[:n]
        eq = np.cumprod(1 + path)
        cagr[s] = eq[-1] ** (1 / years) - 1
        mdd[s] = (eq / np.maximum.accumulate(eq) - 1).min()
    q = [5, 50, 95]
    return {
        "sims": sims,
        "cagr_p5": np.percentile(cagr, q[0]), "cagr_p50": np.percentile(cagr, q[1]), "cagr_p95": np.percentile(cagr, q[2]),
        "mdd_p5": np.percentile(mdd, q[0]), "mdd_p50": np.percentile(mdd, q[1]), "mdd_p95": np.percentile(mdd, q[2]),
        "prob_loss": float((cagr < 0).mean()),
    }


def exposure_stats(exposure: pd.Series, positions: pd.Series) -> dict:
    return {
        "time_in_market": (positions > 0).mean(),
        "avg_exposure": exposure.mean(),
        "max_positions_held": int(positions.max()),
        "avg_positions_when_invested": positions[positions > 0].mean() if (positions > 0).any() else 0.0,
    }
