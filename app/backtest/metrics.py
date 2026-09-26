"""Performance statistics for a backtest.

All ratios use *daily* returns of end-of-day equity (the strategy is flat
overnight, so intraday noise would only inflate the sample count), annualised
with 252 trading days. With fewer than ~20 trading days every ratio is
statistically meaningless; ``summary["n_days"]`` is reported so a reader can
see that, and the report prints a warning below that threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_DAYS_FOR_RATIOS = 20


def daily_equity(equity: pd.Series, starting_capital: float) -> pd.Series:
    """End-of-day equity, with the starting capital as the day-0 anchor."""
    eod = equity.groupby(equity.index.date).last()
    eod.index = pd.to_datetime(eod.index)
    anchor = pd.Series([starting_capital], index=[eod.index[0] - pd.Timedelta(days=1)])
    return pd.concat([anchor, eod])


def drawdown(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak (<= 0)."""
    peak = equity.cummax()
    return equity / peak - 1.0


@dataclass(frozen=True)
class DrawdownPeriod:
    peak_time: pd.Timestamp
    trough_time: pd.Timestamp
    recovery_time: pd.Timestamp | None
    depth_pct: float
    depth_dollars: float


def drawdown_periods(equity: pd.Series, top: int = 5) -> list[DrawdownPeriod]:
    """The ``top`` deepest peak-to-trough episodes, deepest first."""
    periods: list[DrawdownPeriod] = []
    peak_val, peak_t = equity.iloc[0], equity.index[0]
    trough_val, trough_t = peak_val, peak_t
    in_dd = False
    for t, v in equity.items():
        if v >= peak_val:
            if in_dd:
                periods.append(
                    DrawdownPeriod(peak_t, trough_t, t, trough_val / peak_val - 1, trough_val - peak_val)
                )
                in_dd = False
            peak_val, peak_t = v, t
            trough_val, trough_t = v, t
        elif v < trough_val:
            trough_val, trough_t = v, t
            in_dd = True
    if in_dd:
        periods.append(
            DrawdownPeriod(peak_t, trough_t, None, trough_val / peak_val - 1, trough_val - peak_val)
        )
    return sorted(periods, key=lambda p: p.depth_pct)[:top]


def _ratio(num: float, den: float) -> float | None:
    if den == 0 or not math.isfinite(den) or not math.isfinite(num):
        return None
    return num / den


def _streak(signs: list[int], target: int) -> int:
    best = cur = 0
    for s in signs:
        cur = cur + 1 if s == target else 0
        best = max(best, cur)
    return best


def return_stats(eod: pd.Series, risk_free_annual: float = 0.0) -> dict:
    rets = eod.pct_change().dropna()
    excess = rets - risk_free_annual / TRADING_DAYS
    std = rets.std(ddof=1) if len(rets) > 1 else float("nan")
    downside = rets[rets < 0]
    downside_dev = math.sqrt((np.minimum(rets, 0) ** 2).mean()) if len(rets) else float("nan")
    sharpe = _ratio(excess.mean(), std)
    sortino = _ratio(excess.mean(), downside_dev)
    total = eod.iloc[-1] / eod.iloc[0] - 1
    span_days = (eod.index[-1] - eod.index[0]).days
    years = span_days / 365.25 if span_days > 0 else float("nan")
    cagr = (1 + total) ** (1 / years) - 1 if years and years > 0 and total > -1 else None
    dd = drawdown(eod)
    return {
        "total_return_pct": 100 * total,
        "cagr_pct": None if cagr is None else 100 * cagr,
        "annual_volatility_pct": None if not math.isfinite(std) else 100 * std * math.sqrt(TRADING_DAYS),
        "sharpe": None if sharpe is None else sharpe * math.sqrt(TRADING_DAYS),
        "sortino": None if sortino is None else sortino * math.sqrt(TRADING_DAYS),
        "max_drawdown_eod_pct": 100 * dd.min(),
        "best_day_pct": 100 * rets.max() if len(rets) else None,
        "worst_day_pct": 100 * rets.min() if len(rets) else None,
        "positive_days_pct": 100 * (rets > 0).mean() if len(rets) else None,
        "n_days": len(rets),
        "downside_days": len(downside),
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    pnl = trades["net_pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    r = trades["r_multiple"]
    win_rate = len(wins) / len(pnl)
    payoff = _ratio(wins.mean() if len(wins) else 0.0, -losses.mean() if len(losses) else 0.0)
    kelly = None
    if payoff:
        kelly = 100 * (win_rate - (1 - win_rate) / payoff)
    sqn = _ratio(r.mean() * math.sqrt(len(r)), r.std(ddof=1)) if len(r) > 1 else None
    signs = [1 if p > 0 else -1 for p in pnl]
    return {
        "n_trades": int(len(pnl)),
        "n_long": int((trades["direction"] == "long").sum()),
        "n_short": int((trades["direction"] == "short").sum()),
        "win_rate_pct": 100 * win_rate,
        "profit_factor": _ratio(gross_win, gross_loss),
        "expectancy_dollars": pnl.mean(),
        "avg_r": r.mean(),
        "median_r": r.median(),
        "sqn": sqn,
        "avg_win": wins.mean() if len(wins) else None,
        "avg_loss": losses.mean() if len(losses) else None,
        "payoff_ratio": payoff,
        "largest_win": pnl.max(),
        "largest_loss": pnl.min(),
        "max_consecutive_wins": _streak(signs, 1),
        "max_consecutive_losses": _streak(signs, -1),
        "avg_hold_minutes": trades["hold_minutes"].mean(),
        "gross_pnl": trades["gross_pnl"].sum(),
        "commissions": trades["commissions"].sum(),
        "slippage": trades["slippage_cost"].sum(),
        "net_pnl": pnl.sum(),
        "kelly_pct": kelly,
        "avg_mfe_r": trades["mfe_r"].mean(),
        "avg_mae_r": trades["mae_r"].mean(),
    }


def exposure_pct(trades: pd.DataFrame, equity: pd.Series) -> float:
    """Share of bar closes on the calendar with at least one position open."""
    if trades.empty:
        return 0.0
    in_mkt = pd.Series(False, index=equity.index)
    for _, row in trades.iterrows():
        in_mkt |= (equity.index > row["entry_time"]) & (equity.index <= row["exit_time"])
    return 100 * in_mkt.mean()


def breakdown(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(by)
    out = pd.DataFrame(
        {
            "trades": g.size(),
            "win_rate_pct": g["net_pnl"].apply(lambda s: 100 * (s > 0).mean()),
            "net_pnl": g["net_pnl"].sum(),
            "avg_r": g["r_multiple"].mean(),
            "total_r": g["r_multiple"].sum(),
        }
    )
    return out.round(2)


def period_returns(eod: pd.Series, freq: str) -> pd.Series:
    """Compounded returns per calendar period ('YE' yearly, 'ME' monthly), in %."""
    rets = eod.pct_change().dropna()
    return 100 * ((1 + rets).groupby(rets.index.to_period(freq[0])).prod() - 1)


def summarize(
    equity: pd.Series,
    trades: pd.DataFrame,
    starting_capital: float,
    benchmarks: dict[str, pd.Series] | None = None,
    risk_free_annual: float = 0.0,
) -> dict:
    """Everything the report needs, as plain data."""
    eod = daily_equity(equity, starting_capital)
    intraday_dd = drawdown(pd.concat([pd.Series([starting_capital], index=[equity.index[0]]), equity]))
    periods = drawdown_periods(pd.concat([pd.Series([starting_capital], index=[equity.index[0]]), equity]))
    worst = periods[0] if periods else None
    stats = return_stats(eod, risk_free_annual)
    max_dd_pct = 100 * intraday_dd.min()
    summary = {
        "start": equity.index[0],
        "end": equity.index[-1],
        "starting_capital": starting_capital,
        "ending_equity": float(equity.iloc[-1]),
        "net_profit": float(equity.iloc[-1] - starting_capital),
        **stats,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_dollars": worst.depth_dollars if worst else 0.0,
        "max_drawdown_peak": worst.peak_time if worst else None,
        "max_drawdown_trough": worst.trough_time if worst else None,
        "max_drawdown_recovery": worst.recovery_time if worst else None,
        "calmar": None,
        "ulcer_index": float(math.sqrt(((100 * intraday_dd) ** 2).mean())),
        "exposure_pct": exposure_pct(trades, equity),
        **trade_stats(trades),
    }
    if summary["cagr_pct"] is not None and max_dd_pct < 0:
        summary["calmar"] = summary["cagr_pct"] / abs(max_dd_pct)

    bench_rows = {}
    yearly = {"Strategy": period_returns(eod, "YE")}
    monthly = {"Strategy": period_returns(eod, "ME")}
    strat_rets = eod.pct_change().dropna()
    for name, curve in (benchmarks or {}).items():
        b_eod = daily_equity(curve, starting_capital)
        b = return_stats(b_eod, risk_free_annual)
        b_rets = b_eod.pct_change().dropna()
        aligned = pd.concat([strat_rets, b_rets], axis=1, join="inner").dropna()
        corr = beta = None
        if len(aligned) >= 3 and aligned.iloc[:, 1].std() > 0:
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            beta = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())
        bench_rows[name] = {
            "total_return_pct": b["total_return_pct"],
            "ending_equity": float(curve.iloc[-1]),
            "max_drawdown_pct": 100 * drawdown(curve).min(),
            "sharpe": b["sharpe"],
            "annual_volatility_pct": b["annual_volatility_pct"],
            "correlation": corr,
            "beta": beta,
        }
        yearly[name] = period_returns(b_eod, "YE")
        monthly[name] = period_returns(b_eod, "ME")

    return {
        "summary": summary,
        "benchmarks": bench_rows,
        "yearly": pd.DataFrame(yearly).round(2),
        "monthly": pd.DataFrame(monthly).round(2),
        "drawdowns": periods,
        "by_symbol": breakdown(trades, "symbol"),
        "by_direction": breakdown(trades, "direction"),
        "by_exit": breakdown(trades, "exit_reason"),
        "eod": eod,
        "drawdown_series": drawdown(equity),
        "few_days": stats["n_days"] < MIN_DAYS_FOR_RATIOS,
    }
