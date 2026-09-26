"""Risk contributions of an allocation run's holdings: how much of the portfolio's volatility, and of its worst
drawdown, each asset accounts for.

Three views of volatility, all adding up to 100%:

- ex ante (Euler) on the average weights w over the run and the covariance S of the assets' daily (or monthly)
  total returns over the run: marginal contribution MCR_i = (S w)_i / sigma, contribution RC_i = w_i * MCR_i,
  sum_i RC_i = sigma = sqrt(w' S w); share_i = RC_i / sigma.
- realised: with the actual (drifting, rebalanced) weights of the day before, each asset's daily contribution
  c_i,t = w_i,t-1 * r_i,t and the portfolio's asset return r_t = sum_i c_i,t; share_i = cov(c_i, r) / var(r).

Drawdown: over the portfolio's maximum drawdown (peak to trough), each asset's contribution to the loss is
sum_t G_t-1 * c_i,t, where G is the growth of the asset return since the peak; these add up exactly to the
fall of the asset return over the window (cash interest, costs and cash flows are left out).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _euler(w: np.ndarray, cov: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    var = float(w @ cov @ w)
    sig = np.sqrt(max(var, 0.0))
    mcr = cov @ w / sig if sig > 0 else np.zeros_like(w)
    return sig, mcr, w * mcr


def risk_contributions(res, nav: pd.Series | None = None, limit: int = 60) -> dict:
    """Risk contribution table for an allocation Result (see the module docstring); {} when not applicable."""
    hw = getattr(res, "holdings", None)
    if getattr(res, "kind", None) != "allocation" or hw is None or hw.empty or len(hw) < 40:
        return {}
    w = hw.drop(columns=[c for c in ("cash", "other") if c in hw])
    w = w.loc[:, (w.abs() > 1e-9).any()]
    cols = [t for t in w.columns if t in (res.prices or {})]
    if not cols:
        return {}
    w = w[cols].fillna(0.0)
    px = pd.DataFrame({t: (res.prices[t]["adj_close"] if "adj_close" in res.prices[t] else res.prices[t]["close"])
                       for t in cols}).reindex(w.index).ffill()
    r = px.pct_change(fill_method=None).fillna(0.0)
    wprev = w.shift(1).fillna(0.0)
    c = wprev * r                                   # daily contribution of each asset
    rp = c.sum(axis=1)
    c, rp, r = c.iloc[1:], rp.iloc[1:], r.iloc[1:]
    if len(rp) < 20 or rp.var() <= 0:
        return {}
    wbar = w.mean().to_numpy()
    sig_d, mcr_d, rc_d = _euler(wbar, r.cov().to_numpy() * TRADING_DAYS)
    me = px.groupby(px.index.to_period("M")).last()
    rm = me.pct_change(fill_method=None).dropna(how="all").fillna(0.0)
    if len(rm) >= 6:
        sig_m, mcr_m, rc_m = _euler(wbar, rm.cov().to_numpy() * 12)
    else:
        sig_m, mcr_m, rc_m = np.nan, np.full(len(cols), np.nan), np.full(len(cols), np.nan)
    real_share = np.array([np.cov(c[t], rp)[0, 1] for t in cols]) / rp.var()
    # drawdown contribution over the portfolio's worst drawdown
    series = nav if nav is not None and len(nav) > 2 else (1 + rp).cumprod()
    series = series[series.index >= w.index[0]]
    dd = series / series.cummax() - 1
    trough = dd.idxmin()
    peak = series[:trough].idxmax()
    win = c[(c.index > peak) & (c.index <= trough)]
    dd_out = None
    if len(win) and dd.min() < 0:
        g = (1 + win.sum(axis=1)).cumprod().shift(1).fillna(1.0)
        contrib = win.mul(g, axis=0).sum()
        total = float(contrib.sum())
        dd_out = {"peak": peak.date(), "trough": trough.date(), "depth": float(dd.min()), "asset_return": total,
                  "by_asset": {t: float(contrib[t]) for t in cols}}
    rows = []
    vols = r.std().to_numpy() * np.sqrt(TRADING_DAYS)
    for j, t in enumerate(cols):
        row = {"ticker": t, "avg_weight": float(wbar[j]), "vol": float(vols[j]),
               "mcr": float(mcr_d[j]), "contribution": float(rc_d[j]), "share": float(rc_d[j] / sig_d) if sig_d > 0 else np.nan,
               "contribution_monthly": float(rc_m[j]), "share_monthly": float(rc_m[j] / sig_m) if sig_m and sig_m > 0 else np.nan,
               "share_realised": float(real_share[j])}
        if dd_out:
            row["drawdown_contribution"] = dd_out["by_asset"][t]
            row["drawdown_share"] = dd_out["by_asset"][t] / dd_out["asset_return"] if dd_out["asset_return"] else np.nan
        rows.append(row)
    rows.sort(key=lambda x: -abs(x["share"]) if np.isfinite(x["share"]) else 0)
    out = {"rows": rows[:limit], "n_assets": len(rows), "vol_daily": sig_d, "vol_monthly": float(sig_m),
           "vol_realised": float(rp.std() * np.sqrt(TRADING_DAYS)), "start": w.index[0].date(), "end": w.index[-1].date()}
    if dd_out:
        out["drawdown"] = {k: v for k, v in dd_out.items() if k != "by_asset"}
    return out


def frame(rc: dict) -> pd.DataFrame:
    """The table as a DataFrame (for the Excel export)."""
    return pd.DataFrame(rc.get("rows") or []).rename(columns={
        "avg_weight": "average weight", "vol": "volatility (daily returns, annualised)",
        "mcr": "marginal contribution (d vol / d weight)", "contribution": "contribution to volatility (daily)",
        "share": "share of volatility (daily)", "contribution_monthly": "contribution to volatility (monthly)",
        "share_monthly": "share of volatility (monthly)", "share_realised": "share of realised variance",
        "drawdown_contribution": "contribution to max drawdown", "drawdown_share": "share of max drawdown"})
