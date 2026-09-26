"""Factor regressions (Portfolio Visualizer's "factor analysis"): CAPM, Fama-French 3, Carhart 4,
Fama-French 5 and FF5 + momentum, on daily or monthly excess returns, with t-stats, R², annualised
alpha and rolling 36-month loadings.

Factor data: data/factors/ff3_daily.csv when present (otherwise the first three factors of
ff5_daily.csv - the market and SMB/HML from the 5-factor file, which French builds slightly
differently), ff5_daily.csv and mom_daily.csv. Monthly factor returns are compounded from the daily
ones: the market as (1 + Mkt-RF + RF) compounded minus compounded RF, the long-short factors
compounded directly.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import data, metrics

MODELS = {
    "capm": ("CAPM", ["Mkt-RF"]),
    "ff3": ("Fama-French 3", ["Mkt-RF", "SMB", "HML"]),
    "carhart": ("Carhart 4", ["Mkt-RF", "SMB", "HML", "Mom"]),
    "ff5": ("Fama-French 5", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]),
    "ff6": ("Fama-French 5 + momentum", ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]),
}
DESCRIBE = {"Mkt-RF": "market excess return", "SMB": "size (small minus big)", "HML": "value (high minus low book/price)",
            "RMW": "profitability (robust minus weak)", "CMA": "investment (conservative minus aggressive)",
            "Mom": "momentum (winners minus losers)"}


def _read(name: str) -> pd.DataFrame:
    p = data.DATA / "factors" / name
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"], index_col="date").sort_index()
    df.columns = [c.strip() for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


@lru_cache(maxsize=4)
def factor_table(model: str) -> pd.DataFrame:
    """Daily factor returns (decimal) for a model, plus RF."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {', '.join(MODELS)}")
    cols = MODELS[model][1]
    ff5, mom = _read("ff5_daily.csv"), _read("mom_daily.csv")
    if model in ("capm", "ff3", "carhart"):
        ff3 = _read("ff3_daily.csv")
        base = ff3 if not ff3.empty else ff5
    else:
        base = ff5
    if base.empty:
        raise ValueError("Factor data is missing (data/factors/ff5_daily.csv); the data job downloads it.")
    parts = [base[[c for c in cols if c in base] + ["RF"]]]
    if "Mom" in cols:
        if mom.empty:
            raise ValueError("Momentum factor data is missing (data/factors/mom_daily.csv).")
        parts.append(mom[["Mom"]])
    df = pd.concat(parts, axis=1, join="inner").dropna()
    return df


def monthly_factors(daily: pd.DataFrame) -> pd.DataFrame:
    g = (1 + daily).resample("ME").prod() - 1
    out = g.copy()
    if "Mkt-RF" in daily:
        mkt = (1 + daily["Mkt-RF"] + daily["RF"]).resample("ME").prod() - 1
        out["Mkt-RF"] = mkt - g["RF"]
    counts = daily["RF"].resample("ME").count()
    return out[counts > 0]


def ols(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS with an intercept column already in X. Coefficients, standard errors, t-stats, p-values."""
    from scipy import stats
    n, k = X.shape
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = n - k
    s2 = resid @ resid / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = coef / se
    p = 2 * stats.t.sf(np.abs(t), dof) if dof > 0 else np.full(k, np.nan)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - resid @ resid / ss_tot if ss_tot > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else np.nan
    return {"coef": coef, "se": se, "t": t, "p": p, "r2": r2, "adj_r2": adj, "n": n,
            "resid_sd": float(np.sqrt(s2)) if np.isfinite(s2) else np.nan}


def returns_for(target) -> tuple[pd.Series, str]:
    """Daily total returns for a ticker, a {ticker: weight} dict (rebalanced monthly), or a spec."""
    from . import parser, portfolio, runner
    if isinstance(target, str) and target.strip() and " " not in target.strip():
        t = data.canonical(target)
        return data.load(t)["adj_close"].pct_change().dropna(), t
    if isinstance(target, dict):
        w = {data.canonical(k): float(v) for k, v in target.items()}
        s = sum(w.values())
        if s <= 0:
            raise ValueError("Weights must add up to more than zero.")
        p = portfolio.Portfolio(tree={"weights": "specified", "w": [v / s for v in w.values()],
                                      "children": [{"asset": k} for k in w]}, rebalance="monthly")
        res = portfolio.run(p)
        return metrics.nav(res.equity, res.extras.get("flows")).pct_change().dropna().iloc[1:], \
            " / ".join(f"{v / s:.0%} {k}" for k, v in w.items())
    spec = parser.parse(target) if isinstance(target, str) else target
    res = runner.run(spec)
    nv = metrics.nav(res.equity, res.extras.get("flows"))
    return nv.pct_change().dropna().iloc[1:], (spec.name or spec.description or "strategy")


def analyze(returns: pd.Series, model: str = "ff3", freq: str = "monthly", start=None, end=None,
            rolling_months: int = 36, name: str = "") -> dict:
    if freq not in ("daily", "monthly"):
        raise ValueError("freq must be daily or monthly")
    label, cols = MODELS[model]
    f = factor_table(model)
    r = returns.dropna()
    if start:
        r = r[r.index >= pd.Timestamp(start)]
    if end:
        r = r[r.index <= pd.Timestamp(end)]
    fd = f.reindex(r.index).dropna()
    r = r.reindex(fd.index)
    if freq == "monthly":
        rm = (1 + r).resample("ME").prod() - 1
        fm = monthly_factors(fd)
        df = pd.concat([rm.rename("r"), fm], axis=1, join="inner").dropna()
        # drop partial first/last months
        days = r.resample("ME").count().reindex(df.index)
        df = df[days >= 0.8 * days.median()]
        per_year, min_obs = 12, 24
    else:
        df = pd.concat([r.rename("r"), fd], axis=1, join="inner").dropna()
        per_year, min_obs = 252, 120
    if len(df) < min_obs:
        raise ValueError(f"Need at least {min_obs} {freq} observations that overlap the factor data "
                         f"(factors end {f.index[-1].date()}); got {len(df)}.")
    y = (df["r"] - df["RF"]).to_numpy()
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
    o = ols(y, X)
    names = ["Alpha"] + cols
    coefs = []
    for i, n in enumerate(names):
        coefs.append({"factor": n, "about": DESCRIBE.get(n, "intercept (return not explained by the factors)"),
                      "loading": float(o["coef"][i]), "std_error": float(o["se"][i]), "t_stat": float(o["t"][i]),
                      "p_value": float(o["p"][i]),
                      "significant": bool(abs(o["t"][i]) >= 1.96)})
    alpha = o["coef"][0]
    out = {"name": name, "model": model, "model_label": label, "freq": freq, "factors": cols,
           "start": df.index[0].date(), "end": df.index[-1].date(), "observations": int(o["n"]),
           "r_squared": o["r2"], "adj_r_squared": o["adj_r2"],
           "alpha_period": float(alpha), "alpha_annual": float((1 + alpha) ** per_year - 1),
           "alpha_annual_simple": float(alpha * per_year),
           "alpha_t": float(o["t"][0]), "alpha_p": float(o["p"][0]),
           "tracking_error_annual": float(o["resid_sd"] * np.sqrt(per_year)),
           "coefficients": coefs,
           "explained": {"mean_excess_annual": float(y.mean() * per_year),
                         "contributions_annual": {c: float(o["coef"][i + 1] * df[c].mean() * per_year) for i, c in enumerate(cols)}},
           "rolling": rolling(df, cols, rolling_months, freq)}
    return out


def rolling(df: pd.DataFrame, cols: list[str], months: int, freq: str) -> dict:
    """Loadings re-estimated at each month end on the trailing `months` months of observations."""
    ends = df.index.to_series().groupby(df.index.to_period("M")).max()
    y_all = (df["r"] - df["RF"]).to_numpy()
    X_all = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
    need = int(months * (0.8 if freq == "monthly" else 15))
    dates, rows = [], {c: [] for c in ["Alpha"] + cols}
    r2 = []
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for d in ends:
        lo = d - pd.DateOffset(months=months)
        sel = pos[(pos.index > lo) & (pos.index <= d)].to_numpy()
        if len(sel) < need or len(sel) <= len(cols) + 2:
            continue
        o = ols(y_all[sel], X_all[sel])
        dates.append(d.strftime("%Y-%m-%d"))
        per_year = 12 if freq == "monthly" else 252
        rows["Alpha"].append(round(float(o["coef"][0] * per_year), 5))
        for i, c in enumerate(cols):
            rows[c].append(round(float(o["coef"][i + 1]), 4))
        r2.append(round(float(o["r2"]), 4))
    return {"window_months": months, "dates": dates, "loadings": rows, "r_squared": r2}


def console(R: dict) -> str:
    L = [f"{R['model_label']} regression of {R['name']} ({R['freq']} excess returns, {R['start']} -> {R['end']}, "
         f"{R['observations']} observations)",
         f"  {'':8s} {'loading':>9s} {'t-stat':>7s} {'p-value':>8s}"]
    for c in R["coefficients"]:
        L.append(f"  {c['factor']:8s} {c['loading']:>9.4f} {c['t_stat']:>7.2f} {c['p_value']:>8.4f}{'  *' if c['significant'] else ''}")
    L.append(f"  Alpha {R['alpha_annual'] * 100:.2f}%/yr (t {R['alpha_t']:.2f})   R² {R['r_squared']:.3f}   adj. R² {R['adj_r_squared']:.3f}"
             f"   residual volatility {R['tracking_error_annual'] * 100:.1f}%/yr")
    L.append("  * |t| >= 1.96 (significant at about 5%)")
    return "\n".join(L)
