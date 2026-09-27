"""Returns-based style analysis (Sharpe, 1992, "Asset allocation: management style and performance
measurement"): the mix of asset-class returns that best tracks a fund's or portfolio's monthly returns,
with the weights constrained to be non-negative and to add up to 100%.

    minimise  Var(R - sum_i w_i F_i)   subject to  w_i >= 0,  sum_i w_i = 1

(Sharpe minimises the variance of the tracking difference, so a constant gap in return - the fund's
"selection" return - does not distort the weights.) R² = 1 - Var(tracking difference) / Var(R) is the share
of the variance explained by the style mix. Rolling weights re-estimate the mix on trailing windows
(36 months by default) to show style drift.

Default asset classes (the first series available in the local data is used for each; the long-history
"SIM" series reach back to the 1920s-1990s, the ETFs start later):
    US large value (VTVSIM, VTV, IWD), US large growth (VUGSIM, VUG, IWF), US small value (VBRSIM, VBR, IWN),
    US small growth (VBKSIM, VBK, IWO), international developed (EFASIM, EFA, VEA), emerging markets
    (EEMSIM, EEM, VWO), US Treasuries (IEFSIM, IEF), US corporates (LQDSIM, LQD), T-bills (BILSIM, BIL, SHY).
Any other set of tickers can be given instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

ASSET_CLASSES: list[tuple[str, list[str]]] = [
    ("US large value", ["VTVSIM", "VTV", "IWD", "IVE"]),
    ("US large growth", ["VUGSIM", "VUG", "IWF", "IVW"]),
    ("US small value", ["VBRSIM", "VBR", "IWN", "IJS"]),
    ("US small growth", ["VBKSIM", "VBK", "IWO", "IJT"]),
    ("Intl developed", ["EFASIM", "EFA", "VEA"]),
    ("Emerging markets", ["EEMSIM", "EEM", "VWO"]),
    ("US Treasuries", ["IEFSIM", "IEF"]),
    ("US corporates", ["LQDSIM", "LQD"]),
    ("T-bills", ["BILSIM", "BIL", "SHY"]),
]


def _have(t: str) -> bool:
    return (data.PRICES / f"{data.canonical(t)}.csv").exists()


def default_assets() -> tuple[dict[str, str], list[str]]:
    """{asset-class label: ticker} for the classes with local data, and notes for the ones without."""
    out, notes = {}, []
    for label, cands in ASSET_CLASSES:
        t = next((c for c in cands if _have(c)), None)
        if t:
            out[label] = t
        else:
            notes.append(f"{label} left out: none of {', '.join(cands)} is in the local data.")
    return out, notes


def monthly_returns(daily: pd.Series) -> pd.Series:
    """Monthly returns (calendar month-end index) of complete months from daily returns."""
    from .metrics import complete_months
    r = daily.dropna()
    if not len(r):
        return pd.Series(dtype=float)
    lvl = (1 + r).cumprod()
    me = complete_months(lvl)
    base = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)]), me])
    m = (base / base.shift(1) - 1).iloc[1:]
    # the first month counts only if the series starts in its first week (else it is a partial month)
    if r.index[0].day > 7:
        m = m.iloc[1:]
    m = m.dropna()
    m.index = pd.DatetimeIndex(m.index).to_period("M").to_timestamp("M")
    return m


def asset_monthly(ticker: str) -> pd.Series:
    px = data.load(ticker)["adj_close"]
    return monthly_returns(px.pct_change().dropna())


def solve(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Style weights: minimise the variance of y - X w with w >= 0 and sum(w) = 1 (quadratic programme
    solved with SLSQP, started from the non-negative least-squares fit)."""
    from scipy.optimize import minimize, nnls
    n, k = X.shape
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    # a good start: NNLS on the de-meaned data with a heavily weighted sum-to-one row
    big = 1e3 * max(1.0, float(np.abs(Xc).max()))
    w0, _ = nnls(np.vstack([Xc, np.full((1, k), big)]), np.concatenate([yc, [big]]))
    w0 = w0 / w0.sum() if w0.sum() > 0 else np.full(k, 1 / k)
    H = Xc.T @ Xc
    g = Xc.T @ yc

    def f(w):
        e = yc - Xc @ w
        return float(e @ e), -2 * (g - H @ w)
    res = minimize(f, w0, jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(k)}],
                   options={"ftol": 1e-14, "maxiter": 500})
    w = res.x if res.success or f(res.x)[0] <= f(w0)[0] else w0
    w = np.clip(w, 0.0, None)
    return w / w.sum()


def _fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    w = solve(y, X)
    e = y - X @ w
    vy = y.var(ddof=1)
    r2 = 1 - e.var(ddof=1) / vy if vy > 0 else np.nan
    return w, float(r2), e


def analyze(returns: pd.Series, assets: dict[str, str] | list[str] | None = None, start=None, end=None,
            window: int = 36, name: str = "") -> dict:
    """Style analysis of daily returns `returns` on monthly asset-class returns. `assets` is {label: ticker},
    a list of tickers, or None for the default asset classes."""
    notes: list[str] = list(returns.attrs.get("input_notes") or [])
    if assets is None:
        assets, dn = default_assets()
        notes += dn
    elif isinstance(assets, (list, tuple, str)):
        from .parser import resolve_asset_list
        ticks, an = resolve_asset_list(assets)          # asset-class names -> their series, as in the parser
        notes += [n for n in an if n not in notes]
        assets = {t: t for t in ticks}
    if len(assets) < 2:
        raise ValueError("Style analysis needs at least two asset classes with data.")
    y = monthly_returns(returns)
    cols = {}
    for label, t in assets.items():
        try:
            cols[label] = asset_monthly(t)
        except (FileNotFoundError, data.DataError, KeyError) as e:
            notes.append(f"{label} ({t}) left out: {e}")
    if len(cols) < 2:
        raise ValueError("Style analysis needs at least two asset classes with data.")
    F = pd.DataFrame(cols)
    df = pd.concat([y.rename("__y"), F], axis=1, join="inner").dropna()
    if start:
        df = df[df.index >= pd.Timestamp(start) + pd.offsets.MonthEnd(0)]
    if end:
        df = df[df.index <= pd.Timestamp(end) + pd.offsets.MonthEnd(0)]
    labels = [c for c in df.columns if c != "__y"]
    if len(df) < max(24, len(labels) + 3):
        raise ValueError(f"Need at least {max(24, len(labels) + 3)} months that {name or 'the returns'} and the "
                         f"asset classes have in common; got {len(df)}.")
    Y, X = df["__y"].to_numpy(), df[labels].to_numpy()
    w, r2, e = _fit(Y, X)
    style_ret = X @ w
    out = {
        "name": name, "start": df.index[0].date(), "end": df.index[-1].date(), "observations": len(df),
        "assets": {lab: assets[lab] for lab in labels},
        "weights": {lab: float(v) for lab, v in zip(labels, w)},
        "r_squared": r2,
        "selection_return_annual": float(e.mean() * 12),
        "tracking_error_annual": float(e.std(ddof=1) * np.sqrt(12)),
        "fund_return_annual": float((np.prod(1 + Y)) ** (12 / len(Y)) - 1),
        "style_return_annual": float((np.prod(1 + style_ret)) ** (12 / len(Y)) - 1),
        "notes": notes,
        "rolling": rolling(df, labels, window),
    }
    return out


def rolling(df: pd.DataFrame, labels: list[str], window: int) -> dict:
    Y, X = df["__y"].to_numpy(), df[labels].to_numpy()
    dates, r2 = [], []
    rows = {lab: [] for lab in labels}
    if window >= 12 and len(df) >= window:
        for i in range(window, len(df) + 1):
            w, rr, _ = _fit(Y[i - window:i], X[i - window:i])
            dates.append(df.index[i - 1].strftime("%Y-%m-%d"))
            r2.append(round(rr, 4))
            for lab, v in zip(labels, w):
                rows[lab].append(round(float(v), 4))
    return {"window_months": window, "dates": dates, "weights": rows, "r_squared": r2}


def console(R: dict) -> str:
    L = [f"Returns-based style analysis of {R['name']} (monthly, {R['start']} -> {R['end']}, {R['observations']} months)",
         f"  {'asset class':20s} {'ticker':8s} {'weight':>8s}"]
    for lab, v in sorted(R["weights"].items(), key=lambda kv: -kv[1]):
        L.append(f"  {lab:20s} {R['assets'][lab]:8s} {v * 100:>7.1f}%")
    L.append(f"  R² {R['r_squared']:.3f}   selection return {R['selection_return_annual'] * 100:.2f}%/yr   "
             f"tracking error {R['tracking_error_annual'] * 100:.1f}%/yr")
    ro = R["rolling"]
    if ro["dates"]:
        last = {k: v[-1] for k, v in ro["weights"].items()}
        L.append(f"  Latest {ro['window_months']}-month window ({ro['dates'][-1]}): "
                 + ", ".join(f"{k} {v:.0%}" for k, v in sorted(last.items(), key=lambda kv: -kv[1]) if v >= 0.005))
    for n in R.get("notes") or []:
        L.append("  Note: " + n)
    return "\n".join(L)
