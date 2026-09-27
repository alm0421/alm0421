"""Principal component analysis of asset returns (Portfolio Visualizer's "Principal Component Analysis").

    python -m backtester pca SPY EFA EEM AGG TLT GLD [--freq monthly] [--basis correlation|covariance] [--start 2005]

The returns are the assets' total returns (adjusted close, dividends reinvested) over the period they all have in
common, monthly (month-end to month-end, complete months only) or daily, as the correlation tool reads them
(correlation.returns_frame). The components are the eigenvectors of their covariance matrix (basis "covariance":
assets with larger swings weigh more) or of their correlation matrix (basis "correlation", the default: every
asset standardised to unit variance first). For each component:

- explained variance: its eigenvalue over the sum of all of them (the share of the total variance it accounts
  for), and the cumulative share;
- loadings: the eigenvector (one weight per asset; the vector has length 1). The sign of an eigenvector is
  arbitrary: it is chosen so that the loadings add up to a positive number (the first component then reads as
  "the market": every asset moving together);
- on the covariance basis, the component's annualised volatility (the square root of the eigenvalue, annualised
  on the returns' frequency); on the correlation basis the eigenvalue itself (in "number of assets": the average
  is 1, so a component above 1 explains more than one asset's worth of variance); and its correlation with each
  asset's returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import correlation

BASES = ("correlation", "covariance")


def analyze(tickers: list[str], freq: str = "monthly", basis: str = "correlation", start=None, end=None) -> dict:
    from .parser import resolve_asset_list
    tickers, notes = resolve_asset_list(tickers)
    notes = list(notes)
    if len(tickers) < 2:
        raise ValueError("Give at least two tickers for a principal component analysis.")
    if len(tickers) > correlation.MAX_ASSETS:
        raise ValueError(f"Up to {correlation.MAX_ASSETS} tickers at a time.")
    if freq not in ("daily", "monthly"):
        raise ValueError("freq must be daily or monthly")
    if basis not in BASES:
        raise ValueError(f"basis must be one of {', '.join(BASES)}")
    r, first = correlation.returns_frame(tickers, freq, start, end)
    r = r.dropna()
    k = len(tickers)
    if len(r) < k + 2:
        raise ValueError(f"Not enough common history: {len(r)} {freq} returns for {k} assets (need at least {k + 2}).")
    comps, sd, share = components(r[tickers], basis, freq)
    late = max(first, key=lambda t: first[t])
    notes.append(f"Common period {r.index[0].date()} to {r.index[-1].date()} ({len(r)} {freq} returns), from when "
                 f"{late} has data.")
    n90 = int(np.searchsorted(np.cumsum(share), 0.9 - 1e-12) + 1)
    per_year = 12 if freq == "monthly" else 252
    return {"tickers": tickers, "freq": freq, "basis": basis, "start": r.index[0].date(), "end": r.index[-1].date(),
            "observations": int(len(r)), "components": comps, "components_for_90pct": min(n90, k),
            "asset_volatility_annual": {t: float(v * np.sqrt(per_year)) for t, v in zip(tickers, sd)},
            "notes": notes}


def components(r: pd.DataFrame, basis: str = "correlation", freq: str = "monthly") -> tuple[list[dict], np.ndarray, np.ndarray]:
    """The principal components of a (periods x assets) frame of returns: (components, each asset's standard
    deviation, the explained shares). See the module docstring."""
    tickers = list(r.columns)
    k = len(tickers)
    X = r.to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    if (sd <= 0).any():
        flat = [t for t, v in zip(tickers, sd) if v <= 0]
        raise ValueError(f"{', '.join(flat)} did not move over the period: no variance to analyse.")
    Z = X / sd if basis == "correlation" else X
    M = np.cov(Z, rowvar=False)
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]
    for j in range(k):                       # sign convention: the loadings add up to a positive number
        s = vecs[:, j].sum()
        if s < 0 or (abs(s) < 1e-12 and vecs[np.argmax(np.abs(vecs[:, j])), j] < 0):
            vecs[:, j] = -vecs[:, j]
    share = vals / vals.sum() if vals.sum() > 0 else np.zeros(k)
    per_year = 12 if freq == "monthly" else 252
    scores = Z @ vecs                        # each period's value of each component
    comps = []
    for j in range(k):
        corr = [float(np.corrcoef(scores[:, j], X[:, i])[0, 1]) if scores[:, j].std() > 0 else float("nan") for i in range(k)]
        comps.append({"component": f"PC{j + 1}", "eigenvalue": float(vals[j]), "explained": float(share[j]),
                      "cumulative": float(share[: j + 1].sum()),
                      # the component's own volatility: meaningful in return units (covariance basis) only
                      "volatility_annual": float(np.sqrt(vals[j] * per_year)) if basis == "covariance" else None,
                      "loadings": {t: float(vecs[i, j]) for i, t in enumerate(tickers)},
                      "correlation_with_assets": {t: c for t, c in zip(tickers, corr)}})
    return comps, sd, share


def console(R: dict) -> str:
    t = R["tickers"]
    w = max(9, max(len(x) for x in t) + 1)
    L = [f"Principal components of {R['freq']} total returns ({R['basis']} matrix), {R['start']} -> {R['end']} "
         f"({R['observations']} observations)",
         f"{'':6s} {'explained':>10s} {'cumulative':>11s} {'vol/yr' if R['basis'] == 'covariance' else 'eigenval':>8s}  loadings: "
         + "".join(f"{x:>{w}s}" for x in t)]
    for c in R["components"]:
        size = (f"{c['volatility_annual'] * 100:>7.1f}%" if R["basis"] == "covariance" else f"{c['eigenvalue']:>8.2f}")
        L.append(f"{c['component']:6s} {c['explained'] * 100:>9.1f}% {c['cumulative'] * 100:>10.1f}% "
                 f"{size}  {'':10s}" + "".join(f"{c['loadings'][x]:>{w}.3f}" for x in t))
    L.append(f"{R['components_for_90pct']} component(s) explain 90% of the variance. Loadings are the eigenvectors "
             "(length 1; signed so they add up to a positive number)"
             + ("; on the correlation basis each asset is standardised first (eigenvalues average 1)."
                if R["basis"] == "correlation" else "; on the covariance basis the more volatile assets weigh more."))
    for n in R["notes"]:
        L.append("Note: " + n)
    return "\n".join(L)
