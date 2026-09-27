"""Asset correlations and statistics (Portfolio Visualizer's "asset correlations" and "asset analysis").

    python -m backtester correlation SPY TLT GLD EFASIM --window 36 --freq monthly [--start 2000] [--pair SPY,TLT]

- The correlation matrix of daily or monthly total returns (adjusted close, dividends reinvested) over the
  period all the assets have in common (optionally narrowed with start / end).
- A rolling correlation of a chosen pair over `window` periods (36 months, or 63 days for daily returns),
  over the pair's own common history, which can be longer than the whole group's.
- Per-asset statistics over the common period (CAGR, volatility, Sharpe against T-bills, max drawdown,
  best / worst calendar year) and each asset's own first date of data, on the matrix's frequency: with
  monthly returns every asset's volatility, Sharpe and Sortino come from monthly returns and its max
  drawdown from month-end values.
Monthly returns are month-end to month-end; a month still in progress at the end of the data is left out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, metrics

MAX_ASSETS = 25


def _prices(tickers: list[str]) -> dict[str, pd.Series]:
    out = {}
    for t in tickers:
        s = data.load(t)["adj_close"].dropna()
        out[t] = s[s > 0]
    return out


def _complete_months(px: pd.DataFrame) -> pd.DataFrame:
    return metrics.complete_months(px)


def returns_frame(tickers: list[str], freq: str = "monthly", start=None, end=None) -> tuple[pd.DataFrame, dict]:
    """(returns over the common period, {ticker: first date of its data})."""
    px_all = _prices(tickers)
    first = {t: s.index[0].date() for t, s in px_all.items()}
    px = pd.concat(px_all, axis=1, sort=True).dropna()
    if start:
        px = px[px.index >= pd.Timestamp(start)]
    if end:
        px = px[px.index <= pd.Timestamp(end)]
    if len(px) < 3:
        raise ValueError(f"{', '.join(tickers)} have no common history in the chosen period "
                         f"(first dates: {', '.join(f'{t} {d}' for t, d in first.items())}).")
    if freq == "monthly":
        r = _complete_months(px).pct_change().iloc[1:]
    else:
        r = px.pct_change().iloc[1:]
    return r.dropna(how="all"), first


def _pair_rolling(a: str, b: str, freq: str, window: int, start=None, end=None) -> dict:
    r, _ = returns_frame([a, b], freq, start, end)
    rc = r[a].rolling(window).corr(r[b]).dropna()
    return {"pair": [a, b], "window": window, "freq": freq,
            "dates": [d.strftime("%Y-%m-%d") for d in rc.index], "values": [round(float(v), 4) for v in rc],
            "full_period": round(float(r[a].corr(r[b])), 4) if len(r) > 2 else None,
            "start": r.index[0].date() if len(r) else None}


def asset_stats(ticker: str, start, end, rf="tbill", first_date=None, basis: str | None = None) -> dict:
    """PV-style statistics of one asset's total return over [start, end]. basis="monthly" puts every asset on
    monthly returns (volatility, Sharpe, Sortino, and the max drawdown from month-end values), as the correlation
    matrix is when it is monthly; otherwise only a series moving in monthly steps is."""
    s = data.load(ticker)["adj_close"].dropna()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    if len(s) < 3:
        return {"ticker": ticker, "start": str(first_date) if first_date else None}
    st = metrics.equity_stats(s / s.iloc[0] * 10_000, rf)
    stepped = bool(data.stepped_in([ticker], s.index[0], s.index[-1])) or basis == "monthly"
    if stepped:
        # monthly returns: volatility, Sharpe, Sortino and drawdown from month-end values
        st = metrics.monthly_basis(st, s, rf)
        st["max_drawdown_daily"] = st.get("max_drawdown")
        st["max_drawdown"] = metrics.monthly_max_drawdown(s)
    mr = metrics.monthly_returns(s)
    return {"ticker": ticker, "data_from": str(first_date or s.index[0].date()), "from": str(s.index[0].date()),
            "to": str(s.index[-1].date()), "cagr": st["cagr"], "volatility": st["volatility"], "sharpe": st["sharpe"],
            "sortino": st["sortino"], "max_drawdown": st["max_drawdown"], "best_year": st["best_year"],
            "worst_year": st["worst_year"], "total_return": st["total_return"],
            **({"max_drawdown_daily": st["max_drawdown_daily"]} if "max_drawdown_daily" in st else {}),
            "monthly_volatility": float(mr.std() * np.sqrt(12)) if len(mr) > 2 else None,
            **({"return_basis": "monthly"} if stepped else {})}


def analyze(tickers: list[str], freq: str = "monthly", window: int | None = None, start=None, end=None,
            pair: list[str] | None = None, rf="tbill") -> dict:
    from .parser import resolve_asset_list
    tickers, name_notes = resolve_asset_list(tickers)       # asset-class names -> their series, as in the parser
    if pair:
        pair = resolve_asset_list(pair)[0]
    if len(tickers) < 2:
        raise ValueError("Give at least two tickers.")
    if len(tickers) > MAX_ASSETS:
        raise ValueError(f"Up to {MAX_ASSETS} tickers at a time.")
    if freq not in ("daily", "monthly"):
        raise ValueError("freq must be daily or monthly")
    if start and end and pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError(f"The start {start} is not before the end {end}.")
    notes = list(name_notes)
    if freq == "daily":
        # a series that moves in monthly steps (a monthly source spread over daily sessions) has no daily
        # co-movement to measure: use monthly returns
        px = pd.concat(_prices(tickers), axis=1, sort=True).dropna()
        lo = max(px.index[0], pd.Timestamp(start)) if start and len(px) else (px.index[0] if len(px) else None)
        hi = min(px.index[-1], pd.Timestamp(end)) if end and len(px) else (px.index[-1] if len(px) else None)
        found = data.stepped_in(tickers, lo, hi) if lo is not None else {}
        if found:
            freq = "monthly"
            window = None
            notes.append(f"Monthly returns used: {data.stepped_text(found)} {'move' if len(found) > 1 else 'moves'} only once a month in this data (a "
                         "monthly source), so daily correlations would be meaningless.")
    window = int(window or (36 if freq == "monthly" else 63))
    if window < 3:
        raise ValueError("The rolling window must be at least 3 periods.")
    r, first = returns_frame(tickers, freq, start, end)
    if len(r) < 3:
        raise ValueError("Not enough common history for a correlation.")
    C = r.corr()
    late = max(first, key=lambda t: first[t])
    if not start:
        notes.append(f"The common period starts {r.index[0].date()}, when {late} has data (it starts {first[late]}); "
                     "each asset's own history may be longer.")
    pair = [data.canonical(x) for x in (pair or tickers[:2])]
    if len(pair) != 2 or pair[0] == pair[1] or any(p not in tickers for p in pair):
        raise ValueError("The rolling pair must be two different tickers from the list.")
    roll = _pair_rolling(pair[0], pair[1], freq, window, start, end)
    if not roll["values"]:
        notes.append(f"Not enough history for a {window}-{'month' if freq == 'monthly' else 'day'} rolling window.")
    a, b = r.index[0], r.index[-1]
    # stats over the common period of prices (the first return's base day onward)
    px0 = pd.concat({t: data.load(t)["adj_close"] for t in tickers}, axis=1, sort=True).dropna()
    if start:
        px0 = px0[px0.index >= pd.Timestamp(start)]
    if end:
        px0 = px0[px0.index <= pd.Timestamp(end)]
    s0, s1 = px0.index[0], px0.index[-1]
    stats = [asset_stats(t, s0, s1, rf, first[t], basis=freq) for t in tickers]
    return {"tickers": tickers, "freq": freq, "stats_basis": freq, "window": window, "start": a.date(), "end": b.date(),
            "stats_start": s0.date(), "stats_end": s1.date(), "observations": int(len(r)),
            "matrix": [[round(float(x), 4) for x in row] for row in C.to_numpy()],
            "average_correlation": {t: round(float((C[t].sum() - 1) / (len(tickers) - 1)), 4) for t in tickers},
            "rolling": roll, "stats": stats, "first_dates": {t: str(d) for t, d in first.items()}, "notes": notes}


def console(R: dict) -> str:
    t = R["tickers"]
    w = max(8, max(len(x) for x in t) + 1)
    L = [f"Correlation of {R['freq']} total returns, {R['start']} -> {R['end']} ({R['observations']} observations)",
         " " * w + "".join(f"{x:>{w}s}" for x in t)]
    for i, x in enumerate(t):
        L.append(f"{x:<{w}s}" + "".join(f"{v:>{w}.2f}" for v in R["matrix"][i]))
    ro = R["rolling"]
    if ro["values"]:
        v = np.array(ro["values"])
        L.append(f"Rolling {ro['window']}-{'month' if R['freq'] == 'monthly' else 'day'} correlation {ro['pair'][0]} / {ro['pair'][1]}: "
                 f"latest {v[-1]:.2f} ({ro['dates'][-1]}), min {v.min():.2f}, max {v.max():.2f}, whole period {ro['full_period']:.2f}")
    basis = ("monthly returns, like the matrix: volatility, Sharpe and Sortino from monthly returns, max drawdown "
             "from month-end values" if R.get("stats_basis") == "monthly" else "daily returns")
    L.append(f"Asset statistics {R['stats_start']} -> {R['stats_end']} (total returns; Sharpe against T-bills; {basis})")
    L.append(f"{'':{w}s} {'CAGR':>8s} {'Vol':>7s} {'Sharpe':>7s} {'MaxDD':>8s} {'Best yr':>8s} {'Worst yr':>9s}  data from")
    p = lambda v, d=1: "n/a" if v is None or not np.isfinite(v) else f"{v * 100:.{d}f}%"  # noqa: E731
    for s in R["stats"]:
        sh = s.get("sharpe")
        L.append(f"{s['ticker']:<{w}s} {p(s.get('cagr'), 2):>8s} {p(s.get('volatility')):>7s} "
                 f"{('n/a' if sh is None or not np.isfinite(sh) else f'{sh:.2f}'):>7s} {p(s.get('max_drawdown')):>8s} "
                 f"{p(s.get('best_year')):>8s} {p(s.get('worst_year')):>9s}  {s.get('data_from')}")
    for n in R["notes"]:
        L.append("Note: " + n)
    return "\n".join(L)
