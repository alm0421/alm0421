"""Research tools: parameter sweeps, walk-forward optimisation and portfolio optimisation.

Parameters are written as placeholders anywhere in a plain-English sentence (or rule), e.g.

    buy QQQ when RSI({2..5}) is below {5..20 step 5}, hold {1,3,5} days

`{a..b}` is an integer range, `{a..b step s}` a stepped range, `{x,y,z}` a list.
"""
from __future__ import annotations

import dataclasses
import itertools
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import data, metrics, parser, runner

PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
OBJECTIVES = {"sharpe": "Sharpe", "cagr": "CAGR", "calmar": "Calmar", "sortino": "Sortino",
              "total_return": "Total return", "profit_factor": "Profit factor"}
MAX_COMBOS = 2000


def _values(spec: str) -> list:
    s = spec.strip()
    m = re.fullmatch(r"(-?[\d.]+)\s*\.\.\s*(-?[\d.]+)(?:\s+step\s+([\d.]+))?", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        step = float(m.group(3)) if m.group(3) else 1.0
        if step <= 0:
            raise ValueError("step must be positive")
        n = int(round((b - a) / step)) + 1
        vals = [a + i * step for i in range(max(n, 0))]
        return [int(v) if float(v).is_integer() else round(v, 10) for v in vals]
    parts = [p.strip() for p in s.split(",")]
    out = []
    for p in parts:
        try:
            v = float(p)
            out.append(int(v) if v.is_integer() else v)
        except ValueError:
            out.append(p)
    return out


def _label(text: str, m: re.Match) -> str:
    """A readable column label for one placeholder, with the placeholder written out in full.

    A placeholder glued to the word before it is labelled by that whole word ("RSI({2..5})"); otherwise the
    previous word is prefixed ("below {5..25 step 5}", "hold {1,3,5}", "at {1..3}%").
    """
    a, b = m.start(), m.end()
    while a > 0 and not text[a - 1].isspace():
        a -= 1
    while b < len(text) and not text[b].isspace():
        b += 1
    while b > m.end() and (text[b - 1] in ",.;:!?" or (text[b - 1] == ")" and text[a:b].count(")") > text[a:b].count("("))):
        b -= 1  # trailing punctuation and unmatched closing brackets are not part of the word
    while a < m.start() and text[a] == "(" and text[a:b].count("(") > text[a:b].count(")"):
        a += 1  # likewise an unmatched opening bracket
    word = text[a:b]
    if a < m.start():  # glued on the left: "RSI({2..5})" names itself
        return word
    before = text[:a].split()  # free-standing or only a suffix ("{1..3}%"): prefix the previous word
    return f"{before[-1].rstrip(',.;:')} {word}" if before else word


def expand(text: str) -> tuple[list[str], list[str], list[tuple]]:
    """-> (parameter labels, filled-in texts, value tuples)."""
    specs = PLACEHOLDER.findall(text)
    if not specs:
        raise ValueError("No {placeholders} found. Write e.g. 'hold {1..5} days' or 'RSI({2,3,4}) below {5..20 step 5}'.")
    grids = [_values(s) for s in specs]
    labels = []
    for m in PLACEHOLDER.finditer(text):
        lab = _label(text, m)
        k, base = 2, lab
        while lab in labels:  # labels key the CSV columns, so keep them unique
            lab, k = f"{base} #{k}", k + 1
        labels.append(lab)
    combos = list(itertools.product(*grids))
    if len(combos) > MAX_COMBOS:
        raise ValueError(f"{len(combos)} combinations; the limit is {MAX_COMBOS}. Use coarser steps.")
    texts = []
    for c in combos:
        it = iter(c)
        texts.append(PLACEHOLDER.sub(lambda m: str(next(it)), text))
    return labels, texts, combos


def _one(args) -> dict:
    text, overrides = args
    try:
        spec = parser.parse(text, **overrides)
        res = runner.run(spec)
        fl = res.extras.get("flows")
        st = metrics.equity_stats(res.equity, flows=fl)
        ts = metrics.trade_stats(res.trades, st["years"])
        return {"ok": True, "cagr": st["cagr"], "sharpe": st["sharpe"], "sortino": st["sortino"], "calmar": st["calmar"],
                "max_drawdown": st["max_drawdown"], "total_return": st["total_return"], "final": st["end_equity"],
                "trades": ts.get("trades", 0), "win_rate": ts.get("win_rate"), "profit_factor": ts.get("profit_factor"),
                "skew": st["skew"], "kurtosis": st["kurtosis"], "n_obs": max(len(res.equity) - 1, 0),
                "equity": res.equity}
    except Exception as e:  # noqa: BLE001 - one bad combination shouldn't sink the sweep
        return {"ok": False, "error": str(e).splitlines()[0]}


def _workers(n: int) -> int:
    return max(1, min(n, (os.cpu_count() or 2)))


def sweep(text: str, objective: str = "sharpe", overrides: dict | None = None, keep_equity: int = 5,
          parallel: bool = True) -> dict:
    labels, texts, combos = expand(text)
    args = [(t, overrides or {}) for t in texts]
    if parallel and len(args) > 3:
        with ProcessPoolExecutor(max_workers=_workers(len(args))) as ex:
            outs = list(ex.map(_one, args, chunksize=max(1, len(args) // 32)))
    else:
        outs = [_one(a) for a in args]
    rows = []
    for c, t, o in zip(combos, texts, outs):
        row = {"params": list(c), "text": t, **{k: v for k, v in o.items() if k != "equity"}}
        rows.append(row)
    ok = [r for r in rows if r["ok"]]
    key = lambda r: (r.get(objective) if r.get(objective) is not None and np.isfinite(r.get(objective) or np.nan) else -np.inf)  # noqa: E731
    ok.sort(key=key, reverse=True)
    best = ok[:keep_equity]
    curves = {}
    for r in best:
        i = rows.index(r)
        curves[" / ".join(str(p) for p in r["params"])] = outs[i]["equity"]
    return {"labels": labels, "objective": objective, "rows": rows, "ranked": ok, "curves": curves,
            "errors": [r for r in rows if not r["ok"]], "multiple_testing": multiple_testing(rows)}


def multiple_testing(rows: list[dict]) -> dict:
    """How impressive the best Sharpe is, given how many combinations were tried (Deflated Sharpe Ratio).

    Only combinations that traded count as trials: a combination that never trades has no Sharpe.
    """
    trials = [r for r in rows if r.get("ok") and r.get("trades") and metrics._finite(r.get("sharpe"))]
    if not trials:
        return {"n_trials": 0, "n_combinations": len(rows)}
    best = max(trials, key=lambda r: r["sharpe"])
    out = metrics.deflated_sharpe(best["sharpe"], [r["sharpe"] for r in trials], best.get("n_obs") or 0,
                                  best.get("skew") or 0.0, best.get("kurtosis") or 0.0)
    out.update({"n_combinations": len(rows), "best_params": list(best["params"])})
    return out


def walk_forward(text: str, objective: str = "sharpe", in_sample_years: float = 5, out_sample_years: float = 1,
                 anchored: bool = False, start: str | None = None, end: str | None = None,
                 overrides: dict | None = None) -> dict:
    """Optimise on each in-sample window, trade the winner on the following out-of-sample window."""
    labels, texts, combos = expand(text)
    first = parser.parse(texts[0], **(overrides or {}))
    probe = runner.run(first)
    t0 = pd.Timestamp(start) if start else probe.equity.index[1]
    t1 = pd.Timestamp(end) if end else probe.equity.index[-1]
    windows = []
    is_start = t0
    oos_start = t0 + pd.DateOffset(months=int(in_sample_years * 12))
    while oos_start < t1:
        oos_end = min(oos_start + pd.DateOffset(months=int(out_sample_years * 12)) - pd.Timedelta(days=1), t1)
        windows.append((is_start, oos_start - pd.Timedelta(days=1), oos_start, oos_end))
        oos_start = oos_end + pd.Timedelta(days=1)
        if not anchored:
            is_start = oos_start - pd.DateOffset(months=int(in_sample_years * 12))
    if not windows:
        raise ValueError("Period too short for the chosen in-sample window.")
    results = []
    oos_curve = []
    level = 10_000.0
    for (a, b, c, d) in windows:
        ov_is = dict(overrides or {}, start=str(a.date()), end=str(b.date()))
        args = [(t, ov_is) for t in texts]
        with ProcessPoolExecutor(max_workers=_workers(len(args))) as ex:
            outs = list(ex.map(_one, args, chunksize=max(1, len(args) // 32)))
        scored = [(o.get(objective) if o["ok"] and o.get(objective) is not None and np.isfinite(o.get(objective)) else -np.inf, i) for i, o in enumerate(outs)]
        score, bi = max(scored)
        if not np.isfinite(score):
            results.append({"in_sample": [a.date(), b.date()], "out_sample": [c.date(), d.date()], "params": None,
                            "is_score": None, "oos_return": None, "oos_sharpe": None})
            continue
        ov_oos = dict(overrides or {}, start=str(c.date()), end=str(d.date()), capital=level)
        o = _one((texts[bi], ov_oos))
        if not o["ok"]:
            continue
        eq = o["equity"]
        seg = eq.iloc[1:] if oos_curve else eq
        oos_curve.append(seg)
        level = float(eq.iloc[-1])
        results.append({"in_sample": [a.date(), b.date()], "out_sample": [c.date(), d.date()],
                        "params": list(combos[bi]), "is_score": score, "oos_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
                        "oos_sharpe": o["sharpe"], "oos_max_drawdown": o["max_drawdown"]})
    stitched = pd.concat(oos_curve) if oos_curve else pd.Series(dtype=float)
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    oos_stats = metrics.equity_stats(stitched) if len(stitched) > 30 else {}
    # full-period in-sample optimum for comparison (the overfitting gap)
    full = sweep(text, objective, dict(overrides or {}, start=str(windows[0][2].date()), end=str(t1.date())), keep_equity=1)
    best_full = full["ranked"][0] if full["ranked"] else None
    return {"labels": labels, "objective": objective, "anchored": anchored, "windows": results,
            "oos_equity": stitched, "oos_stats": oos_stats, "hindsight_best": best_full}


# ------------------------------------------------------------------ portfolio optimisation

def optimize(tickers: list[str], start: str | None = None, end: str | None = None, max_weight: float = 1.0,
             min_weight: float = 0.0, test_start: str | None = None, points: int = 30) -> dict:
    """Mean-variance optimisation on monthly total returns (long-only).

    If test_start is given, weights are estimated before it and evaluated after it (out of sample).
    """
    from scipy.optimize import minimize

    tickers = [data.canonical(t) for t in tickers]
    px = pd.concat({t: data.load(t)["adj_close"] for t in tickers}, axis=1).dropna()
    if start:
        px = px[px.index >= pd.Timestamp(start)]
    if end:
        px = px[px.index <= pd.Timestamp(end)]
    fit = px if not test_start else px[px.index < pd.Timestamp(test_start)]
    mr = metrics.monthly_returns_frame(fit)
    if len(mr) < 24:
        raise ValueError("Need at least 24 months of overlapping history for these tickers.")
    mu = mr.mean().to_numpy() * 12
    cov = mr.cov().to_numpy() * 12
    n = len(tickers)
    rf = float(data.tbill_rate().reindex(mr.index, method="ffill").mean()) if not data.tbill_rate().empty else 0.0
    bounds = [(min_weight, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    w0 = np.full(n, 1 / n)

    def vol(w):
        return float(np.sqrt(w @ cov @ w))

    def solve(obj, extra=()):
        r = minimize(obj, w0, bounds=bounds, constraints=cons + list(extra), method="SLSQP", options={"maxiter": 500})
        return r.x if r.success else None

    w_minvar = solve(vol)
    w_maxsharpe = solve(lambda w: -((w @ mu - rf) / vol(w)))
    inv = 1 / np.sqrt(np.diag(cov))
    w_rp = inv / inv.sum()
    frontier = []
    lo_r = float(w_minvar @ mu) if w_minvar is not None else float(mu.min())
    for target in np.linspace(lo_r, float(mu.max()), points):
        w = solve(vol, [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}])
        if w is not None:
            frontier.append({"return": float(w @ mu), "vol": vol(w), "weights": w.round(4).tolist()})
    ports = {}
    for name, w in (("Max Sharpe", w_maxsharpe), ("Min variance", w_minvar), ("Inverse volatility", w_rp), ("Equal weight", w0)):
        if w is None:
            continue
        w = np.clip(w, 0, None)
        w = w / w.sum()
        ports[name] = {"weights": {t: float(x) for t, x in zip(tickers, w) if x > 1e-4},
                       "exp_return": float(w @ mu), "exp_vol": vol(w), "exp_sharpe": float((w @ mu - rf) / vol(w))}
    assets = [{"ticker": t, "return": float(mu[i]), "vol": float(np.sqrt(cov[i, i]))} for i, t in enumerate(tickers)]
    out = {"tickers": tickers, "fit_start": mr.index[0].date(), "fit_end": mr.index[-1].date(), "rf": rf,
           "portfolios": ports, "frontier": frontier, "assets": assets,
           "correlation": mr.corr().round(3).to_numpy().tolist()}
    if test_start:
        test = px[px.index >= pd.Timestamp(test_start)]
        out["test_start"] = test.index[0].date() if len(test) else None
        out["test"] = {}
        for name, p in ports.items():
            w = pd.Series(p["weights"]).reindex(tickers).fillna(0)
            ret = (test.pct_change().fillna(0) @ w)
            eq = 10_000 * (1 + ret).cumprod()
            if len(eq) > 20:
                st = metrics.equity_stats(eq)
                out["test"][name] = {"cagr": st["cagr"], "volatility": st["volatility"], "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"]}
    return out
