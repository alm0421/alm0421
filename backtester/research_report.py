"""Console output and self-contained HTML pages for sweeps, walk-forward runs and portfolio optimisation."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .report import _clean, num, pct

PAGE = Path(__file__).with_name("research_template.html")


def _page(kind: str, title: str, payload: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(_clean({"kind": kind, "title": title, **payload}), separators=(",", ":")).replace("</", "<\\/")
    p = out / "report.html"
    p.write_text(PAGE.read_text().replace("__TITLE__", html.escape(title[:80])).replace("__DATA__", blob))
    return p


def _series(eq: pd.Series) -> dict:
    return {"dates": [d.strftime("%Y-%m-%d") for d in eq.index], "values": [round(float(v), 2) for v in eq]}


# ------------------------------------------------------------------ sweep

def sweep_console(R: dict) -> str:
    L = [f"Parameter sweep: {len(R['rows'])} combinations, ranked by {R['objective']}",
         f"{'':3s} " + " ".join(f"{l[:14]:>14s}" for l in R["labels"]) + f" {'CAGR':>8s} {'Sharpe':>7s} {'MaxDD':>8s} {'Trades':>7s}"]
    for i, r in enumerate(R["ranked"][:15]):
        L.append(f"{i + 1:>3d} " + " ".join(f"{str(p):>14s}" for p in r["params"])
                 + f" {pct(r['cagr']):>8s} {num(r['sharpe']):>7s} {pct(r['max_drawdown'], 1):>8s} {r['trades']:>7d}")
    if R["errors"]:
        L.append(f"{len(R['errors'])} combination(s) failed, e.g. {R['errors'][0]['error']}")
    ok = R["ranked"]
    if len(ok) >= 5:
        vals = [r[R["objective"]] for r in ok if r[R["objective"]] is not None]
        L.append(f"Spread of {R['objective']}: best {num(vals[0])}, median {num(float(np.median(vals)))}, worst {num(vals[-1])}. "
                 "A best result far above the median is a sign of overfitting.")
    return "\n".join(L)


def write_sweep(R: dict, text: str, out: Path) -> Path:
    rows = [dict(r) for r in R["rows"]]
    table = [{**dict(zip(R["labels"], r["params"])),
              **{k: r.get(k) for k in ("cagr", "sharpe", "sortino", "calmar", "max_drawdown", "total_return", "trades",
                                       "win_rate", "profit_factor", "error")}} for r in rows]
    pd.DataFrame(table).to_csv(_mk(out) / "sweep.csv", index=False)
    curves = {k: _series(v) for k, v in R["curves"].items()}
    return _page("sweep", "Parameter sweep: " + text, {"text": text, "labels": R["labels"], "objective": R["objective"],
                                                          "rows": rows, "curves": curves}, out)


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ walk-forward

def walk_console(R: dict) -> str:
    L = [f"Walk-forward ({'anchored' if R['anchored'] else 'rolling'}), optimising {R['objective']} in-sample:",
         f"{'In-sample':25s} {'Out-of-sample':25s} {'Chosen parameters':28s} {'IS score':>9s} {'OOS return':>11s}"]
    for w in R["windows"]:
        L.append(f"{str(w['in_sample'][0]) + ' - ' + str(w['in_sample'][1]):25s} {str(w['out_sample'][0]) + ' - ' + str(w['out_sample'][1]):25s} "
                 f"{str(w['params']):28s} {num(w['is_score']):>9s} {pct(w['oos_return']):>11s}")
    s = R["oos_stats"]
    if s:
        L.append(f"Stitched out-of-sample: CAGR {pct(s['cagr'])}, Sharpe {num(s['sharpe'])}, max drawdown {pct(s['max_drawdown'], 1)}")
    hb = R.get("hindsight_best")
    if hb:
        L.append(f"Hindsight best over the same period {hb['params']}: CAGR {pct(hb['cagr'])}, Sharpe {num(hb['sharpe'])} "
                 "- the gap to the out-of-sample result is the price of not knowing the future.")
    return "\n".join(L)


def write_walk(R: dict, text: str, out: Path) -> Path:
    _mk(out)
    pd.DataFrame(R["windows"]).to_csv(out / "walkforward.csv", index=False)
    return _page("walk", "Walk-forward: " + text, {"text": text, "labels": R["labels"], "objective": R["objective"],
                                                  "anchored": R["anchored"], "windows": R["windows"],
                                                  "oos": _series(R["oos_equity"]) if len(R["oos_equity"]) else None,
                                                  "oos_stats": R["oos_stats"],
                                                  "hindsight": {k: v for k, v in (R.get("hindsight_best") or {}).items() if k != "equity"}}, out)


# ------------------------------------------------------------------ optimiser

def optimize_console(R: dict) -> str:
    L = [f"Mean-variance optimisation on monthly total returns {R['fit_start']} -> {R['fit_end']} (T-bill {pct(R['rf'])}):"]
    for name, p in R["portfolios"].items():
        w = ", ".join(f"{t} {x:.0%}" for t, x in sorted(p["weights"].items(), key=lambda kv: -kv[1]))
        L.append(f"  {name:18s} return {pct(p['exp_return'])}  vol {pct(p['exp_vol'])}  Sharpe {num(p['exp_sharpe'])}   [{w}]")
    if R.get("test"):
        L.append(f"Out of sample from {R.get('test_start')}:")
        for name, s in R["test"].items():
            L.append(f"  {name:18s} CAGR {pct(s['cagr'])}  vol {pct(s['volatility'])}  Sharpe {num(s['sharpe'])}  maxDD {pct(s['max_drawdown'], 1)}")
    L.append("Optimised weights fit the past; check them out of sample (--test-start) before trusting them.")
    return "\n".join(L)


def write_optimize(R: dict, out: Path) -> Path:
    _mk(out)
    return _page("optimize", "Portfolio optimisation: " + ", ".join(R["tickers"]), R, out)
