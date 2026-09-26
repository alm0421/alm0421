"""Console output and self-contained HTML pages for sweeps, walk-forward runs and portfolio optimisation."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .report import _clean, _fit, num, pct

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
    widths = [max(len(lab), 6) for lab in R["labels"]]
    L = [f"Parameter sweep: {len(R['rows'])} combinations, ranked by {R['objective']}",
         f"{'':3s} " + " ".join(f"{lab:>{w}s}" for lab, w in zip(R["labels"], widths))
         + f" {'CAGR':>8s} {'Sharpe':>7s} {'MaxDD':>8s} {'Trades':>7s}"]
    for i, r in enumerate(R["ranked"][:15]):
        L.append(f"{i + 1:>3d} " + " ".join(f"{str(p):>{w}s}" for p, w in zip(r["params"], widths))
                 + f" {pct(r['cagr']):>8s} {num(r['sharpe']):>7s} {pct(r['max_drawdown'], 1):>8s} {r['trades']:>7d}")
    if R["errors"]:
        L.append(f"{len(R['errors'])} combination(s) failed, e.g. {R['errors'][0]['error']}")
    ok = R["ranked"]
    if len(ok) >= 5:
        vals = [r[R["objective"]] for r in ok if r[R["objective"]] is not None]
        L.append(f"Spread of {R['objective']}: best {num(vals[0])}, median {num(float(np.median(vals)))}, worst {num(vals[-1])}. "
                 "A best result far above the median is a sign of overfitting.")
    note = multiple_testing_note(R.get("multiple_testing") or {})
    if note:
        L.append(note)
    return "\n".join(L)


def multiple_testing_note(mt: dict) -> str:
    """One paragraph on selection bias: best Sharpe vs the best expected from luck, and the Deflated Sharpe."""
    n = mt.get("n_trials") or 0
    if n < 2 or mt.get("expected_max_sharpe") is None or not np.isfinite(mt.get("expected_max_sharpe", np.nan)):
        return ""
    dsr = mt.get("dsr")
    s = (f"Multiple testing: {n} combinations traded. Best Sharpe {num(mt['best_sharpe'])} {mt.get('best_params')}; "
         f"with Sharpes spread by {num(mt['sd_sharpe'])}, the best of {n} strategies with no edge would be expected "
         f"to reach about {num(mt['expected_max_sharpe'])} by luck alone (the threshold the Deflated Sharpe below tests against).")
    if dsr is not None and np.isfinite(dsr):
        s += (f" Deflated Sharpe Ratio {pct(dsr, 1)}: the probability that the best combination's true Sharpe is above zero "
              f"after allowing for the {n} tries" + (" (below 95%: not significant)." if dsr < 0.95 else "."))
    return s


def write_sweep(R: dict, text: str, out: Path) -> Path:
    rows = [dict(r) for r in R["rows"]]
    table = [{**dict(zip(R["labels"], r["params"])),
              **{k: r.get(k) for k in ("cagr", "sharpe", "sortino", "calmar", "max_drawdown", "total_return", "trades",
                                       "win_rate", "profit_factor", "error")}} for r in rows]
    pd.DataFrame(table).to_csv(_mk(out) / "sweep.csv", index=False)
    curves = {k: _series(v) for k, v in R["curves"].items()}
    return _page("sweep", "Parameter sweep: " + text, {"text": text, "labels": R["labels"], "objective": R["objective"],
                                                          "rows": rows, "curves": curves,
                                                          "multiple_testing": R.get("multiple_testing") or {},
                                                          "multiple_testing_note": multiple_testing_note(R.get("multiple_testing") or {})}, out)


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ walk-forward

def walk_console(R: dict) -> str:
    L = [f"Walk-forward ({'anchored' if R['anchored'] else 'rolling'}), optimising {R['objective']} in-sample:",
         f"{'In-sample':25s} {'Out-of-sample':25s} {'Chosen parameters':28s} {'IS score':>9s} {'OOS return':>11s}"]
    for w in R["windows"]:
        L.append(f"{str(w['in_sample'][0]) + ' - ' + str(w['in_sample'][1]):25s} {str(w['out_sample'][0]) + ' - ' + str(w['out_sample'][1]):25s} "
                 f"{_fit(str(w['params']), 28)} {num(w['is_score']):>9s} {pct(w['oos_return']):>11s}")
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
    L = [f"Portfolio optimisation on monthly total returns {R['fit_start']} -> {R['fit_end']} (T-bill {pct(R['rf'])}):"]
    c = R.get("constraints") or {}
    lim = [("" if a is None else pct(a, 0) + " <= ") + t + ("" if b is None else " <= " + pct(b, 0))
           for t, (a, b) in (c.get("bounds") or {}).items()] + [g["text"] for g in c.get("groups") or []]
    if lim:
        L.append("Constraints: " + "; ".join(lim))
    inp = R.get("inputs") or {}
    if inp.get("source") in ("forecast", "black_litterman"):
        L.append("Inputs: " + ("forecasts given" if inp["source"] == "forecast" else "Black-Litterman posterior")
                 + " (historical in brackets)")
        for r in inp.get("table") or []:
            L.append(f"  {r['ticker']:8s} return {pct(r['return'], 1):>7s} ({pct(r['hist_return'], 1)})   "
                     f"vol {pct(r['vol'], 1):>6s} ({pct(r['hist_vol'], 1)})")
    bl = R.get("black_litterman")
    if bl:
        L.append(f"Black-Litterman: prior {bl['prior']}, tau {bl['tau']:g}, risk aversion {bl['risk_aversion']:g}")
        L.append(f"  {'':8s} {'prior w':>8s} {'equilib.':>9s} {'posterior':>9s} {'history':>8s}")
        for t, w in bl["prior_weights"].items():
            L.append(f"  {t:8s} {pct(w, 1):>8s} {pct(bl['equilibrium_returns'][t], 1):>9s} {pct(bl['posterior_returns'][t], 1):>9s} "
                     f"{pct(bl['historical_returns'][t], 1):>8s}")
        for v in bl["views"]:
            L.append(f"  view {v['text']!r}: confidence {pct(v['confidence'], 0)}, prior implied {pct(v['prior_value'], 2)}"
                     f" -> posterior {pct(v['posterior_value'], 2)}")
    b = R.get("benchmark")
    if b:
        L.append(f"Benchmark {b['name']}: expected return {pct(b['return'], 1)}, volatility {pct(b['vol'], 1)}"
                 + ("" if b.get("investable") else " (not made of these tickers: tracking error from its monthly returns)"))
    L.append(f"  {'':26s} {'return':>7s} {'vol':>7s} {'Sharpe':>6s} {'Sortino':>7s} {'CVaR95m':>7s} {'DivR':>5s}"
             + (f" {'TE':>6s} {'IR':>6s}" if b else ""))
    for name, p in R["portfolios"].items():
        L.append(f"  {_fit(name, 26)} {pct(p['exp_return'], 1):>7s} {pct(p['exp_vol'], 1):>7s} {num(p['exp_sharpe']):>6s} "
                 f"{num(p.get('exp_sortino')):>7s} {pct(p.get('cvar_95_monthly'), 1):>7s} {num(p.get('diversification_ratio')):>5s}"
                 + (f" {pct(p.get('tracking_error'), 1):>6s} {num(p.get('information_ratio')):>6s}" if b else ""))
        L.append(f"  {'':26s} {p.get('sentence', '')}")
        if p.get("weights_sd"):
            L.append(f"  {'':26s} spread across draws: " + ", ".join(f"{t} ±{pct(x, 0)}" for t, x in p["weights_sd"].items()))
    rs = R.get("resampled")
    if rs:
        L.append(rs["note"])
    for n in R.get("notes") or []:
        L.append("Note: " + n)
    if R.get("test"):
        L.append(f"Out of sample from {R.get('test_start')}:")
        for name, s in R["test"].items():
            L.append(f"  {_fit(name, 26)} CAGR {pct(s['cagr'])}  vol {pct(s['volatility'])}  Sharpe {num(s['sharpe'])}  maxDD {pct(s['max_drawdown'], 1)}")
    ro = R.get("rolling")
    if ro:
        L.append(f"Walk-forward: re-optimised every {ro['every_months']} months on the trailing {ro['lookback_months']} months, "
                 f"{ro['start']} -> {ro['end']} (out of sample) vs the same weights fitted on the whole period (hindsight):")
        for name, s in ro["stats"].items():
            a, b = s.get("rolling", {}), s.get("static", {})
            L.append(f"  {_fit(name, 26)} rolling CAGR {pct(a.get('cagr'))} Sharpe {num(a.get('sharpe'))} maxDD {pct(a.get('max_drawdown'), 1)}"
                     + (f"   static CAGR {pct(b.get('cagr'))} Sharpe {num(b.get('sharpe'))} maxDD {pct(b.get('max_drawdown'), 1)}" if b else ""))
    L.append("Optimised weights fit the past; check them out of sample (--test-start) before trusting them.")
    return "\n".join(L)


def write_optimize(R: dict, out: Path) -> Path:
    _mk(out)
    return _page("optimize", "Portfolio optimisation: " + ", ".join(R["tickers"]), R, out)
