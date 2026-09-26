"""Build the backtest report: console summary, CSV/JSON outputs and a self-contained HTML page."""
from __future__ import annotations

import dataclasses
import html
import json
import math
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine, metrics
from .engine import Result

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).with_name("report_template.html")


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "backtest"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) else (f if math.isfinite(f) else ("inf" if f > 0 else "-inf"))
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.strftime("%Y-%m-%d")
    return o


def cost_sensitivity(res: Result, levels=(0, 5, 10, 25)) -> list[dict]:
    rows = []
    base = res.strategy.slippage_bps
    for bps in sorted(set(levels) | {base}):
        if bps == base:
            r = res
        else:
            s = dataclasses.replace(res.strategy, slippage_bps=float(bps), notes=list(res.strategy.notes))
            r = engine.run(s)
        st = metrics.equity_stats(r.equity)
        rows.append({"slippage_bps": bps, "final_equity": st["end_equity"], "cagr": st["cagr"],
                     "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"], "current": bps == base})
    return rows


def analyze(res: Result, rf: float = 0.0, sensitivity: bool = True, mc: bool = True) -> dict:
    s = res.strategy
    eq = res.equity
    stats = metrics.equity_stats(eq, rf)
    tstats = metrics.trade_stats(res.trades, stats["years"])
    expo = metrics.exposure_stats(res.exposure, res.positions)

    benches: dict[str, pd.Series] = {}
    if len(s.universe) == 1 and s.universe[0] not in ("SPY", "QQQ"):
        b = metrics.buy_and_hold(s.universe[0], eq.index, s.capital)
        if b is not None:
            benches[f"{s.universe[0]} buy & hold"] = b
    for t in ("SPY", "QQQ"):
        b = metrics.buy_and_hold(t, eq.index, s.capital)
        if b is not None:
            benches[f"{t} buy & hold"] = b
    bench_stats = {n: metrics.equity_stats(b, rf) for n, b in benches.items()}
    spy = benches.get("SPY buy & hold")
    rel = metrics.relative_stats(eq, spy, rf)

    yearly = metrics.yearly_detail(eq, res.trades, res.exposure)
    yr_bench = metrics.yearly_returns({n: b for n, b in benches.items()})
    for n in yr_bench:
        yearly[n] = yr_bench[n].reindex(yearly.index)

    return {
        "strategy": s,
        "stats": stats,
        "trade_stats": tstats,
        "exposure": expo,
        "relative": rel,
        "benchmarks": benches,
        "bench_stats": bench_stats,
        "yearly": yearly,
        "monthly": metrics.monthly_table(eq),
        "drawdowns": metrics.drawdown_table(eq, 5),
        "monte_carlo": metrics.monte_carlo(eq) if mc else {},
        "sensitivity": cost_sensitivity(res) if sensitivity else [],
        "rf": rf,
    }


# ---------------------------------------------------------------- console

def pct(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:,.{d}f}"


def console_summary(res: Result, a: dict) -> str:
    s, st, ts, ex = a["strategy"], a["stats"], a["trade_stats"], a["exposure"]
    L = []
    L.append("=" * 78)
    L.append(s.description or "(strategy)")
    L.append("-" * 78)
    L.append(s.summary())
    for n in s.notes:
        L.append(f"Note: {n}")
    L.append("-" * 78)
    L.append(f"Period            {st['start']} -> {st['end']}  ({st['years']:.1f} years)")
    L.append(f"Start / end       ${st['start_equity']:,.2f} -> ${st['end_equity']:,.2f}")
    L.append(f"Total return      {pct(st['total_return'])}      CAGR {pct(st['cagr'])}")
    L.append(f"Sharpe            {num(st['sharpe'])}   Sortino {num(st['sortino'])}   Calmar {num(st['calmar'])}   (rf {a['rf']:.2%})")
    L.append(f"Max drawdown      {pct(st['max_drawdown'])}  peak {st['max_dd_peak']}  trough {st['max_dd_trough']}  recovered {st['max_dd_recovery'] or 'not yet'}")
    L.append(f"Volatility        {pct(st['volatility'])}   Time in market {pct(ex['time_in_market'])}   Avg exposure {pct(ex['avg_exposure'])}")
    if ts.get("trades"):
        L.append(f"Trades            {ts['trades']}  ({num(ts['trades_per_year'], 1)}/yr)   Win rate {pct(ts['win_rate'], 1)}   Profit factor {num(ts['profit_factor'])}")
        L.append(f"Avg trade         {pct(ts['avg_return'])}   Avg win {pct(ts['avg_win'])}   Avg loss {pct(ts['avg_loss'])}   t-stat {num(ts['t_stat'])}")
    else:
        L.append("Trades            0 - the entry condition never triggered")
    if a["relative"]:
        r = a["relative"]
        L.append(f"vs SPY            beta {num(r['beta'])}   alpha {pct(r['alpha_annual'])}/yr   correlation {num(r['correlation'])}")
    L.append("-" * 78)
    L.append(f"{'Benchmark':24s} {'Final $':>14s} {'CAGR':>8s} {'Sharpe':>7s} {'MaxDD':>8s}")
    L.append(f"{'Strategy':24s} {st['end_equity']:>14,.0f} {pct(st['cagr']):>8s} {num(st['sharpe']):>7s} {pct(st['max_drawdown'], 1):>8s}")
    for n, b in a["bench_stats"].items():
        L.append(f"{n:24s} {b['end_equity']:>14,.0f} {pct(b['cagr']):>8s} {num(b['sharpe']):>7s} {pct(b['max_drawdown'], 1):>8s}  (from {b['start']})")
    L.append("-" * 78)
    L.append("Returns by year")
    y = a["yearly"]
    cols = [c for c in y.columns if c.endswith("buy & hold")]
    hdr = f"{'Year':6s} {'Strategy':>9s} {'MaxDD':>8s} {'Trades':>6s} " + " ".join(f"{c.replace(' buy & hold', ''):>8s}" for c in cols)
    L.append(hdr)
    for yr, row in y.iterrows():
        L.append(f"{yr:<6d} {pct(row['return'], 1):>9s} {pct(row['max_drawdown'], 1):>8s} {int(row['trades']):>6d} "
                 + " ".join(f"{pct(row[c], 1) if pd.notna(row[c]) else '':>8s}" for c in cols))
    if res.trades is not None and not res.trades.empty:
        L.append("-" * 78)
        n = len(res.trades)
        show = res.trades if n <= 20 else pd.concat([res.trades.head(10), res.trades.tail(10)])
        L.append(f"Trades ({n} total{', first and last 10 shown' if n > 20 else ''}; full list in trades.csv / report.html)")
        for i, t in show.iterrows():
            L.append(f"{i:>5d} {t['ticker']:6s} {t['entry_date']} -> {t['exit_date']} {pct(t['return']):>8s}  ${t['pnl']:>11,.2f}  {t['exit_reason']}")
    L.append("=" * 78)
    return "\n".join(L)


# ---------------------------------------------------------------- files

def _series_payload(eq: pd.Series) -> list:
    return [None if pd.isna(v) else round(float(v), 4) for v in eq.to_numpy()]


def write_outputs(res: Result, a: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = res.strategy
    tr = res.trades.copy() if res.trades is not None else pd.DataFrame()
    if not tr.empty:
        # quoted (split-adjusted, not dividend-adjusted) prices for readability
        def quote(row, which):
            df = res.prices[row["ticker"]]
            d = pd.Timestamp(row[f"{which}_date"])
            f = df["quote_close"].get(d, np.nan) / df["close"].get(d, np.nan)
            return row[f"{which}_price"] * f
        tr["entry_quote"] = tr.apply(lambda r: quote(r, "entry"), axis=1)
        tr["exit_quote"] = tr.apply(lambda r: quote(r, "exit"), axis=1)
    tr.to_csv(out_dir / "trades.csv", index_label="trade")

    eqdf = pd.DataFrame({"equity": res.equity, "drawdown": metrics.drawdown(res.equity),
                         "exposure": res.exposure, "positions": res.positions})
    for n, b in a["benchmarks"].items():
        eqdf[n] = b
    eqdf.to_csv(out_dir / "equity.csv", index_label="date")
    a["yearly"].to_csv(out_dir / "yearly.csv")
    a["monthly"].to_csv(out_dir / "monthly.csv")
    summary = {
        "description": s.description,
        "interpretation": s.summary(),
        "strategy": dataclasses.asdict(s),
        "stats": a["stats"], "trade_stats": a["trade_stats"], "exposure": a["exposure"],
        "relative_to_spy": a["relative"], "benchmarks": a["bench_stats"],
        "monte_carlo": a["monte_carlo"], "cost_sensitivity": a["sensitivity"],
        "drawdowns": a["drawdowns"].to_dict("records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2))
    (out_dir / "strategy.json").write_text(s.to_json())

    # ---- HTML
    idx = res.equity.index
    payload = {
        "title": s.description or s.summary().splitlines()[0],
        "interpretation": s.summary().splitlines(),
        "notes": s.notes,
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "series": [{"name": "Strategy", "values": _series_payload(res.equity)}]
                  + [{"name": n, "values": _series_payload(b.reindex(idx))} for n, b in a["benchmarks"].items()],
        "drawdown": _series_payload(metrics.drawdown(res.equity) * 100),
        "stats": a["stats"], "trade_stats": a["trade_stats"], "exposure": a["exposure"],
        "relative": a["relative"], "bench_stats": a["bench_stats"],
        "yearly": [{"year": int(y), **{k: row[k] for k in row.index}} for y, row in a["yearly"].iterrows()],
        "monthly": {"cols": list(a["monthly"].columns),
                    "rows": [{"year": int(y), "values": [None if pd.isna(v) else float(v) for v in row]} for y, row in a["monthly"].iterrows()]},
        "drawdowns": a["drawdowns"].to_dict("records"),
        "monte_carlo": a["monte_carlo"],
        "sensitivity": a["sensitivity"],
        "trades": tr.reset_index().rename(columns={"index": "trade"}).to_dict("records") if not tr.empty else [],
        "rf": a["rf"],
        "capital": s.capital,
        "universe": s.universe,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    blob = json.dumps(_clean(payload), separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("__TITLE__", html.escape(payload["title"][:80])).replace("__DATA__", blob)
    path = out_dir / "report.html"
    path.write_text(page)
    return path
