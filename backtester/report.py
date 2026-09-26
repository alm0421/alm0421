"""Analysis, console summary, file exports (CSV/JSON/Excel/PDF) and the self-contained HTML report.

A report shows one run or compares several runs (signal strategies and/or allocation portfolios)
side by side against benchmark buy-and-hold curves.
"""
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

from . import data, expr, metrics
from .engine import Result

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).with_name("report_template.html")
MAX_PRICE_POINTS = 60_000
INDICATOR_RE = re.compile(r"\b(sma|ema|wma|rma|bb_upper|bb_lower|keltner_upper|keltner_lower|donchian_upper|"
                          r"donchian_lower|supertrend|sar|monthly_sma|weekly_sma|vwap)\(([^()]*)\)")


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "backtest"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if o is None or o is pd.NaT:
        return None
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        if math.isnan(f):
            return None
        return f if math.isfinite(f) else ("inf" if f > 0 else "-inf")
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.strftime("%Y-%m-%d")
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return o


def _run_name(res: Result, i: int) -> str:
    s = res.strategy
    return s.name or (f"Strategy {chr(65 + i)}" if i < 26 else f"Strategy {i + 1}")


# ------------------------------------------------------------------ analysis

def cost_sensitivity(res: Result, levels=(0, 5, 10, 25)) -> list[dict]:
    from . import runner
    rows = []
    base = res.strategy.slippage_bps
    for bps in sorted(set(levels) | {base}):
        if bps == base:
            r = res
        else:
            s = dataclasses.replace(res.strategy, slippage_bps=float(bps), notes=list(res.strategy.notes))
            r = runner.run(s)
        fl = r.extras.get("flows")
        st = metrics.equity_stats(r.equity, flows=fl)
        rows.append({"slippage_bps": bps, "final_equity": st["end_equity"], "cagr": st["cagr"],
                     "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"], "current": bps == base})
    return rows


def benchmark_series(res: Result) -> dict[str, pd.Series]:
    """Benchmark buy-and-hold curves starting from the strategy's first day with the same capital."""
    s = res.strategy
    idx = res.equity.index
    cap = float(res.equity.iloc[0])
    names: list[str] = []
    primary = getattr(s, "benchmark", None) or "SPY"
    names.append(primary)
    uni = getattr(s, "universe", [])
    if res.kind == "signal" and len(uni) == 1 and uni[0] not in names and not uni[0].startswith("^"):
        names.append(uni[0])
    for t in ("SPY", "QQQ"):
        if t not in names:
            names.append(t)
    out = {}
    for t in names:
        b = metrics.buy_and_hold(t, idx, cap)
        if b is not None and len(b) > 30:
            out[f"{t} buy & hold"] = b
    return out


def _primary_bench(res: Result, benches: dict[str, pd.Series]) -> tuple[str, pd.Series | None]:
    primary = (getattr(res.strategy, "benchmark", None) or "SPY") + " buy & hold"
    return primary, benches.get(primary)


def price_payload(res: Result, budget: int = MAX_PRICE_POINTS) -> dict:
    """OHLC + indicator overlays for the traded tickers (most-traded first), within a size budget."""
    if res.trades is None or res.trades.empty or res.kind != "signal":
        return {}
    counts = res.trades["ticker"].value_counts()
    start = res.equity.index[1]
    end = res.equity.index[-1]
    rules = " ".join(r for r in (res.strategy.entry, getattr(res.strategy, "short_entry", None),
                                 res.strategy.exit_when) if r)
    calls = []
    for m in INDICATOR_RE.finditer(rules):
        args = m.group(2)
        if re.fullmatch(r"\s*(close\s*,\s*)?[\d.\s,]*", args):
            calls.append(m.group(0))
    calls = list(dict.fromkeys(calls))[:6]
    out, used = {}, 0
    for t in counts.index:
        df = res.prices.get(t)
        if df is None:
            continue
        seg = df[(df.index >= start) & (df.index <= end)]
        cost = len(seg) * (5 + len(calls))
        if used + cost > budget and out:
            break
        ns = expr.Namespace(df, ticker=t)
        overlays = {}
        for c in calls:
            try:
                v = expr.evaluate_value(c, ns).reindex(seg.index)
                overlays[c] = [None if not np.isfinite(x) else round(float(x), 4) for x in v]
            except Exception:  # noqa: BLE001 - overlays are best effort
                pass
        out[t] = {
            "dates": [d.strftime("%Y-%m-%d") for d in seg.index],
            "o": seg["open"].round(4).tolist(), "h": seg["high"].round(4).tolist(),
            "l": seg["low"].round(4).tolist(), "c": seg["close"].round(4).tolist(),
            "overlays": overlays,
        }
        used += cost
    return out


def holdings_payload(res: Result) -> dict:
    hw = res.holdings
    if hw is None or hw.empty:
        return {}
    w = hw.copy()
    if "cash" not in w:
        w["cash"] = (1 - w.sum(axis=1)).clip(lower=0)
    # keep the biggest holdings, fold the rest into "other"
    avg = w.drop(columns="cash").abs().mean().sort_values(ascending=False)
    keep = list(avg.index[:6])  # 6 hues + "other" + cash stay inside the validated palette
    other = [c for c in avg.index if c not in keep]
    if other:
        w["other"] = w[other].sum(axis=1)
        w = w.drop(columns=other)
    wk = w.resample("W-FRI").last().dropna(how="all")
    cols = keep + (["other"] if other else []) + ["cash"]
    last = hw.iloc[-1]
    current = [{"ticker": t, "weight": float(v)} for t, v in last.sort_values(ascending=False).items() if abs(v) > 1e-6]
    return {"dates": [d.strftime("%Y-%m-%d") for d in wk.index], "columns": cols,
            "values": {c: [round(float(x), 5) for x in wk[c].fillna(0)] for c in cols}, "current": current,
            "as_of": hw.index[-1].strftime("%Y-%m-%d")}


def analyze(res: Result, rf="tbill", sensitivity: bool = True, mc: bool = True, detail: bool = True) -> dict:
    s = res.strategy
    flows = res.extras.get("flows")
    has_flows = flows is not None and float(flows.abs().sum()) > 0
    nv = metrics.nav(res.equity, flows) if has_flows else res.equity
    stats = metrics.equity_stats(res.equity, rf, flows if has_flows else None)
    tstats = metrics.trade_stats(res.trades, stats["years"])
    expo = metrics.exposure_stats(res.exposure, res.positions, res.in_market)
    benches = benchmark_series(res)
    pname, pseries = _primary_bench(res, benches)
    rel = metrics.relative_stats(nv, pseries, rf)
    yearly = metrics.yearly_detail(nv, res.trades, res.exposure)
    yr_b = metrics.yearly_returns(benches)
    for n in yr_b:
        yearly[n] = yr_b[n].reindex(yearly.index)
    A = {
        "result": res, "strategy": s, "nav": nv, "flows": flows if has_flows else None,
        "stats": stats, "cash": metrics.cashflow_stats(res.equity, flows if has_flows else None),
        "trade_stats": tstats, "exposure": expo, "relative": rel, "primary_benchmark": pname,
        "benchmarks": benches, "yearly": yearly, "monthly": metrics.monthly_table(nv),
        "drawdowns": metrics.drawdown_table(nv, 5), "rf": rf, "interest": res.interest,
        "turnover": res.extras.get("turnover_annual"), "rebalances": res.extras.get("rebalances"),
    }
    if detail:
        A["monte_carlo"] = metrics.monte_carlo(res.equity, flows if has_flows else None) if mc else {}
        A["sensitivity"] = cost_sensitivity(res) if sensitivity else []
        A["rolling"] = metrics.rolling_series(nv, pseries, rf)
        A["rolling_summary"] = metrics.rolling_summary(nv)
        A["crises"] = metrics.crisis_table({"Strategy": nv, **benches})
        A["factors"] = metrics.factor_regression(nv, rf)
        corr_in = {"Strategy": nv, **{k.replace(" buy & hold", ""): v for k, v in benches.items()}}
        if res.holdings is not None and not res.holdings.empty and res.kind == "allocation":
            for t in [c for c in res.holdings.columns if c != "cash"][:6]:
                b = metrics.buy_and_hold(t, nv.index, 1.0)
                if b is not None and t not in corr_in:
                    corr_in[t] = b
        A["correlation"] = metrics.correlation_matrix(corr_in)
    return A


def common_window_stats(analyses: list[dict], rf="tbill") -> dict:
    """Stats for every run and benchmark over their common period (like-for-like comparison)."""
    series = {}
    for i, A in enumerate(analyses):
        series[_run_name(A["result"], i) if len(analyses) > 1 else "Strategy"] = A["nav"]
    for k, v in analyses[0]["benchmarks"].items():
        series[k] = v
    start = max(s.index[0] for s in series.values())
    end = min(s.index[-1] for s in series.values())
    out = {"start": start.date(), "end": end.date(), "columns": {}}
    for k, s in series.items():
        seg = s[(s.index >= start) & (s.index <= end)]
        if len(seg) < 30:
            continue
        st = metrics.equity_stats(seg / seg.iloc[0] * 10_000, rf)
        out["columns"][k] = st
    return out


# ------------------------------------------------------------------ console

def pct(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:,.{d}f}"


def console_summary(A: dict) -> str:
    s, st, ts, ex = A["strategy"], A["stats"], A["trade_stats"], A["exposure"]
    L = ["=" * 78, s.description or s.name or "(strategy)", "-" * 78, s.summary()]
    for n in s.notes:
        L.append(f"Note: {n}")
    L.append("-" * 78)
    L.append(f"Period            {st['start']} -> {st['end']}  ({st['years']:.1f} years)")
    if A.get("cash"):
        c = A["cash"]
        L.append(f"Money             start ${c['starting_balance']:,.0f} + contributions ${c['total_contributions']:,.0f} "
                 f"- withdrawals ${c['total_withdrawals']:,.0f} -> ${c['ending_balance']:,.0f}")
        L.append(f"                  money-weighted return {pct(c['money_weighted_return'])}/yr"
                 + ("   (money ran out)" if c["ran_out"] else ""))
    L.append(f"Start / end       ${st['start_equity']:,.2f} -> ${st['end_equity']:,.2f}")
    L.append(f"Total return      {pct(st['total_return'])}   CAGR {pct(st['cagr'])}   real (after inflation) {pct(st['real_cagr'])}")
    L.append(f"Sharpe            {num(st['sharpe'])}   Sortino {num(st['sortino'])}   Calmar {num(st['calmar'])}   "
             f"(excess over {'T-bills' if A['rf'] == 'tbill' else pct(float(A['rf'] or 0))})")
    L.append(f"Max drawdown      {pct(st['max_drawdown'])}  peak {st['max_dd_peak']}  trough {st['max_dd_trough']}  recovered {st['max_dd_recovery'] or 'not yet'}")
    L.append(f"Volatility        {pct(st['volatility'])}   Time in market {pct(ex['time_in_market'])}   Avg exposure {pct(ex['avg_exposure'])}")
    if ts.get("trades") and A["result"].kind == "allocation":
        L.append(f"Holding periods   {ts['trades']}   win rate {pct(ts['win_rate'], 1)}   (a ticker held from first purchase until sold out)")
    elif ts.get("trades"):
        L.append(f"Trades            {ts['trades']}  ({num(ts['trades_per_year'], 1)}/yr)   Win rate {pct(ts['win_rate'], 1)}   Profit factor {num(ts['profit_factor'])}")
        L.append(f"Avg trade         {pct(ts['avg_return'])}   Avg win {pct(ts['avg_win'])}   Avg loss {pct(ts['avg_loss'])}   t-stat {num(ts['t_stat'])}")
    else:
        L.append("Trades            0")
    if A.get("turnover") is not None:
        L.append(f"Turnover          {pct(A['turnover'] / 2, 0)} per year (one-sided)   Rebalances {A['rebalances']}")
    if A["relative"]:
        r = A["relative"]
        L.append(f"vs {A['primary_benchmark'].replace(' buy & hold', ''):14s} beta {num(r['beta'])}   alpha {pct(r['alpha_annual'])}/yr   "
                 f"correlation {num(r['correlation'])}   up/down capture {pct(r['up_capture'], 0)}/{pct(r['down_capture'], 0)}")
    L.append("-" * 78)
    L.append(f"{'':24s} {'Final $':>14s} {'CAGR':>8s} {'Sharpe':>7s} {'MaxDD':>8s}  (each from its first date)")
    L.append(f"{'Strategy':24s} {st['end_equity']:>14,.0f} {pct(st['cagr']):>8s} {num(st['sharpe']):>7s} {pct(st['max_drawdown'], 1):>8s}")
    for n, b in A["benchmarks"].items():
        bs = metrics.equity_stats(b, A["rf"])
        L.append(f"{n:24s} {bs['end_equity']:>14,.0f} {pct(bs['cagr']):>8s} {num(bs['sharpe']):>7s} {pct(bs['max_drawdown'], 1):>8s}  (from {bs['start']})")
    L.append("-" * 78)
    L.append("Returns by year")
    y = A["yearly"]
    cols = [c for c in y.columns if str(c).endswith("buy & hold")]
    L.append(f"{'Year':8s} {'Strategy':>9s} {'MaxDD':>8s} {'Trades':>6s} " + " ".join(f"{c.replace(' buy & hold', ''):>8s}" for c in cols))
    for yr, row in y.iterrows():
        lab = f"{yr}{'*' if row.get('partial') else ''}"
        L.append(f"{lab:8s} {pct(row['return'], 1):>9s} {pct(row['max_drawdown'], 1):>8s} {int(row['trades']):>6d} "
                 + " ".join(f"{pct(row[c], 1) if pd.notna(row[c]) else '':>8s}" for c in cols))
    if y["partial"].any():
        L.append("* partial year")
    tr = A["result"].trades
    if tr is not None and not tr.empty:
        L.append("-" * 78)
        n = len(tr)
        show = tr if n <= 20 else pd.concat([tr.head(10), tr.tail(10)])
        L.append(f"Trades ({n} total{', first and last 10 shown' if n > 20 else ''}; full list in trades.csv / report.html)")
        for i, t in show.iterrows():
            L.append(f"{i:>5d} {t['ticker']:6s} {t['side']:5s} {t['entry_date']} -> {t['exit_date']} {pct(t['return']):>8s}  ${t['pnl']:>11,.2f}  {t['exit_reason']}")
    L.append("=" * 78)
    return "\n".join(L)


# ------------------------------------------------------------------ payload

def _ser(s: pd.Series, idx: pd.DatetimeIndex | None = None, digits: int = 4) -> list:
    if idx is not None:
        s = s.reindex(idx)
    return [None if (v is None or not np.isfinite(v)) else round(float(v), digits) for v in s.to_numpy(dtype=float)]


def _trades_records(res: Result) -> list[dict]:
    tr = res.trades
    if tr is None or tr.empty:
        return []
    tr = tr.copy()
    return tr.reset_index().rename(columns={"index": "trade"}).to_dict("records")


def run_payload(A: dict, i: int, idx: pd.DatetimeIndex) -> dict:
    res = A["result"]
    s = A["strategy"]
    p = {
        "name": _run_name(res, i),
        "kind": res.kind,
        "title": s.description or s.summary().splitlines()[0],
        "interpretation": s.summary().splitlines(),
        "notes": list(s.notes),
        "spec": dataclasses.asdict(s),
        "equity": _ser(res.equity, idx, 2),
        "nav": _ser(A["nav"], idx, 2),
        "drawdown": _ser(metrics.drawdown(A["nav"]) * 100, idx, 3),
        "stats": A["stats"], "cash": A["cash"], "trade_stats": A["trade_stats"], "exposure": A["exposure"],
        "relative": A["relative"], "primary_benchmark": A["primary_benchmark"],
        "yearly": [{"year": int(y), **{k: row[k] for k in row.index}} for y, row in A["yearly"].iterrows()],
        "monthly": {"cols": list(A["monthly"].columns),
                    "rows": [{"year": int(y), "values": [None if pd.isna(v) else float(v) for v in row]} for y, row in A["monthly"].iterrows()]},
        "drawdowns": A["drawdowns"].to_dict("records"),
        "monte_carlo": A.get("monte_carlo", {}),
        "sensitivity": A.get("sensitivity", []),
        "rolling": {k: _ser(v, idx, 4) for k, v in A.get("rolling", {}).items()},
        "rolling_summary": A.get("rolling_summary", {}),
        "crises": A.get("crises", []),
        "factors": A.get("factors", {}),
        "correlation": A.get("correlation", {}),
        "interest": A["interest"], "turnover": A["turnover"], "rebalances": A["rebalances"],
        "trades": _trades_records(res),
        "orders": res.orders.to_dict("records") if res.orders is not None and not res.orders.empty and len(res.orders) <= 20000 else [],
        "holdings": holdings_payload(res),
        "prices": price_payload(res),
        "universe_size": len(getattr(s, "universe", []) or []),
    }
    return p


def build_payload(analyses: list[dict]) -> dict:
    idx = analyses[0]["result"].equity.index
    for A in analyses[1:]:
        idx = idx.union(A["result"].equity.index)
    benches = analyses[0]["benchmarks"]
    first = analyses[0]
    title = first["strategy"].description or first["strategy"].summary().splitlines()[0]
    if len(analyses) > 1:
        title = "Comparison: " + " vs ".join(_run_name(A["result"], i) for i, A in enumerate(analyses))
    return {
        "title": title,
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "runs": [run_payload(A, i, idx) for i, A in enumerate(analyses)],
        "benchmarks": [{"name": n, "values": _ser(b, idx, 2)} for n, b in benches.items()],
        "benchmark_yearly": {n: {int(y): float(v) for y, v in metrics.yearly_returns({n: b})[n].items()} for n, b in benches.items()},
        "common": common_window_stats(analyses, analyses[0]["rf"]),
        "rf": analyses[0]["rf"],
        "data": data.data_status(),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ------------------------------------------------------------------ files

def write_outputs(analyses: list[dict] | dict, out_dir: Path, excel: bool = True, pdf: bool = False) -> Path:
    if isinstance(analyses, dict):
        analyses = [analyses]
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, A in enumerate(analyses):
        res = A["result"]
        pre = "" if len(analyses) == 1 else f"{slug(_run_name(res, i))}_"
        tr = res.trades if res.trades is not None else pd.DataFrame()
        tr.to_csv(out_dir / f"{pre}trades.csv", index_label="trade")
        if res.orders is not None and not res.orders.empty:
            res.orders.to_csv(out_dir / f"{pre}orders.csv", index=False)
        eq = pd.DataFrame({"equity": res.equity, "twr_index": A["nav"], "drawdown": metrics.drawdown(A["nav"]),
                           "exposure": res.exposure, "positions": res.positions})
        if A["flows"] is not None:
            eq["cash_flow"] = A["flows"]
        for n, b in A["benchmarks"].items():
            eq[n] = b
        eq.to_csv(out_dir / f"{pre}equity.csv", index_label="date")
        if res.holdings is not None and not res.holdings.empty:
            res.holdings.to_csv(out_dir / f"{pre}holdings.csv", index_label="date")
        A["yearly"].to_csv(out_dir / f"{pre}yearly.csv")
        A["monthly"].to_csv(out_dir / f"{pre}monthly.csv")
        (out_dir / f"{pre}strategy.json").write_text(A["strategy"].to_json())
        summary = {k: A.get(k) for k in ("stats", "cash", "trade_stats", "exposure", "relative", "monte_carlo",
                                         "sensitivity", "rolling_summary", "crises", "factors")}
        summary.update({"description": A["strategy"].description, "interpretation": A["strategy"].summary(),
                        "notes": A["strategy"].notes, "kind": res.kind,
                        "drawdowns": A["drawdowns"].to_dict("records")})
        (out_dir / f"{pre}summary.json").write_text(json.dumps(_clean(summary), indent=2))
        if excel:
            _excel(A, out_dir / f"{pre}report.xlsx")
    payload = build_payload(analyses)
    blob = json.dumps(_clean(payload), separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("__TITLE__", html.escape(payload["title"][:80])).replace("__DATA__", blob)
    path = out_dir / "report.html"
    path.write_text(page)
    if pdf:
        to_pdf(path, out_dir / "report.pdf")
    return path


def _excel(A: dict, path: Path) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return
    res = A["result"]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        rows = [("Strategy", A["strategy"].description or ""), ("Interpretation", A["strategy"].summary())]
        rows += [(k, v) for k, v in _clean(A["stats"]).items()]
        rows += [(f"cash: {k}", v) for k, v in _clean(A["cash"]).items()]
        rows += [(f"trades: {k}", v) for k, v in _clean(A["trade_stats"]).items()]
        rows += [(f"vs benchmark: {k}", v) for k, v in _clean(A["relative"]).items()]
        pd.DataFrame(rows, columns=["metric", "value"]).to_excel(xw, sheet_name="Summary", index=False)
        A["yearly"].to_excel(xw, sheet_name="Yearly")
        A["monthly"].to_excel(xw, sheet_name="Monthly")
        A["drawdowns"].to_excel(xw, sheet_name="Drawdowns", index=False)
        if res.trades is not None and not res.trades.empty:
            res.trades.to_excel(xw, sheet_name="Trades", index_label="trade")
        if res.orders is not None and not res.orders.empty:
            res.orders.to_excel(xw, sheet_name="Orders", index=False)
        eq = pd.DataFrame({"equity": res.equity, "twr_index": A["nav"], "drawdown": metrics.drawdown(A["nav"])})
        for n, b in A["benchmarks"].items():
            eq[n] = b
        eq.index = eq.index.tz_localize(None)
        eq.to_excel(xw, sheet_name="Equity", index_label="date")
        if res.holdings is not None and not res.holdings.empty:
            res.holdings.resample("ME").last().to_excel(xw, sheet_name="Holdings (month-end)", index_label="date")


def to_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Print the HTML report to PDF with headless Chromium (Playwright), if available."""
    try:
        import glob

        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    exe = (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        pg = b.new_page(viewport={"width": 1200, "height": 900})
        pg.goto(html_path.resolve().as_uri())
        pg.wait_for_timeout(800)
        pg.emulate_media(media="print")
        pg.pdf(path=str(pdf_path), format="A4", print_background=True, margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"})
        b.close()
    return True
