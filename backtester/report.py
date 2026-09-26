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
MAX_PRICE_POINTS = 200_000  # ~1.5 MB of JSON at most
MAX_PRICE_TICKERS = 40
# Price-scale indicators, drawn over the candles. The lookbehind skips sym("SPY").sma(200)-style calls on
# another ticker, which would otherwise be evaluated on the charted one.
INDICATOR_RE = re.compile(r"(?<![\w.])(sma|ema|wma|rma|bb_upper|bb_lower|keltner_upper|keltner_lower|donchian_upper|"
                          r"donchian_lower|supertrend|sar|monthly_sma|weekly_sma|vwap)\(([^()]*)\)")
# Oscillators, drawn in sub-panes below the price: function -> (pane, fixed y-range or None).
OSCILLATORS = {
    "rsi": ("RSI", (0, 100)), "stoch_k": ("Stochastic", (0, 100)), "stoch_d": ("Stochastic", (0, 100)),
    "macd": ("MACD", None), "macd_signal": ("MACD", None), "macd_hist": ("MACD", None),
    "adx": ("ADX / DI", None), "plus_di": ("ADX / DI", None), "minus_di": ("ADX / DI", None),
    "cci": ("CCI", None), "willr": ("Williams %R", (-100, 0)), "mfi": ("MFI", (0, 100)),
    "zscore": ("Z-score", None), "pct_rank": ("Percent rank", None), "atr": ("ATR", None), "natr": ("NATR", None),
}
OSC_RE = re.compile(r"(?<![\w.])(" + "|".join(sorted(OSCILLATORS, key=len, reverse=True)) + r")\(([^()]*)\)")
SIMPLE_ARGS = re.compile(r"\s*(close\s*,\s*)?[\d.\s,]*")
MAX_PANES = 3


def _nums(args: str) -> list[str]:
    return [a.strip() for a in args.split(",") if a.strip() and a.strip() != "close"]


def _companions(fn: str, args: str) -> list[str]:
    """The calls that belong on the same pane: MACD line/signal/histogram, stochastic %K/%D."""
    n = _nums(args)
    if fn.startswith("macd"):
        if not n:
            return ["macd()", "macd_signal()", "macd_hist()"]
        f, s = n[0], (n[1] if len(n) > 1 else "26")
        g = n[2] if fn != "macd" and len(n) >= 3 else "9"
        return [f"macd({f}, {s})", f"macd_signal({f}, {s}, {g})", f"macd_hist({f}, {s}, {g})"]
    if fn in ("stoch_k", "stoch_d"):
        if not n:
            return ["stoch_k()", "stoch_d()"]
        k = n[:2]
        d = n[2:3] if fn == "stoch_d" else []
        return [f"stoch_k({', '.join(k)})", f"stoch_d({', '.join(k + d)})"]
    return [f"{fn}({args})"]


def _levels(call: str, rules: str) -> list[float]:
    """Numeric thresholds the rule compares this call against, e.g. rsi(close, 2) < 10 -> [10]."""
    c = re.escape(call)
    num_ = r"(-?\d+(?:\.\d+)?)"
    out = [float(m.group(1)) for m in re.finditer(c + r"\s*(?:<=|>=|<|>|==)\s*" + num_ + r"(?![\w.(])", rules)]
    out += [float(m.group(1)) for m in re.finditer(r"(?<![\w.)])" + num_ + r"\s*(?:<=|>=|<|>|==)\s*" + c, rules)]
    return out


def oscillator_panes(rules: str) -> list[dict]:
    """Group the oscillators a rule uses into sub-panes: [{name, calls, levels, range}]."""
    panes: dict[str, dict] = {}
    for m in OSC_RE.finditer(rules):
        fn, args = m.group(1), m.group(2)
        if not SIMPLE_ARGS.fullmatch(args):
            continue
        name, rng = OSCILLATORS[fn]
        p = panes.setdefault(name, {"name": name, "calls": [], "levels": [], "range": list(rng) if rng else None})
        for c in _companions(fn, args):
            if c not in p["calls"] and len(p["calls"]) < 4:
                p["calls"].append(c)
        for lv in _levels(m.group(0), rules):
            if lv not in p["levels"]:
                p["levels"].append(lv)
    return list(panes.values())[:MAX_PANES]


def _values(call: str, ns, index) -> list | None:
    try:
        v = expr.evaluate_value(call, ns).reindex(index)
    except Exception:  # noqa: BLE001 - indicator plots are best effort
        return None
    return [None if not np.isfinite(x) else round(float(x), 4) for x in v.to_numpy(dtype=float)]


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


def _aligned_buy_and_hold(t: str, idx: pd.DatetimeIndex, cap: float) -> pd.Series | None:
    """Buy and hold bought at the close of the strategy's first real bar (idx[1]); the synthetic
    day-before point idx[0] carries the starting capital, as it does for the strategy."""
    if len(idx) < 2:
        return metrics.buy_and_hold(t, idx, cap)
    b = metrics.buy_and_hold(t, idx[1:], cap)
    if b is not None and len(b) and b.index[0] == idx[1]:
        b = pd.concat([pd.Series([cap], index=idx[:1]), b]).rename(t)
    return b


def benchmark_series(res: Result) -> dict[str, pd.Series]:
    """Benchmark buy-and-hold curves (growth of the starting capital, no cash flows), bought at the
    close of the strategy's first bar with the same capital."""
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
        b = _aligned_buy_and_hold(t, idx, cap)
        if b is not None and len(b) > 30:
            out[f"{t} buy & hold"] = b
    return out


def benchmarks_with_flows(benches: dict[str, pd.Series], flows: pd.Series | None) -> dict[str, pd.Series]:
    """The benchmarks' dollar values when they receive the portfolio's own contributions and
    withdrawals (a like-for-like 'account value' comparison)."""
    if flows is None or float(flows.abs().sum()) == 0:
        return {}
    return {k: metrics.with_flows(b, flows) for k, b in benches.items()}


def real_equity(equity: pd.Series) -> pd.Series | None:
    """Equity in dollars of the first day (deflated by CPI)."""
    c = data.cpi()
    if c.empty:
        return None
    ci = c.reindex(equity.index.union(c.index)).ffill().reindex(equity.index)
    if ci.isna().all():
        return None
    base = ci.dropna().iloc[0]
    return equity / (ci / base)


def _primary_bench(res: Result, benches: dict[str, pd.Series]) -> tuple[str, pd.Series | None]:
    primary = (getattr(res.strategy, "benchmark", None) or "SPY") + " buy & hold"
    return primary, benches.get(primary)


def price_payload(res: Result, budget: int = MAX_PRICE_POINTS) -> dict:
    """OHLC + indicators for the traded tickers (most-traded first), within a size budget.

    Price-scale indicators (moving averages, bands, stops) go in "overlays"; oscillators used by the rules
    (RSI, MACD, stochastic, ADX, CCI, Williams %R, MFI...) go in "panes", one sub-pane per indicator family,
    with the rule's numeric thresholds as "levels".
    """
    if res.trades is None or res.trades.empty or res.kind != "signal":
        return {}
    counts = res.trades["ticker"].value_counts()
    start = res.equity.index[1]
    end = res.equity.index[-1]
    rules = " ".join(r for r in (res.strategy.entry, getattr(res.strategy, "short_entry", None),
                                 res.strategy.exit_when) if isinstance(r, str) and r)
    calls = [m.group(0) for m in INDICATOR_RE.finditer(rules) if SIMPLE_ARGS.fullmatch(m.group(2))]
    calls = list(dict.fromkeys(calls))[:6]
    panes = oscillator_panes(rules)
    n_series = 5 + len(calls) + sum(len(p["calls"]) for p in panes)
    out, used = {}, 0
    for t in counts.index:
        df = res.prices.get(t)
        if df is None:
            continue
        seg = df[(df.index >= start) & (df.index <= end)]
        if len(counts) > 1:
            # many tickers: ship only the stretch around this ticker's trades so more tickers fit the budget
            tt = res.trades[res.trades["ticker"] == t]
            lo = seg.index.searchsorted(pd.Timestamp(min(tt["entry_date"])))
            hi = seg.index.searchsorted(pd.Timestamp(max(tt["exit_date"])), side="right")
            seg = seg.iloc[max(0, lo - 300): hi + 60]
        cost = len(seg) * n_series
        if len(out) >= MAX_PRICE_TICKERS:
            break
        if used + cost > budget and out:
            continue  # a less-traded ticker with a shorter stretch may still fit
        ns = expr.Namespace(df, ticker=t)
        overlays = {}
        for c in calls:
            v = _values(c, ns, seg.index)
            if v is not None:
                overlays[c] = v
        tpanes = []
        for p in panes:
            series = {}
            for c in p["calls"]:
                v = _values(c, ns, seg.index)
                if v is not None:
                    series[c] = v
            if series:
                tpanes.append({"name": p["name"], "series": series, "levels": p["levels"], "range": p["range"]})
        out[t] = {
            "dates": [d.strftime("%Y-%m-%d") for d in seg.index],
            "o": seg["open"].round(4).tolist(), "h": seg["high"].round(4).tolist(),
            "l": seg["low"].round(4).tolist(), "c": seg["close"].round(4).tolist(),
            "overlays": overlays, "panes": tpanes,
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
    first_bar = res.equity.index[1] if len(res.equity) > 1 else None
    stats = metrics.equity_stats(res.equity, rf, flows if has_flows else None, first_bar=first_bar)
    tstats = metrics.trade_stats(res.trades, stats["years"])
    no_trades = res.kind == "signal" and not tstats.get("trades")
    warnings = metrics.result_warnings(res.kind, stats, tstats, res.interest, has_flows)
    if no_trades:
        stats = metrics.suppress_degenerate(stats)
    expo = metrics.exposure_stats(res.exposure, res.positions, res.in_market)
    benches = benchmark_series(res)
    pname, pseries = _primary_bench(res, benches)
    rel = {} if no_trades else metrics.relative_stats(nv, pseries, rf)  # beta/alpha of idle cash mean nothing
    yearly = metrics.yearly_detail(nv, res.trades, res.exposure, first_bar=first_bar)
    yr_b = metrics.yearly_returns(benches)
    for n in yr_b:
        yearly[n] = yr_b[n].reindex(yearly.index)
    A = {
        "result": res, "strategy": s, "nav": nv, "flows": flows if has_flows else None,
        "stats": stats, "cash": metrics.cashflow_stats(res.equity, flows if has_flows else None),
        "trade_stats": tstats, "exposure": expo, "relative": rel, "primary_benchmark": pname,
        "benchmarks": benches, "yearly": yearly, "monthly": metrics.monthly_table(nv),
        "drawdowns": metrics.drawdown_table(nv, 5, first_bar=first_bar), "rf": rf, "interest": res.interest,
        "turnover": res.extras.get("turnover_annual"), "rebalances": res.extras.get("rebalances"),
        "warnings": warnings, "no_trades": no_trades,
        "first_bar": first_bar, "fees": res.extras.get("fees", 0.0),
        "equity_real": real_equity(res.equity),
        "attribution": res.extras.get("attribution"),
    }
    bwf = benchmarks_with_flows(benches, flows if has_flows else None)
    A["benchmarks_with_flows"] = bwf
    A["benchmark_cash"] = {k: metrics.cashflow_stats(v, flows) for k, v in bwf.items()}
    A["withdrawal_rates"] = {}
    if res.kind == "allocation" and (getattr(s, "withdrawal", 0) or getattr(s, "withdrawal_pct", 0)):
        from . import montecarlo
        A["withdrawal_rates"] = montecarlo.historical_withdrawal_rates(metrics.monthly_returns(nv))
        if A["withdrawal_rates"]:
            A["withdrawal_rates"].update({"from": stats["start"], "to": stats["end"]})
    if detail:
        A["monte_carlo"] = metrics.monte_carlo(res.equity, flows if has_flows else None) if mc and not no_trades else {}
        A["sensitivity"] = cost_sensitivity(res) if sensitivity and not no_trades else []
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
    series, blank = {}, set()
    for i, A in enumerate(analyses):
        name = _run_name(A["result"], i) if len(analyses) > 1 else "Strategy"
        series[name] = A["nav"]
        if A.get("no_trades"):
            blank.add(name)
    for k, v in analyses[0]["benchmarks"].items():
        series[k] = v
    start = max(s.index[0] for s in series.values())
    end = min(s.index[-1] for s in series.values())
    fb = next((A.get("first_bar") for A in analyses if A.get("first_bar") is not None and A["result"].equity.index[0] == start), None)
    out = {"start": metrics.display_date(start, fb).date(), "end": end.date(), "columns": {}}
    for k, s in series.items():
        seg = s[(s.index >= start) & (s.index <= end)]
        if len(seg) < 30:
            continue
        st = metrics.equity_stats(seg / seg.iloc[0] * 10_000, rf, first_bar=fb)
        out["columns"][k] = metrics.suppress_degenerate(st) if k in blank else st
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
    for w in A.get("warnings") or []:
        tag = {"error": "WARNING", "warn": "Warning", "info": "Note"}.get(w["level"], "Warning")
        L.append(f"{tag}: {w['message']}. {w['detail']}")
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
    bwf = A.get("benchmarks_with_flows") or {}
    for n, b in A["benchmarks"].items():
        bs = metrics.equity_stats(b, A["rf"], first_bar=A.get("first_bar"))
        final = float(bwf[n].iloc[-1]) if n in bwf else bs["end_equity"]
        L.append(f"{n:24s} {final:>14,.0f} {pct(bs['cagr']):>8s} {num(bs['sharpe']):>7s} {pct(bs['max_drawdown'], 1):>8s}  (from {bs['start']})")
    if bwf:
        L.append("(benchmark final values include the same contributions and withdrawals as the strategy)")
    wr = A.get("withdrawal_rates") or {}
    if wr:
        L.append(f"Withdrawal rates  safe {pct(wr.get('swr'), 2)}   perpetual {pct(wr.get('pwr'), 2)}   "
                 f"(inflation-adjusted, over this history {wr['from']} -> {wr['to']}); 95% bootstrap safe rate {pct(wr.get('swr_mc95'), 2)}")
    at = A.get("attribution")
    if at is not None and len(at):
        L.append("P&L by holding  " + "   ".join(f"{r.ticker} ${r.pnl:,.0f}" for r in at.head(8).itertuples())
                 + f"   interest ${A['interest']:,.0f}" + (f"   fees -${A['fees']:,.0f}" if A.get("fees") else ""))
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
        "warnings": A.get("warnings", []),
        "trades": _trades_records(res),
        "orders": res.orders.to_dict("records") if res.orders is not None and not res.orders.empty and len(res.orders) <= 20000 else [],
        "holdings": holdings_payload(res),
        "prices": price_payload(res),
        "universe_size": len(getattr(s, "universe", []) or []),
        # account value in dollars of the first day (CPI-deflated), for a real/nominal toggle
        "equity_real": _ser(A["equity_real"], idx, 2) if A.get("equity_real") is not None else None,
        # per-ticker P&L (allocation runs): sum(pnl) + interest - fees == end - start - net flows
        "attribution": _attribution_payload(A),
        "withdrawal_rates": A.get("withdrawal_rates") or {},
        "benchmark_cash": A.get("benchmark_cash") or {},
        "first_bar": A.get("first_bar"),
    }
    return p


def _attribution_payload(A: dict) -> dict:
    at = A.get("attribution")
    if at is None or not len(at):
        return {}
    res = A["result"]
    fl = A.get("flows")
    net_flows = float(fl.sum()) if fl is not None else 0.0
    return {"rows": at.to_dict("records"), "interest": A["interest"], "fees": A.get("fees") or 0.0,
            "total_pnl": float(at["pnl"].sum()), "net_flows": net_flows,
            "start_equity": float(res.equity.iloc[0]), "end_equity": float(res.equity.iloc[-1]),
            "check": float(res.equity.iloc[-1] - res.equity.iloc[0] - net_flows
                           - (at["pnl"].sum() + A["interest"] - (A.get("fees") or 0.0)))}


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
        # values: growth of the starting capital; with_flows: the same benchmark receiving the first
        # run's contributions/withdrawals (compare with the runs' "equity" account values)
        "benchmarks": [{"name": n, "values": _ser(b, idx, 2),
                        **({"with_flows": _ser(first["benchmarks_with_flows"][n], idx, 2)}
                           if n in (first.get("benchmarks_with_flows") or {}) else {})} for n, b in benches.items()],
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
        if A.get("equity_real") is not None:
            eq["equity_real"] = A["equity_real"]
        for n, b in A["benchmarks"].items():
            eq[n] = b
        for n, b in (A.get("benchmarks_with_flows") or {}).items():
            eq[n + " (with cash flows)"] = b
        eq.to_csv(out_dir / f"{pre}equity.csv", index_label="date")
        if A.get("attribution") is not None and len(A["attribution"]):
            A["attribution"].to_csv(out_dir / f"{pre}attribution.csv", index=False)
        if res.holdings is not None and not res.holdings.empty:
            res.holdings.to_csv(out_dir / f"{pre}holdings.csv", index_label="date")
        A["yearly"].to_csv(out_dir / f"{pre}yearly.csv")
        A["monthly"].to_csv(out_dir / f"{pre}monthly.csv")
        (out_dir / f"{pre}strategy.json").write_text(A["strategy"].to_json())
        summary = {k: A.get(k) for k in ("stats", "cash", "trade_stats", "exposure", "relative", "monte_carlo",
                                         "sensitivity", "rolling_summary", "crises", "factors")}
        summary.update({"description": A["strategy"].description, "interpretation": A["strategy"].summary(),
                        "notes": A["strategy"].notes, "kind": res.kind, "warnings": A.get("warnings", []),
                        "drawdowns": A["drawdowns"].to_dict("records"),
                        "withdrawal_rates": A.get("withdrawal_rates") or {},
                        "benchmark_cash": A.get("benchmark_cash") or {},
                        "attribution": _attribution_payload(A)})
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
        for n, b in (A.get("benchmarks_with_flows") or {}).items():
            eq[n + " (with cash flows)"] = b
        eq.index = eq.index.tz_localize(None)
        eq.to_excel(xw, sheet_name="Equity", index_label="date")
        if A.get("attribution") is not None and len(A["attribution"]):
            A["attribution"].to_excel(xw, sheet_name="Attribution", index=False)
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
