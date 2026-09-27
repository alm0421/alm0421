"""Analysis, console summary, file exports (CSV/JSON/Excel/PDF) and the self-contained HTML report.

A report shows one run or compares several runs (signal strategies and/or allocation portfolios)
side by side against benchmark buy-and-hold curves.
"""
from __future__ import annotations

import ast
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
MAX_EMBED_TICKERS = 12      # tickers embedded in report.html; the others load from charts/<TICKER>.js on demand
# Price-scale indicators, drawn over the candles. The lookbehind skips sym("SPY").sma(200)-style calls on
# another ticker, which would otherwise be evaluated on the charted one. The chart payload itself is laid out
# from the rule's syntax tree (chart_layout); the regexes serve oscillator_panes / other_ticker_panes.
PRICE_FNS = ("sma", "ma", "ema", "wma", "rma", "highest", "lowest", "bb_upper", "bb_lower", "keltner_upper",
             "keltner_lower", "donchian_upper", "donchian_lower", "supertrend", "sar", "vwap", "weekly_sma",
             "monthly_sma", "weekly_ema", "monthly_ema", "weekly_close", "monthly_close", "cummax", "cummin",
             "hma", "vwma", "linreg", "alma", "kama", "tenkan", "kijun", "senkou_a", "senkou_b", "avwap",
             "pivothigh", "pivotlow")
INDICATOR_RE = re.compile(r"(?<![\w.])(" + "|".join(sorted(PRICE_FNS, key=len, reverse=True)) + r")\(([^()]*)\)")
# Oscillators and other own-scale series, drawn in sub-panes below the price: function -> (pane, fixed y-range).
OSCILLATORS = {
    "rsi": ("RSI", (0, 100)), "weekly_rsi": ("RSI (weekly)", (0, 100)), "monthly_rsi": ("RSI (monthly)", (0, 100)),
    "stoch_k": ("Stochastic", (0, 100)), "stoch_d": ("Stochastic", (0, 100)),
    "stoch_rsi_k": ("Stochastic RSI", (0, 100)), "stoch_rsi_d": ("Stochastic RSI", (0, 100)),
    "stoch": ("Stochastic", (0, 100)),
    "macd": ("MACD", None), "macd_signal": ("MACD", None), "macd_hist": ("MACD", None),
    "adx": ("ADX / DMI", None), "plus_di": ("ADX / DMI", None), "minus_di": ("ADX / DMI", None),
    "cci": ("CCI", None), "willr": ("Williams %R", (-100, 0)), "mfi": ("MFI", (0, 100)),
    "zscore": ("Z-score", None), "pct_rank": ("Percent rank", (0, 1)), "atr": ("ATR", None), "natr": ("NATR", None),
    "ret": ("Return", None), "roc": ("Return", None), "tret": ("Total return", None),
    "weekly_ret": ("Return (weekly)", None), "monthly_ret": ("Return (monthly)", None),
    "volatility": ("Volatility", None), "stdev": ("Std dev", None), "obv": ("OBV", None),
    "drawdown": ("Drawdown", None), "max_drawdown": ("Drawdown", None),
    "ma_return": ("Return stats", None), "stdev_return": ("Return stats", None),
    "down_streak": ("Streak", None), "up_streak": ("Streak", None),
    "count": ("Count", None), "bars_since": ("Bars since", None),
    "aroon_up": ("Aroon", (0, 100)), "aroon_down": ("Aroon", (0, 100)), "aroon_osc": ("Aroon oscillator", (-100, 100)),
    "cmf": ("CMF", None), "supertrend_dir": ("Supertrend direction", (-1, 1)),
}
# Rule variables with their own scale (a bare name in the rule): name -> (pane, fixed y-range).
VARIABLE_PANES = {
    "down_days": ("Streak", None), "up_days": ("Streak", None), "ibs": ("IBS", (0, 1)), "gap": ("Gap", None),
    "change": ("Change", None), "range": ("Range", None), "volume": ("Volume", None),
    "dollar_volume": ("Dollar volume", None), "market_cap": ("Market cap", None),
    "true_range": ("True range", None),
}
# price-scale rule variables (Heikin Ashi bars, price averages): drawn over the price
OVERLAY_NAMES = {"ha_open", "ha_high", "ha_low", "ha_close", "hl2", "hlc3", "ohlc4", "hlcc4"}
OSC_RE = re.compile(r"(?<![\w.])(" + "|".join(sorted(OSCILLATORS, key=len, reverse=True)) + r")\(([^()]*)\)")
SIMPLE_ARGS = re.compile(r"\s*(close\s*,\s*)?[\d.\s,]*")
MAX_SERIES = 48   # a safety cap on the series one chart carries; the number of panes is not limited


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
    if fn in ("stoch_rsi_k", "stoch_rsi_d"):
        if fn == "stoch_rsi_d":
            full = (n + ["3", "3", "14", "14"][len(n):])[:4]
            k, d = [full[0], full[2], full[3]], full
        else:
            full = (n + ["3", "14", "14"][len(n):])[:3]
            k, d = full, [full[0], "3", full[1], full[2]]
        return [f"stoch_rsi_k({', '.join(k)})", f"stoch_rsi_d({', '.join(d)})"]
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
    return list(panes.values())


def _series_expr(call: str) -> str:
    """The rule expression of a charted series: "highest(high, 20)[1]" (the series as the rule reads it, one bar
    back) is ref(highest(high, 20), 1)."""
    m = re.fullmatch(r"(?s)(.+)\[(\d+)\]", call.strip())
    return f"ref({m.group(1)}, {m.group(2)})" if m else call


def _values(call: str, ns, index) -> list | None:
    try:
        v = expr.evaluate_value(_series_expr(call), ns).reindex(index)
    except Exception:  # noqa: BLE001 - indicator plots are best effort
        return None
    return [None if not np.isfinite(x) else round(float(x), 4) for x in v.to_numpy(dtype=float)]


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "backtest"


def run_slug(text: str, key) -> str:
    """A report folder name: the readable start of `text` plus a short hash of `key` (the full spec, or whatever
    else defines the run), so two runs whose descriptions start alike get folders of their own, and re-running the
    same spec reuses its folder: 'if-spy-is-above-its-200-day-moving-average-then-q-3f9a1c2e'."""
    import hashlib
    if not isinstance(key, str):
        key = json.dumps(key, sort_keys=True, default=str)
    base = slug(text)[:50].rstrip("-") or "backtest"
    return f"{base}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def _clean(o):
    t = type(o)       # fast paths for the exact built-in types that make up most of a report payload
    if t is float:
        return None if o != o else (o if math.isfinite(o) else ("inf" if o > 0 else "-inf"))
    if t is str or t is int:
        return o
    if t is list:
        return [_clean(v) for v in o]
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

def cost_sensitivity(res: Result, levels=(0, 5, 10, 25), warm=None) -> list[dict]:
    """The run repeated at several slippage levels (stats from `warm`, the end of the indicator warm-up)."""
    from . import portfolio, runner
    rows = []
    base = res.strategy.slippage_bps
    for bps in sorted(set(levels) | {base}):
        if bps == base:
            r = res
        else:
            s = dataclasses.replace(res.strategy, slippage_bps=float(bps), notes=list(res.strategy.notes))
            with portfolio.reuse_evaluations(res, light=True):   # the targets do not depend on costs: evaluated once
                r = runner.run(s)
        r = trim_result(r, warm)
        fl = r.extras.get("flows")
        st = metrics.equity_stats(r.equity, flows=fl)
        rows.append({"slippage_bps": bps, "final_equity": st["end_equity"], "cagr": st["cagr"],
                     "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"], "current": bps == base})
    return rows


def _aligned_buy_and_hold(t: str, idx: pd.DatetimeIndex, cap: float, day0: bool = False) -> pd.Series | None:
    """Buy and hold bought at the close of the strategy's first real bar (idx[1]); the synthetic
    day-before point idx[0] carries the starting capital, as it does for the strategy. day0: the strategy was
    bought at idx[0]'s close (a portfolio starting from the prior session, as Portfolio Visualizer does), so the
    benchmark is too, when it has a price that day."""
    if len(idx) < 2 or day0:
        return metrics.buy_and_hold(t, idx, cap)
    b = metrics.buy_and_hold(t, idx[1:], cap)
    if b is not None and len(b) and b.index[0] == idx[1]:
        b = pd.concat([pd.Series([cap], index=idx[:1]), b]).rename(t)
    return b


LATE_DAYS = 7  # a benchmark whose data starts this many days after the strategy's first bar covers only part


def _first_date(t: str):
    try:
        return data.load(t).index[0]
    except (FileNotFoundError, data.DataError, KeyError):
        return None


def default_benchmark(res: Result) -> str:
    """The comparison ticker: the user's choice, else SPY; when the run starts before SPY existed (1993),
    SPYSIM (the US market spliced into SPY) so alpha, beta and the head-to-head cover the whole period."""
    b = getattr(res.strategy, "benchmark", None)
    if b:
        return metrics.benchmark_label(b)
    if len(res.equity.index) > 1:
        spy = _first_date("SPY")
        first = res.equity.index[1]
        if spy is not None and spy > first + pd.Timedelta(days=LATE_DAYS):
            sim = _first_date("SPYSIM")
            if sim is not None and sim <= first + pd.Timedelta(days=LATE_DAYS):
                return "SPYSIM"
    return "SPY"


def benchmark_series(res: Result, nav_: pd.Series | None = None, skipped: dict | None = None) -> dict[str, pd.Series]:
    """Benchmark buy-and-hold curves (growth of the starting capital, no cash flows), bought at the
    close of the strategy's first bar with the same capital. A benchmark that starts later (its data
    begins after the strategy's first bar) is bought on its first day at the strategy's growth index
    (`nav_`) of that day, so the curves are comparable from then on; its stats cover only its own period."""
    s = res.strategy
    idx = res.equity.index
    cap = float(res.equity.iloc[0])
    if cap <= 0:
        # an account that starts at $0 (funded by contributions): growth curves start at its first funded balance
        cap = float(nav_.iloc[0]) if nav_ is not None and len(nav_) and nav_.iloc[0] > 0 else 10_000.0
    names: list[str] = []
    primary = default_benchmark(res)
    names.append(primary)
    uni = getattr(s, "universe", [])
    if res.kind == "signal" and len(uni) == 1 and uni[0] not in names and not uni[0].startswith("^"):
        names.append(uni[0])
    for t in ("SPY", "QQQ"):
        if t not in names:
            names.append(t)
    out = {}
    for t in names:
        b = _aligned_buy_and_hold(t, idx, cap, bool(res.extras.get("day0")))
        if b is not None and len(b) > 30:
            if nav_ is not None and len(idx) > 1 and b.index[0] > idx[1]:
                base = nav_.reindex(nav_.index.union(b.index[:1])).ffill().get(b.index[0])
                if base is not None and np.isfinite(base) and base > 0:
                    b = b / float(b.iloc[0]) * float(base)
            out[bench_name(t)] = b
        elif skipped is not None:
            skipped[bench_name(t)] = _first_date(t)
    return out


def bench_name(t: str) -> str:
    """The column name of a benchmark: "SPY buy & hold", or "60% SPY / 40% AGG blend" (rebalanced monthly)."""
    try:
        blend = metrics.parse_blend(t) is not None
    except ValueError:
        blend = False
    return f"{metrics.benchmark_label(t)} blend" if blend else f"{t} buy & hold"


def is_bench_col(c) -> bool:
    return str(c).endswith(" buy & hold") or str(c).endswith(" blend")


def benchmark_coverage(benches: dict[str, pd.Series], first_bar) -> dict[str, str]:
    """{benchmark: first date} for benchmarks that start after the strategy's first bar."""
    if first_bar is None:
        return {}
    lim = pd.Timestamp(first_bar) + pd.Timedelta(days=LATE_DAYS)
    return {k: str(b.index[0].date()) for k, b in benches.items() if len(b) and b.index[0] > lim}


def benchmark_partial_years(benches: dict[str, pd.Series], first_bar) -> dict[str, dict]:
    """{benchmark: {"year", "from"}} for a benchmark that starts after the strategy's first bar partway through a
    year (SPY from 1993-01-29, QQQ from 1999-03-10): its first yearly return is for part of that year only."""
    out = {}
    for k, d in benchmark_coverage(benches, first_bar).items():
        f = pd.Timestamp(d)
        if f.month > 1 or f.day > 7:
            out[k] = {"year": int(f.year), "from": str(f.date())}
    return out


def benchmarks_with_flows(benches: dict[str, pd.Series], flows: pd.Series | None,
                          equity: pd.Series | None = None, with_paid: bool = False):
    """The benchmarks' dollar values when they receive the portfolio's own contributions and
    withdrawals (a like-for-like 'account value' comparison), each withdrawal capped at the benchmark's own
    balance. A benchmark that starts after the portfolio starts with the portfolio's account balance on its first
    day and receives the flows after that day, on their original schedule and indexing (so it does not run out
    from starting with the initial capital). with_paid: also return {name: the flows actually made}."""
    if flows is None or float(flows.abs().sum()) == 0:
        return ({}, {}) if with_paid else {}
    out, paid = {}, {}
    for k, b in benches.items():
        if equity is not None and len(equity) > 1 and b.index[0] > equity.index[1]:
            b0 = b.index[0]
            bal = equity.reindex(equity.index.union([b0])).ffill().get(b0)
            if bal is None or not np.isfinite(bal) or bal <= 0:
                continue
            g = b / float(b.iloc[0]) * float(bal)
            out[k], paid[k] = metrics.apply_flows(g, flows[flows.index > b0])
        else:
            start = float(equity.iloc[0]) if equity is not None and len(equity) else None
            out[k], paid[k] = metrics.apply_flows(b, flows, start=start)
    return (out, paid) if with_paid else out


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


def _deflator(idx: pd.DatetimeIndex) -> list | None:
    c = data.cpi()
    if c.empty or not len(idx):
        return None
    ci = c.reindex(idx.union(c.index)).ffill().reindex(idx)
    if ci.isna().all():
        return None
    return _ser(ci / ci.dropna().iloc[0], None, 6)


def _primary_bench(res: Result, benches: dict[str, pd.Series]) -> tuple[str, pd.Series | None]:
    primary = bench_name(default_benchmark(res))
    return primary, benches.get(primary)


# ------------------------------------------------------------------ indicator warm-up

rule_first_defined = expr.first_defined


def _ns(res: Result, t: str):
    df = (res.prices or {}).get(t)
    if df is None:
        try:
            df = data.load(t)
        except (FileNotFoundError, data.DataError, KeyError):
            return None
    from .engine import strategy_namespace
    return strategy_namespace(res.strategy, df, t) if getattr(res, "kind", "") == "signal" else expr.Namespace(df, ticker=t)


def warmup(res: Result) -> tuple[pd.Timestamp | None, list[str]]:
    """(first bar on which every rule can be evaluated, notes) when that is after the run's first bar.

    Signal runs: the entry rules on the ticker with the longest history (only when no trade entered before
    that date). Allocation runs: if-conditions on their ticker, top-N rankings / requirements on their assets
    (the date enough assets have values to fill the N slots), and look-back weightings. Assets of a ranking
    that only get a value later are listed in the notes (excluded until then for lack of history)."""
    idx = res.equity.index
    if len(idx) < 3:
        return None, []
    first = idx[1]
    notes: list[str] = []
    s = res.strategy
    dates: list[pd.Timestamp] = []
    if res.kind == "signal":
        prices = res.prices or {}
        cands = [t for t, df in prices.items() if len(df)]
        if not cands:
            return None, []
        ns = _ns(res, min(cands, key=lambda t: prices[t].index[0]))
        for rule in (s.entry, getattr(s, "short_entry", None)):
            d = rule_first_defined(rule, ns)
            if d is not None:
                dates.append(d)
        warm = max(dates) if dates else None
        if warm is not None and res.trades is not None and not res.trades.empty:
            if pd.Timestamp(min(res.trades["entry_date"])) < warm:
                return None, []
    else:
        tree = getattr(s, "tree", None)
        if not isinstance(tree, dict):
            return None, []
        from . import portfolio as _pf
        # the simulation itself starts on the warm-up day (portfolio.run), so this normally finds nothing to trim;
        # it still lists ranked assets that only get a value later (warmup "first")
        warm, late, waited = _pf.warmup_dates(s, res.prices)
        start = warm if warm is not None and warm > first else first
        for t, d in sorted(late.items(), key=lambda kv: kv[1]):
            if d > start:
                notes.append(f"Lookback: {t} had no value for the ranking until {d.date()} (not enough history), "
                             "so it could not be selected before then.")
    if warm is None or warm <= first or warm > idx[-1]:
        return None, notes
    wd = idx[idx >= warm][0]
    bars = int(((idx >= first) & (idx < wd)).sum())
    if res.kind != "signal" and waited:
        slow = sorted(((t, d) for t, d in waited.items() if d > first), key=lambda kv: kv[1], reverse=True)
        who = ", ".join(f"{t} ({d.date()})" for t, d in slow[:6]) + (f" and {len(slow) - 6} more" if len(slow) > 6 else "")
        notes.insert(0, f"Warm-up: stats start on {wd.date()}, after the {bars}-day warm-up: the first day every rule's "
                        f"indicators have values and all ranked assets have their full lookback"
                        + (f" (waited for {who})" if who else "") + f"; the simulation starts {first.date()}."
                        + (" Set warmup to \"first\" to start once enough assets can fill the slots." if slow else ""))
    else:
        notes.insert(0, f"Warm-up: stats start on {wd.date()}, after the {bars}-day warm-up (the first day every rule's "
                        f"indicators have values; the simulation starts {first.date()}).")
    return wd, notes


def trim_result(res: Result, start) -> Result:
    """A copy of the result whose equity, exposure and holdings begin at `start` (the bar before it becomes the
    starting point). Trades, orders and attribution are kept whole."""
    if start is None:
        return res
    idx = res.equity.index
    pos = int(idx.get_indexer([pd.Timestamp(start)])[0])
    if pos <= 1:
        return res
    cut = idx[pos - 1]

    def sl(x):
        return x[x.index >= cut] if x is not None else None
    extras = dict(res.extras)
    for k in ("flows", "flows_requested"):
        if extras.get(k) is not None:
            extras[k] = extras[k][extras[k].index > cut]
    hold = res.holdings[res.holdings.index >= pd.Timestamp(start)] if res.holdings is not None else None
    return dataclasses.replace(res, equity=sl(res.equity), exposure=sl(res.exposure), positions=sl(res.positions),
                               in_market=sl(res.in_market), holdings=hold, extras=extras)


def trim_end(res: Result, end) -> Result:
    """A copy of the result whose equity, exposure, holdings and flows end at `end` (the day the money ran out):
    after it the account is $0, so charts, yearly rows, holdings statistics and benchmarks stop there too."""
    end = pd.Timestamp(end)

    def sl(x):
        return x[x.index <= end] if x is not None else None
    extras = dict(res.extras)
    for k in ("flows", "flows_requested"):
        if extras.get(k) is not None:
            extras[k] = extras[k][extras[k].index <= end]
    return dataclasses.replace(res, equity=sl(res.equity), exposure=sl(res.exposure), positions=sl(res.positions),
                               in_market=sl(res.in_market), holdings=sl(res.holdings), extras=extras)


def stepped_holdings(res: Result, min_weight: float = 0.05) -> dict[str, list]:
    """{ticker: [(start, end), ...]} for holdings that move in monthly steps (data.stepped_ranges) while held with a
    material weight (an average of at least `min_weight` over the stepped stretch; any trade for a signal run)."""
    idx = res.equity.index
    if len(idx) < 3:
        return {}
    a, b = idx[1], idx[-1]
    out = {}
    if res.kind == "allocation":
        hw = res.holdings
        if hw is None or hw.empty:
            return {}
        cols = [c for c in hw.columns if c not in ("cash", "other")]
        for t, rng in data.stepped_in(cols, a, b).items():
            keep = []
            for ra, rb in rng:
                w = hw[t][(hw.index >= ra) & (hw.index <= rb)].abs()
                if len(w) and float(w.mean()) >= min_weight:
                    keep.append((ra, rb))
            if keep:
                out[t] = keep
        # an opted-in proxy before a holding's inception (Portfolio.proxies): its monthly steps are that holding's
        for t, px in (getattr(res.strategy, "proxies", None) or {}).items():
            if t not in hw.columns:
                continue
            try:
                t0 = data.load(t).index[0]
            except (FileNotFoundError, data.DataError, KeyError, IndexError):
                continue
            for ra, rb in data.stepped_in([px], a, min(b, t0)).get(px, []):
                rb = min(rb, t0)
                w = hw[t][(hw.index >= ra) & (hw.index <= rb)].abs()
                if len(w) and float(w.mean()) >= min_weight:
                    out.setdefault(t, []).append((ra, rb))
            if t in out:
                out[t] = sorted(out[t])
    else:
        tr = res.trades
        if tr is None or tr.empty:
            return {}
        for t, rng in data.stepped_in(list(tr["ticker"].unique()), a, b).items():
            tt = tr[tr["ticker"] == t]
            e0 = pd.to_datetime(tt["entry_date"])
            e1 = pd.to_datetime(tt["exit_date"]).fillna(b)
            keep = [(ra, rb) for ra, rb in rng if ((e0 <= rb) & (e1 >= ra)).any()]
            if keep:
                out[t] = keep
    return out


PRICE_NAMES = {"close", "open", "high", "low", "price", "tr"}
_TRANSPARENT = {"ref", "crossover", "crossunder", "abs", "maximum", "minimum", "log", "sqrt"}   # plotted through
_PAIRS = {"bb_upper": "bb_lower", "bb_lower": "bb_upper", "keltner_upper": "keltner_lower",
          "keltner_lower": "keltner_upper", "donchian_upper": "donchian_lower", "donchian_lower": "donchian_upper"}


def _num_node(n) -> float | None:
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return float(n.value)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        v = _num_node(n.operand)
        return None if v is None else (-v if isinstance(n.op, ast.USub) else v)
    return None


def _sym_ticker(n) -> str | None:
    """'SPY' for the node sym("SPY") (the call itself, not an attribute of it)."""
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "sym" and n.args
            and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
        return data.canonical(n.args[0].value)
    return None


def chart_layout(rules, own: str = "") -> dict:
    """Every series a rule uses, laid out for the price chart of ticker `own`, from the rules' syntax trees.

    Returns {"overlays": [expr], "panes": [{name, calls, levels, range}], "step": {expr}}. Price-scale indicators
    of the charted ticker (moving averages, bands, channels, stops, VWAP, weekly/monthly averages) are overlays;
    oscillators, returns, volatility, volume and the other rule variables get one pane per family; another
    ticker's series (sym("SPY")...) get that ticker's pane with its close. Weekly / monthly series (weekly_*,
    monthly_*, weekly(...), monthly(...)) are listed in "step": they hold each completed period's value, as the
    engine reads them, and are drawn as steps. Levels are the numbers the rule compares a series with."""
    overlays: list[str] = []
    panes: dict[str, dict] = {}
    step: set[str] = set()
    where: dict[str, str | None] = {}   # expr -> pane key (None: overlay)
    own = data.canonical(own) if own else ""

    def pane(key, rng=None):
        return panes.setdefault(key, {"name": key, "calls": [], "levels": [], "range": list(rng) if rng else None})

    def put(key, call, rng=None, periodic=False):
        if call in where or len(where) >= MAX_SERIES:
            return
        where[call] = key
        if periodic:
            step.add(call)
        if key is None:
            overlays.append(call)
        else:
            pane(key, rng)["calls"].append(call)

    def base(n):
        """Where a series argument lives: ("price",), ("sym", T), ("pane", key)."""
        if isinstance(n, ast.Name):
            if n.id in PRICE_NAMES or n.id in OVERLAY_NAMES:
                return ("price",)
            if n.id in VARIABLE_PANES:
                return ("pane", VARIABLE_PANES[n.id][0])
            return None
        if isinstance(n, ast.Attribute):
            tk = _sym_ticker(n.value)
            if tk:
                if n.attr == "volume":
                    return ("pane", "Volume" if tk == own else f"{tk} volume")
                return ("price",) if tk == own else ("sym", tk)
            return None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            fn = n.func.id
            if fn in PRICE_FNS:
                return base(_series_arg(n)) or ("price",)
            if fn in OSCILLATORS:
                b = base(_series_arg(n))
                fam = OSCILLATORS[fn][0]
                return ("pane", f"{b[1]} {fam}") if b and b[0] == "sym" else ("pane", fam)
            if fn in ("weekly", "monthly") and n.args:
                return base(n.args[0])
        return None

    def _series_arg(n):
        for a in n.args:
            if _num_node(a) is None:
                return a
        return None

    def add_call(n, periodic=False, shift=0):
        fn = n.func.id
        src = _src(n)
        # a series the rule reads n bars back (ref(x, n), Pine's x[n]) is drawn with that offset, labelled x[n]
        emit = put if not shift else (lambda key, call, rng=None, periodic=False:
                                      put(key, f"{call}[{shift}]", rng, periodic))
        arg = _series_arg(n)
        b = base(arg) if arg is not None else ("price",)
        periodic = periodic or fn.startswith(("weekly_", "monthly_"))
        if fn in PRICE_FNS:
            if b is None or b[0] == "price":
                calls = [src]
                if fn in _PAIRS and all(_num_node(a) is not None for a in n.args):
                    calls.append(f"{_PAIRS[fn]}({', '.join(_src(a) for a in n.args)})")
                    if fn.startswith("bb_"):   # the Bollinger basis
                        calls.append(f"sma({_src(n.args[0]) if n.args else '20'})")
                for c in calls:
                    emit(None, c, periodic=periodic)
            elif b[0] == "sym":
                put(b[1], f'sym("{b[1]}").close')
                emit(b[1], src, periodic=periodic)
            else:   # an average of an oscillator, of volume...: on that series' pane
                emit(b[1], src, periodic=periodic)
            return
        fam, rng = OSCILLATORS[fn]
        key = f"{b[1]} {fam}" if b and b[0] == "sym" else fam
        if fn in ("macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d", "stoch_rsi_k", "stoch_rsi_d") \
                and (b is None or b[0] == "price") \
                and all(_num_node(a) is not None for a in n.args):
            for c in _companions(fn, ", ".join(_src(a) for a in n.args)):
                emit(key, src if _norm(c) == _norm(src) else c, rng, periodic)
            return
        if fn in ("adx", "plus_di", "minus_di") and all(_num_node(a) is not None for a in n.args):
            args = ", ".join(_src(a) for a in n.args)
            for f in ("adx", "plus_di", "minus_di"):
                c = f"{f}({args})"
                emit(key, src if f == fn else c, rng, periodic)
            return
        emit(key, src, rng, periodic)

    def walk(n, periodic=False):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            fn = n.func.id
            if fn in ("weekly", "monthly") and len(n.args) == 1:
                inner = n.args[0]
                src = _src(n)
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id in PRICE_FNS \
                        and (base(inner) or ("price",))[0] == "price":
                    put(None, src, periodic=True)
                else:
                    b = base(inner)
                    ifn = inner.func.id if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) else ""
                    rng = OSCILLATORS[ifn][1] if ifn in OSCILLATORS else None
                    put(f"{b[1]} ({fn})" if b and b[0] in ("pane", "sym") else src, src, rng, periodic=True)
                return
            if fn == "sym":
                return
            if fn == "ref" and len(n.args) == 2 and isinstance(_num_node(n.args[1]), (int, float)) \
                    and float(_num_node(n.args[1])).is_integer() and _num_node(n.args[1]) > 0:
                inner, k = n.args[0], int(_num_node(n.args[1]))
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and \
                        (inner.func.id in PRICE_FNS or inner.func.id in OSCILLATORS):
                    add_call(inner, periodic, shift=k)
                    shifted = f"{_src(inner)}[{k}]"
                    if shifted in where:
                        where[_src(n)] = where[shifted]
                    for a in inner.args:     # the series it is computed from (e.g. an oscillator of another ticker)
                        if not (isinstance(a, ast.Name) or _num_node(a) is not None):
                            walk(a, periodic)
                    return
                if isinstance(inner, ast.Name) and inner.id in VARIABLE_PANES:
                    fam, rng = VARIABLE_PANES[inner.id]
                    put(fam, f"{inner.id}[{k}]", rng)
                    where[_src(n)] = fam
                    return
            if fn in PRICE_FNS or fn in OSCILLATORS:
                add_call(n, periodic)
            for a in n.args:
                walk(a, periodic)
            return
        if isinstance(n, ast.Attribute):
            tk = _sym_ticker(n.value)
            if tk and tk != own and n.attr in PRICE_FIELDS:
                put(tk, f'sym("{tk}").close')
            elif tk and n.attr == "volume":
                put("Volume" if tk == own else f"{tk} volume", _src(n))
            return
        if isinstance(n, ast.Name):
            if n.id in VARIABLE_PANES:
                fam, rng = VARIABLE_PANES[n.id]
                put(fam, n.id, rng)
            elif n.id in OVERLAY_NAMES:
                put(None, n.id)
            return
        for c in ast.iter_child_nodes(n):
            walk(c, periodic)

    trees = []
    for r in ([rules] if isinstance(rules, str) else rules or []):
        if not isinstance(r, str) or not r.strip():
            continue
        try:
            t = ast.parse(r.strip(), mode="eval")
        except SyntaxError:
            continue
        trees.append(t)
        walk(t.body)
    # thresholds: numbers the rule compares a charted series with
    for t in trees:
        for n in ast.walk(t):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("crossover", "crossunder", "cross")
                    and len(n.args) == 2):
                ops = list(n.args)       # crossover(rsi(close, 14), 50): the 50 line is a level of the RSI pane
            elif isinstance(n, ast.Compare):
                ops = [n.left] + list(n.comparators)
            else:
                continue
            for x, y in zip(ops, ops[1:]):
                for a, b in ((x, y), (y, x)):
                    v, src = _num_node(b), _src(a)
                    key = where.get(src, "?")
                    if v is not None and key not in ("?", None) and v not in panes[key]["levels"]:
                        panes[key]["levels"].append(v)
    return {"overlays": overlays, "panes": list(panes.values()), "step": step}


def _src(n) -> str:
    """The node as rule text, with sym("X") in double quotes as the rules are written."""
    return re.sub(r"sym\('([^'\"]*)'\)", r'sym("\1")', ast.unparse(n))


def _norm(call: str) -> str:
    return re.sub(r"\s+", "", call)


def _chart_setup(res: Result) -> tuple[list[str], list[dict], int]:
    L = chart_layout(_rules_list(res))
    return L["overlays"], L["panes"], 6 + len(L["overlays"]) + sum(len(p["calls"]) for p in L["panes"])


def chart_tickers(res: Result) -> list[str]:
    """Every traded ticker with price data, most-traded first (signal runs only)."""
    if res.trades is None or res.trades.empty or res.kind != "signal":
        return []
    return [t for t in res.trades["ticker"].value_counts().index if (res.prices or {}).get(t) is not None]


def _chart_segment(res: Result, t: str, many: bool) -> pd.DataFrame:
    df = res.prices[t]
    seg = df[(df.index >= res.equity.index[1]) & (df.index <= res.equity.index[-1])]
    if many:
        # many tickers: only the stretch around this ticker's trades, so more of them fit the budget
        tt = res.trades[res.trades["ticker"] == t]
        lo = seg.index.searchsorted(pd.Timestamp(min(tt["entry_date"])))
        hi = seg.index.searchsorted(pd.Timestamp(max(tt["exit_date"])), side="right")
        seg = seg.iloc[max(0, lo - 300): hi + 60]
    return seg


SYM_CALL_RE = re.compile(r"""sym\(\s*["']([^"']+)["']\s*\)\.(\w+)(\(([^()]*)\))?""")
PRICE_FIELDS = ("close", "open", "high", "low", "tr", "adj_close")


def other_ticker_panes(rules: str, own: str) -> list[dict]:
    """Other tickers a rule filters on (sym("SPY").close > sym("SPY").sma(200)): one pane per ticker with its
    close and the indicators the rule applies to it. Oscillators (RSI...) get their own range."""
    out: dict[str, dict] = {}
    for m in SYM_CALL_RE.finditer(rules or ""):
        tk, fn, call, args = data.canonical(m.group(1)), m.group(2), m.group(3), m.group(4)
        if tk == own:
            continue
        p = out.setdefault(tk, {"name": tk, "calls": [f'sym("{m.group(1)}").close'], "levels": [], "range": None})
        if call is None and fn in PRICE_FIELDS:
            continue
        expr_ = m.group(0)
        if fn in OSCILLATORS:
            p.setdefault("osc", []).append(expr_)
            for lv in _levels(expr_, rules):
                if lv not in p["levels"]:
                    p["levels"].append(lv)
        elif expr_ not in p["calls"] and len(p["calls"]) < 4:
            p["calls"].append(expr_)
    # fn(sym("SPY").close, 200): an indicator of another ticker's series
    for m in re.finditer(r"""(?<![\w.])(\w+)\(\s*sym\(\s*["']([^"']+)["']\s*\)\.(\w+)\s*((?:,[^()]*)?)\)""", rules or ""):
        fn, tk = m.group(1), data.canonical(m.group(2))
        if tk == own or fn == "sym":
            continue
        p = out.setdefault(tk, {"name": tk, "calls": [f'sym("{m.group(2)}").close'], "levels": [], "range": None})
        if fn in OSCILLATORS:
            p.setdefault("osc", []).append(m.group(0))
            for lv in _levels(m.group(0), rules):
                if lv not in p["levels"]:
                    p["levels"].append(lv)
        elif m.group(0) not in p["calls"] and len(p["calls"]) < 4:
            p["calls"].append(m.group(0))
    panes = []
    for tk, p in out.items():
        if p.get("osc"):   # the ticker's oscillator on its own scale; its price would flatten it
            m0 = SYM_CALL_RE.match(p["osc"][0])
            fam = OSCILLATORS[m0.group(2) if m0 else p["osc"][0].split("(")[0]]
            panes.append({"name": f"{tk} {fam[0]}", "calls": p["osc"][:3], "levels": p["levels"],
                          "range": list(fam[1]) if fam[1] else None})
            if len(p["calls"]) > 1:
                panes.append({"name": tk, "calls": p["calls"], "levels": [], "range": None})
        else:
            panes.append({"name": tk, "calls": p["calls"], "levels": [], "range": None})
    return panes


def _label(call: str) -> str:
    m = SYM_CALL_RE.fullmatch(call)
    if m:
        return f"{data.canonical(m.group(1))} {m.group(2)}{m.group(3) or ''}"
    m = re.fullmatch(r"""(\w+)\(\s*sym\(\s*["']([^"']+)["']\s*\)\.(\w+)\s*,?\s*([^()]*)\)""", call)
    if m:
        return f"{data.canonical(m.group(2))} {m.group(1)}({m.group(3) + ', ' if m.group(3) != 'close' else ''}{m.group(4)})"
    return call


def rule_state(res: Result, t: str, index: pd.DatetimeIndex) -> str | None:
    """Entry / exit rule state on each bar as one character: "0" neither, "1" entry, "2" exit, "3" both (the
    booleans the engine acted on). None when the run has no rule state for this ticker."""
    rs = (res.extras or {}).get("rule_state")
    if not rs or t not in rs["tick"]:
        return None
    j = rs["tick"].index(t)
    cal = rs["cal"]
    e = pd.Series(rs["entry"][:, j], index=cal).reindex(index, fill_value=False).to_numpy(dtype=bool)
    code = e.astype(int)
    if rs.get("exit") is not None:
        x = pd.Series(rs["exit"][:, j], index=cal).reindex(index, fill_value=False).to_numpy(dtype=bool)
        code = code + 2 * x.astype(int)
    return "".join("0123"[c] for c in code)


def ticker_chart(res: Result, t: str, setup=None, seg: pd.DataFrame | None = None) -> dict:
    """OHLC + every series the rules use, for one ticker (see chart_layout): overlays on the price, one pane per
    oscillator family and per other ticker, "step" for weekly / monthly series (drawn as steps), and the
    bar-by-bar entry / exit rule state. Values come from the same Namespace the engine evaluates the rules on."""
    L = chart_layout(_rules_list(res), t)
    if seg is None:
        seg = _chart_segment(res, t, res.trades["ticker"].nunique() > 1)
    from .engine import strategy_namespace
    ns = (strategy_namespace(res.strategy, res.prices[t], t) if getattr(res, "kind", "") == "signal"
          else expr.Namespace(res.prices[t], ticker=t))
    overlays, step = {}, []
    for c in L["overlays"]:
        v = _values(c, ns, seg.index)
        if v is not None:
            overlays[c] = v
            if c in L["step"]:
                step.append(c)
    tpanes = []
    for p in L["panes"]:
        series = {}
        for c in p["calls"]:
            v = _values(c, ns, seg.index)
            if v is not None:
                series[_label(c)] = v
                if c in L["step"]:
                    step.append(_label(c))
        if series:
            tpanes.append({"name": p["name"], "series": series, "levels": p["levels"], "range": p["range"]})
    out = {
        "dates": [d.strftime("%Y-%m-%d") for d in seg.index],
        "o": seg["open"].round(4).tolist(), "h": seg["high"].round(4).tolist(),
        "l": seg["low"].round(4).tolist(), "c": seg["close"].round(4).tolist(),
        "overlays": overlays, "panes": tpanes,
    }
    if step:
        out["step"] = step
    rs = rule_state(res, t, seg.index)
    if rs is not None:
        out["rs"] = rs
        out["rs_exit"] = (res.extras.get("rule_state") or {}).get("exit") is not None
    return out


def _rules_list(res: Result) -> list[str]:
    s = res.strategy
    return [r for r in (s.entry, getattr(s, "short_entry", None), s.exit_when, getattr(s, "entry_level", None))
            if isinstance(r, str) and r]


def _rules_text(res: Result) -> str:
    return " ".join(_rules_list(res))


def price_payload(res: Result, budget: int = MAX_PRICE_POINTS, max_tickers: int = MAX_PRICE_TICKERS) -> dict:
    """OHLC + indicators for the traded tickers (most-traded first), within a size budget.

    Price-scale indicators (moving averages, bands, stops) go in "overlays"; oscillators used by the rules
    (RSI, MACD, stochastic, ADX, CCI, Williams %R, MFI...) go in "panes", one sub-pane per indicator family,
    with the rule's numeric thresholds as "levels".
    """
    tks = chart_tickers(res)
    if not tks:
        return {}
    setup = _chart_setup(res)
    many = len(tks) > 1
    out, used = {}, 0
    for t in tks:
        if len(out) >= max_tickers:
            break
        seg = _chart_segment(res, t, many)
        cost = len(seg) * setup[2]
        if used + cost > budget and out:
            continue  # a less-traded ticker with a shorter stretch may still fit
        out[t] = ticker_chart(res, t, setup, seg)
        used += cost
    return out


def chart_file_name(t: str, run: int | None = None) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", t)
    return f"charts/{'' if run is None else f'r{run + 1}-'}{safe}.js"


def write_chart_files(res: Result, out_dir: Path, skip: set, run: int | None = None) -> dict[str, str]:
    """One file per traded ticker not embedded in the report (charts/<TICKER>.js next to report.html), loaded
    by the report when that ticker is picked. It is a script that registers the JSON payload, so it loads
    both from a web server and from a report opened as a local file (file://, where fetch() is blocked)."""
    files = {}
    tks = [t for t in chart_tickers(res) if t not in skip]
    if not tks:
        return files
    setup = _chart_setup(res)
    many = len(chart_tickers(res)) > 1
    (out_dir / "charts").mkdir(parents=True, exist_ok=True)
    for t in tks:
        name = chart_file_name(t, run)
        P = ticker_chart(res, t, setup, _chart_segment(res, t, many))
        ds = pd.to_datetime(P.pop("dates"))
        if len(ds):  # dates as the first date + day steps (the report rebuilds the list): ~40% smaller files
            P["d0"] = ds[0].strftime("%Y-%m-%d")
            P["dd"] = [int(x) for x in np.diff(ds.values).astype("timedelta64[D]").astype(int)]
        blob = json.dumps(_clean(P), separators=(",", ":"))
        (out_dir / name).write_text(f"(window.__charts=window.__charts||{{}})[{json.dumps(name)}]={blob};\n")
        files[t] = name
    return files


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


def signal_attribution(res: Result) -> pd.DataFrame | None:
    """P&L per ticker for a signal run, from its trades (closed and still open): sum(pnl) + interest equals
    final equity - starting capital."""
    tr = res.trades
    if res.kind != "signal" or tr is None or tr.empty:
        return None
    g = tr.groupby("ticker")
    closed, opened = metrics.split_open(tr)
    df = pd.DataFrame({
        "pnl": g["pnl"].sum(),
        "trades": closed.groupby("ticker").size().reindex(g.size().index).fillna(0).astype(int),
        "open_pnl": opened.groupby("ticker")["pnl"].sum().reindex(g.size().index).fillna(0.0),
        "win_rate": closed.groupby("ticker")["pnl"].apply(lambda p: (p > 0).mean()).reindex(g.size().index),
        "dividends": g["income"].sum() if "income" in tr else 0.0,
        "commissions": g["commission"].sum() if "commission" in tr else 0.0,
    }).reset_index()
    tot = df["pnl"].sum()
    df["share_of_pnl"] = df["pnl"] / tot if abs(tot) > 1e-9 else np.nan
    return df.sort_values("pnl", key=lambda s: -s.abs()).reset_index(drop=True)


def _has_rules(tree) -> bool:
    """Does a portfolio tree switch or rank on data (if-nodes, filters)?"""
    if not isinstance(tree, dict):
        return False
    if "if" in tree or "filter" in tree:
        return True
    return any(_has_rules(k) for k in (tree.get("children") or []))


def holdings_stats(res: Result, rf="tbill", limit: int = 12) -> list[dict]:
    """Per-asset statistics of an allocation run's holdings over the run (PV's asset table): each asset's own
    total return from the later of the run's first bar and its first data, with its average weight."""
    hw = res.holdings
    if res.kind != "allocation" or hw is None or hw.empty or len(res.equity) < 3:
        return []
    from . import correlation
    w = hw.drop(columns=[c for c in ("cash", "other") if c in hw])
    avg = w.abs().mean().sort_values(ascending=False)
    a, b = res.equity.index[1], res.equity.index[-1]
    out = []
    for t in list(avg.index[:limit]):
        try:
            first = data.load(t).index[0]
        except (FileNotFoundError, data.DataError, KeyError):
            continue
        st = correlation.asset_stats(t, max(a, first), b, rf, first.date())
        if st.get("cagr") is None:
            continue
        st["avg_weight"] = float(w[t].mean())
        st["time_held"] = float((w[t].abs() > 1e-6).mean())
        out.append(st)
    return out


def flow_free_nav(spec, start=None) -> pd.Series | None:
    """The allocation re-run with no contributions or withdrawals: its equity is the time-weighted growth of the
    asset mix over the whole requested period, whatever the cash flows did (even after they emptied the account).
    From `start` on (the first day of the reported run). None if it cannot be run."""
    from . import portfolio as _pf
    try:
        p0 = dataclasses.replace(spec, contribution=0.0, withdrawal=0.0, withdrawal_pct=0.0,
                                 notes=[], tree=json.loads(json.dumps(spec.tree)))
        eq = _pf.run(p0).equity
    except Exception:  # noqa: BLE001 - the rates then fall back to the run's own returns
        return None
    if start is not None:
        eq = eq[eq.index >= pd.Timestamp(start)]
    return eq if len(eq) > 2 else None


def analyze(res: Result, rf="tbill", sensitivity: bool = True, mc: bool = True, detail: bool = True) -> dict:
    s = res.strategy
    full = res
    warm, warm_notes = warmup(res)
    for n in warm_notes:
        if n not in s.notes:
            s.notes.append(n)
    res = trim_result(res, warm)
    flows = res.extras.get("flows")
    has_flows = flows is not None and float(flows.abs().sum()) > 0
    nv = metrics.floor_at_zero(metrics.nav(res.equity, flows) if has_flows else res.equity)
    nv_all = nv
    first_bar = res.equity.index[1] if len(res.equity) > 1 else None
    # money ran out: everything (statistics, yearly rows, charts, holdings, benchmarks) stops on that day (a $0
    # balance earns no return); the Monte Carlo still replays the whole scheduled period
    dep = res.extras.get("depleted") if has_flows else None
    res_run = res
    if dep is not None and res.equity.index[0] < dep < res.equity.index[-1]:
        res = trim_end(res, dep)
        flows = res.extras.get("flows")
        nv = nv[nv.index <= dep]
        nv_all = nv
        stats = metrics.equity_stats(res.equity, rf, flows, first_bar=first_bar,
                                     years_from_first_bar=res.kind == "signal")
        stats["depleted"] = dep.date()
    else:
        dep = None
        stats = metrics.equity_stats(res.equity, rf, flows if has_flows else None, first_bar=first_bar,
                                     years_from_first_bar=res.kind == "signal")
    tstats = metrics.trade_stats(res.trades, stats["years"])
    no_trades = res.kind == "signal" and not (tstats.get("trades") or tstats.get("open_trades"))
    warnings = metrics.result_warnings(res.kind, stats, tstats, res.interest, has_flows)
    if no_trades and res.prices:
        # no trades because an indicator's look-back is longer than the history: say so, not "never triggered"
        t_long = max(res.prices, key=lambda t: len(res.prices[t]))
        ns_ = _ns(res, t_long)
        for rule in (getattr(s, "entry", None), getattr(s, "short_entry", None)):
            cold = expr.never_defined(rule, ns_) if ns_ is not None else []
            if cold:
                seg, n = cold[0]
                for w in warnings:
                    if w.get("code") == "no_trades":
                        w["detail"] = (f"The indicator never warmed up: {seg} needs {n or 'more'} bars, and {t_long}'s data "
                                       f"has {len(res.prices[t_long])}. Shorten the look-back.")
                break
    if any(isinstance(n, str) and n.startswith("Warning: insufficient funds") for n in s.notes):
        for w in warnings:
            if w.get("code") == "no_trades":
                w["detail"] = ("The entry condition triggered, but every order was skipped for insufficient funds: the fixed "
                               "size costs more than the account can pay (see the warning below).")
    # notes that reinterpret the user's words ("it" resolved to the entry's indicator...) are shown as warnings
    for n in s.notes:
        if isinstance(n, str) and n.startswith("Warning:"):
            warnings.append({"code": "interpretation", "level": "warn", "message": "Check the interpretation",
                             "detail": n[len("Warning:"):].strip()})
    # Nasdaq-100 point-in-time universes: the survivorship coverage belongs in the headline, not only in the notes
    sv_note = next((n for n in s.notes if isinstance(n, str) and n.startswith("Survivorship:")), None)
    if sv_note and len(res.equity):
        try:
            # the period the run's note was computed for (its first and last trading day), so the headline quotes the
            # same figures; for a saved run without it, the equity's first real day (index 0 is the anchor before it)
            win = data.coverage_window(sv_note) or (str(res.equity.index[min(1, len(res.equity) - 1)].date()),
                                                    str(res.equity.index[-1].date()))
            sv = data.survivorship(*win)
        except Exception:  # noqa: BLE001 - a caveat must never break a report
            sv = None
        if sv:
            detail = []
            if sv.get("bias"):
                detail.append(sv["bias"]["text"])
            if sv.get("missing"):
                detail.append("Biggest missing members: " + ", ".join(f"{t} ({k} member-months)" for t, k in sv["missing"])
                              + ". " + data.TIINGO_HINT)
            warnings.insert(0, {"code": "survivorship", "level": "warn", "message": sv["headline"],
                                "detail": " ".join(detail), "coverage": sv["coverage"]})
    if not no_trades:
        for c in metrics.caveats(stats):
            warnings.append({"code": "volatility_drag", "level": "info", "message": "Negative CAGR with a positive Sharpe ratio",
                             "detail": c})
    if res.kind == "allocation" and getattr(s, "fill", "close") == "close" and _has_rules(getattr(s, "tree", None)):
        moc = ("Same-bar fills: the rules are evaluated on each rebalance day's close and traded at that same close, "
               "which assumes a market-on-close order placed just before the close with near-close prices. "
               "Choose 'trade at the next open' (fill: next_open) for a fill that only uses the finished bar.")
        if moc not in s.notes:
            s.notes.append(moc)
    if no_trades:
        stats = metrics.suppress_degenerate(stats)
    expo = metrics.exposure_stats(res.exposure, res.positions, res.in_market)
    # a holding that moves in monthly steps (EEMSIM before 2003, VNQSIM before 2004...): its daily returns are 0 on
    # most days and the month's move on one, so daily statistics describe the steps, not the asset
    stepped = {} if no_trades else stepped_holdings(res)
    basis = "monthly" if stepped else "daily"
    if stepped:
        stats = metrics.monthly_basis(stats, nv, rf)
        msg = (f"{data.stepped_text(stepped)} {'move' if len(stepped) > 1 else 'moves'} only once a month in this data "
               "(a monthly source spread over daily sessions). Volatility, Sharpe, Sortino, skew, kurtosis, beta/alpha and "
               "the factor regression are computed from monthly returns; daily figures (best/worst day, positive days, "
               "daily VaR/CVaR) are left blank.")
        warnings.append({"code": "monthly_steps", "level": "warn", "message": "Monthly-stepped data", "detail": msg})
    skipped: dict = {}
    benches = benchmark_series(res, nv, skipped)
    if has_flows:
        # a like-for-like comparison with cash flows: a benchmark that cannot receive the same flows from a balance
        # the portfolio actually had (its data starts after the money ran out) is left out, never shown without flows
        sched0 = res.extras.get("flows_requested")
        sched0 = sched0[sched0.index >= res.equity.index[0]] if sched0 is not None else flows
        can = benchmarks_with_flows(benches, sched0, res.equity)
        for k in [k for k in benches if k not in can]:
            skipped[k] = benches.pop(k).index[0]
        for k, d0 in skipped.items():
            if d0 is None:
                continue
            d0 = pd.Timestamp(d0)
            why = (f"after the portfolio ran out of money on {dep.date()}" if dep is not None and d0 >= dep
                   else "too close to the end of the period")
            n = (f"{k} is left out of the comparison: its data starts {d0.date()}, {why}. Benchmarks here receive the "
                 "same contributions and withdrawals as the portfolio, starting from the portfolio's balance on their "
                 "first day, so one that starts after the money ran out has nothing to compare.")
            if n not in s.notes:
                s.notes.append(n)
    pname, pseries = _primary_bench(res, benches)
    rel = {} if no_trades else metrics.relative_stats(nv, pseries, rf, freq=basis)  # beta/alpha of idle cash mean nothing
    yearly = metrics.yearly_detail(nv_all, res.trades, res.exposure, first_bar=first_bar)
    bal = metrics.yearly_balances(res.equity, nv_all, flows if has_flows else None)
    for c in bal.columns:
        yearly[c] = bal[c].reindex(yearly.index)
    monthly = metrics.monthly_table(nv_all)
    if dep is not None:
        # after the money ran out there is nothing to earn a return on: blank, not 0%
        after = yearly.index.astype(int) > dep.year
        for c in ("return", "max_drawdown", "real_return"):
            if c in yearly:
                yearly.loc[after, c] = np.nan
        monthly.loc[monthly.index > dep.year] = np.nan
        monthly.loc[dep.year, monthly.columns[dep.month:]] = np.nan
    yr_b = metrics.yearly_returns(benches)
    for n in yr_b:
        yearly[n] = yr_b[n].reindex(yearly.index)
    attribution = res.extras.get("attribution")
    if attribution is None:
        attribution = signal_attribution(full)
    A = {
        "result": res, "result_full": full, "warmup_start": warm, "strategy": s, "nav": nv,
        "flows": flows if has_flows else None,
        "stats": stats, "cash": metrics.cashflow_stats(res.equity, flows if has_flows else None),
        "trade_stats": tstats, "exposure": expo, "relative": rel, "primary_benchmark": pname,
        "benchmarks": benches, "benchmark_from": benchmark_coverage(benches, first_bar),
        "benchmark_partial": benchmark_partial_years(benches, first_bar),
        "yearly": yearly, "monthly": monthly, "depleted": dep.date() if dep is not None else None,
        "return_basis": basis,
        "stepped": {t: [[str(a.date()), str(b.date())] for a, b in r] for t, r in stepped.items()},
        "drawdowns": metrics.drawdown_table(nv, 5, first_bar=first_bar), "rf": rf, "interest": res.interest,
        "turnover": res.extras.get("turnover_annual"), "rebalances": res.extras.get("rebalances"),
        "warnings": warnings, "no_trades": no_trades,
        "first_bar": first_bar, "fees": res.extras.get("fees", 0.0),
        "equity_real": real_equity(res.equity),
        "attribution": attribution,
        "trailing": {"Strategy": metrics.trailing_returns(nv, first_bar),
                     **{k: metrics.trailing_returns(b, first_bar, end=nv.index[-1]) for k, b in benches.items()}},
    }
    # the benchmarks receive the flows as scheduled (each capped at its own balance), not the strategy's capped ones
    sched = res.extras.get("flows_requested") if has_flows else None
    sched = sched[sched.index >= res.equity.index[0]] if sched is not None else (flows if has_flows else None)
    bwf, bflows = benchmarks_with_flows(benches, sched, res.equity, with_paid=True)
    A["benchmarks_with_flows"] = bwf
    A["benchmark_cash"] = {}
    for k, v in bwf.items():
        b0 = benches[k].index[0]
        late = len(res.equity) > 1 and b0 > res.equity.index[1]
        cs = metrics.cashflow_stats(v, bflows[k])
        if late:
            cs["from"] = b0.date()
            cs["note"] = "starts with the portfolio's balance on its first day"
        A["benchmark_cash"][k] = cs
    if res.kind == "allocation":
        from . import risk
        A["risk_contributions"] = risk.risk_contributions(res, nv)
        # Portfolio Visualizer's annual income table and each holding's calendar-year returns
        A["income_yearly"] = metrics.income_yearly(res.equity, res.extras.get("income"))
        held = [c for c in (res.holdings.columns if res.holdings is not None else []) if c != "cash"]
        A["asset_yearly"] = metrics.asset_yearly_returns(res.prices or {}, held[:40], res.equity.index)
    A["withdrawal_rates"] = {}
    if res.kind == "allocation" and (getattr(s, "withdrawal", 0) or getattr(s, "withdrawal_pct", 0)):
        from . import montecarlo
        # with contributions first (save, then retire), a rate "of the starting balance" means nothing: the rates
        # are measured from the balance on the first withdrawal, over the withdrawal years
        wd = sched[sched < 0] if sched is not None else pd.Series(dtype=float)
        contrib = sched is not None and bool((sched > 0).any())
        base = None
        if contrib and len(wd) and first_bar is not None and wd.index[0] > pd.Timestamp(first_bar) + pd.Timedelta(days=31):
            base = wd.index[0]
        # measured on the portfolio's own returns without any cash flows, over the whole requested period (or the
        # whole withdrawal phase): the rate must not depend on the amount entered, which can empty the account early
        tw = flow_free_nav(s, res.equity.index[0])
        if tw is None:
            tw = nv
        nv_w = tw[tw.index >= base] if base is not None else tw
        every = {"monthly": 1, "quarterly": 3, "semiannual": 6}.get(getattr(s, "withdrawal_freq", "yearly"), 12)
        A["withdrawal_rates"] = (montecarlo.historical_withdrawal_rates(metrics.monthly_returns(nv_w), every=every)
                                 if len(nv_w) > 2 else {})
        if A["withdrawal_rates"]:
            A["withdrawal_rates"].update({"from": stats["start"] if base is None else base.date(), "to": tw.index[-1].date(),
                                          "basis": "flow_free_returns", "freq": getattr(s, "withdrawal_freq", "yearly")})
            if base is not None:
                bal = res.equity.reindex(res.equity.index.union([base])).ffill().get(base)
                if bal is not None and flows is not None and base in flows.index:
                    bal = bal - float(flows.get(base, 0.0))      # the balance before the first withdrawal
                A["withdrawal_rates"].update({"base": "withdrawal_start", "base_date": base.date(),
                                              "base_balance": float(bal) if bal is not None else None})
    if detail:
        sched_run = res_run.extras.get("flows_requested") if has_flows else None
        sched_run = (sched_run[sched_run.index >= res_run.equity.index[0]] if sched_run is not None
                     else res_run.extras.get("flows") if has_flows else None)
        A["monte_carlo"] = (metrics.monte_carlo(res_run.equity, res_run.extras.get("flows") if has_flows else None,
                                                schedule=sched_run, until=dep)
                            if mc and not no_trades else {})
        A["sensitivity"] = cost_sensitivity(full, warm=warm) if sensitivity and not no_trades else []
        A["rolling"] = metrics.rolling_series(nv, pseries, rf)
        A["rolling_summary"] = metrics.rolling_summary(nv)
        A["crises"] = metrics.crisis_table({"Strategy": nv, **benches})
        A["factors"] = metrics.factor_regression(nv, rf, freq=basis)
        corr_in = {"Strategy": nv, **{k.replace(" buy & hold", ""): v for k, v in benches.items()}}
        if res.holdings is not None and not res.holdings.empty and res.kind == "allocation":
            for t in [c for c in res.holdings.columns if c != "cash"][:6]:
                b = metrics.buy_and_hold(t, nv.index, 1.0)
                if b is not None and t not in corr_in:
                    corr_in[t] = b
        A["correlation"] = metrics.correlation_matrix(corr_in)
        A["asset_stats"] = holdings_stats(res, rf)
    full.__dict__.pop("_evaluations", None)   # the cost reruns are done: free the evaluator's indicator caches
    return A


def _window_stats(series: dict, start, end, fb, rf, blank, monthly=(), base: float = 10_000,
                  balances: dict | None = None) -> dict:
    """Stats of each series over start..end as the growth of `base` (the runs' starting amount). `balances`:
    {name: dollar account value with the cash flows}, whose value at the end is added as "final_balance"."""
    out = {}
    for k, s in series.items():
        seg = s[(s.index >= start) & (s.index <= end)]
        seg = seg[seg.index >= seg[seg > 0].index[0]] if (seg > 0).any() else seg.iloc[:0]
        if len(seg) < 30:
            continue
        st = metrics.equity_stats(seg / seg.iloc[0] * base, rf, first_bar=fb)
        if balances and k in balances and len(balances[k]):
            b = balances[k][balances[k].index <= seg.index[-1]]
            if len(b):
                st["final_balance"] = float(b.iloc[-1])
        if k in monthly:   # holds a series that moves in monthly steps: as in its own report
            st = metrics.monthly_basis(st, seg, rf)
        out[k] = metrics.suppress_degenerate(st) if k in blank else st
    return out


def common_window_stats(analyses: list[dict], rf="tbill") -> dict:
    """Stats for every run and benchmark over the runs' common period ("columns": everything in it covers exactly
    start -> end, like for like). A benchmark whose data starts after the common start is NOT in "columns": it is
    listed separately in "benchmarks_own", measured over its own dates ("benchmarks_own_from" / "_to"), so a young
    default benchmark such as QQQ neither cuts the runs' period short nor sits under the common-period heading with
    other dates. Also over the whole test period where each exists ("full"; later starters in "full_from")."""
    series, blank, runs, monthly = {}, set(), [], set()
    # growth of the runs' own starting amount (not a fixed $10,000); with cash flows the growth is time-weighted
    # (the flows left out) and each account's real final balance is given as well
    base = float(analyses[0]["nav"].iloc[0]) if len(analyses[0]["nav"]) and analyses[0]["nav"].iloc[0] > 0 else 10_000.0
    balances: dict = {}
    has_flows = False
    for i, A in enumerate(analyses):
        name = _run_name(A["result"], i) if len(analyses) > 1 else "Strategy"
        series[name] = A["nav"]
        runs.append(name)
        fl = (getattr(A["result"], "extras", None) or {}).get("flows")
        if fl is not None and float(fl.abs().sum()) > 0:
            has_flows = True
            balances[name] = getattr(A["result"], "equity", A["nav"])
        if A.get("no_trades"):
            blank.add(name)
        if A.get("return_basis") == "monthly":
            monthly.add(name)
    for k, v in analyses[0]["benchmarks"].items():
        series[k] = v
    if has_flows:
        balances.update(analyses[0].get("benchmarks_with_flows") or {})
    start = max(series[k].index[0] for k in runs)
    end = min(series[k].index[-1] for k in runs)
    fb = next((A.get("first_bar") for A in analyses if A.get("first_bar") is not None and A["result"].equity.index[0] == start), None)
    out = {"start": metrics.display_date(start, fb).date(), "end": end.date(), "columns": {}, "initial": base,
           "has_flows": has_flows}
    clim = (pd.Timestamp(fb) if fb is not None else start) + pd.Timedelta(days=LATE_DAYS)
    late = [k for k in series if k not in runs and len(series[k]) and series[k].index[0] > clim]
    out["columns"] = _window_stats({k: v for k, v in series.items() if k not in late}, start, end, fb, rf, blank, monthly,
                                   base, balances)
    out["columns_from"] = {}   # kept for older readers: nothing in "columns" starts later any more
    own = _window_stats({k: series[k] for k in late}, start, end, fb, rf, blank, base=base, balances=balances)
    out["benchmarks_own"] = own
    out["benchmarks_own_from"] = {k: str(metrics.display_date(series[k].index[0], fb).date()) for k in own}
    out["benchmarks_own_to"] = {k: str(min(series[k].index[-1], end).date()) for k in own}
    # full period: from the earliest run's start to the latest end
    fstart = min(A["nav"].index[0] for A in analyses)
    fend = max(A["nav"].index[-1] for A in analyses)
    ffb = next((A.get("first_bar") for A in analyses if A["nav"].index[0] == fstart), None)
    out["full"] = _window_stats(series, fstart, fend, ffb, rf, blank, monthly, base, balances)
    out["full_start"] = metrics.display_date(fstart, ffb).date()
    out["full_end"] = fend.date()
    lim = (pd.Timestamp(ffb) if ffb is not None else fstart) + pd.Timedelta(days=LATE_DAYS)
    out["full_from"] = {k: str(metrics.display_date(s.index[0], ffb).date()) for k, s in series.items()
                        if len(s) and s.index[0] > lim}
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


def interpretation(s) -> str:
    """The spec's summary(), completed where it leaves something out: a portfolio's expense ratio is a cost
    (Portfolio.summary() only lists trading costs)."""
    text = s.summary()
    er = getattr(s, "expense_ratio", 0) or 0
    if er and "expense" not in text:
        fee = f"{er:.2%}/yr expense ratio on invested assets"
        lines = text.splitlines()
        for j, ln in enumerate(lines):
            if ln.startswith("Costs:"):
                lines[j] = "Costs: " + fee if ln.strip() == "Costs: none" else ln + ", " + fee
                break
        else:
            lines.append("Costs: " + fee)
        text = "\n".join(lines)
    return text


def headline(A: dict) -> dict:
    """The headline numbers and the one rule behind them, shared by the report tiles, the console and
    summary.json: every headline figure (final value, total return, CAGR, Sharpe, drawdown) is measured from the
    first day of the statistics, and the start value is the account value on that day. An allocation portfolio
    starts trading after its warm-up, so it starts with the starting capital; a signal strategy's statistics start
    after its rules' warm-up with the cash (plus interest) it held then."""
    st = A["stats"]
    res = A.get("result_full") or A["result"]
    cap = float(res.equity.iloc[0]) if len(res.equity) else None
    warm = A.get("warmup_start")
    out = {"stats_start": st.get("start"), "stats_end": st.get("end"), "start_value": st.get("start_equity"),
           "final_value": st.get("end_equity"), "capital": cap, "warmup": warm is not None}
    if warm is not None:
        out["rule"] = (f"Stats from {st.get('start')} (value ${st.get('start_equity'):,.2f}) after the warm-up; the "
                       f"simulation starts {res.equity.index[1].date() if len(res.equity) > 1 else ''} with "
                       f"${cap:,.2f}. Final value, returns and CAGR are all measured from the stats start.")
    else:
        out["rule"] = (f"Stats from {st.get('start')} (value ${st.get('start_equity'):,.2f}): final value, returns and "
                       "CAGR are all measured from the start.")
    return out


def short_label(name) -> str:
    """A benchmark or blend name shortened without losing what it is: "60% SPYSIM / 40% IEFSIM buy & hold" ->
    "60/40 SPYSIM/IEFSIM" (weights, then tickers); " buy & hold" dropped."""
    s = str(name).replace(" buy & hold", "").strip()
    pairs = re.findall(r"(\d+(?:\.\d+)?)% ([\w^$.\-]+)", s)
    if len(pairs) >= 2 and re.fullmatch(r"(?:\s*(?:/|,|\+|and)?\s*\d+(?:\.\d+)?% [\w^$.\-]+)+(?:\s*(?:blend|mix|portfolio))?\s*", s):
        return "/".join(w for w, _ in pairs) + " " + "/".join(t for _, t in pairs)
    return s


def _fit(s, w: int, right: bool = False) -> str:
    """Text cut to width w (ending in '~' when cut) and padded, so console table columns stay aligned."""
    s = str(s)
    if len(s) > w:
        s = s[: max(w - 1, 0)] + "~"
    return s.rjust(w) if right else s.ljust(w)


def console_summary(A: dict) -> str:
    s, st, ts, ex = A["strategy"], A["stats"], A["trade_stats"], A["exposure"]
    L = ["=" * 78, s.description or s.name or "(strategy)", "-" * 78, interpretation(s)]
    for n in s.notes:
        if not n.startswith("Warning:"):   # those are listed with the warnings below
            L.append(f"Note: {n}")
    for w in A.get("warnings") or []:
        tag = {"error": "WARNING", "warn": "Warning", "info": "Note"}.get(w["level"], "Warning")
        L.append(f"{tag}: {w['message']}. {w['detail']}")
    L.append("-" * 78)
    L.append(f"Period            {st['start']} -> {st['end']}  ({st['years']:.1f} years)"
             + ("  (returns up to the day the money ran out)" if st.get("depleted") else ""))
    hl = headline(A)
    if hl.get("warmup"):
        L.append(f"                  {hl['rule']}")
    sv = next((w for w in A.get("warnings") or [] if w.get("code") == "survivorship"), None)
    if sv:
        L.append(f"!! {sv['message']}")
    if A.get("cash"):
        c = A["cash"]
        L.append(f"Money             start ${c['starting_balance']:,.0f} + contributions ${c['total_contributions']:,.0f} "
                 f"- withdrawals ${c['total_withdrawals']:,.0f} -> ${c['ending_balance']:,.0f}")
        L.append(f"                  money-weighted return {pct(c['money_weighted_return'])}/yr"
                 + (f"   (money ran out: portfolio depleted on {c['depleted_on']})" if c.get("depleted_on") else
                    "   (money ran out)" if c["ran_out"] else ""))
    L.append(f"Start / end       ${st['start_equity']:,.2f} -> ${st['end_equity']:,.2f}")
    if A.get("no_trades"):
        L.append("Result            No trades: the entry rule never triggered.")
        L.append(f"                  (cash interest alone: total {pct(st['total_return'])}, CAGR {pct(st['cagr'])}; not a trading result)")
    else:
        L.append(f"Total return      {pct(st['total_return'])}   CAGR {pct(st['cagr'])}   real (after inflation) {pct(st['real_cagr'])}")
    L.append(f"Sharpe            {num(st['sharpe'])}   Sortino {num(st['sortino'])}   Calmar {num(st['calmar'])}   "
             f"(excess over {'T-bills' if A['rf'] == 'tbill' else pct(float(A['rf'] or 0))})")
    if st.get("max_dd_peak") is None:
        L.append("Max drawdown      none (the account never fell below a previous high)")
    else:
        L.append(f"Max drawdown      {pct(st['max_drawdown'])}  peak {st['max_dd_peak']}  trough {st['max_dd_trough']}  recovered {st['max_dd_recovery'] or 'not yet'}"
                 + (f"   (month-end {pct(st['max_drawdown_monthly'], 1)})" if st.get("max_drawdown_monthly") is not None
                    and np.isfinite(st["max_drawdown_monthly"]) else ""))
    L.append(f"Volatility        {pct(st['volatility'])}   Time in market {pct(ex['time_in_market'])}   Avg exposure {pct(ex['avg_exposure'])}")
    if A["result"].kind == "allocation":
        n_hp = int(ts.get("trades") or 0) + int(ts.get("open_trades") or 0)
        if n_hp:
            L.append(f"Holding periods   {n_hp}   ({ts.get('open_trades') or 0} still held; a ticker held from first "
                     "purchase until sold out)")
    elif ts.get("trades"):
        L.append(f"Trades            {ts['trades']} closed  ({num(ts['trades_per_year'], 1)}/yr)   Win rate {pct(ts['win_rate'], 1)}   Profit factor {num(ts['profit_factor'])}")
        L.append(f"Avg trade         {pct(ts['avg_return'])}   Avg win {pct(ts['avg_win'])}   Avg loss {pct(ts['avg_loss'])}   t-stat {num(ts['t_stat'])}")
    else:
        L.append("Trades            0 closed")
    if A["result"].kind == "signal" and ts.get("open_trades"):
        L.append(f"Open P&L          ${ts['open_pnl']:,.2f} on {ts['open_trades']} position(s) still open at the end "
                 "(marked at the last close; not in the trade statistics)")
    if A.get("turnover") is not None:
        L.append(f"Turnover          {pct(A['turnover'] / 2, 0)} per year (one-sided)   Rebalances {A['rebalances']}")
    if A["relative"]:
        r = A["relative"]
        L.append(f"vs {short_label(A['primary_benchmark'])} beta {num(r['beta'])}   alpha {pct(r['alpha_annual'])}/yr   "
                 f"correlation {num(r['correlation'])}   up/down capture {pct(r['up_capture'], 0)}/{pct(r['down_capture'], 0)}")
    L.append("-" * 78)
    nw = max([24] + [min(44, len(short_label(n))) for n in A["benchmarks"]])
    L.append(f"{'':{nw}s} {'Final $':>14s} {'CAGR':>8s} {'Sharpe':>7s} {'MaxDD':>8s} {'MaxDD me':>8s}  (each from its first date; "
             "MaxDD me = from month-end values)")
    L.append(f"{'Strategy':{nw}s} {st['end_equity']:>14,.0f} {pct(st['cagr']):>8s} {num(st['sharpe']):>7s} {pct(st['max_drawdown'], 1):>8s} "
             f"{pct(st.get('max_drawdown_monthly'), 1):>8s}")
    bwf = A.get("benchmarks_with_flows") or {}
    late = A.get("benchmark_from") or {}
    for n, b in A["benchmarks"].items():
        bs = metrics.equity_stats(b, A["rf"], first_bar=A.get("first_bar"))
        final = float(bwf[n].iloc[-1]) if n in bwf else bs["end_equity"]
        L.append(f"{_fit(short_label(n), nw)} {final:>14,.0f} {pct(bs['cagr']):>8s} {num(bs['sharpe']):>7s} {pct(bs['max_drawdown'], 1):>8s} "
                 f"{pct(bs.get('max_drawdown_monthly'), 1):>8s}  (from {bs['start']})")
    if bwf:
        L.append("(benchmark final values include the same contributions and withdrawals as the strategy"
                 + ("; one that starts later starts with the strategy's balance on its first day" if late else "")
                 + (f"; all values on {A['depleted']}, the day the money ran out" if A.get("depleted") else "") + ")")
    elif late:
        L.append("(a benchmark that starts later is bought at the strategy's growth index on its first day)")
    tr_ = A.get("trailing") or {}
    if A["result"].kind == "allocation" and tr_.get("Strategy"):
        L.append("-" * 78)
        keys = [k for k, *_ in metrics.TRAILING] + ["Full"]
        L.append(f"Trailing returns as of {tr_['Strategy'].get('as_of')} (3Y and longer annualised)")
        tw = max([24] + [min(44, len(short_label(n))) for n in tr_])
        L.append(f"{'':{tw}s} " + " ".join(f"{k:>7s}" for k in keys))
        for n, t in tr_.items():
            if t:
                L.append(f"{_fit(short_label(n), tw)} " + " ".join(f"{pct(t.get(k), 1):>7s}" for k in keys))
    wr = A.get("withdrawal_rates") or {}
    if wr:
        base = (f"of the balance when withdrawals start ({wr['base_date']}, ${wr['base_balance']:,.0f}), "
                if wr.get("base") == "withdrawal_start" and wr.get("base_balance") is not None else "of the starting balance, ")
        L.append(f"Historical withdrawal rates over the full period {wr['from']} -> {wr['to']}: safe {pct(wr.get('swr'), 2)}   "
                 f"perpetual {pct(wr.get('pwr'), 2)}")
        L.append(f"  ({base}inflation-adjusted, {wr.get('freq', 'yearly')} withdrawals; from the portfolio's returns "
                 f"without the cash flows, so they do not depend on the amount entered); "
                 f"95% bootstrap safe rate {pct(wr.get('swr_mc95'), 2)}")
        if wr.get("percentiles"):
            wp = wr["percentiles"]
            L.append("  bootstrapped safe / perpetual by percentile  " + "  ".join(
                f"{p}th {pct(wp['safe'][p], 1)}/{pct(wp['perpetual'][p], 1)}" for p in wp["safe"]))
    rc = A.get("risk_contributions") or {}
    if rc.get("rows"):
        L.append("Risk contribution (share of volatility, avg weights x daily covariance)  " + "   ".join(
            f"{x['ticker']} {pct(x['share'], 0)}" for x in rc["rows"][:8])
            + (f"; of the max drawdown  " + "   ".join(f"{x['ticker']} {pct(x.get('drawdown_share'), 0)}" for x in rc["rows"][:8])
               if rc.get("drawdown") else ""))
    at = A.get("attribution")
    if at is not None and len(at):
        L.append(("P&L by holding  " if A["result"].kind == "allocation" else "P&L by ticker   ") + "   ".join(f"{r.ticker} ${r.pnl:,.0f}" for r in at.head(8).itertuples())
                 + f"   interest ${round(A['interest']) + 0:,.0f}" + (f"   fees -${A['fees']:,.0f}" if A.get("fees") else ""))
    L.append("-" * 78)
    L.append("Returns by year")
    y = A["yearly"]
    cols = [c for c in y.columns if is_bench_col(c)]
    alloc = A["result"].kind == "allocation"
    # each benchmark column is as wide as its name (8 to 24 characters; blends as "60/40 A/B"; longer names are cut)
    labels = [short_label(c.replace(" blend", "")) for c in cols]
    widths = [min(max(8, len(x)), 24) for x in labels]
    head = " ".join(_fit(x, w, right=True) for x, w in zip(labels, widths))
    if alloc:
        L.append(f"{'Year':8s} {'Return':>8s} {'Real':>7s} {'Infl.':>6s} {'Start $':>13s} {'Added $':>11s} {'Withdrawn $':>11s} {'End $':>13s} "
                 + head)
    else:
        L.append(f"{'Year':8s} {'Strategy':>9s} {'MaxDD':>8s} {'Trades':>6s} " + head)
    bpart = A.get("benchmark_partial") or {}
    for yr, row in y.iterrows():
        lab = f"{yr}{'*' if row.get('partial') else ''}"   # the footnote names the dates
        # a benchmark's own partial first year (it starts later in that year) is starred too
        tail = " ".join(_fit((pct(row[c], 1) + ("*" if (bpart.get(c) or {}).get("year") == int(yr) else ""))
                             if pd.notna(row[c]) else '', w, right=True) for c, w in zip(cols, widths))
        if alloc:
            L.append(f"{lab:8s} {pct(row['return'], 1):>8s} {pct(row.get('real_return'), 1):>7s} {pct(row.get('inflation'), 1):>6s} "
                     f"{row.get('start_balance', np.nan):>13,.0f} {row.get('contributions', 0):>11,.0f} {row.get('withdrawals', 0):>11,.0f} "
                     f"{row.get('end_balance', np.nan):>13,.0f} " + tail)
        else:
            L.append(f"{lab:8s} {pct(row['return'], 1):>9s} {pct(row['max_drawdown'], 1):>8s} {int(row['trades']):>6d} " + tail)
    years_shown = {int(x) for x in y.index}
    bparts = [f"{short_label(c.replace(' blend', ''))} {_partial_text(v['year'], {'from': v['from']})}"
              for c, v in bpart.items() if c in cols and v["year"] in years_shown]
    if y["partial"].any() or bparts:
        own = ", ".join(_partial_text(yr, row) for yr, row in y[y["partial"].astype(bool)].iterrows())
        L.append("* partial year: " + "; ".join(([own] if own else [])
                                                + ([", ".join(bparts) + ", each benchmark from its first day of data"] if bparts else [])))
    if alloc and "inflation" in y:
        L.append("Infl. = CPI inflation over the calendar year (December to December; a partial year to the latest month "
                 "published). Real = the return after that inflation.")
    if A.get("depleted"):
        L.append(f"Portfolio depleted on {A['depleted']}: the report stops there (the balance was $0 after it).")
    if A.get("return_basis") == "monthly":
        L.append(f"Monthly steps: {data.stepped_text({t: [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in r] for t, r in A['stepped'].items()})}"
                 " - volatility, Sharpe, Sortino, skew and kurtosis from monthly returns; daily figures blank.")
    tr = A["result"].trades
    if tr is not None and not tr.empty:
        L.append("-" * 78)
        n = len(tr)
        show = tr if n <= 20 else pd.concat([tr.head(10), tr.tail(10)])
        L.append(f"Trades ({n} total{', first and last 10 shown' if n > 20 else ''}; full list in trades.csv / report.html)")
        for i, t in show.iterrows():
            L.append(f"{i:>5d} {_fit(t['ticker'], 6)} {_fit(t['side'], 5)} {t['entry_date']} -> {t['exit_date']} {pct(t['return']):>8s}  ${t['pnl']:>11,.2f}  {t['exit_reason']}")
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
    recs = tr.reset_index().rename(columns={"index": "trade"}).to_dict("records")
    paths = (res.extras or {}).get("level_paths") or {}
    if paths:
        for r in recs:      # the stop / target bar by bar where they move (dynamic levels, TradingView's trailing stops)
            lv = paths.get(f"{r.get('ticker')}|{r.get('entry_date')}")
            if lv:
                r["lv"] = lv
    return recs


def _partial_text(year, row) -> str:
    """'2010 (from Mar 3)', '2026 (to Sep 25)' or '2020 (from Mar 3, to Sep 25)' for a partial year of yearly_detail."""
    bits = []
    f, t = row.get("from"), row.get("to")
    if f is not None and not pd.isna(f) and (pd.Timestamp(f).month > 1 or pd.Timestamp(f).day > 7):
        bits.append(f"from {pd.Timestamp(f).strftime('%b')} {pd.Timestamp(f).day}")
    if t is not None and not pd.isna(t) and (pd.Timestamp(t).month < 12 or pd.Timestamp(t).day < 24):
        bits.append(f"to {pd.Timestamp(t).strftime('%b')} {pd.Timestamp(t).day}")
    return f"{year}" + (f" ({', '.join(bits)})" if bits else "")


def run_payload(A: dict, i: int, idx: pd.DatetimeIndex) -> dict:
    res = A["result"]
    s = A["strategy"]
    p = {
        "name": _run_name(res, i),
        "kind": res.kind,
        "title": s.description or s.summary().splitlines()[0],
        "interpretation": interpretation(s).splitlines(),
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
        "prices": {},        # filled by build_payload (embedded charts) / chart files
        "universe_size": len(getattr(s, "universe", []) or []),
        "universe": list(getattr(s, "universe", []) or [])[:500],
        # account value in dollars of the first day (CPI-deflated), for a real/nominal toggle
        "equity_real": _ser(A["equity_real"], idx, 2) if A.get("equity_real") is not None else None,
        # per-ticker P&L (allocation runs): sum(pnl) + interest - fees == end - start - net flows
        "attribution": _attribution_payload(A),
        "withdrawal_rates": A.get("withdrawal_rates") or {},
        "risk_contributions": A.get("risk_contributions") or {},
        "benchmark_cash": A.get("benchmark_cash") or {},
        "first_bar": A.get("first_bar"),
        "warmup_start": A.get("warmup_start"),
        "headline": headline(A),
        "benchmark_from": A.get("benchmark_from") or {},
        "no_trades": bool(A.get("no_trades")),
        "trailing": A.get("trailing") or {},
        "asset_stats": A.get("asset_stats") or [],
        "fill": getattr(s, "fill", None) or getattr(s, "entry_fill", None),
        "depleted": A.get("depleted"),
        "return_basis": A.get("return_basis") or "daily",
        "stepped": A.get("stepped") or {},
        "income_yearly": _frame_records(A.get("income_yearly")),
        "asset_yearly": ({"tickers": [str(c) for c in A["asset_yearly"].columns],
                          "rows": [{"year": int(y), "values": [None if pd.isna(v) else float(v) for v in row]}
                                   for y, row in A["asset_yearly"].iterrows()]}
                         if A.get("asset_yearly") is not None and len(A["asset_yearly"]) else {}),
    }
    return p


def _frame_records(df) -> list:
    """A year-indexed frame as [{"year": ..., column: value}] (NaN -> None)."""
    if df is None or not len(df):
        return []
    return [{"year": int(y), **{k: (None if pd.isna(v) else float(v)) for k, v in row.items()}} for y, row in df.iterrows()]


def _attribution_payload(A: dict) -> dict:
    at = A.get("attribution")
    if at is None or not len(at):
        return {}
    res = A["result"]
    res = A.get("result_full") or res  # attribution covers the whole simulation (incl. any warm-up)
    fl = res.extras.get("flows")
    net_flows = float(fl.sum()) if fl is not None else 0.0
    return {"rows": at.to_dict("records"), "interest": A["interest"], "fees": A.get("fees") or 0.0,
            "kind": res.kind, "total_pnl": float(at["pnl"].sum()), "net_flows": net_flows,
            "start_equity": float(res.equity.iloc[0]), "end_equity": float(res.equity.iloc[-1]),
            "check": float(res.equity.iloc[-1] - res.equity.iloc[0] - net_flows
                           - (at["pnl"].sum() + A["interest"] - (A.get("fees") or 0.0)))}


def build_payload(analyses: list[dict], out_dir: Path | None = None) -> dict:
    """The report's data. With `out_dir`, price charts of up to MAX_EMBED_TICKERS tickers per run are embedded
    and every other traded ticker is written to out_dir/charts/ for loading on demand; without it, only the
    embedded ones (within MAX_PRICE_TICKERS and the size budget) are charted."""
    idx = analyses[0]["result"].equity.index
    for A in analyses[1:]:
        idx = idx.union(A["result"].equity.index)
    benches = analyses[0]["benchmarks"]
    first = analyses[0]
    title = first["strategy"].description or first["strategy"].summary().splitlines()[0]
    if len(analyses) > 1:
        title = "Comparison: " + " vs ".join(_run_name(A["result"], i) for i, A in enumerate(analyses))
    runs = [run_payload(A, i, idx) for i, A in enumerate(analyses)]
    if out_dir is not None:
        cd = out_dir / "charts"
        if cd.is_dir():
            for f in cd.glob("*.js"):
                f.unlink()
    for i, (A, rp) in enumerate(zip(analyses, runs)):
        full = A.get("result_full") or A["result"]
        if out_dir is None:
            rp["prices"] = price_payload(full)
            rp["chart_tickers"], rp["chart_files"] = list(rp["prices"]), {}
            continue
        rp["prices"] = price_payload(full, max_tickers=MAX_EMBED_TICKERS)
        rp["chart_tickers"] = chart_tickers(full)
        rp["chart_files"] = write_chart_files(full, out_dir, set(rp["prices"]), i if len(analyses) > 1 else None)
    return {
        "title": title,
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "runs": runs,
        # values: growth of the starting capital; with_flows: the same benchmark receiving the first
        # run's contributions/withdrawals (compare with the runs' "equity" account values)
        "benchmarks": [{"name": n, "values": _ser(b, idx, 2),
                        **({"with_flows": _ser(first["benchmarks_with_flows"][n], idx, 2)}
                           if n in (first.get("benchmarks_with_flows") or {}) else {})} for n, b in benches.items()],
        "benchmark_yearly": {n: {int(y): float(v) for y, v in metrics.yearly_returns({n: b})[n].items()} for n, b in benches.items()},
        "common": common_window_stats(analyses, analyses[0]["rf"]),
        "benchmark_from": first.get("benchmark_from") or {},
        "benchmark_partial": first.get("benchmark_partial") or {},
        # CPI relative to the first date (divide a dollar series by it for dollars of the start date)
        "deflator": _deflator(idx),
        "rf": analyses[0]["rf"],
        "data": data.data_status(),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ------------------------------------------------------------------ files

def export_frame(eq: pd.DataFrame, A: dict) -> pd.DataFrame:
    """The daily table as exported: without the synthetic starting point one calendar day before the first
    bar (often a weekend), which only carries the starting capital for the return calculations. A run
    trimmed for an indicator warm-up starts on a real bar (the last warm-up day) and keeps it, and so does a
    portfolio bought at the close of the session before its first day (day 0: the starting balance)."""
    res, full = A["result"], A.get("result_full") or A["result"]
    idx = res.equity.index
    # both simulators prepend that point to the full result; a warm-up trim starts on a real bar instead
    if len(idx) > 1 and idx[0] == full.equity.index[0] and not full.extras.get("day0"):
        return eq[eq.index != idx[0]]
    return eq


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
        export_frame(eq, A).to_csv(out_dir / f"{pre}equity.csv", index_label="date")
        if A.get("attribution") is not None and len(A["attribution"]):
            A["attribution"].to_csv(out_dir / f"{pre}attribution.csv", index=False)
        if res.holdings is not None and not res.holdings.empty:
            res.holdings.to_csv(out_dir / f"{pre}holdings.csv", index_label="date")
        A["yearly"].to_csv(out_dir / f"{pre}yearly.csv")
        A["monthly"].to_csv(out_dir / f"{pre}monthly.csv")
        (out_dir / f"{pre}strategy.json").write_text(A["strategy"].to_json())
        summary = {k: A.get(k) for k in ("stats", "cash", "trade_stats", "exposure", "relative", "monte_carlo",
                                         "sensitivity", "rolling_summary", "crises", "factors")}
        summary.update({"trailing": A.get("trailing") or {}, "asset_stats": A.get("asset_stats") or [],
                        "risk_contributions": A.get("risk_contributions") or {},
                        "no_trades": bool(A.get("no_trades")),
                        "description": A["strategy"].description, "interpretation": interpretation(A["strategy"]),
                        "open_pnl": A["trade_stats"].get("open_pnl", 0.0), "open_trades": A["trade_stats"].get("open_trades", 0),
                        "warmup_start": A.get("warmup_start"), "benchmark_from": A.get("benchmark_from") or {},
                        "headline": headline(A),
                        "notes": A["strategy"].notes, "kind": res.kind, "warnings": A.get("warnings", []),
                        "drawdowns": A["drawdowns"].to_dict("records"),
                        "withdrawal_rates": A.get("withdrawal_rates") or {},
                        "benchmark_cash": A.get("benchmark_cash") or {},
                        "attribution": _attribution_payload(A)})
        (out_dir / f"{pre}summary.json").write_text(json.dumps(_clean(summary), indent=2))
        if excel:
            _excel(A, out_dir / f"{pre}report.xlsx")
    payload = build_payload(analyses, out_dir)
    blob = json.dumps(_clean(payload), separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("__TITLE__", html.escape(payload["title"][:80])).replace("__DATA__", blob)
    path = out_dir / "report.html"
    path.write_text(page)
    if pdf:
        to_pdf(path, out_dir / "report.pdf")
    return path


def _xml_escape(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class _FastBook:
    """A minimal .xlsx writer with DataFrame.to_excel's layout (bold header row, the index as the first column,
    dates as Excel dates). It streams the sheet XML directly: pandas' openpyxl writer builds a Python object per
    cell and takes seconds on the tens of thousands of orders of a daily rebalance."""

    _EPOCH = pd.Timestamp("1899-12-30")

    def __init__(self):
        self.sheets: list[tuple[str, str]] = []

    @staticmethod
    def _col(j: int) -> str:
        out = ""
        j += 1
        while j:
            j, r = divmod(j - 1, 26)
            out = chr(65 + r) + out
        return out

    _CTRL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")   # control characters XML 1.0 does not allow

    @classmethod
    def _cell(cls, ref: str, x, bold: bool = False) -> str:
        """One <c> element: numbers as numbers, dates as serial days (style 2, or 3 with a time), text inline."""
        st = ' s="1"' if bold else ""
        if x is None or x is pd.NaT:
            return f'<c r="{ref}"{st}/>' if bold else ""
        if isinstance(x, (bool, np.bool_)):
            return f'<c r="{ref}" t="b"{st}><v>{int(bool(x))}</v></c>'
        if isinstance(x, (int, np.integer)):
            return f'<c r="{ref}"{st}><v>{int(x)}</v></c>'
        if isinstance(x, (float, np.floating)):
            if not np.isfinite(x):
                return ""
            return f'<c r="{ref}"{st}><v>{repr(float(x))}</v></c>'
        if isinstance(x, (np.datetime64, date)):
            ts = pd.Timestamp(x)
            if ts is pd.NaT:
                return ""
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            days = (ts - cls._EPOCH) / pd.Timedelta(days=1)
            style = "3" if (ts.hour or ts.minute or ts.second) else "2"
            return f'<c r="{ref}" s="{style}"><v>{repr(float(days))}</v></c>'
        text = cls._CTRL.sub("", _xml_escape(str(x)).replace("\r", "&#13;"))
        return f'<c r="{ref}" t="inlineStr"{st}><is><t xml:space="preserve">{text}</t></is></c>'

    @classmethod
    def _column(cls, ref: str, col: pd.Series, first: int) -> list[str]:
        """The <c> elements of one column (rows first, first + 1, ...), with fast paths for numbers and dates."""
        rows = range(first, first + len(col))
        kind = col.dtype.kind
        if kind == "f":
            return ["" if v != v or v in (np.inf, -np.inf) else f'<c r="{ref}{n}"><v>{v!r}</v></c>'
                    for n, v in zip(rows, col.to_numpy(dtype=float).tolist())]
        if kind in "iu":
            return [f'<c r="{ref}{n}"><v>{v}</v></c>' for n, v in zip(rows, col.to_numpy().tolist())]
        if kind == "M":
            ix = pd.DatetimeIndex(col)
            if ix.tz is not None:
                ix = ix.tz_localize(None)
            days = ((ix - cls._EPOCH) / pd.Timedelta(days=1)).to_numpy(dtype=float).tolist()
            timed = (ix != ix.normalize()).tolist()
            return ["" if d != d else f'<c r="{ref}{n}" s="{3 if t else 2}"><v>{d!r}</v></c>'
                    for n, d, t in zip(rows, days, timed)]
        cell = cls._cell
        return [cell(f"{ref}{n}", x) for n, x in zip(rows, col.to_numpy(dtype=object).tolist())]

    def put(self, df: pd.DataFrame, sheet_name: str, index: bool = True, index_label: str | None = None) -> None:
        cols = list(df.columns)
        names = ([index_label if index_label is not None else df.index.name] if index else []) + cols
        refs = [self._col(j) for j in range(len(names))]
        head = "".join(self._cell(f"{refs[j]}1", None if v is None else str(v), bold=True) for j, v in enumerate(names))
        columns = [self._column(refs[j + (1 if index else 0)], df.iloc[:, j], 2) for j in range(len(cols))]
        if index:
            columns.insert(0, [self._cell(f"A{r + 2}", k, bold=True) for r, k in enumerate(df.index)])
        rows = [f'<row r="1">{head}</row>']
        rows += [f'<row r="{r + 2}">' + "".join(cells) + "</row>" for r, cells in enumerate(zip(*columns))]
        xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
               + "".join(rows) + "</sheetData></worksheet>")
        title = "".join(ch for ch in sheet_name if ch not in '[]:*?/\\')[:31]
        self.sheets.append((title, xml))

    def save(self, path) -> None:
        import zipfile
        from xml.sax.saxutils import quoteattr
        n = len(self.sheets)
        ns = "http://schemas.openxmlformats.org"
        ctypes = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                  f'<Types xmlns="{ns}/package/2006/content-types">'
                  '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                  '<Default Extension="xml" ContentType="application/xml"/>'
                  '<Override PartName="/xl/workbook.xml" '
                  'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                  '<Override PartName="/xl/styles.xml" '
                  'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                  + "".join(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" '
                            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                            for i in range(n)) + "</Types>")
        rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<Relationships xmlns="{ns}/package/2006/relationships">'
                f'<Relationship Id="rId1" Type="{ns}/officeDocument/2006/relationships/officeDocument" '
                'Target="xl/workbook.xml"/></Relationships>')
        wb = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              f'<workbook xmlns="{ns}/spreadsheetml/2006/main" xmlns:r="{ns}/officeDocument/2006/relationships"><sheets>'
              + "".join(f'<sheet name={quoteattr(t)} sheetId="{i + 1}" r:id="rId{i + 1}"/>'
                        for i, (t, _) in enumerate(self.sheets)) + "</sheets></workbook>")
        wbrels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                  f'<Relationships xmlns="{ns}/package/2006/relationships">'
                  + "".join(f'<Relationship Id="rId{i + 1}" Type="{ns}/officeDocument/2006/relationships/worksheet" '
                            f'Target="worksheets/sheet{i + 1}.xml"/>' for i in range(n))
                  + f'<Relationship Id="rId{n + 1}" Type="{ns}/officeDocument/2006/relationships/styles" '
                    'Target="styles.xml"/></Relationships>')
        # cell styles: 0 plain, 1 bold (headers and the index), 2 date, 3 date and time
        styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                  f'<styleSheet xmlns="{ns}/spreadsheetml/2006/main">'
                  '<numFmts count="1"><numFmt numFmtId="164" formatCode="yyyy-mm-dd hh:mm:ss"/></numFmts>'
                  '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
                  '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
                  '<fills count="2"><fill><patternFill patternType="none"/></fill>'
                  '<fill><patternFill patternType="gray125"/></fill></fills>'
                  '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
                  '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                  '<cellXfs count="4"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
                  '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
                  '<xf numFmtId="14" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
                  '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>'
                  '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
                  '</styleSheet>')
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", ctypes)
            z.writestr("_rels/.rels", rels)
            z.writestr("xl/workbook.xml", wb)
            z.writestr("xl/_rels/workbook.xml.rels", wbrels)
            z.writestr("xl/styles.xml", styles)
            for i, (_, xml) in enumerate(self.sheets):
                z.writestr(f"xl/worksheets/sheet{i + 1}.xml", xml)


def _excel(A: dict, path: Path) -> None:
    res = A["result"]
    xw = _FastBook()
    rows = [("Strategy", A["strategy"].description or ""), ("Interpretation", interpretation(A["strategy"]))]
    rows += [(k, v) for k, v in _clean(A["stats"]).items()]
    rows += [(f"cash: {k}", v) for k, v in _clean(A["cash"]).items()]
    rows += [(f"trades: {k}", v) for k, v in _clean(A["trade_stats"]).items()]
    rows += [(f"vs benchmark: {k}", v) for k, v in _clean(A["relative"]).items()]
    xw.put(pd.DataFrame(rows, columns=["metric", "value"]), sheet_name="Summary", index=False)
    xw.put(A["yearly"], sheet_name="Yearly")
    if A.get("income_yearly") is not None and len(A["income_yearly"]):
        xw.put(A["income_yearly"], sheet_name="Income")
    if A.get("asset_yearly") is not None and len(A["asset_yearly"]):
        xw.put(A["asset_yearly"], sheet_name="Asset returns by year")
    xw.put(A["monthly"], sheet_name="Monthly")
    xw.put(A["drawdowns"], sheet_name="Drawdowns", index=False)
    if res.trades is not None and not res.trades.empty:
        xw.put(res.trades, sheet_name="Trades", index_label="trade")
    if res.orders is not None and not res.orders.empty:
        xw.put(res.orders, sheet_name="Orders", index=False)
    eq = pd.DataFrame({"equity": res.equity, "twr_index": A["nav"], "drawdown": metrics.drawdown(A["nav"])})
    for n, b in A["benchmarks"].items():
        eq[n] = b
    for n, b in (A.get("benchmarks_with_flows") or {}).items():
        eq[n + " (with cash flows)"] = b
    eq = export_frame(eq, A)
    eq.index = eq.index.tz_localize(None)
    xw.put(eq, sheet_name="Equity", index_label="date")
    if A.get("attribution") is not None and len(A["attribution"]):
        xw.put(A["attribution"], sheet_name="Attribution", index=False)
    if res.holdings is not None and not res.holdings.empty:
        xw.put(res.holdings.resample("ME").last(), sheet_name="Holdings (month-end)", index_label="date")
    if A.get("risk_contributions"):
        from . import risk
        xw.put(risk.frame(A["risk_contributions"]), sheet_name="Risk contributions", index=False)
    xw.save(path)


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
