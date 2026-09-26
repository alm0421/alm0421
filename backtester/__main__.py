"""Command line.

    python -m backtester "buy MSFT at the close when it is down 5 days in a row, hold 1 day"
    python -m backtester compare "60/40 SPY/TLT rebalanced quarterly" "buy and hold QQQ"
    python -m backtester sweep "buy QQQ when RSI({2..5}) is below {5..20 step 5}, hold {1,3,5} days"
    python -m backtester walkforward "buy QQQ when RSI(2) is below {5..20 step 5}, hold {1..5} days"
    python -m backtester optimize SPY QQQ TLT GLD --max-weight 0.6 --constraint "SPY+QQQ <= 70%" --rolling 12
    python -m backtester montecarlo --weights "SPY 60 TLT 40" --years 30 --withdrawal 40000 --balance 1000000
    python -m backtester factors QQQ --model ff5 --freq monthly      (models: capm ff3 carhart ff5 ff6 bonds ff3+bonds,
                                    <region>_ff3/ff5/carhart/ff6 for developed, dev (ex US), europe, japan,
                                    asia_pacific_ex_japan, north_america, emerging; add-ons ff5+qmj+bab; auto)
    python -m backtester style QQQ                                   (returns-based style analysis)
    python -m backtester correlation SPY TLT GLD EFASIM --window 36 --freq monthly
    python -m backtester signals "buy Nasdaq 100 stocks when RSI(2) is below 5, hold 3 days"
    python -m backtester paper add "..." --name rsi2 ; python -m backtester paper report
    python -m backtester import-composer symphony.json [--out spec.json] [--run]
    python -m backtester web            # the backtesting site on http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import data, expr, parser, report, runner
from .montecarlo import parse_weights

SUBCOMMANDS = {"run", "compare", "sweep", "walkforward", "optimize", "signals", "paper", "web", "tickers", "library", "montecarlo",
               "factors", "style", "import-composer", "correlation", "correlations", "trade"}


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--capital", type=float)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--slippage-bps", type=float)
    p.add_argument("--commission", type=float, help="$ per order")
    p.add_argument("--rf", default="tbill", help="risk-free rate for Sharpe/alpha: 'tbill' (default) or an annual rate like 0.02")


def build_run_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m backtester", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?", help="strategy in plain English")
    p.add_argument("--spec", help="JSON strategy/portfolio file (e.g. a strategy.json written by an earlier run)")
    p.add_argument("--tickers", help="comma-separated tickers, or NDX for the Nasdaq-100 (explicit rule mode)")
    p.add_argument("--entry", help="entry rule in the expression language (explicit mode)")
    p.add_argument("--exit-when", help="exit rule in the expression language")
    p.add_argument("--hold", type=int, help="exit after N bars")
    p.add_argument("--short", action="store_true", help="short instead of long (explicit mode)")
    p.add_argument("--entry-fill", choices=["close", "open", "next_open", "next_close"])
    p.add_argument("--exit-fill", choices=["close", "open"], help="fill for --hold exits")
    p.add_argument("--stop-loss", type=float, help="e.g. 0.05 for 5%%")
    p.add_argument("--take-profit", type=float)
    p.add_argument("--trailing-stop", type=float)
    p.add_argument("--max-positions", type=int)
    p.add_argument("--position-size", type=float, help="fraction of equity per position")
    p.add_argument("--rank-by", help="expression used to prioritise signals when slots are limited (highest first)")
    p.add_argument("--rank-ascending", action="store_true")
    p.add_argument("--whole-shares", action="store_true", help="disallow fractional shares")
    p.add_argument("--benchmark", help="benchmark ticker for alpha/beta (default SPY)")
    p.add_argument("--name", help="label for this run")
    _common(p)
    p.add_argument("--out", help="output folder (default reports/<slug>)")
    p.add_argument("--no-sensitivity", action="store_true", help="skip the transaction-cost re-runs")
    p.add_argument("--pdf", action="store_true", help="also write report.pdf (needs Playwright + Chromium)")
    p.add_argument("--dry-run", action="store_true", help="only show how the text was interpreted")
    p.add_argument("--help-expr", action="store_true", help="show the rule language reference")
    p.add_argument("--json", action="store_true", help="print the summary statistics as JSON")
    return p


def _overrides(a) -> dict:
    ov = dict(capital=a.capital, start=a.start, end=a.end, slippage_bps=a.slippage_bps, commission=a.commission)
    for k in ("max_positions", "position_size", "rank_by", "stop_loss", "take_profit", "trailing_stop",
              "exit_when", "entry_fill", "benchmark", "name"):
        if hasattr(a, k):
            ov[k] = getattr(a, k)
    if getattr(a, "hold", None) is not None:
        ov["hold_bars"] = a.hold
    if getattr(a, "exit_fill", None):
        ov["hold_exit_fill"] = a.exit_fill
    if getattr(a, "rank_ascending", False):
        ov["rank_ascending"] = True
    if getattr(a, "whole_shares", False):
        ov["fractional_shares"] = False
    return ov


def make_spec(text: str | None, a, spec_path: str | None = None):
    ov = _overrides(a)
    if spec_path:
        spec = runner.load(spec_path)
        for k, v in ov.items():
            if v is not None and hasattr(spec, k):
                setattr(spec, k, v)
    elif text:
        spec = parser.parse(text, **ov)
    elif getattr(a, "tickers", None) and getattr(a, "entry", None):
        from .strategy import Strategy
        ndx = a.tickers.upper() == "NDX"
        uni = data.nasdaq100_ever() if ndx else [data.canonical(t) for t in a.tickers.split(",")]
        spec = Strategy(universe=uni, universe_name="NDX" if ndx else None, entry=a.entry,
                        side="short" if a.short else "long", max_positions=a.max_positions or (1 if len(uni) == 1 else 10),
                        description=f"{'short' if a.short else 'long'} {a.tickers} when {a.entry}")
        for k, v in ov.items():
            if v is not None and hasattr(spec, k):
                setattr(spec, k, v)
    else:
        raise parser.ParseError("Give a strategy in plain English, --spec FILE, or --tickers with --entry.")
    spec.validate()
    return spec


def preflight(spec) -> list[str]:
    """The checks a real run makes before simulating, so --dry-run never prints a clean interpretation for a
    spec that will fail: the spec's own validation, the date range (reversed, or without data for the
    tickers), unknown tickers and rule names (every rule evaluated once on real data), and rules whose
    look-back needs more history than the period has. Returns warnings that don't stop the run."""
    import pandas as pd

    from . import portfolio as pf
    from . import web
    warn: list[str] = []
    spec.validate()
    start = pd.Timestamp(spec.start) if spec.start else None
    end = pd.Timestamp(spec.end) if spec.end else None
    if start is not None and end is not None and start >= end:
        raise ValueError(f"The period is reversed or empty: it starts on {start.date()} but ends on {end.date()}.")
    web._probe_rules(spec)
    is_pf = spec.__class__.__name__ == "Portfolio"
    tickers = pf.fixed_tickers(spec.tree) if is_pf else [data.canonical(t) for t in spec.universe]
    frames = {}
    for t in tickers:
        try:
            frames[t] = data.load(t)
        except data.DataError:
            if is_pf or not getattr(spec, "universe_name", None):
                raise
    if not frames and is_pf and not tickers:
        return warn  # only index-universe filters: the universe's own data is checked when it runs
    if not frames:
        raise ValueError("None of the tickers has price data.")

    def in_range(df):
        ix = df.index
        if start is not None:
            ix = ix[ix >= start]
        if end is not None:
            ix = ix[ix <= end]
        return ix
    if is_pf:
        first_common = max(df.index[0] for df in frames.values())
        last_common = min(df.index[-1] for df in frames.values())
        lo = max(first_common, start) if start is not None else first_common
        hi = min(last_common, end) if end is not None else last_common
        if lo >= hi:
            late = max(frames, key=lambda t: frames[t].index[0])
            raise ValueError(f"No price data in the requested period: the holdings trade together from {first_common.date()} "
                             f"({late} starts then) to {last_common.date()}.")
    elif not any(len(in_range(df)) >= 2 for df in frames.values()):
        rng = f"{start.date() if start is not None else 'the start'} to {end.date() if end is not None else 'the end'}"
        raise ValueError(f"No price data in the requested period ({rng}) for {', '.join(list(frames)[:5])}.")
    # look-backs: a rule that has no value before the end of the period can never trigger
    from .report import rule_first_defined
    from .expr import Namespace
    rules = web._rules(spec)
    if is_pf:
        def ons(n, acc):
            if isinstance(n, dict):
                if "if" in n:
                    acc.append((n["if"], data.canonical(n.get("on", "SPY"))))
                for k in pf._kids(n):
                    ons(k, acc)
            return acc
        pairs = ons(spec.tree, [])
    else:
        t0 = min(frames, key=lambda t: frames[t].index[0])
        pairs = [(r, t0) for r in rules if isinstance(r, str)]
    for rule, t in pairs:
        df = frames.get(t) if t in frames else data.load(t)
        ns_ = Namespace(df, ticker=t)
        d = rule_first_defined(rule, ns_)
        stop = end if end is not None else df.index[-1]
        if d is None:
            from .expr import never_defined
            cold = never_defined(rule, ns_)
            if cold:
                seg, n = cold[0]
                need = f"needs {n} bars" if n else "needs more bars than there are"
                raise ValueError(f"Indicator never warmed up: {seg} {need}, and {t}'s data has {len(df)} "
                                 f"({df.index[0].date()} to {df.index[-1].date()}), so the rule {rule!r} can never be "
                                 "true. Shorten the look-back.")
        if d is not None and d > stop:
            raise ValueError(f"The rule {rule!r} needs more history than there is: on {t} its indicators first have a "
                             f"value on {d.date()}, after the end of the period ({stop.date()}). Shorten the look-back "
                             "or extend the period.")
        if d is not None and start is not None and d > start:
            warn.append(f"Warm-up: {rule!r} on {t} has no value until {d.date()}, so nothing can trigger before then.")
    return warn


def _rf(v):
    return "tbill" if v in (None, "tbill") else float(v)


def cmd_run(argv: list[str]) -> int:
    a = build_run_parser().parse_args(argv)
    if a.help_expr:
        print(expr.HELP)
        return 0
    spec = make_spec(a.text, a, a.spec)
    extra = preflight(spec)   # the same checks for a dry run and a real run: fail before printing a clean interpretation
    if a.dry_run:
        print(spec.summary())
        allnotes = spec.notes + extra
        for n in [n for n in allnotes if n.startswith("Warning:")]:   # reinterpreted words: shown first
            print("WARNING:", n[len("Warning:"):].strip())
        for n in [n for n in allnotes if not n.startswith("Warning:")]:
            print("Note:", n)
        print(spec.to_json())
        return 0
    t0 = time.time()
    res = runner.run(spec)
    A = report.analyze(res, rf=_rf(a.rf), sensitivity=not a.no_sensitivity)
    if a.json:
        print(json.dumps(report._clean({"stats": A["stats"], "trade_stats": A["trade_stats"], "relative": A["relative"],
                                        "cash": A["cash"]}), indent=2))
    else:
        print(report.console_summary(A))
    out = Path(a.out) if a.out else report.ROOT / "reports" / report.slug(spec.name or spec.description or "run")
    path = report.write_outputs(A, out, pdf=a.pdf)
    print(f"Report: {path}\nFiles:  {out}/ (trades.csv, equity.csv, yearly.csv, monthly.csv, report.xlsx, summary.json, strategy.json)")
    print(f"({time.time() - t0:.1f}s)")
    return 0


def cmd_compare(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="python -m backtester compare", description="Run several strategies/portfolios and compare them in one report.")
    p.add_argument("texts", nargs="*", help="strategies in plain English (quote each one)")
    p.add_argument("--spec", action="append", default=[], help="JSON spec file (repeatable)")
    p.add_argument("--names", help="comma-separated labels")
    _common(p)
    p.add_argument("--out")
    p.add_argument("--pdf", action="store_true")
    a = p.parse_args(argv)
    specs = [parser.parse(t, **_overrides(a)) for t in a.texts] + [runner.load(s) for s in a.spec]
    if len(specs) < 2:
        raise parser.ParseError("compare needs at least two strategies")
    names = a.names.split(",") if a.names else [chr(65 + i) for i in range(len(specs))]
    analyses = []
    for s, n in zip(specs, names):
        s.name = n.strip() if a.names else f"Strategy {n}"
        res = runner.run(s)
        analyses.append(report.analyze(res, rf=_rf(a.rf), sensitivity=False))
        st = analyses[-1]["stats"]
        print(f"{report._fit(s.name, 14)} CAGR {report.pct(st['cagr']):>8s}  Sharpe {report.num(st['sharpe']):>5s}  MaxDD {report.pct(st['max_drawdown'], 1):>7s}  "
              f"final ${st['end_equity']:,.0f}   {s.description[:70]}")
    C = report.common_window_stats(analyses, _rf(a.rf))
    print(f"\nCommon period {C['start']} -> {C['end']}:")
    for k, st in C["columns"].items():
        print(f"  {report._fit(k, 24)} CAGR {report.pct(st['cagr']):>8s}  Sharpe {report.num(st['sharpe']):>5s}  MaxDD {report.pct(st['max_drawdown'], 1):>7s}")
    out = Path(a.out) if a.out else report.ROOT / "reports" / ("compare-" + report.slug("-".join(n for n in names)))
    path = report.write_outputs(analyses, out, pdf=a.pdf)
    print(f"Report: {path}")
    return 0


def cmd_sweep(argv: list[str], walk: bool = False) -> int:
    from . import research, research_report
    p = argparse.ArgumentParser(prog=f"python -m backtester {'walkforward' if walk else 'sweep'}")
    p.add_argument("text", help="strategy with {placeholders}, e.g. 'hold {1..5} days'")
    p.add_argument("--objective", default="sharpe", choices=list(research.OBJECTIVES))
    if walk:
        p.add_argument("--in-sample", type=float, default=5, help="years")
        p.add_argument("--out-sample", type=float, default=1, help="years")
        p.add_argument("--anchored", action="store_true")
    _common(p)
    p.add_argument("--out")
    a = p.parse_args(argv)
    ov = {k: v for k, v in dict(capital=a.capital, start=a.start, end=a.end, slippage_bps=a.slippage_bps, commission=a.commission).items() if v is not None}
    t0 = time.time()
    if walk:
        ov.pop("start", None)
        ov.pop("end", None)
        R = research.walk_forward(a.text, a.objective, a.in_sample, a.out_sample, a.anchored, a.start, a.end, ov)
        print(research_report.walk_console(R))
        out = Path(a.out) if a.out else report.ROOT / "reports" / ("walkforward-" + report.slug(a.text))
        path = research_report.write_walk(R, a.text, out)
    else:
        R = research.sweep(a.text, a.objective, ov)
        print(research_report.sweep_console(R))
        out = Path(a.out) if a.out else report.ROOT / "reports" / ("sweep-" + report.slug(a.text))
        path = research_report.write_sweep(R, a.text, out)
    print(f"Report: {path}  ({time.time() - t0:.1f}s)")
    return 0


def cmd_optimize(argv: list[str]) -> int:
    from . import research, research_report
    p = argparse.ArgumentParser(prog="python -m backtester optimize", description="Efficient frontier / max-Sharpe / min-variance weights.")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--max-weight", type=float, default=1.0)
    p.add_argument("--min-weight", type=float, default=0.0)
    p.add_argument("--test-start", help="estimate before this date, evaluate after it (out of sample)")
    p.add_argument("--constraint", action="append", default=[],
                   help="repeatable: 'SPY <= 50%%', 'TLT >= 10%%', 'SPY+QQQ <= 70%%', '20%% <= TLT+IEF <= 60%%'")
    p.add_argument("--target-return", type=float, help="annual, e.g. 0.07: minimum volatility with at least this return")
    p.add_argument("--target-vol", type=float, help="annual, e.g. 0.10: maximum return with at most this volatility")
    p.add_argument("--methods", help="comma-separated objectives to run (default all): " + ", ".join(research.OPT_METHODS))
    p.add_argument("--omega-threshold", type=float, default=0.0,
                   help="annual threshold return of the Omega ratio (default 0, e.g. 0.03)")
    p.add_argument("--rolling", type=int, metavar="MONTHS", help="walk-forward: re-optimise every N months")
    p.add_argument("--lookback", type=int, default=60, metavar="MONTHS", help="trailing window for --rolling (default 60)")
    p.add_argument("--rebalance", default="quarterly", choices=["monthly", "quarterly", "yearly"],
                   help="rebalancing used in the suggested sentences")
    p.add_argument("--out")
    a = p.parse_args(argv)
    R = research.optimize(a.tickers, a.start, a.end, a.max_weight, a.min_weight, a.test_start,
                          constraints=a.constraint, target_return=a.target_return, target_vol=a.target_vol,
                          rolling_months=a.rolling, lookback_months=a.lookback, rebalance=a.rebalance,
                          methods=[m.strip() for m in a.methods.split(",") if m.strip()] if a.methods else None,
                          omega_threshold=a.omega_threshold)
    print(research_report.optimize_console(R))
    out = Path(a.out) if a.out else report.ROOT / "reports" / ("optimize-" + report.slug("-".join(a.tickers)))
    path = research_report.write_optimize(R, out)
    print(f"Report: {path}")
    return 0


def _load_target(a):
    """(weights or None, spec or None) from --weights / --spec / --run / a sentence."""
    if getattr(a, "weights", None):
        return parse_weights(a.weights), None
    if getattr(a, "spec", None):
        return None, runner.load(a.spec)
    if getattr(a, "run", None):
        p = report.ROOT / "reports" / "runs" / a.run / "strategy.json"
        if not p.exists():
            raise ValueError(f"No saved run {a.run} (looked for {p}).")
        return None, runner.load(p)
    if getattr(a, "text", None):
        return None, parser.parse(a.text)
    raise ValueError("Give --weights 'SPY 60 TLT 40', a sentence, --spec FILE or --run ID.")


def cmd_montecarlo(argv: list[str]) -> int:
    from . import montecarlo as mc
    p = argparse.ArgumentParser(prog="python -m backtester montecarlo",
                                description="Monte Carlo simulation of a portfolio's future balance (percentile bands, "
                                            "chance of success, safe and perpetual withdrawal rates).")
    p.add_argument("text", nargs="?", help="a portfolio or strategy sentence (its monthly returns are resampled)")
    p.add_argument("--weights", help="tickers and weights, e.g. 'SPY 60 TLT 40' (use SPYSIM/TLTSIM for long history)")
    p.add_argument("--spec", help="JSON spec file")
    p.add_argument("--run", help="id of a saved run from the site (reports/runs/<id>)")
    p.add_argument("--balance", type=float, default=1_000_000)
    p.add_argument("--years", type=int, default=30)
    p.add_argument("--model", default="historical", choices=list(mc.MODELS))
    p.add_argument("--block", type=int, default=12, help="bootstrap block length in months (historical model)")
    p.add_argument("--forecast", action="append", default=[], metavar="TICKER:RET:VOL",
                   help="forecast model: e.g. SPY:0.06:0.16 (annual expected return and volatility)")
    p.add_argument("--contribution", type=float, default=0.0, help="$ added per period")
    p.add_argument("--withdrawal", type=float, default=0.0, help="$ withdrawn per period")
    p.add_argument("--withdrawal-pct", type=float, default=0.0, help="share of the balance withdrawn per year, e.g. 0.04")
    p.add_argument("--freq", default="yearly", choices=["monthly", "quarterly", "yearly"])
    p.add_argument("--nominal", action="store_true", help="do not grow $ flows with inflation")
    p.add_argument("--inflation", default="historical", help="'historical' (bootstrap CPI) or a fixed annual rate like 0.025")
    p.add_argument("--rebalance", default="yearly", choices=["monthly", "quarterly", "yearly", "none"])
    p.add_argument("--sims", type=int, default=5000)
    p.add_argument("--success", type=float, default=0.95, help="success target for the safe withdrawal rate")
    p.add_argument("--start", help="history window start")
    p.add_argument("--end", help="history window end")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--stress", choices=["worst_sequence", "shock"],
                   help="sequence-of-returns stress: start every path with the worst historical --stress-years years, "
                        "or with a first-year --shock")
    p.add_argument("--stress-years", type=int, default=10)
    p.add_argument("--shock", type=float, default=-0.30, help="first-year return for --stress shock (default -0.30)")
    p.add_argument("--age", type=float, help="current age; with --until-age the horizon is the difference")
    p.add_argument("--until-age", type=float, help="e.g. 95: withdraw until this age")
    p.add_argument("--horizon", choices=["fixed", "mortality"], default="fixed",
                   help="mortality: run until the SSA life table says nobody is left and weight success by survival (needs --age)")
    p.add_argument("--sex", choices=["male", "female", "joint"], default="male", help="for --horizon mortality")
    p.add_argument("--age2", type=float, help="--sex joint: the second person's age (default: --age)")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    weights, spec = _load_target(a)
    flows = []
    if a.contribution:
        flows.append(mc.CashFlow(amount=a.contribution, freq=a.freq, inflation_adjusted=not a.nominal))
    if a.withdrawal:
        flows.append(mc.CashFlow(amount=-a.withdrawal, freq=a.freq, inflation_adjusted=not a.nominal))
    if a.withdrawal_pct:
        flows.append(mc.CashFlow(pct=-a.withdrawal_pct, freq=a.freq))
    s = mc.Settings(start_balance=a.balance, years=a.years, model=a.model, block_months=a.block, rebalance=a.rebalance,
                    sims=a.sims, seed=a.seed, success_target=a.success, start=a.start, end=a.end,
                    inflation=a.inflation if a.inflation == "historical" else float(a.inflation),
                    stress=a.stress, stress_years=a.stress_years, stress_shock=a.shock, age=a.age, until_age=a.until_age,
                    horizon=a.horizon, sex=a.sex, age2=a.age2)
    if a.horizon == "mortality":
        if a.age is None:
            raise ValueError("--horizon mortality needs --age.")
    elif (a.age is None) != (a.until_age is None):
        raise ValueError("Give both --age and --until-age (the horizon is the difference).")
    for f in a.forecast:
        t, r, v = f.split(":")
        s.forecast[data.canonical(t)] = (float(r), float(v))
    if weights:
        s.weights = weights
    else:
        s.weights, s.series_name = mc.settings_from_spec(spec, s)
    if not flows and spec is not None and hasattr(spec, "contribution"):
        flows = mc.flows_from_portfolio(spec)
    s.flows = flows
    R = mc.run(s)
    if a.json:
        print(json.dumps(report._clean(R), indent=2))
    else:
        print(mc.console(R))
    return 0


def cmd_factors(argv: list[str]) -> int:
    from . import factors as F
    p = argparse.ArgumentParser(prog="python -m backtester factors",
                                description="Regress a ticker, portfolio or strategy on CAPM / Fama-French / Carhart factors.")
    p.add_argument("text", nargs="?", help="a ticker (QQQ) or a strategy/portfolio sentence")
    p.add_argument("--weights", help="portfolio, e.g. 'SPY 60 TLT 40' (rebalanced monthly)")
    p.add_argument("--spec")
    p.add_argument("--run", help="id of a saved run from the site")
    p.add_argument("--model", default="ff3",
                   help="auto (the ticker's region), a model, or a model plus add-ons such as ff5+qmj+bab or "
                        "europe_ff5+qmj. " + "; ".join(f"{m['key']}: {m['label']}" for m in F.model_list()))
    p.add_argument("--freq", default="monthly", choices=["monthly", "daily"])
    p.add_argument("--rolling", type=int, default=36, help="rolling window in months (default 36)")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.text and not a.weights and " " not in a.text.strip():
        target = a.text.strip()
    else:
        w, spec = _load_target(a)
        target = w if w is not None else spec
    r, name = F.returns_for(target)
    R = F.analyze(r, a.model, a.freq, a.start, a.end, a.rolling, name=name)
    print(json.dumps(report._clean(R), indent=2) if a.json else F.console(R))
    return 0


def cmd_style(argv: list[str]) -> int:
    from . import factors as F
    from . import style as ST
    p = argparse.ArgumentParser(prog="python -m backtester style",
                                description="Returns-based style analysis (Sharpe 1992): the non-negative mix of asset "
                                            "classes, adding up to 100%, that best tracks a fund's monthly returns.")
    p.add_argument("text", nargs="?", help="a ticker (QQQ) or a strategy/portfolio sentence")
    p.add_argument("--weights", help="portfolio, e.g. 'SPY 60 TLT 40' (rebalanced monthly)")
    p.add_argument("--spec")
    p.add_argument("--run", help="id of a saved run from the site")
    p.add_argument("--assets", help="asset-class tickers to use instead of the defaults, e.g. 'SPY EFA AGG BIL'")
    p.add_argument("--window", type=int, default=36, help="rolling window in months (default 36)")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.text and not a.weights and " " not in a.text.strip():
        target = a.text.strip()
    else:
        w, spec = _load_target(a)
        target = w if w is not None else spec
    r, name = F.returns_for(target)
    assets = a.assets.replace(",", " ").split() if a.assets else None
    R = ST.analyze(r, assets, a.start, a.end, a.window, name=name)
    print(json.dumps(report._clean(R), indent=2) if a.json else ST.console(R))
    return 0


def cmd_correlation(argv: list[str]) -> int:
    from . import correlation as K
    p = argparse.ArgumentParser(prog="python -m backtester correlation",
                                description="Correlation matrix, rolling correlation of a pair and per-asset statistics "
                                            "(total returns, adjusted close).")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--freq", default="monthly", choices=["monthly", "daily"])
    p.add_argument("--window", type=int, help="rolling window in periods (default 36 months or 63 days)")
    p.add_argument("--pair", help="the pair for the rolling correlation, e.g. SPY,TLT (default: the first two)")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    tickers = [t for x in a.tickers for t in x.replace(",", " ").split()]
    pair = a.pair.replace(",", " ").split() if a.pair else None
    R = K.analyze(tickers, a.freq, a.window, a.start, a.end, pair)
    print(json.dumps(report._clean(R), indent=2) if a.json else K.console(R))
    return 0


def cmd_signals(argv: list[str]) -> int:
    from . import signals
    p = argparse.ArgumentParser(prog="python -m backtester signals", description="What the strategy says to do on the latest bar.")
    p.add_argument("text", nargs="?")
    p.add_argument("--spec")
    p.add_argument("--webhook", help="POST the result to this URL (Slack/Discord/any JSON endpoint)")
    a = p.parse_args(argv)
    spec = runner.load(a.spec) if a.spec else parser.parse(a.text)
    s = signals.scan(spec)
    print(signals.format_alert(s))
    print(json.dumps(s, indent=2, default=str))
    if a.webhook:
        print("webhook:", "sent" if signals.post_webhook(a.webhook, s) else "FAILED")
    return 0


def cmd_paper(argv: list[str]) -> int:
    from . import signals
    p = argparse.ArgumentParser(prog="python -m backtester paper", description="Forward-test (paper trade) saved strategies.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("add")
    pa.add_argument("text")
    pa.add_argument("--name", required=True)
    sub.add_parser("report")
    a = p.parse_args(argv)
    if a.cmd == "add":
        path = signals.paper_add(parser.parse(a.text), a.name)
        print(f"Saved {path}; forward results accumulate from today. Commit the paper/ folder to keep it.")
    else:
        rows = signals.paper_report()
        for r in rows:
            print(f"{report._fit(r['name'], 16)} since {r['registered']}  days {r.get('days', 0):>4}  return {report.pct(r.get('return'))}  "
                  f"maxDD {report.pct(r.get('max_drawdown'))}   today: {r['today'].get('action', '')}")
        if not rows:
            print("No paper strategies yet: python -m backtester paper add \"...\" --name NAME")
    return 0


def cmd_trade(argv: list[str]) -> int:
    from . import broker
    p = argparse.ArgumentParser(prog="python -m backtester trade",
                                description="Send today's orders for a strategy to a broker (Alpaca; the paper account unless "
                                            "ALPACA_LIVE=1 and --live). Keys: ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY.")
    p.add_argument("text", nargs="?")
    p.add_argument("--spec")
    p.add_argument("--broker", default="alpaca", choices=["alpaca"])
    p.add_argument("--dry-run", action="store_true", help="print the orders, send nothing")
    p.add_argument("--live", action="store_true", help="trade the live account (also needs ALPACA_LIVE=1)")
    p.add_argument("--account-value", type=float, help="size to this $ amount instead of the account's equity "
                                                         "(needed for a dry run without keys)")
    p.add_argument("--whole-shares", action="store_true", help="whole shares only (keeps market-on-close/-open timing)")
    p.add_argument("--keep-open-orders", action="store_true", help="don't cancel orders this tool placed earlier")
    a = p.parse_args(argv)
    if not (a.text or a.spec):
        p.error("give the strategy sentence or --spec")
    spec = runner.load(a.spec) if a.spec else parser.parse(a.text)
    spec.validate()
    try:
        client = broker.client_for(a.live, a.live)
    except broker.BrokerError as e:
        raise ValueError(str(e)) from None
    print(f"Broker: {client}")
    positions: dict[str, float] = {}
    if client.has_keys:
        positions = broker.broker_positions(client)
        value = a.account_value or broker.account_equity(client)
    elif a.dry_run:
        value = a.account_value or 10_000.0
        print(f"No Alpaca keys: dry run for a ${value:,.0f} account with no positions.")
    else:
        raise ValueError(f"Alpaca keys are not set: export {broker.KEY_ENV} and {broker.SECRET_ENV}, or use --dry-run.")
    pl = broker.plan(spec, value, positions, fractional=not a.whole_shares)
    print(f"As of {pl.as_of}: {len(pl.orders)} order(s) for a ${value:,.0f} account")
    for n in pl.notes:
        print("Note:", n)
    if client.live and not a.dry_run:
        print("*** LIVE ACCOUNT: these orders use real money ***")
    res = broker.submit(client, pl, dry_run=a.dry_run, cancel_stale=not a.keep_open_orders)
    bad = [r for r in res if r["status"] == "error"]
    return 1 if bad else 0


def cmd_tickers(argv: list[str]) -> int:
    st = data.data_status()
    m = data.universe_meta()
    print(json.dumps(st, indent=1))
    print("\nNasdaq-100 (current):", " ".join(data.nasdaq100()))
    print("\nETFs:", " ".join(data.etfs()))
    print("\nIndexes:", " ".join(m.get("indexes", [])))
    print("\nFormer Nasdaq-100 members with data:", " ".join(m.get("former_members", [])))
    return 0


def cmd_import_composer(argv: list[str]) -> int:
    from . import composer_import
    from .portfolio import Portfolio
    p = argparse.ArgumentParser(prog="python -m backtester import-composer",
                                description="Convert a Composer (composer.trade) symphony JSON file into this tool's "
                                            "portfolio spec, and optionally run it.")
    p.add_argument("file", help="the symphony's JSON (exported or copied from Composer); - reads stdin")
    p.add_argument("--out", help="write the portfolio spec here (default: print it)")
    p.add_argument("--run", action="store_true", help="also run the backtest and write the report")
    _common(p)
    p.add_argument("--no-sensitivity", action="store_true", help="skip the transaction-cost re-runs")
    a = p.parse_args(argv)
    src = sys.stdin.read() if a.file == "-" else Path(a.file).read_text()
    d = composer_import.convert(src)
    spec = Portfolio.from_dict(dict(d))
    for k, v in dict(capital=a.capital, start=a.start, end=a.end, slippage_bps=a.slippage_bps, commission=a.commission).items():
        if v is not None:
            setattr(spec, k, v)
            d[k] = v
    spec.validate()
    print(spec.summary())
    for n in d.get("notes", []):
        print("Note:", n)
    js = json.dumps(d, indent=2)
    if a.out:
        Path(a.out).write_text(js + "\n")
        print(f"Spec: {a.out}  (run it with: python -m backtester --spec {a.out})")
    elif not a.run:
        print(js)
    if a.run:
        t0 = time.time()
        res = runner.run(spec)
        A = report.analyze(res, rf=_rf(a.rf), sensitivity=not a.no_sensitivity)
        print(report.console_summary(A))
        out = report.ROOT / "reports" / report.slug(spec.name or "composer-symphony")
        path = report.write_outputs(A, out)
        print(f"Report: {path}  ({time.time() - t0:.1f}s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--tickers-list":
        argv[0] = "tickers"
    cmd = argv[0] if argv and argv[0] in SUBCOMMANDS else "run"
    rest = argv[1:] if argv and argv[0] in SUBCOMMANDS else argv
    try:
        if cmd == "run":
            return cmd_run(rest)
        if cmd == "compare":
            return cmd_compare(rest)
        if cmd in ("sweep", "walkforward"):
            return cmd_sweep(rest, walk=cmd == "walkforward")
        if cmd == "montecarlo":
            return cmd_montecarlo(rest)
        if cmd == "factors":
            return cmd_factors(rest)
        if cmd == "style":
            return cmd_style(rest)
        if cmd in ("correlation", "correlations"):
            return cmd_correlation(rest)
        if cmd == "optimize":
            return cmd_optimize(rest)
        if cmd == "signals":
            return cmd_signals(rest)
        if cmd == "paper":
            return cmd_paper(rest)
        if cmd == "tickers":
            return cmd_tickers(rest)
        if cmd == "trade":
            return cmd_trade(rest)
        if cmd == "library":
            from .library import LIBRARY
            for x in LIBRARY:
                how = f"python -m backtester \"{x['text']}\"" if x.get("text") else "(JSON tree: Gallery page, or library.entry_spec)"
                print(f"[{x['category']}] {x['name']} ({', '.join(x.get('tags', []))}): {x['about']}\n    {how}")
            return 0
        if cmd == "import-composer":
            return cmd_import_composer(rest)
        if cmd == "web":
            from . import web
            return web.main(rest)
    except (parser.ParseError, ValueError, data.DataError) as e:
        print(f"Could not run that:\n  {e}", file=sys.stderr)
        return 2
    except (SyntaxError, TypeError, NameError, KeyError, ZeroDivisionError) as e:
        # a malformed rule (unbalanced brackets, wrong argument types...): say so instead of a traceback
        what = {"SyntaxError": "the rule has a syntax error", "TypeError": "a function got the wrong kind of argument",
                "NameError": "the rule uses an unknown name", "KeyError": "an unknown field was referenced",
                "ZeroDivisionError": "the rule divides by zero"}[type(e).__name__]
        print(f"Could not run that:\n  {what}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
