"""Command line.

    python -m backtester "buy MSFT at the close when it is down 5 days in a row, hold 1 day"
    python -m backtester compare "60/40 SPY/TLT rebalanced quarterly" "buy and hold QQQ"
    python -m backtester sweep "buy QQQ when RSI({2..5}) is below {5..20 step 5}, hold {1,3,5} days"
    python -m backtester walkforward "buy QQQ when RSI(2) is below {5..20 step 5}, hold {1..5} days"
    python -m backtester optimize SPY QQQ TLT GLD --max-weight 0.6
    python -m backtester signals "buy Nasdaq 100 stocks when RSI(2) is below 5, hold 3 days"
    python -m backtester paper add "..." --name rsi2 ; python -m backtester paper report
    python -m backtester web            # the backtesting site on http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import data, expr, parser, report, runner

SUBCOMMANDS = {"run", "compare", "sweep", "walkforward", "optimize", "signals", "paper", "web", "tickers", "library"}


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


def _rf(v):
    return "tbill" if v in (None, "tbill") else float(v)


def cmd_run(argv: list[str]) -> int:
    a = build_run_parser().parse_args(argv)
    if a.help_expr:
        print(expr.HELP)
        return 0
    spec = make_spec(a.text, a, a.spec)
    if a.dry_run:
        print(spec.summary())
        for n in spec.notes:
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
        print(f"{s.name:14s} CAGR {report.pct(st['cagr']):>8s}  Sharpe {report.num(st['sharpe']):>5s}  MaxDD {report.pct(st['max_drawdown'], 1):>7s}  "
              f"final ${st['end_equity']:,.0f}   {s.description[:70]}")
    C = report.common_window_stats(analyses, _rf(a.rf))
    print(f"\nCommon period {C['start']} -> {C['end']}:")
    for k, st in C["columns"].items():
        print(f"  {k:24s} CAGR {report.pct(st['cagr']):>8s}  Sharpe {report.num(st['sharpe']):>5s}  MaxDD {report.pct(st['max_drawdown'], 1):>7s}")
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
    p.add_argument("--out")
    a = p.parse_args(argv)
    R = research.optimize(a.tickers, a.start, a.end, a.max_weight, a.min_weight, a.test_start)
    print(research_report.optimize_console(R))
    out = Path(a.out) if a.out else report.ROOT / "reports" / ("optimize-" + report.slug("-".join(a.tickers)))
    path = research_report.write_optimize(R, out)
    print(f"Report: {path}")
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
            print(f"{r['name']:16s} since {r['registered']}  days {r.get('days', 0):>4}  return {report.pct(r.get('return'))}  "
                  f"maxDD {report.pct(r.get('max_drawdown'))}   today: {r['today'].get('action', '')}")
        if not rows:
            print("No paper strategies yet: python -m backtester paper add \"...\" --name NAME")
    return 0


def cmd_tickers(argv: list[str]) -> int:
    st = data.data_status()
    m = data.universe_meta()
    print(json.dumps(st, indent=1))
    print("\nNasdaq-100 (current):", " ".join(data.nasdaq100()))
    print("\nETFs:", " ".join(data.etfs()))
    print("\nIndexes:", " ".join(m.get("indexes", [])))
    print("\nFormer Nasdaq-100 members with data:", " ".join(m.get("former_members", [])))
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
        if cmd == "optimize":
            return cmd_optimize(rest)
        if cmd == "signals":
            return cmd_signals(rest)
        if cmd == "paper":
            return cmd_paper(rest)
        if cmd == "tickers":
            return cmd_tickers(rest)
        if cmd == "library":
            from .library import LIBRARY
            for x in LIBRARY:
                print(f"[{x['category']}] {x['name']}: {x['about']}\n    python -m backtester \"{x['text']}\"")
            return 0
        if cmd == "web":
            from . import web
            return web.main(rest)
    except (parser.ParseError, ValueError, data.DataError) as e:
        print(f"Could not run that:\n  {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
