"""Command line: python -m backtester "buy MSFT at the close when it is down 5 days in a row, hold 1 day"."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import engine, expr, parser, report
from .strategy import Strategy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backtester",
        description="Backtest a trading strategy described in plain English (or with explicit rules).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m backtester "buy Microsoft at the close when it trades down 5 days in a row, hold 1 day and sell at the close"
  python -m backtester "buy Nasdaq 100 stocks at the close when RSI(2) is below 5 and it is above its 200-day moving average, sell when it closes above its 5-day moving average, max 10 positions"
  python -m backtester "short QQQ at the open when it gaps up 1%, cover at the close"
  python -m backtester --tickers MSFT --entry "down_days >= 5" --hold 1
  python -m backtester --spec my_strategy.json
""",
    )
    p.add_argument("text", nargs="?", help="strategy in plain English")
    p.add_argument("--spec", help="JSON strategy file (fields of backtester.strategy.Strategy)")
    p.add_argument("--tickers", help="comma-separated tickers, or NDX for the Nasdaq-100 (explicit mode)")
    p.add_argument("--entry", help="entry rule in the expression language (explicit mode)")
    p.add_argument("--exit-when", help="exit rule in the expression language")
    p.add_argument("--hold", type=int, help="exit after N bars")
    p.add_argument("--short", action="store_true", help="short instead of long (explicit mode)")
    p.add_argument("--entry-fill", choices=["close", "next_open", "next_close"])
    p.add_argument("--exit-fill", choices=["close", "open"], help="fill for --hold exits")
    p.add_argument("--stop-loss", type=float, help="e.g. 0.05 for 5%%")
    p.add_argument("--take-profit", type=float)
    p.add_argument("--trailing-stop", type=float)
    p.add_argument("--capital", type=float)
    p.add_argument("--max-positions", type=int)
    p.add_argument("--position-size", type=float, help="fraction of equity per position")
    p.add_argument("--rank-by", help="expression used to prioritise signals when slots are limited (highest first)")
    p.add_argument("--rank-ascending", action="store_true")
    p.add_argument("--slippage-bps", type=float)
    p.add_argument("--commission", type=float, help="$ per order")
    p.add_argument("--commission-per-share", type=float)
    p.add_argument("--whole-shares", action="store_true", help="disallow fractional shares")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--rf", type=float, default=0.0, help="annual risk-free rate for Sharpe/Sortino (default 0)")
    p.add_argument("--out", help="output folder (default reports/<slug>)")
    p.add_argument("--no-sensitivity", action="store_true", help="skip the transaction-cost re-runs")
    p.add_argument("--dry-run", action="store_true", help="only show how the text was interpreted")
    p.add_argument("--help-expr", action="store_true", help="show the rule language reference")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.help_expr:
        print(expr.HELP)
        return 0

    overrides = dict(
        capital=a.capital, max_positions=a.max_positions, position_size=a.position_size,
        slippage_bps=a.slippage_bps, commission=a.commission, commission_per_share=a.commission_per_share,
        start=a.start, end=a.end, rank_by=a.rank_by, stop_loss=a.stop_loss, take_profit=a.take_profit,
        trailing_stop=a.trailing_stop, hold_bars=a.hold, exit_when=a.exit_when, entry_fill=a.entry_fill,
        hold_exit_fill=a.exit_fill,
    )
    if a.rank_ascending:
        overrides["rank_ascending"] = True
    if a.whole_shares:
        overrides["fractional_shares"] = False
    try:
        if a.spec:
            strat = Strategy.from_dict(json.loads(Path(a.spec).read_text()))
            for k, v in overrides.items():
                if v is not None:
                    setattr(strat, k, v)
        elif a.text:
            strat = parser.parse(a.text, **overrides)
        elif a.tickers and a.entry:
            from . import data
            uni = data.nasdaq100() if a.tickers.upper() == "NDX" else [t.strip().upper() for t in a.tickers.split(",")]
            strat = Strategy(universe=uni, entry=a.entry, side="short" if a.short else "long",
                             max_positions=a.max_positions or (1 if len(uni) == 1 else 10),
                             description=f"{'short' if a.short else 'long'} {a.tickers} when {a.entry}")
            for k, v in overrides.items():
                if v is not None:
                    setattr(strat, k, v)
        else:
            build_parser().print_help()
            return 2
        strat.validate()
    except parser.ParseError as e:
        print(f"Could not understand the strategy:\n  {e}", file=sys.stderr)
        return 2

    if a.dry_run:
        print(strat.summary())
        for n in strat.notes:
            print("Note:", n)
        print(strat.to_json())
        return 0

    t0 = time.time()
    res = engine.run(strat)
    analysis = report.analyze(res, rf=a.rf, sensitivity=not a.no_sensitivity)
    print(report.console_summary(res, analysis))
    out = Path(a.out) if a.out else report.ROOT / "reports" / report.slug(strat.description or strat.entry)
    path = report.write_outputs(res, analysis, out)
    print(f"Report: {path}\nFiles:  {out}/trades.csv, equity.csv, yearly.csv, monthly.csv, summary.json, strategy.json")
    print(f"({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
