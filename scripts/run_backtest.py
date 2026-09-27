"""Backtest the HitchHiker scalp on 1-minute bars.

Two ways to generate trades:

  --alerts FILE   Alert mode. Each row (date,time,symbol,direction) arms one
                  HitchHiker break starting at that minute. Add --market-entry to
                  instead enter at the alert bar's open.
  --scan          Scanner mode. Every symbol, every day in the data, first
                  qualifying break in either direction between --scan-start and
                  --scan-end.

Data comes from ``<data-dir>/<SYMBOL>_1min.csv``. With ``--fetch-alpaca`` the
files are first downloaded from Alpaca (IEX feed, needs ALPACA_PAPER_API_KEY /
ALPACA_PAPER_SECRET_KEY in .env) for --start..--end and cached there.

Nothing here constructs a broker or transmits an order.

Examples
--------
  python scripts/run_backtest.py --alerts backtests/alerts/hitchhiker_2026-09-25.csv \\
      --data-dir backtests/data/2026-09-25 --out backtests/reports/alerts
  python scripts/run_backtest.py --scan --symbols NBIS CRCL GOOGL USDE SPCX PURR SPY QQQ \\
      --data-dir backtests/data/2026-09-25 --out backtests/reports/scan
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import date, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backtest import data as bd  # noqa: E402
from app.backtest.hitchhiker import HitchhikerParams  # noqa: E402
from app.backtest.metrics import summarize  # noqa: E402
from app.backtest.portfolio import CostModel, SizingModel, run_portfolio  # noqa: E402
from app.backtest.report import write_report  # noqa: E402
from app.backtest.runner import buy_and_hold, calendar, load_alerts, run_alerts, run_scan  # noqa: E402

BENCHMARKS = ("SPY", "QQQ")

BASE_NOTES = [
    "Entries are buy-stop / sell-stop orders one tick through the consolidation, filled at the "
    "trigger price or at the bar open if it gaps through. Every entry and exit pays the "
    "slippage in the parameters, and IBKR-style commissions are charged on every fill.",
    "Where a single 1-minute bar could have hit both the stop and a profit exit, the stop is "
    "assumed to have come first (pessimistic).",
    "Position size = risk % of realised equity ÷ (entry-to-stop distance + slippage), capped "
    "so total open notional never exceeds max_leverage × equity (2× = Reg T margin). Shorts "
    "assume shares were available to borrow at no fee. A US margin account under $25,000 is "
    "limited by the pattern-day-trader rule to 3 day trades per rolling 5 days; the backtest "
    "does not enforce that.",
    "Waves are mechanical: a wave ends after <code>wave_pause_bars</code> consecutive bars "
    "without a new high (low for shorts). A discretionary trader reading the tape will exit "
    "differently.",
    "Buy &amp; hold benchmarks buy at the first regular-session open in the window and are "
    "marked at every bar close. No commissions are charged to them.",
    "Only regular-session bars (09:30–16:00 ET) are traded. Premarket bars are used only for "
    "the premarket high/low context column.",
]


def parse_time(text: str) -> time:
    h, m = text.split(":")[:2]
    return time(int(h), int(m))


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest the HitchHiker scalp on 1-minute bars.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--alerts", type=Path, help="Alert CSV: date,time,symbol,direction")
    mode.add_argument("--scan", action="store_true", help="Scan every symbol/day for setups")
    p.add_argument("--symbols", nargs="*", default=None, help="Scanner universe (default: all files)")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output directory for the report")
    p.add_argument("--title", default=None)
    p.add_argument("--market-entry", action="store_true", help="Alert mode: enter at the alert bar open")
    p.add_argument("--relaxed", action="store_true", help="Drop the location and drive rules")
    p.add_argument("--require-market", action="store_true", help="Require SPY/QQQ/IWM alignment")
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--risk-pct", type=float, default=1.0, help="Percent of equity risked per trade")
    p.add_argument("--max-leverage", type=float, default=2.0, help="Reg T margin = 2")
    p.add_argument("--slippage", type=float, default=0.01, help="Per share, per fill")
    p.add_argument("--wave-pause-bars", type=int, default=2)
    p.add_argument("--max-hold", type=int, default=60, help="Minutes")
    p.add_argument("--alert-window", type=int, default=30, help="Minutes after an alert to allow a trigger")
    p.add_argument("--scan-start", type=parse_time, default=time(9, 35))
    p.add_argument("--scan-end", type=parse_time, default=time(10, 30))
    p.add_argument("--fetch-alpaca", action="store_true", help="Download bars from Alpaca first")
    p.add_argument("--start", type=date.fromisoformat, help="--fetch-alpaca start date")
    p.add_argument("--end", type=date.fromisoformat, help="--fetch-alpaca end date")
    args = p.parse_args()

    alerts = load_alerts(args.alerts) if args.alerts else []
    symbols = sorted({a.symbol for a in alerts}) if alerts else [s.upper() for s in (args.symbols or [])]
    if not symbols:
        symbols = sorted(f.name.split("_")[0] for f in args.data_dir.glob("*_1min.csv"))
    wanted = sorted(set(symbols) | set(BENCHMARKS) | {"IWM"})

    if args.fetch_alpaca:  # pragma: no cover - network
        from dotenv import load_dotenv

        load_dotenv()
        key, secret = os.getenv("ALPACA_PAPER_API_KEY"), os.getenv("ALPACA_PAPER_SECRET_KEY")
        if not (args.start and args.end):
            p.error("--fetch-alpaca needs --start and --end")
        # Keys are optional: when absent, the environment's outbound proxy injects
        # the APCA-API-KEY-ID / APCA-API-SECRET-KEY headers for data.alpaca.markets.
        for sym in wanted:
            try:
                bars = bd.fetch_alpaca_minute_bars(sym, args.start, args.end, key, secret)
            except bd.BarDataError as exc:
                # A symbol that had not listed yet in the window returns no bars;
                # skip it rather than aborting the whole run.
                print(f"skipped {sym}: {exc}")
                continue
            bd.save_minute_bars(bars, bd.csv_path(args.data_dir, sym))
            print(f"fetched {sym}: {len(bars):,} bars")

    bars = {s: bd.load_minute_bars(bd.csv_path(args.data_dir, s)) for s in wanted if bd.csv_path(args.data_dir, s).exists()}
    missing = [s for s in symbols if s not in bars]
    if missing:
        p.error(f"no bar files for {missing} in {args.data_dir}")

    params = HitchhikerParams(
        wave_pause_bars=args.wave_pause_bars,
        max_hold_minutes=args.max_hold,
        alert_window_minutes=args.alert_window,
        scan_start=args.scan_start,
        scan_end=args.scan_end,
        require_market_alignment=args.require_market,
    )
    if args.relaxed:
        params = replace(params, require_location=False, require_drive=False)

    if alerts:
        paths, diagnostics = run_alerts(bars, alerts, params, market_entry=args.market_entry)
        days = sorted({a.time.date() for a in alerts})
    else:
        paths, diagnostics = run_scan(bars, symbols, params)
        days = None

    cal = calendar({s: bars[s] for s in symbols}, days)
    sizing = SizingModel(args.capital, args.risk_pct / 100, args.max_leverage)
    costs = CostModel(slippage_per_share=args.slippage)
    result = run_portfolio(paths, cal, sizing, costs)
    bench_curves = {b: buy_and_hold(bars[b], cal, args.capital) for b in BENCHMARKS if b in bars}
    stats = summarize(result.equity, result.trades, args.capital, bench_curves)

    notes = list(BASE_NOTES)
    notes.insert(0, f"Data: {args.data_dir} — {len(cal) // 390} trading day(s), "
                 f"{cal[0]:%Y-%m-%d} to {cal[-1]:%Y-%m-%d}, symbols {', '.join(symbols)}.")
    if args.relaxed:
        notes.append("Relaxed run: the upper/lower-third location rule and the drive rule are off.")
    if args.market_entry:
        notes.append("Market-entry run: each alert is entered at the open of the alert minute, "
                     "not on a consolidation break. The stop sits beyond the consolidation "
                     "before the alert.")

    title = args.title or ("HitchHiker scalp — " + (f"alerts {args.alerts.stem}" if alerts else "scanner"))
    path = write_report(args.out, title, result, stats, bench_curves, params, notes, diagnostics)

    s = stats["summary"]
    print(f"{title}")
    print(f"  trades {s.get('n_trades', 0)}  total return {s['total_return_pct']:+.2f}%  "
          f"ending equity ${s['ending_equity']:,.2f}  max DD {s['max_drawdown_pct']:.2f}%")
    for b, row in stats["benchmarks"].items():
        print(f"  {b} buy&hold {row['total_return_pct']:+.2f}%")
    print(f"  report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
