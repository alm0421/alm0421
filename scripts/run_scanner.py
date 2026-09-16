"""Run the scan -> signal pipeline, once or on a loop.

Order placement requires BOTH ``--execute`` and a mode that transmits orders.
In signal_only mode (the default) ``--execute`` still transmits nothing; the
null broker records intents only.

Outputs are written atomically to ``outputs/``: a temporary file is written and
then renamed, so a reader never sees a half-written file.
"""

from __future__ import annotations

import argparse
import json
import signal as signal_module
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.clock import MarketClock, to_display_tz, utcnow  # noqa: E402
from app.config import ConfigError, load_config  # noqa: E402
from app.data.alpaca_provider import AlpacaDataProvider  # noqa: E402
from app.enums import TradingMode  # noqa: E402
from app.execution.router import build_broker  # noqa: E402
from app.logging import AuditLog, configure_logging, get_logger  # noqa: E402
from app.pipeline import TradingPipeline  # noqa: E402
from app.risk.engine import RiskEngine  # noqa: E402
from app.scanners.base import build_scanners  # noqa: E402
from app.strategies.base import build_strategies  # noqa: E402

log = get_logger("run_scanner")

_STOP = False


def _handle_stop(signum, frame) -> None:  # noqa: ARG001
    global _STOP
    _STOP = True
    print("\nStop requested; finishing the current cycle...")


def write_atomic(path: Path, payload: str) -> None:
    """Write via a temporary file and rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(payload, encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the trading scan pipeline.")
    parser.add_argument("--loop", action="store_true", help="Run continuously.")
    parser.add_argument(
        "--interval", type=int, default=None,
        help="Seconds between cycles (default: scanner.refresh_seconds from config).",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help=(
            "Permit order placement. The trading mode and every risk check still "
            "decide whether anything is actually transmitted."
        ),
    )
    parser.add_argument("--once", action="store_true", help="Run exactly one cycle (default).")
    args = parser.parse_args()

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    log_path = configure_logging(
        level=config.logging.level, log_dir=config.logging.dir,
        json_logs=config.logging.json_logs, console=config.logging.console,
    )
    audit = AuditLog(config.logging.dir / "audit.jsonl")

    print(f"Mode: {config.mode.value.upper()}   Log: {log_path}   Audit: {audit.path}")
    if args.execute and config.mode is TradingMode.SIGNAL_ONLY:
        print(
            "NOTE: --execute was passed but the mode is signal_only. Order intents "
            "will be recorded and NOT transmitted."
        )
    if config.mode is TradingMode.PAPER:
        print("WARNING: PAPER mode. Orders will be transmitted to the Alpaca paper endpoint.")

    try:
        provider = AlpacaDataProvider(config)
        broker = build_broker(config, audit=audit)
        pipeline = TradingPipeline(
            config,
            provider=provider,
            scanners=build_scanners(config.scanner.enabled),
            strategies=build_strategies(config.strategies.enabled, config.strategies.params),
            broker=broker,
            risk_engine=RiskEngine(config, audit=audit),
            audit=audit,
            clock=MarketClock(display_tz=config.data.display_timezone),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Startup failed: {exc}", file=sys.stderr)
        log.exception("Startup failed")
        return 1

    signal_module.signal(signal_module.SIGINT, _handle_stop)
    signal_module.signal(signal_module.SIGTERM, _handle_stop)

    interval = args.interval or config.scanner.refresh_seconds
    cycle = 0

    while True:
        cycle += 1
        started = utcnow()
        try:
            result = pipeline.run(execute=args.execute)
        except Exception as exc:  # noqa: BLE001 - a bad cycle must not kill the loop
            log.exception("Pipeline cycle failed")
            print(f"Cycle {cycle} FAILED: {exc}", file=sys.stderr)
            if not args.loop or _STOP:
                return 1
            time.sleep(interval)
            continue

        local = to_display_tz(started, config.data.display_timezone)
        print(
            f"[{local:%Y-%m-%d %H:%M:%S %Z}] cycle {cycle}: "
            f"{result.bar_sets_received}/{result.symbols_requested} symbols with data, "
            f"{len(result.scan_results)} scan hits, {len(result.signals)} signals, "
            f"{len(result.order_results)} orders, "
            f"{len(result.data_failures)} data failures, {len(result.errors)} errors "
            f"({result.duration_seconds:.2f}s)"
        )
        # A symbol with no data is a visible failure, never a silent skip.
        if result.data_failures:
            shown = list(result.data_failures.items())[:5]
            for symbol, why in shown:
                print(f"    NO DATA: {symbol}: {why}")
            if len(result.data_failures) > len(shown):
                print(f"    NO DATA: ... and {len(result.data_failures) - len(shown)} more")
        if result.bar_sets_received == 0 and result.symbols_requested > 0:
            print(
                "    WARNING: no market data was received for ANY symbol. "
                "Signals cannot be generated. Check connectivity and credentials "
                "with: python scripts/preflight.py"
            )
        for sig in result.signals:
            print(
                f"    {sig.direction.value.upper():5s} {sig.symbol:10s} "
                f"conf={sig.confidence:.2f} @ {sig.reference_price:.4f} "
                f"stop={sig.stop_price:.4f} target={sig.target_price:.4f} "
                f"[{sig.data_freshness.value}]"
            )
        for error in result.errors[:5]:
            print(f"    ERROR: {error}")

        # Latest snapshot (overwritten) plus a timestamped history file.
        payload = json.dumps(result.to_dict(), indent=2, default=str)
        write_atomic(config.outputs_dir / "latest_scan.json", payload)
        write_atomic(
            config.outputs_dir / f"scan_{started:%Y%m%d_%H%M%S}.json", payload
        )

        if not args.loop or _STOP:
            break
        time.sleep(interval)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
