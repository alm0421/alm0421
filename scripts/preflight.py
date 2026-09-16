"""Non-destructive preflight check.

Run this before anything else, and after any config or credential change. It
validates everything that can be validated without placing an order:

* configuration loads and is internally consistent
* log and output directories exist and are writable
* the trading mode and every live-trading gate
* credential presence (never their values)
* market data connectivity and the freshness label in force
* broker connectivity and the account mode actually reached
* every universe symbol's tradability
* the kill-switch state

It NEVER places an order and never modifies configuration.

Exit codes: 0 all checks passed, 1 one or more checks FAILED, 2 the config
could not be loaded at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/preflight.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.clock import MarketClock, to_display_tz, utcnow  # noqa: E402
from app.config import AppConfig, ConfigError, load_config  # noqa: E402
from app.enums import AssetClass, TradingMode  # noqa: E402
from app.logging import configure_logging  # noqa: E402

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_SYMBOLS = {PASS: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, check: str, detail: str = "") -> None:
        self.rows.append((status, check, detail))
        print(f"{_SYMBOLS[status]} {check}" + (f"\n         {detail}" if detail else ""))

    @property
    def failed(self) -> int:
        return sum(1 for status, _, _ in self.rows if status == FAIL)

    @property
    def warned(self) -> int:
        return sum(1 for status, _, _ in self.rows if status == WARN)


def _section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(4, 60 - len(title)))


def check_config(report: Report) -> AppConfig | None:
    _section("Configuration")
    try:
        config = load_config()
    except ConfigError as exc:
        report.add(FAIL, "Config loads", str(exc))
        return None
    report.add(PASS, "Config loads", f"{config.source_path}")
    report.add(PASS, "Trading mode", f"{config.mode.value.upper()}")

    if config.mode is TradingMode.LIVE:
        report.add(FAIL, "Mode safety", "Mode is LIVE. Live trading is disabled in this release.")
    elif config.mode is TradingMode.PAPER:
        report.add(
            WARN, "Mode safety",
            "PAPER mode: orders WILL be transmitted to the Alpaca paper endpoint. "
            "No real money is at risk.",
        )
    else:
        report.add(PASS, "Mode safety", "No orders will be transmitted.")
    return config


def check_directories(report: Report, config: AppConfig) -> None:
    _section("Directories")
    for label, directory in (("Log", config.logging.dir), ("Output", config.outputs_dir)):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".preflight_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            report.add(PASS, f"{label} directory writable", str(directory))
        except Exception as exc:  # noqa: BLE001
            report.add(FAIL, f"{label} directory writable", f"{directory}: {exc}")


def check_live_gates(report: Report, config: AppConfig) -> None:
    _section("Live-trading gates")
    from app.execution.live_gate import live_gate_status

    status = live_gate_status(config)
    if status["all_gates_open"]:
        report.add(FAIL, "Live gates", "ALL LIVE GATES ARE OPEN. Real money is at risk.")
    else:
        closed = [name for name, value in status.items() if name != "all_gates_open" and not value]
        report.add(PASS, "Live trading blocked", f"closed gate(s): {', '.join(closed)}")


def check_kill_switch(report: Report, config: AppConfig) -> None:
    _section("Kill switch")
    path = config.risk.kill_switch_file
    if path.exists():
        report.add(WARN, "Kill switch", f"ENGAGED ({path}). All orders will be rejected.")
    else:
        report.add(PASS, "Kill switch", f"disengaged (create {path} to halt trading)")


def check_credentials(report: Report, config: AppConfig) -> None:
    _section("Credentials")
    which = "LIVE" if config.mode is TradingMode.LIVE else "PAPER"
    if config.credentials.is_complete:
        report.add(PASS, f"Alpaca {which} credentials present", "(values not shown)")
    elif config.mode.submits_real_orders:
        report.add(
            FAIL, f"Alpaca {which} credentials present",
            f"Set ALPACA_{which}_API_KEY and ALPACA_{which}_SECRET_KEY in .env",
        )
    else:
        report.add(
            WARN, "Alpaca credentials present",
            "Not set. Signal-only mode works without them, but equity and option "
            "market data will be unavailable.",
        )


def check_market_sessions(report: Report, config: AppConfig) -> None:
    _section("Market sessions")
    clock = MarketClock(display_tz=config.data.display_timezone)
    now = utcnow()
    report.add(
        PASS, "Local time",
        f"{to_display_tz(now, config.data.display_timezone).isoformat()} "
        f"({config.data.display_timezone})",
    )
    for asset_class in (AssetClass.US_EQUITY, AssetClass.CRYPTO):
        info = clock.session_for(asset_class, now=now)
        report.add(PASS, f"{asset_class.value} session", info.session.value)


def check_data(report: Report, config: AppConfig) -> None:
    _section("Market data")
    from app.data.alpaca_provider import AlpacaDataProvider

    provider = AlpacaDataProvider(config)
    health = provider.health()
    report.add(PASS, "Data feed", f"{health['feed']} -> labelled '{health['freshness_label']}'")

    if health["freshness_label"] in ("delayed", "unknown"):
        report.add(
            WARN, "Feed suitable for execution",
            f"Feed '{health['feed']}' is labelled '{health['freshness_label']}' and will "
            "be REFUSED for order sizing and triggering.",
        )
    else:
        report.add(PASS, "Feed suitable for execution", "")

    if health.get("connected"):
        report.add(
            PASS, "Data connectivity",
            f"probe {health.get('probe_symbol')} age={health.get('probe_age_seconds')}s",
        )
    else:
        status = FAIL if config.mode.submits_real_orders else WARN
        report.add(status, "Data connectivity", str(health.get("error", "unknown error"))[:200])

    report.add(PASS, "Futures data", "not available (Alpaca offers no futures data)")


def check_broker(report: Report, config: AppConfig) -> None:
    _section("Broker")
    if not config.mode.submits_real_orders:
        report.add(
            PASS, "Broker", f"not required in {config.mode.value} mode (no orders transmitted)"
        )
        return

    from app.execution.base import BrokerError
    from app.execution.live_gate import LiveTradingDisabled
    from app.execution.router import ExecutionRouterError, build_broker

    try:
        broker = build_broker(config)
    except (BrokerError, ExecutionRouterError, LiveTradingDisabled) as exc:
        report.add(FAIL, "Broker connection", str(exc).splitlines()[0][:200])
        return

    health = broker.health()
    if health.get("connected"):
        report.add(
            PASS, "Broker connection",
            f"{health['broker']} account={health.get('account')} "
            f"mode={health.get('account_mode')}",
        )
        if health.get("trading_blocked"):
            report.add(FAIL, "Broker trading enabled", "The broker has BLOCKED trading.")
        else:
            report.add(PASS, "Broker trading enabled", "")
        report.add(PASS, "Market open (broker clock)", str(health.get("market_is_open")))
    else:
        report.add(FAIL, "Broker connection", str(health.get("error", "unknown"))[:200])


def check_symbols(report: Report, config: AppConfig, *, verify_tradable: bool) -> None:
    _section("Universe")
    total = sum(len(s) for s in config.universe.values())
    report.add(PASS, "Universe loaded", f"{total} symbol(s)")

    for asset_class, symbols in config.universe.items():
        if not symbols:
            continue
        if asset_class is AssetClass.FUTURES:
            report.add(
                FAIL, f"{asset_class.value} symbols",
                f"{len(symbols)} futures symbol(s) configured, but no shipped adapter "
                "supports futures. They will be rejected.",
            )
            continue
        if asset_class not in config.execution.allowed_asset_classes:
            report.add(
                WARN, f"{asset_class.value} symbols",
                f"{len(symbols)} symbol(s) present but the asset class is not in "
                "execution.allowed_asset_classes; signals only, no orders.",
            )
        else:
            report.add(PASS, f"{asset_class.value} symbols", f"{len(symbols)} symbol(s)")

    if not verify_tradable or not config.mode.submits_real_orders:
        return

    from app.execution.router import build_broker

    try:
        broker = build_broker(config)
    except Exception:  # noqa: BLE001 - already reported by check_broker
        return
    for asset_class, symbols in config.universe.items():
        if asset_class is AssetClass.FUTURES or not symbols:
            continue
        untradable = [s for s in symbols if not broker.is_symbol_tradable(s, asset_class)]
        if untradable:
            report.add(WARN, f"{asset_class.value} tradability", f"not tradable: {untradable}")
        else:
            report.add(PASS, f"{asset_class.value} tradability", "all symbols tradable")


def check_components(report: Report, config: AppConfig) -> None:
    _section("Components")
    from app.scanners.base import build_scanners
    from app.strategies.base import build_strategies

    try:
        scanners = build_scanners(config.scanner.enabled)
        report.add(PASS, "Scanners build", ", ".join(s.name for s in scanners) or "none")
    except Exception as exc:  # noqa: BLE001
        report.add(FAIL, "Scanners build", str(exc)[:200])
    try:
        strategies = build_strategies(config.strategies.enabled, config.strategies.params)
        report.add(PASS, "Strategies build", ", ".join(s.name for s in strategies) or "none")
    except Exception as exc:  # noqa: BLE001
        report.add(FAIL, "Strategies build", str(exc)[:200])


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-destructive preflight check.")
    parser.add_argument(
        "--skip-network", action="store_true",
        help="Skip data and broker connectivity checks (offline validation only).",
    )
    parser.add_argument(
        "--verify-symbols", action="store_true",
        help="Query the broker for each universe symbol's tradability.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  TRADING PLATFORM PREFLIGHT")
    print("=" * 70)

    report = Report()
    config = check_config(report)
    if config is None:
        print("\nPreflight ABORTED: configuration could not be loaded.")
        return 2

    configure_logging(
        level=config.logging.level, log_dir=config.logging.dir,
        json_logs=config.logging.json_logs, console=False,
    )

    check_directories(report, config)
    check_live_gates(report, config)
    check_kill_switch(report, config)
    check_credentials(report, config)
    check_market_sessions(report, config)
    check_components(report, config)

    if args.skip_network:
        _section("Network checks")
        report.add(WARN, "Network checks", "SKIPPED (--skip-network). Connectivity is UNVERIFIED.")
    else:
        check_data(report, config)
        check_broker(report, config)

    check_symbols(report, config, verify_tradable=args.verify_symbols)

    print("\n" + "=" * 70)
    print(
        f"  {len(report.rows)} checks | {report.failed} failed | {report.warned} warnings"
    )
    print("=" * 70)
    if report.failed:
        print("\nPreflight FAILED. Resolve the items above before running the platform.")
        return 1
    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
