"""Execution routing: the single decision point for "which broker, if any".

Every order path in the platform obtains its broker from :func:`build_broker`.
Nothing else constructs a broker directly. That makes this module the one place
where mode separation is enforced, and the one place to audit for it.

Routing table
-------------
=============  =======================================================
Mode           Broker
=============  =======================================================
SIGNAL_ONLY    :class:`~app.execution.null_broker.NullBroker` - records
               intents, transmits nothing.
BACKTEST       Refused. No backtest engine ships in this release, and
               silently falling back to another broker would be worse
               than failing.
PAPER          :class:`~app.execution.alpaca_broker.AlpacaBroker` bound
               to the paper endpoint with paper credentials.
LIVE           Refused unless every gate in
               :mod:`app.execution.live_gate` is open. They are not.
=============  =======================================================
"""

from __future__ import annotations

from app.config import AppConfig
from app.enums import TradingMode
from app.execution.base import Broker
from app.execution.live_gate import assert_live_trading_allowed
from app.execution.null_broker import NullBroker
from app.logging import AuditLog, get_logger

log = get_logger(__name__)


class ExecutionRouterError(RuntimeError):
    """Raised when a mode cannot be routed to a broker."""


def build_broker(config: AppConfig, *, audit: AuditLog | None = None) -> Broker:
    """Return the broker for the configured mode.

    Raises rather than falling back. A mode that cannot be honoured must stop
    the platform, not quietly become a different mode.
    """
    mode = config.mode

    if mode is TradingMode.SIGNAL_ONLY:
        log.info("Execution mode: SIGNAL_ONLY. No orders will be transmitted.")
        if audit is not None:
            audit.record("broker_selected", mode=mode.value, broker="null", transmits=False)
        return NullBroker(audit=audit)

    if mode is TradingMode.BACKTEST:
        raise ExecutionRouterError(
            "Mode 'backtest' has no execution route. A historical simulation "
            "engine is not part of this release, so there is nothing to route "
            "orders to. Use 'signal_only' to generate and inspect signals."
        )

    if mode is TradingMode.PAPER:
        # Imported lazily so signal-only operation never requires the broker SDK
        # to be importable or credentials to be present.
        from app.execution.alpaca_broker import AlpacaBroker

        log.warning(
            "Execution mode: PAPER. Orders WILL be transmitted to the Alpaca "
            "paper endpoint. No real money is at risk."
        )
        broker = AlpacaBroker(config, mode=mode, audit=audit)
        if audit is not None:
            audit.record("broker_selected", mode=mode.value, broker=broker.name, transmits=True)
        return broker

    if mode is TradingMode.LIVE:
        # Raises LiveTradingDisabled unless every gate is open. Gate 1 is closed
        # in this release, so this always raises.
        assert_live_trading_allowed(config)

        from app.execution.alpaca_broker import AlpacaBroker

        log.critical(
            "Execution mode: LIVE. Orders WILL be transmitted with REAL MONEY."
        )
        broker = AlpacaBroker(config, mode=mode, audit=audit)
        if audit is not None:
            audit.record(
                "broker_selected", mode=mode.value, broker=broker.name,
                transmits=True, real_money=True,
            )
        return broker

    raise ExecutionRouterError(f"Unhandled trading mode: {mode!r}")
