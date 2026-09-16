"""Shared vocabulary for the platform.

These enums live in a dependency-free module so every subsystem can import them
without creating import cycles.
"""

from __future__ import annotations

from enum import Enum


class TradingMode(str, Enum):
    """Execution mode. Determines which broker the router will hand orders to.

    The modes are strictly separated (see ``app.execution.router``):

    SIGNAL_ONLY  Signals are generated, ranked, logged and displayed. Orders are
                 recorded as *intents* only and are never sent to any broker.
                 This is the default and the only mode that needs no credentials.
    BACKTEST     Reserved for the historical simulation engine. No engine ships
                 in this release, so the router refuses this mode explicitly
                 rather than silently falling back to another one.
    PAPER        Orders are submitted to the Alpaca paper endpoint using paper
                 credentials. No real money is at risk.
    LIVE         Orders would be submitted to the Alpaca live endpoint with real
                 money. Disabled at build level - see ``app.execution.live_gate``.
    """

    SIGNAL_ONLY = "signal_only"
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

    @property
    def submits_real_orders(self) -> bool:
        """True when the mode reaches an external broker in any form."""
        return self in (TradingMode.PAPER, TradingMode.LIVE)

    @property
    def risks_real_money(self) -> bool:
        """True only for LIVE. Paper reaches a broker but risks nothing."""
        return self is TradingMode.LIVE


class AssetClass(str, Enum):
    """Asset classes the platform models.

    FUTURES is modelled so that sizing, risk and routing are not hardcoded to
    equities, but no shipped broker adapter supports it. ``AlpacaBroker``
    rejects futures explicitly instead of mis-routing it as an equity.
    """

    US_EQUITY = "us_equity"
    US_OPTION = "us_option"
    CRYPTO = "crypto"
    FUTURES = "futures"


class Freshness(str, Enum):
    """How current a market data point is. Never inferred - always carried
    explicitly from the provider that produced the data."""

    REALTIME = "realtime"
    DELAYED = "delayed"
    HISTORICAL = "historical"
    SNAPSHOT = "snapshot"
    DERIVED = "derived"
    UNKNOWN = "unknown"

    @property
    def safe_for_execution(self) -> bool:
        """Only genuinely current data may drive an order.

        DELAYED, HISTORICAL and UNKNOWN data must never size or trigger a live
        or paper order. DERIVED inherits the freshness of its inputs and is
        resolved by the caller before this is consulted.
        """
        return self in (Freshness.REALTIME, Freshness.SNAPSHOT)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"
    CLS = "cls"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class MarketSession(str, Enum):
    PREMARKET = "premarket"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"
    ALWAYS_OPEN = "always_open"  # crypto
