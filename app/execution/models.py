"""Order intents and results.

An :class:`OrderIntent` is what the platform *wants* to do. An
:class:`OrderResult` is what actually happened. They are separate types on
purpose: in signal-only mode an intent is produced and recorded but no result
ever comes back from a broker, and conflating the two is how a system ends up
reporting simulated activity as though it were real.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from app.clock import ensure_aware, utcnow
from app.enums import AssetClass, OrderSide, OrderType, TimeInForce, TradingMode


class OrderStatus(str, Enum):
    """Lifecycle of an order intent."""

    #: Created but not yet validated.
    DRAFT = "draft"
    #: Failed validation or risk checks. Never reached a broker.
    REJECTED_LOCALLY = "rejected_locally"
    #: Recorded in signal-only mode. Deliberately never sent to a broker.
    RECORDED_INTENT = "recorded_intent"
    #: Accepted by the broker.
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    #: The broker refused it.
    REJECTED_BY_BROKER = "rejected_by_broker"
    #: Submission failed in transit; the true state is unknown and must be
    #: reconciled against the broker before retrying.
    UNKNOWN = "unknown"

    @property
    def reached_broker(self) -> bool:
        return self in (
            OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED,
            OrderStatus.REJECTED_BY_BROKER, OrderStatus.UNKNOWN,
        )

    @property
    def is_terminal(self) -> bool:
        return self in (
            OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED,
            OrderStatus.REJECTED_BY_BROKER, OrderStatus.REJECTED_LOCALLY,
            OrderStatus.RECORDED_INTENT,
        )


class RejectionReason(str, Enum):
    """Why an order was refused before reaching a broker."""

    KILL_SWITCH = "kill_switch"
    MODE_NOT_EXECUTABLE = "mode_not_executable"
    LIVE_TRADING_DISABLED = "live_trading_disabled"
    UNSUPPORTED_ASSET_CLASS = "unsupported_asset_class"
    UNSUPPORTED_ORDER_TYPE = "unsupported_order_type"
    INVALID_ORDER = "invalid_order"
    STALE_DATA = "stale_data"
    UNKNOWN_ACCOUNT_STATE = "unknown_account_state"
    MARKET_CLOSED = "market_closed"
    EXTENDED_HOURS_NOT_ALLOWED = "extended_hours_not_allowed"
    RISK_LIMIT = "risk_limit"
    DUPLICATE_ORDER = "duplicate_order"
    MISSING_STOP_LOSS = "missing_stop_loss"
    INSUFFICIENT_BUYING_POWER = "insufficient_buying_power"
    BROKER_ERROR = "broker_error"
    SYMBOL_NOT_TRADABLE = "symbol_not_tradable"


@dataclass(frozen=True)
class OrderIntent:
    """A fully specified order the platform intends to place.

    Every field a broker needs is explicit. Nothing is defaulted at submission
    time, so what was validated is exactly what is sent.
    """

    symbol: str
    asset_class: AssetClass
    side: OrderSide
    quantity: float
    order_type: OrderType
    time_in_force: TimeInForce
    #: The mode in force when the intent was created. Carried so an intent can
    #: never be replayed into a different mode than the one that produced it.
    mode: TradingMode
    limit_price: float | None = None
    stop_price: float | None = None
    trail_percent: float | None = None
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    extended_hours: bool = False
    #: Idempotency key sent to the broker to prevent duplicate submission.
    client_order_id: str = field(default_factory=lambda: f"tp-{uuid.uuid4().hex[:20]}")
    #: Provenance.
    signal_id: str | None = None
    strategy: str | None = None
    reference_price: float | None = None
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(
            self, "created_at", ensure_aware(self.created_at, field="order.created_at")
        )

    @property
    def is_bracket(self) -> bool:
        """True when the intent carries both a protective stop and a target."""
        return self.take_profit_price is not None and self.stop_loss_price is not None

    @property
    def estimated_notional(self) -> float | None:
        """Best available notional estimate, for risk checks."""
        price = self.limit_price or self.reference_price or self.stop_price
        return abs(self.quantity) * price if price else None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            asset_class=self.asset_class.value,
            side=self.side.value,
            order_type=self.order_type.value,
            time_in_force=self.time_in_force.value,
            mode=self.mode.value,
            created_at=self.created_at.isoformat(),
            estimated_notional=self.estimated_notional,
        )
        return payload


@dataclass(frozen=True)
class OrderResult:
    """The outcome of an order intent."""

    intent: OrderIntent
    status: OrderStatus
    #: Broker-assigned id. None when the order never reached a broker.
    broker_order_id: str | None = None
    filled_quantity: float = 0.0
    filled_avg_price: float | None = None
    rejection_reason: RejectionReason | None = None
    #: Human-readable detail: which limit was breached, what the broker said.
    message: str = ""
    #: Which broker handled it, e.g. "alpaca-paper", "null(signal_only)".
    broker: str = ""
    submitted_at: datetime | None = None
    recorded_at: datetime = field(default_factory=utcnow)
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "recorded_at", ensure_aware(self.recorded_at, field="result.recorded_at")
        )

    @property
    def succeeded(self) -> bool:
        """True when the order reached a broker and was not refused by it."""
        return self.status in (
            OrderStatus.SUBMITTED, OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
        )

    @property
    def is_simulated(self) -> bool:
        """True when nothing was sent anywhere - a recorded intent only."""
        return self.status is OrderStatus.RECORDED_INTENT

    @classmethod
    def rejected(
        cls, intent: OrderIntent, reason: RejectionReason, message: str, *, broker: str = "n/a"
    ) -> "OrderResult":
        return cls(
            intent=intent, status=OrderStatus.REJECTED_LOCALLY,
            rejection_reason=reason, message=message, broker=broker,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "broker": self.broker,
            "broker_order_id": self.broker_order_id,
            "filled_quantity": self.filled_quantity,
            "filled_avg_price": self.filled_avg_price,
            "rejection_reason": self.rejection_reason.value if self.rejection_reason else None,
            "message": self.message,
            "is_simulated": self.is_simulated,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "recorded_at": self.recorded_at.isoformat(),
            "intent": self.intent.to_dict(),
        }
