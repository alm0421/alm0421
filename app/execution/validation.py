"""Structural order validation.

This module answers one question: *is this order internally coherent and
permitted by configuration?* It deliberately does not consult account state,
market data or position limits - that is the risk engine's job
(:mod:`app.risk.engine`), which runs after this.

Both layers must pass before an order can reach a broker.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import AppConfig
from app.enums import AssetClass, OrderSide, OrderType, TimeInForce
from app.execution.models import OrderIntent, RejectionReason

#: Order types that require a limit price.
_NEEDS_LIMIT = frozenset({OrderType.LIMIT, OrderType.STOP_LIMIT})
#: Order types that require a stop/trigger price.
_NEEDS_STOP = frozenset({OrderType.STOP, OrderType.STOP_LIMIT})

#: Crypto cannot rest a day order the way equities do, and Alpaca requires GTC
#: or IOC for crypto. This is a broker constraint, enforced before submission so
#: the failure message is actionable rather than an opaque 422.
_CRYPTO_ALLOWED_TIF = frozenset({TimeInForce.GTC, TimeInForce.IOC})


@dataclass(frozen=True)
class ValidationIssue:
    reason: RejectionReason
    message: str


def validate_order_intent(intent: OrderIntent, config: AppConfig) -> list[ValidationIssue]:
    """Return every problem found. An empty list means structurally valid.

    All issues are collected rather than short-circuiting on the first, so an
    operator sees everything wrong with an order at once.
    """
    issues: list[ValidationIssue] = []

    # --- symbol ---
    if not intent.symbol or not intent.symbol.strip():
        issues.append(ValidationIssue(RejectionReason.INVALID_ORDER, "Symbol is empty."))

    # --- asset class permitted ---
    if intent.asset_class is AssetClass.FUTURES:
        issues.append(
            ValidationIssue(
                RejectionReason.UNSUPPORTED_ASSET_CLASS,
                f"{intent.symbol}: futures is not supported by any shipped broker "
                "adapter. Alpaca does not offer futures trading or futures market "
                "data. A futures-capable adapter must be added first.",
            )
        )
    elif intent.asset_class not in config.execution.allowed_asset_classes:
        allowed = ", ".join(sorted(a.value for a in config.execution.allowed_asset_classes))
        issues.append(
            ValidationIssue(
                RejectionReason.UNSUPPORTED_ASSET_CLASS,
                f"{intent.symbol}: asset class '{intent.asset_class.value}' is not in "
                f"execution.allowed_asset_classes ({allowed}).",
            )
        )

    # --- order type permitted ---
    if intent.order_type not in config.execution.allowed_order_types:
        allowed = ", ".join(sorted(t.value for t in config.execution.allowed_order_types))
        issues.append(
            ValidationIssue(
                RejectionReason.UNSUPPORTED_ORDER_TYPE,
                f"{intent.symbol}: order type '{intent.order_type.value}' is not in "
                f"execution.allowed_order_types ({allowed}).",
            )
        )

    # --- quantity ---
    if intent.quantity is None or intent.quantity <= 0:
        issues.append(
            ValidationIssue(
                RejectionReason.INVALID_ORDER,
                f"{intent.symbol}: quantity must be positive, got {intent.quantity}. "
                "Direction is carried by `side`, never by a negative quantity.",
            )
        )
    elif intent.asset_class in (AssetClass.US_EQUITY, AssetClass.US_OPTION):
        # Fractional equity shares cannot carry a bracket, and option contracts
        # are always whole.
        if intent.asset_class is AssetClass.US_OPTION and intent.quantity != int(intent.quantity):
            issues.append(
                ValidationIssue(
                    RejectionReason.INVALID_ORDER,
                    f"{intent.symbol}: option quantity must be a whole number of "
                    f"contracts, got {intent.quantity}.",
                )
            )
        if (
            intent.asset_class is AssetClass.US_EQUITY
            and intent.quantity != int(intent.quantity)
            and intent.is_bracket
        ):
            issues.append(
                ValidationIssue(
                    RejectionReason.INVALID_ORDER,
                    f"{intent.symbol}: fractional share quantity ({intent.quantity}) "
                    "cannot carry a bracket (take-profit/stop-loss). Use a whole "
                    "share count.",
                )
            )

    # --- side ---
    if intent.side not in (OrderSide.BUY, OrderSide.SELL):
        issues.append(
            ValidationIssue(RejectionReason.INVALID_ORDER, f"Invalid side {intent.side!r}.")
        )

    # --- price fields required by order type ---
    if intent.order_type in _NEEDS_LIMIT and (intent.limit_price is None or intent.limit_price <= 0):
        issues.append(
            ValidationIssue(
                RejectionReason.INVALID_ORDER,
                f"{intent.symbol}: order type '{intent.order_type.value}' requires a "
                f"positive limit_price, got {intent.limit_price}.",
            )
        )
    if intent.order_type in _NEEDS_STOP and (intent.stop_price is None or intent.stop_price <= 0):
        issues.append(
            ValidationIssue(
                RejectionReason.INVALID_ORDER,
                f"{intent.symbol}: order type '{intent.order_type.value}' requires a "
                f"positive stop_price, got {intent.stop_price}.",
            )
        )
    if intent.order_type is OrderType.TRAILING_STOP:
        if intent.trail_percent is None or not (0 < intent.trail_percent < 100):
            issues.append(
                ValidationIssue(
                    RejectionReason.INVALID_ORDER,
                    f"{intent.symbol}: trailing stop requires trail_percent strictly "
                    f"between 0 and 100, got {intent.trail_percent}.",
                )
            )

    # --- bracket coherence ---
    # An inverted bracket would fill its own stop the instant it is placed.
    reference = intent.limit_price or intent.reference_price
    if reference and reference > 0:
        if intent.side is OrderSide.BUY:
            if intent.stop_loss_price is not None and intent.stop_loss_price >= reference:
                issues.append(
                    ValidationIssue(
                        RejectionReason.INVALID_ORDER,
                        f"{intent.symbol}: BUY stop-loss {intent.stop_loss_price} must be "
                        f"below the entry reference {reference}; as written it would "
                        "trigger immediately.",
                    )
                )
            if intent.take_profit_price is not None and intent.take_profit_price <= reference:
                issues.append(
                    ValidationIssue(
                        RejectionReason.INVALID_ORDER,
                        f"{intent.symbol}: BUY take-profit {intent.take_profit_price} must "
                        f"be above the entry reference {reference}.",
                    )
                )
        else:
            if intent.stop_loss_price is not None and intent.stop_loss_price <= reference:
                issues.append(
                    ValidationIssue(
                        RejectionReason.INVALID_ORDER,
                        f"{intent.symbol}: SELL stop-loss {intent.stop_loss_price} must be "
                        f"above the entry reference {reference}; as written it would "
                        "trigger immediately.",
                    )
                )
            if intent.take_profit_price is not None and intent.take_profit_price >= reference:
                issues.append(
                    ValidationIssue(
                        RejectionReason.INVALID_ORDER,
                        f"{intent.symbol}: SELL take-profit {intent.take_profit_price} must "
                        f"be below the entry reference {reference}.",
                    )
                )

    # --- time in force ---
    if intent.asset_class is AssetClass.CRYPTO and intent.time_in_force not in _CRYPTO_ALLOWED_TIF:
        allowed = ", ".join(sorted(t.value for t in _CRYPTO_ALLOWED_TIF))
        issues.append(
            ValidationIssue(
                RejectionReason.INVALID_ORDER,
                f"{intent.symbol}: crypto orders require time in force of {allowed}, "
                f"got '{intent.time_in_force.value}'.",
            )
        )

    # --- extended hours ---
    if intent.extended_hours:
        if not config.risk.allow_extended_hours:
            issues.append(
                ValidationIssue(
                    RejectionReason.EXTENDED_HOURS_NOT_ALLOWED,
                    f"{intent.symbol}: extended-hours order requested but "
                    "risk.allow_extended_hours is false.",
                )
            )
        if intent.order_type is not OrderType.LIMIT:
            issues.append(
                ValidationIssue(
                    RejectionReason.INVALID_ORDER,
                    f"{intent.symbol}: extended-hours orders must be LIMIT orders; "
                    f"got '{intent.order_type.value}'. A market order outside regular "
                    "hours can fill at an arbitrary price.",
                )
            )
        if intent.time_in_force is not TimeInForce.DAY:
            issues.append(
                ValidationIssue(
                    RejectionReason.INVALID_ORDER,
                    f"{intent.symbol}: extended-hours orders require time in force "
                    f"'day', got '{intent.time_in_force.value}'.",
                )
            )

    # --- mode coherence ---
    if intent.mode is not config.mode:
        issues.append(
            ValidationIssue(
                RejectionReason.MODE_NOT_EXECUTABLE,
                f"{intent.symbol}: order was created in mode '{intent.mode.value}' but "
                f"the platform is now in mode '{config.mode.value}'. An intent may not "
                "be replayed into a different mode.",
            )
        )

    return issues
