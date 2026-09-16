"""Broker adapters, order validation and execution-mode gating.

Nothing in this package may submit an order without passing through
:func:`app.execution.router.build_broker`, which is the single place where the
trading mode decides which broker (if any) receives the order.
"""

from app.execution.models import (
    OrderIntent,
    OrderResult,
    OrderStatus,
    RejectionReason,
)
from app.execution.base import Broker, BrokerError
from app.execution.router import build_broker, ExecutionRouterError

__all__ = [
    "OrderIntent",
    "OrderResult",
    "OrderStatus",
    "RejectionReason",
    "Broker",
    "BrokerError",
    "build_broker",
    "ExecutionRouterError",
]
