"""Broker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.enums import AssetClass, TradingMode
from app.execution.models import OrderIntent, OrderResult
from app.portfolio.positions import PortfolioState


class BrokerError(RuntimeError):
    """Raised when a broker operation fails in a way the caller must handle."""


class Broker(ABC):
    """Interface every execution adapter implements.

    Implementations must never widen their own permissions: a broker bound to
    ``TradingMode.PAPER`` must be incapable of reaching a live endpoint.
    """

    #: Identifier used in logs, audit entries and the dashboard, e.g. "alpaca-paper".
    name: str = "unknown"

    #: The mode this broker instance is bound to.
    mode: TradingMode = TradingMode.SIGNAL_ONLY

    @property
    def submits_real_orders(self) -> bool:
        """True when this adapter actually transmits to an external venue."""
        return self.mode.submits_real_orders

    @abstractmethod
    def submit_order(self, intent: OrderIntent) -> OrderResult:
        """Submit an order. Must not raise for ordinary rejections - return an
        :class:`OrderResult` carrying the rejection instead."""

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True when the cancel was accepted."""

    @abstractmethod
    def get_order(self, broker_order_id: str) -> OrderResult | None:
        """Fetch the current state of a previously submitted order."""

    @abstractmethod
    def get_portfolio(self) -> PortfolioState:
        """Fetch account and open positions.

        On failure this must return :meth:`PortfolioState.unknown` rather than
        an empty portfolio, so risk checks can tell the two apart.
        """

    @abstractmethod
    def is_symbol_tradable(self, symbol: str, asset_class: AssetClass) -> bool:
        """Whether the broker will accept orders for this symbol right now."""

    @abstractmethod
    def health(self) -> dict[str, object]:
        """Connectivity and account-mode status for the dashboard."""
