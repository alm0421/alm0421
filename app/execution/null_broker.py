"""Signal-only broker.

Records order intents and sends nothing anywhere. This is what backs
``mode: signal_only`` and it is the default for the whole platform.

It is not a simulator: it does not model fills, slippage or position changes,
and it deliberately reports :class:`OrderStatus.RECORDED_INTENT` so that nothing
downstream can mistake a recorded intent for a real execution. Simulated fills
belong in a backtest engine, where their assumptions can be stated explicitly.
"""

from __future__ import annotations

from app.clock import utcnow
from app.enums import AssetClass, TradingMode
from app.execution.base import Broker
from app.execution.models import OrderIntent, OrderResult, OrderStatus
from app.logging import AuditLog
from app.portfolio.positions import PortfolioState


class NullBroker(Broker):
    """Records intents; never transmits."""

    name = "null(signal_only)"
    mode = TradingMode.SIGNAL_ONLY

    def __init__(self, audit: AuditLog | None = None) -> None:
        self._audit = audit
        self._recorded: list[OrderResult] = []

    @property
    def recorded_intents(self) -> tuple[OrderResult, ...]:
        return tuple(self._recorded)

    def submit_order(self, intent: OrderIntent) -> OrderResult:
        result = OrderResult(
            intent=intent,
            status=OrderStatus.RECORDED_INTENT,
            broker=self.name,
            message=(
                "Signal-only mode: order intent recorded. Nothing was sent to any "
                "broker and no position was opened."
            ),
            submitted_at=None,
            recorded_at=utcnow(),
        )
        self._recorded.append(result)
        if self._audit is not None:
            self._audit.record(
                "order_intent_recorded",
                mode=self.mode.value,
                symbol=intent.symbol,
                side=intent.side.value,
                quantity=intent.quantity,
                order_type=intent.order_type.value,
                client_order_id=intent.client_order_id,
                signal_id=intent.signal_id,
                strategy=intent.strategy,
                transmitted=False,
            )
        return result

    def cancel_order(self, broker_order_id: str) -> bool:
        # There is nothing to cancel: no order was ever transmitted.
        return False

    def get_order(self, broker_order_id: str) -> OrderResult | None:
        for result in self._recorded:
            if result.intent.client_order_id == broker_order_id:
                return result
        return None

    def get_portfolio(self) -> PortfolioState:
        """No broker is connected, so account state is genuinely unknown.

        Returning a zeroed account here would let risk checks compare against
        fabricated equity, so the state is reported as unknown instead.
        """
        return PortfolioState.unknown(
            "Signal-only mode: no broker is connected, so account and position "
            "state is unknown."
        )

    def is_symbol_tradable(self, symbol: str, asset_class: AssetClass) -> bool:
        # Without a broker there is no tradability information to report.
        return False

    def health(self) -> dict[str, object]:
        return {
            "broker": self.name,
            "mode": self.mode.value,
            "connected": False,
            "transmits_orders": False,
            "account_mode": "none",
            "recorded_intents": len(self._recorded),
            "note": "Signal-only mode. Order intents are recorded but never sent.",
            "last_checked": utcnow().isoformat(),
        }
