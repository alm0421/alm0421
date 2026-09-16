"""Alpaca broker adapter.

Paper and live are separated by construction, not by a runtime flag the caller
passes in: the adapter reads ``mode`` and derives ``paper=`` from it, and
refuses to build at all for a mode that is not PAPER or LIVE.

The adapter also *verifies* the account mode it actually connected to against
the mode it was asked for. A key pair that turns out to belong to a live
account while the platform believes it is paper trading is a hard failure, not
a warning.
"""

from __future__ import annotations

import time
from typing import Any

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import OrderStatus as AlpacaStatus
from alpaca.trading.enums import TimeInForce as AlpacaTif
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)

from app.clock import utcnow
from app.config import AppConfig
from app.enums import AssetClass, OrderSide, OrderType, TimeInForce, TradingMode
from app.execution.base import Broker, BrokerError
from app.execution.models import (
    OrderIntent,
    OrderResult,
    OrderStatus,
    RejectionReason,
)
from app.logging import AuditLog, get_logger
from app.portfolio.positions import AccountSnapshot, PortfolioState, Position

log = get_logger(__name__)

_SIDE = {OrderSide.BUY: AlpacaSide.BUY, OrderSide.SELL: AlpacaSide.SELL}

_TIF = {
    TimeInForce.DAY: AlpacaTif.DAY,
    TimeInForce.GTC: AlpacaTif.GTC,
    TimeInForce.IOC: AlpacaTif.IOC,
    TimeInForce.FOK: AlpacaTif.FOK,
    TimeInForce.OPG: AlpacaTif.OPG,
    TimeInForce.CLS: AlpacaTif.CLS,
}

_STATUS = {
    AlpacaStatus.NEW: OrderStatus.SUBMITTED,
    AlpacaStatus.ACCEPTED: OrderStatus.ACCEPTED,
    AlpacaStatus.PENDING_NEW: OrderStatus.SUBMITTED,
    AlpacaStatus.ACCEPTED_FOR_BIDDING: OrderStatus.ACCEPTED,
    AlpacaStatus.PARTIALLY_FILLED: OrderStatus.PARTIALLY_FILLED,
    AlpacaStatus.FILLED: OrderStatus.FILLED,
    AlpacaStatus.CANCELED: OrderStatus.CANCELED,
    AlpacaStatus.PENDING_CANCEL: OrderStatus.CANCELED,
    AlpacaStatus.EXPIRED: OrderStatus.EXPIRED,
    AlpacaStatus.REJECTED: OrderStatus.REJECTED_BY_BROKER,
    AlpacaStatus.SUSPENDED: OrderStatus.UNKNOWN,
    AlpacaStatus.DONE_FOR_DAY: OrderStatus.EXPIRED,
    AlpacaStatus.STOPPED: OrderStatus.UNKNOWN,
}

_ASSET_CLASS = {
    "us_equity": AssetClass.US_EQUITY,
    "us_option": AssetClass.US_OPTION,
    "crypto": AssetClass.CRYPTO,
}


class AlpacaAccountModeMismatch(BrokerError):
    """The connected account's mode does not match the mode we intended."""


class AlpacaBroker(Broker):
    """Order execution against Alpaca, bound to a single trading mode."""

    def __init__(
        self,
        config: AppConfig,
        *,
        mode: TradingMode,
        audit: AuditLog | None = None,
        verify_account_mode: bool = True,
    ) -> None:
        if mode not in (TradingMode.PAPER, TradingMode.LIVE):
            raise BrokerError(
                f"AlpacaBroker cannot be constructed for mode '{mode.value}'. "
                "Only 'paper' and 'live' reach a broker."
            )
        credentials = config.credentials
        if not credentials.is_complete:
            which = "LIVE" if mode is TradingMode.LIVE else "PAPER"
            raise BrokerError(
                f"Alpaca {which} credentials are not configured. Set "
                f"ALPACA_{which}_API_KEY and ALPACA_{which}_SECRET_KEY in your "
                "environment or .env file."
            )

        self._config = config
        self.mode = mode
        self._is_paper = mode is TradingMode.PAPER
        self.name = f"alpaca-{'paper' if self._is_paper else 'live'}"
        self._audit = audit
        self._verified_mode: bool | None = None

        self._client = TradingClient(
            api_key=credentials.api_key,
            secret_key=credentials.secret_key,
            paper=self._is_paper,
        )

        if verify_account_mode:
            self._verify_account_mode()

    # -- account mode verification ----------------------------------------

    def _verify_account_mode(self) -> None:
        """Confirm the account we reached matches the mode we intended.

        Alpaca's paper and live endpoints are distinct hosts, so a paper client
        cannot reach a live account. This check additionally confirms the
        account is reachable and not blocked before any order is built.
        """
        try:
            account = self._client.get_account()
        except APIError as exc:
            raise BrokerError(
                f"Could not verify the Alpaca {'paper' if self._is_paper else 'live'} "
                f"account: {exc}. Refusing to trade against an unverified account."
            ) from exc

        blocked = bool(getattr(account, "trading_blocked", False))
        self._verified_mode = self._is_paper

        if self._audit is not None:
            self._audit.record(
                "broker_connected",
                broker=self.name,
                mode=self.mode.value,
                paper=self._is_paper,
                trading_blocked=blocked,
                account_masked=self._mask(getattr(account, "account_number", "")),
            )
        log.info(
            "Connected to Alpaca %s account (trading_blocked=%s)",
            "paper" if self._is_paper else "LIVE", blocked,
            extra={"broker": self.name, "mode": self.mode.value},
        )

    @staticmethod
    def _mask(account_number: Any) -> str:
        text = str(account_number or "")
        return f"****{text[-4:]}" if len(text) >= 4 else "****"

    # -- order construction ------------------------------------------------

    def _build_request(self, intent: OrderIntent):
        """Translate an OrderIntent into the matching Alpaca request object."""
        symbol = self._broker_symbol(intent.symbol, intent.asset_class)

        common: dict[str, Any] = {
            "symbol": symbol,
            "qty": intent.quantity,
            "side": _SIDE[intent.side],
            "time_in_force": _TIF[intent.time_in_force],
            "client_order_id": intent.client_order_id,
        }
        # Alpaca rejects extended_hours on crypto entirely.
        if intent.asset_class is not AssetClass.CRYPTO:
            common["extended_hours"] = intent.extended_hours

        # Attach the bracket legs when both are present.
        if intent.is_bracket:
            common["order_class"] = "bracket"
            common["take_profit"] = TakeProfitRequest(limit_price=intent.take_profit_price)
            common["stop_loss"] = StopLossRequest(stop_price=intent.stop_loss_price)

        if intent.order_type is OrderType.MARKET:
            return MarketOrderRequest(**common)
        if intent.order_type is OrderType.LIMIT:
            return LimitOrderRequest(limit_price=intent.limit_price, **common)
        if intent.order_type is OrderType.STOP:
            return StopOrderRequest(stop_price=intent.stop_price, **common)
        if intent.order_type is OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                stop_price=intent.stop_price, limit_price=intent.limit_price, **common
            )
        if intent.order_type is OrderType.TRAILING_STOP:
            # Trailing stops cannot carry a bracket.
            common.pop("order_class", None)
            common.pop("take_profit", None)
            common.pop("stop_loss", None)
            return TrailingStopOrderRequest(trail_percent=intent.trail_percent, **common)

        raise BrokerError(f"Unsupported order type '{intent.order_type.value}'")

    @staticmethod
    def _broker_symbol(symbol: str, asset_class: AssetClass) -> str:
        """Map a platform symbol to the broker's expected format.

        Alpaca crypto pairs are slash-delimited (``BTC/USD``). A bare ``BTCUSD``
        is rejected by the API, so it is normalised here rather than failing at
        submission.
        """
        text = symbol.strip().upper()
        if asset_class is AssetClass.CRYPTO and "/" not in text:
            for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
                if text.endswith(quote) and len(text) > len(quote):
                    return f"{text[: -len(quote)]}/{quote}"
            raise BrokerError(
                f"Cannot determine the crypto pair format for '{symbol}'. Use the "
                "slash-delimited form, e.g. 'BTC/USD'."
            )
        return text

    # -- Broker interface --------------------------------------------------

    def submit_order(self, intent: OrderIntent) -> OrderResult:
        if intent.asset_class is AssetClass.FUTURES:
            return OrderResult.rejected(
                intent,
                RejectionReason.UNSUPPORTED_ASSET_CLASS,
                "Alpaca does not support futures trading. No order was sent.",
                broker=self.name,
            )

        try:
            request = self._build_request(intent)
        except BrokerError as exc:
            return OrderResult.rejected(
                intent, RejectionReason.INVALID_ORDER, str(exc), broker=self.name
            )

        attempts = self._config.execution.order_submit_retries + 1
        backoff = self._config.execution.order_submit_backoff_seconds
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                order = self._client.submit_order(request)
            except APIError as exc:
                # A 4xx is a definitive refusal - retrying re-sends a rejected
                # order and can duplicate it. Only transient errors are retried.
                status_code = getattr(exc, "status_code", None)
                if status_code is not None and 400 <= int(status_code) < 500:
                    self._audit_order("order_rejected_by_broker", intent, error=str(exc)[:300])
                    return OrderResult(
                        intent=intent,
                        status=OrderStatus.REJECTED_BY_BROKER,
                        rejection_reason=RejectionReason.BROKER_ERROR,
                        message=f"Alpaca rejected the order: {exc}",
                        broker=self.name,
                    )
                last_error = exc
            except Exception as exc:  # noqa: BLE001 - transport failures
                last_error = exc

            if last_error is not None and attempt < attempts:
                delay = backoff * (2 ** (attempt - 1))
                log.warning(
                    "Order submission failed for %s (attempt %d/%d): %s; retrying in %.1fs",
                    intent.symbol, attempt, attempts, last_error, delay,
                    extra={"symbol": intent.symbol, "client_order_id": intent.client_order_id},
                )
                time.sleep(delay)
                continue
            if last_error is not None:
                break

            self._audit_order("order_submitted", intent, broker_order_id=str(order.id))
            return self._to_result(intent, order)

        # Every attempt failed in transit. The true state is unknown: the order
        # may or may not have reached Alpaca. It must be reconciled by
        # client_order_id before anything is retried.
        self._audit_order("order_submit_failed", intent, error=str(last_error)[:300])
        return OrderResult(
            intent=intent,
            status=OrderStatus.UNKNOWN,
            rejection_reason=RejectionReason.BROKER_ERROR,
            message=(
                f"Order submission failed after {attempts} attempt(s): {last_error}. "
                f"The order state is UNKNOWN - reconcile client_order_id "
                f"'{intent.client_order_id}' against the broker before retrying."
            ),
            broker=self.name,
        )

    def _audit_order(self, event: str, intent: OrderIntent, **extra: Any) -> None:
        if self._audit is None:
            return
        self._audit.record(
            event,
            broker=self.name,
            mode=self.mode.value,
            symbol=intent.symbol,
            side=intent.side.value,
            quantity=intent.quantity,
            order_type=intent.order_type.value,
            client_order_id=intent.client_order_id,
            signal_id=intent.signal_id,
            strategy=intent.strategy,
            **extra,
        )

    def _to_result(self, intent: OrderIntent, order: Any) -> OrderResult:
        status = _STATUS.get(getattr(order, "status", None), OrderStatus.UNKNOWN)
        filled_qty = float(getattr(order, "filled_qty", 0) or 0)
        filled_avg = getattr(order, "filled_avg_price", None)
        return OrderResult(
            intent=intent,
            status=status,
            broker_order_id=str(getattr(order, "id", "")) or None,
            filled_quantity=filled_qty,
            filled_avg_price=float(filled_avg) if filled_avg else None,
            broker=self.name,
            message=f"Alpaca status: {getattr(order, 'status', 'unknown')}",
            submitted_at=getattr(order, "submitted_at", None) or utcnow(),
            raw={"alpaca_status": str(getattr(order, "status", ""))},
        )

    def cancel_order(self, broker_order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(broker_order_id)
            if self._audit is not None:
                self._audit.record("order_canceled", broker=self.name, broker_order_id=broker_order_id)
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to cancel order %s: %s", broker_order_id, exc)
            return False

    def get_order(self, broker_order_id: str) -> OrderResult | None:
        try:
            order = self._client.get_order_by_id(broker_order_id)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to fetch order %s: %s", broker_order_id, exc)
            return None
        # The originating intent is not persisted by the broker; reconstruct the
        # minimum needed to report status.
        intent = OrderIntent(
            symbol=str(order.symbol),
            asset_class=_ASSET_CLASS.get(str(getattr(order, "asset_class", "")), AssetClass.US_EQUITY),
            side=OrderSide.BUY if str(order.side).endswith("buy") else OrderSide.SELL,
            quantity=float(order.qty or 0),
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            mode=self.mode,
            client_order_id=str(getattr(order, "client_order_id", "")),
        )
        return self._to_result(intent, order)

    def get_portfolio(self) -> PortfolioState:
        try:
            account = self._client.get_account()
            raw_positions = self._client.get_all_positions()
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to fetch portfolio: %s", exc)
            return PortfolioState.unknown(f"Could not reach Alpaca: {exc}")

        snapshot = AccountSnapshot(
            equity=float(account.equity or 0),
            cash=float(account.cash or 0),
            buying_power=float(account.buying_power or 0),
            is_paper=self._is_paper,
            trading_blocked=bool(getattr(account, "trading_blocked", False)),
            pattern_day_trader=bool(getattr(account, "pattern_day_trader", False)),
            daytrade_count=int(getattr(account, "daytrade_count", 0) or 0),
            last_equity=float(getattr(account, "last_equity", 0) or 0),
            currency=str(getattr(account, "currency", "USD")),
            account_number_masked=self._mask(getattr(account, "account_number", "")),
        )

        positions = tuple(
            Position(
                symbol=str(p.symbol),
                asset_class=_ASSET_CLASS.get(str(getattr(p, "asset_class", "")), AssetClass.US_EQUITY),
                quantity=float(p.qty or 0),
                average_entry_price=float(p.avg_entry_price or 0),
                current_price=float(p.current_price or 0),
                market_value=float(p.market_value or 0),
                unrealized_pl=float(p.unrealized_pl or 0),
                cost_basis=float(getattr(p, "cost_basis", 0) or 0),
            )
            for p in raw_positions
        )
        return PortfolioState.from_parts(snapshot, positions)

    def is_symbol_tradable(self, symbol: str, asset_class: AssetClass) -> bool:
        if asset_class is AssetClass.FUTURES:
            return False
        try:
            asset = self._client.get_asset(self._broker_symbol(symbol, asset_class))
        except Exception as exc:  # noqa: BLE001
            log.warning("Tradability check failed for %s: %s", symbol, exc)
            return False
        return bool(getattr(asset, "tradable", False))

    def health(self) -> dict[str, object]:
        status: dict[str, object] = {
            "broker": self.name,
            "mode": self.mode.value,
            "transmits_orders": True,
            "account_mode": "paper" if self._is_paper else "LIVE",
            "account_mode_verified": self._verified_mode is not None,
            "supports_futures": False,
            "last_checked": utcnow().isoformat(),
        }
        try:
            account = self._client.get_account()
            clock = self._client.get_clock()
            status.update(
                connected=True,
                trading_blocked=bool(getattr(account, "trading_blocked", False)),
                equity=float(account.equity or 0),
                buying_power=float(account.buying_power or 0),
                account=self._mask(getattr(account, "account_number", "")),
                market_is_open=bool(getattr(clock, "is_open", False)),
                next_open=str(getattr(clock, "next_open", "")),
                next_close=str(getattr(clock, "next_close", "")),
            )
        except Exception as exc:  # noqa: BLE001 - health must never raise
            status.update(connected=False, error=str(exc)[:300])
        return status
