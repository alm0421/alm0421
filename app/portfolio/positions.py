"""Account and position snapshots.

These are read-only views of broker state at a point in time. They carry an
``as_of`` timestamp and an explicit ``is_known`` flag, because "we could not
reach the broker" and "the account has no positions" are completely different
situations and must never collapse into the same empty result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from app.clock import ensure_aware, utcnow
from app.enums import AssetClass


@dataclass(frozen=True)
class Position:
    """An open position."""

    symbol: str
    asset_class: AssetClass
    #: Signed: positive is long, negative is short.
    quantity: float
    average_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    cost_basis: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def abs_market_value(self) -> float:
        return abs(self.market_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class.value,
            "quantity": self.quantity,
            "average_entry_price": self.average_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pl": self.unrealized_pl,
            "side": "long" if self.is_long else "short",
        }


@dataclass(frozen=True)
class AccountSnapshot:
    """Account state at a point in time.

    ``is_known`` is False when the broker could not be reached. Risk checks
    treat an unknown account as a hard block rather than assuming zero or
    unlimited buying power.
    """

    equity: float
    cash: float
    buying_power: float
    #: Broker-reported account mode. None means it could not be determined.
    is_paper: bool | None
    #: True when the broker has restricted or blocked trading on this account.
    trading_blocked: bool = False
    pattern_day_trader: bool = False
    daytrade_count: int = 0
    #: Equity at the start of the current trading day, for daily-loss limits.
    last_equity: float = 0.0
    currency: str = "USD"
    account_number_masked: str = "****"
    as_of: datetime = field(default_factory=utcnow)
    is_known: bool = True
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", ensure_aware(self.as_of, field="account.as_of"))

    @classmethod
    def unknown(cls, error: str) -> "AccountSnapshot":
        """An explicitly unknown account. Every risk check must refuse this."""
        return cls(
            equity=0.0, cash=0.0, buying_power=0.0, is_paper=None,
            is_known=False, error=error,
        )

    @property
    def daily_pl(self) -> float:
        """Change in equity since the start of the trading day."""
        if not self.is_known or self.last_equity <= 0:
            return 0.0
        return self.equity - self.last_equity

    @property
    def daily_pl_pct(self) -> float:
        if not self.is_known or self.last_equity <= 0:
            return 0.0
        return (self.daily_pl / self.last_equity) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_known": self.is_known,
            "equity": self.equity,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "is_paper": self.is_paper,
            "trading_blocked": self.trading_blocked,
            "daily_pl": round(self.daily_pl, 2),
            "daily_pl_pct": round(self.daily_pl_pct, 4),
            "daytrade_count": self.daytrade_count,
            "account": self.account_number_masked,
            "as_of": self.as_of.isoformat(),
            "error": self.error,
        }


@dataclass(frozen=True)
class PortfolioState:
    """Account plus open positions, fetched together."""

    account: AccountSnapshot
    positions: tuple[Position, ...] = ()
    as_of: datetime = field(default_factory=utcnow)

    @classmethod
    def unknown(cls, error: str) -> "PortfolioState":
        return cls(account=AccountSnapshot.unknown(error), positions=())

    @property
    def is_known(self) -> bool:
        return self.account.is_known

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    @property
    def gross_exposure(self) -> float:
        return sum(p.abs_market_value for p in self.positions)

    def position_for(self, symbol: str) -> Position | None:
        target = symbol.strip().upper()
        for position in self.positions:
            if position.symbol.upper() == target:
                return position
        return None

    def exposure_pct_for(self, symbol: str) -> float:
        """This symbol's share of account equity, as a fraction 0-1."""
        position = self.position_for(symbol)
        if position is None or not self.account.is_known or self.account.equity <= 0:
            return 0.0
        return position.abs_market_value / self.account.equity

    @classmethod
    def from_parts(
        cls, account: AccountSnapshot, positions: Iterable[Position]
    ) -> "PortfolioState":
        return cls(account=account, positions=tuple(positions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "account": self.account.to_dict(),
            "positions": [p.to_dict() for p in self.positions],
            "open_position_count": self.open_position_count,
            "gross_exposure": round(self.gross_exposure, 2),
            "as_of": self.as_of.isoformat(),
        }
