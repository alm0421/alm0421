"""The signal schema.

A :class:`Signal` is the single hand-off object between strategy evaluation and
everything downstream (ranking, persistence, display, risk, execution). It
carries the provenance of the data that produced it - freshness label and data
timestamp - so the risk engine can refuse to act on a signal derived from stale
or unlabelled data without having to re-fetch anything.

Signals are immutable. A changed view of the market produces a new signal.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Iterator

from app.clock import ensure_aware, utcnow
from app.enums import AssetClass, Freshness, SignalDirection


@dataclass(frozen=True)
class Signal:
    """A directional trade idea produced by one strategy for one symbol."""

    symbol: str
    asset_class: AssetClass
    direction: SignalDirection
    strategy: str
    #: 0.0-1.0. Strategy-relative conviction; only comparable within a strategy.
    confidence: float
    #: Price the signal was computed against (usually the last bar close).
    reference_price: float
    #: Bar timeframe the signal was computed on, e.g. "5Min".
    timeframe: str
    #: Freshness of the data the signal was derived from.
    data_freshness: Freshness
    #: Timestamp of the most recent data point used.
    data_timestamp: datetime
    stop_price: float | None = None
    target_price: float | None = None
    #: Human-readable reasons plus the indicator values behind the decision.
    rationale: dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=utcnow)
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Signal confidence must be within 0.0-1.0, got {self.confidence}"
            )
        if self.reference_price <= 0:
            raise ValueError(f"Signal reference_price must be positive, got {self.reference_price}")
        object.__setattr__(
            self, "data_timestamp", ensure_aware(self.data_timestamp, field="signal.data_timestamp")
        )
        object.__setattr__(
            self, "generated_at", ensure_aware(self.generated_at, field="signal.generated_at")
        )
        object.__setattr__(self, "symbol", self.symbol.strip().upper())

        # A long signal's stop below and target above (and vice versa) is an
        # invariant, not a preference: an inverted bracket would place a stop
        # that fills instantly.
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive when supplied")
        if self.target_price is not None and self.target_price <= 0:
            raise ValueError("target_price must be positive when supplied")
        if self.direction is SignalDirection.LONG:
            if self.stop_price is not None and self.stop_price >= self.reference_price:
                raise ValueError(
                    f"LONG signal for {self.symbol}: stop {self.stop_price} must be "
                    f"below reference price {self.reference_price}"
                )
            if self.target_price is not None and self.target_price <= self.reference_price:
                raise ValueError(
                    f"LONG signal for {self.symbol}: target {self.target_price} must be "
                    f"above reference price {self.reference_price}"
                )
        elif self.direction is SignalDirection.SHORT:
            if self.stop_price is not None and self.stop_price <= self.reference_price:
                raise ValueError(
                    f"SHORT signal for {self.symbol}: stop {self.stop_price} must be "
                    f"above reference price {self.reference_price}"
                )
            if self.target_price is not None and self.target_price >= self.reference_price:
                raise ValueError(
                    f"SHORT signal for {self.symbol}: target {self.target_price} must be "
                    f"below reference price {self.reference_price}"
                )

    @property
    def risk_per_share(self) -> float | None:
        """Absolute distance from entry to stop, used for position sizing."""
        if self.stop_price is None:
            return None
        return abs(self.reference_price - self.stop_price)

    @property
    def reward_risk_ratio(self) -> float | None:
        risk = self.risk_per_share
        if risk is None or risk == 0 or self.target_price is None:
            return None
        return abs(self.target_price - self.reference_price) / risk

    @property
    def is_actionable(self) -> bool:
        """A FLAT signal is informational; only LONG/SHORT can become an order."""
        return self.direction in (SignalDirection.LONG, SignalDirection.SHORT)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["asset_class"] = self.asset_class.value
        payload["direction"] = self.direction.value
        payload["data_freshness"] = self.data_freshness.value
        payload["data_timestamp"] = self.data_timestamp.isoformat()
        payload["generated_at"] = self.generated_at.isoformat()
        payload["risk_per_share"] = self.risk_per_share
        payload["reward_risk_ratio"] = self.reward_risk_ratio
        return payload


@dataclass(frozen=True)
class SignalSet:
    """An ordered collection of signals from one evaluation pass."""

    signals: tuple[Signal, ...]
    generated_at: datetime = field(default_factory=utcnow)

    @classmethod
    def from_iterable(cls, signals: Iterable[Signal]) -> "SignalSet":
        return cls(signals=tuple(signals))

    def __len__(self) -> int:
        return len(self.signals)

    def __iter__(self) -> Iterator[Signal]:
        return iter(self.signals)

    def actionable(self) -> "SignalSet":
        return SignalSet(tuple(s for s in self.signals if s.is_actionable))

    def ranked(self, limit: int | None = None) -> "SignalSet":
        """Highest confidence first; ties broken by newest data, then symbol.

        The tiebreak is deterministic so that two runs over the same data
        produce the same ordering - a reproducibility requirement.
        """
        ordered = sorted(
            self.signals,
            key=lambda s: (-s.confidence, -s.data_timestamp.timestamp(), s.symbol),
        )
        return SignalSet(tuple(ordered[:limit] if limit is not None else ordered))

    def to_records(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.signals]
