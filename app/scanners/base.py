"""Scanner interface and registry.

A scanner reduces a universe to a ranked shortlist. It never produces orders and
never consults the broker; its only job is to decide which symbols are worth
spending strategy evaluation on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from app.clock import utcnow
from app.data.base import BarSet
from app.enums import AssetClass, Freshness


class ScannerError(RuntimeError):
    """Raised when a scanner is misconfigured."""


@dataclass(frozen=True)
class ScanResult:
    """One symbol that passed a scanner's filters."""

    symbol: str
    asset_class: AssetClass
    scanner: str
    #: Composite rank score. Only comparable within the same scanner.
    score: float
    last_price: float
    #: The filter values that produced the score, for dashboard display.
    metrics: dict[str, Any] = field(default_factory=dict)
    data_freshness: Freshness = Freshness.UNKNOWN
    data_timestamp: datetime | None = None
    scanned_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class.value,
            "scanner": self.scanner,
            "score": round(self.score, 6),
            "last_price": self.last_price,
            "data_freshness": self.data_freshness.value,
            "data_timestamp": self.data_timestamp.isoformat() if self.data_timestamp else None,
            "scanned_at": self.scanned_at.isoformat(),
            **{f"metric_{k}": v for k, v in self.metrics.items()},
        }


class Scanner(ABC):
    """Base class for scanners."""

    name: str = "base"

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params = dict(params or {})
        self._validate_params()

    def _validate_params(self) -> None:
        """Override to reject bad parameters at construction time."""

    @property
    @abstractmethod
    def min_bars(self) -> int:
        """Minimum bars required to evaluate one symbol."""

    @abstractmethod
    def evaluate_symbol(self, bars: BarSet) -> ScanResult | None:
        """Score a single symbol, or return None if it fails the filters."""

    def scan(self, bar_sets: Mapping[str, BarSet], *, limit: int | None = None) -> list[ScanResult]:
        """Evaluate every symbol and return results ranked highest score first.

        A symbol that raises is skipped and does not abort the scan - one bad
        symbol must not blind the operator to the rest of the universe.
        """
        results: list[ScanResult] = []
        for symbol, bars in bar_sets.items():
            if len(bars) < self.min_bars:
                continue
            try:
                result = self.evaluate_symbol(bars)
            except Exception:  # noqa: BLE001 - one symbol must not kill the scan
                from app.logging import get_logger

                get_logger(__name__).exception(
                    "Scanner %s failed on %s", self.name, symbol, extra={"symbol": symbol}
                )
                continue
            if result is not None:
                results.append(result)

        # Deterministic ordering: score desc, then symbol asc for stable ties.
        results.sort(key=lambda r: (-r.score, r.symbol))
        return results[:limit] if limit is not None else results


_REGISTRY: dict[str, type[Scanner]] = {}


def register(scanner_cls: type[Scanner]) -> type[Scanner]:
    _REGISTRY[scanner_cls.name] = scanner_cls
    return scanner_cls


def available_scanners() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_scanners(enabled: tuple[str, ...], params: Mapping[str, Any] | None = None) -> list[Scanner]:
    params = params or {}
    built: list[Scanner] = []
    for name in enabled:
        scanner_cls = _REGISTRY.get(name)
        if scanner_cls is None:
            raise ScannerError(
                f"Unknown scanner {name!r}. Available: {', '.join(available_scanners()) or 'none'}"
            )
        built.append(scanner_cls(params.get(name, {})))
    return built
