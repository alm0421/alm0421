"""Strategy interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from app.data.base import BarSet
from app.signals.models import Signal


class StrategyError(RuntimeError):
    """Raised when a strategy is misconfigured or asked to do something invalid."""


class Strategy(ABC):
    """Base class for all strategies.

    Contract
    --------
    * ``evaluate`` receives only **completed** bars, oldest first.
    * ``evaluate`` returns a Signal or ``None``. It must never raise for ordinary
      "no setup" conditions; ``None`` is the answer for that.
    * Implementations must not perform I/O or consult wall-clock time, so that a
      given BarSet always yields the same Signal. This is what makes results
      reproducible between live and simulated runs.
    """

    #: Registry key, matching the name used in ``config/config.yaml``.
    name: str = "base"

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params = dict(params or {})
        self._validate_params()

    def _validate_params(self) -> None:
        """Override to reject bad parameters at construction rather than at run time."""

    @property
    @abstractmethod
    def min_bars(self) -> int:
        """Minimum bars required before the strategy can produce a signal."""

    @abstractmethod
    def evaluate(self, bars: BarSet) -> Signal | None:
        """Evaluate the most recent bar and return a Signal, or None."""

    def _param(self, key: str, default: Any) -> Any:
        return self.params.get(key, default)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r}, params={self.params!r})"


#: Populated at import time by ``app.strategies``.
_REGISTRY: dict[str, type[Strategy]] = {}


def register(strategy_cls: type[Strategy]) -> type[Strategy]:
    """Class decorator adding a strategy to the registry."""
    _REGISTRY[strategy_cls.name] = strategy_cls
    return strategy_cls


def available_strategies() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_strategies(
    enabled: tuple[str, ...], params: Mapping[str, Mapping[str, Any]]
) -> list[Strategy]:
    """Instantiate the enabled strategies.

    An unknown strategy name is a hard error: silently running fewer strategies
    than configured would misrepresent what the platform is doing.
    """
    built: list[Strategy] = []
    for name in enabled:
        strategy_cls = _REGISTRY.get(name)
        if strategy_cls is None:
            raise StrategyError(
                f"Unknown strategy {name!r}. Available: {', '.join(available_strategies()) or 'none'}"
            )
        built.append(strategy_cls(params.get(name, {})))
    return built
