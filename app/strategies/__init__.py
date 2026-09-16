"""Strategy evaluation.

A strategy consumes a :class:`~app.data.base.BarSet` and produces at most one
:class:`~app.signals.models.Signal`. Strategies are pure: they hold no broker
handle, place no orders, and perform no I/O. That separation is what lets the
same strategy code be driven by a backtest engine and by live data without
behavioural drift.
"""

from app.strategies.base import Strategy, StrategyError, build_strategies
from app.strategies.ema_pullback import EmaPullbackStrategy

__all__ = ["Strategy", "StrategyError", "EmaPullbackStrategy", "build_strategies"]
