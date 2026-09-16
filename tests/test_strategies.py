"""Strategy evaluation."""

from __future__ import annotations

import pytest

from app.enums import AssetClass, SignalDirection
from app.strategies import EmaPullbackStrategy
from app.strategies.base import StrategyError, build_strategies
from tests.conftest import flat_closes, make_bars, uptrend_pullback_closes


@pytest.fixture()
def strategy() -> EmaPullbackStrategy:
    return EmaPullbackStrategy({"min_confidence": 0.0})


def test_uptrend_pullback_produces_long_signal(strategy):
    bars = make_bars("AAPL", closes=uptrend_pullback_closes())
    signal = strategy.evaluate(bars)
    assert signal is not None, "the canonical pullback setup should produce a signal"
    assert signal.direction is SignalDirection.LONG
    assert signal.symbol == "AAPL"
    assert signal.strategy == "ema_pullback"


def test_long_signal_has_coherent_bracket(strategy):
    signal = strategy.evaluate(make_bars("AAPL", closes=uptrend_pullback_closes()))
    assert signal.stop_price < signal.reference_price < signal.target_price
    assert signal.reward_risk_ratio == pytest.approx(2.0, rel=0.01)


def test_downtrend_pullback_produces_short_signal(strategy):
    closes = [200.0 - c for c in uptrend_pullback_closes()]
    signal = strategy.evaluate(make_bars("AAPL", closes=closes))
    assert signal is not None
    assert signal.direction is SignalDirection.SHORT
    assert signal.target_price < signal.reference_price < signal.stop_price


def test_flat_market_produces_no_signal(strategy):
    """Zero ATR and no trend must yield nothing, not a degenerate signal."""
    assert strategy.evaluate(make_bars("AAPL", closes=flat_closes(), spread=0.0)) is None


def test_insufficient_bars_produces_no_signal(strategy):
    assert strategy.evaluate(make_bars("AAPL", closes=[100.0] * 10)) is None


def test_min_confidence_filters(strategy):
    bars = make_bars("AAPL", closes=uptrend_pullback_closes())
    assert strategy.evaluate(bars) is not None
    assert EmaPullbackStrategy({"min_confidence": 1.0}).evaluate(bars) is None


def test_allow_short_false_suppresses_shorts():
    closes = [200.0 - c for c in uptrend_pullback_closes()]
    bars = make_bars("AAPL", closes=closes)
    assert EmaPullbackStrategy({"min_confidence": 0.0}).evaluate(bars) is not None
    suppressed = EmaPullbackStrategy({"min_confidence": 0.0, "allow_short": False})
    assert suppressed.evaluate(bars) is None


def test_signal_carries_data_provenance(strategy):
    bars = make_bars("AAPL", closes=uptrend_pullback_closes())
    signal = strategy.evaluate(bars)
    assert signal.data_freshness is bars.freshness
    assert signal.data_timestamp == bars.last_timestamp
    assert signal.rationale["reasons"]
    assert "ema_fast" in signal.rationale["indicators"]


def test_confidence_bounded(strategy):
    signal = strategy.evaluate(make_bars("AAPL", closes=uptrend_pullback_closes()))
    assert 0.0 <= signal.confidence <= 1.0


def test_evaluation_is_deterministic(strategy):
    """The same bars must always yield the same signal - a reproducibility
    requirement for comparing live and simulated runs."""
    bars = make_bars("AAPL", closes=uptrend_pullback_closes())
    first, second = strategy.evaluate(bars), strategy.evaluate(bars)
    assert first.direction == second.direction
    assert first.confidence == second.confidence
    assert first.stop_price == second.stop_price


def test_strategy_does_no_io(strategy):
    """A strategy that touched the network would be non-reproducible."""
    import socket

    original = socket.socket

    def forbidden(*args, **kwargs):
        raise AssertionError("strategy attempted network I/O")

    socket.socket = forbidden
    try:
        strategy.evaluate(make_bars("AAPL", closes=uptrend_pullback_closes()))
    finally:
        socket.socket = original


# --- parameter validation ---------------------------------------------------


@pytest.mark.parametrize(
    "params,match",
    [
        ({"fast_period": 30}, "fast < slow < trend"),
        ({"trend_period": 5}, "fast < slow < trend"),
        ({"target_atr_multiple": 1.0}, "must exceed"),
        ({"stop_atr_multiple": -1}, "positive"),
        ({"rsi_floor": 80}, "rsi_floor < rsi_ceiling"),
        ({"min_confidence": 2.0}, "min_confidence"),
    ],
)
def test_invalid_params_rejected_at_construction(params, match):
    with pytest.raises(StrategyError, match=match):
        EmaPullbackStrategy(params)


def test_unknown_strategy_name_is_a_hard_error():
    """Running fewer strategies than configured would misrepresent the system."""
    with pytest.raises(StrategyError, match="Unknown strategy"):
        build_strategies(("no_such_strategy",), {})


def test_build_from_real_config(config):
    built = build_strategies(config.strategies.enabled, config.strategies.params)
    assert [s.name for s in built] == list(config.strategies.enabled)
