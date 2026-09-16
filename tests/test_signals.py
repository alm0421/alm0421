"""Signal schema invariants."""

from __future__ import annotations

import pytest

from app.enums import AssetClass, Freshness, SignalDirection
from app.signals.models import Signal, SignalSet
from tests.conftest import MARKET_OPEN_MOMENT


def make(**overrides) -> Signal:
    defaults = dict(
        symbol="AAPL",
        asset_class=AssetClass.US_EQUITY,
        direction=SignalDirection.LONG,
        strategy="test",
        confidence=0.7,
        reference_price=100.0,
        timeframe="5Min",
        data_freshness=Freshness.HISTORICAL,
        data_timestamp=MARKET_OPEN_MOMENT,
        stop_price=98.0,
        target_price=106.0,
    )
    defaults.update(overrides)
    return Signal(**defaults)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_must_be_a_fraction(confidence):
    with pytest.raises(ValueError, match="confidence"):
        make(confidence=confidence)


def test_reference_price_must_be_positive():
    with pytest.raises(ValueError, match="reference_price"):
        make(reference_price=0)


def test_long_stop_above_entry_rejected():
    """An inverted bracket would fill its own stop immediately."""
    with pytest.raises(ValueError, match="must be.*below"):
        make(stop_price=105.0)


def test_long_target_below_entry_rejected():
    with pytest.raises(ValueError, match="must be.*above"):
        make(target_price=95.0)


def test_short_stop_below_entry_rejected():
    with pytest.raises(ValueError, match="must be.*above"):
        make(direction=SignalDirection.SHORT, stop_price=95.0, target_price=90.0)


def test_valid_short_bracket_accepted():
    signal = make(direction=SignalDirection.SHORT, stop_price=102.0, target_price=94.0)
    assert signal.risk_per_share == pytest.approx(2.0)
    assert signal.reward_risk_ratio == pytest.approx(3.0)


def test_naive_timestamp_rejected():
    from datetime import datetime

    with pytest.raises(Exception):
        make(data_timestamp=datetime(2026, 9, 16, 15, 0))


def test_symbol_is_normalised():
    assert make(symbol=" aapl ").symbol == "AAPL"


def test_flat_signal_is_not_actionable():
    signal = make(direction=SignalDirection.FLAT, stop_price=None, target_price=None)
    assert signal.is_actionable is False


def test_ranking_is_deterministic_on_ties():
    """Two runs over the same signals must order them identically."""
    signals = [make(symbol=s, confidence=0.7) for s in ("MSFT", "AAPL", "NVDA")]
    first = [s.symbol for s in SignalSet.from_iterable(signals).ranked()]
    second = [s.symbol for s in SignalSet.from_iterable(signals).ranked()]
    assert first == second == ["AAPL", "MSFT", "NVDA"]


def test_ranking_orders_by_confidence():
    signals = [
        make(symbol="LOW", confidence=0.3),
        make(symbol="HIGH", confidence=0.9),
        make(symbol="MID", confidence=0.6),
    ]
    ranked = [s.symbol for s in SignalSet.from_iterable(signals).ranked()]
    assert ranked == ["HIGH", "MID", "LOW"]


def test_actionable_filters_flat():
    signals = [
        make(symbol="A"),
        make(symbol="B", direction=SignalDirection.FLAT, stop_price=None, target_price=None),
    ]
    assert len(SignalSet.from_iterable(signals).actionable()) == 1


def test_to_dict_is_json_serialisable():
    import json

    json.dumps(make().to_dict())
