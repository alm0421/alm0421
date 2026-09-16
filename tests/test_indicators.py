"""Indicator correctness and lookahead safety."""

from __future__ import annotations

import pytest

from app.signals import indicators as ind

RISING = [float(x) for x in range(1, 41)]
CHOPPY = [10.0, 11, 12, 11, 10, 9, 10, 12, 14, 13, 12, 11, 13, 15, 16, 15, 14, 16, 18, 17,
          16, 18, 20, 19, 18, 17, 19, 21, 22, 21, 20, 22, 24, 23, 22, 21, 23, 25, 26, 25]


def test_sma_known_values():
    assert ind.sma([1.0, 2, 3, 4, 5], 3) == [None, None, 2.0, 3.0, 4.0]


def test_ema_seeded_with_sma():
    result = ind.ema([1.0, 2, 3, 4, 5], 3)
    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(2.0)  # SMA seed
    assert result[3] == pytest.approx((4 - 2) * 0.5 + 2)


def test_rsi_is_100_for_monotonic_rise():
    assert ind.rsi(RISING, 14)[-1] == pytest.approx(100.0)


def test_rsi_is_zero_for_monotonic_fall():
    assert ind.rsi(list(reversed(RISING)), 14)[-1] == pytest.approx(0.0)


def test_rsi_bounded_0_to_100():
    for value in ind.rsi(CHOPPY, 14):
        if value is not None:
            assert 0.0 <= value <= 100.0


def test_atr_constant_range():
    """A series with a constant 2.0 true range must produce ATR of exactly 2.0."""
    closes = [100.0] * 30
    highs = [101.0] * 30
    lows = [99.0] * 30
    assert ind.atr(highs, lows, closes, 14)[-1] == pytest.approx(2.0)


def test_relative_volume_excludes_current_bar():
    """The current bar must not dampen its own baseline."""
    volumes = [100.0] * 20 + [300.0]
    assert ind.relative_volume(volumes, 20)[-1] == pytest.approx(3.0)


def test_percent_change():
    assert ind.percent_change([100.0, 110.0], 1)[-1] == pytest.approx(10.0)


def test_rolling_high_low():
    values = [1.0, 5, 3, 2, 7]
    assert ind.rolling_high(values, 3)[-1] == 7.0
    assert ind.rolling_low(values, 3)[-1] == 2.0


def test_last_value_skips_none():
    assert ind.last_value([None, None, 3.0]) == 3.0
    assert ind.last_value([None, None]) is None


# --- lookahead safety -------------------------------------------------------


@pytest.mark.parametrize("func,period", [(ind.sma, 5), (ind.ema, 5), (ind.rsi, 14)])
def test_output_length_matches_input(func, period):
    assert len(func(CHOPPY, period)) == len(CHOPPY)


@pytest.mark.parametrize("func,period", [(ind.sma, 5), (ind.ema, 5), (ind.rsi, 14)])
def test_prefix_stability_proves_no_lookahead(func, period):
    """Truncating the future must not change any past value.

    If an indicator consulted a later bar, the truncated series would produce
    different values at the same indices. This is the strongest available
    property test for lookahead leakage.
    """
    full = func(CHOPPY, period)
    for cut in (20, 25, 30, 35):
        truncated = func(CHOPPY[:cut], period)
        assert truncated == full[:cut], f"{func.__name__} leaked future data at cut={cut}"


def test_atr_prefix_stability():
    highs = [c + 1 for c in CHOPPY]
    lows = [c - 1 for c in CHOPPY]
    full = ind.atr(highs, lows, CHOPPY, 14)
    for cut in (20, 25, 30):
        truncated = ind.atr(highs[:cut], lows[:cut], CHOPPY[:cut], 14)
        assert truncated == full[:cut], f"ATR leaked future data at cut={cut}"


def test_relative_volume_prefix_stability():
    volumes = [100.0 + (i % 7) * 20 for i in range(40)]
    full = ind.relative_volume(volumes, 20)
    for cut in (25, 30, 35):
        assert ind.relative_volume(volumes[:cut], 20) == full[:cut]


# --- input validation -------------------------------------------------------


@pytest.mark.parametrize("period", [0, -1])
def test_non_positive_period_rejected(period):
    with pytest.raises(ValueError, match="positive"):
        ind.sma([1.0, 2.0], period)


def test_warmup_returns_none_not_garbage():
    assert ind.ema([1.0, 2.0], 10) == [None, None]
    assert ind.rsi([1.0, 2.0], 14) == [None, None]


def test_true_range_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal length"):
        ind.true_range([1.0, 2.0], [1.0], [1.0, 2.0])
