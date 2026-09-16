"""Risk-based position sizing."""

from __future__ import annotations

import pytest

from app.enums import AssetClass, Freshness, SignalDirection
from app.risk.sizing import size_position
from app.signals.models import Signal
from tests.conftest import MARKET_OPEN_MOMENT


def make_signal(
    *, price: float = 100.0, stop: float | None = 98.0,
    asset_class: AssetClass = AssetClass.US_EQUITY,
) -> Signal:
    target = price * 1.06 if asset_class is AssetClass.US_EQUITY else price * 1.06
    return Signal(
        symbol="AAPL" if asset_class is AssetClass.US_EQUITY else "BTC/USD",
        asset_class=asset_class,
        direction=SignalDirection.LONG,
        strategy="test",
        confidence=0.7,
        reference_price=price,
        timeframe="5Min",
        data_freshness=Freshness.HISTORICAL,
        data_timestamp=MARKET_OPEN_MOMENT,
        stop_price=stop,
        target_price=target,
    )


def test_risk_budget_binds_on_a_small_account(config):
    """1% of 10,000 = 100 risk budget / 2.00 per share = 50 shares, but the
    10%-of-equity cap (10 shares) is tighter and must win."""
    sized = size_position(make_signal(), equity=10_000, risk_config=config.risk)
    assert sized.quantity == 10
    assert sized.binding_constraint == "max_position_pct_equity"


def test_notional_cap_binds_on_a_large_account(config):
    sized = size_position(make_signal(), equity=1_000_000, risk_config=config.risk)
    assert sized.notional <= config.risk.max_position_notional
    assert sized.binding_constraint == "max_position_notional"


def test_no_stop_means_no_size(config):
    """Without a stop there is no defined risk, so no size can be derived."""
    sized = size_position(make_signal(stop=None), equity=100_000, risk_config=config.risk)
    assert sized.quantity == 0
    assert sized.is_tradable is False
    assert sized.binding_constraint == "no_stop_loss"


def test_unknown_equity_means_no_size(config):
    sized = size_position(make_signal(), equity=0.0, risk_config=config.risk)
    assert sized.quantity == 0
    assert sized.binding_constraint == "unknown_or_zero_equity"


def test_account_too_small_returns_zero_not_a_fraction(config):
    """Equities are whole-share; a sub-one-share result must be zero."""
    sized = size_position(
        make_signal(price=5000.0, stop=4900.0), equity=1_000, risk_config=config.risk
    )
    assert sized.quantity == 0
    assert "below_one_unit" in sized.binding_constraint


def test_equity_quantity_is_always_whole(config):
    for equity in (10_000, 37_531, 250_000):
        sized = size_position(make_signal(), equity=equity, risk_config=config.risk)
        assert sized.quantity == int(sized.quantity)


def test_crypto_allows_fractional(config):
    sized = size_position(
        make_signal(price=60_000.0, stop=58_800.0, asset_class=AssetClass.CRYPTO),
        equity=10_000, risk_config=config.risk,
    )
    assert 0 < sized.quantity < 1


def test_quantity_never_exceeds_any_cap(config):
    """Property check across a wide range of accounts and prices."""
    for equity in (1_000, 10_000, 100_000, 1_000_000):
        for price in (10.0, 100.0, 750.0):
            signal = make_signal(price=price, stop=price * 0.98)
            sized = size_position(signal, equity=equity, risk_config=config.risk)
            if sized.quantity == 0:
                continue
            assert sized.notional <= config.risk.max_position_notional + 1e-6
            assert sized.notional <= equity * config.risk.max_position_pct_equity + 1e-6


def test_risk_amount_matches_stop_distance(config):
    sized = size_position(make_signal(), equity=100_000, risk_config=config.risk)
    assert sized.risk_amount == pytest.approx(sized.quantity * 2.0)


def test_rounding_is_floor_never_ceiling(config):
    """Rounding up could breach the cap that was binding."""
    sized = size_position(
        make_signal(price=333.0, stop=330.0), equity=100_000, risk_config=config.risk
    )
    assert sized.notional <= config.risk.max_position_notional
