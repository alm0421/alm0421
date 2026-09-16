"""Structural order validation."""

from __future__ import annotations

import dataclasses

import pytest

from app.enums import (
    AssetClass,
    OrderSide,
    OrderType,
    TimeInForce,
    TradingMode,
)
from app.execution.models import OrderIntent, OrderStatus, RejectionReason
from app.execution.validation import validate_order_intent


def make_intent(**overrides) -> OrderIntent:
    defaults = dict(
        symbol="AAPL",
        asset_class=AssetClass.US_EQUITY,
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        mode=TradingMode.SIGNAL_ONLY,
        reference_price=100.0,
        stop_loss_price=98.0,
        take_profit_price=106.0,
    )
    defaults.update(overrides)
    return OrderIntent(**defaults)


def reasons(issues) -> set[RejectionReason]:
    return {i.reason for i in issues}


def test_valid_order_passes(config):
    assert validate_order_intent(make_intent(), config) == []


def test_futures_always_rejected(config):
    issues = validate_order_intent(
        make_intent(symbol="ESZ6", asset_class=AssetClass.FUTURES), config
    )
    assert RejectionReason.UNSUPPORTED_ASSET_CLASS in reasons(issues)
    assert any("futures" in i.message.lower() for i in issues)


def test_disallowed_asset_class_rejected(config):
    issues = validate_order_intent(
        make_intent(symbol="AAPL250117C00150000", asset_class=AssetClass.US_OPTION), config
    )
    assert RejectionReason.UNSUPPORTED_ASSET_CLASS in reasons(issues)


@pytest.mark.parametrize("quantity", [0, -5, -0.001])
def test_non_positive_quantity_rejected(config, quantity):
    issues = validate_order_intent(make_intent(quantity=quantity), config)
    assert RejectionReason.INVALID_ORDER in reasons(issues)


def test_limit_order_without_limit_price_rejected(config):
    issues = validate_order_intent(
        make_intent(order_type=OrderType.LIMIT, limit_price=None), config
    )
    assert any("limit_price" in i.message for i in issues)


def test_stop_order_without_stop_price_rejected(config):
    issues = validate_order_intent(
        make_intent(order_type=OrderType.STOP, stop_price=None), config
    )
    assert any("stop_price" in i.message for i in issues)


def test_trailing_stop_requires_valid_percent(config):
    for bad in (None, 0, 100, 150):
        issues = validate_order_intent(
            make_intent(order_type=OrderType.TRAILING_STOP, trail_percent=bad), config
        )
        assert any("trail_percent" in i.message for i in issues), bad


def test_inverted_buy_bracket_rejected(config):
    """A BUY stop above entry would trigger the instant it is placed."""
    issues = validate_order_intent(make_intent(stop_loss_price=105.0), config)
    assert any("trigger immediately" in i.message for i in issues)


def test_buy_take_profit_below_entry_rejected(config):
    issues = validate_order_intent(make_intent(take_profit_price=95.0), config)
    assert any("take-profit" in i.message for i in issues)


def test_inverted_sell_bracket_rejected(config):
    issues = validate_order_intent(
        make_intent(side=OrderSide.SELL, stop_loss_price=95.0, take_profit_price=90.0), config
    )
    assert any("trigger immediately" in i.message for i in issues)


def test_valid_sell_bracket_passes(config):
    assert (
        validate_order_intent(
            make_intent(side=OrderSide.SELL, stop_loss_price=102.0, take_profit_price=94.0),
            config,
        )
        == []
    )


def test_crypto_day_order_rejected(config):
    """Alpaca requires GTC/IOC for crypto; catch it before the 422."""
    issues = validate_order_intent(
        make_intent(
            symbol="BTC/USD", asset_class=AssetClass.CRYPTO,
            time_in_force=TimeInForce.DAY, quantity=0.01,
            reference_price=60000.0, stop_loss_price=58000.0, take_profit_price=64000.0,
        ),
        config,
    )
    assert any("time in force" in i.message for i in issues)


def test_crypto_gtc_order_passes(config):
    assert (
        validate_order_intent(
            make_intent(
                symbol="BTC/USD", asset_class=AssetClass.CRYPTO,
                time_in_force=TimeInForce.GTC, quantity=0.01,
                reference_price=60000.0, stop_loss_price=58000.0, take_profit_price=64000.0,
            ),
            config,
        )
        == []
    )


def test_extended_hours_blocked_by_config(config):
    issues = validate_order_intent(
        make_intent(extended_hours=True, order_type=OrderType.LIMIT, limit_price=100.0), config
    )
    assert RejectionReason.EXTENDED_HOURS_NOT_ALLOWED in reasons(issues)


def test_extended_hours_market_order_rejected(config):
    """A market order outside regular hours can fill at an arbitrary price."""
    allow = dataclasses.replace(config, risk=dataclasses.replace(config.risk, allow_extended_hours=True))
    issues = validate_order_intent(make_intent(extended_hours=True), allow)
    assert any("must be LIMIT" in i.message for i in issues)


def test_mode_mismatch_rejected(config):
    """An intent created in one mode must not be replayed into another."""
    issues = validate_order_intent(make_intent(mode=TradingMode.LIVE), config)
    assert RejectionReason.MODE_NOT_EXECUTABLE in reasons(issues)


def test_all_issues_collected_not_just_the_first(config):
    issues = validate_order_intent(
        make_intent(quantity=-1, order_type=OrderType.LIMIT, limit_price=None), config
    )
    assert len(issues) >= 2


def test_order_status_classification():
    assert OrderStatus.RECORDED_INTENT.reached_broker is False
    assert OrderStatus.FILLED.reached_broker is True
    assert OrderStatus.REJECTED_LOCALLY.reached_broker is False
    assert OrderStatus.UNKNOWN.reached_broker is True
