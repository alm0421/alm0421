"""Mode separation - the most safety-critical behaviour in the platform.

These tests exist to fail loudly if anyone ever makes it possible to reach a
live broker without every gate being deliberately opened.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.config import Credentials
from app.enums import TradingMode
from app.execution.live_gate import (
    LIVE_ACK_ENV_VAR,
    LIVE_ACK_PHRASE,
    LIVE_TRADING_APPROVED,
    LiveTradingDisabled,
    assert_live_trading_allowed,
    live_gate_status,
)
from app.execution.null_broker import NullBroker
from app.execution.router import ExecutionRouterError, build_broker


# --- the build gate ---------------------------------------------------------


def test_build_gate_is_closed_in_this_release():
    """Live trading has never been validated; the build gate must ship closed."""
    assert LIVE_TRADING_APPROVED is False


def test_live_blocked_even_with_config_and_env_and_credentials(config, monkeypatch):
    """Opening every gate EXCEPT the build gate must still block.

    This is the whole point of a three-gate design: a leaked .env plus an edited
    config must not be sufficient.
    """
    monkeypatch.setenv(LIVE_ACK_ENV_VAR, LIVE_ACK_PHRASE)
    live_config = dataclasses.replace(
        config,
        mode=TradingMode.LIVE,
        execution=dataclasses.replace(config.execution, live_trading_enabled=True),
        _credentials=Credentials(api_key="live-key", secret_key="live-secret", paper=False),
    )
    with pytest.raises(LiveTradingDisabled, match="build gate"):
        assert_live_trading_allowed(live_config)


def test_live_blocked_without_runtime_ack(config, monkeypatch):
    monkeypatch.delenv(LIVE_ACK_ENV_VAR, raising=False)
    live_config = dataclasses.replace(
        config,
        mode=TradingMode.LIVE,
        execution=dataclasses.replace(config.execution, live_trading_enabled=True),
    )
    with pytest.raises(LiveTradingDisabled, match="runtime gate"):
        assert_live_trading_allowed(live_config)


def test_live_gate_status_reports_all_gates_closed(config):
    status = live_gate_status(config)
    assert status["all_gates_open"] is False
    assert status["build_approved"] is False


def test_live_gate_status_never_raises(config):
    """Status is a query; it must be safe to call from the dashboard."""
    assert isinstance(live_gate_status(config), dict)


# --- routing ----------------------------------------------------------------


def test_signal_only_routes_to_null_broker(config, audit):
    broker = build_broker(config, audit=audit)
    assert isinstance(broker, NullBroker)
    assert broker.submits_real_orders is False
    assert broker.mode is TradingMode.SIGNAL_ONLY


def test_backtest_mode_refuses_rather_than_falling_back(config, audit):
    """A mode with no engine must fail, not silently become another mode."""
    backtest_config = dataclasses.replace(config, mode=TradingMode.BACKTEST)
    with pytest.raises(ExecutionRouterError, match="no execution route"):
        build_broker(backtest_config, audit=audit)


def test_live_mode_router_blocks(config, audit, monkeypatch):
    monkeypatch.setenv(LIVE_ACK_ENV_VAR, LIVE_ACK_PHRASE)
    live_config = dataclasses.replace(
        config,
        mode=TradingMode.LIVE,
        execution=dataclasses.replace(config.execution, live_trading_enabled=True),
        _credentials=Credentials(api_key="k", secret_key="s", paper=False),
    )
    with pytest.raises(LiveTradingDisabled):
        build_broker(live_config, audit=audit)


def test_paper_mode_requires_real_credentials(config, audit, monkeypatch):
    """Paper mode must not silently fall back to the null broker."""
    from app.execution.base import BrokerError

    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
    paper_config = dataclasses.replace(
        config,
        mode=TradingMode.PAPER,
        _credentials=Credentials(api_key=None, secret_key=None, paper=True),
    )
    with pytest.raises(BrokerError, match="credentials are not configured"):
        build_broker(paper_config, audit=audit)


# --- the null broker transmits nothing --------------------------------------


def test_null_broker_records_but_never_transmits(config, audit):
    from tests.test_execution_validation import make_intent

    broker = NullBroker(audit=audit)
    result = broker.submit_order(make_intent())

    assert result.is_simulated is True
    assert result.succeeded is False
    assert result.broker_order_id is None
    assert result.submitted_at is None
    assert result.status.reached_broker is False


def test_null_broker_reports_account_as_unknown_not_empty(audit):
    """An unknown account and an empty account are different; never conflate them."""
    portfolio = NullBroker(audit=audit).get_portfolio()
    assert portfolio.is_known is False
    assert portfolio.account.equity == 0.0
    assert portfolio.account.error


def test_null_broker_reports_nothing_tradable(audit):
    from app.enums import AssetClass

    assert NullBroker(audit=audit).is_symbol_tradable("AAPL", AssetClass.US_EQUITY) is False


def test_audit_trail_records_non_transmission(config, audit):
    from tests.test_execution_validation import make_intent

    NullBroker(audit=audit).submit_order(make_intent())
    entries = audit.tail()
    assert entries[-1]["event"] == "order_intent_recorded"
    assert entries[-1]["payload"]["transmitted"] is False
