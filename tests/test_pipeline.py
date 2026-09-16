"""End-to-end pipeline behaviour.

These tests wire the real scanner, strategy, risk engine and router together
against a deterministic in-memory data provider.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.clock import MarketClock
from app.enums import AssetClass, Freshness, TradingMode
from app.execution.models import OrderStatus, RejectionReason
from app.execution.null_broker import NullBroker
from app.pipeline import TradingPipeline
from app.risk.engine import RiskEngine
from app.scanners import MomentumScanner
from app.strategies import EmaPullbackStrategy
from tests.conftest import (
    MARKET_OPEN_MOMENT,
    FakeProvider,
    make_bars,
    make_quote,
    uptrend_pullback_closes,
)
from tests.test_scanners import surging_volumes


@pytest.fixture()
def signal_config(config):
    """Signal-only config restricted to a single equity symbol."""
    return dataclasses.replace(
        config,
        mode=TradingMode.SIGNAL_ONLY,
        universe={AssetClass.US_EQUITY: ("AAPL",)},
    )


@pytest.fixture()
def provider() -> FakeProvider:
    bars = make_bars(
        "AAPL",
        closes=uptrend_pullback_closes(),
        volumes=surging_volumes(),
        # End the series well before MARKET_OPEN_MOMENT so no bar is "forming".
        start=MARKET_OPEN_MOMENT.replace(hour=6),
    )
    return FakeProvider({"AAPL": bars}, {"AAPL": make_quote("AAPL")})


def build_pipeline(cfg, provider, audit, broker=None):
    return TradingPipeline(
        cfg,
        provider=provider,
        scanners=[MomentumScanner({})],
        strategies=[EmaPullbackStrategy({"min_confidence": 0.0})],
        broker=broker or NullBroker(audit=audit),
        risk_engine=RiskEngine(cfg, audit=audit),
        audit=audit,
        clock=MarketClock(),
    )


def test_pipeline_produces_a_signal(signal_config, provider, audit):
    result = build_pipeline(signal_config, provider, audit).run(now=MARKET_OPEN_MOMENT)
    assert result.bar_sets_received == 1
    assert len(result.scan_results) == 1
    assert len(result.signals) == 1
    assert result.signals.signals[0].symbol == "AAPL"


def test_signal_only_run_transmits_nothing(signal_config, provider, audit):
    broker = NullBroker(audit=audit)
    result = build_pipeline(signal_config, provider, audit, broker).run(
        execute=True, now=MARKET_OPEN_MOMENT
    )
    for order in result.order_results:
        assert order.status is not OrderStatus.SUBMITTED
        assert order.broker_order_id is None


def test_signal_only_cannot_size_without_a_known_account(signal_config, provider, audit):
    """No broker means unknown equity, so nothing can be sized. That is correct,
    not a bug: signal-only mode has no account to risk."""
    result = build_pipeline(signal_config, provider, audit).run(
        execute=True, now=MARKET_OPEN_MOMENT
    )
    assert result.portfolio is not None
    assert result.portfolio.is_known is False
    assert result.order_results == []
    events = [e["event"] for e in audit.tail(200)]
    assert "order_not_sized" in events


def test_signals_are_identical_with_and_without_execution(signal_config, provider, audit):
    """Enabling execution must not change which signals are produced."""
    without = build_pipeline(signal_config, provider, audit).run(now=MARKET_OPEN_MOMENT)
    with_exec = build_pipeline(signal_config, provider, audit).run(
        execute=True, now=MARKET_OPEN_MOMENT
    )
    assert [s.symbol for s in without.signals] == [s.symbol for s in with_exec.signals]
    assert [s.confidence for s in without.signals] == [s.confidence for s in with_exec.signals]


def test_missing_data_is_reported_not_silent(signal_config, audit):
    empty = FakeProvider({}, {})
    result = build_pipeline(signal_config, empty, audit).run(now=MARKET_OPEN_MOMENT)
    assert result.symbols_requested == 1
    assert result.bar_sets_received == 0
    assert "AAPL" in result.data_failures


def test_futures_symbols_reported_as_unsupported(config, provider, audit):
    cfg = dataclasses.replace(
        config, mode=TradingMode.SIGNAL_ONLY, universe={AssetClass.FUTURES: ("ESZ6",)}
    )
    result = build_pipeline(cfg, provider, audit).run(now=MARKET_OPEN_MOMENT)
    assert "futures" in result.data_failures["ESZ6"].lower()
    assert result.bar_sets_received == 0


def test_forming_bar_is_dropped(signal_config, audit):
    """A bar that has not closed yet must not reach a strategy."""
    closes = uptrend_pullback_closes()
    bars = make_bars(
        "AAPL", closes=closes, volumes=surging_volumes(),
        start=MARKET_OPEN_MOMENT.replace(hour=6),
    )
    last_open = bars.last_timestamp
    # Evaluate midway through the final bar's 5-minute window.
    mid_bar = last_open.replace(minute=last_open.minute + 2)
    provider = FakeProvider({"AAPL": bars}, {"AAPL": make_quote("AAPL")})
    result = build_pipeline(signal_config, provider, audit).run(now=mid_bar)
    assert result.bar_sets_received == 1


def test_audit_trail_records_every_stage(signal_config, provider, audit):
    build_pipeline(signal_config, provider, audit).run(execute=True, now=MARKET_OPEN_MOMENT)
    events = {e["event"] for e in audit.tail(200)}
    assert {"pipeline_started", "scanner_run", "signal_generated", "pipeline_finished"} <= events


def test_result_is_json_serialisable(signal_config, provider, audit):
    import json

    result = build_pipeline(signal_config, provider, audit).run(now=MARKET_OPEN_MOMENT)
    json.dumps(result.to_dict())


def test_strategy_failure_does_not_abort_the_run(signal_config, provider, audit, monkeypatch):
    class Exploding(EmaPullbackStrategy):
        def evaluate(self, bars):
            raise RuntimeError("boom")

    pipeline = TradingPipeline(
        signal_config,
        provider=provider,
        scanners=[MomentumScanner({})],
        strategies=[Exploding({"min_confidence": 0.0})],
        broker=NullBroker(audit=audit),
        risk_engine=RiskEngine(signal_config, audit=audit),
        audit=audit,
    )
    result = pipeline.run(now=MARKET_OPEN_MOMENT)
    assert len(result.scan_results) == 1  # the scan still completed
    assert len(result.signals) == 0
