"""Pre-trade risk engine.

Every check is fail-closed: an unknown input must reject, never approve.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from app.enums import AssetClass, Freshness, OrderSide, TimeInForce, TradingMode
from app.execution.models import RejectionReason
from app.portfolio.positions import Position, PortfolioState
from app.risk.engine import RiskEngine
from app.risk.limits import RiskDecision, RiskViolation
from tests.conftest import (
    MARKET_CLOSED_MOMENT,
    MARKET_OPEN_MOMENT,
    PREMARKET_MOMENT,
    make_portfolio,
    make_quote,
)
from tests.test_execution_validation import make_intent


@pytest.fixture()
def paper_config(config):
    """Paper mode, so quote-freshness and account checks are enforced."""
    return dataclasses.replace(config, mode=TradingMode.PAPER)


@pytest.fixture()
def engine(paper_config, audit):
    return RiskEngine(paper_config, audit=audit)


def reasons(decision) -> set[RejectionReason]:
    return {v.reason for v in decision.violations}


# --- the happy path ---------------------------------------------------------


def test_clean_order_is_approved(engine):
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert decision.approved, decision.summary
    assert "kill_switch" in decision.checks_passed
    assert "quote_freshness" in decision.checks_passed


# --- kill switch ------------------------------------------------------------


def test_kill_switch_blocks_everything(paper_config, audit, tmp_path):
    switch = tmp_path / "KILL_SWITCH"
    switch.write_text("halt")
    cfg = dataclasses.replace(
        paper_config, risk=dataclasses.replace(paper_config.risk, kill_switch_file=switch)
    )
    engine = RiskEngine(cfg, audit=audit)
    assert engine.kill_switch_engaged()

    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    assert decision.primary_reason is RejectionReason.KILL_SWITCH


# --- fail-closed on unknown state -------------------------------------------


def test_unknown_account_is_rejected(engine):
    """Unreachable broker must never be treated as an empty account."""
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=PortfolioState.unknown("broker unreachable"),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    assert RejectionReason.UNKNOWN_ACCOUNT_STATE in reasons(decision)


def test_unknown_account_mode_is_rejected(engine):
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(is_paper=None),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    assert RejectionReason.UNKNOWN_ACCOUNT_STATE in reasons(decision)


def test_live_account_in_paper_mode_is_rejected(engine):
    """A paper-mode platform connected to a live account is a hard stop."""
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(is_paper=False),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    assert RejectionReason.MODE_NOT_EXECUTABLE in reasons(decision)


def test_broker_trading_block_is_respected(engine):
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(trading_blocked=True),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved


# --- data freshness ---------------------------------------------------------


def test_missing_quote_rejected_in_paper_mode(engine):
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(),
        quote=None,
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    assert RejectionReason.STALE_DATA in reasons(decision)


def test_delayed_data_rejected(engine):
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(),
        quote=make_quote("AAPL", freshness=Freshness.DELAYED),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    assert RejectionReason.STALE_DATA in reasons(decision)


def test_unknown_freshness_rejected(engine):
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(),
        quote=make_quote("AAPL", freshness=Freshness.UNKNOWN),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved


def test_stale_quote_rejected(engine):
    stale = make_quote("AAPL", timestamp=MARKET_OPEN_MOMENT - timedelta(minutes=10))
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(),
        quote=stale,
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    assert RejectionReason.STALE_DATA in reasons(decision)


def test_quote_not_required_in_signal_only_mode(config, audit):
    """Signal-only intents transmit nothing, so quote currency is not required."""
    engine = RiskEngine(config, audit=audit)
    decision = engine.evaluate(
        make_intent(),
        portfolio=make_portfolio(),
        quote=None,
        now=MARKET_OPEN_MOMENT,
    )
    assert "quote_freshness_not_required" in decision.checks_passed


# --- market sessions --------------------------------------------------------


def test_market_closed_blocks_equity_order(engine):
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(),
        quote=make_quote("AAPL", timestamp=MARKET_CLOSED_MOMENT),
        now=MARKET_CLOSED_MOMENT,
    )
    assert not decision.approved
    assert RejectionReason.MARKET_CLOSED in reasons(decision)


def test_premarket_blocked_when_extended_hours_disabled(engine):
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(),
        quote=make_quote("AAPL", timestamp=PREMARKET_MOMENT),
        now=PREMARKET_MOMENT,
    )
    assert not decision.approved
    assert RejectionReason.EXTENDED_HOURS_NOT_ALLOWED in reasons(decision)


def test_crypto_trades_when_equity_market_is_closed(engine):
    """Crypto is 24/7 and must not be blocked by the equity calendar."""
    decision = engine.evaluate(
        make_intent(
            symbol="BTC/USD", asset_class=AssetClass.CRYPTO, quantity=0.01,
            time_in_force=TimeInForce.GTC, mode=TradingMode.PAPER,
            reference_price=60000.0, stop_loss_price=58000.0, take_profit_price=64000.0,
        ),
        portfolio=make_portfolio(),
        quote=make_quote(
            "BTC/USD", price=60000.0, asset_class=AssetClass.CRYPTO,
            timestamp=MARKET_CLOSED_MOMENT,
        ),
        now=MARKET_CLOSED_MOMENT,
    )
    assert decision.approved, decision.summary


# --- limits -----------------------------------------------------------------


def test_daily_loss_limit_blocks(engine, paper_config):
    limit = paper_config.risk.max_daily_loss
    portfolio = make_portfolio(equity=100_000 - limit - 1, last_equity=100_000)
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=portfolio,
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    assert RejectionReason.RISK_LIMIT in reasons(decision)


def test_max_open_positions_blocks_new_symbol(engine, paper_config):
    positions = tuple(
        Position(f"SYM{i}", AssetClass.US_EQUITY, 1, 10, 10, 10, 0)
        for i in range(paper_config.risk.max_open_positions)
    )
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(positions=positions),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    assert any(v.check == "max_open_positions" for v in decision.violations)


def test_position_limit_not_applied_to_existing_symbol(engine, paper_config):
    """Adding to an existing position does not open a new one."""
    positions = tuple(
        Position(f"SYM{i}", AssetClass.US_EQUITY, 1, 10, 10, 10, 0)
        for i in range(paper_config.risk.max_open_positions - 1)
    ) + (Position("AAPL", AssetClass.US_EQUITY, 1, 100, 100, 100, 0),)
    decision = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(positions=positions),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert "max_open_positions_not_applicable" in decision.checks_passed


def test_notional_limit_blocks(engine, paper_config):
    oversized = int(paper_config.risk.max_position_notional / 100.0) + 50
    decision = engine.evaluate(
        make_intent(quantity=oversized, mode=TradingMode.PAPER),
        portfolio=make_portfolio(),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    assert any(v.check == "max_position_notional" for v in decision.violations)


def test_concentration_counts_existing_exposure(engine, paper_config):
    """Repeated small adds must not creep past the concentration cap.

    Existing exposure is 24% of equity and the new order is only 1.5%, so every
    per-order limit passes in isolation. Only the combined figure (25.5%)
    breaches the 25% concentration cap - which is exactly the creep this check
    exists to stop.
    """
    existing = Position("AAPL", AssetClass.US_EQUITY, 240, 100, 100, 24_000, 0)
    decision = engine.evaluate(
        make_intent(quantity=15, mode=TradingMode.PAPER),
        portfolio=make_portfolio(equity=100_000, positions=(existing,)),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    breached = {v.check for v in decision.violations}
    # Isolate the concentration check: no other size limit may have fired.
    assert breached == {"max_symbol_concentration"}, breached


def test_concentration_allows_add_that_stays_under_cap(engine):
    """The mirror case: 20% existing plus 1.5% new is 21.5%, under the 25% cap."""
    existing = Position("AAPL", AssetClass.US_EQUITY, 200, 100, 100, 20_000, 0)
    decision = engine.evaluate(
        make_intent(quantity=15, mode=TradingMode.PAPER),
        portfolio=make_portfolio(equity=100_000, positions=(existing,)),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert decision.approved, decision.summary


def test_insufficient_buying_power_blocks(engine):
    decision = engine.evaluate(
        make_intent(quantity=15, mode=TradingMode.PAPER),
        portfolio=make_portfolio(equity=100_000, buying_power=100.0),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    assert RejectionReason.INSUFFICIENT_BUYING_POWER in reasons(decision)


def test_missing_stop_loss_blocks_when_required(engine):
    decision = engine.evaluate(
        make_intent(stop_loss_price=None, take_profit_price=None, mode=TradingMode.PAPER),
        portfolio=make_portfolio(),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved
    assert RejectionReason.MISSING_STOP_LOSS in reasons(decision)


def test_minimum_equity_blocks(engine, paper_config):
    tiny = paper_config.risk.min_account_equity - 1
    decision = engine.evaluate(
        make_intent(quantity=1, mode=TradingMode.PAPER),
        portfolio=make_portfolio(equity=tiny, last_equity=tiny, buying_power=tiny),
        quote=make_quote("AAPL"),
        now=MARKET_OPEN_MOMENT,
    )
    assert not decision.approved


# --- duplicate suppression --------------------------------------------------


def test_duplicate_order_blocked_within_window(engine):
    kwargs = dict(
        portfolio=make_portfolio(), quote=make_quote("AAPL"), now=MARKET_OPEN_MOMENT
    )
    first = engine.evaluate(make_intent(mode=TradingMode.PAPER), **kwargs)
    assert first.approved

    # A different client_order_id must still be caught: same symbol/side/qty/type.
    second = engine.evaluate(
        make_intent(mode=TradingMode.PAPER, client_order_id="a-different-id"), **kwargs
    )
    assert not second.approved
    assert RejectionReason.DUPLICATE_ORDER in reasons(second)


def test_duplicate_allowed_after_window_expires(engine, paper_config):
    kwargs = dict(portfolio=make_portfolio(), quote=make_quote("AAPL"))
    assert engine.evaluate(
        make_intent(mode=TradingMode.PAPER), now=MARKET_OPEN_MOMENT, **kwargs
    ).approved

    later = MARKET_OPEN_MOMENT + timedelta(
        seconds=paper_config.risk.duplicate_order_window_seconds + 1
    )
    assert engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        now=later,
        portfolio=make_portfolio(),
        quote=make_quote("AAPL", timestamp=later),
    ).approved


def test_rejected_order_does_not_consume_daily_budget(engine):
    """A rejected order must not block a corrected retry."""
    engine.evaluate(
        make_intent(stop_loss_price=None, take_profit_price=None, mode=TradingMode.PAPER),
        portfolio=make_portfolio(), quote=make_quote("AAPL"), now=MARKET_OPEN_MOMENT,
    )
    corrected = engine.evaluate(
        make_intent(mode=TradingMode.PAPER),
        portfolio=make_portfolio(), quote=make_quote("AAPL"), now=MARKET_OPEN_MOMENT,
    )
    assert corrected.approved


# --- decision invariants ----------------------------------------------------


def test_decision_cannot_be_approved_with_violations():
    """The type system must make a silently-overridden limit impossible."""
    violation = RiskViolation(
        check="x", reason=RejectionReason.RISK_LIMIT, message="breached"
    )
    with pytest.raises(ValueError, match="cannot be approved"):
        RiskDecision(approved=True, violations=(violation,))


def test_rejection_requires_a_violation():
    with pytest.raises(ValueError, match="at least one violation"):
        RiskDecision.reject(())


def test_every_rejection_is_audited(engine, audit):
    engine.evaluate(
        make_intent(stop_loss_price=None, take_profit_price=None, mode=TradingMode.PAPER),
        portfolio=make_portfolio(), quote=make_quote("AAPL"), now=MARKET_OPEN_MOMENT,
    )
    entries = [e for e in audit.tail() if e["event"] == "risk_decision"]
    assert entries and entries[-1]["payload"]["approved"] is False
    assert entries[-1]["payload"]["violations"]
