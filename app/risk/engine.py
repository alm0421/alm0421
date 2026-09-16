"""Pre-trade risk engine.

Runs after structural validation (:mod:`app.execution.validation`) and before
any broker sees an order. Every check is fail-closed: if a required input is
missing or unknown, the order is rejected rather than approved on an assumption.

Check order is deliberate - the cheapest and most absolute blocks run first, so
a halted system rejects immediately without touching the network.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from app.clock import MarketClock, utcnow
from app.config import AppConfig
from app.data.base import Quote
from app.enums import AssetClass, MarketSession, OrderSide, TradingMode
from app.execution.models import OrderIntent, RejectionReason
from app.logging import AuditLog, get_logger
from app.portfolio.positions import PortfolioState
from app.risk.limits import RiskDecision, RiskViolation

log = get_logger(__name__)


class RiskEngine:
    """Evaluates every pre-trade limit against an order intent."""

    def __init__(
        self,
        config: AppConfig,
        *,
        clock: MarketClock | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._config = config
        self._clock = clock or MarketClock(display_tz=config.data.display_timezone)
        self._audit = audit
        #: client_order_id fingerprints of recently approved orders, used for
        #: duplicate suppression within the configured window.
        self._recent: list[tuple[str, datetime]] = []
        #: Orders approved today, for the per-day activity cap.
        self._approved_today: list[datetime] = []

    # -- public API --------------------------------------------------------

    def kill_switch_engaged(self) -> bool:
        """True when the kill-switch file exists.

        A file rather than a flag so it can be engaged from outside the process
        - by an operator, a script, or a scheduled task - without a restart.
        """
        return self._config.risk.kill_switch_file.exists()

    def evaluate(
        self,
        intent: OrderIntent,
        *,
        portfolio: PortfolioState,
        quote: Quote | None = None,
        now: datetime | None = None,
    ) -> RiskDecision:
        """Evaluate every limit and return an approval or a rejection."""
        moment = now or utcnow()
        violations: list[RiskViolation] = []
        passed: list[str] = []
        risk = self._config.risk

        def check(name: str, ok: bool, violation: RiskViolation | None) -> None:
            if ok:
                passed.append(name)
            elif violation is not None:
                violations.append(violation)

        # --- 1. Kill switch: absolute, checked first ---
        if self.kill_switch_engaged():
            violations.append(
                RiskViolation(
                    check="kill_switch",
                    reason=RejectionReason.KILL_SWITCH,
                    message=(
                        f"Kill switch is ENGAGED ({risk.kill_switch_file}). All order "
                        "submission is halted. Delete the file to resume."
                    ),
                    observed="engaged",
                    limit="disengaged",
                )
            )
            # Nothing else matters; return immediately.
            decision = RiskDecision.reject(tuple(violations), tuple(passed))
            self._audit_decision(intent, decision)
            return decision
        passed.append("kill_switch")

        # --- 2. Mode must be able to execute ---
        if not self._config.mode.submits_real_orders:
            # Not a violation: signal-only intents are expected and are routed to
            # the null broker. Recorded so the audit trail shows the mode.
            passed.append("mode_is_signal_only")

        # --- 3. Account state must be KNOWN ---
        account = portfolio.account
        if not account.is_known:
            violations.append(
                RiskViolation(
                    check="account_known",
                    reason=RejectionReason.UNKNOWN_ACCOUNT_STATE,
                    message=(
                        "Account state is unknown, so no limit can be evaluated "
                        f"against it: {account.error or 'broker unreachable'}."
                    ),
                    observed="unknown",
                    limit="known",
                )
            )
            # Every remaining check depends on account state; stop here.
            decision = RiskDecision.reject(tuple(violations), tuple(passed))
            self._audit_decision(intent, decision)
            return decision
        passed.append("account_known")

        # --- 4. Account mode must match the trading mode ---
        expected_paper = self._config.mode is not TradingMode.LIVE
        if account.is_paper is None:
            violations.append(
                RiskViolation(
                    check="account_mode",
                    reason=RejectionReason.UNKNOWN_ACCOUNT_STATE,
                    message="Broker did not report whether the account is paper or live.",
                    observed=None, limit="paper" if expected_paper else "live",
                )
            )
        elif account.is_paper != expected_paper:
            violations.append(
                RiskViolation(
                    check="account_mode",
                    reason=RejectionReason.MODE_NOT_EXECUTABLE,
                    message=(
                        f"Account mode mismatch: platform mode is "
                        f"'{self._config.mode.value}' but the connected account is "
                        f"{'paper' if account.is_paper else 'LIVE'}."
                    ),
                    observed="paper" if account.is_paper else "live",
                    limit="paper" if expected_paper else "live",
                )
            )
        else:
            passed.append("account_mode")

        # --- 5. Broker-side trading block ---
        check(
            "trading_not_blocked",
            not account.trading_blocked,
            RiskViolation(
                check="trading_not_blocked",
                reason=RejectionReason.RISK_LIMIT,
                message="The broker has blocked trading on this account.",
                observed=True, limit=False,
            ),
        )

        # --- 6. Minimum account equity ---
        check(
            "min_account_equity",
            account.equity >= risk.min_account_equity,
            RiskViolation(
                check="min_account_equity",
                reason=RejectionReason.RISK_LIMIT,
                message=(
                    f"Account equity {account.equity:,.2f} is below the minimum "
                    f"{risk.min_account_equity:,.2f}."
                ),
                observed=round(account.equity, 2), limit=risk.min_account_equity,
            ),
        )

        # --- 7. Daily loss limits ---
        daily_pl = account.daily_pl
        check(
            "max_daily_loss",
            daily_pl > -risk.max_daily_loss,
            RiskViolation(
                check="max_daily_loss",
                reason=RejectionReason.RISK_LIMIT,
                message=(
                    f"Daily loss {daily_pl:,.2f} has reached the limit of "
                    f"-{risk.max_daily_loss:,.2f}. No new orders today."
                ),
                observed=round(daily_pl, 2), limit=-risk.max_daily_loss,
            ),
        )
        daily_pl_fraction = (daily_pl / account.last_equity) if account.last_equity > 0 else 0.0
        check(
            "max_daily_loss_pct",
            daily_pl_fraction > -risk.max_daily_loss_pct,
            RiskViolation(
                check="max_daily_loss_pct",
                reason=RejectionReason.RISK_LIMIT,
                message=(
                    f"Daily loss {daily_pl_fraction * 100:.2f}% has reached the limit "
                    f"of -{risk.max_daily_loss_pct * 100:.2f}%."
                ),
                observed=round(daily_pl_fraction, 6), limit=-risk.max_daily_loss_pct,
            ),
        )

        # --- 8. Open position count (only for orders that OPEN exposure) ---
        existing = portfolio.position_for(intent.symbol)
        opens_new_position = existing is None
        if opens_new_position:
            check(
                "max_open_positions",
                portfolio.open_position_count < risk.max_open_positions,
                RiskViolation(
                    check="max_open_positions",
                    reason=RejectionReason.RISK_LIMIT,
                    message=(
                        f"Already holding {portfolio.open_position_count} positions; "
                        f"the limit is {risk.max_open_positions}."
                    ),
                    observed=portfolio.open_position_count, limit=risk.max_open_positions,
                ),
            )
        else:
            passed.append("max_open_positions_not_applicable")

        # --- 9. Trades per day ---
        self._prune_daily(moment)
        check(
            "max_trades_per_day",
            len(self._approved_today) < risk.max_trades_per_day,
            RiskViolation(
                check="max_trades_per_day",
                reason=RejectionReason.RISK_LIMIT,
                message=(
                    f"{len(self._approved_today)} orders already approved today; "
                    f"the limit is {risk.max_trades_per_day}."
                ),
                observed=len(self._approved_today), limit=risk.max_trades_per_day,
            ),
        )

        # --- 10. Notional and concentration ---
        notional = self._notional_for(intent, quote)
        if notional is None:
            violations.append(
                RiskViolation(
                    check="notional_known",
                    reason=RejectionReason.STALE_DATA,
                    message=(
                        f"{intent.symbol}: cannot determine order notional - no limit "
                        "price, reference price or usable quote is available. Size "
                        "limits cannot be evaluated."
                    ),
                    observed=None, limit=risk.max_position_notional,
                )
            )
        else:
            check(
                "max_position_notional",
                notional <= risk.max_position_notional,
                RiskViolation(
                    check="max_position_notional",
                    reason=RejectionReason.RISK_LIMIT,
                    message=(
                        f"{intent.symbol}: order notional {notional:,.2f} exceeds the "
                        f"limit {risk.max_position_notional:,.2f}."
                    ),
                    observed=round(notional, 2), limit=risk.max_position_notional,
                ),
            )
            pct_of_equity = notional / account.equity if account.equity > 0 else float("inf")
            check(
                "max_position_pct_equity",
                pct_of_equity <= risk.max_position_pct_equity,
                RiskViolation(
                    check="max_position_pct_equity",
                    reason=RejectionReason.RISK_LIMIT,
                    message=(
                        f"{intent.symbol}: order is {pct_of_equity * 100:.2f}% of equity, "
                        f"above the {risk.max_position_pct_equity * 100:.2f}% limit."
                    ),
                    observed=round(pct_of_equity, 6), limit=risk.max_position_pct_equity,
                ),
            )
            # Concentration counts the EXISTING position plus this order, so
            # repeated adds cannot creep past the cap one order at a time.
            existing_value = existing.abs_market_value if existing else 0.0
            combined_pct = (
                (existing_value + notional) / account.equity if account.equity > 0 else float("inf")
            )
            check(
                "max_symbol_concentration",
                combined_pct <= risk.max_symbol_concentration_pct,
                RiskViolation(
                    check="max_symbol_concentration",
                    reason=RejectionReason.RISK_LIMIT,
                    message=(
                        f"{intent.symbol}: existing plus new exposure would be "
                        f"{combined_pct * 100:.2f}% of equity, above the "
                        f"{risk.max_symbol_concentration_pct * 100:.2f}% concentration limit."
                    ),
                    observed=round(combined_pct, 6), limit=risk.max_symbol_concentration_pct,
                ),
            )

            # --- 11. Buying power ---
            if intent.side is OrderSide.BUY:
                check(
                    "buying_power",
                    notional <= account.buying_power,
                    RiskViolation(
                        check="buying_power",
                        reason=RejectionReason.INSUFFICIENT_BUYING_POWER,
                        message=(
                            f"{intent.symbol}: order notional {notional:,.2f} exceeds "
                            f"available buying power {account.buying_power:,.2f}."
                        ),
                        observed=round(notional, 2), limit=round(account.buying_power, 2),
                    ),
                )

        # --- 12. Mandatory stop loss ---
        if risk.require_stop_loss:
            check(
                "require_stop_loss",
                intent.stop_loss_price is not None,
                RiskViolation(
                    check="require_stop_loss",
                    reason=RejectionReason.MISSING_STOP_LOSS,
                    message=(
                        f"{intent.symbol}: risk.require_stop_loss is true but the order "
                        "carries no protective stop."
                    ),
                    observed=None, limit="stop_loss_price required",
                ),
            )

        # --- 13. Market session ---
        session_info = self._clock.session_for(intent.asset_class, now=moment)
        self._check_session(intent, session_info, violations, passed)

        # --- 14. Quote freshness ---
        self._check_quote(intent, quote, violations, passed, moment)

        # --- 15. Duplicate suppression ---
        fingerprint = self._fingerprint(intent)
        self._prune_recent(moment)
        duplicate = any(fp == fingerprint for fp, _ in self._recent)
        check(
            "duplicate_order",
            not duplicate,
            RiskViolation(
                check="duplicate_order",
                reason=RejectionReason.DUPLICATE_ORDER,
                message=(
                    f"{intent.symbol}: an identical order was approved within the last "
                    f"{risk.duplicate_order_window_seconds}s. Refusing to submit a duplicate."
                ),
                observed=fingerprint, limit=f"{risk.duplicate_order_window_seconds}s window",
            ),
        )

        # --- decide ---
        if violations:
            decision = RiskDecision.reject(tuple(violations), tuple(passed))
        else:
            decision = RiskDecision.approve(tuple(passed))
            # Only record state for approved orders, so a rejected order does
            # not consume the daily budget or block a corrected retry.
            self._recent.append((fingerprint, moment))
            self._approved_today.append(moment)

        self._audit_decision(intent, decision)
        return decision

    # -- individual checks -------------------------------------------------

    def _check_session(
        self, intent: OrderIntent, session_info, violations: list[RiskViolation], passed: list[str]
    ) -> None:
        risk = self._config.risk
        session = session_info.session

        if intent.asset_class is AssetClass.CRYPTO:
            passed.append("market_session")
            return

        if session is MarketSession.CLOSED:
            if risk.block_when_market_closed:
                violations.append(
                    RiskViolation(
                        check="market_session",
                        reason=RejectionReason.MARKET_CLOSED,
                        message=(
                            f"{intent.symbol}: the market is closed "
                            f"(next open: {session_info.next_open.isoformat() if session_info.next_open else 'unknown'})."
                        ),
                        observed=session.value, limit="regular/extended session",
                    )
                )
            else:
                passed.append("market_session_closed_allowed")
            return

        if session in (MarketSession.PREMARKET, MarketSession.AFTER_HOURS):
            if not risk.allow_extended_hours:
                violations.append(
                    RiskViolation(
                        check="market_session",
                        reason=RejectionReason.EXTENDED_HOURS_NOT_ALLOWED,
                        message=(
                            f"{intent.symbol}: it is the {session.value} session and "
                            "risk.allow_extended_hours is false."
                        ),
                        observed=session.value, limit="regular session only",
                    )
                )
                return
            if not intent.extended_hours:
                violations.append(
                    RiskViolation(
                        check="market_session",
                        reason=RejectionReason.INVALID_ORDER,
                        message=(
                            f"{intent.symbol}: it is the {session.value} session but the "
                            "order is not flagged extended_hours; it would not fill "
                            "until the next regular session."
                        ),
                        observed=False, limit=True,
                    )
                )
                return

        passed.append("market_session")

    def _check_quote(
        self,
        intent: OrderIntent,
        quote: Quote | None,
        violations: list[RiskViolation],
        passed: list[str],
        moment: datetime,
    ) -> None:
        """A live order requires current, correctly-labelled market data."""
        if not self._config.mode.submits_real_orders:
            # Signal-only intents are not transmitted, so quote currency is not
            # a safety requirement for them.
            passed.append("quote_freshness_not_required")
            return

        max_age = self._config.data.stale_quote_seconds

        if quote is None:
            violations.append(
                RiskViolation(
                    check="quote_freshness",
                    reason=RejectionReason.STALE_DATA,
                    message=(
                        f"{intent.symbol}: no quote was supplied. An order that reaches "
                        "a broker requires current market data."
                    ),
                    observed=None, limit=f"<= {max_age}s old",
                )
            )
            return

        if not quote.freshness.safe_for_execution:
            violations.append(
                RiskViolation(
                    check="quote_freshness",
                    reason=RejectionReason.STALE_DATA,
                    message=(
                        f"{intent.symbol}: market data is labelled "
                        f"'{quote.freshness.value}' ({quote.source}) and must not drive "
                        "an order."
                    ),
                    observed=quote.freshness.value, limit="realtime or snapshot",
                )
            )
            return

        age = quote.age_seconds(now=moment)
        if quote.is_stale(max_age, now=moment):
            violations.append(
                RiskViolation(
                    check="quote_freshness",
                    reason=RejectionReason.STALE_DATA,
                    message=(
                        f"{intent.symbol}: quote is {age:.1f}s old, beyond the "
                        f"{max_age}s limit."
                    ),
                    observed=round(age, 2), limit=max_age,
                )
            )
            return

        passed.append("quote_freshness")

    # -- helpers -----------------------------------------------------------

    def _notional_for(self, intent: OrderIntent, quote: Quote | None) -> float | None:
        estimated = intent.estimated_notional
        if estimated is not None:
            return estimated
        if quote is not None and quote.mid > 0:
            return abs(intent.quantity) * quote.mid
        return None

    @staticmethod
    def _fingerprint(intent: OrderIntent) -> str:
        """Identity of an order for duplicate detection.

        Deliberately excludes client_order_id: two orders with different ids but
        the same symbol, side, quantity and type within the window are exactly
        the duplicate this check exists to catch.
        """
        return "|".join(
            [
                intent.symbol,
                intent.side.value,
                f"{intent.quantity:.8f}",
                intent.order_type.value,
                f"{intent.limit_price or 0:.6f}",
                f"{intent.stop_price or 0:.6f}",
            ]
        )

    def _prune_recent(self, moment: datetime) -> None:
        window = self._config.risk.duplicate_order_window_seconds
        self._recent = [
            (fp, ts) for fp, ts in self._recent
            if (moment - ts).total_seconds() <= window
        ]

    def _prune_daily(self, moment: datetime) -> None:
        """Drop approvals from previous trading days.

        The boundary is the operator's display timezone, so "today" matches what
        the dashboard shows and what the broker counts as a trading day.
        """
        from app.clock import to_display_tz

        today = to_display_tz(moment, self._config.data.display_timezone).date()
        self._approved_today = [
            ts for ts in self._approved_today
            if to_display_tz(ts, self._config.data.display_timezone).date() == today
        ]

    def _audit_decision(self, intent: OrderIntent, decision: RiskDecision) -> None:
        if self._audit is None:
            return
        self._audit.record(
            "risk_decision",
            approved=decision.approved,
            symbol=intent.symbol,
            side=intent.side.value,
            quantity=intent.quantity,
            client_order_id=intent.client_order_id,
            mode=self._config.mode.value,
            violations=[v.to_dict() for v in decision.violations],
            checks_passed=list(decision.checks_passed),
        )
        if not decision.approved:
            log.warning(
                "Risk REJECTED %s: %s",
                intent.symbol, decision.summary,
                extra={"symbol": intent.symbol, "client_order_id": intent.client_order_id},
            )
