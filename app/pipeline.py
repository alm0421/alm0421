"""End-to-end orchestration.

Flow
----
::

    universe
      -> market data (bars, freshness-labelled)
      -> drop forming bar
      -> scanner filter + rank
      -> strategy evaluation
      -> signals (ranked)
      -> [execution enabled?] position sizing
      -> order intent
      -> structural validation
      -> risk engine
      -> broker (null / paper / live-gated)
      -> audit trail + outputs

Each stage is separately observable: :class:`PipelineResult` carries the state
after every step, so the dashboard can show where symbols dropped out rather
than only the final signals.

Signal generation never depends on execution being enabled. The same signals are
produced in signal-only mode as in paper mode; only what happens *after* them
differs. That is what keeps the modes comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from app.clock import MarketClock, utcnow
from app.config import AppConfig
from app.data.base import (
    BarSet,
    DataUnavailable,
    MarketDataProvider,
    Quote,
    drop_forming_bar,
)
from app.data.alpaca_provider import timeframe_seconds
from app.enums import AssetClass, OrderSide, OrderType, SignalDirection, TimeInForce
from app.execution.base import Broker
from app.execution.models import OrderIntent, OrderResult, RejectionReason
from app.execution.validation import validate_order_intent
from app.logging import AuditLog, get_logger
from app.portfolio.positions import PortfolioState
from app.risk.engine import RiskEngine
from app.risk.sizing import size_position
from app.scanners.base import ScanResult, Scanner
from app.signals.models import Signal, SignalSet
from app.strategies.base import Strategy

log = get_logger(__name__)


@dataclass
class PipelineResult:
    """Everything one pipeline run produced, stage by stage."""

    started_at: datetime
    finished_at: datetime | None = None
    symbols_requested: int = 0
    bar_sets_received: int = 0
    scan_results: list[ScanResult] = field(default_factory=list)
    signals: SignalSet = field(default_factory=lambda: SignalSet(()))
    order_results: list[OrderResult] = field(default_factory=list)
    portfolio: PortfolioState | None = None
    errors: list[str] = field(default_factory=list)
    #: Symbols that produced no data, keyed by reason.
    data_failures: dict[str, str] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": round(self.duration_seconds, 3),
            "symbols_requested": self.symbols_requested,
            "bar_sets_received": self.bar_sets_received,
            "scan_results": [r.to_dict() for r in self.scan_results],
            "signals": self.signals.to_records(),
            "order_results": [r.to_dict() for r in self.order_results],
            "portfolio": self.portfolio.to_dict() if self.portfolio else None,
            "errors": self.errors,
            "data_failures": self.data_failures,
        }


class TradingPipeline:
    """Runs one full scan -> signal -> (optional) execution cycle."""

    def __init__(
        self,
        config: AppConfig,
        *,
        provider: MarketDataProvider,
        scanners: Sequence[Scanner],
        strategies: Sequence[Strategy],
        broker: Broker,
        risk_engine: RiskEngine,
        audit: AuditLog,
        clock: MarketClock | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._scanners = list(scanners)
        self._strategies = list(strategies)
        self._broker = broker
        self._risk = risk_engine
        self._audit = audit
        self._clock = clock or MarketClock(display_tz=config.data.display_timezone)

    # -- main entry point --------------------------------------------------

    def run(self, *, execute: bool = False, now: datetime | None = None) -> PipelineResult:
        """Run one cycle.

        ``execute`` only *permits* order placement; the mode and every risk check
        still decide whether anything is actually sent. In signal-only mode the
        broker records intents and transmits nothing regardless of this flag.
        """
        moment = now or utcnow()
        result = PipelineResult(started_at=moment)
        self._audit.record(
            "pipeline_started",
            mode=self._config.mode.value,
            execute_requested=execute,
            scanners=[s.name for s in self._scanners],
            strategies=[s.name for s in self._strategies],
        )

        # --- 1. market data ---
        bar_sets = self._fetch_bars(result, now=moment)
        result.bar_sets_received = len(bar_sets)

        # --- 2. scan ---
        result.scan_results = self._run_scanners(bar_sets)

        # --- 3. strategies ---
        shortlist = {r.symbol for r in result.scan_results}
        result.signals = self._evaluate_strategies(bar_sets, shortlist).ranked(
            limit=self._config.scanner.max_results
        )

        # --- 4. portfolio ---
        result.portfolio = self._broker.get_portfolio()

        # --- 5. execution ---
        if execute:
            result.order_results = self._execute_signals(result, moment)
        else:
            log.info(
                "Execution not requested; %d signal(s) generated and recorded only.",
                len(result.signals),
            )

        result.finished_at = utcnow()
        self._audit.record(
            "pipeline_finished",
            duration_seconds=round(result.duration_seconds, 3),
            signals=len(result.signals),
            scan_results=len(result.scan_results),
            orders=len(result.order_results),
            errors=len(result.errors),
        )
        return result

    # -- stages ------------------------------------------------------------

    def _fetch_bars(
        self, result: PipelineResult, *, now: datetime
    ) -> dict[str, tuple[BarSet, AssetClass]]:
        """Fetch bars for the whole universe.

        ``now`` is threaded through to the forming-bar guard rather than letting
        it fall back to wall-clock time, so a run is fully determined by its
        timestamp. Without this a replay or a test would silently classify bars
        as forming based on when the code happened to execute.
        """
        timeframe = self._config.scanner.bar_timeframe
        limit = self._config.scanner.lookback_bars
        bar_seconds = timeframe_seconds(timeframe)
        collected: dict[str, tuple[BarSet, AssetClass]] = {}

        for asset_class, symbols in self._config.universe.items():
            if not symbols:
                continue
            result.symbols_requested += len(symbols)

            if asset_class is AssetClass.FUTURES:
                for symbol in symbols:
                    result.data_failures[symbol] = (
                        "futures is not supported by any shipped data provider"
                    )
                continue

            try:
                fetched = self._provider.get_bars_batch(
                    symbols, asset_class, timeframe=timeframe, limit=limit
                )
            except DataUnavailable as exc:
                message = f"{asset_class.value}: {exc}"
                result.errors.append(message)
                log.error("Bar fetch failed for %s: %s", asset_class.value, exc)
                continue

            for symbol in symbols:
                bars = fetched.get(symbol)
                if bars is None:
                    result.data_failures[symbol] = "no bars returned"
                    continue
                trimmed = drop_forming_bar(bars, bar_seconds, now=now)
                if len(trimmed) == 0:
                    result.data_failures[symbol] = "no completed bars"
                    continue
                collected[symbol] = (trimmed, asset_class)

        return collected

    def _run_scanners(self, bar_sets: dict[str, tuple[BarSet, AssetClass]]) -> list[ScanResult]:
        plain = {symbol: bars for symbol, (bars, _) in bar_sets.items()}
        results: list[ScanResult] = []
        for scanner in self._scanners:
            found = scanner.scan(plain, limit=self._config.scanner.max_results)
            results.extend(found)
            self._audit.record(
                "scanner_run",
                scanner=scanner.name,
                symbols_evaluated=len(plain),
                results=len(found),
                top=[r.symbol for r in found[:5]],
            )
        results.sort(key=lambda r: (-r.score, r.symbol))
        return results

    def _evaluate_strategies(
        self, bar_sets: dict[str, tuple[BarSet, AssetClass]], shortlist: set[str]
    ) -> SignalSet:
        signals: list[Signal] = []
        # When no scanner shortlisted anything, evaluate nothing rather than
        # silently falling back to the whole universe.
        for symbol in sorted(shortlist):
            entry = bar_sets.get(symbol)
            if entry is None:
                continue
            bars, _ = entry
            for strategy in self._strategies:
                try:
                    signal = strategy.evaluate(bars)
                except Exception:  # noqa: BLE001 - one symbol must not kill the run
                    log.exception(
                        "Strategy %s failed on %s", strategy.name, symbol,
                        extra={"symbol": symbol, "strategy": strategy.name},
                    )
                    continue
                if signal is not None:
                    signals.append(signal)
                    self._audit.record(
                        "signal_generated",
                        signal_id=signal.signal_id,
                        symbol=signal.symbol,
                        direction=signal.direction.value,
                        strategy=signal.strategy,
                        confidence=signal.confidence,
                        reference_price=signal.reference_price,
                        stop_price=signal.stop_price,
                        target_price=signal.target_price,
                        data_freshness=signal.data_freshness.value,
                        data_timestamp=signal.data_timestamp.isoformat(),
                    )
        return SignalSet.from_iterable(signals)

    def _execute_signals(self, result: PipelineResult, moment: datetime) -> list[OrderResult]:
        portfolio = result.portfolio or PortfolioState.unknown("portfolio not fetched")
        outcomes: list[OrderResult] = []

        for signal in result.signals.actionable():
            intent = self._build_intent(signal, portfolio)
            if intent is None:
                continue

            # --- structural validation ---
            issues = validate_order_intent(intent, self._config)
            if issues:
                rejection = OrderResult.rejected(
                    intent, issues[0].reason,
                    "; ".join(i.message for i in issues),
                    broker=self._broker.name,
                )
                self._audit.record(
                    "order_validation_failed",
                    symbol=intent.symbol,
                    signal_id=signal.signal_id,
                    issues=[{"reason": i.reason.value, "message": i.message} for i in issues],
                )
                outcomes.append(rejection)
                continue

            # --- market data for risk evaluation ---
            quote = self._fetch_quote(signal, result)

            # --- risk ---
            decision = self._risk.evaluate(
                intent, portfolio=portfolio, quote=quote, now=moment
            )
            if not decision.approved:
                outcomes.append(
                    OrderResult.rejected(
                        intent,
                        decision.primary_reason or RejectionReason.RISK_LIMIT,
                        decision.summary,
                        broker=self._broker.name,
                    )
                )
                continue

            # --- submit ---
            outcomes.append(self._broker.submit_order(intent))

        return outcomes

    def _fetch_quote(self, signal: Signal, result: PipelineResult) -> Quote | None:
        """Fetch a current quote. Only needed when orders actually transmit."""
        if not self._config.mode.submits_real_orders:
            return None
        try:
            return self._provider.get_latest_quote(signal.symbol, signal.asset_class)
        except DataUnavailable as exc:
            result.errors.append(f"quote for {signal.symbol}: {exc}")
            # Returning None makes the risk engine reject on STALE_DATA, which
            # is the correct outcome: no data means no order.
            return None

    def _build_intent(self, signal: Signal, portfolio: PortfolioState) -> OrderIntent | None:
        """Turn a signal into a fully specified order intent.

        Returns None when the signal cannot be sized - for example when account
        equity is unknown, which is the normal case in signal-only mode.
        """
        equity = portfolio.account.equity if portfolio.account.is_known else 0.0

        sized = size_position(signal, equity=equity, risk_config=self._config.risk)
        if not sized.is_tradable:
            self._audit.record(
                "order_not_sized",
                symbol=signal.symbol,
                signal_id=signal.signal_id,
                binding_constraint=sized.binding_constraint,
                account_known=portfolio.account.is_known,
                detail=sized.detail,
            )
            return None

        side = OrderSide.BUY if signal.direction is SignalDirection.LONG else OrderSide.SELL
        # Crypto cannot rest a DAY order on Alpaca.
        tif = (
            TimeInForce.GTC
            if signal.asset_class is AssetClass.CRYPTO
            else self._config.execution.default_time_in_force
        )

        return OrderIntent(
            symbol=signal.symbol,
            asset_class=signal.asset_class,
            side=side,
            quantity=sized.quantity,
            order_type=OrderType.MARKET,
            time_in_force=tif,
            mode=self._config.mode,
            take_profit_price=signal.target_price,
            stop_loss_price=signal.stop_price,
            reference_price=signal.reference_price,
            signal_id=signal.signal_id,
            strategy=signal.strategy,
            extended_hours=False,
        )
