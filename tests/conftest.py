"""Shared fixtures and deterministic fakes.

Nothing in the test suite touches the network or a real broker. The fakes are
in-memory and fully deterministic so failures point at logic, not at a flaky
connection.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.config import AppConfig, load_config
from app.data.base import Bar, BarSet, DataUnavailable, MarketDataProvider, Quote
from app.enums import AssetClass, Freshness, TradingMode
from app.logging import AuditLog
from app.portfolio.positions import AccountSnapshot, PortfolioState, Position

UTC = timezone.utc

#: Wednesday 2026-09-16, 15:00 UTC = 11:00 America/New_York - regular session.
MARKET_OPEN_MOMENT = datetime(2026, 9, 16, 15, 0, tzinfo=UTC)
#: Saturday.
MARKET_CLOSED_MOMENT = datetime(2026, 9, 19, 15, 0, tzinfo=UTC)
#: 12:00 UTC = 08:00 ET on a weekday - premarket.
PREMARKET_MOMENT = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


@pytest.fixture()
def config() -> AppConfig:
    """The repository's real configuration, loaded from disk."""
    return load_config()


@pytest.fixture()
def audit(tmp_path: Path) -> AuditLog:
    return AuditLog(tmp_path / "audit.jsonl")


def make_bars(
    symbol: str = "TEST",
    *,
    closes: list[float],
    asset_class: AssetClass = AssetClass.US_EQUITY,
    timeframe: str = "5Min",
    start: datetime = datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
    volume: float = 100_000.0,
    volumes: list[float] | None = None,
    spread: float = 0.5,
    freshness: Freshness = Freshness.HISTORICAL,
) -> BarSet:
    """Build a BarSet from a list of closes.

    Highs and lows are placed symmetrically around each close so ATR is a
    predictable function of ``spread``.
    """
    step = timedelta(minutes=5)
    bars = tuple(
        Bar(
            timestamp=start + step * i,
            open=close,
            high=close + spread,
            low=close - spread,
            close=close,
            volume=(volumes[i] if volumes else volume),
            trade_count=10,
        )
        for i, close in enumerate(closes)
    )
    return BarSet(
        symbol=symbol, asset_class=asset_class, timeframe=timeframe,
        bars=bars, freshness=freshness, source="test",
    )


def uptrend_pullback_closes(n: int = 80) -> list[float]:
    """A realistic uptrend ending in a pullback and reclaim.

    The trend is a sawtooth - three bars up, one bar back - rather than a
    straight ramp. That matters: a relentless ramp drives RSI to ~80, which the
    strategy correctly refuses as over-extended, so it would not exercise the
    entry path at all. This shape keeps RSI near 55, inside the permitted band.

    The final two bars are the setup itself: bar n-2 closes below the fast EMA
    (the pullback) and bar n-1 closes back above it (the reclaim).
    """
    closes: list[float] = []
    price = 100.0
    for i in range(n - 2):
        # Three bars up, one bar back.
        price += 0.65 if (i % 4) != 3 else -1.6
        closes.append(price)
    closes.append(closes[-1] - 4.0)   # pullback below the fast EMA
    closes.append(closes[-1] + 4.2)   # reclaim above it
    return closes


def flat_closes(n: int = 80, value: float = 100.0) -> list[float]:
    """A dead-flat series: no trend, no ATR, so no signal is possible."""
    return [value] * n


class FakeProvider(MarketDataProvider):
    """Deterministic in-memory market data provider."""

    name = "fake"

    def __init__(
        self,
        bar_sets: dict[str, BarSet] | None = None,
        quotes: dict[str, Quote] | None = None,
    ) -> None:
        self._bar_sets = bar_sets or {}
        self._quotes = quotes or {}
        self.quote_calls: list[str] = []

    def get_latest_quote(self, symbol: str, asset_class: AssetClass) -> Quote:
        self.quote_calls.append(symbol)
        quote = self._quotes.get(symbol)
        if quote is None:
            raise DataUnavailable(f"no fake quote for {symbol}")
        return quote

    def get_bars(self, symbol, asset_class, *, timeframe, limit) -> BarSet:
        bars = self._bar_sets.get(symbol)
        if bars is None:
            raise DataUnavailable(f"no fake bars for {symbol}")
        return bars

    def get_bars_batch(self, symbols, asset_class, *, timeframe, limit) -> dict[str, BarSet]:
        return {
            s: self._bar_sets[s]
            for s in symbols
            if s in self._bar_sets and self._bar_sets[s].asset_class is asset_class
        }

    def health(self) -> dict[str, object]:
        return {"provider": self.name, "connected": True}


def make_quote(
    symbol: str = "TEST",
    *,
    price: float = 100.0,
    timestamp: datetime = MARKET_OPEN_MOMENT,
    freshness: Freshness = Freshness.REALTIME,
    asset_class: AssetClass = AssetClass.US_EQUITY,
) -> Quote:
    return Quote(
        symbol=symbol, asset_class=asset_class,
        bid=price - 0.01, ask=price + 0.01,
        timestamp=timestamp, freshness=freshness, source="test",
    )


def make_portfolio(
    *,
    equity: float = 100_000.0,
    buying_power: float = 200_000.0,
    last_equity: float = 100_000.0,
    is_paper: bool | None = True,
    positions: tuple[Position, ...] = (),
    trading_blocked: bool = False,
) -> PortfolioState:
    account = AccountSnapshot(
        equity=equity, cash=equity, buying_power=buying_power,
        is_paper=is_paper, last_equity=last_equity,
        trading_blocked=trading_blocked, as_of=MARKET_OPEN_MOMENT,
    )
    return PortfolioState.from_parts(account, positions)
