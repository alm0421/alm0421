"""Market data contracts.

The provider interface is deliberately narrow: quotes for execution decisions,
bars for indicator and scanner work. Both carry freshness and a source label so
the dashboard and the risk engine can reason about data quality instead of
assuming it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from app.clock import age_seconds, ensure_aware, is_stale
from app.enums import AssetClass, Freshness


class DataUnavailable(RuntimeError):
    """Raised when a provider cannot supply data for a symbol.

    This is distinct from returning stale data: callers must be able to tell
    "no data" apart from "old data", because they are handled differently.
    """


@dataclass(frozen=True)
class Quote:
    """A top-of-book quote at a point in time."""

    symbol: str
    asset_class: AssetClass
    bid: float
    ask: float
    timestamp: datetime
    freshness: Freshness
    source: str
    bid_size: float = 0.0
    ask_size: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_aware(self.timestamp, field="quote.timestamp"))

    @property
    def mid(self) -> float:
        """Midpoint. Falls back to whichever side is present if the book is one-sided."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.ask if self.ask > 0 else self.bid

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        return (self.spread / mid) * 10_000 if mid > 0 else 0.0

    def age_seconds(self, *, now: datetime | None = None) -> float:
        return age_seconds(self.timestamp, now=now)

    def is_stale(self, max_age_seconds: float, *, now: datetime | None = None) -> bool:
        return is_stale(self.timestamp, max_age_seconds, now=now)

    def usable_for_execution(self, max_age_seconds: float, *, now: datetime | None = None) -> bool:
        """A quote may size or trigger an order only if it is both current and
        labelled as genuinely live data."""
        if not self.freshness.safe_for_execution:
            return False
        if self.bid <= 0 and self.ask <= 0:
            return False
        return not self.is_stale(max_age_seconds, now=now)


@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar. ``timestamp`` is the bar's OPEN time."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int = 0
    vwap: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_aware(self.timestamp, field="bar.timestamp"))


@dataclass(frozen=True)
class BarSet:
    """An ordered series of bars for one symbol.

    Bars are guaranteed ascending by timestamp. Strategies rely on this ordering
    to avoid lookahead, so it is asserted at construction rather than trusted.
    """

    symbol: str
    asset_class: AssetClass
    timeframe: str
    bars: tuple[Bar, ...]
    freshness: Freshness
    source: str

    def __post_init__(self) -> None:
        timestamps = [b.timestamp for b in self.bars]
        if timestamps != sorted(timestamps):
            raise ValueError(
                f"BarSet for {self.symbol} is not in ascending timestamp order; "
                "out-of-order bars would corrupt indicator state and permit lookahead."
            )

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    @property
    def highs(self) -> list[float]:
        return [b.high for b in self.bars]

    @property
    def lows(self) -> list[float]:
        return [b.low for b in self.bars]

    @property
    def volumes(self) -> list[float]:
        return [b.volume for b in self.bars]

    @property
    def last(self) -> Bar | None:
        return self.bars[-1] if self.bars else None

    @property
    def last_timestamp(self) -> datetime | None:
        return self.bars[-1].timestamp if self.bars else None


class MarketDataProvider(ABC):
    """Interface every market data source implements."""

    #: Human-readable provider name used in logs, audit entries and the dashboard.
    name: str = "unknown"

    @abstractmethod
    def get_latest_quote(self, symbol: str, asset_class: AssetClass) -> Quote:
        """Return the most recent quote. Raises :class:`DataUnavailable` if none."""

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        asset_class: AssetClass,
        *,
        timeframe: str,
        limit: int,
    ) -> BarSet:
        """Return up to ``limit`` most recent completed bars, oldest first."""

    def get_bars_batch(
        self,
        symbols: Sequence[str],
        asset_class: AssetClass,
        *,
        timeframe: str,
        limit: int,
    ) -> dict[str, BarSet]:
        """Fetch bars for several symbols.

        The default implementation loops; providers with a native batch endpoint
        should override for efficiency. Symbols that fail are omitted from the
        result rather than aborting the whole batch - one bad symbol must not
        blind the scanner to the rest of the universe.
        """
        results: dict[str, BarSet] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_bars(
                    symbol, asset_class, timeframe=timeframe, limit=limit
                )
            except DataUnavailable:
                continue
        return results

    @abstractmethod
    def health(self) -> dict[str, object]:
        """Return a status dict for the dashboard: connectivity, feed, freshness."""


def drop_forming_bar(bars: BarSet, bar_seconds: int, *, now: datetime | None = None) -> BarSet:
    """Remove a trailing bar that has not closed yet.

    Intraday bar endpoints return the *current, still-forming* bar alongside
    completed ones. Evaluating a strategy on a partial bar is a lookahead-style
    error in reverse: the bar's high, low and close will still change, so a
    signal taken from it is not reproducible and cannot be backtested honestly.

    A bar whose open timestamp plus its duration is still in the future has not
    closed, and is dropped.
    """
    from app.clock import utcnow  # local import: avoids a cycle at module load

    if not bars.bars:
        return bars
    reference = ensure_aware(now) if now is not None else utcnow()
    kept = tuple(
        bar for bar in bars.bars
        if (bar.timestamp.timestamp() + bar_seconds) <= reference.timestamp()
    )
    if len(kept) == len(bars.bars):
        return bars
    return BarSet(
        symbol=bars.symbol,
        asset_class=bars.asset_class,
        timeframe=bars.timeframe,
        bars=kept,
        freshness=bars.freshness,
        source=bars.source,
    )
