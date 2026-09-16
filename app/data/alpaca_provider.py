"""Alpaca market data provider.

Freshness labelling is driven by the configured feed and is never guessed:

==============  ==========================================================
Feed            Label
==============  ==========================================================
``sip``         REALTIME - consolidated tape, all US venues.
``iex``         REALTIME - genuinely live, but IEX venue only (~2-3% of
                consolidated volume). Prices can differ from the NBBO.
``delayed_sip`` DELAYED - consolidated tape delayed 15 minutes. Refused for
                order sizing and triggering by ``Freshness.safe_for_execution``.
``otc``         REALTIME - OTC venue.
anything else   UNKNOWN - refused for execution.
==============  ==========================================================

Historical bars are always labelled HISTORICAL. Bars describe the past, so they
drive indicators and scanners but never size an order on their own; a fresh
quote is required for that.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Sequence

from alpaca.data.enums import DataFeed
from alpaca.data.historical import (
    CryptoHistoricalDataClient,
    StockHistoricalDataClient,
)
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
    OptionBarsRequest,
    OptionLatestQuoteRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from app.clock import utcnow
from app.config import AppConfig
from app.data.base import Bar, BarSet, DataUnavailable, MarketDataProvider, Quote
from app.enums import AssetClass, Freshness
from app.logging import get_logger

log = get_logger(__name__)

_FEED_FRESHNESS: dict[str, Freshness] = {
    "sip": Freshness.REALTIME,
    "iex": Freshness.REALTIME,
    "otc": Freshness.REALTIME,
    "delayed_sip": Freshness.DELAYED,
}

_UNIT_BY_SUFFIX: dict[str, TimeFrameUnit] = {
    "min": TimeFrameUnit.Minute,
    "hour": TimeFrameUnit.Hour,
    "day": TimeFrameUnit.Day,
    "week": TimeFrameUnit.Week,
    "month": TimeFrameUnit.Month,
}


def parse_timeframe(text: str) -> TimeFrame:
    """Parse strings like ``5Min``, ``1Hour``, ``1Day`` into an Alpaca TimeFrame."""
    cleaned = str(text).strip()
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    suffix = "".join(ch for ch in cleaned if ch.isalpha()).lower()
    if not suffix:
        raise ValueError(f"Cannot parse timeframe {text!r}: missing unit (e.g. '5Min')")
    unit = _UNIT_BY_SUFFIX.get(suffix)
    if unit is None:
        allowed = ", ".join(sorted(_UNIT_BY_SUFFIX))
        raise ValueError(f"Unsupported timeframe unit {suffix!r} in {text!r}. Allowed: {allowed}")
    amount = int(digits) if digits else 1
    if amount <= 0:
        raise ValueError(f"Timeframe amount must be positive in {text!r}")
    return TimeFrame(amount, unit)


def timeframe_seconds(text: str) -> int:
    """Approximate duration of one bar, used for staleness thresholds."""
    cleaned = str(text).strip()
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    suffix = "".join(ch for ch in cleaned if ch.isalpha()).lower()
    amount = int(digits) if digits else 1
    per_unit = {"min": 60, "hour": 3600, "day": 86400, "week": 604800, "month": 2592000}
    return amount * per_unit.get(suffix, 60)


class AlpacaDataProvider(MarketDataProvider):
    """Market data from Alpaca for equities, options and crypto.

    Crypto data requires no credentials on Alpaca; equity and option data do.
    The provider reports this honestly through :meth:`health` rather than
    failing opaquely at the first request.
    """

    name = "alpaca"

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._feed_name = config.data.feed
        self._feed = self._resolve_feed(config.data.feed)
        self._freshness = _FEED_FRESHNESS.get(config.data.feed, Freshness.UNKNOWN)

        creds = config.credentials
        self._has_credentials = creds.is_complete
        key, secret = creds.api_key, creds.secret_key

        # Crypto data is public on Alpaca; the client works without keys.
        self._crypto = CryptoHistoricalDataClient(api_key=key, secret_key=secret)
        self._stock = (
            StockHistoricalDataClient(api_key=key, secret_key=secret)
            if self._has_credentials
            else None
        )
        self._option = (
            OptionHistoricalDataClient(api_key=key, secret_key=secret)
            if self._has_credentials
            else None
        )

        if self._freshness is Freshness.UNKNOWN:
            log.warning(
                "Unrecognised data feed %r; data will be labelled UNKNOWN and "
                "refused for order execution.",
                config.data.feed,
                extra={"feed": config.data.feed},
            )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _resolve_feed(feed: str) -> DataFeed | None:
        try:
            return DataFeed(feed)
        except ValueError:
            return None

    @property
    def source_label(self) -> str:
        return f"alpaca:{self._feed_name}"

    def _require_stock_client(self, symbol: str) -> StockHistoricalDataClient:
        if self._stock is None:
            raise DataUnavailable(
                f"Cannot fetch equity data for {symbol}: Alpaca API credentials are "
                "not configured. Set ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY."
            )
        return self._stock

    def _require_option_client(self, symbol: str) -> OptionHistoricalDataClient:
        if self._option is None:
            raise DataUnavailable(
                f"Cannot fetch option data for {symbol}: Alpaca API credentials are "
                "not configured."
            )
        return self._option

    def _with_retries(self, operation: str, symbol: str, func):
        """Run ``func`` with bounded retries and exponential backoff."""
        attempts = self._config.data.max_retries + 1
        backoff = self._config.data.retry_backoff_seconds
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001 - provider errors are opaque
                last_error = exc
                if attempt >= attempts:
                    break
                delay = backoff * (2 ** (attempt - 1))
                log.warning(
                    "%s failed for %s (attempt %d/%d): %s; retrying in %.1fs",
                    operation, symbol, attempt, attempts, exc, delay,
                    extra={"symbol": symbol, "operation": operation, "attempt": attempt},
                )
                time.sleep(delay)
        raise DataUnavailable(
            f"{operation} failed for {symbol} after {attempts} attempt(s): {last_error}"
        ) from last_error

    # -- quotes ------------------------------------------------------------

    def get_latest_quote(self, symbol: str, asset_class: AssetClass) -> Quote:
        if asset_class is AssetClass.FUTURES:
            raise DataUnavailable(
                "Alpaca does not provide futures market data. Futures requires a "
                "different provider and broker adapter."
            )

        if asset_class is AssetClass.CRYPTO:
            request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            payload = self._with_retries(
                "get_latest_quote", symbol, lambda: self._crypto.get_crypto_latest_quote(request)
            )
            # Crypto quotes come from Alpaca's own venue and are live.
            freshness, source = Freshness.REALTIME, "alpaca:crypto"
        elif asset_class is AssetClass.US_OPTION:
            client = self._require_option_client(symbol)
            request = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
            payload = self._with_retries(
                "get_latest_quote", symbol, lambda: client.get_option_latest_quote(request)
            )
            freshness, source = self._freshness, "alpaca:options"
        else:
            client = self._require_stock_client(symbol)
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._feed)
            payload = self._with_retries(
                "get_latest_quote", symbol, lambda: client.get_stock_latest_quote(request)
            )
            freshness, source = self._freshness, self.source_label

        raw = payload.get(symbol) if isinstance(payload, dict) else payload
        if raw is None:
            raise DataUnavailable(f"No quote returned for {symbol}")

        return Quote(
            symbol=symbol,
            asset_class=asset_class,
            bid=float(getattr(raw, "bid_price", 0.0) or 0.0),
            ask=float(getattr(raw, "ask_price", 0.0) or 0.0),
            bid_size=float(getattr(raw, "bid_size", 0.0) or 0.0),
            ask_size=float(getattr(raw, "ask_size", 0.0) or 0.0),
            timestamp=raw.timestamp,
            freshness=freshness,
            source=source,
        )

    # -- bars --------------------------------------------------------------

    def _bar_lookback_start(self, timeframe: str, limit: int) -> datetime:
        """How far back to request so ``limit`` bars are available.

        Padded generously: intraday bars only exist during sessions, so a naive
        limit * bar_duration window would fall short across nights and weekends.
        """
        span = timeframe_seconds(timeframe) * max(limit, 1)
        return utcnow() - timedelta(seconds=span * 6 + 86_400)

    def get_bars(
        self, symbol: str, asset_class: AssetClass, *, timeframe: str, limit: int
    ) -> BarSet:
        if asset_class is AssetClass.FUTURES:
            raise DataUnavailable("Alpaca does not provide futures market data.")

        tf = parse_timeframe(timeframe)
        start = self._bar_lookback_start(timeframe, limit)

        if asset_class is AssetClass.CRYPTO:
            request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start)
            payload = self._with_retries(
                "get_bars", symbol, lambda: self._crypto.get_crypto_bars(request)
            )
            source = "alpaca:crypto"
        elif asset_class is AssetClass.US_OPTION:
            client = self._require_option_client(symbol)
            request = OptionBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start)
            payload = self._with_retries(
                "get_bars", symbol, lambda: client.get_option_bars(request)
            )
            source = "alpaca:options"
        else:
            client = self._require_stock_client(symbol)
            request = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=tf, start=start, feed=self._feed
            )
            payload = self._with_retries(
                "get_bars", symbol, lambda: client.get_stock_bars(request)
            )
            source = self.source_label

        raw_bars = payload.data.get(symbol, []) if hasattr(payload, "data") else []
        if not raw_bars:
            raise DataUnavailable(f"No {timeframe} bars returned for {symbol}")

        bars = tuple(
            Bar(
                timestamp=b.timestamp,
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=float(b.volume or 0.0),
                trade_count=int(b.trade_count or 0),
                vwap=float(b.vwap) if b.vwap is not None else None,
            )
            for b in raw_bars
        )
        # Alpaca returns ascending order; sort defensively rather than trusting it,
        # because BarSet ordering is a lookahead-safety invariant.
        bars = tuple(sorted(bars, key=lambda b: b.timestamp))[-limit:]

        return BarSet(
            symbol=symbol,
            asset_class=asset_class,
            timeframe=timeframe,
            bars=bars,
            freshness=Freshness.HISTORICAL,
            source=source,
        )

    def get_bars_batch(
        self, symbols: Sequence[str], asset_class: AssetClass, *, timeframe: str, limit: int
    ) -> dict[str, BarSet]:
        """Batch fetch using Alpaca's native multi-symbol endpoint."""
        symbols = [s for s in symbols if s]
        if not symbols:
            return {}
        if asset_class is AssetClass.FUTURES:
            return {}

        tf = parse_timeframe(timeframe)
        start = self._bar_lookback_start(timeframe, limit)

        try:
            if asset_class is AssetClass.CRYPTO:
                request = CryptoBarsRequest(symbol_or_symbols=list(symbols), timeframe=tf, start=start)
                payload = self._with_retries(
                    "get_bars_batch", ",".join(symbols[:3]),
                    lambda: self._crypto.get_crypto_bars(request),
                )
                source = "alpaca:crypto"
            elif asset_class is AssetClass.US_OPTION:
                client = self._require_option_client(symbols[0])
                request = OptionBarsRequest(symbol_or_symbols=list(symbols), timeframe=tf, start=start)
                payload = self._with_retries(
                    "get_bars_batch", ",".join(symbols[:3]),
                    lambda: client.get_option_bars(request),
                )
                source = "alpaca:options"
            else:
                client = self._require_stock_client(symbols[0])
                request = StockBarsRequest(
                    symbol_or_symbols=list(symbols), timeframe=tf, start=start, feed=self._feed
                )
                payload = self._with_retries(
                    "get_bars_batch", ",".join(symbols[:3]),
                    lambda: client.get_stock_bars(request),
                )
                source = self.source_label
        except DataUnavailable:
            log.error("Batch bar fetch failed for %d symbols", len(symbols))
            return {}

        results: dict[str, BarSet] = {}
        data = payload.data if hasattr(payload, "data") else {}
        for symbol in symbols:
            raw_bars = data.get(symbol, [])
            if not raw_bars:
                continue
            bars = tuple(
                Bar(
                    timestamp=b.timestamp,
                    open=float(b.open), high=float(b.high), low=float(b.low),
                    close=float(b.close), volume=float(b.volume or 0.0),
                    trade_count=int(b.trade_count or 0),
                    vwap=float(b.vwap) if b.vwap is not None else None,
                )
                for b in raw_bars
            )
            bars = tuple(sorted(bars, key=lambda b: b.timestamp))[-limit:]
            results[symbol] = BarSet(
                symbol=symbol, asset_class=asset_class, timeframe=timeframe,
                bars=bars, freshness=Freshness.HISTORICAL, source=source,
            )
        return results

    # -- health ------------------------------------------------------------

    def health(self) -> dict[str, object]:
        """Connectivity and feed status for the dashboard.

        Probes crypto (credential-free) so a status is available even before
        keys are configured, and reports equity availability separately.
        """
        status: dict[str, object] = {
            "provider": self.name,
            "feed": self._feed_name,
            "freshness_label": self._freshness.value,
            "credentials_configured": self._has_credentials,
            "equity_data_available": self._has_credentials,
            "option_data_available": self._has_credentials,
            "futures_data_available": False,
            "futures_note": "Alpaca does not offer futures market data.",
        }
        try:
            probe = self.get_latest_quote("BTC/USD", AssetClass.CRYPTO)
            status["connected"] = True
            status["probe_symbol"] = "BTC/USD"
            status["probe_age_seconds"] = round(probe.age_seconds(), 2)
            status["last_checked"] = utcnow().isoformat()
        except Exception as exc:  # noqa: BLE001 - health must never raise
            status["connected"] = False
            status["error"] = str(exc)[:300]
            status["last_checked"] = utcnow().isoformat()
        return status
