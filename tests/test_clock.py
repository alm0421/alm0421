"""Market sessions, timezones and staleness."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.clock import ClockError, MarketClock, ensure_aware, is_stale, to_display_tz
from app.enums import AssetClass, MarketSession

UTC = timezone.utc


@pytest.fixture()
def clock() -> MarketClock:
    return MarketClock()


@pytest.mark.parametrize(
    "moment,expected",
    [
        (datetime(2026, 9, 16, 12, 0, tzinfo=UTC), MarketSession.PREMARKET),
        (datetime(2026, 9, 16, 15, 0, tzinfo=UTC), MarketSession.REGULAR),
        (datetime(2026, 9, 16, 21, 0, tzinfo=UTC), MarketSession.AFTER_HOURS),
        (datetime(2026, 9, 16, 5, 0, tzinfo=UTC), MarketSession.CLOSED),
        (datetime(2026, 9, 19, 15, 0, tzinfo=UTC), MarketSession.CLOSED),  # Saturday
        (datetime(2026, 9, 20, 15, 0, tzinfo=UTC), MarketSession.CLOSED),  # Sunday
    ],
)
def test_equity_sessions(clock, moment, expected):
    assert clock.session_for(AssetClass.US_EQUITY, now=moment).session is expected


def test_market_holiday_is_closed(clock):
    """2026-07-03 is the observed Independence Day holiday."""
    moment = datetime(2026, 7, 3, 15, 0, tzinfo=UTC)
    assert clock.session_for(AssetClass.US_EQUITY, now=moment).session is MarketSession.CLOSED


def test_crypto_is_always_open(clock):
    saturday = datetime(2026, 9, 19, 3, 0, tzinfo=UTC)
    info = clock.session_for(AssetClass.CRYPTO, now=saturday)
    assert info.session is MarketSession.ALWAYS_OPEN
    assert info.is_open_regular


def test_futures_reported_closed_not_fabricated(clock):
    """Futures sessions are not modelled, so the clock must not claim they are open."""
    moment = datetime(2026, 9, 16, 15, 0, tzinfo=UTC)
    info = clock.session_for(AssetClass.FUTURES, now=moment)
    assert info.session is MarketSession.CLOSED
    assert info.source == "unsupported_asset_class"


def test_naive_datetime_rejected():
    with pytest.raises(ClockError, match="timezone-naive"):
        ensure_aware(datetime(2026, 9, 16, 15, 0))


def test_staleness_boundaries():
    base = datetime(2026, 9, 16, 15, 0, tzinfo=UTC)
    assert not is_stale(base, 60, now=base + timedelta(seconds=30))
    assert not is_stale(base, 60, now=base + timedelta(seconds=60))
    assert is_stale(base, 60, now=base + timedelta(seconds=61))


def test_future_timestamp_treated_as_stale():
    """Clock skew must not be mistaken for very fresh data."""
    base = datetime(2026, 9, 16, 15, 0, tzinfo=UTC)
    assert is_stale(base + timedelta(minutes=5), 60, now=base)


def test_display_timezone_conversion():
    moment = datetime(2026, 9, 16, 15, 0, tzinfo=UTC)
    local = to_display_tz(moment, "America/New_York")
    assert local.hour == 11  # EDT in September


def test_next_open_is_in_the_future(clock):
    saturday = datetime(2026, 9, 19, 15, 0, tzinfo=UTC)
    info = clock.session_for(AssetClass.US_EQUITY, now=saturday)
    assert info.next_open is not None and info.next_open > saturday
