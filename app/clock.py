"""Market sessions, timezone handling and staleness arithmetic.

Two sources of truth, in priority order:

1. The broker's own clock/calendar endpoint, when a connected broker is
   supplied. This is authoritative - it reflects unscheduled closures.
2. The offline NYSE exchange calendar, which correctly models weekends,
   holidays and half-days but cannot know about an unscheduled halt.

Every timestamp handled here is timezone-aware. A naive datetime is rejected
rather than being silently assumed to be UTC or local time, because that
assumption is exactly how session and staleness bugs are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

from app.enums import AssetClass, MarketSession

UTC = timezone.utc
DEFAULT_DISPLAY_TZ = "America/New_York"


class ClockError(RuntimeError):
    """Raised when a timestamp cannot be interpreted unambiguously."""


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def ensure_aware(value: datetime, *, field: str = "timestamp") -> datetime:
    """Return ``value`` in UTC, refusing naive datetimes.

    A naive datetime is an error, not something to guess about.
    """
    if not isinstance(value, datetime):
        raise ClockError(f"{field} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ClockError(
            f"{field} is timezone-naive. Timestamps must carry an explicit "
            "timezone so session and staleness checks are unambiguous."
        )
    return value.astimezone(UTC)


def to_display_tz(value: datetime, tz_name: str = DEFAULT_DISPLAY_TZ) -> datetime:
    """Convert an aware timestamp into the operator's display timezone."""
    return ensure_aware(value).astimezone(_zone(tz_name))


@lru_cache(maxsize=8)
def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception as exc:  # pragma: no cover - depends on host tz database
        raise ClockError(
            f"Unknown timezone {name!r}. On Windows the 'tzdata' package must "
            "be installed for the standard library to resolve IANA zones."
        ) from exc


def age_seconds(timestamp: datetime, *, now: datetime | None = None) -> float:
    """Seconds elapsed since ``timestamp``. Negative if it is in the future."""
    reference = ensure_aware(now) if now is not None else utcnow()
    return (reference - ensure_aware(timestamp)).total_seconds()


def is_stale(
    timestamp: datetime, max_age_seconds: float, *, now: datetime | None = None
) -> bool:
    """True when ``timestamp`` is older than the permitted age.

    A future-dated timestamp beyond a small tolerance is also treated as stale:
    it indicates clock skew, which we must not trade on.
    """
    elapsed = age_seconds(timestamp, now=now)
    if elapsed < -5.0:
        return True
    return elapsed > max_age_seconds


@dataclass(frozen=True)
class SessionInfo:
    """The market state for one asset class at one instant."""

    session: MarketSession
    as_of: datetime
    next_open: datetime | None = None
    next_close: datetime | None = None
    source: str = "exchange_calendar"

    @property
    def is_open_regular(self) -> bool:
        return self.session in (MarketSession.REGULAR, MarketSession.ALWAYS_OPEN)

    @property
    def is_tradable_extended(self) -> bool:
        """True in any session where an extended-hours order could rest."""
        return self.session in (
            MarketSession.REGULAR,
            MarketSession.PREMARKET,
            MarketSession.AFTER_HOURS,
            MarketSession.ALWAYS_OPEN,
        )


class MarketClock:
    """Resolves the market session for a given asset class and instant."""

    def __init__(self, *, calendar_name: str = "NYSE", display_tz: str = DEFAULT_DISPLAY_TZ) -> None:
        self._calendar_name = calendar_name
        self._calendar = mcal.get_calendar(calendar_name)
        self.display_tz = display_tz

    # -- schedule lookup ---------------------------------------------------

    def _schedule(self, day: date) -> pd.DataFrame:
        """Session boundaries for ``day``. Empty frame when the market is shut."""
        return self._calendar.schedule(
            start_date=day.isoformat(), end_date=day.isoformat(), market_times="all"
        )

    def _next_session_open(self, after: datetime, *, horizon_days: int = 10) -> datetime | None:
        start = after.astimezone(UTC).date()
        schedule = self._calendar.schedule(
            start_date=start.isoformat(),
            end_date=(start + timedelta(days=horizon_days)).isoformat(),
        )
        for open_ts in schedule["market_open"]:
            candidate = open_ts.to_pydatetime().astimezone(UTC)
            if candidate > after:
                return candidate
        return None

    # -- public API --------------------------------------------------------

    def session_for(
        self, asset_class: AssetClass, *, now: datetime | None = None
    ) -> SessionInfo:
        """Resolve the current market session for ``asset_class``."""
        moment = ensure_aware(now) if now is not None else utcnow()

        if asset_class is AssetClass.CRYPTO:
            return SessionInfo(
                session=MarketSession.ALWAYS_OPEN, as_of=moment, source="crypto_24_7"
            )

        if asset_class is AssetClass.FUTURES:
            # Futures sessions are contract- and venue-specific and are not
            # modelled by the NYSE calendar. Reporting REGULAR here would be a
            # fabrication, so the state is reported as closed/unsupported and
            # order validation rejects the asset class outright.
            return SessionInfo(
                session=MarketSession.CLOSED, as_of=moment, source="unsupported_asset_class"
            )

        # US equities and US options both follow the NYSE session calendar.
        local_day = moment.astimezone(_zone(self._calendar.tz.key if hasattr(self._calendar.tz, "key") else DEFAULT_DISPLAY_TZ)).date()
        schedule = self._schedule(local_day)

        if schedule.empty:
            return SessionInfo(
                session=MarketSession.CLOSED,
                as_of=moment,
                next_open=self._next_session_open(moment),
                source="exchange_calendar",
            )

        row = schedule.iloc[0]
        pre = row["pre"].to_pydatetime().astimezone(UTC)
        open_ = row["market_open"].to_pydatetime().astimezone(UTC)
        close = row["market_close"].to_pydatetime().astimezone(UTC)
        post = row["post"].to_pydatetime().astimezone(UTC)

        if open_ <= moment < close:
            session = MarketSession.REGULAR
        elif pre <= moment < open_:
            session = MarketSession.PREMARKET
        elif close <= moment < post:
            session = MarketSession.AFTER_HOURS
        else:
            session = MarketSession.CLOSED

        return SessionInfo(
            session=session,
            as_of=moment,
            next_open=open_ if moment < open_ else self._next_session_open(moment),
            next_close=close if moment < close else None,
            source="exchange_calendar",
        )

    def is_half_day(self, day: date) -> bool:
        """True when ``day`` is a scheduled early close."""
        schedule = self._schedule(day)
        if schedule.empty:
            return False
        close = schedule.iloc[0]["market_close"].to_pydatetime()
        local_close = close.astimezone(_zone(DEFAULT_DISPLAY_TZ))
        return (local_close.hour, local_close.minute) < (16, 0)
