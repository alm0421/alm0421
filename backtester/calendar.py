"""NYSE trading calendar: the published (scheduled) schedule, point in time.

History uses the bars themselves as the calendar. Anything that asks "is this the last trading day of
the month/week/quarter?" about a bar needs to know the sessions that come next, which the data can't tell
yet. That is what this module is for.

The schedule is the regular holiday rules (the historical ones before 1971: Washington's Birthday on Feb 22,
Memorial Day on May 30, Lincoln's Birthday to 1953 and in 1968, Columbus Day 1909-1953, Armistice/Veterans Day 1934-1953 and 1968, no Friday closure for a Saturday holiday before 1954,
Election Day every year to 1968 and in presidential years to 1980, Thanksgiving on the last Thursday before
1939) plus the closures announced in advance (state funerals: SPECIAL_CLOSURES), each known from its
announcement day on - so a question asked on day D uses only the schedule as published on D. Closures
nobody could know in advance (9/11, Hurricane Sandy) are never in the schedule: on 2001-09-10 the next
scheduled session was 2001-09-11. Saturday sessions (to 1952) are not modelled: the data is weekday bars.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd


def _observed_modern(d: dt.date) -> dt.date | None:
    if d.weekday() == 5:  # Saturday -> Friday, except New Year's Day (NYSE does not close Dec 31)
        return None if (d.month, d.day) == (1, 1) else d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1) if month < 12 else dt.date(year, 12, 31)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _easter(y: int) -> dt.date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(y, month, day)


# Every NYSE closure on a weekday that is not a regular holiday: {closed day: the day the closure was announced}.
# From the announcement on, the schedule knows it (e.g. on 2018-12-03 the week of 2018-12-05 had four sessions);
# before it, the day was a scheduled session. Unscheduled closures (9/11, blackouts, storms) are "announced" on the
# day itself (or the evening/weekend before, when that was so): no session before them knew. The list matches the
# weekdays missing from the ^GSPC / SPYSIM bars since 1950 (tests/test_calendar_closures.py). Where the exact
# announcement day is uncertain the latest plausible one is used (the schedule never knows a closure too early).
_PAPERWORK_1968 = {   # the 1968 paperwork crisis: Wednesdays closed, voted in batches
    **{dt.date(1968, 6, d): dt.date(1968, 6, 10) for d in (12, 19, 26)},       # first batch (voted early June 1968)
    dt.date(1968, 7, 5): dt.date(1968, 6, 10),                                 # ... with the Friday after July 4
    **{dt.date(1968, 7, d): dt.date(1968, 6, 25) for d in (10, 17, 24, 31)},   # NYT 1968-06-26 "extend closings 4 weeks"
    **{dt.date(1968, 8, d): dt.date(1968, 7, 31) for d in (7, 14, 21, 28)},
    **{dt.date(1968, 9, d): dt.date(1968, 8, 30) for d in (11, 18, 25)},       # reviewed by the exchanges 1968-08-23
    **{dt.date(1968, 10, d): dt.date(1968, 9, 27) for d in (2, 9, 16, 23, 30)},
    dt.date(1968, 11, 20): dt.date(1968, 11, 15),
    **{dt.date(1968, 12, d): dt.date(1968, 11, 29) for d in (4, 11, 18)},      # the last ones (4-day weeks from 1969)
}
SPECIAL_CLOSURES: dict[dt.date, dt.date] = {
    dt.date(1956, 12, 24): dt.date(1956, 12, 14),  # Christmas Eve (Monday), closed by board vote
    dt.date(1958, 12, 26): dt.date(1958, 12, 12),  # day after Christmas (Friday)
    dt.date(1961, 5, 29): dt.date(1961, 5, 12),    # day before Memorial Day (Monday)
    dt.date(1963, 11, 25): dt.date(1963, 11, 23),  # President Kennedy's funeral (day of mourning proclaimed Nov 23)
    dt.date(1968, 4, 9): dt.date(1968, 4, 8),      # Dr. Martin Luther King Jr.'s funeral, national day of mourning
    **_PAPERWORK_1968,
    dt.date(1969, 2, 10): dt.date(1969, 2, 10),    # snowstorm (unscheduled)
    dt.date(1969, 3, 31): dt.date(1969, 3, 29),    # President Eisenhower's funeral (died March 28)
    dt.date(1969, 7, 21): dt.date(1969, 7, 17),    # Apollo 11 moon landing, national day of participation
    dt.date(1972, 12, 28): dt.date(1972, 12, 27),  # President Truman's funeral (died December 26)
    dt.date(1973, 1, 25): dt.date(1973, 1, 23),    # President Johnson's funeral (died January 22)
    dt.date(1977, 7, 14): dt.date(1977, 7, 14),    # New York City blackout (unscheduled)
    dt.date(1985, 9, 27): dt.date(1985, 9, 27),    # Hurricane Gloria (decided that morning)
    dt.date(1994, 4, 27): dt.date(1994, 4, 24),    # President Nixon's funeral (died April 22; the closure was set over
                                                   # the weekend and reported on Monday April 25)
    dt.date(2001, 9, 11): dt.date(2001, 9, 11),    # September 11 attacks (unscheduled) ...
    dt.date(2001, 9, 12): dt.date(2001, 9, 11),    # ... each further day announced the day before
    dt.date(2001, 9, 13): dt.date(2001, 9, 12),
    dt.date(2001, 9, 14): dt.date(2001, 9, 13),
    dt.date(2004, 6, 11): dt.date(2004, 6, 6),     # President Reagan's funeral (Reagan died June 5)
    dt.date(2007, 1, 2): dt.date(2006, 12, 27),    # President Ford's national day of mourning
    dt.date(2012, 10, 29): dt.date(2012, 10, 28),  # Hurricane Sandy (announced Sunday October 28) ...
    dt.date(2012, 10, 30): dt.date(2012, 10, 29),  # ... and extended on the 29th
    dt.date(2018, 12, 5): dt.date(2018, 12, 1),    # President G. H. W. Bush's national day of mourning
    dt.date(2025, 1, 9): dt.date(2024, 12, 30),    # President Carter's national day of mourning
}


# Fridays before a Saturday holiday on which the NYSE traded anyway (from 1954 the Friday was usually closed)
_FRIDAYS_OPEN = {dt.date(1958, 2, 21), dt.date(1959, 5, 29), dt.date(1970, 5, 29)}


def _election_day(year: int) -> dt.date:
    return _nth_weekday(year, 11, 0, 1) + dt.timedelta(days=1)   # the Tuesday after the first Monday


@lru_cache(maxsize=None)
def holidays(year: int) -> frozenset:
    """The regular NYSE holidays of `year` on the rules of that year (not the special closures)."""
    modern = year >= 1971          # Uniform Monday Holiday Act

    def _observed(d: dt.date) -> dt.date | None:
        # a holiday on a Saturday closed no weekday before 1954 (Saturday sessions ran to 1952; 1953-07-03 traded),
        # and in a few later years the Friday stayed open too
        if d.weekday() == 5 and (year < 1954 or d - dt.timedelta(days=1) in _FRIDAYS_OPEN):
            return None
        return _observed_modern(d)
    out = [
        _observed(dt.date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3) if year >= 1998 else None,     # Martin Luther King Jr. Day
        _observed(dt.date(year, 2, 12)) if (year <= 1953 or year == 1968) else None,  # Lincoln's Birthday
        _nth_weekday(year, 2, 0, 3) if modern else _observed(dt.date(year, 2, 22)),   # Washington's Birthday
        None if year in (1898, 1906, 1907) else _easter(year) - dt.timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0) if modern else _observed(dt.date(year, 5, 30)),     # Memorial Day
        _observed(dt.date(year, 6, 19)) if year >= 2022 else None,  # Juneteenth
        _observed(dt.date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),                               # Labor Day
        _observed(dt.date(year, 10, 12)) if 1909 <= year <= 1953 else None,           # Columbus Day
        _election_day(year) if (year <= 1968 or (year <= 1980 and year % 4 == 0)) else None,
        _observed(dt.date(year, 11, 11)) if (1934 <= year <= 1953 or year == 1968) else None,  # Armistice/Veterans Day
        (_nth_weekday(year, 11, 3, 4) if year >= 1942 else                           # Thanksgiving
         _last_weekday(year, 11, 3) - dt.timedelta(days=7) if year >= 1939 else _last_weekday(year, 11, 3)),
        _observed(dt.date(year, 12, 25)),
    ]
    return frozenset(d for d in out if d is not None and d.weekday() < 5)


def closures(year: int, as_of=None) -> frozenset:
    """The scheduled closures of `year` as published on `as_of` (default: today - every announced closure)."""
    a = pd.Timestamp(as_of).date() if as_of is not None else None
    sp = {d for d, ann in SPECIAL_CLOSURES.items() if d.year == year and (a is None or ann <= a)}
    return holidays(year) | frozenset(sp)


def is_session(d, as_of=None) -> bool:
    """A scheduled NYSE session: every closure counted (default), or only those announced by `as_of`."""
    d = pd.Timestamp(d).date()
    return d.weekday() < 5 and d not in closures(d.year, as_of)


def next_sessions(after, n: int = 1) -> pd.DatetimeIndex:
    """The next `n` NYSE sessions strictly after `after`, on the schedule as published on `after` (a closure
    announced later - 9/11, Hurricane Sandy - is not known then)."""
    d = pd.Timestamp(after).normalize()
    asof = d
    out = []
    while len(out) < n:
        d += pd.Timedelta(days=1)
        if is_session(d, asof):
            out.append(d)
    return pd.DatetimeIndex(out)


def anchor_day(first, traded: pd.DatetimeIndex | None = None) -> pd.Timestamp:
    """The day an equity curve's starting-capital row sits on: the session before `first` on the traded
    calendar `traded` (the union of the run's price dates), or - when the data starts at `first` - the
    previous NYSE session (never a weekend or holiday such as New Year's Day)."""
    first = pd.Timestamp(first)
    if traded is not None and len(traded):
        k = int(traded.searchsorted(first))
        if k > 0:
            return pd.Timestamp(traded[k - 1])
    d = first.normalize() - pd.Timedelta(days=1)
    for _ in range(10):
        if is_session(d):
            return d
        d -= pd.Timedelta(days=1)
    return first - pd.Timedelta(days=1)


def extend(idx: pd.DatetimeIndex, n: int = 70) -> pd.DatetimeIndex:
    """`idx` followed by the next `n` sessions (for period-end logic at the data edge)."""
    if len(idx) == 0:
        return idx
    return idx.append(next_sessions(idx[-1], n))


def _holiday_array(start_year: int, end_year: int, specials: bool = True, skip=()) -> np.ndarray:
    days = {d for y in range(start_year, end_year + 1) for d in holidays(y)}
    if specials:
        days |= {d for d in SPECIAL_CLOSURES if start_year <= d.year <= end_year and d not in skip}
    return np.array(sorted(days), dtype="datetime64[D]")


def _point_in_time(idx: pd.DatetimeIndex, fn):
    """fn(dates as datetime64[D], holiday array) evaluated with the schedule as published on each date: every
    special closure counted, except - for the dates before its announcement (and within 120 days of it) - the
    ones not yet announced."""
    d = idx.values.astype("datetime64[D]")
    y0, y1 = int(idx.min().year) - 1, int(idx.max().year) + 1
    out = fn(d, _holiday_array(y0, y1))
    days, early = [], []
    for day, ann in SPECIAL_CLOSURES.items():
        if not (y0 <= day.year <= y1):
            continue
        e = (d < np.datetime64(ann)) & (d >= np.datetime64(day - dt.timedelta(days=120)))
        if e.any():
            days.append(day)
            early.append(e)
    if not days:
        return out
    # each date without the closures not yet announced on it (several can be pending at once: 9/11-9/14 2001)
    E = np.vstack(early)
    groups: dict[tuple, list[int]] = {}
    for i in np.flatnonzero(E.any(axis=0)):
        groups.setdefault(tuple(np.flatnonzero(E[:, i])), []).append(int(i))
    for ks, rows in groups.items():
        rows_a = np.array(rows)
        out[rows_a] = fn(d[rows_a], _holiday_array(y0, y1, skip={days[k] for k in ks}))
    return out


def next_scheduled(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """For each date, the next session on the NYSE schedule as published that day (weekends, holidays and the
    closures announced by then skipped).

    Unscheduled closures (9/11, hurricanes) are deliberately not known in advance: on 2001-09-10 the next
    scheduled session was 2001-09-11. A pre-announced closure is known from its announcement: on 2018-12-04 the
    next session was 2018-12-06 (the Bush day of mourning was announced on 2018-12-01), on 2018-11-30 it was not
    yet known."""
    if len(idx) == 0:
        return idx
    nxt = _point_in_time(idx, lambda d, hol: np.busday_offset(d, 1, roll="forward", holidays=hol))
    return pd.DatetimeIndex(nxt.astype("datetime64[ns]"))


def scheduled_period_end(idx: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True where the date is the last scheduled session of its week/month/quarter/year."""
    if len(idx) == 0:
        return np.zeros(0, bool)
    return np.asarray(next_scheduled(idx).to_period(freq) != idx.to_period(freq))


def scheduled_sessions_left(idx: pd.DatetimeIndex, freq: str = "M") -> np.ndarray:
    """Scheduled sessions AFTER each date to the end of its period (0 on the period's last scheduled session),
    on the schedule as published that day. Known at the open: it depends only on the published schedule."""
    if len(idx) == 0:
        return np.zeros(0, int)
    end = (idx.to_period(freq).end_time.normalize() + pd.Timedelta(days=1)).values.astype("datetime64[D]")
    pos = {k: j for j, k in enumerate(idx.values.astype("datetime64[D]"))}

    def count(d, hol):
        e = end[[pos[x] for x in d]]
        return np.busday_count(d + np.timedelta64(1, "D"), e, holidays=hol)
    return _point_in_time(idx, count)
