"""NYSE trading calendar for dates beyond the last bar of data.

History uses the bars themselves as the calendar. Anything that asks "is this the last trading day of
the month/week/quarter?" about the latest bar needs to know the sessions that come next, which the
data can't tell us yet. That is what this module is for.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas as pd


def _observed(d: dt.date) -> dt.date | None:
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


@lru_cache(maxsize=None)
def holidays(year: int) -> frozenset:
    out = [
        _observed(dt.date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3) if year >= 1998 else None,     # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),                               # Washington's Birthday
        _easter(year) - dt.timedelta(days=2),                      # Good Friday
        _last_weekday(year, 5, 0),                                 # Memorial Day
        _observed(dt.date(year, 6, 19)) if year >= 2022 else None,  # Juneteenth
        _observed(dt.date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),                               # Labor Day
        _nth_weekday(year, 11, 3, 4),                              # Thanksgiving
        _observed(dt.date(year, 12, 25)),
    ]
    return frozenset(d for d in out if d is not None)


def is_session(d) -> bool:
    d = pd.Timestamp(d).date()
    return d.weekday() < 5 and d not in holidays(d.year)


def next_sessions(after, n: int = 1) -> pd.DatetimeIndex:
    """The next `n` NYSE sessions strictly after `after`."""
    d = pd.Timestamp(after).normalize()
    out = []
    while len(out) < n:
        d += pd.Timedelta(days=1)
        if is_session(d):
            out.append(d)
    return pd.DatetimeIndex(out)


def extend(idx: pd.DatetimeIndex, n: int = 70) -> pd.DatetimeIndex:
    """`idx` followed by the next `n` sessions (for period-end logic at the data edge)."""
    if len(idx) == 0:
        return idx
    return idx.append(next_sessions(idx[-1], n))
