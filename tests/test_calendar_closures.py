"""NYSE special closures: complete against the index bars, and known only from their announcement."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from backtester import calendar as cal, data


def test_nixon_funeral_closure():
    assert dt.date(1994, 4, 27) in cal.SPECIAL_CLOSURES
    left = cal.scheduled_sessions_left(pd.DatetimeIndex(["1994-04-26"]), "M")
    assert int(left[0]) == 2                                  # April 28 and 29 (the 27th closed)
    # before the announcement the 27th was a scheduled session
    left = cal.scheduled_sessions_left(pd.DatetimeIndex(["1994-04-22"]), "M")
    assert int(left[0]) == 5
    assert not cal.is_session("1994-04-27")


def test_unscheduled_closures_are_not_known_before_they_happen():
    nxt = cal.next_scheduled(pd.DatetimeIndex(["2001-09-10", "2012-10-26", "1977-07-13", "1985-09-26"]))
    assert [str(d.date()) for d in nxt] == ["2001-09-11", "2012-10-29", "1977-07-14", "1985-09-27"]
    assert [str(d.date()) for d in cal.next_sessions("2001-09-10", 2)] == ["2001-09-11", "2001-09-12"]
    assert not cal.is_session("2001-09-12") and cal.is_session("2001-09-12", as_of="2001-09-10")


def test_1968_paperwork_wednesdays():
    wed = [d for d in cal.SPECIAL_CLOSURES if d.year == 1968 and d.weekday() == 2]
    assert len(wed) == 23 and min(wed) == dt.date(1968, 6, 12) and max(wed) == dt.date(1968, 12, 18)
    # on 1968-06-24 the July Wednesdays were not yet voted (1968-06-25)
    assert int(cal.scheduled_sessions_left(pd.DatetimeIndex(["1968-06-24"]), "M")[0]) == 3   # 25, 27, 28
    assert int(cal.scheduled_sessions_left(pd.DatetimeIndex(["1968-06-27"]), "M")[0]) == 1


@pytest.mark.parametrize("ticker", ["^GSPC", "SPYSIM"])
def test_closures_match_the_index_bars(ticker):
    path = data.PRICES / f"{ticker}.csv"
    if not path.exists():
        pytest.skip("no data")
    idx = pd.read_csv(path, parse_dates=["date"], index_col="date").index
    lo = max(idx[0], pd.Timestamp("1952-01-01"))
    have = set(idx)
    days = pd.bdate_range(lo, idx[-1])
    # no bar on any closure, and every weekday without a bar is a holiday or a listed closure
    assert not [d for d in idx if d >= lo and d.weekday() < 5 and d.date() in cal.closures(d.year)]
    missing = [str(d.date()) for d in days if d not in have and d.date() not in cal.closures(d.year)]
    assert not missing, missing
