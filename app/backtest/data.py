"""Load 1-minute bars for backtesting.

Bars are stored one CSV per symbol as ``<data_dir>/<SYMBOL>_1min.csv`` with
columns ``time_utc, open, high, low, close, volume``. ``time_utc`` is the bar's
*start* time in ISO-8601 UTC, which is what both IBKR and Alpaca return.

Everything downstream works in America/New_York, because every rule in an
opening-drive strategy ("before 9:59", "the first 30 minutes") is stated in
exchange time and DST shifts the UTC offset twice a year.
"""

from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Iterator

import pandas as pd

ET = "America/New_York"
RTH_OPEN = time(9, 30)
RTH_LAST_BAR = time(15, 59)
PREMARKET_OPEN = time(4, 0)

_COLUMNS = ["open", "high", "low", "close", "volume"]


class BarDataError(RuntimeError):
    """Raised when bar data is missing or malformed."""


def csv_path(data_dir: Path, symbol: str) -> Path:
    return Path(data_dir) / f"{symbol.upper()}_1min.csv"


def load_minute_bars(path: Path) -> pd.DataFrame:
    """Load one symbol's bars, indexed by bar-start time in US/Eastern."""
    path = Path(path)
    if not path.exists():
        raise BarDataError(f"No bar file at {path}")
    frame = pd.read_csv(path)
    time_col = "time_utc" if "time_utc" in frame.columns else "timestamp"
    missing = [c for c in [time_col, *_COLUMNS] if c not in frame.columns]
    if missing:
        raise BarDataError(f"{path} is missing columns {missing}")
    index = pd.to_datetime(frame[time_col], utc=True).dt.tz_convert(ET)
    bars = frame[_COLUMNS].astype(float).set_index(index)
    bars.index.name = "time_et"
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    # IBKR pads empty minutes with a zero-volume bar repeating the prior price.
    # Those are harmless for highs/lows, so they are kept rather than guessed at.
    return bars


def load_universe(data_dir: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    return {s.upper(): load_minute_bars(csv_path(data_dir, s)) for s in symbols}


def sessions(bars: pd.DataFrame) -> list[date]:
    """Trading dates that have at least one regular-session bar."""
    rth = regular_session(bars)
    return sorted(set(rth.index.date))


def regular_session(bars: pd.DataFrame) -> pd.DataFrame:
    return bars.between_time(RTH_OPEN, RTH_LAST_BAR)


def _calendar_day(bars: pd.DataFrame, day: date) -> pd.DataFrame:
    """All of one calendar date's bars via the sorted index (O(log n)).

    ``bars.index.date == day`` rebuilds a Python-date array over every row on
    each call, which is O(n) per day and O(n^2) across a multi-year scan. The
    index is sorted in :func:`load_minute_bars`, so partial-string label
    indexing slices the same rows without touching the rest of the frame.
    """
    try:
        same_day = bars.loc[day.isoformat()]
    except KeyError:
        return bars.iloc[0:0]
    # A full-timestamp label would yield a Series; a date string never does,
    # but guard anyway so callers always get a frame.
    if isinstance(same_day, pd.Series):
        same_day = same_day.to_frame().T
    return same_day


def day_bars(bars: pd.DataFrame, day: date) -> pd.DataFrame:
    """Regular-session bars for one date, oldest first."""
    return regular_session(_calendar_day(bars, day))


def premarket_bars(bars: pd.DataFrame, day: date) -> pd.DataFrame:
    return _calendar_day(bars, day).between_time(PREMARKET_OPEN, time(9, 29))


def previous_session(bars: pd.DataFrame, day: date) -> pd.DataFrame:
    """Regular-session bars of the last trading date before ``day`` (may be empty).

    Walks back through the sorted index with ``searchsorted`` (O(log n)) rather
    than recomputing the full session list on every call, which is what made a
    multi-year scan quadratic once a setup was found on most days.
    """
    idx = bars.index
    pos = idx.searchsorted(pd.Timestamp(day, tz=ET))  # first bar on/after `day`
    while pos > 0:
        prev_date = idx[pos - 1].date()
        rth = day_bars(bars, prev_date)
        if not rth.empty:  # skip calendar dates that only had premarket bars
            return rth
        pos = idx.searchsorted(pd.Timestamp(prev_date, tz=ET))
    return bars.iloc[0:0]


def iter_days(bars: pd.DataFrame) -> Iterator[tuple[date, pd.DataFrame]]:
    # Group the regular session by calendar date once (computing the date array
    # a single time) rather than re-slicing the whole frame per day, which keeps
    # a multi-year scan linear instead of quadratic.
    rth = regular_session(bars)
    for day, group in rth.groupby(rth.index.date, sort=True):
        yield day, group


def save_minute_bars(bars: pd.DataFrame, path: Path) -> None:
    """Write bars in the canonical CSV layout (UTC bar-start timestamps)."""
    out = bars[_COLUMNS].copy()
    out.insert(0, "time_utc", bars.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"


def fetch_alpaca_minute_bars(
    symbol: str,
    start: date,
    end: date,
    api_key: str | None = None,
    secret_key: str | None = None,
    feed: str = "iex",
) -> pd.DataFrame:  # pragma: no cover - needs network and credentials
    """Download 1-minute bars from Alpaca's market-data REST API, paging automatically.

    Authentication has two supported paths:

    * Pass ``api_key`` / ``secret_key`` and they are sent as the
      ``APCA-API-KEY-ID`` / ``APCA-API-SECRET-KEY`` headers Alpaca expects.
    * Leave them ``None`` and let the environment's outbound proxy inject those
      headers for ``data.alpaca.markets`` (a stored API credential). The keys
      then never touch this process.

    The REST endpoint is used directly rather than the SDK so the proxy path
    works. Alpaca's free IEX feed reaches back to 2016 and needs only paper-account
    keys. IEX is a single venue (~2-3% of consolidated volume), so bar volumes are
    far smaller than the tape and highs/lows can differ slightly from SIP.
    """
    import requests

    headers = {"Accept": "application/json"}
    if api_key and secret_key:
        headers["APCA-API-KEY-ID"] = api_key
        headers["APCA-API-SECRET-KEY"] = secret_key

    sym = symbol.upper()
    params = {
        "symbols": sym,
        "timeframe": "1Min",
        "start": pd.Timestamp(start, tz=ET).tz_convert("UTC").isoformat(),
        "end": (pd.Timestamp(end, tz=ET).tz_convert("UTC") + pd.Timedelta(days=1)).isoformat(),
        "limit": 10000,
        "feed": feed,
        "adjustment": "all",
        "sort": "asc",
    }

    rows: list[dict] = []
    page_token: str | None = None
    while True:
        if page_token:
            params["page_token"] = page_token
        resp = requests.get(ALPACA_DATA_URL, params=params, headers=headers, timeout=60)
        if resp.status_code in (401, 403):
            raise BarDataError(
                f"Alpaca rejected the request for {sym} ({resp.status_code}). Check that the "
                "APCA-API-KEY-ID / APCA-API-SECRET-KEY credential is set for data.alpaca.markets."
            )
        resp.raise_for_status()
        payload = resp.json()
        rows.extend((payload.get("bars") or {}).get(sym) or [])
        page_token = payload.get("next_page_token")
        if not page_token:
            break

    if not rows:
        raise BarDataError(f"Alpaca returned no bars for {sym} {start}..{end}")
    frame = pd.DataFrame(rows)
    index = pd.to_datetime(frame["t"], utc=True).dt.tz_convert(ET)
    bars = pd.DataFrame(
        {
            "open": frame["o"].astype(float),
            "high": frame["h"].astype(float),
            "low": frame["l"].astype(float),
            "close": frame["c"].astype(float),
            "volume": frame["v"].astype(float),
        },
        index=index,
    )
    bars.index.name = "time_et"
    return bars[~bars.index.duplicated(keep="last")].sort_index()
