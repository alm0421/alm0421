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


def day_bars(bars: pd.DataFrame, day: date) -> pd.DataFrame:
    """Regular-session bars for one date, oldest first."""
    mask = bars.index.date == day
    return regular_session(bars[mask])


def premarket_bars(bars: pd.DataFrame, day: date) -> pd.DataFrame:
    mask = bars.index.date == day
    return bars[mask].between_time(PREMARKET_OPEN, time(9, 29))


def previous_session(bars: pd.DataFrame, day: date) -> pd.DataFrame:
    """Regular-session bars of the last trading date before ``day`` (may be empty)."""
    earlier = [d for d in sessions(bars) if d < day]
    if not earlier:
        return bars.iloc[0:0]
    return day_bars(bars, earlier[-1])


def iter_days(bars: pd.DataFrame) -> Iterator[tuple[date, pd.DataFrame]]:
    for day in sessions(bars):
        yield day, day_bars(bars, day)


def save_minute_bars(bars: pd.DataFrame, path: Path) -> None:
    """Write bars in the canonical CSV layout (UTC bar-start timestamps)."""
    out = bars[_COLUMNS].copy()
    out.insert(0, "time_utc", bars.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def fetch_alpaca_minute_bars(
    symbol: str,
    start: date,
    end: date,
    api_key: str,
    secret_key: str,
    feed: str = "iex",
) -> pd.DataFrame:  # pragma: no cover - needs network and credentials
    """Download 1-minute bars from Alpaca market data, paging automatically.

    Alpaca's free IEX feed reaches back to 2016 and needs only paper-account
    keys, which makes it the practical source for multi-year 1-minute history.
    IEX is a single venue (~2-3% of consolidated volume), so bar volumes are far
    smaller than the tape and highs/lows can differ slightly from SIP.
    """
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=symbol.upper(),
        timeframe=TimeFrame.Minute,
        start=pd.Timestamp(start, tz=ET).tz_convert("UTC").to_pydatetime(),
        end=pd.Timestamp(end, tz=ET).tz_convert("UTC").to_pydatetime() + pd.Timedelta(days=1),
        feed=DataFeed(feed),
    )
    frame = client.get_stock_bars(request).df
    if frame.empty:
        raise BarDataError(f"Alpaca returned no bars for {symbol} {start}..{end}")
    frame = frame.reset_index()
    frame = frame[frame["symbol"] == symbol.upper()]
    index = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(ET)
    bars = frame[_COLUMNS].astype(float).set_index(index)
    bars.index.name = "time_et"
    return bars.sort_index()
