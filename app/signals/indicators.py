"""Technical indicators.

Lookahead safety
----------------
Every function here returns a sequence the same length as its input, where
element ``i`` is computed **only** from elements ``0..i``. Warm-up positions
hold ``None``. No function may consult a future element, and none re-centres or
back-fills, because either would leak future information into a historical
decision and make a backtest unreproducible in live trading.

Values are plain floats/None rather than NaN so that "not yet computable" is
explicit and cannot silently propagate through arithmetic.
"""

from __future__ import annotations

from typing import Sequence

Series = Sequence[float]
OptSeries = list[float | None]


def _validate(values: Series, period: int, name: str) -> None:
    if period <= 0:
        raise ValueError(f"{name} period must be positive, got {period}")
    if not isinstance(values, Sequence):
        raise TypeError(f"{name} expects a sequence of floats")


def sma(values: Series, period: int) -> OptSeries:
    """Simple moving average."""
    _validate(values, period, "sma")
    out: OptSeries = []
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= period:
            running -= values[i - period]
        out.append(running / period if i >= period - 1 else None)
    return out


def ema(values: Series, period: int) -> OptSeries:
    """Exponential moving average.

    Seeded with the SMA of the first ``period`` values, which is the standard
    convention and avoids the unstable early values of a zero-seeded EMA.
    """
    _validate(values, period, "ema")
    out: OptSeries = [None] * len(values)
    if len(values) < period:
        return out
    multiplier = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    previous = seed
    for i in range(period, len(values)):
        previous = (values[i] - previous) * multiplier + previous
        out[i] = previous
    return out


def rsi(values: Series, period: int = 14) -> OptSeries:
    """Wilder's Relative Strength Index, 0-100."""
    _validate(values, period, "rsi")
    out: OptSeries = [None] * len(values)
    if len(values) <= period:
        return out

    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_from(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        # Wilder smoothing.
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_from(avg_gain, avg_loss)
    return out


def _rsi_from(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(highs: Series, lows: Series, closes: Series) -> OptSeries:
    """True range. Index 0 is None because it has no previous close."""
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("true_range requires highs, lows and closes of equal length")
    out: OptSeries = [None] * len(closes)
    for i in range(1, len(closes)):
        previous_close = closes[i - 1]
        out[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - previous_close),
            abs(lows[i] - previous_close),
        )
    return out


def atr(highs: Series, lows: Series, closes: Series, period: int = 14) -> OptSeries:
    """Wilder's Average True Range."""
    _validate(closes, period, "atr")
    tr = true_range(highs, lows, closes)
    out: OptSeries = [None] * len(closes)
    # First ATR is the simple mean of TR[1..period].
    if len(closes) <= period:
        return out
    window = [v for v in tr[1 : period + 1] if v is not None]
    if len(window) < period:
        return out
    previous = sum(window) / period
    out[period] = previous
    for i in range(period + 1, len(closes)):
        current_tr = tr[i]
        if current_tr is None:
            out[i] = previous
            continue
        previous = (previous * (period - 1) + current_tr) / period
        out[i] = previous
    return out


def relative_volume(volumes: Series, period: int = 20) -> OptSeries:
    """Current volume divided by the average of the *preceding* ``period`` bars.

    The current bar is excluded from its own baseline; including it would dampen
    exactly the spike the measure exists to detect.
    """
    _validate(volumes, period, "relative_volume")
    out: OptSeries = [None] * len(volumes)
    for i in range(period, len(volumes)):
        baseline = sum(volumes[i - period : i]) / period
        out[i] = (volumes[i] / baseline) if baseline > 0 else None
    return out


def percent_change(values: Series, lookback: int = 1) -> OptSeries:
    """Percent change versus ``lookback`` bars ago, expressed as a percentage."""
    _validate(values, lookback, "percent_change")
    out: OptSeries = [None] * len(values)
    for i in range(lookback, len(values)):
        previous = values[i - lookback]
        out[i] = ((values[i] - previous) / previous) * 100.0 if previous else None
    return out


def rolling_high(values: Series, period: int) -> OptSeries:
    """Highest value over the trailing ``period`` bars, inclusive of the current bar."""
    _validate(values, period, "rolling_high")
    out: OptSeries = [None] * len(values)
    for i in range(period - 1, len(values)):
        out[i] = max(values[i - period + 1 : i + 1])
    return out


def rolling_low(values: Series, period: int) -> OptSeries:
    """Lowest value over the trailing ``period`` bars, inclusive of the current bar."""
    _validate(values, period, "rolling_low")
    out: OptSeries = [None] * len(values)
    for i in range(period - 1, len(values)):
        out[i] = min(values[i - period + 1 : i + 1])
    return out


def last_value(series: OptSeries) -> float | None:
    """Most recent non-None value, or None if the series never warmed up."""
    for value in reversed(series):
        if value is not None:
            return value
    return None
