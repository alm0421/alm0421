"""The HitchHiker scalp, as a mechanical rule set on 1-minute bars.

Source: "HitchHiker Scalp cheat sheet". Every discretionary phrase in the sheet is
mapped to an explicit, tunable parameter below. The mapping is the
backtest's central assumption, so it is spelled out here rather than hidden:

===============================  ============================================
Cheat-sheet wording              Mechanical rule (HitchhikerParams field)
===============================  ============================================
"distinct drive off the open"    Before the consolidation starts, price has
                                 moved from the session open by at least
                                 ``min_drive_multiple`` x the consolidation's
                                 own height (``require_drive``).
"holds up and moves sideways",   The ``min_consol_bars``..``max_consol_bars``
"5 to 20 minute consolidation"   bars immediately before the trigger bar,
                                 taking the longest window whose height is at
                                 most ``max_consol_range_frac`` of the session
                                 range so far.
"low of consolidation in upper   Consolidation low >= session low +
1/3 of the day's range"          (1 - ``location_frac``) x session range
                                 (``require_location``).
"aggressively buy the break,     Buy-stop at consolidation high + 1 tick.
don't wait for the bar close"    Filled at that price, or at the bar's open if
                                 it gaps through.
"stop .02 below the low of the   Stop = consolidation low - ``stop_offset``.
consolidation; one-and-done"     Hard stop; one trade per symbol per day.
"exit 1/2 into the first wave    Wave 1 ends after ``wave_pause_bars``
... when the first rush slows"   consecutive bars fail to make a new extreme;
                                 half is sold at that bar's close.
"exit 1/2 into the second wave"  After the rest, once price exceeds the wave-1
                                 extreme, wave 2 ends by the same pause rule;
                                 the rest is sold at that bar's close.
(not in the sheet)               Time stop after ``max_hold_minutes`` and a
                                 flat-by-close rule, so no trade runs forever.
===============================  ============================================

Shorts are the exact mirror. The implementation achieves that by negating
prices (high <-> -low) and running the long logic, so the two sides can never
drift apart.

Intrabar ordering is unknowable from OHLC bars. Wherever a bar could have hit
both the stop and something favourable, the stop is assumed to come first.

Lookahead: setup detection at bar ``j`` reads only bars ``< j`` plus bar ``j``'s
high/low for the trigger itself. The trigger bar's *volume* is recorded for the
report (the sheet's "30% more volume" factor) but is never used to decide a
trade, because it is not known until that bar closes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import Mapping

import pandas as pd

from app.enums import SignalDirection

LONG = SignalDirection.LONG
SHORT = SignalDirection.SHORT


@dataclass(frozen=True)
class HitchhikerParams:
    min_consol_bars: int = 5
    max_consol_bars: int = 20
    max_consol_range_frac: float = 0.5
    location_frac: float = 1 / 3
    require_location: bool = True
    min_drive_multiple: float = 1.5
    require_drive: bool = True
    stop_offset: float = 0.02
    entry_tick: float = 0.01
    wave_pause_bars: int = 2
    max_hold_minutes: int = 60
    flat_by: time = time(15, 55)
    #: Scanner mode: earliest and latest trigger-bar start times.
    scan_start: time = time(9, 35)
    scan_end: time = time(10, 30)
    #: Alert mode: minutes after the alert during which a break may trigger.
    alert_window_minutes: int = 30
    #: Require SPY/QQQ(/IWM) to be trading on the same side of their open.
    require_market_alignment: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.min_consol_bars <= self.max_consol_bars:
            raise ValueError("need 1 <= min_consol_bars <= max_consol_bars")
        if not 0 < self.max_consol_range_frac <= 1:
            raise ValueError("max_consol_range_frac must be in (0, 1]")
        if not 0 < self.location_frac <= 1:
            raise ValueError("location_frac must be in (0, 1]")
        if self.wave_pause_bars < 1 or self.max_hold_minutes < 1:
            raise ValueError("wave_pause_bars and max_hold_minutes must be >= 1")


@dataclass(frozen=True)
class Setup:
    symbol: str
    day: date
    direction: SignalDirection
    trigger_time: pd.Timestamp
    entry_price: float
    stop_price: float
    consol_start: pd.Timestamp
    consol_bars: int
    consol_high: float
    consol_low: float
    session_high: float
    session_low: float
    drive: float
    in_location: bool
    drive_ok: bool
    #: Trigger-bar volume / previous-bar volume. Informational only (lookahead).
    volume_ratio: float | None
    source: str = "scan"
    alert_time: pd.Timestamp | None = None
    notes: tuple[str, ...] = ()

    @property
    def sign(self) -> int:
        return 1 if self.direction is LONG else -1

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_price)


@dataclass(frozen=True)
class ExitLeg:
    time: pd.Timestamp
    fraction: float
    price: float
    reason: str


@dataclass
class TradePath:
    """One simulated trade, independent of position size."""

    setup: Setup
    legs: list[ExitLeg]
    #: Close of every bar from the trigger bar to the final exit bar, for marking.
    marks: pd.Series
    context: dict = field(default_factory=dict)

    @property
    def exit_time(self) -> pd.Timestamp:
        return self.legs[-1].time

    @property
    def avg_exit_price(self) -> float:
        return sum(l.fraction * l.price for l in self.legs) / sum(l.fraction for l in self.legs)

    @property
    def r_multiple(self) -> float:
        s = self.setup
        return s.sign * (self.avg_exit_price - s.entry_price) / s.risk_per_share


# --------------------------------------------------------------------------- helpers


def _oriented(day: pd.DataFrame, direction: SignalDirection) -> pd.DataFrame:
    """View the day so that 'up' is always profitable."""
    if direction is LONG:
        return day[["open", "high", "low", "close", "volume"]]
    return pd.DataFrame(
        {
            "open": -day["open"],
            "high": -day["low"],
            "low": -day["high"],
            "close": -day["close"],
            "volume": day["volume"],
        },
        index=day.index,
    )


def _bar_close_time(ts: pd.Timestamp) -> pd.Timestamp:
    return ts + pd.Timedelta(minutes=1)


# --------------------------------------------------------------------------- detection


def find_setup(
    symbol: str,
    day: pd.DataFrame,
    direction: SignalDirection,
    window_start: time,
    window_end: time,
    params: HitchhikerParams,
    *,
    source: str = "scan",
    alert_time: pd.Timestamp | None = None,
) -> Setup | None:
    """First bar in [window_start, window_end] that breaks a qualifying consolidation."""
    if day.empty or direction not in (LONG, SHORT):
        return None
    o = _oriented(day, direction)
    opens, highs, lows = o["open"].to_numpy(), o["high"].to_numpy(), o["low"].to_numpy()
    vols = o["volume"].to_numpy()
    idx = o.index
    session_open = opens[0]
    sign = 1 if direction is LONG else -1

    for j in range(len(o)):
        t = idx[j].time()
        if t < window_start or t > window_end:
            continue
        if j < params.min_consol_bars + 1:
            continue
        s_hi, s_lo = highs[:j].max(), lows[:j].min()
        s_range = s_hi - s_lo
        if s_range <= 0:
            continue

        # The consolidation is the longest window that satisfies every rule in
        # force, so a stray early bar cannot disqualify an otherwise valid base.
        chosen = None
        for n in range(min(params.max_consol_bars, j - 1), params.min_consol_bars - 1, -1):
            w_hi, w_lo = highs[j - n : j].max(), lows[j - n : j].min()
            w_range = w_hi - w_lo
            if w_range > params.max_consol_range_frac * s_range:
                continue
            drive = highs[: j - n].max() - session_open
            drive_ok = drive > 0 and drive >= params.min_drive_multiple * w_range
            in_location = w_lo >= s_lo + (1 - params.location_frac) * s_range
            if params.require_drive and not drive_ok:
                continue
            if params.require_location and not in_location:
                continue
            chosen = (n, w_hi, w_lo, drive, drive_ok, in_location)
            break
        if chosen is None:
            continue
        n, w_hi, w_lo, drive, drive_ok, in_location = chosen

        trigger = w_hi + params.entry_tick
        if highs[j] < trigger - 1e-9:
            continue
        entry = max(trigger, opens[j])
        stop = w_lo - params.stop_offset
        vol_ratio = float(vols[j] / vols[j - 1]) if vols[j - 1] > 0 else None

        return Setup(
            symbol=symbol,
            day=idx[j].date(),
            direction=direction,
            trigger_time=idx[j],
            entry_price=round(sign * entry, 4),
            stop_price=round(sign * stop, 4),
            consol_start=idx[j - n],
            consol_bars=n,
            consol_high=round(max(sign * w_hi, sign * w_lo), 4),
            consol_low=round(min(sign * w_hi, sign * w_lo), 4),
            session_high=round(max(sign * s_hi, sign * s_lo), 4),
            session_low=round(min(sign * s_hi, sign * s_lo), 4),
            drive=round(float(drive), 4),
            in_location=bool(in_location),
            drive_ok=bool(drive_ok),
            volume_ratio=vol_ratio,
            source=source,
            alert_time=alert_time,
        )
    return None


# --------------------------------------------------------------------------- simulation


def simulate(day: pd.DataFrame, setup: Setup, params: HitchhikerParams) -> TradePath:
    """Walk the bars after entry and apply the stop / wave / time exits."""
    o = _oriented(day, setup.direction)
    sign = setup.sign
    j = o.index.get_loc(setup.trigger_time)
    opens, highs, lows, closes = (o[c].to_numpy() for c in ("open", "high", "low", "close"))
    idx = o.index
    stop = sign * setup.stop_price
    entry_time = idx[j]

    legs: list[ExitLeg] = []
    remaining = 1.0
    last = j

    def exit_leg(k: int, fraction: float, oriented_price: float, reason: str) -> None:
        legs.append(ExitLeg(_bar_close_time(idx[k]), fraction, round(sign * oriented_price, 4), reason))

    if lows[j] <= stop:
        exit_leg(j, 1.0, stop, "stop")
        remaining = 0.0
    else:
        peak = highs[j]
        pause = 0
        phase = "wave1"
        wave1_peak = None
        for k in range(j + 1, len(o)):
            last = k
            if opens[k] <= stop:
                exit_leg(k, remaining, opens[k], "stop")
                remaining = 0.0
                break
            if lows[k] <= stop:
                exit_leg(k, remaining, stop, "stop")
                remaining = 0.0
                break
            if highs[k] > peak:
                peak = highs[k]
                pause = 0
                if phase == "rest" and peak > wave1_peak:
                    phase = "wave2"
            else:
                pause += 1
            if phase == "wave1" and pause >= params.wave_pause_bars:
                exit_leg(k, 0.5, closes[k], "wave1")
                remaining = 0.5
                phase, wave1_peak, pause = "rest", peak, 0
            elif phase == "wave2" and pause >= params.wave_pause_bars:
                exit_leg(k, remaining, closes[k], "wave2")
                remaining = 0.0
                break
            held = (idx[k] - entry_time) / pd.Timedelta(minutes=1) + 1
            if idx[k].time() >= params.flat_by:
                exit_leg(k, remaining, closes[k], "eod")
                remaining = 0.0
                break
            if held >= params.max_hold_minutes:
                exit_leg(k, remaining, closes[k], "time")
                remaining = 0.0
                break
        if remaining > 0:
            exit_leg(last, remaining, closes[last], "data_end")

    marks = day["close"].iloc[j : last + 1]
    # Excursions are measured from the entry price over the bars the trade was
    # open, in the trade's favour (MFE) and against it (MAE), per share.
    entry = sign * setup.entry_price
    context = {
        "mfe_per_share": round(float(max(highs[j : last + 1].max() - entry, 0.0)), 4),
        "mae_per_share": round(float(max(entry - lows[j : last + 1].min(), 0.0)), 4),
    }
    return TradePath(setup=setup, legs=legs, marks=marks, context=context)


# --------------------------------------------------------------------------- context


def market_alignment(
    trigger_time: pd.Timestamp,
    direction: SignalDirection,
    benchmarks: Mapping[str, pd.DataFrame],
) -> dict[str, bool]:
    """Per benchmark: was it on the trade's side of its session open one bar before entry?"""
    out: dict[str, bool] = {}
    for name, day in benchmarks.items():
        before = day[day.index < trigger_time]
        if before.empty:
            continue
        move = before["close"].iloc[-1] - day["open"].iloc[0]
        out[name] = bool(move > 0) if direction is LONG else bool(move < 0)
    return out


def key_level_context(setup: Setup, premarket: pd.DataFrame, prior_day: pd.DataFrame) -> dict:
    """The sheet's 'consolidation above premarket high / prior-day high' factor."""
    ctx: dict = {}
    if not premarket.empty:
        ctx["pm_high"], ctx["pm_low"] = float(premarket["high"].max()), float(premarket["low"].min())
    if not prior_day.empty:
        ctx["pd_high"], ctx["pd_low"] = float(prior_day["high"].max()), float(prior_day["low"].min())
    if setup.direction is LONG:
        levels = [ctx[k] for k in ("pm_high", "pd_high") if k in ctx]
        ctx["beyond_key_level"] = bool(levels) and setup.consol_low > max(levels)
    else:
        levels = [ctx[k] for k in ("pm_low", "pd_low") if k in ctx]
        ctx["beyond_key_level"] = bool(levels) and setup.consol_high < min(levels)
    return ctx


def alert_window(alert_time: pd.Timestamp, params: HitchhikerParams) -> tuple[time, time]:
    end = alert_time + timedelta(minutes=params.alert_window_minutes)
    return alert_time.time(), end.time()


def scan_day(
    symbol: str, day: pd.DataFrame, params: HitchhikerParams
) -> Setup | None:
    """Scanner mode: earliest qualifying break in either direction (one-and-done)."""
    found = [
        s
        for d in (LONG, SHORT)
        if (s := find_setup(symbol, day, d, params.scan_start, params.scan_end, params))
    ]
    return min(found, key=lambda s: s.trigger_time) if found else None


__all__ = [
    "ExitLeg",
    "HitchhikerParams",
    "LONG",
    "SHORT",
    "Setup",
    "TradePath",
    "alert_window",
    "find_setup",
    "key_level_context",
    "market_alignment",
    "scan_day",
    "simulate",
]
