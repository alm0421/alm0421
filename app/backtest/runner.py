"""Glue: alerts or scanner -> trade paths -> portfolio -> metrics."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd

from app.backtest import data as bd
from app.backtest.hitchhiker import (
    LONG,
    SHORT,
    HitchhikerParams,
    Setup,
    TradePath,
    alert_window,
    find_setup,
    key_level_context,
    market_alignment,
    scan_day,
    simulate,
)
from app.enums import SignalDirection

MARKET_PROXIES = ("SPY", "QQQ", "IWM")


@dataclass(frozen=True)
class Alert:
    symbol: str
    time: pd.Timestamp  # ET
    direction: SignalDirection


def load_alerts(path: Path) -> list[Alert]:
    """CSV with columns ``date,time,symbol,direction`` (time in ET, HH:MM[:SS])."""
    alerts = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            ts = pd.Timestamp(f"{row['date']} {row['time']}", tz=bd.ET)
            direction = SignalDirection(row["direction"].strip().lower())
            alerts.append(Alert(row["symbol"].strip().upper(), ts, direction))
    return alerts


def calendar(bars_by_symbol: dict[str, pd.DataFrame], days: list[date] | None = None) -> pd.DatetimeIndex:
    """Every regular-session minute of every trading day present in the data."""
    all_days = sorted({d for b in bars_by_symbol.values() for d in bd.sessions(b)})
    if days is not None:
        all_days = [d for d in all_days if d in set(days)]
    stamps = []
    for d in all_days:
        start = pd.Timestamp.combine(d, bd.RTH_OPEN).tz_localize(bd.ET)
        stamps.append(pd.date_range(start, periods=390, freq="min"))
    return stamps[0].append(stamps[1:]) if stamps else pd.DatetimeIndex([], tz=bd.ET)


def buy_and_hold(bars: pd.DataFrame, cal: pd.DatetimeIndex, capital: float) -> pd.Series:
    """Buy at the first regular-session open on the calendar, mark at every bar close."""
    rth = bd.regular_session(bars)
    rth = rth[(rth.index >= cal[0]) & (rth.index <= cal[-1])]
    shares = capital / rth["open"].iloc[0]
    closes = rth["close"].reindex(cal).ffill().bfill()
    curve = shares * closes
    curve.index = cal + pd.Timedelta(minutes=1)
    return curve


def _benchmark_days(bars_by_symbol: dict[str, pd.DataFrame], day: date) -> dict[str, pd.DataFrame]:
    return {
        s: bd.day_bars(bars_by_symbol[s], day)
        for s in MARKET_PROXIES
        if s in bars_by_symbol and not bd.day_bars(bars_by_symbol[s], day).empty
    }


def _annotate(path: TradePath, bars: pd.DataFrame, bench: dict[str, pd.DataFrame]) -> TradePath:
    s = path.setup
    others = {k: v for k, v in bench.items() if k != s.symbol}
    align = market_alignment(s.trigger_time, s.direction, others)
    path.context["market_detail"] = align
    path.context["market_aligned"] = all(align.values()) if align else None
    path.context.update(
        key_level_context(s, bd.premarket_bars(bars, s.day), bd.previous_session(bars, s.day))
    )
    return path


def market_entry_setup(
    symbol: str, day: pd.DataFrame, alert: Alert, params: HitchhikerParams
) -> Setup | None:
    """Sensitivity variant: take the alert immediately at the alert bar's open.

    The stop sits beyond the tightest qualifying consolidation before the alert
    (same height rule as the scanner), or beyond the prior
    ``min_consol_bars`` bars if nothing qualifies.
    """
    if alert.time not in day.index:
        return None
    j = day.index.get_loc(alert.time)
    if j < params.min_consol_bars:
        return None
    prior = day.iloc[:j]
    s_range = prior["high"].max() - prior["low"].min()
    window = prior.iloc[-params.min_consol_bars :]
    for n in range(min(params.max_consol_bars, j), params.min_consol_bars - 1, -1):
        w = prior.iloc[-n:]
        if w["high"].max() - w["low"].min() <= params.max_consol_range_frac * s_range:
            window = w
            break
    entry = float(day["open"].iloc[j])
    sign = 1 if alert.direction is LONG else -1
    if sign > 0:
        stop = float(window["low"].min()) - params.stop_offset
    else:
        stop = float(window["high"].max()) + params.stop_offset
    if sign * (entry - stop) <= 0:
        return None
    return Setup(
        symbol=symbol,
        day=alert.time.date(),
        direction=alert.direction,
        trigger_time=alert.time,
        entry_price=round(entry, 4),
        stop_price=round(stop, 4),
        consol_start=window.index[0],
        consol_bars=len(window),
        consol_high=float(window["high"].max()),
        consol_low=float(window["low"].min()),
        session_high=float(prior["high"].max()),
        session_low=float(prior["low"].min()),
        drive=0.0,
        in_location=False,
        drive_ok=False,
        volume_ratio=None,
        source="alert_market",
        alert_time=alert.time,
    )


def run_alerts(
    bars_by_symbol: dict[str, pd.DataFrame],
    alerts: list[Alert],
    params: HitchhikerParams,
    *,
    market_entry: bool = False,
) -> tuple[list[TradePath], list[dict]]:
    """Alert mode: each alert arms one HitchHiker break (or a market entry)."""
    paths, diagnostics = [], []
    for a in alerts:
        diag = {"symbol": a.symbol, "alert_time": a.time, "direction": a.direction.value}
        bars = bars_by_symbol.get(a.symbol)
        day = bd.day_bars(bars, a.time.date()) if bars is not None else pd.DataFrame()
        if day.empty:
            diagnostics.append({**diag, "status": "no data"})
            continue
        if market_entry:
            setup = market_entry_setup(a.symbol, day, a, params)
        else:
            start, end = alert_window(a.time, params)
            setup = find_setup(
                a.symbol, day, a.direction, start, end, params, source="alert", alert_time=a.time
            )
        if setup is None:
            diagnostics.append({**diag, "status": "no trigger", "detail": _why_not(day, a, params)})
            continue
        bench = _benchmark_days(bars_by_symbol, a.time.date())
        path = _annotate(simulate(day, setup, params), bars, bench)
        if params.require_market_alignment and path.context.get("market_aligned") is False:
            diagnostics.append({**diag, "status": "filtered", "detail": "market not aligned"})
            continue
        paths.append(path)
        diagnostics.append(
            {**diag, "status": "signal", "detail": f"entry {setup.trigger_time:%H:%M} @ {setup.entry_price:.2f}"}
        )
    return paths, diagnostics


def _why_not(day: pd.DataFrame, alert: Alert, params: HitchhikerParams) -> str:
    """Re-run with filters relaxed one at a time to name the blocking rule."""
    start, end = alert_window(alert.time, params)
    relaxed = replace(params, require_location=False, require_drive=False)
    if find_setup(alert.symbol, day, alert.direction, start, end, relaxed) is None:
        side = "above" if alert.direction is LONG else "below"
        return (
            f"price never broke {side} a {params.min_consol_bars}-{params.max_consol_bars} bar "
            f"consolidation between {start:%H:%M} and {end:%H:%M}"
        )
    blockers = []
    if find_setup(alert.symbol, day, alert.direction, start, end, replace(params, require_location=False)):
        blockers.append("consolidation not in the outer third of the day's range")
    if find_setup(alert.symbol, day, alert.direction, start, end, replace(params, require_drive=False)):
        blockers.append("no qualifying drive off the open")
    return "; ".join(blockers) or "location and drive rules both failed"


def run_scan(
    bars_by_symbol: dict[str, pd.DataFrame],
    symbols: list[str],
    params: HitchhikerParams,
) -> tuple[list[TradePath], list[dict]]:
    """Scanner mode: every symbol, every day, first qualifying break either way."""
    paths, diagnostics = [], []
    for sym in symbols:
        bars = bars_by_symbol[sym]
        for day_date, day in bd.iter_days(bars):
            setup = scan_day(sym, day, params)
            if setup is None:
                diagnostics.append({"symbol": sym, "date": day_date.isoformat(), "status": "no setup"})
                continue
            bench = _benchmark_days(bars_by_symbol, day_date)
            path = _annotate(simulate(day, setup, params), bars, bench)
            if params.require_market_alignment and path.context.get("market_aligned") is False:
                diagnostics.append(
                    {"symbol": sym, "date": day_date.isoformat(), "status": "filtered: market"}
                )
                continue
            paths.append(path)
            diagnostics.append(
                {
                    "symbol": sym,
                    "date": day_date.isoformat(),
                    "status": f"signal {setup.direction.value} {setup.trigger_time:%H:%M}",
                }
            )
    return paths, diagnostics


__all__ = [
    "Alert",
    "MARKET_PROXIES",
    "SHORT",
    "buy_and_hold",
    "calendar",
    "load_alerts",
    "market_entry_setup",
    "run_alerts",
    "run_scan",
]
