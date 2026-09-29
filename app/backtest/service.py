"""Programmatic entry point for a backtest, shared by the CLI and the HTTP API.

``run_backtest`` takes a plain ``BacktestRequest`` and returns a JSON-safe dict:
the same numbers the HTML report shows, but as data rather than a document. The
web UI (``app/webui``) renders it live; ``scripts/serve_api.py`` exposes it.

Nothing here constructs a broker. It is the same offline engine as the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from app.backtest import data as bd
from app.backtest.hitchhiker import HitchhikerParams
from app.backtest.metrics import summarize
from app.backtest.portfolio import CostModel, SizingModel, run_portfolio
from app.backtest.runner import Alert, buy_and_hold, calendar, run_alerts, run_scan
from app.enums import SignalDirection

MARKET_SYMBOLS = ("SPY", "QQQ", "IWM")
BENCHMARKS = ("SPY", "QQQ")


class BacktestError(RuntimeError):
    """Raised for a bad request (unknown symbol, missing data, bad dates)."""


@dataclass
class BacktestRequest:
    data_dir: Path
    symbols: list[str]
    mode: str = "scan"  # "scan" or "alerts"
    alerts: list[dict] = field(default_factory=list)  # {date,time,symbol,direction}
    capital: float = 10_000.0
    risk_pct: float = 1.0
    max_leverage: float = 2.0
    slippage: float = 0.01
    # Strategy knobs (subset of HitchhikerParams that the UI exposes).
    min_consol_bars: int = 5
    max_consol_bars: int = 20
    require_location: bool = True
    require_drive: bool = True
    wave_pause_bars: int = 2
    max_hold_minutes: int = 60
    alert_window_minutes: int = 30
    market_entry: bool = False


def _num(v):
    """Make a scalar JSON-safe (NaN/inf -> None, numpy -> python, Timestamp -> str)."""
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, (pd.Timedelta, pd.Period)):
        return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, TypeError):
            return v
    return v


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    return [{k: _num(v) for k, v in row.items()} for _, row in frame.reset_index().iterrows()]


def _equity_payload(equity: pd.Series, benchmarks: dict[str, pd.Series], dd: pd.Series) -> dict:
    """Downsample to <= ~1500 points for the chart."""
    step = max(1, len(equity) // 1500)
    idx = equity.index[::step]
    if len(idx) and idx[-1] != equity.index[-1]:
        idx = idx.append(equity.index[-1:])
    series = [{"name": "Strategy", "values": [round(float(x), 2) for x in equity.reindex(idx)]}]
    for name, curve in benchmarks.items():
        series.append(
            {"name": f"{name} buy & hold", "values": [round(float(x), 2) for x in curve.reindex(idx).ffill()]}
        )
    return {
        "t": [t.strftime("%Y-%m-%d %H:%M") for t in idx],
        "series": series,
        "drawdown_pct": [round(100 * float(x), 3) for x in dd.reindex(idx)],
    }


def _params(req: BacktestRequest) -> HitchhikerParams:
    return HitchhikerParams(
        min_consol_bars=req.min_consol_bars,
        max_consol_bars=req.max_consol_bars,
        require_location=req.require_location,
        require_drive=req.require_drive,
        wave_pause_bars=req.wave_pause_bars,
        max_hold_minutes=req.max_hold_minutes,
        alert_window_minutes=req.alert_window_minutes,
    )


def _load_bars(data_dir: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    wanted = sorted(set(symbols) | set(MARKET_SYMBOLS))
    bars: dict[str, pd.DataFrame] = {}
    for sym in wanted:
        path = bd.csv_path(data_dir, sym)
        if path.exists():
            bars[sym] = bd.load_minute_bars(path)
    missing = [s for s in symbols if s not in bars]
    if missing:
        raise BacktestError(f"No bar data for {missing} in {data_dir}. Fetch or upload it first.")
    return bars


def available_symbols(data_dir: Path) -> list[str]:
    d = Path(data_dir)
    if not d.exists():
        return []
    return sorted(p.name.split("_")[0] for p in d.glob("*_1min.csv"))


def run_backtest(req: BacktestRequest) -> dict:
    """Run one backtest and return a JSON-safe result dict."""
    if not req.symbols:
        raise BacktestError("Pick at least one symbol.")
    bars = _load_bars(req.data_dir, req.symbols)
    params = _params(req)
    sizing = SizingModel(req.capital, req.risk_pct / 100.0, req.max_leverage)
    costs = CostModel(slippage_per_share=req.slippage)

    if req.mode == "alerts":
        alerts = [
            Alert(
                a["symbol"].upper(),
                pd.Timestamp(f"{a['date']} {a['time']}", tz=bd.ET),
                SignalDirection(a["direction"].lower()),
            )
            for a in req.alerts
        ]
        if not alerts:
            raise BacktestError("Alert mode needs at least one alert row.")
        paths, diagnostics = run_alerts(bars, alerts, params, market_entry=req.market_entry)
        days = sorted({a.time.date() for a in alerts})
    else:
        paths, diagnostics = run_scan(bars, req.symbols, params)
        days = None

    cal = calendar({s: bars[s] for s in req.symbols}, days)
    if len(cal) == 0:
        raise BacktestError("No regular-session bars in the selected data.")
    result = run_portfolio(paths, cal, sizing, costs)
    bench = {b: buy_and_hold(bars[b], cal, req.capital) for b in BENCHMARKS if b in bars}
    stats = summarize(result.equity, result.trades, req.capital, bench)

    summary = {k: _num(v) for k, v in stats["summary"].items()}
    return {
        "summary": summary,
        "benchmarks": {k: {kk: _num(vv) for kk, vv in v.items()} for k, v in stats["benchmarks"].items()},
        "equity": _equity_payload(result.equity, bench, stats["drawdown_series"]),
        "trades": _records(result.trades),
        "skipped": _records(result.skipped),
        "yearly": _records(stats["yearly"].rename_axis("period")),
        "monthly": _records(stats["monthly"].rename_axis("period")),
        "by_symbol": _records(stats["by_symbol"].rename_axis("symbol")),
        "by_direction": _records(stats["by_direction"].rename_axis("direction")),
        "by_exit": _records(stats["by_exit"].rename_axis("exit_reason")),
        "drawdowns": [
            {
                "peak": _num(p.peak_time),
                "trough": _num(p.trough_time),
                "recovered": _num(p.recovery_time),
                "depth_pct": round(100 * p.depth_pct, 3),
                "depth_dollars": round(p.depth_dollars, 2),
            }
            for p in stats["drawdowns"]
        ],
        "diagnostics": [{k: _num(v) for k, v in d.items()} for d in diagnostics],
        "few_days": stats["few_days"],
        "meta": {
            "symbols": req.symbols,
            "mode": req.mode,
            "trading_days": len(cal) // 390,
            "start": _num(result.equity.index[0]),
            "end": _num(result.equity.index[-1]),
            "capital": req.capital,
        },
    }
