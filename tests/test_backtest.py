"""HitchHiker backtester: setup detection, exits, sizing and metrics."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from app.backtest.hitchhiker import LONG, SHORT, HitchhikerParams, find_setup, scan_day, simulate
from app.backtest.metrics import drawdown, drawdown_periods, return_stats, summarize
from app.backtest.portfolio import CostModel, SizingModel, run_portfolio
from app.backtest.runner import Alert, buy_and_hold, calendar, run_alerts
from datetime import time

ET = "America/New_York"


def make_day(rows: list[tuple[float, float, float, float]], day: str = "2026-09-16", vol: float = 1000.0):
    """Bars from 09:30 ET, one per minute, as (open, high, low, close)."""
    idx = pd.date_range(f"{day} 09:30", periods=len(rows), freq="min", tz=ET)
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    frame["volume"] = vol
    return frame


def long_setup_rows(after: list[tuple[float, float, float, float]]) -> list[tuple]:
    """Drive 10 -> 12 in 2 bars, then a 6-bar base at 11.80-12.00, then ``after``."""
    drive = [(10.0, 11.0, 10.0, 11.0), (11.0, 12.0, 11.0, 12.0)]
    base = [(11.9, 12.0, 11.8, 11.9)] * 6
    return drive + base + after


PARAMS = HitchhikerParams(scan_start=time(9, 30), scan_end=time(10, 30), wave_pause_bars=2)


def mirror(rows):
    return [(-o, -l, -h, -c) for o, h, l, c in rows]


# --------------------------------------------------------------------------- detection


def test_break_of_base_triggers_long_with_stop_below_base():
    day = make_day(long_setup_rows([(11.95, 12.30, 11.95, 12.25)]))
    s = find_setup("X", day, LONG, time(9, 30), time(10, 30), PARAMS)
    assert s is not None
    assert s.trigger_time == day.index[8]
    assert s.entry_price == pytest.approx(12.01)
    assert s.stop_price == pytest.approx(11.78)
    assert s.consol_bars == 6
    assert s.in_location and s.drive_ok


def test_gap_through_trigger_fills_at_open():
    day = make_day(long_setup_rows([(12.20, 12.30, 12.15, 12.25)]))
    s = find_setup("X", day, LONG, time(9, 30), time(10, 30), PARAMS)
    assert s.entry_price == pytest.approx(12.20)


def test_no_trigger_without_break():
    day = make_day(long_setup_rows([(11.9, 12.0, 11.85, 11.9)] * 5))
    assert find_setup("X", day, LONG, time(9, 30), time(10, 30), PARAMS) is None


def test_setup_ignores_future_bars():
    """Changing bars after the trigger cannot change the setup (no lookahead)."""
    base = long_setup_rows([(11.95, 12.30, 11.95, 12.25)])
    a = make_day(base + [(12.25, 12.5, 12.2, 12.4)] * 5)
    b = make_day(base + [(12.25, 12.26, 5.0, 5.0)] * 5)
    sa = find_setup("X", a, LONG, time(9, 30), time(10, 30), PARAMS)
    sb = find_setup("X", b, LONG, time(9, 30), time(10, 30), PARAMS)
    assert (sa.trigger_time, sa.entry_price, sa.stop_price) == (sb.trigger_time, sb.entry_price, sb.stop_price)


def test_location_rule_rejects_base_low_in_range():
    # Base sits in the middle of the day's range: drive to 12, flush to 9, base at 10.4-10.6.
    rows = [(10, 12, 10, 11.9), (11.9, 11.9, 9, 9.2)] + [(10.5, 10.6, 10.4, 10.5)] * 6 + [(10.5, 10.9, 10.5, 10.8)]
    day = make_day(rows)
    assert find_setup("X", day, LONG, time(9, 30), time(10, 30), PARAMS) is None
    relaxed = replace(PARAMS, require_location=False, require_drive=False)
    assert find_setup("X", day, LONG, time(9, 30), time(10, 30), relaxed) is not None


def test_short_is_exact_mirror_of_long():
    rows = long_setup_rows([(11.95, 12.30, 11.95, 12.25), (12.25, 12.4, 12.2, 12.35)] + [(12.3, 12.3, 12.2, 12.25)] * 3)
    lng = find_setup("X", make_day(rows), LONG, time(9, 30), time(10, 30), PARAMS)
    sht = find_setup("X", make_day(mirror(rows)), SHORT, time(9, 30), time(10, 30), PARAMS)
    assert sht.entry_price == pytest.approx(-lng.entry_price)
    assert sht.stop_price == pytest.approx(-lng.stop_price)
    pl = simulate(make_day(rows), lng, PARAMS)
    ps = simulate(make_day(mirror(rows)), sht, PARAMS)
    assert ps.r_multiple == pytest.approx(pl.r_multiple)
    assert [l.reason for l in ps.legs] == [l.reason for l in pl.legs]


def test_scan_day_picks_earliest_direction():
    day = make_day(long_setup_rows([(11.95, 12.30, 11.95, 12.25)]))
    s = scan_day("X", day, PARAMS)
    assert s is not None and s.direction is LONG


# --------------------------------------------------------------------------- exits


def test_stop_exits_whole_position():
    day = make_day(long_setup_rows([(11.95, 12.05, 11.95, 12.0), (12.0, 12.0, 11.5, 11.6)]))
    s = find_setup("X", day, LONG, time(9, 30), time(10, 30), PARAMS)
    path = simulate(day, s, PARAMS)
    assert [(l.reason, l.fraction, l.price) for l in path.legs] == [("stop", 1.0, 11.78)]
    assert path.r_multiple == pytest.approx(-1.0)


def test_gap_below_stop_fills_at_open():
    day = make_day(long_setup_rows([(11.95, 12.05, 11.95, 12.0), (11.5, 11.6, 11.4, 11.5)]))
    s = find_setup("X", day, LONG, time(9, 30), time(10, 30), PARAMS)
    assert simulate(day, s, PARAMS).legs[0].price == pytest.approx(11.5)


def test_two_waves_exit_halves():
    after = [
        (11.95, 12.10, 11.95, 12.10),  # trigger, peak 12.10
        (12.10, 12.30, 12.10, 12.30),  # new high
        (12.30, 12.25, 12.20, 12.22),  # pause 1
        (12.22, 12.24, 12.15, 12.20),  # pause 2 -> wave1 half @ 12.20
        (12.20, 12.25, 12.10, 12.20),  # rest
        (12.20, 12.50, 12.20, 12.50),  # beyond 12.30 -> wave 2
        (12.50, 12.70, 12.45, 12.70),
        (12.70, 12.65, 12.55, 12.60),  # pause 1
        (12.60, 12.62, 12.50, 12.55),  # pause 2 -> exit rest @ 12.55
    ]
    day = make_day(long_setup_rows(after))
    s = find_setup("X", day, LONG, time(9, 30), time(10, 30), PARAMS)
    legs = simulate(day, s, PARAMS).legs
    assert [(l.reason, l.fraction, l.price) for l in legs] == [("wave1", 0.5, 12.20), ("wave2", 0.5, 12.55)]


def test_stop_checked_before_wave_on_same_bar():
    after = [(11.95, 12.10, 11.95, 12.10), (12.10, 12.40, 11.70, 12.3)]
    day = make_day(long_setup_rows(after))
    s = find_setup("X", day, LONG, time(9, 30), time(10, 30), PARAMS)
    assert simulate(day, s, PARAMS).legs[-1].reason == "stop"


def test_time_stop():
    after = [(11.95, 12.10, 11.95, 12.10)] + [(12.05, 12.08, 12.0, 12.05)] * 10
    params = replace(PARAMS, max_hold_minutes=5, wave_pause_bars=50)
    day = make_day(long_setup_rows(after))
    s = find_setup("X", day, LONG, time(9, 30), time(10, 30), params)
    legs = simulate(day, s, params).legs
    assert legs[-1].reason == "time"
    assert legs[-1].time == s.trigger_time + pd.Timedelta(minutes=5)


# --------------------------------------------------------------------------- portfolio


def _one_trade(after):
    day = make_day(long_setup_rows(after) + [(12.0, 12.0, 12.0, 12.0)] * 5)
    s = find_setup("X", day, LONG, time(9, 30), time(10, 30), PARAMS)
    return day, simulate(day, s, PARAMS)


def test_risk_based_sizing_and_costs():
    day, path = _one_trade([(11.95, 12.05, 11.95, 12.0), (12.0, 12.0, 11.5, 11.6)])
    costs = CostModel(slippage_per_share=0.01)
    res = run_portfolio([path], day.index, SizingModel(10_000, 0.01, 4.0), costs)
    t = res.trades.iloc[0]
    # 1% of 10k = $100 / (0.23 stop distance + 0.01 slippage) = 416 shares.
    assert t.shares == 416
    assert t.entry_fill == pytest.approx(12.02)
    expected = 416 * (11.77 - 12.02) - costs.commission(416, 12.02) - costs.commission(416, 11.77)
    assert t.net_pnl == pytest.approx(expected, abs=0.01)
    assert res.equity.iloc[-1] == pytest.approx(10_000 + expected, abs=0.01)


def test_buying_power_caps_size():
    day, path = _one_trade([(11.95, 12.05, 11.95, 12.0), (12.0, 12.0, 11.5, 11.6)])
    res = run_portfolio([path], day.index, SizingModel(10_000, 0.01, 0.5))
    assert res.trades.iloc[0].shares == int(5000 / 12.02)


def test_equity_is_marked_to_market_while_open():
    day, path = _one_trade([(11.95, 12.05, 11.95, 12.0), (12.0, 12.0, 11.9, 11.9), (11.9, 11.9, 11.5, 11.6)])
    res = run_portfolio([path], day.index, SizingModel(10_000, 0.01, 4.0), CostModel(slippage_per_share=0))
    mark_bar_close = day.index[9] + pd.Timedelta(minutes=1)
    shares = res.trades.iloc[0].shares
    comm = CostModel(slippage_per_share=0).commission(shares, 12.01)
    assert res.equity[mark_bar_close] == pytest.approx(10_000 - comm + shares * (11.9 - 12.01), abs=0.01)


# --------------------------------------------------------------------------- metrics


def test_drawdown_and_periods():
    eq = pd.Series([100, 110, 99, 105, 120, 90, 95], index=pd.date_range("2026-01-01", periods=7, freq="D"))
    assert drawdown(eq).min() == pytest.approx(90 / 120 - 1)
    worst = drawdown_periods(eq)[0]
    assert worst.depth_pct == pytest.approx(-0.25)
    assert worst.recovery_time is None
    second = drawdown_periods(eq)[1]
    assert second.recovery_time == eq.index[4]


def test_sharpe_matches_hand_calculation():
    eod = pd.Series([100.0, 101.0, 100.0, 102.0], index=pd.date_range("2026-01-01", periods=4, freq="D"))
    rets = eod.pct_change().dropna()
    expected = rets.mean() / rets.std(ddof=1) * (252**0.5)
    assert return_stats(eod)["sharpe"] == pytest.approx(expected)


def test_summary_end_to_end_with_benchmark():
    day, path = _one_trade([(11.95, 12.05, 11.95, 12.0), (12.0, 12.0, 11.5, 11.6)])
    cal = calendar({"X": pd.concat([day, make_day([(1, 1, 1, 1)] * 390)])})
    res = run_portfolio([path], cal)
    bench = buy_and_hold(make_day([(10, 10, 10, 10)] * 390), cal, 10_000)
    stats = summarize(res.equity, res.trades, 10_000, {"SPY": bench})
    s = stats["summary"]
    assert s["n_trades"] == 1 and s["win_rate_pct"] == 0
    assert s["total_return_pct"] == pytest.approx(100 * (res.equity.iloc[-1] / 10_000 - 1))
    assert stats["benchmarks"]["SPY"]["total_return_pct"] == pytest.approx(0.0)
    assert s["max_drawdown_pct"] < 0


def test_alert_mode_only_triggers_after_alert():
    rows = long_setup_rows([(11.95, 12.30, 11.95, 12.25)] + [(12.2, 12.3, 12.2, 12.25)] * 40)
    day = make_day(rows)
    early = Alert("X", day.index[5], LONG)
    late = Alert("X", day.index[30], LONG)
    params = replace(PARAMS, alert_window_minutes=5)
    paths, _ = run_alerts({"X": day}, [early], params)
    assert paths and paths[0].setup.trigger_time == day.index[8]
    paths, diag = run_alerts({"X": day}, [late], params)
    assert not paths and diag[0]["status"] == "no trigger"
