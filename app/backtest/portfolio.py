"""Turn simulated trade paths into sized trades and a marked-to-market equity curve.

Sizing follows the platform's live rule (``app.risk.sizing``): shares come from
the distance to the stop, then get clamped by buying power. Equity used for
sizing is *realised* equity at the moment of entry, so a trade never borrows
unrealised gains from a position that is still open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from app.backtest.hitchhiker import TradePath


@dataclass(frozen=True)
class CostModel:
    #: Paid on entry and on every exit leg, per share, against the trade.
    slippage_per_share: float = 0.01
    #: IBKR Pro fixed-style commission.
    commission_per_share: float = 0.005
    commission_min: float = 1.00
    commission_max_pct: float = 0.01

    def commission(self, shares: int, price: float) -> float:
        if shares <= 0:
            return 0.0
        fee = max(self.commission_min, self.commission_per_share * shares)
        return round(min(fee, self.commission_max_pct * shares * price), 4)


@dataclass(frozen=True)
class SizingModel:
    starting_capital: float = 10_000.0
    #: Fraction of realised equity risked per trade (entry-to-stop, incl. slippage).
    risk_pct: float = 0.01
    #: Gross notional of all open positions may not exceed this multiple of equity.
    max_leverage: float = 2.0


@dataclass
class PortfolioResult:
    trades: pd.DataFrame
    skipped: pd.DataFrame
    #: Equity at every bar close in the backtest calendar (ET index).
    equity: pd.Series
    sizing: SizingModel
    costs: CostModel


def _split_shares(qty: int, fractions: list[float]) -> list[int]:
    """Integer shares per leg; the last leg takes the remainder."""
    shares, used = [], 0
    for f in fractions[:-1]:
        n = min(qty - used, math.floor(qty * f))
        shares.append(n)
        used += n
    shares.append(qty - used)
    return shares


def run_portfolio(
    paths: list[TradePath],
    calendar: pd.DatetimeIndex,
    sizing: SizingModel | None = None,
    costs: CostModel | None = None,
) -> PortfolioResult:
    """Size every path in entry order and build the equity curve.

    ``calendar`` is the set of bar-start times (ET) the curve is reported on,
    normally every regular-session minute of every day in the data.
    """
    sizing = sizing or SizingModel()
    costs = costs or CostModel()
    slip = costs.slippage_per_share

    rows: list[dict] = []
    skipped: list[dict] = []
    # (time, cash delta) for realised P&L; (trade row, path, legs shares) for marking.
    cash_events: list[tuple[pd.Timestamp, float]] = []
    open_book: list[tuple[dict, TradePath, list[int]]] = []

    for path in sorted(paths, key=lambda p: p.setup.trigger_time):
        s = path.setup
        t = s.trigger_time
        realised = sizing.starting_capital + sum(v for ts, v in cash_events if ts <= t)
        open_notional = 0.0
        for row, p, leg_shares in open_book:
            if p.setup.trigger_time <= t < p.exit_time:
                still = row["shares"] - sum(
                    q for leg, q in zip(p.legs, leg_shares) if leg.time <= t
                )
                open_notional += still * row["entry_fill"]

        per_share_risk = s.risk_per_share + slip
        qty_risk = math.floor(sizing.risk_pct * realised / per_share_risk)
        room = sizing.max_leverage * realised - open_notional
        qty_bp = math.floor(max(room, 0.0) / (s.entry_price + slip))
        qty = min(qty_risk, qty_bp)
        if qty < 1:
            skipped.append(
                {
                    "symbol": s.symbol,
                    "trigger_time": t,
                    "reason": "no buying power" if qty_bp < 1 else "risk budget < 1 share",
                }
            )
            continue

        entry_fill = s.entry_price + s.sign * slip
        leg_shares = _split_shares(qty, [leg.fraction for leg in path.legs])
        entry_comm = costs.commission(qty, entry_fill)
        cash_events.append((t + pd.Timedelta(minutes=1), -entry_comm))

        gross = 0.0
        exit_comm = 0.0
        exit_value = 0.0
        exits_desc = []
        for leg, q in zip(path.legs, leg_shares):
            if q <= 0:
                continue
            fill = leg.price - s.sign * slip
            pnl = s.sign * (fill - entry_fill) * q
            comm = costs.commission(q, fill)
            gross += pnl
            exit_comm += comm
            exit_value += fill * q
            cash_events.append((leg.time, pnl - comm))
            exits_desc.append(f"{q}@{fill:.2f} {leg.reason} {leg.time:%H:%M}")

        commissions = entry_comm + exit_comm
        net = gross - commissions
        risk_dollars = qty * per_share_risk
        row = {
            "trade_id": len(rows) + 1,
            "symbol": s.symbol,
            "date": s.day.isoformat(),
            "direction": s.direction.value,
            "source": s.source,
            "alert_time": s.alert_time.strftime("%H:%M") if s.alert_time is not None else "",
            "entry_time": t,
            "exit_time": path.exit_time,
            "hold_minutes": int((path.exit_time - t) / pd.Timedelta(minutes=1)),
            "consol_start": s.consol_start.strftime("%H:%M"),
            "consol_bars": s.consol_bars,
            "consol_high": s.consol_high,
            "consol_low": s.consol_low,
            "entry_price": s.entry_price,
            "entry_fill": round(entry_fill, 4),
            "stop_price": s.stop_price,
            "shares": qty,
            "notional": round(qty * entry_fill, 2),
            "risk_dollars": round(risk_dollars, 2),
            "avg_exit_fill": round(exit_value / qty, 4),
            "exits": "; ".join(exits_desc),
            "exit_reason": path.legs[-1].reason,
            "gross_pnl": round(gross, 2),
            "commissions": round(commissions, 2),
            "slippage_cost": round(slip * qty * 2, 2),
            "net_pnl": round(net, 2),
            "r_multiple": round(net / risk_dollars, 3),
            "equity_before": round(realised, 2),
            "return_on_equity_pct": round(100 * net / realised, 3),
            "mfe_r": round(path.context.get("mfe_per_share", 0.0) / per_share_risk, 2),
            "mae_r": round(path.context.get("mae_per_share", 0.0) / per_share_risk, 2),
            "in_location": s.in_location,
            "drive_ok": s.drive_ok,
            "volume_ratio": None if s.volume_ratio is None else round(s.volume_ratio, 2),
            "market_aligned": path.context.get("market_aligned"),
            "beyond_key_level": path.context.get("beyond_key_level"),
        }
        rows.append(row)
        open_book.append((row, path, leg_shares))

    trades = pd.DataFrame(rows)
    equity = _equity_curve(calendar, sizing.starting_capital, cash_events, open_book, slip)
    return PortfolioResult(trades, pd.DataFrame(skipped), equity, sizing, costs)


def _equity_curve(
    calendar: pd.DatetimeIndex,
    start: float,
    cash_events: list[tuple[pd.Timestamp, float]],
    open_book: list[tuple[dict, TradePath, list[int]]],
    slip: float,
) -> pd.Series:
    """Equity at each bar *close* (bar start + 1 minute)."""
    closes = pd.DatetimeIndex(calendar).sort_values() + pd.Timedelta(minutes=1)
    realised = pd.Series(0.0, index=closes)
    if cash_events:
        times = pd.DatetimeIndex([t for t, _ in cash_events])
        # Every event is stamped at a bar close; snap anything off-calendar to
        # the next close so no cash flow can be dropped by the reindex below.
        pos = closes.searchsorted(times).clip(max=len(closes) - 1)
        ev = pd.Series([v for _, v in cash_events], index=closes[pos])
        ev = ev.groupby(level=0).sum()
        realised = realised.add(ev.reindex(closes, fill_value=0.0), fill_value=0.0).cumsum()

    unrealised = pd.Series(0.0, index=closes)
    for row, path, leg_shares in open_book:
        sign = path.setup.sign
        marks = path.marks.copy()
        marks.index = marks.index + pd.Timedelta(minutes=1)
        held = pd.Series(float(row["shares"]), index=marks.index)
        for leg, q in zip(path.legs, leg_shares):
            held[held.index >= leg.time] -= q
        # Marks are taken on the side a closing order would trade into.
        mtm = sign * ((marks - sign * slip) - row["entry_fill"]) * held
        unrealised = unrealised.add(mtm.reindex(closes, fill_value=0.0), fill_value=0.0)

    equity = start + realised + unrealised
    equity.name = "equity"
    return equity
