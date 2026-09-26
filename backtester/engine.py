"""Daily bar-by-bar portfolio simulator.

Order of events on each bar i:
  1. OPEN   - scheduled exits at the open, gap-through stops/targets, then
              entries pending from yesterday's signals ("next_open").
  2. INTRA  - stop loss / trailing stop / take profit using the bar's high/low
              (if both stop and target are touched on one bar, assume the stop
              hit first - the conservative choice).
  3. CLOSE  - time-based and rule-based exits at the close, then entries at
              the close (same-bar signals, or yesterday's "next_close" signals).
  4. Mark the portfolio to market at the close.
Signals are computed from data available at the close of the signal bar, so
"buy at the close when X" assumes you can evaluate X a moment before the
close (standard market-on-close convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data, expr
from .strategy import Strategy


@dataclass
class Position:
    k: int
    sign: int
    shares: float
    entry_bar: int
    entry_price: float           # fill price incl. slippage
    entry_at_open: bool
    commission: float
    exit_due_bar: int | None
    exit_due_open: bool
    peak: float                  # best price since entry (high for longs, low for shorts)
    exit_sig: np.ndarray | None = None   # per-trade exit rule when it uses position variables
    pending_open_exit: bool = False
    pending_reason: str = ""


@dataclass
class Result:
    strategy: Strategy
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series
    positions: pd.Series
    prices: dict[str, pd.DataFrame] = field(repr=False, default_factory=dict)


def _prepare(strat: Strategy):
    dfs = data.load_many(strat.universe)
    start = pd.Timestamp(strat.start) if strat.start else None
    end = pd.Timestamp(strat.end) if strat.end else None
    cal = None
    for df in dfs.values():
        cal = df.index if cal is None else cal.union(df.index)
    if start is not None:
        cal = cal[cal >= start]
    if end is not None:
        cal = cal[cal <= end]
    if len(cal) < 2:
        raise ValueError("no data in the requested period")

    tick = list(dfs)
    T, N = len(cal), len(tick)
    O, H, L, C = (np.full((T, N), np.nan) for _ in range(4))
    Q = np.full((T, N), np.nan)
    entry = np.zeros((T, N), bool)
    exit_ = np.zeros((T, N), bool)
    rank = np.zeros((T, N))
    per_trade_exit = bool(strat.exit_when) and bool(expr.names_in(strat.exit_when) & expr.POSITION_VARS)
    namespaces = {}
    for j, t in enumerate(tick):
        df = dfs[t]
        ns = expr.Namespace(df)
        namespaces[t] = ns
        a = df.reindex(cal)
        O[:, j], H[:, j], L[:, j], C[:, j] = a["open"], a["high"], a["low"], a["close"]
        Q[:, j] = a["quote_close"]
        entry[:, j] = expr.evaluate(strat.entry, ns).reindex(cal, fill_value=False).to_numpy()
        if strat.exit_when and not per_trade_exit:
            exit_[:, j] = expr.evaluate(strat.exit_when, ns).reindex(cal, fill_value=False).to_numpy()
        if strat.rank_by:
            r = expr.evaluate_value(strat.rank_by, ns).reindex(cal)
        else:  # default preference: most liquid (20-day average dollar volume)
            r = (df["close"] * df["volume"]).rolling(20, min_periods=1).mean().reindex(cal)
        rank[:, j] = r.fillna(-np.inf if not strat.rank_ascending else np.inf).to_numpy()
    entry &= ~np.isnan(C)
    if N > 1:
        live = (~np.isnan(C)).sum(axis=1)
        need = max(2, int(0.5 * N))
        thin = np.flatnonzero(live < need)
        if len(thin) and thin[-1] > 20 and not any(n.startswith("Coverage:") for n in strat.notes):
            upto = cal[thin[-1]].date()
            strat.notes.append(
                f"Coverage: until {upto} fewer than half of the {N} tickers had price history "
                f"(most of today's members listed later), so early results come from a small subset. "
                f"Add 'since <year>' to focus on the period with full coverage.")
    return dfs, cal, tick, O, H, L, C, Q, entry, exit_, rank, per_trade_exit, namespaces


def run(strat: Strategy) -> Result:
    strat.validate()
    dfs, cal, tick, O, H, L, C, Q, entry, exit_sig, rank, per_trade_exit, namespaces = _prepare(strat)
    T, N = C.shape
    sign = 1 if strat.side == "long" else -1
    slip = strat.slippage_bps / 1e4
    size = strat.position_size
    # last bar with data for each ticker (to close positions if a series ends)
    has = ~np.isnan(C)
    last_bar = np.array([np.flatnonzero(has[:, j])[-1] if has[:, j].any() else -1 for j in range(N)])
    last_close = np.full(N, np.nan)

    cash = strat.capital
    positions: dict[int, Position] = {}
    pending_open: list[int] = []
    pending_close: list[int] = []
    trades: list[dict] = []
    equity = np.zeros(T)
    exposure = np.zeros(T)
    npos = np.zeros(T, int)

    def mark(prices: np.ndarray) -> float:
        v = cash
        for p in positions.values():
            px = prices[p.k] if not np.isnan(prices[p.k]) else last_close[p.k]
            v += p.sign * p.shares * px
        return v

    def commission(shares: float) -> float:
        return strat.commission + strat.commission_per_share * shares

    def open_pos(i: int, k: int, px: float, at_open: bool, eq: float) -> None:
        nonlocal cash
        fill = px * (1 + sign * slip)
        target = eq * size
        if sign == 1:
            avail = cash + (strat.leverage - 1) * eq
            target = min(target, avail - commission(target / fill))
        if target <= 0 or not np.isfinite(fill) or fill <= 0:
            return
        shares = target / fill
        if not strat.fractional_shares:
            shares = np.floor(shares)
        if shares <= 0:
            return
        com = commission(shares)
        cash -= sign * shares * fill + com
        hb = strat.hold_bars
        if hb is None:
            due, due_open = None, False
        elif strat.hold_exit_fill == "close":
            due, due_open = (i + max(hb - 1, 0) if at_open else i + max(hb, 1)), False
        else:
            due, due_open = i + max(hb, 1), True
        p = Position(k, sign, shares, i, fill, at_open, com, due, due_open, peak=px)
        if per_trade_exit:
            ns = namespaces[tick[k]]
            idx = ns.df.index
            pos_in_df = idx.get_indexer([cal[i]])[0]
            bars_held = pd.Series(np.arange(len(idx)) - pos_in_df, index=idx, dtype=float)
            extra = {"bars_held": bars_held, "entry_price": fill,
                     "pnl": sign * (ns.df["close"] / fill - 1)}
            ns2 = expr.Namespace(ns.df, extra)
            p.exit_sig = expr.evaluate(strat.exit_when, ns2).reindex(cal, fill_value=False).to_numpy()
        positions[k] = p

    def close_pos(i: int, p: Position, px: float, reason: str, at_open: bool) -> None:
        nonlocal cash
        fill = px * (1 - p.sign * slip)
        com = commission(p.shares)
        cash += p.sign * p.shares * fill - com
        pnl = p.sign * p.shares * (fill - p.entry_price) - p.commission - com
        # excursions over the bars the position was actually exposed to
        a = p.entry_bar + (0 if p.entry_at_open else 1)
        b = i - (1 if at_open else 0)
        hi = np.nanmax(H[a:b + 1, p.k]) if b >= a else np.nan
        lo = np.nanmin(L[a:b + 1, p.k]) if b >= a else np.nan
        hi = np.nanmax([hi, fill]) if not np.isnan(hi) else fill
        lo = np.nanmin([lo, fill]) if not np.isnan(lo) else fill
        if p.sign == 1:
            mfe, mae = hi / p.entry_price - 1, lo / p.entry_price - 1
        else:
            mfe, mae = 1 - lo / p.entry_price, 1 - hi / p.entry_price
        cost = p.shares * p.entry_price
        trades.append({
            "ticker": tick[p.k],
            "side": "long" if p.sign == 1 else "short",
            "entry_date": cal[p.entry_bar].date(),
            "entry_fill": "open" if p.entry_at_open else "close",
            "entry_price": p.entry_price,
            "exit_date": cal[i].date(),
            "exit_fill": "open" if at_open else ("close" if reason not in ("stop loss", "trailing stop", "take profit") else "intraday"),
            "exit_price": fill,
            "shares": p.shares,
            "position_value": cost,
            "pnl": pnl,
            "return": pnl / cost,
            "bars_held": i - p.entry_bar,
            "exit_reason": reason,
            "mae": mae,
            "mfe": mfe,
            "commission": p.commission + com,
        })
        del positions[p.k]

    def ranked(cands: list[int], i: int) -> list[int]:
        return sorted(cands, key=lambda k: rank[i, k], reverse=not strat.rank_ascending)

    def enter(i: int, cands: list[int], prices: np.ndarray, at_open: bool) -> None:
        eq = mark(prices)
        for k in cands:
            if len(positions) >= strat.max_positions:
                break
            if k in positions or np.isnan(prices[k]):
                continue
            open_pos(i, k, prices[k], at_open, eq)

    for i in range(T):
        o, h, lo_, c = O[i], H[i], L[i], C[i]

        # ---- 1. OPEN
        for p in list(positions.values()):
            k = p.k
            if np.isnan(o[k]) or p.entry_bar == i:
                continue
            if p.pending_open_exit or (p.exit_due_bar is not None and p.exit_due_open and i >= p.exit_due_bar):
                close_pos(i, p, o[k], p.pending_reason or "time exit", at_open=True)
                continue
            fav = (o[k] / p.entry_price - 1) * p.sign
            if strat.stop_loss and fav <= -strat.stop_loss:
                close_pos(i, p, o[k], "stop loss", True)
                continue
            if strat.trailing_stop:
                lvl = p.peak * (1 - p.sign * strat.trailing_stop)
                if (o[k] - lvl) * p.sign <= 0:
                    close_pos(i, p, o[k], "trailing stop", True)
                    continue
            if strat.take_profit and fav >= strat.take_profit:
                close_pos(i, p, o[k], "take profit", True)
        if strat.entry_fill == "open":
            # rank with yesterday's values: today's close is not known at the open
            pending_open = ranked(list(np.flatnonzero(entry[i] & ~np.isnan(o))), max(i - 1, 0))
        if pending_open:
            enter(i, pending_open, o, at_open=True)
            pending_open = []

        # ---- 2. INTRADAY stops / targets
        for p in list(positions.values()):
            k = p.k
            if np.isnan(h[k]) or (p.entry_bar == i and not p.entry_at_open):
                continue
            s, e = p.sign, p.entry_price
            adverse = lo_[k] if s == 1 else h[k]
            best = h[k] if s == 1 else lo_[k]
            exit_px, reason = None, ""
            if strat.stop_loss:
                lvl = e * (1 - s * strat.stop_loss)
                if (adverse - lvl) * s <= 0:
                    exit_px, reason = lvl, "stop loss"
            if exit_px is None and strat.trailing_stop:
                lvl = p.peak * (1 - s * strat.trailing_stop)
                if (adverse - lvl) * s <= 0:
                    exit_px, reason = lvl, "trailing stop"
            if exit_px is None and strat.take_profit:
                lvl = e * (1 + s * strat.take_profit)
                if (best - lvl) * s >= 0:
                    exit_px, reason = lvl, "take profit"
            if exit_px is not None:
                close_pos(i, p, exit_px, reason, at_open=False)
                continue
            p.peak = max(p.peak, h[k]) if s == 1 else min(p.peak, lo_[k])

        # ---- 3. CLOSE exits
        for p in list(positions.values()):
            k = p.k
            if np.isnan(c[k]):
                if i > last_bar[k]:
                    close_pos(i, p, last_close[k], "data ended", at_open=False)
                continue
            if p.entry_bar == i and not p.entry_at_open:
                continue
            if p.exit_due_bar is not None and not p.exit_due_open and i >= p.exit_due_bar:
                close_pos(i, p, c[k], "time exit", at_open=False)
                continue
            sig = p.exit_sig[i] if p.exit_sig is not None else exit_sig[i, k]
            if strat.exit_when and sig:
                if strat.exit_when_fill == "close":
                    close_pos(i, p, c[k], "exit rule", at_open=False)
                else:
                    p.pending_open_exit, p.pending_reason = True, "exit rule"
        np.copyto(last_close, c, where=~np.isnan(c))

        # ---- 3b. CLOSE entries
        todays = list(np.flatnonzero(entry[i]))
        if strat.entry_fill == "close":
            enter(i, ranked(todays, i), c, at_open=False)
        elif strat.entry_fill == "open":
            pass
        elif strat.entry_fill == "next_close":
            if pending_close:
                enter(i, pending_close, c, at_open=False)
            pending_close = ranked(todays, i)
        else:
            pending_open = ranked(todays, i)

        # ---- 4. MARK
        equity[i] = mark(c)
        gross = sum(p.shares * last_close[p.k] for p in positions.values())
        exposure[i] = gross / equity[i] if equity[i] > 0 else 0.0
        npos[i] = len(positions)

    # close anything still open at the last bar
    for p in list(positions.values()):
        close_pos(T - 1, p, last_close[p.k], "open at end", at_open=False)
    equity[-1] = cash

    tr = pd.DataFrame(trades)
    if not tr.empty:
        tr = tr.sort_values(["exit_date", "entry_date", "ticker"]).reset_index(drop=True)
        tr.index = tr.index + 1
        tr["cum_pnl"] = tr["pnl"].cumsum()
    return Result(
        strategy=strat,
        equity=pd.Series(equity, index=cal, name="equity"),
        trades=tr,
        exposure=pd.Series(exposure, index=cal, name="exposure"),
        positions=pd.Series(npos, index=cal, name="positions"),
        prices=dfs,
    )
