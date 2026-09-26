"""Daily bar-by-bar portfolio simulator for rule-based strategies.

Order of events on each bar i:
  0. OVERNIGHT - interest on cash (T-bill rate) or margin interest on borrowed cash, short borrow
                 fees, and cash dividends for positions held into the ex-date.
  1. OPEN      - scheduled exits at the open, stops/targets gapped through at the open, then market
                 entries at the open (yesterday's "next_open" signals or open-safe same-day rules),
                 then limit/stop entry orders whose level the open already crossed.
  2. INTRADAY  - limit/stop entries touched during the bar, then stop loss / ATR stop / trailing
                 stop / take profit / scale-outs from the bar's high and low. If a stop and a target
                 are both touched on one bar the stop is assumed to have hit first (conservative).
  3. CLOSE     - time exits, rule exits and reversals at the close, then entries at the close.
  4. MARK      - portfolio marked to market at the close; liquidation if equity is exhausted.

Prices are split-adjusted as quoted; dividends are paid in cash on the ex-date (and charged to
shorts), so share counts, per-share commissions and whole-share sizing are realistic.
Signals use data up to the close of the signal bar ("at the close" = market-on-close order).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data, expr
from .strategy import Strategy

INTRADAY_REASONS = ("stop loss", "ATR stop", "trailing stop", "chandelier stop", "take profit", "scale out")


@dataclass
class Lot:
    shares: float
    price: float            # fill price incl. slippage
    bar: int
    at_open: bool
    commission: float
    income: float = 0.0     # dividends received (paid, for shorts) and borrow fees, allocated to the lot


@dataclass
class Position:
    k: int
    sign: int
    lots: list[Lot]
    entry_bar: int
    peak: float                          # best price since entry (high for longs, low for shorts)
    atr_at_entry: float
    exit_due_bar: int | None
    exit_due_open: bool
    exit_sig: np.ndarray | None = None
    pending_open_exit: bool = False
    pending_reason: str = ""
    scaled: set = field(default_factory=set)

    @property
    def shares(self) -> float:
        return sum(l.shares for l in self.lots)

    @property
    def avg_price(self) -> float:
        s = self.shares
        return sum(l.shares * l.price for l in self.lots) / s if s else np.nan


@dataclass
class Result:
    strategy: Strategy
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series
    positions: pd.Series
    prices: dict[str, pd.DataFrame] = field(repr=False, default_factory=dict)
    holdings: pd.DataFrame | None = None     # daily weight per ticker (end of day)
    interest: float = 0.0                    # net interest earned on cash
    in_market: pd.Series | None = None       # held a position at any time during the day
    kind: str = "signal"
    orders: pd.DataFrame | None = None
    extras: dict = field(default_factory=dict)


# ------------------------------------------------------------------ preparation

def _daily_rate(index: pd.DatetimeIndex, spec) -> np.ndarray:
    """Per-trading-day interest rate for idle cash."""
    if spec is None or spec is False or spec == 0:
        return np.zeros(len(index))
    if spec == "tbill":
        s = data.tbill_rate()
        if s.empty:
            return np.zeros(len(index))
        s = s.reindex(index.union(s.index)).ffill().reindex(index).fillna(0.0)
        return s.to_numpy() / 252.0
    return np.full(len(index), float(spec) / 252.0)


def _prepare(strat: Strategy):
    tickers = [data.canonical(t) for t in strat.universe]
    dfs = data.load_many(tickers)
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
        raise ValueError("no price data in the requested period")

    tick = list(dfs)
    T, N = len(cal), len(tick)
    arr = lambda: np.full((T, N), np.nan)  # noqa: E731
    O, H, L, C, V, DIV, ATR, VOL, LEVEL = (arr() for _ in range(9))
    long_sig = np.zeros((T, N), bool)
    short_sig = np.zeros((T, N), bool)
    exit_ = np.zeros((T, N), bool)
    rank = np.zeros((T, N))
    per_trade_exit = bool(strat.exit_when) and bool(expr.names_in(strat.exit_when) & expr.POSITION_VARS)
    namespaces = {}
    if strat.entry_fill == "open":
        # the static check (Strategy.validate) is backed by an empirical one on the longest history
        t0 = max(tick, key=lambda t: len(dfs[t]))
        for rule in (strat.entry, strat.short_entry):
            if rule:
                bad = expr.open_time_probe(rule, dfs[t0], t0)
                if bad:
                    raise ValueError(f"Lookahead: the entry is filled at the open but {bad} ({t0}). "
                                     "Use ref(..., 1) for yesterday's values or fill at the next open.")
    for j, t in enumerate(tick):
        df = dfs[t]
        ns = expr.Namespace(df, ticker=t)
        namespaces[t] = ns
        a = df.reindex(cal)
        O[:, j], H[:, j], L[:, j], C[:, j] = a["open"], a["high"], a["low"], a["close"]
        V[:, j] = a["volume"]
        DIV[:, j] = a["dividend"].fillna(0.0)
        ATR[:, j] = ns["atr"](strat.atr_period).reindex(cal)
        VOL[:, j] = ns["volatility"](20).reindex(cal)
        if strat.side in ("long", "both"):
            long_sig[:, j] = expr.evaluate(strat.entry, ns).reindex(cal, fill_value=False).to_numpy()
        if strat.side == "short":
            short_sig[:, j] = expr.evaluate(strat.entry, ns).reindex(cal, fill_value=False).to_numpy()
        if strat.side == "both":
            short_sig[:, j] = expr.evaluate(strat.short_entry, ns).reindex(cal, fill_value=False).to_numpy()
        if strat.exit_when and not per_trade_exit:
            exit_[:, j] = expr.evaluate(strat.exit_when, ns).reindex(cal, fill_value=False).to_numpy()
        if strat.entry_level:
            LEVEL[:, j] = expr.evaluate_value(strat.entry_level, ns).reindex(cal)
        if strat.rank_by:
            r = expr.evaluate_value(strat.rank_by, ns).reindex(cal)
        else:  # default preference: most liquid (20-day average dollar volume)
            r = (df["close"] * df["volume"]).rolling(20, min_periods=1).mean().reindex(cal)
        rank[:, j] = r.fillna(-np.inf if not strat.rank_ascending else np.inf).to_numpy()
    valid = ~np.isnan(C)
    long_sig &= valid
    short_sig &= valid

    if strat.universe_name == "NDX" and strat.point_in_time:
        mask, first = data.member_mask(tick, cal)
        long_sig &= mask
        short_sig &= mask
        cov = data.coverage_note(str(cal[0].date()), str(cal[-1].date()))
        if cov and not any(n.startswith("Survivorship:") for n in strat.notes):
            strat.notes.append(cov)
        if first is not None and cal[0] < first and not any(n.startswith("Membership:") for n in strat.notes):
            strat.notes.append(
                f"Membership: point-in-time Nasdaq-100 membership starts {first.date()}; before that the "
                f"earliest known member list is used, so results before {first.year} still carry survivorship bias.")
    elif N > 1:
        live = valid.sum(axis=1)
        need = max(2, int(0.5 * N))
        thin = np.flatnonzero(live < need)
        if len(thin) and thin[-1] > 20 and not any(n.startswith("Coverage:") for n in strat.notes):
            strat.notes.append(
                f"Coverage: until {cal[thin[-1]].date()} fewer than half of the {N} tickers had price history, "
                f"so early results come from a small subset. Add 'since <year>' to focus on the period with full coverage.")
    return dict(dfs=dfs, cal=cal, tick=tick, O=O, H=H, L=L, C=C, V=V, DIV=DIV, ATR=ATR, VOL=VOL,
                LEVEL=LEVEL, long_sig=long_sig, short_sig=short_sig, exit_sig=exit_, rank=rank,
                per_trade_exit=per_trade_exit, namespaces=namespaces)


# ------------------------------------------------------------------ simulation

def run(strat: Strategy) -> Result:
    strat.validate()
    P = _prepare(strat)
    cal, tick = P["cal"], P["tick"]
    O, H, L, C, V, DIV, ATR, VOL = P["O"], P["H"], P["L"], P["C"], P["V"], P["DIV"], P["ATR"], P["VOL"]
    LEVEL, long_sig, short_sig, exit_sig, rank = P["LEVEL"], P["long_sig"], P["short_sig"], P["exit_sig"], P["rank"]
    namespaces = P["namespaces"]
    T, N = C.shape
    slip = strat.slippage_bps / 1e4
    rate = _daily_rate(cal, strat.cash_rate)
    has = ~np.isnan(C)
    last_bar = np.array([np.flatnonzero(has[:, j])[-1] if has[:, j].any() else -1 for j in range(N)])
    last_close = np.full(N, np.nan)

    S = dict(cash=strat.capital, interest=0.0, halted=False)
    positions: dict[int, Position] = {}
    pending_mkt_open: list[tuple[int, int]] = []    # (k, sign) market orders for the next open
    pending_mkt_close: list[tuple[int, int]] = []   # (k, sign) for the next close
    pending_lvl: list[dict] = []                    # limit / stop entry orders
    trades: list[dict] = []
    equity = np.zeros(T)
    exposure = np.zeros(T)
    npos = np.zeros(T, int)
    in_mkt = np.zeros(T, bool)
    hold_w = np.zeros((T, N))
    bar_gross = [0.0]

    def price_or_last(k: int, prices: np.ndarray) -> float:
        px = prices[k]
        return px if not np.isnan(px) else last_close[k]

    def mark(prices: np.ndarray) -> float:
        v = S["cash"]
        for p in positions.values():
            v += p.sign * p.shares * price_or_last(p.k, prices)
        return v

    def gross(prices: np.ndarray) -> float:
        return sum(p.shares * price_or_last(p.k, prices) for p in positions.values())

    def note_gross(prices: np.ndarray) -> None:
        bar_gross[0] = max(bar_gross[0], gross(prices))

    def commission(shares: float, value: float) -> float:
        return strat.commission + strat.commission_per_share * shares + strat.commission_pct * abs(value)

    def size_shares(i: int, k: int, fill: float, eq: float, prices: np.ndarray, sgn: int, at_open: bool) -> float:
        # indicators used for sizing must be known when the order is placed
        ib = i - 1 if (at_open and i > 0) else i
        if strat.sizing == "percent":
            value = eq * strat.position_size
        elif strat.sizing == "fixed_dollars":
            value = strat.fixed_amount
        elif strat.sizing == "fixed_shares":
            value = strat.fixed_amount * fill
        elif strat.sizing == "risk":
            dist = fill * strat.stop_loss if strat.stop_loss else strat.stop_atr * ATR[ib, k]
            if not np.isfinite(dist) or dist <= 0:
                return 0.0
            value = eq * strat.risk_per_trade / dist * fill
        else:  # volatility
            v = VOL[ib, k]
            if not np.isfinite(v) or v <= 0:
                return 0.0
            value = eq * strat.target_vol / v
        # gross exposure cap and (for longs) cash / margin availability
        room = strat.leverage * eq - gross(prices)
        value = min(value, room)
        if sgn == 1:
            avail = S["cash"] + (strat.leverage - 1) * max(eq, 0)
            value = min(value, avail)
        if value <= 0 or not np.isfinite(fill) or fill <= 0:
            return 0.0
        shares = value / fill
        if strat.max_volume_pct and V[i, k] > 0:
            shares = min(shares, strat.max_volume_pct * V[i, k])
        if not strat.fractional_shares:
            shares = np.floor(shares)
        if sgn == 1 and shares * fill + commission(shares, shares * fill) > S["cash"] + (strat.leverage - 1) * max(eq, 0) + 1e-9:
            shares = max(0.0, (S["cash"] + (strat.leverage - 1) * max(eq, 0) - strat.commission) / (fill * (1 + strat.commission_pct) + strat.commission_per_share))
            if not strat.fractional_shares:
                shares = np.floor(shares)
        return max(shares, 0.0)

    def open_pos(i: int, k: int, px: float, at_open: bool, sgn: int, prices: np.ndarray) -> bool:
        if S["halted"]:
            return False
        fill = px * (1 + sgn * slip)
        eq = mark(prices)
        if eq <= 0:
            return False
        shares = size_shares(i, k, fill, eq, prices, sgn, at_open)
        if shares <= 0:
            return False
        com = commission(shares, shares * fill)
        S["cash"] -= sgn * shares * fill + com
        lot = Lot(shares, fill, i, at_open, com)
        if k in positions:  # pyramiding
            positions[k].lots.append(lot)
            note_gross(prices)
            return True
        hb = strat.hold_bars
        if hb is None:
            due, due_open = None, False
        elif strat.hold_exit_fill == "close":
            due, due_open = (i + hb - 1 if at_open else i + hb), False
        else:
            due, due_open = i + hb, True
        atr_ref = ATR[i - 1, k] if at_open and i > 0 else ATR[i, k]
        p = Position(k, sgn, [lot], i, px, atr_ref, due, due_open)
        if P["per_trade_exit"]:
            ns = namespaces[tick[k]]
            dfk = ns.df
            idx = dfk.index
            pos_in_df = idx.get_indexer([cal[i]])[0]
            after = idx > cal[i] if not at_open else idx >= cal[i]
            hs = dfk["high"].where(after).cummax()
            ls = dfk["low"].where(after).cummin()
            extra = {"bars_held": pd.Series(np.arange(len(idx)) - pos_in_df, index=idx, dtype=float),
                     "entry_price": fill,
                     "pnl": sgn * (dfk["close"] / fill - 1),
                     "highest_since_entry": np.maximum(hs.fillna(fill), fill),
                     "lowest_since_entry": np.minimum(ls.fillna(fill), fill)}
            p.exit_sig = expr.evaluate(strat.exit_when, expr.Namespace(dfk, extra, ticker=tick[k])).reindex(cal, fill_value=False).to_numpy()
        positions[k] = p
        note_gross(prices)
        return True

    def close_part(i: int, p: Position, px: float, reason: str, at_open: bool, fraction: float = 1.0) -> None:
        fill = px * (1 - p.sign * slip)
        a = p.entry_bar + (0 if p.lots[0].at_open else 1)
        b = i - (1 if at_open else 0)
        hi = np.nanmax(H[a:b + 1, p.k]) if b >= a else np.nan
        lo = np.nanmin(L[a:b + 1, p.k]) if b >= a else np.nan
        hi = fill if np.isnan(hi) else max(hi, fill)
        lo = fill if np.isnan(lo) else min(lo, fill)
        remaining = []
        for lot in p.lots:
            q = lot.shares * fraction
            com = commission(q, q * fill)
            S["cash"] += p.sign * q * fill - com
            share_in = lot.commission * fraction
            inc = lot.income * fraction
            pnl = p.sign * q * (fill - lot.price) - share_in - com + inc
            cost = q * lot.price
            if p.sign == 1:
                mfe, mae = hi / lot.price - 1, lo / lot.price - 1
            else:
                mfe, mae = 1 - lo / lot.price, 1 - hi / lot.price
            trades.append({
                "ticker": tick[p.k], "side": "long" if p.sign == 1 else "short",
                "entry_date": cal[lot.bar].date(), "entry_fill": "open" if lot.at_open else "close",
                "entry_price": lot.price,
                "exit_date": cal[i].date(),
                "exit_fill": "open" if at_open else ("intraday" if reason in INTRADAY_REASONS else "close"),
                "exit_price": fill, "shares": q, "position_value": cost, "pnl": pnl,
                "return": pnl / cost if cost else 0.0, "bars_held": i - lot.bar, "exit_reason": reason,
                "mae": mae, "mfe": mfe, "commission": share_in + com, "income": inc,
            })
            if fraction < 1.0:
                lot.shares -= q
                lot.commission -= share_in
                lot.income -= inc
                remaining.append(lot)
        if fraction >= 1.0 or not remaining:
            del positions[p.k]
        else:
            p.lots = remaining

    def ranked(cands: list[int], i: int) -> list[int]:
        return sorted(cands, key=lambda k: rank[i, k], reverse=not strat.rank_ascending)

    def want(i: int, k: int, sgn: int, prices: np.ndarray, at_open: bool) -> None:
        """Handle an entry signal for ticker k in direction sgn."""
        if np.isnan(prices[k]):
            return
        p = positions.get(k)
        if p is not None and p.sign != sgn:
            if strat.side == "both" and strat.reverse:
                close_part(i, p, prices[k], "reversal", at_open)
            else:
                return
        p = positions.get(k)
        if p is not None:
            if len(p.lots) < strat.pyramiding and p.lots[-1].bar != i:
                open_pos(i, k, prices[k], at_open, sgn, prices)
            return
        if len(positions) >= strat.max_positions:
            return
        open_pos(i, k, prices[k], at_open, sgn, prices)

    def signals(i: int) -> list[tuple[int, int]]:
        out = [(k, 1) for k in np.flatnonzero(long_sig[i])] + [(k, -1) for k in np.flatnonzero(short_sig[i])]
        return out

    def ranked_pairs(pairs: list[tuple[int, int]], i: int) -> list[tuple[int, int]]:
        return sorted(pairs, key=lambda kp: rank[i, kp[0]], reverse=not strat.rank_ascending)

    stops_used = any([strat.stop_loss, strat.stop_atr, strat.trailing_stop, strat.trailing_atr,
                      strat.take_profit, strat.take_profit_atr, strat.scale_out])

    def levels(p: Position):
        """(stop level or None, stop reason, target level or None)"""
        s, e = p.sign, p.avg_price
        stop, why = None, ""
        cands = []
        if strat.stop_loss:
            cands.append((e * (1 - s * strat.stop_loss), "stop loss"))
        if strat.stop_atr and np.isfinite(p.atr_at_entry):
            cands.append((e - s * strat.stop_atr * p.atr_at_entry, "ATR stop"))
        if strat.trailing_stop:
            cands.append((p.peak * (1 - s * strat.trailing_stop), "trailing stop"))
        if strat.trailing_atr and np.isfinite(p.atr_at_entry):
            cands.append((p.peak - s * strat.trailing_atr * p.atr_at_entry, "chandelier stop"))
        if cands:  # the tightest stop (closest to price) triggers first
            stop, why = max(cands, key=lambda c: c[0] * s)
        tgt = None
        if strat.take_profit:
            tgt = e * (1 + s * strat.take_profit)
        if strat.take_profit_atr and np.isfinite(p.atr_at_entry):
            t2 = e + s * strat.take_profit_atr * p.atr_at_entry
            tgt = t2 if tgt is None else (min(tgt, t2) if s == 1 else max(tgt, t2))
        return stop, why, tgt

    for i in range(T):
        o, h, lo_, c = O[i], H[i], L[i], C[i]
        bar_gross[0] = 0.0
        had_position = bool(positions)

        # ---- 0. OVERNIGHT: interest, borrow fees, dividends
        if i > 0:
            r = rate[i - 1]
            if S["cash"] >= 0:
                earned = S["cash"] * r
            else:
                earned = S["cash"] * (r + strat.margin_rate / 252.0)
            S["cash"] += earned
            S["interest"] += earned
        for p in positions.values():
            k = p.k
            if i > 0 and p.sign == -1 and strat.borrow_fee:
                fee = p.shares * price_or_last(k, C[i - 1]) * strat.borrow_fee / 252.0
                S["cash"] -= fee
                for lot in p.lots:
                    lot.income -= fee * lot.shares / p.shares
            if DIV[i, k] > 0:
                for lot in p.lots:
                    if lot.bar < i:  # held into the ex-date
                        amt = p.sign * lot.shares * DIV[i, k]
                        S["cash"] += amt
                        lot.income += amt

        # ---- 1. OPEN
        for p in list(positions.values()):
            k = p.k
            if np.isnan(o[k]) or (p.entry_bar == i):
                continue
            if p.pending_open_exit or (p.exit_due_bar is not None and p.exit_due_open and i >= p.exit_due_bar):
                close_part(i, p, o[k], p.pending_reason or "time exit", at_open=True)
                continue
            if stops_used:
                stop, why, tgt = levels(p)
                if stop is not None and (o[k] - stop) * p.sign <= 0:
                    close_part(i, p, o[k], why, True)
                    continue
                if tgt is not None and (o[k] - tgt) * p.sign >= 0:
                    close_part(i, p, o[k], "take profit", True)
        # market entries at the open
        todays_open = []
        if strat.entry_fill == "open" and strat.entry_order == "market":
            todays_open = ranked_pairs(signals(i), max(i - 1, 0))  # rank with yesterday's values
        for k, sgn in pending_mkt_open + todays_open:
            want(i, k, sgn, o, at_open=True)
        pending_mkt_open = []

        # ---- limit / stop entry orders (open, then intraday)
        still = []
        for od in pending_lvl:
            k, sgn, lvl = od["k"], od["sign"], od["level"]
            if i > od["expires"] or S["halted"]:
                continue
            if np.isnan(o[k]) or (k in positions and positions[k].sign == sgn and len(positions[k].lots) >= strat.pyramiding):
                still.append(od)
                continue
            fill = None
            buyish = sgn == 1
            if strat.entry_order == "limit":
                if (o[k] <= lvl) if buyish else (o[k] >= lvl):
                    fill, at_o = o[k], True
                elif (lo_[k] <= lvl) if buyish else (h[k] >= lvl):
                    fill, at_o = lvl, False
            else:  # stop entry
                if (o[k] >= lvl) if buyish else (o[k] <= lvl):
                    fill, at_o = o[k], True
                elif (h[k] >= lvl) if buyish else (lo_[k] <= lvl):
                    fill, at_o = lvl, False
            if fill is None:
                still.append(od)
                continue
            if k not in positions and len(positions) >= strat.max_positions:
                still.append(od)
                continue
            if open_pos(i, k, fill, True, sgn, o):
                positions[k].lots[-1].at_open = True  # exposed from the fill onward
        pending_lvl = still

        # ---- 2. INTRADAY stops / targets / scale-outs
        if stops_used:
            for p in list(positions.values()):
                k = p.k
                if np.isnan(h[k]) or (p.lots[-1].bar == i and not p.lots[-1].at_open):
                    continue
                s = p.sign
                adverse = lo_[k] if s == 1 else h[k]
                best = h[k] if s == 1 else lo_[k]
                stop, why, tgt = levels(p)
                if stop is not None and (adverse - stop) * s <= 0:
                    close_part(i, p, stop if (o[k] - stop) * s > 0 else o[k], why, at_open=False)
                    continue
                if tgt is not None and (best - tgt) * s >= 0:
                    close_part(i, p, tgt if (o[k] - tgt) * s < 0 else o[k], "take profit", at_open=False)
                    continue
                for j, so in enumerate(strat.scale_out):
                    if j in p.scaled or k not in positions:
                        continue
                    lvl = p.avg_price * (1 + s * so["at"])
                    if (best - lvl) * s >= 0:
                        p.scaled.add(j)
                        close_part(i, p, lvl, "scale out", at_open=False, fraction=float(so["fraction"]))
                if k in positions:
                    p.peak = max(p.peak, h[k]) if s == 1 else min(p.peak, lo_[k])

        # ---- 3. CLOSE exits
        for p in list(positions.values()):
            k = p.k
            if np.isnan(c[k]):
                if i > last_bar[k]:
                    close_part(i, p, last_close[k], "data ended", at_open=False)
                continue
            if p.entry_bar == i and not p.lots[0].at_open:
                continue
            if p.exit_due_bar is not None and not p.exit_due_open and i >= p.exit_due_bar:
                close_part(i, p, c[k], "time exit", at_open=False)
                continue
            sig = p.exit_sig[i] if p.exit_sig is not None else exit_sig[i, k]
            if strat.exit_when and sig:
                if strat.exit_when_fill == "close":
                    close_part(i, p, c[k], "exit rule", at_open=False)
                else:
                    p.pending_open_exit, p.pending_reason = True, "exit rule"
        np.copyto(last_close, c, where=~np.isnan(c))

        # ---- 3b. entries at the close / orders for tomorrow
        if not S["halted"]:
            todays = ranked_pairs(signals(i), i)
            if strat.entry_order != "market":
                for k, sgn in todays:
                    lvl = LEVEL[i, k]
                    if np.isfinite(lvl):
                        pending_lvl = [od for od in pending_lvl if od["k"] != k]
                        pending_lvl.append({"k": k, "sign": sgn, "level": lvl, "expires": i + strat.order_valid_bars})
            elif strat.entry_fill == "close":
                for k, sgn in todays:
                    want(i, k, sgn, c, at_open=False)
            elif strat.entry_fill == "next_close":
                for k, sgn in pending_mkt_close:
                    want(i, k, sgn, c, at_open=False)
                pending_mkt_close = todays
            elif strat.entry_fill == "next_open":
                pending_mkt_open = todays

        # ---- 4. MARK
        equity[i] = mark(c)
        if equity[i] <= 0 and positions and not S["halted"]:
            for p in list(positions.values()):
                close_part(i, p, price_or_last(p.k, c), "liquidated (equity exhausted)", at_open=False)
            S["halted"] = True
            equity[i] = S["cash"]
            if not any(n.startswith("Liquidated") for n in strat.notes):
                strat.notes.append(f"Liquidated: equity fell to zero on {cal[i].date()}; trading stopped.")
        g = gross(c)
        eq_pos = equity[i] if equity[i] > 0 else np.nan
        exposure[i] = max(g, bar_gross[0]) / eq_pos if eq_pos == eq_pos else 0.0
        npos[i] = len(positions)
        in_mkt[i] = bool(positions) or had_position or bar_gross[0] > 0
        if positions and equity[i] > 0:
            for p in positions.values():
                hold_w[i, p.k] = p.sign * p.shares * price_or_last(p.k, c) / equity[i]

    # close anything still open at the last bar
    for p in list(positions.values()):
        close_part(T - 1, p, last_close[p.k], "open at end", at_open=False)
    equity[-1] = S["cash"]

    tr = pd.DataFrame(trades)
    if not tr.empty:
        tr = tr.sort_values(["exit_date", "entry_date", "ticker"], kind="stable").reset_index(drop=True)
        tr.index = tr.index + 1
        tr["cum_pnl"] = tr["pnl"].cumsum()

    # prepend the starting capital one day before the first bar so returns include the first bar
    start = cal[0] - pd.Timedelta(days=1)
    idx = pd.DatetimeIndex([start]).append(cal)
    eq = pd.Series(np.concatenate([[strat.capital], equity]), index=idx, name="equity")
    ex = pd.Series(np.concatenate([[0.0], exposure]), index=idx, name="exposure")
    npo = pd.Series(np.concatenate([[0], npos]), index=idx, name="positions")
    inm = pd.Series(np.concatenate([[False], in_mkt]), index=idx, name="in_market")
    hw = pd.DataFrame(hold_w, index=cal, columns=tick)
    hw = hw.loc[:, (hw != 0).any()]
    return Result(strategy=strat, equity=eq, trades=tr, exposure=ex, positions=npo, prices=P["dfs"],
                  holdings=hw, interest=S["interest"], in_market=inm)
