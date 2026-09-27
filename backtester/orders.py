"""Today's orders: turn a strategy's current target into a broker order list.

`todays_orders(spec, account_value, holdings_text)` works out what the strategy wants to hold after
the latest bar, sizes it to `account_value`, subtracts what you already hold and returns the orders,
plus a generic CSV and an Interactive Brokers basket CSV (TWS "Basket Trader" import format).

- Allocation portfolios: the tree is evaluated on the latest bar, as if rebalancing today (leverage
  applied). That is the target a rebalance today would trade to.
- Signal strategies: the positions the backtest holds after the latest bar (sized as in the
  backtest, relative to its equity), plus new entry signals that fill at the next open/close,
  sized with the strategy's sizing rule, up to the free position slots.
"""
from __future__ import annotations

import csv
import io
import math
import re

import numpy as np
import pandas as pd

from . import data, runner
from .portfolio import Portfolio, _Evaluator, tickers_in


def parse_holdings(text: str) -> dict[str, float]:
    """'TICKER,shares' lines (commas, tabs, spaces or semicolons; an optional header) -> {ticker: shares}."""
    out: dict[str, float] = {}
    for n, line in enumerate((text or "").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in re.split(r"[,;\t ]+", line) if p]
        if len(parts) < 2:
            raise ValueError(f"Holdings line {n} ({line!r}) needs a ticker and a share count, e.g. SPY,10")
        try:
            q = float(parts[1].replace("_", ""))
        except ValueError:
            if n == 1:  # header row such as "symbol,quantity"
                continue
            raise ValueError(f"Holdings line {n}: {parts[1]!r} is not a number of shares")
        t = data.canonical(parts[0])
        out[t] = out.get(t, 0.0) + q
    return out


def _last_close(t: str) -> tuple[float | None, str | None]:
    try:
        df = data.load(t)
    except data.DataError:
        return None, None
    return float(df["close"].iloc[-1]), str(df.index[-1].date())


def portfolio_targets(p: Portfolio) -> tuple[str, dict[str, float], list[str]]:
    p.validate()
    dfs = data.load_many(tickers_in(p.tree))
    cal = None
    for df in dfs.values():
        cal = df.index if cal is None else cal.union(df.index)
    cal = cal[cal >= cal[-1] - pd.Timedelta(days=3 * 366)]  # enough history for any lookback
    ev = _Evaluator(p, cal, dfs)
    w = ev.eval(p.tree, len(cal) - 1)
    w = {t: x * p.leverage for t, x in w.items() if t != "cash" and abs(x) > 1e-9}
    return str(cal[-1].date()), w, list(p.notes)


def active_branches(p: Portfolio) -> dict:
    """Which branch of each if/else the tree takes on the latest bar (the Build page's "active today" badges):
    {"as_of": date, "branches": ["then" | "else" | None, ...]}, one entry per if-node in pre-order (a node, then its
    children in order: a group's or filter's list, then an if's then and else branches, then a filter's fallback), the
    order the editor draws them. None when the condition cannot be evaluated yet (no data)."""
    p.validate()
    dfs = data.load_many(tickers_in(p.tree))
    cal = None
    for df in dfs.values():
        cal = df.index if cal is None else cal.union(df.index)
    if cal is None or not len(cal):
        return {"as_of": None, "branches": []}
    cal = cal[cal >= cal[-1] - pd.Timedelta(days=3 * 366)]  # enough history for any lookback
    ev = _Evaluator(p, cal, dfs)
    i = len(cal) - 1
    out: list = []

    def walk(n) -> None:
        if not isinstance(n, dict):
            return
        if "if" in n:
            try:
                v = ev.series(n["if"], data.canonical(n.get("on", "SPY")), "bool")[i]
                out.append(None if v is None or (isinstance(v, float) and np.isnan(v)) else ("then" if bool(v) else "else"))
            except Exception:  # noqa: BLE001 - a rule the page is still editing: no badge
                out.append(None)
            walk(n.get("then"))
            walk(n.get("else"))
            return
        for k in n.get("children") or []:
            walk(k)
        if "filter" in n and n.get("fallback"):
            walk(n["fallback"])
    walk(p.tree)
    return {"as_of": str(cal[-1].date()), "branches": out}


def signal_targets(s, account_value: float, entry_orders: list | None = None, catch_up: dict | None = None
                   ) -> tuple[str, dict[str, float], list[str], dict[str, float]]:
    """-> as_of, {ticker: weight}, notes, {ticker: fixed shares} (fixed_shares sizing, and limit/stop entries,
    which are sized at their order price). For limit / stop entries, the engine's working entry orders
    (signals.entry_orders) are appended to `entry_orders` when a list is given. `catch_up` (a dict, when given)
    receives {ticker: entry date} for the positions the backtest already holds from an entry that has passed (an
    earlier session, or today's open / intraday level): an account without them would be catching up, not
    following a signal due now."""
    from . import signals
    res = runner.run(s)
    notes: list[str] = []
    eq = float(res.equity.iloc[-1])
    as_of = str(res.equity.index[-1].date())
    tr = res.trades
    held: dict[str, float] = {}
    exits, standing = signals.exit_instructions(s, res)
    # positions the engine sells at the next open (an exit rule filled at the open, a holding period ending
    # at the open) are not part of the target: they are SELL orders now
    leaving = {e["ticker"] for e in exits if not e.get("done") and e["when"] == "at the next open"}
    for e in exits:
        if e["ticker"] in leaving and not e.get("done"):
            notes.append(f"{e['ticker']}: {e['action'].lower()} at the next open ({e['reason']}), as the backtest does.")
    for o in standing:
        if o.get("price") is not None:
            notes.append(f"{o['ticker']}: place a {o['action']} {o['order']} order at {o['price']} for the next session "
                         f"({o['reason']}{'; one-cancels-other with the other exit order' if o.get('oca') else ''}).")
        else:
            notes.append(f"{o['ticker']}: {o['action'].lower()} at the next open if {o['reason']}.")
    if tr is not None and not tr.empty:
        for _, r in tr[tr["exit_reason"] == "open at end"].iterrows():
            if r.ticker in leaving:
                continue
            px, _ = _last_close(r.ticker)
            sign = -1 if r.side == "short" else 1
            if px and eq > 0:
                held[r.ticker] = held.get(r.ticker, 0.0) + sign * float(getattr(r, "exit_shares", r.shares)) * px / eq
            if catch_up is not None:
                ed = str(pd.Timestamp(r.entry_date).date())
                # a market entry at today's close is the usual one-day-late market-on-close order, not a catch-up
                due_now = ed == as_of and s.entry_fill == "close" and s.entry_order == "market"
                if not due_now:
                    catch_up[r.ticker] = min(catch_up.get(r.ticker, ed), ed)
    fixed: dict[str, float] = {}
    if s.entry_order != "market":
        # limit / stop entries: the orders the engine has working for the next session, at the engine's levels
        pend = signals.entry_orders(s, res)
        free = max(int(s.max_positions) - len(held), 0)
        placed = 0
        for e in pend:
            t = e["ticker"]
            if t in held or t in fixed:
                notes.append(f"{t}: an entry order at {e['price']} is working while the position is held (pyramiding); "
                             "add it by hand if wanted.")
                continue
            sign = -1 if e["side"] == "short" else 1
            lvl = float(e["price"])
            if s.sizing == "fixed_shares" and s.fixed_amount:
                q = float(s.fixed_amount)
            else:
                if s.sizing == "fixed_dollars" and s.fixed_amount:
                    w = float(s.fixed_amount) / account_value
                else:
                    if s.sizing in ("risk", "volatility"):
                        notes.append(f"{s.sizing} sizing needs the fill-day stop/volatility; entry orders use "
                                     f"{s.position_size:.0%} of the account each instead.")
                    w = float(s.position_size)
                q = float(math.floor(w * account_value / lvl + 1e-9)) if lvl > 0 else 0.0
            fixed[t] = sign * q
            held[t] = sign * q * lvl / account_value
            placed += 1
            if entry_orders is not None:
                entry_orders.append({**e, "shares": q})
            span = ("the next session" if e["valid_sessions"] == 1
                    else f"{e['valid_sessions']} sessions (until {e['until']})")
            br = ", ".join(x for x in (f"stop loss {e['stop_loss']}" if "stop_loss" in e else "",
                                       f"take profit {e['take_profit']}" if "take_profit" in e else "") if x)
            notes.append(f"{t}: {e['action'].lower()} with a {e['order'].lower()} order at {e['price']} "
                         f"(signal on {e['placed']}, close {e['close']}), working for {span}, as the backtest does"
                         + (f"; bracket (the two exits one-cancels-other): {br}" if "," in br
                            else f"; attached exit once filled: {br}" if br else "") + ".")
        if placed > free:
            notes.append(f"{placed} entry orders for {free} free position slot(s): as in the backtest, the first to fill "
                         "take the slots; cancel the others once the slots are full.")
        if not pend:
            notes.append(f"No {s.entry_order} entry orders are working for the next session.")
    elif s.entry_fill != "close":
        sc = signals.scan(s)
        free = max(int(s.max_positions) - len(held), 0)
        new = [e for e in sc.get("entry_signals", []) if e["ticker"] not in held][:free]
        if len(sc.get("entry_signals", [])) > free + len([e for e in sc.get("entry_signals", []) if e["ticker"] in held]):
            notes.append("More entry signals than free position slots: kept the best-ranked ("
                         + (f"{'lowest' if s.rank_ascending else 'highest'} {s.rank_by}" if s.rank_by else
                            "highest 20-day average dollar volume, the default") + "), as the backtest does.")
        for e in new:
            sign = -1 if e["side"] == "short" else 1
            if s.sizing == "fixed_dollars" and s.fixed_amount:
                w = float(s.fixed_amount) / account_value
            elif s.sizing == "fixed_shares" and s.fixed_amount:
                fixed[e["ticker"]] = sign * float(s.fixed_amount)
                w = sign * float(s.fixed_amount) * e["close"] / account_value
            else:
                if s.sizing in ("risk", "volatility"):
                    notes.append(f"{s.sizing} sizing needs the entry-day stop/volatility; new entries use "
                                 f"{s.position_size:.0%} of the account each instead.")
                w = float(s.position_size)
            held[e["ticker"]] = held.get(e["ticker"], 0.0) + sign * w
        if new:
            notes.append(f"{len(new)} new entry signal(s) fill at the {s.entry_fill.replace('_', ' ')}.")
    else:
        notes.append("Entries fill at the close of the signal day, so today's signals are already in the positions.")
    return as_of, held, notes, fixed


def todays_orders(spec, account_value: float, holdings_text: str = "", whole_shares: bool = True) -> dict:
    if not (account_value and account_value > 0):
        raise ValueError("Enter the current account value (a positive dollar amount).")
    current = parse_holdings(holdings_text)
    fixed: dict[str, float] = {}
    pend: list[dict] = []
    catch = {}
    if isinstance(spec, Portfolio):
        as_of, target, notes = portfolio_targets(spec)
        order_type = "MOC" if spec.fill == "close" else "MKT"
    else:
        spec.validate()
        catch: dict = {}
        as_of, target, notes, fixed = signal_targets(spec, account_value, pend, catch)
        order_type = "MKT"
    lvl_orders = {e["ticker"]: e for e in pend}
    rows = []
    for t in list(dict.fromkeys([*target, *current])):
        px, px_date = _last_close(t)
        w = target.get(t, 0.0)
        cur = current.get(t, 0.0)
        if t in fixed:
            tgt = fixed[t]
        elif px:
            tgt = w * account_value / px
            if whole_shares:
                tgt = float(math.trunc(tgt))
        else:
            tgt = 0.0 if not w else float("nan")
        if not np.isfinite(tgt):
            notes.append(f"No price for {t}: it was skipped.")
            continue
        delta = tgt - cur
        if abs(delta) < 1e-9:
            side = "hold"
        else:
            side = "BUY" if delta > 0 else "SELL"
        row = {"ticker": t, "side": side, "shares": round(abs(delta), 6), "est_price": px,
               "value": round(abs(delta) * px, 2) if px else None, "current_shares": cur,
               "target_shares": tgt, "target_weight": round(w, 6), "price_date": px_date}
        if t in catch and side != "hold" and abs(tgt) > abs(cur) + 1e-9 and tgt * cur >= 0 and t not in lvl_orders:
            # the backtest already holds this position (entered catch[t]); the account has less of it
            row["catch_up"] = catch[t]
        e = lvl_orders.get(t)
        if e is not None and side != "hold":
            row.update({"order_type": "LMT" if e["order"] == "LIMIT" else "STP", "order_price": e["price"],
                        "valid_sessions": e["valid_sessions"], "est_price": e["price"],
                        "value": round(abs(delta) * e["price"], 2)})
            for k in ("stop_loss", "take_profit"):
                if k in e:
                    row[k] = e[k]
        rows.append(row)
    rows.sort(key=lambda r: (r["side"] == "hold", r["side"] != "SELL", r["ticker"]))
    trades = [r for r in rows if r["side"] != "hold"]
    untradable = [r["ticker"] for r in trades if r["ticker"].startswith("^")]
    if untradable:
        notes.append(f"{', '.join(untradable)} is an index, not a tradable security: use an ETF that tracks it.")
    buys = sum(r["value"] or 0 for r in trades if r["side"] == "BUY")
    sells = sum(r["value"] or 0 for r in trades if r["side"] == "SELL")
    return {"as_of": as_of, "account_value": account_value, "orders": rows, "notes": list(dict.fromkeys(notes)),
            "buy_value": round(buys, 2), "sell_value": round(sells, 2),
            "invested_after": round(sum(abs(r["target_shares"]) * (r["est_price"] or 0) for r in rows), 2),
            "entry_orders": pend, "csv": generic_csv(trades), "ib_csv": ib_basket_csv(trades, order_type)}


def generic_csv(rows: list[dict]) -> str:
    f = io.StringIO()
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["ticker", "side", "shares", "est_price", "value", "current_shares", "target_shares", "target_weight"])
    for r in rows:
        w.writerow([r["ticker"], r["side"], _n(r["shares"]), _n(r["est_price"]), _n(r["value"]), _n(r["current_shares"]),
                    _n(r["target_shares"]), _n(r["target_weight"])])
    return f.getvalue()


def ib_basket_csv(rows: list[dict], order_type: str = "MKT") -> str:
    """Interactive Brokers TWS basket file (File > Import/Export > Import Basket)."""
    f = io.StringIO()
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["Action", "Quantity", "Symbol", "SecType", "Exchange", "Currency", "TimeInForce", "OrderType",
                "LmtPrice", "BasketTag", "OrderRef", "AuxPrice"])
    for r in rows:
        if r["ticker"].startswith("^") or not r["shares"]:
            continue
        ot = r.get("order_type") or order_type          # limit / stop entries carry their own type and price
        tif = "DAY" if int(r.get("valid_sessions") or 1) <= 1 else "GTC"
        w.writerow([r["side"], _n(r["shares"]), r["ticker"].replace("-", " "), "STK", "SMART", "USD", tif, ot,
                    _n(float(r["order_price"])) if r.get("order_type") == "LMT" else "", "Backtester",
                    "entry" if r.get("order_type") else "rebalance", _n(float(r["order_price"])) if r.get("order_type") == "STP" else ""])
    return f.getvalue()


def _n(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:.6f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
