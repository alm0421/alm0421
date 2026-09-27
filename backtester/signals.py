"""What does a strategy say to do today? And forward ("paper") tracking of saved strategies.

`scan(spec)` runs the strategy through the latest data and reports: entry signals on the latest
bar, positions that would be open now, and (for allocations) the current target weights.

Paper trading: `paper_add` stores a strategy with today's date; `paper_report` re-runs every stored
strategy from its registration date, so the results only contain days that happened after it was
saved - a genuine forward test. The "Signals" GitHub Action posts the daily scan to a webhook.
"""
from __future__ import annotations

import dataclasses
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, expr, metrics, runner
from .portfolio import Portfolio
from .strategy import Strategy
from .engine import strategy_namespace

PAPER = data.ROOT / "paper"


def _sessions_after(last: pd.Timestamp, n: int, ticker: str) -> pd.Timestamp:
    """The n-th trading session after `last` (calendar days for crypto, NYSE sessions otherwise)."""
    if expr.is_crypto(ticker):
        return last + pd.Timedelta(days=n)
    from . import calendar as _cal
    return _cal.next_sessions(last, n)[-1]


def exit_instructions(spec: Strategy, res) -> tuple[list[dict], list[dict]]:
    """What the engine does with the open positions, as orders: (exits, standing orders).

    exits: market exits - the ones the engine made at the latest close (MOC orders, sold "at today's close"),
    exits scheduled for the next open (an exit rule filled at the next open, a holding period ending at an open),
    and holding periods ending at a later close/open (with the date). standing orders: for every position still
    open, the stop (stop order) and target / scale-out (limit orders) levels that apply at the next session, as
    the engine uses them; a stop and a target form a one-cancels-other (OCA) pair."""
    last = res.equity.index[-1]
    exits: list[dict] = []
    standing: list[dict] = []
    tr = res.trades
    word = {"long": ("SELL", "sell"), "short": ("BUY TO COVER", "cover")}
    if tr is not None and not tr.empty:
        today = tr[(pd.to_datetime(tr["exit_date"]) == last) & (tr["exit_reason"] != "open at end")]
        for (t, side, reason, fill), g in today.groupby(["ticker", "side", "exit_reason", "exit_fill"], sort=False):
            when = {"close": "at today's close", "open": "at today's open", "intraday": "today (intraday)"}.get(fill, fill)
            exits.append({"ticker": t, "side": side, "action": word[side][0], "when": when,
                          "order": {"close": "MOC", "open": "MOO"}.get(fill, "STOP/LIMIT"), "reason": reason,
                          "date": str(last.date()), "shares": round(float(g["exit_shares" if "exit_shares" in g else "shares"].sum()), 6),
                          "price": round(float(g["exit_price"].iloc[-1]), 4), "done": True})
    for st in res.extras.get("open_state", []) or []:
        t, side = st["ticker"], st["side"]
        act = word[side][0]
        base = {"ticker": t, "side": side, "action": act, "shares": round(st["shares"], 6)}
        left = st.get("hold_bars_left")
        if st.get("pending_open_exit"):
            nxt = _sessions_after(last, 1, t)
            exits.append({**base, "when": "at the next open", "order": "MOO", "reason": st.get("pending_reason") or "exit rule",
                          "date": str(nxt.date())})
            continue       # sold at the open, before any stop or target can act
        if left is not None and left >= 1:
            day = _sessions_after(last, left, t)
            at_open = st.get("hold_exit_open")
            when = (("at the next open" if at_open else "at the next close") if left == 1
                    else f"at the {'open' if at_open else 'close'} on {day.date()}")
            exits.append({**base, "when": when, "order": "MOO" if at_open else "MOC", "reason": "time exit",
                          "date": str(day.date())})
            if left == 1 and at_open:
                continue
        oca = f"{t}-exit" if (st.get("stop") is not None and st.get("target") is not None) else None
        if st.get("stop") is not None:
            standing.append({**base, "order": "STOP", "price": round(st["stop"], 4), "reason": st.get("stop_reason") or "stop",
                             "valid": "next session (update after each close)", "oca": oca})
        if st.get("target") is not None:
            standing.append({**base, "order": "LIMIT", "price": round(st["target"], 4), "reason": "take profit",
                             "valid": "next session (update after each close)", "oca": oca})
        for so in st.get("scale_out") or []:
            standing.append({**base, "order": "LIMIT", "price": round(so["level"], 4), "shares": round(st["shares"] * so["fraction"], 6),
                             "reason": f"scale out {so['fraction']:.0%} at " + (f"{so['r']:g}R" if "r" in so else f"${so['points']:g}" if "points" in so else f"+{so['at']:.1%}"), "valid": "next session", "oca": None})
        if spec.exit_when and spec.exit_when_fill == "open" and isinstance(spec.exit_when, str):
            standing.append({**base, "order": "MOO if", "price": None, "reason": f"exit rule {spec.exit_when} (checked at the open)",
                             "valid": "next open", "oca": None})
    return exits, standing


def scan(spec) -> dict:
    res = runner.run(spec)
    last = res.equity.index[-1]
    out: dict = {"as_of": str(last.date()), "kind": res.kind, "description": spec.description,
                 "interpretation": spec.summary()}
    if isinstance(spec, Portfolio):
        hw = res.holdings
        out["target_weights"] = {t: round(float(w), 4) for t, w in hw.iloc[-1].items() if abs(w) > 1e-6}
        out["action"] = "hold " + ", ".join(f"{w:.0%} {t}" for t, w in sorted(out["target_weights"].items(), key=lambda kv: -kv[1]))
        return out
    tr = res.trades
    open_now = tr[tr["exit_reason"] == "open at end"] if tr is not None and not tr.empty else pd.DataFrame()
    out["open_positions"] = [
        {"ticker": r.ticker, "side": r.side, "since": str(r.entry_date), "entry_price": round(float(r.entry_price), 4),
         "last_price": round(float(r.exit_price), 4), "unrealized_return": round(float(r["return"]), 4)}
        for _, r in open_now.iterrows()]
    exits, standing = exit_instructions(spec, res)
    out["exit_signals"] = exits
    out["exit_orders"] = standing
    held = {r["ticker"] for r in out["open_positions"]}
    # entries at the close: the engine has already filled today's signals at today's close (market-on-close), so
    # they are open positions "since today"; they are today's BUY (or SELL SHORT) actions, not positions held before
    # the signal. A ticker sold at today's close and bought back at the same close is, net, still held.
    filled_today: list[dict] = []
    at_close = spec.entry_fill == "close" and spec.entry_order == "market"
    if at_close and not open_now.empty:
        sold_moc = {e["ticker"] for e in exits if e.get("done") and e["order"] == "MOC"}
        since = pd.to_datetime(open_now["entry_date"])
        new_rows = open_now[(since == last) & (open_now["entry_fill"] == "close")]
        for _, r in new_rows.iterrows():
            filled_today.append({"ticker": r.ticker, "side": r.side, "action": "BUY" if r.side == "long" else "SELL SHORT",
                                 "when": "at today's close", "order": "MOC", "shares": round(float(r.shares), 6),
                                 "price": round(float(r.entry_price), 4), "close": round(float(r.exit_price), 4),
                                 "done": True, "rebought": r.ticker in sold_moc})
        before = set(open_now.loc[since < last, "ticker"])
        held = {t for t in held if t in before}      # held before today's close (a pyramided add keeps it held)
    sig, assumed = [], False
    if spec.entry_order != "market":
        return _scan_level_orders(spec, res, out, exits, standing)
    xtab = {}
    if any(expr.xrank_calls(r) for r in (spec.entry, spec.short_entry, spec.rank_by)):
        # xrank(): the universe's percentiles on the scanned day, as the engine computes them
        from .engine import _xrank_tables, _union_index
        dfs_x = {}
        for t in spec.universe:
            try:
                d_ = data.load(t).loc[:last]
            except data.DataError:
                continue
            if len(d_):
                dfs_x[data.canonical(t)] = d_
        cal_x = _union_index([d_.index for d_ in dfs_x.values()])
        if cal_x is not None and len(dfs_x) > 1:
            mem = data.member_mask(list(dfs_x), cal_x)[0] if spec.universe_name == "NDX" and spec.point_in_time else None
            xtab = _xrank_tables(spec, dfs_x, list(dfs_x), cal_x, mem)
    for t in spec.universe:
        t = data.canonical(t)
        df = data.load(t).loc[:last]   # a run that ends earlier (spec.end) is scanned as of its last day
        if not len(df) or df.index[-1] != last:
            continue
        if spec.entry_fill == "open":
            # an "at the open" rule decides on the NEXT bar's open: evaluate it on a provisional next bar
            # whose open is today's close (exact for the parts that use earlier bars; open-dependent parts
            # such as gaps are only a preview and are flagged)
            from . import calendar as _cal
            nxt = _cal.next_sessions(last)[0]
            c = float(df["close"].iloc[-1])
            row = {col: df[col].iloc[-1] for col in df.columns}
            row.update({"open": c, "high": c, "low": c, "close": c, "volume": 0.0, "dividend": 0.0})
            if "adj_close" in df:
                row["adj_close"] = float(df["adj_close"].iloc[-1])
            if "open_ok" in df:
                row["open_ok"] = True
            ext = pd.concat([df, pd.DataFrame([row], index=pd.DatetimeIndex([nxt]))])
            ns = strategy_namespace(spec, ext, t)
            assumed = assumed or bool({"open", "gap"} & expr.names_in(spec.entry if isinstance(spec.entry, str) else ""))
        else:
            ns = strategy_namespace(spec, df, t)
        if xtab:
            ns.xrank_table = xtab.get(t, {})
        rules = [("long" if spec.side != "short" else "short", spec.entry)]
        if spec.side == "both":
            rules.append(("short", spec.short_entry))
        for side, rule in rules:
            if bool(expr.evaluate(rule, ns).iloc[-1]):
                rank = None
                if spec.rank_by and isinstance(spec.rank_by, str):
                    try:
                        rank = float(expr.evaluate_value(spec.rank_by, expr.Namespace(df, ticker=t)).iloc[-1])
                    except Exception:  # noqa: BLE001
                        rank = None
                if rank is None:
                    rank = float((df["close"] * df["volume"]).rolling(20, min_periods=1).mean().iloc[-1])
                sig.append({"ticker": t, "side": side, "close": round(float(df["close"].iloc[-1]), 4), "_rank": rank})
    if spec.universe_name == "NDX" and spec.point_in_time:
        cur = set(data.nasdaq100())
        sig = [s for s in sig if s["ticker"] in cur]
    # like the engine: no second entry in a ticker already held (unless pyramiding), and only as many
    # new positions as there are free slots, best-ranked first
    skipped = []
    if getattr(spec, "pyramiding", 1) <= 1:
        skipped = [s["ticker"] for s in sig if s["ticker"] in held]
        sig = [s for s in sig if s["ticker"] not in held]
    free = max(0, int(spec.max_positions) - len(held))
    sig.sort(key=lambda s: s["_rank"], reverse=not getattr(spec, "rank_ascending", False))
    if at_close:
        # the engine's own fills at today's close are the entries (its ranking, slots and sizing decided them)
        done = {f["ticker"] for f in filled_today}
        over = [s["ticker"] for s in sig if s["ticker"] not in done]
        sig = filled_today
    else:
        over = [s["ticker"] for s in sig[free:]]
        sig = sig[:free]
    for s_ in sig:
        s_.pop("_rank", None)
    out["entry_signals"] = sig
    out["skipped_already_held"] = skipped
    out["skipped_no_free_slot"] = over
    if assumed:
        out["note"] = ("The rule depends on the next open (e.g. a gap): this preview assumes the market opens at "
                       "today's close; the order only triggers if the actual open meets the condition.")
    fill = {"close": "at today's close (market-on-close)", "open": "at the next open (rule is open-time)",
            "next_open": "at the next open", "next_close": "at the next close"}[spec.entry_fill]
    todo = [e for e in exits if not e.get("done") and e["when"] in ("at the next open", "at the next close")]
    # an exit at today's close is a market-on-close order (like entries at the close); earlier ones already happened
    moc = [e for e in exits if e.get("done") and e["order"] == "MOC"]
    rebought = {f["ticker"] for f in sig if f.get("rebought")}
    exit_txt = "; ".join([f"{e['action']} {e['ticker']} {e['when']} ({e['reason']})" for e in moc + todo
                          if not (e.get("done") and e["ticker"] in rebought)])
    still_open = len(out["open_positions"]) - len({e["ticker"] for e in todo if e["when"] == "at the next open"})
    if at_close:
        parts = [f"{f['action']} {f['ticker']} at today's close (new entry)" for f in sig if not f.get("rebought")]
        for f in sig:
            if f.get("rebought"):
                e = next((e for e in moc if e["ticker"] == f["ticker"]), None)
                parts.append(f"{f['ticker']}: {e['action'] if e else 'SELL'} at today's close ({e['reason'] if e else 'exit'}) "
                             f"and {f['action']} again at the same close (the entry signal fired again): net, keep holding "
                             f"{f['ticker']} as a new position from today")
        entry_txt = ("; ".join(parts) + f" ({len(sig)} entr{'y' if len(sig) == 1 else 'ies'} filled at today's close, "
                     "market-on-close)") if sig else "No new entry signals"
    else:
        entry_txt = f"{len(sig)} entry signal(s) {fill}" if sig else "No new entry signals"
    out["action"] = ((exit_txt + "; ") if exit_txt else "") + entry_txt + \
                    (f"; {still_open} position(s) open" if still_open > 0 else "") + \
                    (f" (exit orders for the next session: " + ", ".join(
                        f"{o['action']} {o['order']} {o['ticker']} @ {o['price']}" for o in standing if o.get("price") is not None) + ")"
                     if any(o.get("price") is not None for o in standing) else "") + \
                    (f"; {len(over)} more signal(s) with no free slot" if over else "")
    return out


def entry_orders(spec: Strategy, res) -> list[dict]:
    """The limit / stop entry orders the engine has working after the last bar, as orders for the next session(s):
    {ticker, side, action, order ("LIMIT"/"STOP"), price, valid_sessions, placed, close, stop_loss, take_profit}.
    The level is the engine's own (the entry_level expression on the signal bar), so a limit 1% below the close
    is 0.99 x the signal day's close; the order works for `valid_sessions` more sessions, as in the backtest
    (order_valid_bars counted from the signal bar). stop_loss / take_profit are the bracket levels the engine
    applies after a fill at that price."""
    out = []
    last = res.equity.index[-1]
    for od in res.extras.get("pending_entries", []) or []:
        t, sgn = od["ticker"], 1 if od["side"] == "long" else -1
        lvl = float(od["level"])
        try:
            close = float(data.load(t)["close"].loc[:last].iloc[-1])
        except Exception:  # noqa: BLE001
            close = None
        row = {"ticker": t, "side": od["side"], "action": "BUY" if sgn == 1 else "SELL SHORT",
               "order": od["order"].upper(), "price": round(lvl, 4), "valid_sessions": int(od["sessions_left"]),
               "placed": od["placed"], "close": None if close is None else round(close, 4)}
        if spec.stop_loss:
            row["stop_loss"] = round(lvl * (1 - sgn * spec.stop_loss), 4)
        if spec.take_profit:
            row["take_profit"] = round(lvl * (1 + sgn * spec.take_profit), 4)
        row["until"] = str(_sessions_after(last, int(od["sessions_left"]), t).date())
        out.append(row)
    return out


def _scan_level_orders(spec: Strategy, res, out: dict, exits: list[dict], standing: list[dict]) -> dict:
    """scan() for limit / stop entries: the signal is an order at a level, not a fill at the close."""
    pend = entry_orders(spec, res)
    out["entry_signals"] = [{"ticker": o["ticker"], "side": o["side"], "close": o["close"], "order": o["order"],
                             "level": o["price"], "valid_sessions": o["valid_sessions"], "until": o["until"],
                             "placed": o["placed"],
                             **{k: o[k] for k in ("stop_loss", "take_profit") if k in o}} for o in pend]
    out["entry_orders"] = pend
    out["skipped_already_held"] = []
    out["skipped_no_free_slot"] = []
    held = len(out["open_positions"])
    if len(pend) > max(0, int(spec.max_positions) - held):
        out["note"] = (f"{len(pend)} entry orders are working but only {max(0, int(spec.max_positions) - held)} position "
                       "slot(s) are free: as in the backtest, the ones that fill first take the slots and the rest are "
                       "not filled.")
    todo = [e for e in exits if not e.get("done") and e["when"] in ("at the next open", "at the next close")]
    moc = [e for e in exits if e.get("done") and e["order"] == "MOC"]
    parts = [f"{e['action']} {e['ticker']} {e['when']} ({e['reason']})" for e in moc + todo]
    for o in pend:
        br = ", ".join(x for x in (f"stop {o['stop_loss']}" if "stop_loss" in o else "",
                                   f"target {o['take_profit']}" if "take_profit" in o else "") if x)
        parts.append(f"{o['action']} {o['order']} {o['ticker']} @ {o['price']} "
                     f"(entry order, {'the next session' if o['valid_sessions'] == 1 else str(o['valid_sessions']) + ' sessions, until ' + o['until']}"
                     + (f"; bracket {br}" if br else "") + ")")
    if not pend:
        parts.append("No new entry orders")
    if out["open_positions"]:
        parts.append(f"{len(out['open_positions'])} position(s) open")
    txt = "; ".join(parts)
    if any(o.get("price") is not None for o in standing):
        txt += " (exit orders for the next session: " + ", ".join(
            f"{o['action']} {o['order']} {o['ticker']} @ {o['price']}" for o in standing if o.get("price") is not None) + ")"
    out["action"] = txt
    return out


def paper_add(spec, name: str) -> Path:
    PAPER.mkdir(exist_ok=True)
    d = runner.to_dict(spec)
    d["start"] = str(date.today())
    rec = {"name": name, "registered": str(date.today()), "spec": d}
    p = PAPER / f"{name}.json"
    p.write_text(json.dumps(rec, indent=2))
    return p


def paper_report() -> list[dict]:
    out = []
    for p in sorted(PAPER.glob("*.json")):
        rec = json.loads(p.read_text())
        spec = runner.from_dict(rec["spec"])
        row = {"name": rec["name"], "registered": rec["registered"], "description": spec.description}
        try:
            res = runner.run(spec)
            st = metrics.equity_stats(res.equity, flows=res.extras.get("flows"))
            row.update({"days": len(res.equity) - 1, "return": st["total_return"], "max_drawdown": st["max_drawdown"],
                        "value": st["end_equity"], "trades": len(res.trades) if res.trades is not None else 0})
        except ValueError:
            row.update({"days": 0, "note": "no trading days since registration yet"})
        row["today"] = scan(dataclasses.replace(spec, start=None, notes=[]))
        out.append(row)
    return out


def post_webhook(url: str, payload: dict) -> bool:
    import urllib.request
    body = json.dumps({"text": format_alert(payload), "data": payload}, default=str).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310 - user-configured webhook
            return 200 <= r.status < 300
    except Exception:  # noqa: BLE001
        return False


def entry_line(e: dict) -> str:
    """One entry signal in words: a fill the engine made at today's close (BUY ... at today's close), or a signal
    for the next fill (long X @ close)."""
    if e.get("done"):
        return (f"{e['action']} {e['ticker']} {e.get('when', 'at the close')} @ {e['price']}"
                + (f", {e['shares']:g} shares" if e.get("shares") is not None else "")
                + (" (sold and bought back at the same close: net, keep holding it)" if e.get("rebought") else " (new entry)"))
    return f"{e['side']} {e['ticker']} @ {e['close']}"


def format_alert(rows) -> str:
    lines = []
    for r in rows if isinstance(rows, list) else [rows]:
        s = r.get("today", r)
        lines.append(f"*{r.get('name', s.get('description', 'strategy'))}* ({s['as_of']}): {s.get('action', '')}")
        for e in s.get("exit_signals", [])[:20]:
            lines.append(f"  • {e['action']} {e['ticker']} {e['when']} ({e['reason']}"
                         + (", already filled" if e.get("done") and e.get("order") != "MOC" else "") + ")")
        for o in s.get("exit_orders", [])[:20]:
            if o.get("price") is not None:
                lines.append(f"  • {o['action']} {o['order']} {o['ticker']} @ {o['price']} ({o['reason']}"
                             + (", OCA with the other exit order" if o.get("oca") else "") + ")")
            else:
                lines.append(f"  • {o['action']} {o['ticker']} at the next open if {o['reason']}")
        for e in s.get("entry_signals", [])[:20]:
            if e.get("order") and not e.get("done"):
                lines.append(f"  • {e['side']} {e['ticker']}: {e['order']} order @ {e['level']} (close {e['close']}, "
                             f"valid {e['valid_sessions']} session(s))")
            else:
                lines.append("  • " + entry_line(e))
        if s.get("target_weights"):
            lines.append("  target: " + ", ".join(f"{t} {w:.0%}" for t, w in s["target_weights"].items()))
    return "\n".join(lines)
