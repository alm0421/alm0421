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

PAPER = data.ROOT / "paper"


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
    held = {r["ticker"] for r in out["open_positions"]}
    sig, assumed = [], False
    for t in spec.universe:
        t = data.canonical(t)
        df = data.load(t)
        if df.index[-1] != last:
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
            ns = expr.Namespace(ext, ticker=t)
            assumed = assumed or bool({"open", "gap"} & expr.names_in(spec.entry if isinstance(spec.entry, str) else ""))
        else:
            ns = expr.Namespace(df, ticker=t)
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
    out["action"] = (f"{len(sig)} entry signal(s) {fill}" if sig else "No new entry signals") + \
                    (f"; {len(out['open_positions'])} position(s) open" if out["open_positions"] else "") + \
                    (f"; {len(over)} more signal(s) with no free slot" if over else "")
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


def format_alert(rows) -> str:
    lines = []
    for r in rows if isinstance(rows, list) else [rows]:
        s = r.get("today", r)
        lines.append(f"*{r.get('name', s.get('description', 'strategy'))}* ({s['as_of']}): {s.get('action', '')}")
        for e in s.get("entry_signals", [])[:20]:
            lines.append(f"  • {e['side']} {e['ticker']} @ {e['close']}")
        if s.get("target_weights"):
            lines.append("  target: " + ", ".join(f"{t} {w:.0%}" for t, w in s["target_weights"].items()))
    return "\n".join(lines)
