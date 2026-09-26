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
    sig = []
    for t in spec.universe:
        t = data.canonical(t)
        df = data.load(t)
        if df.index[-1] != last:
            continue
        ns = expr.Namespace(df)
        rules = [("long" if spec.side != "short" else "short", spec.entry)]
        if spec.side == "both":
            rules.append(("short", spec.short_entry))
        for side, rule in rules:
            if bool(expr.evaluate(rule, ns).iloc[-1]):
                sig.append({"ticker": t, "side": side, "close": round(float(df["close"].iloc[-1]), 4)})
    if spec.universe_name == "NDX" and spec.point_in_time:
        cur = set(data.nasdaq100())
        sig = [s for s in sig if s["ticker"] in cur]
    out["entry_signals"] = sig
    fill = {"close": "at today's close (market-on-close)", "open": "at the next open (rule is open-time)",
            "next_open": "at the next open", "next_close": "at the next close"}[spec.entry_fill]
    out["action"] = (f"{len(sig)} entry signal(s) {fill}" if sig else "No new entry signals") + \
                    (f"; {len(out['open_positions'])} position(s) open" if out["open_positions"] else "")
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
