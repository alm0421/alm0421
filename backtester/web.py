"""The backtesting site: a local web app (standard library only).

    python -m backtester web [--port 8000] [--host 127.0.0.1]

Pages: Backtest (plain English, live interpretation), Community (published strategies: search, sort,
fork, run), Build (block editor for portfolios and
rule forms for signal strategies), Compare, Research (parameter sweep, walk-forward, portfolio
optimiser), Signals & paper trading, History (saved runs with share links) and Data.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import difflib
import hashlib
import json
import mimetypes
import os
import re
import threading
import time
import traceback
import urllib.parse
import zlib
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import data, expr, parser, report, runner

APP = Path(__file__).with_name("webapp.html")
RUNS = report.ROOT / "reports" / "runs"
LOCK = threading.Lock()

EXAMPLES = [
    "buy at the close Microsoft when it trades down 5 days in a row, hold for 1 day, and sell at the close",
    "buy Nasdaq 100 stocks at the close when RSI(2) is below 5 and it is above its 200-day moving average, sell when it closes above its 5-day moving average, max 5 positions, since 2005",
    "short QQQ at the open when it gaps up 1%, cover at the close",
    "buy AAPL when MACD crosses above its signal line, sell when MACD crosses below its signal line, 5 bps slippage",
    "buy NVDA when it closes above its 20 day high, with a 2 ATR stop and a 3 ATR trailing stop",
    "hold 60% SPY and 40% TLT, rebalance quarterly",
    "if SPY is above its 200-day moving average hold QQQ, otherwise hold TLT, rebalance daily",
    "hold the top 5 Nasdaq 100 stocks by 6 month momentum, rebalance monthly, since 2006",
    "dual momentum between SPY and EFA with AGG as the safe asset",
    "hold 60% SPY and 40% AGG, withdraw 4% per year adjusted for inflation, starting with $1,000,000, rebalance annually",
    "buy and hold SPY, add $500 every month",
]

OPTION_KEYS = {"capital": float, "start": str, "end": str, "slippage_bps": float, "commission": float,
               "benchmark": str, "name": str}


class ClientError(Exception):
    pass


# ------------------------------------------------------------------ favicon (reports and the site ask for /favicon.ico)

FAVICON_SVG = (b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" rx="3" '
               b'fill="#2a78d6"/><path d="M2 12 L6 7.5 L9 9.5 L14 4" stroke="#fff" stroke-width="2" fill="none" '
               b'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _favicon_ico() -> bytes:
    """A 32x32 .ico (PNG inside) of the same rising line, drawn without any imaging library."""
    import struct
    n = 32
    px = [[(42, 120, 214, 255)] * n for _ in range(n)]
    for (x0, y0), (x1, y1) in zip([(4, 24), (12, 15), (18, 19)], [(12, 15), (18, 19), (28, 8)]):
        steps = max(abs(x1 - x0), abs(y1 - y0)) * 4
        for s in range(steps + 1):
            x, y = x0 + (x1 - x0) * s / steps, y0 + (y1 - y0) * s / steps
            for dx in (-1.5, -0.5, 0.5, 1.5):
                for dy in (-1.5, -0.5, 0.5, 1.5):
                    xi, yi = int(x + dx), int(y + dy)
                    if 0 <= xi < n and 0 <= yi < n and dx * dx + dy * dy <= 4.5:
                        px[yi][xi] = (255, 255, 255, 255)
    for y in range(n):  # rounded corners
        for x in range(n):
            cx, cy = min(x, n - 1 - x), min(y, n - 1 - y)
            if cx < 4 and cy < 4 and (4 - cx) ** 2 + (4 - cy) ** 2 > 18:
                px[y][x] = (0, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(c for p in row for c in p) for row in px)

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return struct.pack("<HHH", 0, 1, 1) + struct.pack("<BBBBHHII", n, n, 0, 0, 1, 32, len(png), 22) + png


FAVICON_ICO = _favicon_ico()


def _index() -> list[dict]:
    p = RUNS / "index.json"
    return json.loads(p.read_text()) if p.exists() else []


def _save_index(rows: list[dict]) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "index.json").write_text(json.dumps(rows, indent=1, default=str))


def _options(body: dict) -> dict:
    out = {}
    for k, typ in OPTION_KEYS.items():
        v = (body.get("options") or {}).get(k)
        if v in (None, ""):
            continue
        try:
            out[k] = typ(v)
        except (TypeError, ValueError):
            raise ClientError(f"Bad value for {k}: {v!r}")
    return out


def _coerce_spec_dict(d: dict) -> dict:
    """Make a hand-edited / form-built spec forgiving: blank values fall back to the field's default and
    numbers typed as text become numbers. Unknown fields are left for from_dict to reject."""
    from .portfolio import Portfolio
    from .strategy import Strategy
    d = dict(d)
    cls = Portfolio if (d.get("kind") == "allocation" or "tree" in d) else Strategy
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        has_default = f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        if (v is None or (isinstance(v, str) and not v.strip())) and has_default:
            del d[f.name]  # blank -> the default (e.g. position_size -> leverage / max_positions)
            continue
        t = str(f.type)
        numeric = ("float" in t or "int" in t) and "str" not in t
        if isinstance(v, str) and numeric:
            try:
                d[f.name] = int(v) if "float" not in t else float(v)
            except ValueError:
                raise ClientError(f"{f.name.replace('_', ' ')}: {v!r} is not a number")
        elif isinstance(v, float) and numeric and "float" not in t and v.is_integer():
            d[f.name] = int(v)
    return d


def share_token(spec, rf="tbill") -> str:
    """A URL-safe token that reproduces a run exactly: the full spec (settings included) + risk-free choice."""
    d = runner.to_dict(spec)
    d.pop("notes", None)
    if d.get("universe_name") == "NDX":
        d.pop("universe", None)  # rebuilt from the membership file on load
    raw = json.dumps({"v": 1, "spec": d, "rf": rf}, separators=(",", ":"), sort_keys=True, default=str).encode()
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode().rstrip("=")


def decode_share(token: str) -> dict:
    try:
        raw = zlib.decompress(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
        obj = json.loads(raw)
        if not (isinstance(obj, dict) and isinstance(obj.get("spec"), dict)):
            raise ValueError
        return obj
    except Exception:  # noqa: BLE001
        raise ClientError("This share link is damaged or incomplete (copy the whole link and try again).")


def _spec(body: dict):
    ov = _options(body)
    if body.get("share"):
        body = {**body, "spec": decode_share(str(body["share"]))["spec"]}
    if body.get("spec"):
        if not isinstance(body["spec"], dict):
            raise ClientError("The strategy JSON must be an object ({...}).")
        d = _coerce_spec_dict(body["spec"])
        if d.get("universe_name") == "NDX" and not d.get("universe") and "tree" not in d:
            d["universe"] = data.nasdaq100_ever()
        spec = runner.from_dict(d)
        for k, v in ov.items():
            if hasattr(spec, k):
                setattr(spec, k, v)
        if getattr(spec, "universe_name", None) == "NDX":
            spec.universe = data.nasdaq100_ever()
        spec.validate()  # fills defaults (e.g. position_size) before the summary uses them
        if not spec.description:
            if spec.__class__.__name__ == "Portfolio":
                from .portfolio import short_name
                spec.description = short_name(spec.tree)   # e.g. "If TQQQ RSI(10) > 79: UVXY, else TQQQ"
            else:
                lines = [ln.strip() for ln in spec.summary().splitlines() if ln.strip()]
                spec.description = lines[0] if lines else ""
    elif body.get("text"):
        spec = parser.parse(body["text"], **ov)
    else:
        raise ClientError("Describe a strategy first.")
    spec.validate()
    _probe_rules(spec)
    return spec


def _rules(spec) -> list[str]:
    if spec.__class__.__name__ == "Portfolio":
        out = []

        def walk(n):
            if not isinstance(n, dict):
                return
            if "if" in n:
                out.append(n["if"])
            if "filter" in n:
                out.extend(x for x in (n["filter"].get("by"), n["filter"].get("require")) if x)
            for k in ("then", "else", "fallback"):
                walk(n.get(k))
            for c in n.get("children") or []:
                walk(c)
        walk(spec.tree)
        return out
    return [r for r in (spec.entry, spec.short_entry, spec.exit_when, spec.entry_level, spec.rank_by) if r]


def _probe_rules(spec) -> None:
    """Evaluate every rule once on a short slice of real data, so unknown names and wrong arguments are
    reported when the strategy is interpreted (not halfway through a run). Also checks the tickers."""
    if spec.__class__.__name__ != "Portfolio" and not spec.universe_name:
        for t in spec.universe:
            data.load(t)  # DataError with "did you mean" suggestions
    if getattr(spec, "benchmark", None):
        from . import metrics
        metrics.check_benchmark(spec.benchmark)   # a ticker or a blend such as "60 SPY 40 AGG"
    try:
        df = data.load("SPY").iloc[-260:]
    except data.DataError:
        return
    import pandas as pd
    z = pd.Series(0.0, index=df.index)
    pos = {k: z for k in ("bars_held", "entry_price", "pnl", "highest_since_entry", "lowest_since_entry")}
    for rule in _rules(spec):
        if isinstance(rule, str):
            expr.evaluate_value(rule, expr.Namespace(df, extra=dict(pos), ticker="SPY"))


def run_key(spec, rf="tbill", sensitivity=True) -> str:
    """Identity of a run: the full spec and settings (as in its share link), the report options and the
    data version. Running the same thing again on the same data reuses the saved run."""
    stamp = str(data.data_status().get("updated_utc"))
    return hashlib.sha1(f"{share_token(spec, rf)}|{int(bool(sensitivity))}|{stamp}".encode()).hexdigest()


def _reusable(key: str) -> dict | None:
    row = next((r for r in _index() if r.get("key") == key), None)
    if row and (RUNS / row["id"] / "report.html").is_file():
        return row
    return None


def _new_id(label: str) -> str:
    h = hashlib.sha1(f"{label}{time.time()}".encode()).hexdigest()[:6]
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + h


def _summary_row(rid: str, A: dict, kind: str, label: str, spec, extra: dict | None = None, res=None,
                 rf="tbill") -> dict:
    st = A["stats"]
    start = st.get("start")  # the first day of the statistics: after any indicator warm-up, never the synthetic point
    row = {"id": rid, "created": datetime.now().isoformat(timespec="seconds"), "kind": kind, "label": label,
           "text": getattr(spec, "description", ""), "spec": runner.to_dict(spec) if spec is not None else None,
           "cagr": st.get("cagr"), "sharpe": st.get("sharpe"), "max_drawdown": st.get("max_drawdown"),
           "final": st.get("end_equity"), "start_value": st.get("start_equity"), "start": str(start), "end": str(st.get("end")),
           "trades": A["trade_stats"].get("trades", 0), **(extra or {})}
    if res is not None and res.kind == "allocation":
        row["rebalances"] = res.extras.get("rebalances")
    if spec is not None:
        row["share"] = share_token(spec, rf)
    return row


# ------------------------------------------------------------------ API handlers

def api_parse(body):
    spec = _spec(body)
    return {"kind": "allocation" if spec.__class__.__name__ == "Portfolio" else "signal",
            "interpretation": spec.summary(), "notes": spec.notes, "spec": runner.to_dict(spec)}


def api_run(body):
    rf = body.get("rf")
    if body.get("share") and rf in (None, ""):
        rf = decode_share(str(body["share"])).get("rf")
    rf = "tbill" if rf in (None, "", "tbill") else rf
    if rf != "tbill":
        try:
            rf = float(rf)
        except (TypeError, ValueError):
            raise ClientError(f"Bad risk-free rate {rf!r}")
    spec = _spec(body)
    sens = body.get("sensitivity", True) is not False
    key = run_key(spec, rf, sens)
    if not body.get("force"):
        old = _reusable(key)
        if old:  # the same spec and settings on the same data: open that run instead of saving a duplicate
            return {"id": old["id"], "url": f"/r/{old['id']}/report.html",
                    "files": sorted(p.name for p in (RUNS / old["id"]).iterdir()), "summary": old,
                    "interpretation": spec.summary(), "notes": (old.get("spec") or {}).get("notes") or spec.notes,
                    "spec": old.get("spec") or runner.to_dict(spec), "share": old.get("share"), "reused": True}
    res = runner.run(spec)
    A = report.analyze(res, rf=rf, sensitivity=sens)
    rid = _new_id(spec.description)
    out = RUNS / rid
    report.write_outputs(A, out)
    row = _summary_row(rid, report._clean(A), res.kind, spec.name or spec.description[:80], spec, res=res, rf=rf)
    row["key"] = key
    with LOCK:
        idx = _index()
        idx.insert(0, report._clean(row))
        _save_index(idx)
    return {"id": rid, "url": f"/r/{rid}/report.html", "files": sorted(p.name for p in out.iterdir()),
            "summary": report._clean(row), "interpretation": spec.summary(), "notes": spec.notes,
            "spec": runner.to_dict(spec), "share": row["share"]}


def api_share(body):
    """Decode a share token -> the spec and settings it carries (the page fills its form from these)."""
    sh = decode_share(str(body.get("share") or ""))
    spec = _spec({"share": body.get("share")})
    return {"spec": runner.to_dict(spec), "rf": sh.get("rf", "tbill"), "interpretation": spec.summary(),
            "kind": "allocation" if spec.__class__.__name__ == "Portfolio" else "signal"}


def api_orders(body):
    from . import orders
    try:
        value = float(str(body.get("account_value") or "").replace(",", "").replace("$", ""))
    except ValueError:
        raise ClientError("Enter the current account value as a number, e.g. 25000.")
    spec = _spec(body)
    return report._clean(orders.todays_orders(spec, value, str(body.get("holdings") or ""),
                                              whole_shares=body.get("whole_shares", True) is not False))


def _gallery_cache_file() -> Path:
    return RUNS / "gallery_stats.json"


def api_gallery():
    from .library import LIBRARY, TAGS
    f = _gallery_cache_file()
    cache = json.loads(f.read_text()) if f.exists() else {}
    stamp = str(data.data_status().get("updated_utc"))
    lib = []
    for x in LIBRARY:
        c = cache.get(x["id"]) or {}
        lib.append({**x, "stats": c.get("stats") if c.get("data") == stamp else None})
    runs = [r for r in _index() if r.get("kind") in ("signal", "allocation")]
    for r in runs:
        _readable_label(r)
    return {"library": lib, "runs": runs, "tags": list(TAGS)}


def _readable_label(row: dict) -> dict:
    """Older saved runs of unnamed Build-page portfolios are labelled with the first line of the interpretation
    ("Portfolio: if rsi(close, 10) > 79 (on TQQQ):"); show the readable default name instead."""
    lab = str(row.get("label") or "")
    spec = row.get("spec") or {}
    if lab.startswith("Portfolio: ") and isinstance(spec.get("tree"), dict) and not spec.get("name"):
        from .portfolio import short_name
        try:
            row["label"] = short_name(spec["tree"])
        except Exception:  # noqa: BLE001 - a label is cosmetic
            pass
    return row


def api_gallery_stats(body):
    """Headline stats for one library strategy (cached until the data is updated)."""
    from . import metrics
    from .library import entry_spec, find
    x = find(str(body.get("id") or body.get("text") or ""))
    if x is None:
        raise ClientError("Not a library strategy.")
    text = x["id"]
    spec = entry_spec(x)
    res = runner.run(spec)
    warm, _ = report.warmup(res)   # the same period as the report's statistics (after any indicator warm-up)
    tr = report.trim_result(res, warm)
    fb = tr.equity.index[1] if len(tr.equity) > 1 else None
    st = metrics.equity_stats(tr.equity, flows=tr.extras.get("flows"), first_bar=fb)
    stats = report._clean({"cagr": st.get("cagr"), "sharpe": st.get("sharpe"), "max_drawdown": st.get("max_drawdown"),
                           "start": str(st.get("start")),
                           "end": str(tr.equity.index[-1].date()), "share": share_token(spec)})
    with LOCK:
        f = _gallery_cache_file()
        cache = json.loads(f.read_text()) if f.exists() else {}
        cache[text] = {"data": str(data.data_status().get("updated_utc")), "stats": stats}
        RUNS.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(cache, default=str))
    return {"id": x["id"], "text": x.get("text"), "stats": stats}


# ------------------------------------------------------------------ community gallery (published strategies)

# shared and committed with the repo (BACKTESTER_COMMUNITY points it elsewhere, e.g. for tests)
COMMUNITY = Path(os.environ.get("BACKTESTER_COMMUNITY") or report.ROOT / "data" / "community.json")
COMMUNITY_MAX = 5000
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _community() -> list[dict]:
    try:
        rows = json.loads(COMMUNITY.read_text()) if COMMUNITY.exists() else []
    except (OSError, json.JSONDecodeError):
        return []
    return rows if isinstance(rows, list) else []


def _save_community(rows: list[dict]) -> None:
    COMMUNITY.parent.mkdir(parents=True, exist_ok=True)
    tmp = COMMUNITY.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=1, default=str))
    tmp.replace(COMMUNITY)


def _clean_text(v, limit: int, what: str, required: bool = False) -> str:
    s = _CTRL.sub("", str(v or "")).strip()
    if required and not s:
        raise ClientError(f"Give the strategy a {what}.")
    if len(s) > limit:
        raise ClientError(f"The {what} is too long ({len(s)} characters; at most {limit}).")
    return s


COMMUNITY_SORTS = {"cagr": ("cagr", True), "sharpe": ("sharpe", True), "max_drawdown": ("max_drawdown", True),
                   "newest": ("created", True), "name": ("name", False)}


def api_community(query: dict | None = None) -> dict:
    """Published strategies, optionally searched (q: every word must appear in the name, author, description,
    sentence or tickers) and sorted (cagr / sharpe: highest first; max_drawdown: the smallest drawdown first;
    newest; name)."""
    q = (query or {}).get("q", [""])[0].strip().lower() if query else ""
    sort = (query or {}).get("sort", ["newest"])[0] if query else "newest"
    if sort not in COMMUNITY_SORTS:
        raise ClientError(f"sort must be one of {', '.join(COMMUNITY_SORTS)}")
    rows = _community()
    if q:
        rows = [r for r in rows if all(w in str(r.get("hay", "")) for w in q.split())]
    key, desc = COMMUNITY_SORTS[sort]

    def k(r):
        v = r.get(key) if key in ("created", "name") else (r.get("stats") or {}).get(key)
        if key == "name":
            return str(v or "").lower()
        return (v is not None, v if v is not None else 0)

    rows = sorted(rows, key=k, reverse=desc)
    return {"strategies": [{x: v for x, v in r.items() if x != "hay"} for r in rows], "count": len(rows)}


def api_community_publish(body):
    """Publish a strategy (a sentence or a spec) to the community gallery with its name, author and description.
    It is backtested first (a saved run with the same spec and data is reused), so every entry carries its
    headline numbers and a report."""
    name = _clean_text(body.get("name"), 80, "name", required=True)
    author = _clean_text(body.get("author"), 60, "author name") or "anonymous"
    about = _clean_text(body.get("description"), 2000, "description")
    text = _clean_text(body.get("text"), 4000, "sentence")
    if not body.get("spec") and not text:
        raise ClientError("Nothing to publish: describe a strategy or build one first.")
    if body.get("spec") and len(json.dumps(body["spec"], default=str)) > 200_000:
        raise ClientError("This strategy is too large to publish.")
    run = api_run({"spec": body["spec"]} if body.get("spec") else {"text": text, "options": body.get("options") or {}})
    spec = dict(run["spec"])
    spec.pop("notes", None)
    if spec.get("universe_name") == "NDX":
        spec.pop("universe", None)
    spec["name"] = name
    s = run["summary"]
    tickers = []
    if isinstance(spec.get("tree"), dict):
        from .portfolio import tickers_in
        try:
            tickers = tickers_in(spec["tree"])[:60]
        except Exception:  # noqa: BLE001 - search words are a convenience
            tickers = []
    else:
        tickers = list(spec.get("universe") or [])[:50]
    entry = {"id": hashlib.sha1(f"{name}|{author}|{time.time()}".encode()).hexdigest()[:10],
             "created": datetime.now().isoformat(timespec="seconds"), "name": name, "author": author,
             "description": about, "text": text or None, "kind": s.get("kind"), "spec": spec,
             "stats": {k: s.get(k) for k in ("cagr", "sharpe", "max_drawdown", "start", "end")},
             "share": run.get("share"), "run_id": run.get("id")}
    entry["hay"] = " ".join([name, author, about, text, " ".join(tickers), str(s.get("kind") or "")]).lower()
    with LOCK:
        rows = _community()
        if len(rows) >= COMMUNITY_MAX:
            raise ClientError("The community gallery is full.")
        rows.insert(0, report._clean(entry))
        _save_community(rows)
    return {"entry": {k: v for k, v in report._clean(entry).items() if k != "hay"}, "url": run.get("url")}


def api_import_composer(body):
    """A Composer symphony (JSON text or object) -> a Portfolio spec for the block editor. Problems that
    don't stop the tree from loading (e.g. a ticker without data) come back in "problems"."""
    from . import composer_import
    src = body.get("json")
    if src in (None, "", {}):
        raise ClientError("Paste a Composer symphony's JSON (or choose the exported file).")
    d = composer_import.convert(src)
    problems = []
    interp = ""
    try:
        spec = runner.from_dict(dict(d))
        spec.validate()
        _probe_rules(spec)
        interp = spec.summary()
    except Exception as e:  # noqa: BLE001 - still load the tree, so the editor can mark what to fix
        problems.append(friendly_error(e))
    return {"spec": d, "interpretation": interp, "notes": d.get("notes", []), "problems": problems}


def api_compare(body):
    specs = []
    for rid in body.get("ids") or []:
        row = next((r for r in _index() if r["id"] == rid), None)
        if not row or not row.get("spec"):
            raise ClientError(f"Unknown run {rid}")
        s = runner.from_dict(row["spec"])
        s.name = row.get("label") or s.name
        specs.append(s)
    for t in body.get("texts") or []:
        if t.strip():
            specs.append(parser.parse(t, **_options(body)))
    if len(specs) < 2:
        raise ClientError("Pick at least two strategies to compare.")
    if len(specs) > 6:
        raise ClientError("Compare up to 6 strategies at a time.")
    analyses = []
    for i, s in enumerate(specs):
        if not s.name or s.name == s.description[:80]:
            s.name = f"Strategy {chr(65 + i)}"
        s.name = s.name[:40]
        analyses.append(report.analyze(runner.run(s), sensitivity=False))
    rid = _new_id("compare")
    report.write_outputs(analyses, RUNS / rid)
    C = report.common_window_stats(analyses)
    row = {"id": rid, "created": datetime.now().isoformat(timespec="seconds"), "kind": "compare",
           "label": " vs ".join(s.name for s in specs), "text": " | ".join(s.description for s in specs)}
    with LOCK:
        idx = _index()
        idx.insert(0, row)
        _save_index(idx)
    return {"id": rid, "url": f"/r/{rid}/report.html", "common": report._clean(C)}


def _method_list(v) -> list[str] | None:
    if not v:
        return None
    items = v if isinstance(v, list) else str(v).replace(";", ",").split(",")
    out = [str(x).strip() for x in items if str(x).strip()]
    return out or None


def api_research(body, kind):
    from . import research, research_report
    rid = _new_id(kind)
    out = RUNS / rid
    ov = _options(body)
    ov.pop("name", None)
    ov.pop("benchmark", None)
    if kind == "sweep":
        R = research.sweep(body["text"], body.get("objective", "sharpe"), ov)
        research_report.write_sweep(R, body["text"], out)
        label = "Sweep: " + body["text"]
    elif kind == "walkforward":
        start, end = ov.pop("start", None), ov.pop("end", None)
        R = research.walk_forward(body["text"], body.get("objective", "sharpe"), float(body.get("in_sample", 5)),
                                  float(body.get("out_sample", 1)), bool(body.get("anchored")), start, end, ov)
        research_report.write_walk(R, body["text"], out)
        label = "Walk-forward: " + body["text"]
    else:
        tickers = [t for t in (body.get("tickers") or "").replace(",", " ").split() if t]
        if len(tickers) < 2:
            raise ClientError("Give at least two tickers.")
        num = lambda k: float(body[k]) if body.get(k) not in (None, "") else None  # noqa: E731
        R = research.optimize(tickers, body.get("start") or None, body.get("end") or None,
                              float(body.get("max_weight") or 1), float(body.get("min_weight") or 0),
                              body.get("test_start") or None, constraints=body.get("constraints") or None,
                              target_return=num("target_return"), target_vol=num("target_vol"),
                              rolling_months=int(num("rolling_months")) if num("rolling_months") else None,
                              lookback_months=int(num("lookback_months") or 60),
                              rebalance=body.get("rebalance") or "quarterly",
                              methods=_method_list(body.get("methods")),
                              omega_threshold=num("omega_threshold") or 0.0,
                              expected_returns=body.get("expected_returns") or None,
                              expected_vols=body.get("expected_vols") or None,
                              correlations=body.get("correlations") or None, views=body.get("views") or None,
                              prior=body.get("prior") or None, tau=num("tau") or 0.05,
                              risk_aversion=num("risk_aversion") or 2.5, benchmark=body.get("benchmark") or None,
                              target_active=num("target_active"), resample=int(num("resample") or 0))
        research_report.write_optimize(R, out)
        label = "Optimise: " + " ".join(tickers)
    row = {"id": rid, "created": datetime.now().isoformat(timespec="seconds"), "kind": kind, "label": label[:120],
           "text": body.get("text") or body.get("tickers")}
    with LOCK:
        idx = _index()
        idx.insert(0, row)
        _save_index(idx)
    return {"id": rid, "url": f"/r/{rid}/report.html"}


def _target_spec(body):
    """A strategy/portfolio for the Monte Carlo and factor pages: a saved run id, a sentence or a spec."""
    rid = body.get("run_id")
    if rid:
        row = next((r for r in _index() if r["id"] == rid), None)
        if not row or not row.get("spec"):
            raise ClientError(f"Unknown saved run {rid}")
        return runner.from_dict(row["spec"])
    if body.get("spec"):
        return runner.from_dict(body["spec"])
    if (body.get("text") or "").strip():
        return parser.parse(body["text"])
    return None


def _weights(body) -> dict | None:
    w = body.get("weights")
    if not w:
        return None
    if isinstance(w, dict):
        return {data.canonical(k): float(v) for k, v in w.items()}
    from .montecarlo import parse_weights
    return parse_weights(str(w))


def api_montecarlo(body):
    from . import montecarlo as mc
    s = mc.Settings()
    try:
        s.start_balance = float(body.get("balance") or 1_000_000)
        s.years = int(body.get("years") or 30)
        s.sims = int(body.get("sims") or 5000)
        s.block_months = int(body.get("block_months") or 12)
        s.success_target = float(body.get("success_target") or 0.95)
        infl = body.get("inflation", "historical")
        s.inflation = "historical" if infl in (None, "", "historical") else float(infl)
        st = body.get("stress")
        s.stress = None if st in (None, "", "none") else str(st)
        s.stress_years = int(body.get("stress_years") or 10)
        if body.get("stress_shock") not in (None, ""):
            s.stress_shock = float(body["stress_shock"])
        s.horizon = str(body.get("horizon") or "fixed")
        if s.horizon == "mortality":
            if body.get("age") in (None, ""):
                raise ClientError("The life-table horizon needs the current age.")
            s.age = float(body["age"])
            s.sex = str(body.get("sex") or "male")
            if s.sex not in ("male", "female", "joint"):
                raise ClientError("sex must be male, female or joint")
            if body.get("age2") not in (None, ""):
                s.age2 = float(body["age2"])
        elif body.get("age") not in (None, "") and body.get("until_age") not in (None, ""):
            s.age, s.until_age = float(body["age"]), float(body["until_age"])
    except (TypeError, ValueError) as e:
        raise ClientError(f"Bad number: {e}")
    s.model = body.get("model") or "historical"
    s.rebalance = body.get("rebalance") or "yearly"
    s.start, s.end = body.get("start") or None, body.get("end") or None
    for t, v in (body.get("forecast") or {}).items():
        if v and v.get("ret") not in (None, "") and v.get("vol") not in (None, ""):
            s.forecast[data.canonical(t)] = (float(v["ret"]), float(v["vol"]))
    w = _weights(body)
    spec = None if w else _target_spec(body)
    if w:
        s.weights = w
    elif spec is not None:
        rb = s.rebalance
        s.weights, s.series_name = mc.settings_from_spec(spec, s)
        if body.get("rebalance"):
            s.rebalance = rb
    else:
        raise ClientError("Give tickers with weights, a sentence or a saved run.")
    flows = []
    for f in body.get("flows") or []:
        kind = f.get("type")
        amt = float(f.get("amount") or 0)
        if kind == "none" or (not amt and kind != "pct_withdrawal"):
            continue
        freq = f.get("freq") or "yearly"
        s_y, e_y = int(f.get("start_year") or 1), (int(f["end_year"]) if f.get("end_year") else None)
        if kind == "contribution":
            flows.append(mc.CashFlow(amount=abs(amt), freq=freq, inflation_adjusted=bool(f.get("inflation_adjusted", True)), start_year=s_y, end_year=e_y))
        elif kind == "withdrawal":
            flows.append(mc.CashFlow(amount=-abs(amt), freq=freq, inflation_adjusted=bool(f.get("inflation_adjusted", True)), start_year=s_y, end_year=e_y))
        elif kind == "pct_withdrawal":
            flows.append(mc.CashFlow(pct=-abs(float(f.get("pct") or 0)), freq=freq, start_year=s_y, end_year=e_y))
    if not flows and not body.get("flows") and spec is not None and hasattr(spec, "contribution"):
        flows = mc.flows_from_portfolio(spec)
    s.flows = flows
    R = mc.run(s)
    return report._clean(R)


def _factor_target(body):
    w = _weights(body)
    if w:
        return w
    if (body.get("ticker") or "").strip():
        return body["ticker"].strip()
    target = _target_spec(body)
    if target is None:
        raise ClientError("Give a ticker, tickers with weights, a sentence or a saved run.")
    return target


def api_factors(body):
    from . import factors as F
    r, name = F.returns_for(_factor_target(body))
    model = (body.get("model") or "ff3").strip()
    addons = "".join(ch for ch in str(body.get("addons") or "").replace(" ", "+").replace(",", "+") if ch.isalnum() or ch in "+_")
    if addons.strip("+") and model != "auto":
        model = model + "+" + "+".join(a for a in addons.split("+") if a)
    elif addons.strip("+"):
        model = F.auto_model(name) + "+" + "+".join(a for a in addons.split("+") if a)
    R = F.analyze(r, model, body.get("freq") or "monthly", body.get("start") or None,
                  body.get("end") or None, int(body.get("rolling_months") or 36), name=name)
    return report._clean(R)


def api_style(body):
    from . import factors as F
    from . import style as ST
    r, name = F.returns_for(_factor_target(body))
    raw = body.get("assets") or ""
    assets = [t for t in (raw if isinstance(raw, list) else str(raw).replace(",", " ").split()) if str(t).strip()] or None
    R = ST.analyze(r, assets, body.get("start") or None, body.get("end") or None,
                   int(body.get("window") or body.get("rolling_months") or 36), name=name)
    return report._clean(R)


def api_signals(body):
    from . import signals
    return report._clean(signals.scan(_spec(body)))


def api_paper(body, method):
    from . import signals
    if method == "POST":
        name = "".join(ch for ch in (body.get("name") or "") if ch.isalnum() or ch in "-_")[:40]
        if not name:
            raise ClientError("Give the paper strategy a name (letters, digits, - or _).")
        path = signals.paper_add(_spec(body), name)
        return {"saved": str(path.relative_to(report.ROOT))}
    return report._clean(signals.paper_report())


def last_bar_date() -> str | None:
    """The date of the latest daily bar (SPY, else the latest of a few broad ETFs): what "data to" means, rather
    than when the data was fetched."""
    best = None
    for t in ("SPY", "QQQ", "IWM", "TLT"):
        try:
            d = data.load(t).index[-1]
        except (FileNotFoundError, data.DataError, KeyError, IndexError):
            continue
        best = d if best is None else max(best, d)
        if t == "SPY":
            break
    return str(best.date()) if best is not None else None


def api_status(_body=None):
    m = data.universe_meta()
    st = data.data_status()
    st["last_bar"] = last_bar_date()
    return report._clean({"data": st, "examples": EXAMPLES, "nasdaq100": data.nasdaq100(),
                          "sims": [{"ticker": t, "about": data.SIMS.get(t, "simulated long history")} for t in data.sims()],
                          "etfs": data.etfs(), "indexes": m.get("indexes", []),
                          "former": m.get("former_members", []), "help": expr.HELP,
                          "tickers": data.available_tickers(), "factor_models": _factor_models()})


def _factor_models() -> list[dict]:
    from . import factors
    return factors.model_list()


def api_correlation(body):
    from . import correlation
    raw = body.get("tickers") or ""
    tickers = [t for t in (raw if isinstance(raw, list) else str(raw).replace(",", " ").split()) if str(t).strip()]
    pair = body.get("pair")
    if isinstance(pair, str):
        pair = pair.replace(",", " ").split()
    try:
        window = int(body["window"]) if body.get("window") not in (None, "") else None
    except (TypeError, ValueError):
        raise ClientError(f"Bad window {body.get('window')!r}: give a number of periods.")
    R = correlation.analyze(tickers, body.get("freq") or "monthly", window, body.get("start") or None,
                            body.get("end") or None, pair or None)
    return report._clean(R)


def api_fetch(body):
    t = data.canonical(str(body.get("ticker") or ""))
    if not t or len(t) > 12:
        raise ClientError("Give a ticker symbol.")
    if t in data.available_tickers():
        return {"ticker": t, "status": "already available"}
    if data.fetch_on_demand(t):
        data.load.cache_clear()
        data._nasdaq100_ever.cache_clear()
        return {"ticker": t, "status": "downloaded"}
    raise ClientError(f"Could not download {t} (unknown symbol, or no internet access from this machine). No data for {t}: "
                      "add it to data/extra_tickers.txt and run the 'Fetch price data' workflow (GitHub Actions), then pull.")


def api_delete(rid):
    import shutil
    with LOCK:
        idx = [r for r in _index() if r["id"] != rid]
        _save_index(idx)
    d = RUNS / rid
    if d.exists() and d.parent == RUNS:
        shutil.rmtree(d)
    return {"deleted": rid}


# ------------------------------------------------------------------ errors

_WORDS = sorted(set(re.findall(r"\b([a-z_]{2,})\(", expr.HELP))
                | set(re.findall(r"\b([a-z_]{3,})\b", expr.HELP.split("Functions")[0])))


def friendly_error(e: BaseException) -> str:
    """A readable message for anything a user's input can trigger (no raw Python exception names)."""
    msg = str(e).strip()
    if isinstance(e, KeyError):
        return f"Missing field {e.args[0]!r} in the strategy." if e.args else "A required field is missing."
    if isinstance(e, NameError):
        m = re.search(r"unknown name '([^']+)'", msg) or re.search(r"name '([^']+)' is not defined", msg)
        if m:
            w = m.group(1)
            close = difflib.get_close_matches(w, _WORDS, n=3, cutoff=0.5)
            return (f"The rule uses '{w}', which is not a known indicator or variable."
                    + (f" Did you mean {' or '.join(close)}?" if close else "")
                    + " See the rule language reference (Build or Data page).")
        return f"Unknown name in the rule: {msg}"
    if isinstance(e, SyntaxError):
        where = f" {e.text.strip()!r}" if getattr(e, "text", None) else ""
        return f"Couldn't read the rule{where}: check the brackets, commas and operators (e.g. rsi(close, 2) < 10)."
    if isinstance(e, TypeError):
        m = re.search(r"missing \d+ required (?:positional )?arguments?: (.+)$", msg)
        if m:
            return f"The strategy is missing required field(s): {m.group(1)}."
        m = re.search(r"(\w+)\(\) (takes|got|missing)(.*)", msg)
        if m:
            return f"Wrong arguments for {m.group(1)}(): {m.group(2)}{m.group(3)}. Check the rule language reference."
        return f"A value has the wrong type ({msg})."
    m = re.search(r"(?:invalid literal for \w+\(\) with base \d+|could not convert string to float): (.+)$", msg)
    if isinstance(e, ValueError) and m:
        return f"A number was expected, but got {m.group(1)}."
    if isinstance(e, ZeroDivisionError):
        return "A calculation divided by zero: check the numbers in the strategy (e.g. max positions, lookbacks)."
    if isinstance(e, (ClientError, parser.ParseError, data.DataError, ValueError)):
        return msg or "Invalid input."
    return (f"Couldn't run this strategy ({msg or type(e).__name__}). Try rephrasing it, "
            "or report it with the sentence you used.")


# ------------------------------------------------------------------ HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "Backtester/1.0"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, default=str).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 10_000_000:
            raise ClientError("Request too large")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            raise ClientError("Invalid JSON")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        path = u.path
        try:
            if path in ("/", "/index.html") or path.startswith("/app"):
                return self._send(200, APP.read_bytes(), "text/html; charset=utf-8")
            if path == "/favicon.ico":
                return self._send(200, FAVICON_ICO, "image/x-icon")
            if path == "/favicon.svg":
                return self._send(200, FAVICON_SVG, "image/svg+xml")
            if path == "/api/status":
                return self._json(200, api_status())
            if path == "/api/runs":
                return self._json(200, [_readable_label(r) for r in _index()])
            if path == "/api/library":
                from .library import LIBRARY
                return self._json(200, LIBRARY)
            if path == "/api/gallery":
                return self._json(200, api_gallery())
            if path == "/api/community":
                try:
                    return self._json(200, api_community(urllib.parse.parse_qs(u.query)))
                except ClientError as e:
                    return self._json(400, {"error": str(e)})
            if path == "/api/paper":
                return self._json(200, api_paper({}, "GET"))
            if path.startswith("/r/"):
                rel = urllib.parse.unquote(path[3:])
                f = (RUNS / rel).resolve()
                if RUNS.resolve() not in f.parents or not f.is_file():
                    return self._send(404, b"not found", "text/plain")
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                return self._send(200, f.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))
            return self._send(404, b"not found", "text/plain")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._json(500, {"error": f"Server error: {e}"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        path = u.path
        # ?soft=1 (live interpretation while typing): report input errors with 200 so the browser console stays quiet
        bad = 200 if urllib.parse.parse_qs(u.query).get("soft") == ["1"] else 400
        try:
            body = self._body()
            handlers = {
                "/api/parse": api_parse, "/api/run": api_run, "/api/compare": api_compare,
                "/api/sweep": lambda b: api_research(b, "sweep"), "/api/walkforward": lambda b: api_research(b, "walkforward"),
                "/api/optimize": lambda b: api_research(b, "optimize"), "/api/signals": api_signals,
                "/api/paper": lambda b: api_paper(b, "POST"),
                "/api/fetch": api_fetch, "/api/share": api_share, "/api/orders": api_orders,
                "/api/gallery/stats": api_gallery_stats, "/api/import/composer": api_import_composer,
                "/api/community/publish": api_community_publish,
                "/api/montecarlo": api_montecarlo, "/api/factors": api_factors, "/api/style": api_style, "/api/correlation": api_correlation,
            }
            if path in handlers:
                return self._json(200, handlers[path](body))
            return self._json(404, {"error": "unknown endpoint"})
        except (ClientError, parser.ParseError, ValueError, data.DataError, KeyError, NameError, SyntaxError,
                TypeError, ZeroDivisionError) as e:
            return self._json(bad, {"error": friendly_error(e)})
        except Exception as e:  # noqa: BLE001 - whatever a request triggers is reported readably, never as a 500
            traceback.print_exc()
            return self._json(bad, {"error": friendly_error(e)})

    def do_DELETE(self):
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/api/runs/"):
            return self._json(200, api_delete(path.rsplit("/", 1)[-1]))
        return self._json(404, {"error": "unknown endpoint"})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m backtester web")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share on your network")
    a = p.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Backtester site running on http://{a.host if a.host != '0.0.0.0' else 'localhost'}:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
