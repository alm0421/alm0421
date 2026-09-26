"""The backtesting site: a local web app (standard library only).

    python -m backtester web [--port 8000] [--host 127.0.0.1]

Pages: Backtest (plain English, live interpretation), Build (block editor for portfolios and
rule forms for signal strategies), Compare, Research (parameter sweep, walk-forward, portfolio
optimiser), Signals & paper trading, History (saved runs with share links) and Data.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import mimetypes
import threading
import time
import traceback
import urllib.parse
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


def _spec(body: dict):
    ov = _options(body)
    if body.get("spec"):
        spec = runner.from_dict(body["spec"])
        for k, v in ov.items():
            if hasattr(spec, k):
                setattr(spec, k, v)
        if getattr(spec, "universe_name", None) == "NDX":
            spec.universe = data.nasdaq100_ever()
        if not spec.description:
            spec.description = spec.summary().splitlines()[0]
    elif body.get("text"):
        spec = parser.parse(body["text"], **ov)
    else:
        raise ClientError("Describe a strategy first.")
    spec.validate()
    return spec


def _new_id(label: str) -> str:
    h = hashlib.sha1(f"{label}{time.time()}".encode()).hexdigest()[:6]
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + h


def _summary_row(rid: str, A: dict, kind: str, label: str, spec, extra: dict | None = None) -> dict:
    st = A["stats"]
    return {"id": rid, "created": datetime.now().isoformat(timespec="seconds"), "kind": kind, "label": label,
            "text": getattr(spec, "description", ""), "spec": runner.to_dict(spec) if spec is not None else None,
            "cagr": st.get("cagr"), "sharpe": st.get("sharpe"), "max_drawdown": st.get("max_drawdown"),
            "final": st.get("end_equity"), "start": str(st.get("start")), "end": str(st.get("end")),
            "trades": A["trade_stats"].get("trades", 0), **(extra or {})}


# ------------------------------------------------------------------ API handlers

def api_parse(body):
    spec = _spec(body)
    return {"kind": "allocation" if spec.__class__.__name__ == "Portfolio" else "signal",
            "interpretation": spec.summary(), "notes": spec.notes, "spec": runner.to_dict(spec)}


def api_run(body):
    spec = _spec(body)
    rf = body.get("rf") or "tbill"
    res = runner.run(spec)
    A = report.analyze(res, rf=rf if rf == "tbill" else float(rf), sensitivity=body.get("sensitivity", True))
    rid = _new_id(spec.description)
    out = RUNS / rid
    report.write_outputs(A, out)
    row = _summary_row(rid, report._clean(A), res.kind, spec.name or spec.description[:80], spec)
    with LOCK:
        idx = _index()
        idx.insert(0, report._clean(row))
        _save_index(idx)
    return {"id": rid, "url": f"/r/{rid}/report.html", "files": sorted(p.name for p in out.iterdir()),
            "summary": report._clean(row), "interpretation": spec.summary(), "notes": spec.notes}


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
        R = research.optimize(tickers, body.get("start") or None, body.get("end") or None,
                              float(body.get("max_weight") or 1), float(body.get("min_weight") or 0),
                              body.get("test_start") or None)
        research_report.write_optimize(R, out)
        label = "Optimise: " + " ".join(tickers)
    row = {"id": rid, "created": datetime.now().isoformat(timespec="seconds"), "kind": kind, "label": label[:120],
           "text": body.get("text") or body.get("tickers")}
    with LOCK:
        idx = _index()
        idx.insert(0, row)
        _save_index(idx)
    return {"id": rid, "url": f"/r/{rid}/report.html"}


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


def api_status(_body=None):
    m = data.universe_meta()
    return report._clean({"data": data.data_status(), "examples": EXAMPLES, "nasdaq100": data.nasdaq100(),
                          "etfs": data.etfs(), "indexes": m.get("indexes", []),
                          "former": m.get("former_members", []), "help": expr.HELP})


def api_fetch(body):
    t = data.canonical(str(body.get("ticker") or ""))
    if not t or len(t) > 12:
        raise ClientError("Give a ticker symbol.")
    if t in data.available_tickers():
        return {"ticker": t, "status": "already available"}
    if data.fetch_on_demand(t):
        data.load.cache_clear()
        return {"ticker": t, "status": "downloaded"}
    raise ClientError(f"Could not download {t} (unknown symbol, or no internet access from this machine).")


def api_delete(rid):
    import shutil
    with LOCK:
        idx = [r for r in _index() if r["id"] != rid]
        _save_index(idx)
    d = RUNS / rid
    if d.exists() and d.parent == RUNS:
        shutil.rmtree(d)
    return {"deleted": rid}


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
        if n > 2_000_000:
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
            if path == "/api/status":
                return self._json(200, api_status())
            if path == "/api/runs":
                return self._json(200, _index())
            if path == "/api/library":
                from .library import LIBRARY
                return self._json(200, LIBRARY)
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
        path = urllib.parse.urlparse(self.path).path
        try:
            body = self._body()
            handlers = {
                "/api/parse": api_parse, "/api/run": api_run, "/api/compare": api_compare,
                "/api/sweep": lambda b: api_research(b, "sweep"), "/api/walkforward": lambda b: api_research(b, "walkforward"),
                "/api/optimize": lambda b: api_research(b, "optimize"), "/api/signals": api_signals,
                "/api/paper": lambda b: api_paper(b, "POST"),
                "/api/fetch": api_fetch,
            }
            if path in handlers:
                return self._json(200, handlers[path](body))
            return self._json(404, {"error": "unknown endpoint"})
        except (ClientError, parser.ParseError, ValueError, data.DataError, KeyError) as e:
            msg = str(e)
            if isinstance(e, KeyError):
                msg = f"Missing field {e}"
            return self._json(400, {"error": msg})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._json(500, {"error": f"Something went wrong while running this ({type(e).__name__}: {e}). "
                                             "Try rephrasing, or report it with the sentence you used."})

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
