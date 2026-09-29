"""Local backtest API + web UI, on the Python standard library only.

    python scripts/serve_api.py            # then open http://localhost:8000
    python scripts/serve_api.py --data-dir backtests/data/2026-09-25 --port 8000

Routes
------
GET  /                      the backtesting UI (app/webui/index.html)
GET  /api/symbols           symbols with data in --data-dir
POST /api/backtest          run one backtest; JSON body -> JSON result
GET  /api/health            liveness

This never constructs a broker. It runs the same offline engine as the report
and the CLI, over the 1-minute CSVs in ``--data-dir`` (fetch them first with
``scripts/run_backtest.py --fetch-alpaca`` or point at an existing set).
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backtest.service import (  # noqa: E402
    BacktestError,
    BacktestRequest,
    available_symbols,
    run_backtest,
)

ROOT = Path(__file__).resolve().parents[1]
UI_FILE = ROOT / "app" / "webui" / "index.html"

_ALLOWED = {
    "symbols", "mode", "alerts", "capital", "risk_pct", "max_leverage", "slippage",
    "min_consol_bars", "max_consol_bars", "require_location", "require_drive",
    "wave_pause_bars", "max_hold_minutes", "alert_window_minutes", "market_entry",
}


def make_handler(data_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HitchHikerBacktest/1.0"

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def log_message(self, *args) -> None:  # quieter console
            pass

        def do_OPTIONS(self) -> None:
            self._send(204, b"", "text/plain")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                if not UI_FILE.exists():
                    self._json(500, {"error": f"UI not found at {UI_FILE}"})
                    return
                self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            elif path == "/api/health":
                self._json(200, {"ok": True, "data_dir": str(data_dir)})
            elif path == "/api/symbols":
                self._json(200, {"symbols": available_symbols(data_dir)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/api/backtest":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "invalid JSON body"})
                return
            kwargs = {k: v for k, v in payload.items() if k in _ALLOWED}
            kwargs.setdefault("symbols", [])
            try:
                result = run_backtest(BacktestRequest(data_dir=data_dir, **kwargs))
            except BacktestError as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - surface the message to the UI
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            else:
                self._json(200, result)

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description="Serve the HitchHiker backtest UI + API.")
    p.add_argument("--data-dir", type=Path, default=ROOT / "backtests" / "data" / "2026-09-25")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    syms = available_symbols(args.data_dir)
    print(f"Backtest data dir : {args.data_dir}  ({len(syms)} symbols: {', '.join(syms) or 'none'})")
    print(f"Serving UI + API  : http://{args.host}:{args.port}")
    if not syms:
        print("  ! No *_1min.csv found there. Fetch data first, e.g.:")
        print("    python scripts/run_backtest.py --scan --symbols SPY QQQ --fetch-alpaca \\")
        print("      --start 2024-01-01 --end 2026-09-25 --data-dir", args.data_dir, "--out /tmp/bt")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.data_dir))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
