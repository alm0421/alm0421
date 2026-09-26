"""Scan every paper-trading strategy on the latest data; write signals/latest.{json,md}; optionally
POST the summary to ALERT_WEBHOOK_URL (Slack/Discord/any JSON webhook). Run by the Signals Action."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtester import report, signals  # noqa: E402


def main() -> None:
    rows = signals.paper_report()
    out = ROOT / "signals"
    out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps(report._clean(rows), indent=1, default=str))
    lines = ["# Daily signals", ""]
    for r in rows:
        lines.append(f"## {r['name']} (paper trading since {r['registered']})")
        lines.append(f"- forward return {report.pct(r.get('return'))}, max drawdown {report.pct(r.get('max_drawdown'))}, {r.get('days', 0)} days")
        lines.append(f"- today ({r['today']['as_of']}): {r['today'].get('action', '')}")
        for e in r["today"].get("entry_signals", [])[:30]:
            lines.append(f"  - {e['side']} {e['ticker']} @ {e['close']}")
        lines.append("")
    if not rows:
        lines.append("No paper strategies. Add one with `python -m backtester paper add \"...\" --name NAME` and commit paper/.")
    (out / "latest.md").write_text("\n".join(lines))
    print("\n".join(lines))
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if url and rows:
        ok = signals.post_webhook(url, rows)
        print("webhook:", "sent" if ok else "FAILED")


if __name__ == "__main__":
    main()
