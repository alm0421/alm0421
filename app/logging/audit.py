"""Append-only audit trail.

Every decision that could lead to an order - signal generated, risk verdict,
order intent, broker response, rejection - is appended here as a JSON line.
The file is never rewritten in place, so the trail is reconstructable.

Writes are append-mode with an explicit flush so a crash mid-session still
leaves a readable trail.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    event: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=False)


class AuditLog:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: Path | str, *, name: str = "audit") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._name = name

    @property
    def path(self) -> Path:
        return self._path

    def record(self, event: str, /, **payload: Any) -> AuditEvent:
        entry = AuditEvent(event=event, payload=payload)
        line = entry.to_json()
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return entry

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent entries, newest last.

        Malformed lines are skipped rather than raising - the audit trail must
        stay readable even if a write was truncated by a crash.
        """
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()[-limit:]
        entries: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
