"""Logging initialisation.

Produces two sinks:

* a rotating human-readable console stream (optional), and
* a line-delimited JSON file under ``logging.dir`` for machine search.

Log directories are created on startup so a fresh checkout or a fresh Windows
machine never fails with "directory not found" mid-run.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONFIGURED = False

# Attributes present on every LogRecord; anything else was supplied by the
# caller via `extra=` and belongs in the structured payload.
_RESERVED = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Renders records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = _coerce(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )


def _coerce(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    return str(value)


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: Path | str = "logs",
    json_logs: bool = True,
    console: bool = True,
    filename: str = "platform.log",
    force: bool = False,
) -> Path:
    """Configure root logging. Idempotent unless ``force`` is set.

    Returns the path of the log file that was opened.
    """
    global _CONFIGURED

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / filename

    root = logging.getLogger()
    if _CONFIGURED and not force:
        return log_path
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter() if json_logs else ConsoleFormatter())
    root.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(ConsoleFormatter())
        root.addHandler(stream)

    # Third-party libraries are noisy at DEBUG; keep them at WARNING.
    for noisy in ("urllib3", "websockets", "alpaca", "streamlit", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
