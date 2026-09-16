"""Structured logging and the immutable audit trail.

Note: this package is named ``app.logging`` but does not shadow the standard
library. Python 3 uses absolute imports, so ``import logging`` anywhere in this
project resolves to the stdlib module.
"""

from app.logging.setup import configure_logging, get_logger
from app.logging.audit import AuditLog, AuditEvent

__all__ = ["configure_logging", "get_logger", "AuditLog", "AuditEvent"]
