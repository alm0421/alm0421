"""Risk decision types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.clock import utcnow
from app.execution.models import RejectionReason


@dataclass(frozen=True)
class RiskViolation:
    """One breached limit."""

    check: str
    reason: RejectionReason
    message: str
    #: The observed value and the limit it breached, for the dashboard.
    observed: Any = None
    limit: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "reason": self.reason.value,
            "message": self.message,
            "observed": self.observed,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class RiskDecision:
    """The outcome of a full pre-trade risk evaluation.

    ``approved`` is True only when *every* check passed. There is no partial
    approval and no override flag: a decision object cannot be constructed in a
    state where violations exist but the order is approved.
    """

    approved: bool
    violations: tuple[RiskViolation, ...] = ()
    #: Checks that passed, for the audit trail.
    checks_passed: tuple[str, ...] = ()
    evaluated_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if self.approved and self.violations:
            raise ValueError(
                "RiskDecision cannot be approved while violations exist. This "
                "would allow a breached limit to be overridden silently."
            )

    @classmethod
    def approve(cls, checks_passed: tuple[str, ...]) -> "RiskDecision":
        return cls(approved=True, violations=(), checks_passed=checks_passed)

    @classmethod
    def reject(
        cls, violations: tuple[RiskViolation, ...], checks_passed: tuple[str, ...] = ()
    ) -> "RiskDecision":
        if not violations:
            raise ValueError("A rejection must carry at least one violation.")
        return cls(approved=False, violations=violations, checks_passed=checks_passed)

    @property
    def primary_reason(self) -> RejectionReason | None:
        return self.violations[0].reason if self.violations else None

    @property
    def summary(self) -> str:
        if self.approved:
            return f"Approved ({len(self.checks_passed)} checks passed)"
        return "; ".join(v.message for v in self.violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "violations": [v.to_dict() for v in self.violations],
            "checks_passed": list(self.checks_passed),
            "evaluated_at": self.evaluated_at.isoformat(),
            "summary": self.summary,
        }
