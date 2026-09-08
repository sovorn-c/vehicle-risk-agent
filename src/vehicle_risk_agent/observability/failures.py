"""Safe failure categorization and redaction without leaking sensitive internals."""

# story: e07s01

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from vehicle_risk_agent.domain.errors import (
    IdempotencyConflictError,
)


class SafeFailureCategory(StrEnum):
    """Stable safe failure categories."""

    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


class SafeFailure(BaseModel):
    """Sanitized failure representation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: SafeFailureCategory
    safe_code: str
    safe_message: str


def classify_safe_failure(exc: Exception) -> SafeFailure:
    """Classify any exception into a safe failure category with sanitized message."""
    if isinstance(exc, IdempotencyConflictError):
        return SafeFailure(
            category=SafeFailureCategory.IDEMPOTENCY_CONFLICT,
            safe_code=SafeFailureCategory.IDEMPOTENCY_CONFLICT.value,
            safe_message="An idempotency conflict occurred for the specified operation",
        )

    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return SafeFailure(
            category=SafeFailureCategory.DEPENDENCY_UNAVAILABLE,
            safe_code=SafeFailureCategory.DEPENDENCY_UNAVAILABLE.value,
            safe_message="An external dependency was temporarily unavailable",
        )

    return SafeFailure(
        category=SafeFailureCategory.INTERNAL_FAILURE,
        safe_code=SafeFailureCategory.INTERNAL_FAILURE.value,
        safe_message="An internal processing failure occurred",
    )
