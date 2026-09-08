"""Tests for mapping domain and dependency failures to stable safe categories."""
# story: e07s01

from vehicle_risk_agent.domain.errors import (
    IdempotencyConflictError,
)
from vehicle_risk_agent.observability.failures import (
    SafeFailureCategory,
    classify_safe_failure,
)


def test_classify_idempotency_conflict() -> None:
    """Verify IdempotencyConflictError maps to IDEMPOTENCY_CONFLICT category."""
    err = IdempotencyConflictError(key="key-1", message="Payload mismatch with key")
    failure = classify_safe_failure(err)
    assert failure.category == SafeFailureCategory.IDEMPOTENCY_CONFLICT
    assert failure.safe_code == "IDEMPOTENCY_CONFLICT"
    assert failure.safe_message == "An idempotency conflict occurred for the specified operation"
    # No stack trace or credentials in safe message
    assert "Traceback" not in failure.safe_message


def test_classify_dependency_network_error() -> None:
    """Verify external network/connection timeout maps to DEPENDENCY_UNAVAILABLE."""
    err = ConnectionRefusedError("Connection to upstream service refused: port 8000")
    failure = classify_safe_failure(err)
    assert failure.category == SafeFailureCategory.DEPENDENCY_UNAVAILABLE
    assert failure.safe_code == "DEPENDENCY_UNAVAILABLE"
    # Never leak internal port numbers or internal hosts
    assert "port 8000" not in failure.safe_message
    assert "An external dependency was temporarily unavailable" in failure.safe_message


def test_classify_unknown_runtime_error() -> None:
    """Verify arbitrary unexpected exception maps to INTERNAL_FAILURE without leaking details."""
    err = RuntimeError("Secret DB password 'p@ssword123' caused null dereference")
    failure = classify_safe_failure(err)
    assert failure.category == SafeFailureCategory.INTERNAL_FAILURE
    assert failure.safe_code == "INTERNAL_FAILURE"
    assert "p@ssword123" not in failure.safe_message
    assert "RuntimeError" not in failure.safe_message
    assert failure.safe_message == "An internal processing failure occurred"
