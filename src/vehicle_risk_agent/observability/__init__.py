"""Observability package."""

from vehicle_risk_agent.observability.failures import (
    SafeFailure,
    SafeFailureCategory,
    classify_safe_failure,
)
from vehicle_risk_agent.observability.logging import (
    JsonFormatter,
    get_logger,
    set_assessment_context,
    setup_logging,
)

__all__ = [
    "JsonFormatter",
    "SafeFailure",
    "SafeFailureCategory",
    "classify_safe_failure",
    "get_logger",
    "set_assessment_context",
    "setup_logging",
]
