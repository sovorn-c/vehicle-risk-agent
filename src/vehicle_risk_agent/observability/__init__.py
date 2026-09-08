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
from vehicle_risk_agent.observability.telemetry import (
    TelemetryManager,
    get_meter,
    get_tracer,
    init_telemetry,
    instrument_app,
    record_boundary_metric,
    record_model_tokens,
    trace_boundary,
)

__all__ = [
    "JsonFormatter",
    "SafeFailure",
    "SafeFailureCategory",
    "TelemetryManager",
    "classify_safe_failure",
    "get_logger",
    "get_meter",
    "get_tracer",
    "init_telemetry",
    "instrument_app",
    "record_boundary_metric",
    "record_model_tokens",
    "set_assessment_context",
    "setup_logging",
    "trace_boundary",
]
