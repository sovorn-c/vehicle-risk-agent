"""Structured JSON logging with correlation, assessment, and redaction support."""

import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

# Context variables for tracing across async workflows
_current_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_correlation_id", default=None
)
_current_assessment_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_assessment_id", default=None
)
_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_run_id", default=None
)
_current_node: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_node", default=None
)

SENSITIVE_KEY_SUBSTRINGS = (
    "authorization",
    "token",
    "password",
    "secret",
    "credential",
    "api_key",
    "apikey",
    "prompt",
    "raw_observation",
    "raw_evidence",
)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(sub in lowered for sub in SENSITIVE_KEY_SUBSTRINGS)


def sanitize_telemetry_value(key: str, value: Any) -> Any:
    """Sanitize sensitive keys or nested dict values."""
    if _is_sensitive_key(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {k: sanitize_telemetry_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_telemetry_value(key, item) for item in value]
    return value


@contextmanager
def set_assessment_context(
    correlation_id: str | None = None,
    assessment_id: str | None = None,
    run_id: str | None = None,
    node: str | None = None,
) -> Iterator[None]:
    """Context manager setting ambient contextvars for structured logging."""
    token_corr = _current_correlation_id.set(correlation_id) if correlation_id is not None else None
    token_asm = _current_assessment_id.set(assessment_id) if assessment_id is not None else None
    token_run = _current_run_id.set(run_id) if run_id is not None else None
    token_node = _current_node.set(node) if node is not None else None
    try:
        yield
    finally:
        if token_corr is not None:
            _current_correlation_id.reset(token_corr)
        if token_asm is not None:
            _current_assessment_id.reset(token_asm)
        if token_run is not None:
            _current_run_id.reset(token_run)
        if token_node is not None:
            _current_node.reset(token_node)


class JsonFormatter(logging.Formatter):
    """Standard-library JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Resolve context fields from record attributes or active contextvars
        correlation_id = getattr(record, "correlation_id", None) or _current_correlation_id.get()
        if correlation_id is not None:
            payload["correlation_id"] = correlation_id

        assessment_id = getattr(record, "assessment_id", None) or _current_assessment_id.get()
        if assessment_id is not None:
            payload["assessment_id"] = assessment_id

        run_id = getattr(record, "run_id", None) or _current_run_id.get()
        if run_id is not None:
            payload["run_id"] = run_id

        node = getattr(record, "node", None) or _current_node.get()
        if node is not None:
            payload["node"] = node

        safe_outcome = getattr(record, "safe_outcome", None)
        if safe_outcome is not None:
            payload["safe_outcome"] = safe_outcome

        elapsed_time_ms = getattr(record, "elapsed_time_ms", None)
        if elapsed_time_ms is not None:
            payload["elapsed_time_ms"] = elapsed_time_ms

        model = getattr(record, "model", None)
        if model is not None:
            payload["model"] = model

        token_count = getattr(record, "token_count", None)
        if token_count is not None:
            payload["token_count"] = token_count

        # Include sanitized extra attributes if present
        for key, val in record.__dict__.items():
            if key not in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "correlation_id",
                "assessment_id",
                "run_id",
                "node",
                "safe_outcome",
                "elapsed_time_ms",
                "model",
                "token_count",
            }:
                payload[key] = sanitize_telemetry_value(key, val)

        return json.dumps(payload, default=str)


def setup_logging(level: str | int = "INFO") -> None:
    """Configure root logging to use JsonFormatter on stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    log_level = getattr(logging, level.upper()) if isinstance(level, str) else level
    logging.basicConfig(level=log_level, handlers=[handler], force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger prefixed with the package namespace."""
    if name.startswith("vehicle_risk_agent"):
        return logging.getLogger(name)
    return logging.getLogger(f"vehicle_risk_agent.{name}")
