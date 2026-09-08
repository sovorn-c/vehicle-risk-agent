"""Tests verifying redaction of credentials, prompts, raw observations, and stack traces."""

# story: e07s01

import io
import json
import logging

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from vehicle_risk_agent.domain.errors import IdempotencyConflictError
from vehicle_risk_agent.observability.failures import classify_safe_failure
from vehicle_risk_agent.observability.logging import JsonFormatter
from vehicle_risk_agent.observability.telemetry import (
    init_telemetry,
    record_boundary_metric,
    trace_boundary,
)


def test_logs_redact_sensitive_keys_and_canaries() -> None:
    """Log formatting must redact tokens, passwords, secrets, prompts, and raw observations."""
    canary_token = "canary-secret-token-xyz-123"
    canary_prompt = "canary-prompt-do-not-leak-assessment-prompt"
    canary_raw_obs = "canary-raw-evidence-payload-456"

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("test_redaction_logger")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info(
        "Boundary call with sensitive payload",
        extra={
            "authorization": f"Bearer {canary_token}",
            "password": canary_token,
            "prompt": canary_prompt,
            "raw_observation": canary_raw_obs,
            "nested": {"client_secret": canary_token, "safe_key": "safe_val"},
        },
    )

    output = stream.getvalue()
    data = json.loads(output)

    assert canary_token not in output
    assert canary_prompt not in output
    assert canary_raw_obs not in output
    assert data["authorization"] == "[REDACTED]"
    assert data["password"] == "[REDACTED]"
    assert data["prompt"] == "[REDACTED]"
    assert data["raw_observation"] == "[REDACTED]"
    assert data["nested"]["client_secret"] == "[REDACTED]"
    assert data["nested"]["safe_key"] == "safe_val"


def test_spans_redact_sensitive_attributes() -> None:
    """OpenTelemetry span attributes must not contain sensitive canaries."""
    canary_password = "canary-super-secret-password-789"
    canary_prompt = "canary-model-prompt-secret-context"

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    init_telemetry(tracer_provider=tracer_provider)

    with trace_boundary(
        "model.draft",
        boundary="model",
        password=canary_password,
        prompt=canary_prompt,
        token="secret-token-val",
    ):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes is not None

    for key, val in span.attributes.items():
        val_str = str(val)
        assert canary_password not in val_str
        assert canary_prompt not in val_str
        if key in ("password", "prompt", "token"):
            assert val == "[REDACTED]"


def test_exceptions_and_traces_do_not_leak_stack_traces_into_spans() -> None:
    """Span error description and attributes must use safe categories, not raw stack traces."""
    canary_db_secret = "canary-database-password-leak-attempt"

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    init_telemetry(tracer_provider=tracer_provider)

    with (
        pytest.raises(RuntimeError),
        trace_boundary("database.connect", boundary="database"),
    ):
        raise RuntimeError(f"Connection failed to db with password={canary_db_secret}")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes is not None

    for _k, val in span.attributes.items():
        assert canary_db_secret not in str(val)

    assert span.status.description is not None
    assert canary_db_secret not in span.status.description
    assert "Traceback" not in span.status.description


def test_metrics_do_not_leak_high_cardinality_or_sensitive_data() -> None:
    """Metric attributes must only contain safe categorical boundaries and outcomes."""
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    init_telemetry(meter_provider=meter_provider)

    record_boundary_metric(
        boundary="api",
        duration_ms=12.3,
        safe_outcome="SUCCESS",
    )

    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                for point in metric.data.data_points:
                    assert point.attributes is not None
                    for k in point.attributes:
                        assert k in ("boundary", "safe_outcome")


def test_safe_failure_classification_redacts_credentials_in_message() -> None:
    """Safe failure mapping always uses generic safe messages for arbitrary runtime errors."""
    canary = "super-secret-canary-never-reveal"
    exc = Exception(f"Internal memory corruption with secret={canary}")
    failure = classify_safe_failure(exc)

    assert canary not in failure.safe_message
    assert canary not in failure.safe_code
    assert failure.safe_message == "An internal processing failure occurred"


def test_logs_redact_canaries_in_message_and_safe_extras() -> None:
    """Log formatting must redact canaries from message and arbitrary safe extra keys."""
    canary = "canary-message-secret-abc-999"
    canary_extra = "canary-extra-secret-def-888"

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("test_message_redaction_logger")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info(
        f"Processing assessment with secret {canary}",
        extra={
            "arbitrary_safe_key": f"Value containing {canary_extra}",
            "custom_metadata": {"inner_note": f"Embedded {canary}"},
        },
    )

    output = stream.getvalue()
    data = json.loads(output)

    assert canary not in output
    assert canary_extra not in output
    assert canary not in data["message"]
    assert canary_extra not in data["arbitrary_safe_key"]
    assert canary not in data["custom_metadata"]["inner_note"]


def test_safe_failure_idempotency_conflict_redacts_canaries() -> None:
    """IdempotencyConflictError must not leak custom message or key details into safe failure."""
    canary_key = "canary-idempotency-key-xyz"
    canary_msg = "canary-sensitive-mismatch-detail"
    err = IdempotencyConflictError(key=canary_key, message=f"Conflict on {canary_msg}")

    failure = classify_safe_failure(err)

    assert failure.safe_code == "IDEMPOTENCY_CONFLICT"
    assert canary_key not in failure.safe_message
    assert canary_msg not in failure.safe_message
    assert failure.safe_message == "An idempotency conflict occurred for the specified operation"
