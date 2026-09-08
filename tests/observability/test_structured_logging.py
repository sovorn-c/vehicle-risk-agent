"""Tests for structured JSON logging with correlation, assessment, node, outcome fields."""

import io
import json
import logging
from unittest.mock import patch

from vehicle_risk_agent.observability.logging import (
    JsonFormatter,
    get_logger,
    set_assessment_context,
    setup_logging,
)


def test_json_formatter_emits_valid_json_with_standard_fields() -> None:
    """Formatter must output valid JSON with timestamp, level, logger, and message."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Assessment created successfully",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    data = json.loads(output)

    assert data["level"] == "INFO"
    assert data["logger"] == "test_logger"
    assert data["message"] == "Assessment created successfully"
    assert "timestamp" in data


def test_structured_telemetry_fields_in_log_record() -> None:
    """Formatter must include correlation_id, assessment_id, run_id, node, outcome, duration."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="vehicle_risk_agent.workflow",
        level=logging.INFO,
        pathname="graph.py",
        lineno=42,
        msg="Node execution completed",
        args=(),
        exc_info=None,
    )
    record.correlation_id = "corr-12345"
    record.assessment_id = "asm-67890"
    record.run_id = "run-001"
    record.node = "evaluate_risk"
    record.safe_outcome = "SUCCESS"
    record.elapsed_time_ms = 45.2

    output = formatter.format(record)
    data = json.loads(output)

    assert data["correlation_id"] == "corr-12345"
    assert data["assessment_id"] == "asm-67890"
    assert data["run_id"] == "run-001"
    assert data["node"] == "evaluate_risk"
    assert data["safe_outcome"] == "SUCCESS"
    assert data["elapsed_time_ms"] == 45.2


def test_contextvars_propagation_to_records() -> None:
    """When context variables are set, logger records automatically inherit them."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("test_context_logger")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    with set_assessment_context(
        correlation_id="corr-ctx-99",
        assessment_id="asm-ctx-88",
        run_id="run-ctx-77",
        node="ingest_policy",
    ):
        logger.info("Processing step within context")

    output = stream.getvalue().strip()
    data = json.loads(output)

    assert data["correlation_id"] == "corr-ctx-99"
    assert data["assessment_id"] == "asm-ctx-88"
    assert data["run_id"] == "run-ctx-77"
    assert data["node"] == "ingest_policy"
    assert data["message"] == "Processing step within context"


def test_setup_logging_attaches_json_formatter() -> None:
    """setup_logging configures root/package handler with JsonFormatter."""
    with patch("logging.basicConfig") as mock_basic_config:
        setup_logging("DEBUG")
        assert mock_basic_config.called
        _, kwargs = mock_basic_config.call_args
        assert kwargs["level"] == logging.DEBUG
        assert len(kwargs["handlers"]) == 1
        assert isinstance(kwargs["handlers"][0].formatter, JsonFormatter)


def test_get_logger_returns_configured_logger() -> None:
    """get_logger returns a Logger instance prefixed with vehicle_risk_agent."""
    log = get_logger("workflow.runner")
    assert isinstance(log, logging.Logger)
    assert log.name == "vehicle_risk_agent.workflow.runner"
