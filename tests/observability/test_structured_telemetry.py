"""Tests for OpenTelemetry spans and metrics across system boundaries."""

# story: e07s01

from typing import cast

import pytest
from httpx import ASGITransport, AsyncClient
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, NumberDataPoint
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.observability.telemetry import (
    TelemetryManager,
    get_tracer,
    init_telemetry,
    record_boundary_metric,
    record_model_tokens,
    trace_boundary,
)


@pytest.fixture
def memory_telemetry() -> tuple[InMemorySpanExporter, InMemoryMetricReader, TelemetryManager]:
    span_exporter = InMemorySpanExporter()
    metric_reader = InMemoryMetricReader()

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    meter_provider = MeterProvider(metric_readers=[metric_reader])

    manager = init_telemetry(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    return span_exporter, metric_reader, manager


def test_tracer_and_meter_initialization(
    memory_telemetry: tuple[InMemorySpanExporter, InMemoryMetricReader, TelemetryManager],
) -> None:
    """Verify tracer and meter instances are properly initialized and accessible."""
    _exporter, _reader, manager = memory_telemetry
    assert manager is not None
    tracer = get_tracer("api")
    assert tracer is not None


def test_trace_boundary_sync_records_span(
    memory_telemetry: tuple[InMemorySpanExporter, InMemoryMetricReader, TelemetryManager],
) -> None:
    """Synchronous boundary execution creates span with correlation and assessment attributes."""
    exporter, _reader, _manager = memory_telemetry

    with trace_boundary(
        "mcp.call_tool",
        boundary="mcp",
        correlation_id="corr-mcp-1",
        assessment_id="asm-mcp-1",
        tool_name="fetch_vehicle_history",
    ) as span:
        span.set_attribute("safe_outcome", "SUCCESS")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.attributes is not None
    assert s.name == "mcp.call_tool"
    assert s.attributes["boundary"] == "mcp"
    assert s.attributes["correlation_id"] == "corr-mcp-1"
    assert s.attributes["assessment_id"] == "asm-mcp-1"
    assert s.attributes["tool_name"] == "fetch_vehicle_history"
    assert s.attributes["safe_outcome"] == "SUCCESS"


@pytest.mark.asyncio
async def test_trace_boundary_async_records_span(
    memory_telemetry: tuple[InMemorySpanExporter, InMemoryMetricReader, TelemetryManager],
) -> None:
    """Asynchronous boundary execution traces workflow and node execution."""
    exporter, _reader, _manager = memory_telemetry

    async def sample_graph_node() -> str:
        with trace_boundary(
            "workflow.node",
            boundary="graph",
            correlation_id="corr-async-1",
            assessment_id="asm-async-1",
            node="assess_risk",
        ):
            return "completed"

    result = await sample_graph_node()
    assert result == "completed"

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s0 = spans[0]
    assert s0.attributes is not None
    assert s0.name == "workflow.node"
    assert s0.attributes["boundary"] == "graph"
    assert s0.attributes["node"] == "assess_risk"


def test_trace_all_required_boundaries(
    memory_telemetry: tuple[InMemorySpanExporter, InMemoryMetricReader, TelemetryManager],
) -> None:
    """Confirm spans can be emitted for all required boundaries."""
    exporter, _reader, _manager = memory_telemetry
    boundaries = ["api", "graph", "mcp", "retrieval", "model", "database", "review"]

    for b in boundaries:
        with trace_boundary(f"{b}.operation", boundary=b, assessment_id="asm-all"):
            pass

    spans = exporter.get_finished_spans()
    emitted_boundaries = {
        s.attributes["boundary"]
        for s in spans
        if s.attributes is not None and "boundary" in s.attributes
    }
    assert emitted_boundaries == set(boundaries)


def test_trace_boundary_handles_safe_failure(
    memory_telemetry: tuple[InMemorySpanExporter, InMemoryMetricReader, TelemetryManager],
) -> None:
    """Span records safe error status when an exception occurs."""
    exporter, _reader, _manager = memory_telemetry

    with (
        pytest.raises(ValueError, match="Invalid payload"),
        trace_boundary("database.query", boundary="database", query_type="select"),
    ):
        raise ValueError("Invalid payload")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.status.status_code == trace.StatusCode.ERROR
    assert s.attributes is not None
    assert s.attributes["safe_outcome"] == "INTERNAL_FAILURE"


def test_model_token_metrics_record_input_and_output_totals(
    memory_telemetry: tuple[InMemorySpanExporter, InMemoryMetricReader, TelemetryManager],
) -> None:
    """Model operations must record token totals without recording prompt content."""
    _exporter, reader, _manager = memory_telemetry

    record_model_tokens(input_tokens=12, output_tokens=8, model="offline")

    data = reader.get_metrics_data()
    assert data is not None
    metrics_by_name = {
        metric.name: metric
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }
    token_metric = metrics_by_name["model_tokens_total"]
    point = cast(NumberDataPoint, token_metric.data.data_points[0])
    assert point.value == 20


def test_metrics_recording(
    memory_telemetry: tuple[InMemorySpanExporter, InMemoryMetricReader, TelemetryManager],
) -> None:
    """Verify metrics recording for duration and outcome."""
    _exporter, reader, _manager = memory_telemetry

    record_boundary_metric(
        boundary="retrieval",
        duration_ms=25.5,
        safe_outcome="SUCCESS",
    )

    data = reader.get_metrics_data()
    assert data is not None
    resource_metrics = data.resource_metrics
    assert len(resource_metrics) > 0


@pytest.mark.asyncio
async def test_app_telemetry_wiring_and_readiness(
    memory_telemetry: tuple[InMemorySpanExporter, InMemoryMetricReader, TelemetryManager],
) -> None:
    """create_app must configure telemetry, instrument app, and provide conditional /ready."""
    _exporter, _reader, _manager = memory_telemetry
    settings = Settings(
        database_url="postgresql+psycopg://postgres:postgres@localhost:54329/postgres"
    )
    app = create_app(settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    # Test unreachable DB returns 503 Service Unavailable
    bad_settings = Settings(
        database_url="postgresql+psycopg://postgres:postgres@localhost:54320/nonexistent"
    )
    bad_app = create_app(bad_settings)
    bad_transport = ASGITransport(app=bad_app)
    async with AsyncClient(transport=bad_transport, base_url="http://test") as bad_client:
        bad_resp = await bad_client.get("/ready")
        assert bad_resp.status_code == 503
        assert bad_resp.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_workflow_runner_emits_trace_boundary_span(
    memory_telemetry: tuple[InMemorySpanExporter, InMemoryMetricReader, TelemetryManager],
) -> None:
    """AssessmentRunner execution must emit a workflow.execute span via trace_boundary."""
    exporter, _reader, _manager = memory_telemetry
    from vehicle_risk_agent.workflow.runner import AssessmentRunner

    runner = AssessmentRunner()
    _res = await runner.run(
        assessment_id="asmt-trace-1",
        run_number=1,
        vin="7AT0BJ03020000001",
        context=AssessmentContext(sale_type=SaleType.DEALER),
    )

    spans = exporter.get_finished_spans()
    workflow_spans = [s for s in spans if s.name == "workflow.execute"]
    assert len(workflow_spans) >= 1
    assert workflow_spans[0].attributes is not None
    assert workflow_spans[0].attributes["boundary"] == "workflow"
    assert workflow_spans[0].attributes["assessment_id"] == "asmt-trace-1"
