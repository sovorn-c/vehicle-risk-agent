"""OpenTelemetry tracing and metrics instrumentation across system boundaries."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.metrics import Counter, Histogram, Meter, MeterProvider
from opentelemetry.trace import Span, StatusCode, Tracer, TracerProvider

from vehicle_risk_agent.observability.failures import classify_safe_failure
from vehicle_risk_agent.observability.logging import sanitize_telemetry_value


class TelemetryManager:
    """Manages OpenTelemetry tracer and meter providers and instruments."""

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        service_name: str = "vehicle-risk-agent",
    ) -> None:
        self.service_name = service_name
        self.tracer_provider = tracer_provider or trace.get_tracer_provider()
        self.meter_provider = meter_provider or metrics.get_meter_provider()

        self.tracer: Tracer = self.tracer_provider.get_tracer(service_name)
        self.meter: Meter = self.meter_provider.get_meter(service_name)

        # Instruments
        self.boundary_calls_counter: Counter = self.meter.create_counter(
            name="boundary_calls_total",
            description="Total boundary invocations across API, graph, MCP, retrieval, etc.",
            unit="1",
        )
        self.boundary_duration_histogram: Histogram = self.meter.create_histogram(
            name="boundary_duration_ms",
            description="Boundary execution latency in milliseconds",
            unit="ms",
        )
        self.boundary_failures_counter: Counter = self.meter.create_counter(
            name="boundary_failures_total",
            description="Total safe failure outcomes by category and boundary",
            unit="1",
        )
        self.model_tokens_counter: Counter = self.meter.create_counter(
            name="model_tokens_total",
            description="Total tokens consumed by model operations",
            unit="1",
        )


_global_telemetry_manager: TelemetryManager | None = None


def init_telemetry(
    tracer_provider: TracerProvider | None = None,
    meter_provider: MeterProvider | None = None,
    service_name: str = "vehicle-risk-agent",
) -> TelemetryManager:
    """Initialize or reset the global telemetry manager."""
    global _global_telemetry_manager
    manager = TelemetryManager(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        service_name=service_name,
    )
    _global_telemetry_manager = manager
    return manager


def get_telemetry_manager() -> TelemetryManager:
    """Get the active telemetry manager or initialize default."""
    global _global_telemetry_manager
    if _global_telemetry_manager is None:
        _global_telemetry_manager = TelemetryManager()
    return _global_telemetry_manager


def get_tracer(name: str = "vehicle_risk_agent") -> Tracer:
    """Get a tracer instance from the telemetry manager."""
    manager = get_telemetry_manager()
    return manager.tracer_provider.get_tracer(name)


def get_meter(name: str = "vehicle_risk_agent") -> Meter:
    """Get a meter instance from the telemetry manager."""
    manager = get_telemetry_manager()
    return manager.meter_provider.get_meter(name)


def record_boundary_metric(
    boundary: str,
    duration_ms: float,
    safe_outcome: str = "SUCCESS",
) -> None:
    """Record metric counts and duration for a boundary execution."""
    manager = get_telemetry_manager()
    attributes = {"boundary": boundary, "safe_outcome": safe_outcome}
    manager.boundary_calls_counter.add(1, attributes)
    manager.boundary_duration_histogram.record(duration_ms, attributes)
    if safe_outcome != "SUCCESS":
        manager.boundary_failures_counter.add(1, attributes)


@contextmanager
def trace_boundary(
    name: str,
    boundary: str,
    **attributes: Any,
) -> Iterator[Span]:
    """Trace a boundary execution, recording timing, safe failures, and span attributes."""
    tracer = get_tracer(f"vehicle_risk_agent.{boundary}")
    start_time = time.perf_counter()

    with tracer.start_as_current_span(
        name,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        span.set_attribute("boundary", boundary)
        for key, val in attributes.items():
            if val is not None:
                sanitized_val = sanitize_telemetry_value(key, val)
                if isinstance(sanitized_val, (str, int, float, bool)):
                    span.set_attribute(key, sanitized_val)
                else:
                    span.set_attribute(key, str(sanitized_val))

        try:
            yield span
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            span.set_status(StatusCode.OK)
            record_boundary_metric(boundary, duration_ms, "SUCCESS")
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            failure = classify_safe_failure(exc)
            span.set_status(StatusCode.ERROR, description=failure.safe_message)
            span.set_attribute("safe_outcome", failure.category.value)
            record_boundary_metric(boundary, duration_ms, failure.category.value)
            raise


def instrument_app(app: FastAPI) -> None:
    """Instrument a FastAPI application with OpenTelemetry."""
    manager = get_telemetry_manager()
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=manager.tracer_provider,
        meter_provider=manager.meter_provider,
    )
