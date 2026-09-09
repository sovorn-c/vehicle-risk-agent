"""FastAPI application factory with exception handlers and lifecycle management."""

# story: e07s01 e07s03

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vehicle_risk_agent.adapters.mcp import create_mcp_adapter
from vehicle_risk_agent.api.docs import DOCS_HTML
from vehicle_risk_agent.api.evidence_routes import router as evidence_router
from vehicle_risk_agent.api.policy_routes import router as policy_router
from vehicle_risk_agent.api.review_routes import router as review_router
from vehicle_risk_agent.api.risk_policy_routes import router as risk_policy_router
from vehicle_risk_agent.api.routes import router as assessment_router
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import Assessment, AssessmentRunPhase
from vehicle_risk_agent.events.broadcaster import ProgressEventBroadcaster
from vehicle_risk_agent.observability.failures import classify_safe_failure
from vehicle_risk_agent.observability.logging import get_logger, setup_logging
from vehicle_risk_agent.observability.telemetry import init_telemetry, instrument_app
from vehicle_risk_agent.retrieval.adapters import (
    CrossEncoderRerankerAdapter,
    EmbeddingAdapter,
    FakeEmbeddingAdapter,
    FakeRerankerAdapter,
    RerankerAdapter,
    SentenceTransformersEmbeddingAdapter,
)
from vehicle_risk_agent.workflow.runner import AssessmentWorkflowRunner, build_initial_state

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings()

    setup_logging(level=settings.log_level)
    init_telemetry()

    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    event_broadcaster = ProgressEventBroadcaster()
    embedding_adapter: EmbeddingAdapter
    reranker_adapter: RerankerAdapter
    if settings.environment.lower() == "production":
        embedding_adapter = SentenceTransformersEmbeddingAdapter()
        reranker_adapter = CrossEncoderRerankerAdapter()
    else:
        embedding_adapter = FakeEmbeddingAdapter()
        reranker_adapter = FakeRerankerAdapter()

    mcp_adapter = create_mcp_adapter(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AssessmentWorkflowRunner.create(
            settings,
            session_factory=session_factory,
            broadcaster=event_broadcaster,
            mcp_adapter=mcp_adapter,
            embedding_adapter=embedding_adapter,
            reranker_adapter=reranker_adapter,
        ) as runner:
            app.state.workflow_runner = runner
            try:
                yield
            finally:
                pending_tasks = list(app.state.workflow_tasks)
                for task in pending_tasks:
                    task.cancel()
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                app.state.workflow_runner = None
        await engine.dispose()

    app = FastAPI(
        title="Vehicle Risk Assessment Agent",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    instrument_app(app)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.event_broadcaster = event_broadcaster
    app.state.embedding_adapter = embedding_adapter
    app.state.reranker_adapter = reranker_adapter
    app.state.mcp_adapter = mcp_adapter
    app.state.workflow_tasks = set()
    app.state.scheduled_workflow_runs = set()

    @app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
    async def documentation() -> HTMLResponse:
        return HTMLResponse(DOCS_HTML)

    @app.get("/reference", include_in_schema=False, response_class=HTMLResponse)
    async def documentation_reference() -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url=app.openapi_url or "/openapi.json",
            title="Vehicle Risk Assessment Agent · Full API reference",
        )

    def schedule_workflow_run(assessment: Assessment) -> bool:
        """Schedule one pending run and retain its task until completion."""
        runner = getattr(app.state, "workflow_runner", None)
        run = next(
            (item for item in assessment.runs if item.run_number == assessment.current_run_number),
            None,
        )
        if runner is None or run is None or run.phase != AssessmentRunPhase.PENDING:
            return False

        key = (assessment.id, run.run_number)
        if key in app.state.scheduled_workflow_runs:
            return False

        app.state.scheduled_workflow_runs.add(key)
        task = asyncio.create_task(
            runner.run(
                initial_state=build_initial_state(
                    assessment.id,
                    run.run_number,
                    assessment.vin,
                    assessment.context,
                )
            ),
            name=f"assessment-workflow:{assessment.id}:{run.run_number}",
        )
        app.state.workflow_tasks.add(task)

        def _task_finished(completed: asyncio.Task[Any]) -> None:
            app.state.workflow_tasks.discard(completed)
            app.state.scheduled_workflow_runs.discard(key)
            try:
                completed.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                failure = classify_safe_failure(exc)
                logger.error(
                    "Assessment workflow task failed",
                    extra={
                        "assessment_id": assessment.id,
                        "run_id": f"{assessment.id}:{run.run_number}",
                        "safe_outcome": failure.category.value,
                    },
                )

        task.add_done_callback(_task_finished)
        return True

    app.state.schedule_workflow_run = schedule_workflow_run

    @app.middleware("http")
    async def asgi_spec_version_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        if "asgi" in request.scope:
            request.scope["asgi"]["spec_version"] = "3.0"
        return await call_next(request)

    # Safe structured HTTPException handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "HTTP_ERROR", "message": str(exc.detail)}},
        )

    # General unexpected exception handler (fails closed, no internal stack leak)
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        if exc.__class__.__name__ in (
            "ClientDisconnect",
            "ClosedResourceError",
            "BrokenResourceError",
            "CancelledError",
        ):
            return JSONResponse(status_code=status.HTTP_200_OK, content={})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred",
                }
            },
        )

    @app.get("/health", tags=["Operational"])
    async def health_check() -> dict[str, str]:
        return {"status": "healthy", "service": "vehicle-risk-agent"}

    @app.get("/ready", tags=["Operational"])
    async def readiness_check() -> dict[str, str]:
        try:
            async with session_factory() as session:
                await session.execute(text("SELECT 1"))
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Database connectivity check failed",
                },
            ) from None
        return {"status": "ready", "service": "vehicle-risk-agent"}

    app.include_router(assessment_router)
    app.include_router(policy_router)
    app.include_router(evidence_router)
    app.include_router(risk_policy_router)
    app.include_router(review_router)

    return app
