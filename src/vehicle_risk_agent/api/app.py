"""FastAPI application factory with exception handlers and lifecycle management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vehicle_risk_agent.api.policy_routes import router as policy_router
from vehicle_risk_agent.api.routes import router as assessment_router
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.events.broadcaster import ProgressEventBroadcaster
from vehicle_risk_agent.retrieval.adapters import (
    EmbeddingAdapter,
    FakeEmbeddingAdapter,
    SentenceTransformersEmbeddingAdapter,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings()

    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    event_broadcaster = ProgressEventBroadcaster()
    embedding_adapter: EmbeddingAdapter
    if settings.environment.lower() == "production":
        embedding_adapter = SentenceTransformersEmbeddingAdapter()
    else:
        embedding_adapter = FakeEmbeddingAdapter()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await engine.dispose()

    app = FastAPI(
        title="Vehicle Risk Assessment Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.event_broadcaster = event_broadcaster
    app.state.embedding_adapter = embedding_adapter

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

    app.include_router(assessment_router)
    app.include_router(policy_router)

    return app
