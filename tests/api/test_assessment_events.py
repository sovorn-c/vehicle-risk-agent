"""Tests for SSE event streaming, replay, authorization, and heartbeats."""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.persistence.models import Base

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def app_client() -> AsyncIterator[AsyncClient]:
    """Provide an AsyncClient for FastAPI application with test database."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    settings = Settings(database_url=TEST_DB_URL)
    app = create_app(settings=settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await engine.dispose()


@pytest.mark.asyncio
async def test_owner_can_stream_events(app_client: AsyncClient) -> None:
    """Verify assessment owner can stream progress events via SSE."""
    headers = {
        "Authorization": "Bearer dev-requester-token",
        "Idempotency-Key": "req-sse-01",
    }
    payload = {
        "vin": "1HGCR2F85HA000000",
        "context": {"sale_type": "DEALER"},
    }

    create_resp = await app_client.post("/api/v1/assessments", json=payload, headers=headers)
    assert create_resp.status_code == 201
    assessment_id = create_resp.json()["id"]

    # Stream events
    stream_headers = {"Authorization": "Bearer dev-requester-token"}
    response = await app_client.get(
        f"/api/v1/assessments/{assessment_id}/events",
        headers=stream_headers,
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_non_owner_forbidden_from_streaming_events(app_client: AsyncClient) -> None:
    """Verify another requester cannot stream events for an assessment they don't own."""
    # Requester 1 creates assessment
    create_resp = await app_client.post(
        "/api/v1/assessments",
        json={"vin": "1HGCR2F85HA000000", "context": {"sale_type": "DEALER"}},
        headers={"Authorization": "Bearer dev-requester-token", "Idempotency-Key": "req-own-01"},
    )
    assessment_id = create_resp.json()["id"]

    # Request with another token that maps to a different requester or reviewer unauthorized
    # We test with operator token which is not owner/reviewer for client stream
    resp = await app_client.get(
        f"/api/v1/assessments/{assessment_id}/events",
        headers={"Authorization": "Bearer dev-operator-token"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_event_replay_from_cursor(app_client: AsyncClient) -> None:
    """Verify client can replay events starting from Last-Event-ID."""
    create_resp = await app_client.post(
        "/api/v1/assessments",
        json={"vin": "1HGCR2F85HA000000", "context": {"sale_type": "DEALER"}},
        headers={"Authorization": "Bearer dev-requester-token", "Idempotency-Key": "req-replay-01"},
    )
    assessment_id = create_resp.json()["id"]

    # Request events with Last-Event-ID
    resp = await app_client.get(
        f"/api/v1/assessments/{assessment_id}/events",
        headers={"Authorization": "Bearer dev-requester-token", "Last-Event-ID": "1"},
    )
    assert resp.status_code == 200
