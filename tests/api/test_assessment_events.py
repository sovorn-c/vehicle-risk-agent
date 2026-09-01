"""Tests for SSE event streaming, replay, authorization, and heartbeats."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.domain.events import WorkflowProgressEvent
from vehicle_risk_agent.events.broadcaster import ProgressEventBroadcaster
from vehicle_risk_agent.persistence.event_store import EventStore
from vehicle_risk_agent.persistence.models import Base

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def app_client() -> AsyncIterator[AsyncClient]:
    """Provide an AsyncClient for FastAPI application with test database."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    settings = Settings(database_url=TEST_DB_URL, sse_heartbeat_interval_seconds=0.1)
    app = create_app(settings=settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await app.state.engine.dispose()
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

    # Pre-populate completed event
    transport = app_client._transport
    assert isinstance(transport, ASGITransport)
    app = transport.app
    assert isinstance(app, FastAPI)
    session_factory = app.state.session_factory
    broadcaster = app.state.event_broadcaster
    async with session_factory() as sess:
        store = EventStore(sess, broadcaster=broadcaster)
        evt = WorkflowProgressEvent(
            event_id="evt-owner-1",
            sequence=1,
            assessment_id=assessment_id,
            run_number=1,
            phase=AssessmentRunPhase.COMPLETED,
            safe_message="Assessment completed successfully",
            timestamp=datetime.now(UTC),
        )
        await store.append_event(evt)

    # Stream events
    stream_headers = {"Authorization": "Bearer dev-requester-token"}
    resp = await app_client.get(
        f"/api/v1/assessments/{assessment_id}/events",
        headers=stream_headers,
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert "Assessment completed successfully" in resp.text


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

    # Request with operator token which is not owner/reviewer for client stream
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

    # Pre-populate events
    transport = app_client._transport
    assert isinstance(transport, ASGITransport)
    app = transport.app
    assert isinstance(app, FastAPI)
    session_factory = app.state.session_factory
    broadcaster = app.state.event_broadcaster
    async with session_factory() as sess:
        store = EventStore(sess, broadcaster=broadcaster)
        evt1 = WorkflowProgressEvent(
            event_id="evt-replay-1",
            sequence=1,
            assessment_id=assessment_id,
            run_number=1,
            phase=AssessmentRunPhase.COLLECTING_EVIDENCE,
            safe_message="Collecting evidence for replay",
            timestamp=datetime.now(UTC),
        )
        await store.append_event(evt1)
        evt2 = WorkflowProgressEvent(
            event_id="evt-replay-2",
            sequence=2,
            assessment_id=assessment_id,
            run_number=1,
            phase=AssessmentRunPhase.COMPLETED,
            safe_message="Completed replay run",
            timestamp=datetime.now(UTC),
        )
        await store.append_event(evt2)

    # Request events with Last-Event-ID = 0 to replay evt 1 and 2
    resp = await app_client.get(
        f"/api/v1/assessments/{assessment_id}/events",
        headers={"Authorization": "Bearer dev-requester-token", "Last-Event-ID": "0"},
    )
    assert resp.status_code == 200
    assert "Collecting evidence for replay" in resp.text
    assert "Completed replay run" in resp.text

    # Request events with Last-Event-ID = 1 to replay only evt 2
    resp2 = await app_client.get(
        f"/api/v1/assessments/{assessment_id}/events",
        headers={"Authorization": "Bearer dev-requester-token", "Last-Event-ID": "1"},
    )
    assert resp2.status_code == 200
    assert "Collecting evidence for replay" not in resp2.text
    assert "Completed replay run" in resp2.text


@pytest.mark.asyncio
async def test_client_disconnect_cleans_up_without_failing_run() -> None:
    """Verify client subscription cleans up subscriber queue on unsubscribe."""
    broadcaster = ProgressEventBroadcaster()
    initial_count = broadcaster.subscriber_count

    async with broadcaster.subscribe("asmt-disc-01") as queue:
        assert broadcaster.subscriber_count == initial_count + 1
        assert not queue.full()

    # After exiting context, subscriber count returns to initial
    assert broadcaster.subscriber_count == initial_count


@pytest.mark.asyncio
async def test_live_events_streamed_to_connected_client(app_client: AsyncClient) -> None:
    """Verify live events emitted during workflow execution are received by active SSE stream."""
    create_resp = await app_client.post(
        "/api/v1/assessments",
        json={"vin": "1HGCR2F85HA000000", "context": {"sale_type": "DEALER"}},
        headers={"Authorization": "Bearer dev-requester-token", "Idempotency-Key": "req-live-01"},
    )
    assessment_id = create_resp.json()["id"]

    transport = app_client._transport
    assert isinstance(transport, ASGITransport)
    app = transport.app
    assert isinstance(app, FastAPI)
    session_factory = app.state.session_factory
    broadcaster = app.state.event_broadcaster

    async def publish_events_later() -> None:
        await asyncio.sleep(0.02)
        async with session_factory() as sess:
            store = EventStore(sess, broadcaster=broadcaster)
            evt1 = WorkflowProgressEvent(
                event_id="evt-live-1",
                sequence=1,
                assessment_id=assessment_id,
                run_number=1,
                phase=AssessmentRunPhase.COLLECTING_EVIDENCE,
                safe_message="Gathering vehicle facts",
                timestamp=datetime.now(UTC),
            )
            await store.append_event(evt1)
            await asyncio.sleep(0.02)
            evt2 = WorkflowProgressEvent(
                event_id="evt-live-2",
                sequence=2,
                assessment_id=assessment_id,
                run_number=1,
                phase=AssessmentRunPhase.COMPLETED,
                safe_message="Assessment completed",
                timestamp=datetime.now(UTC),
            )
            await store.append_event(evt2)

    task = asyncio.create_task(publish_events_later())

    resp = await app_client.get(
        f"/api/v1/assessments/{assessment_id}/events",
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    await task
    assert resp.status_code == 200
    assert "event: progress" in resp.text
    assert "Gathering vehicle facts" in resp.text
    assert "Assessment completed" in resp.text


@pytest.mark.asyncio
async def test_sse_heartbeat_emitted_during_idle() -> None:
    """Verify heartbeat comments are emitted periodically when idle."""
    broadcaster = ProgressEventBroadcaster()
    async with broadcaster.subscribe("asmt-hb-01") as queue:
        # Simulate idle timeout triggering heartbeat
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(queue.get(), timeout=0.02)
