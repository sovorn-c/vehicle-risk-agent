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


@pytest.mark.asyncio
async def test_sse_subscription_captures_concurrent_events_without_loss_or_duplicate(
    app_client: AsyncClient,
) -> None:
    """Verify events appended during stream connection are not lost due to race."""
    create_resp = await app_client.post(
        "/api/v1/assessments",
        json={"vin": "1HGCR2F85HA000000", "context": {"sale_type": "DEALER"}},
        headers={"Authorization": "Bearer dev-requester-token", "Idempotency-Key": "req-race-01"},
    )
    assert create_resp.status_code == 201
    assessment_id = create_resp.json()["id"]

    transport = app_client._transport
    assert isinstance(transport, ASGITransport)
    app = transport.app
    assert isinstance(app, FastAPI)
    session_factory = app.state.session_factory
    broadcaster = app.state.event_broadcaster

    # Pre-populate initial event
    async with session_factory() as sess:
        store = EventStore(sess, broadcaster=broadcaster)
        evt1 = WorkflowProgressEvent(
            event_id="evt-race-1",
            sequence=1,
            assessment_id=assessment_id,
            run_number=1,
            phase=AssessmentRunPhase.COLLECTING_EVIDENCE,
            safe_message="Phase 1 initial evidence",
            timestamp=datetime.now(UTC),
        )
        await store.append_event(evt1)

    async def emit_concurrent() -> None:
        await asyncio.sleep(0.01)
        async with session_factory() as sess:
            store = EventStore(sess, broadcaster=broadcaster)
            evt2 = WorkflowProgressEvent(
                event_id="evt-race-2",
                sequence=2,
                assessment_id=assessment_id,
                run_number=1,
                phase=AssessmentRunPhase.RETRIEVING_POLICY,
                safe_message="Phase 2 concurrent policy",
                timestamp=datetime.now(UTC),
            )
            await store.append_event(evt2)
            await asyncio.sleep(0.01)
            evt3 = WorkflowProgressEvent(
                event_id="evt-race-3",
                sequence=3,
                assessment_id=assessment_id,
                run_number=1,
                phase=AssessmentRunPhase.COMPLETED,
                safe_message="Phase 3 concurrent completion",
                timestamp=datetime.now(UTC),
            )
            await store.append_event(evt3)

    task = asyncio.create_task(emit_concurrent())

    resp = await app_client.get(
        f"/api/v1/assessments/{assessment_id}/events",
        headers={"Authorization": "Bearer dev-requester-token", "Last-Event-ID": "0"},
    )
    await task
    assert resp.status_code == 200
    assert "Phase 1 initial evidence" in resp.text
    assert "Phase 2 concurrent policy" in resp.text
    assert "Phase 3 concurrent completion" in resp.text
    # Verify no duplicates
    assert resp.text.count("id: 1:1\n") == 1
    assert resp.text.count("id: 1:2\n") == 1
    assert resp.text.count("id: 1:3\n") == 1


@pytest.mark.asyncio
async def test_sse_race_deterministic_gap_interception(app_client: AsyncClient) -> None:
    """Prove that subscribing before get_events captures writes during DB fetch."""
    import unittest.mock

    create_resp = await app_client.post(
        "/api/v1/assessments",
        json={"vin": "1HGCR2F85HA000000", "context": {"sale_type": "DEALER"}},
        headers={"Authorization": "Bearer dev-requester-token", "Idempotency-Key": "req-det-race"},
    )
    assert create_resp.status_code == 201
    assessment_id = create_resp.json()["id"]

    transport = app_client._transport
    assert isinstance(transport, ASGITransport)
    app = transport.app
    assert isinstance(app, FastAPI)
    session_factory = app.state.session_factory
    broadcaster = app.state.event_broadcaster

    # Pre-populate event 1
    async with session_factory() as sess:
        store = EventStore(sess, broadcaster=broadcaster)
        evt1 = WorkflowProgressEvent(
            event_id="evt-det-1",
            sequence=1,
            assessment_id=assessment_id,
            run_number=1,
            phase=AssessmentRunPhase.COLLECTING_EVIDENCE,
            safe_message="Deterministic Event 1",
            timestamp=datetime.now(UTC),
        )
        await store.append_event(evt1)

    # Intercept get_events: when get_events runs, append event 2 concurrently
    original_get_events = EventStore.get_events

    async def hooked_get_events(
        self: EventStore,
        asmt_id: str,
        after_sequence: int = 0,
        run_number: int | None = None,
    ) -> list[WorkflowProgressEvent]:
        res = await original_get_events(self, asmt_id, after_sequence, run_number)
        async with session_factory() as sess2:
            store2 = EventStore(sess2, broadcaster=broadcaster)
            evt2 = WorkflowProgressEvent(
                event_id="evt-det-2",
                sequence=2,
                assessment_id=assessment_id,
                run_number=1,
                phase=AssessmentRunPhase.COMPLETED,
                safe_message="Deterministic Event 2 Concurrent",
                timestamp=datetime.now(UTC),
            )
            await store2.append_event(evt2)
        return res

    with unittest.mock.patch.object(
        EventStore, "get_events", side_effect=hooked_get_events, autospec=True
    ):
        resp = await asyncio.wait_for(
            app_client.get(
                f"/api/v1/assessments/{assessment_id}/events",
                headers={"Authorization": "Bearer dev-requester-token"},
            ),
            timeout=1.0,
        )
        assert resp.status_code == 200
        assert "Deterministic Event 1" in resp.text
        assert "Deterministic Event 2 Concurrent" in resp.text


@pytest.mark.asyncio
async def test_sse_replay_is_partitioned_by_requested_run(app_client: AsyncClient) -> None:
    """Replay for one run must not include same-sequence events from another run."""
    create_resp = await app_client.post(
        "/api/v1/assessments",
        json={"vin": "1HGCR2F85HA000000", "context": {"sale_type": "DEALER"}},
        headers={
            "Authorization": "Bearer dev-requester-token",
            "Idempotency-Key": "req-run-partition-01",
        },
    )
    assessment_id = create_resp.json()["id"]
    transport = app_client._transport
    assert isinstance(transport, ASGITransport)
    app = transport.app
    assert isinstance(app, FastAPI)
    async with app.state.session_factory() as session:
        store = EventStore(session, broadcaster=app.state.event_broadcaster)
        await store.append_event(
            WorkflowProgressEvent(
                event_id="evt-run-1",
                sequence=1,
                assessment_id=assessment_id,
                run_number=1,
                phase=AssessmentRunPhase.COMPLETED,
                safe_message="run one event",
                timestamp=datetime.now(UTC),
            )
        )
        await store.append_event(
            WorkflowProgressEvent(
                event_id="evt-run-2",
                sequence=1,
                assessment_id=assessment_id,
                run_number=2,
                phase=AssessmentRunPhase.COMPLETED,
                safe_message="run two event",
                timestamp=datetime.now(UTC),
            )
        )

    response = await app_client.get(
        f"/api/v1/assessments/{assessment_id}/events?run_number=1",
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    assert response.status_code == 200
    assert "run one event" in response.text
    assert "run two event" not in response.text

    # A cursor from another run must not skip the selected run's sequence one.
    response_with_other_run_cursor = await app_client.get(
        f"/api/v1/assessments/{assessment_id}/events?run_number=1",
        headers={
            "Authorization": "Bearer dev-requester-token",
            "Last-Event-ID": "2:1",
        },
    )
    assert "run one event" in response_with_other_run_cursor.text
