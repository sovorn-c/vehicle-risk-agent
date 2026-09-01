import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress

import anyio
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send

from vehicle_risk_agent.api.deps import (
    get_current_principal,
    get_db_session,
    get_event_broadcaster,
    get_settings,
    intake_rate_limiter,
    require_role,
)
from vehicle_risk_agent.api.models import AssessmentCreateRequest
from vehicle_risk_agent.api.schemas import AssessmentResponse, AssessmentRunResponse
from vehicle_risk_agent.auth import Principal, Role
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.domain.errors import IdempotencyConflictError
from vehicle_risk_agent.events.broadcaster import ProgressEventBroadcaster
from vehicle_risk_agent.persistence.event_store import EventStore
from vehicle_risk_agent.persistence.repository import AssessmentRepository

router = APIRouter(prefix="/api/v1/assessments", tags=["assessments"])


@router.post("", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
async def create_assessment(
    request: AssessmentCreateRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    principal: Principal = Depends(require_role(Role.REQUESTER)),
    session: AsyncSession = Depends(get_db_session),
) -> AssessmentResponse:
    """Create a new Assessment and its initial Assessment Run.

    Enforces idempotency and intake rate limiting.
    """
    if not idempotency_key or not idempotency_key.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "MISSING_IDEMPOTENCY_KEY",
                "message": "Idempotency-Key header is required",
            },
        )

    # Check rate limit
    if not intake_rate_limiter.check(principal.principal_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Assessment intake rate limit exceeded",
            },
        )

    repo = AssessmentRepository(session)
    try:
        assessment = await repo.create_assessment(
            requester_id=principal.principal_id,
            idempotency_key=idempotency_key.strip(),
            request=request,
        )
    except IdempotencyConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "IDEMPOTENCY_CONFLICT",
                "message": str(err),
            },
        ) from err

    return AssessmentResponse(
        id=assessment.id,
        requester_id=assessment.requester_id,
        vin=assessment.vin,
        context=assessment.context,
        lifecycle_state=assessment.lifecycle_state,
        current_run_number=assessment.current_run_number,
        runs=[
            AssessmentRunResponse(
                id=r.id,
                run_number=r.run_number,
                phase=r.phase,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in assessment.runs
        ],
        created_at=assessment.created_at,
        updated_at=assessment.updated_at,
    )


@router.get("/{assessment_id}", response_model=AssessmentResponse)
async def get_assessment(
    assessment_id: str,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
) -> AssessmentResponse:
    """Retrieve an Assessment by ID, enforcing owner or reviewer authorization."""
    repo = AssessmentRepository(session)
    assessment = await repo.get_assessment(assessment_id)
    if assessment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Assessment {assessment_id} not found"},
        )

    # Requesters can only read their own assessments
    if principal.role == Role.REQUESTER and assessment.requester_id != principal.principal_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "Access to assessment is restricted to owner"},
        )
    if principal.role not in (Role.REQUESTER, Role.REVIEWER):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "Role not authorized to read assessment"},
        )

    return AssessmentResponse(
        id=assessment.id,
        requester_id=assessment.requester_id,
        vin=assessment.vin,
        context=assessment.context,
        lifecycle_state=assessment.lifecycle_state,
        current_run_number=assessment.current_run_number,
        runs=[
            AssessmentRunResponse(
                id=r.id,
                run_number=r.run_number,
                phase=r.phase,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in assessment.runs
        ],
        created_at=assessment.created_at,
        updated_at=assessment.updated_at,
    )


class SSEStreamingResponse(StreamingResponse):
    """StreamingResponse that cleanly streams response and handles client disconnects."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:  # noqa: ARG002
        with suppress(
            anyio.ClosedResourceError,
            anyio.BrokenResourceError,
            ClientDisconnect,
            asyncio.CancelledError,
        ):
            await self.stream_response(send)
        if self.background is not None:
            await self.background()


@router.get("/{assessment_id}/events")
async def stream_assessment_events(
    assessment_id: str,
    run_number: int | None = Query(default=None, ge=1),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    broadcaster: ProgressEventBroadcaster = Depends(get_event_broadcaster),
    settings: Settings = Depends(get_settings),
) -> SSEStreamingResponse:
    """Stream sanitized assessment progress events via SSE."""
    repo = AssessmentRepository(session)
    assessment = await repo.get_assessment(assessment_id)
    if assessment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Assessment {assessment_id} not found"},
        )

    if principal.role == Role.REQUESTER and assessment.requester_id != principal.principal_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "Access to assessment is restricted to owner"},
        )
    if principal.role not in (Role.REQUESTER, Role.REVIEWER):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "Role not authorized for event stream"},
        )

    event_store = EventStore(session, broadcaster=broadcaster)
    stream_run_number = run_number or assessment.current_run_number
    after_seq = 0
    if last_event_id:
        cursor_parts = last_event_id.split(":", maxsplit=1)
        if len(cursor_parts) == 2 and all(part.isdigit() for part in cursor_parts):
            cursor_run, cursor_sequence = (int(part) for part in cursor_parts)
            if cursor_run == stream_run_number:
                after_seq = cursor_sequence
        elif last_event_id.isdigit():
            # Accept legacy sequence-only cursors for the selected run.
            after_seq = int(last_event_id)

    async def event_generator() -> AsyncIterator[str]:
        last_seq = after_seq
        heartbeat_timeout = settings.sse_heartbeat_interval_seconds

        async with broadcaster.subscribe(assessment_id) as queue:
            # Replay historical events within subscription context to eliminate race window.
            events = await event_store.get_events(
                assessment_id,
                after_sequence=after_seq,
                run_number=stream_run_number,
            )
            for evt in events:
                payload = evt.model_dump_json()
                yield (
                    f"id: {stream_run_number}:{evt.sequence}\nevent: progress\ndata: {payload}\n\n"
                )
                last_seq = max(last_seq, evt.sequence)

            # Heartbeat comment to establish stream
            yield ": heartbeat\n\n"

            # If already terminal and past events cover terminal phase, terminate cleanly
            if events and events[-1].phase in (
                AssessmentRunPhase.COMPLETED,
                AssessmentRunPhase.FAILED,
            ):
                return

            while True:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=heartbeat_timeout)
                    if evt.run_number != stream_run_number:
                        continue
                    if evt.sequence > last_seq:
                        payload = evt.model_dump_json()
                        yield (
                            f"id: {stream_run_number}:{evt.sequence}\n"
                            f"event: progress\ndata: {payload}\n\n"
                        )
                        last_seq = evt.sequence
                        if evt.phase in (
                            AssessmentRunPhase.COMPLETED,
                            AssessmentRunPhase.FAILED,
                        ):
                            break
                except TimeoutError:
                    yield ": heartbeat\n\n"
                except (asyncio.CancelledError, GeneratorExit):
                    break

    return SSEStreamingResponse(event_generator(), media_type="text/event-stream")
