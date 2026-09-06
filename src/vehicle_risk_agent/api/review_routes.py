"""API route handlers for Reviewer report decisions and approval actions."""

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.api.deps import get_db_session, require_role
from vehicle_risk_agent.api.review_schemas import (
    ApproveReportRequest,
    RejectReportRequest,
    RequestReinvestigationRequest,
    ReviewDecisionResponse,
)
from vehicle_risk_agent.auth import Principal, Role
from vehicle_risk_agent.domain.errors import IdempotencyConflictError
from vehicle_risk_agent.review.errors import (
    AssessmentNotFoundError,
    AssessmentNotReviewableError,
    DraftNotFoundError,
    ReinvestigationLimitReachedError,
    ReviewActionConflictError,
)
from vehicle_risk_agent.review.models import (
    ApproveReportCommand,
    RejectReportCommand,
    RequestReinvestigationCommand,
)
from vehicle_risk_agent.review.service import ReviewDecisionResult, ReviewDecisionService

router = APIRouter(prefix="/api/v1/assessments", tags=["review"])


def _to_response(res: ReviewDecisionResult) -> ReviewDecisionResponse:
    """Map ReviewDecisionResult to public API response schema."""
    return ReviewDecisionResponse(
        action_id=res.action.id,
        assessment_id=res.assessment_id,
        run_number=res.action.run_number,
        reviewer_id=res.action.reviewer_id,
        action_type=res.action.action_type.value,
        disposition=res.disposition.value,
        assessment_state=res.assessment_state.value,
        notes=res.action.notes,
        rationale=res.action.rationale,
        created_at=res.action.created_at,
        action_hash=res.action.action_hash,
        released_report_id=res.released_report.id if res.released_report else None,
        next_run_number=res.next_run_number,
        questions=list(res.action.questions),
        evidence_targets=list(res.action.evidence_targets),
    )


async def _handle_approve(
    assessment_id: str,
    request: ApproveReportRequest,
    idempotency_key: str | None,
    principal: Principal,
    session: AsyncSession,
) -> ReviewDecisionResponse:
    if not idempotency_key or not idempotency_key.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "MISSING_IDEMPOTENCY_KEY",
                "message": "Idempotency-Key header is required",
            },
        )

    try:
        cmd = ApproveReportCommand(
            assessment_id=assessment_id,
            run_number=request.run_number,
            reviewer_id=principal.principal_id,
            idempotency_key=idempotency_key.strip(),
            notes=request.notes,
            acknowledge_missing_evidence=request.acknowledge_missing_evidence,
            rationale=request.rationale,
            draft_outcome=request.draft_outcome,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR", "message": str(e)},
        ) from e

    service = ReviewDecisionService(session)
    try:
        result = await service.record_review_action(cmd)
    except IdempotencyConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "IDEMPOTENCY_CONFLICT", "message": str(e)},
        ) from e
    except ReviewActionConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "REVIEW_ACTION_CONFLICT", "message": str(e)},
        ) from e
    except AssessmentNotReviewableError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "ASSESSMENT_NOT_REVIEWABLE", "message": str(e)},
        ) from e
    except AssessmentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ASSESSMENT_NOT_FOUND", "message": str(e)},
        ) from e
    except DraftNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DRAFT_NOT_FOUND", "message": str(e)},
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR", "message": str(e)},
        ) from e

    return _to_response(result)


async def _handle_reject(
    assessment_id: str,
    request: RejectReportRequest,
    idempotency_key: str | None,
    principal: Principal,
    session: AsyncSession,
) -> ReviewDecisionResponse:
    if not idempotency_key or not idempotency_key.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "MISSING_IDEMPOTENCY_KEY",
                "message": "Idempotency-Key header is required",
            },
        )

    try:
        cmd = RejectReportCommand(
            assessment_id=assessment_id,
            run_number=request.run_number,
            reviewer_id=principal.principal_id,
            idempotency_key=idempotency_key.strip(),
            rationale=request.rationale,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR", "message": str(e)},
        ) from e

    service = ReviewDecisionService(session)
    try:
        result = await service.record_review_action(cmd)
    except IdempotencyConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "IDEMPOTENCY_CONFLICT", "message": str(e)},
        ) from e
    except ReviewActionConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "REVIEW_ACTION_CONFLICT", "message": str(e)},
        ) from e
    except AssessmentNotReviewableError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "ASSESSMENT_NOT_REVIEWABLE", "message": str(e)},
        ) from e
    except AssessmentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ASSESSMENT_NOT_FOUND", "message": str(e)},
        ) from e
    except DraftNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DRAFT_NOT_FOUND", "message": str(e)},
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR", "message": str(e)},
        ) from e

    return _to_response(result)


@router.post(
    "/{assessment_id}/review/approve",
    response_model=ReviewDecisionResponse,
    status_code=status.HTTP_200_OK,
)
@router.post(
    "/{assessment_id}/reviews/approve",
    response_model=ReviewDecisionResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def approve_report(
    assessment_id: str,
    request: ApproveReportRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    principal: Principal = Depends(require_role(Role.REVIEWER)),
    session: AsyncSession = Depends(get_db_session),
) -> ReviewDecisionResponse:
    """Reviewer approves a Report Draft and derives Released Report."""
    return await _handle_approve(assessment_id, request, idempotency_key, principal, session)


@router.post(
    "/{assessment_id}/review/reject",
    response_model=ReviewDecisionResponse,
    status_code=status.HTTP_200_OK,
)
@router.post(
    "/{assessment_id}/reviews/reject",
    response_model=ReviewDecisionResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def reject_report(
    assessment_id: str,
    request: RejectReportRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    principal: Principal = Depends(require_role(Role.REVIEWER)),
    session: AsyncSession = Depends(get_db_session),
) -> ReviewDecisionResponse:
    """Reviewer rejects a Report Draft with bounded mandatory rationale."""
    return await _handle_reject(assessment_id, request, idempotency_key, principal, session)


async def _handle_reinvestigate(
    assessment_id: str,
    request: RequestReinvestigationRequest,
    idempotency_key: str | None,
    principal: Principal,
    session: AsyncSession,
) -> ReviewDecisionResponse:
    if not idempotency_key or not idempotency_key.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "MISSING_IDEMPOTENCY_KEY",
                "message": "Idempotency-Key header is required",
            },
        )

    try:
        cmd = RequestReinvestigationCommand(
            assessment_id=assessment_id,
            run_number=request.run_number,
            reviewer_id=principal.principal_id,
            idempotency_key=idempotency_key.strip(),
            rationale=request.rationale,
            questions=tuple(request.questions),
            evidence_targets=tuple(request.evidence_targets),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR", "message": str(e)},
        ) from e

    service = ReviewDecisionService(session)
    try:
        result = await service.record_review_action(cmd)
    except ReinvestigationLimitReachedError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "REINVESTIGATION_LIMIT_REACHED", "message": str(e)},
        ) from e
    except IdempotencyConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "IDEMPOTENCY_CONFLICT", "message": str(e)},
        ) from e
    except ReviewActionConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "REVIEW_ACTION_CONFLICT", "message": str(e)},
        ) from e
    except AssessmentNotReviewableError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "ASSESSMENT_NOT_REVIEWABLE", "message": str(e)},
        ) from e
    except AssessmentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ASSESSMENT_NOT_FOUND", "message": str(e)},
        ) from e
    except DraftNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DRAFT_NOT_FOUND", "message": str(e)},
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR", "message": str(e)},
        ) from e

    return _to_response(result)


@router.post(
    "/{assessment_id}/review/reinvestigate",
    response_model=ReviewDecisionResponse,
    status_code=status.HTTP_200_OK,
)
@router.post(
    "/{assessment_id}/reviews/reinvestigate",
    response_model=ReviewDecisionResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def request_reinvestigation(
    assessment_id: str,
    request: RequestReinvestigationRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    principal: Principal = Depends(require_role(Role.REVIEWER)),
    session: AsyncSession = Depends(get_db_session),
) -> ReviewDecisionResponse:
    """Reviewer requests additive reinvestigation within the 3-run limit."""
    return await _handle_reinvestigate(assessment_id, request, idempotency_key, principal, session)
