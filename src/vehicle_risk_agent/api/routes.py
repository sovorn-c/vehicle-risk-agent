"""API routes for Assessment intake and retrieval."""

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.api.deps import (
    get_current_principal,
    get_db_session,
    intake_rate_limiter,
    require_role,
)
from vehicle_risk_agent.api.models import AssessmentCreateRequest
from vehicle_risk_agent.api.schemas import AssessmentResponse, AssessmentRunResponse
from vehicle_risk_agent.auth import Principal, Role
from vehicle_risk_agent.domain.errors import IdempotencyConflictError
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
