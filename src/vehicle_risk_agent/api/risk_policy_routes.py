"""FastAPI routes for Risk Policy management, validation, and operator activation."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.api.deps import (
    get_current_principal,
    get_db_session,
    require_role,
)
from vehicle_risk_agent.api.risk_policy_schemas import (
    RiskPolicyActivationResponse,
    RiskPolicyCreateRequest,
    RiskPolicyResponse,
)
from vehicle_risk_agent.auth import Principal, Role
from vehicle_risk_agent.risk.models import RiskBand, RiskBandDefinition
from vehicle_risk_agent.risk.repository import RiskPolicyRepository
from vehicle_risk_agent.risk.service import (
    RiskPolicyLifecycleError,
    RiskPolicyService,
)

router = APIRouter(prefix="/api/v1/risk-policies", tags=["risk-policies"])


@router.post(
    "",
    response_model=RiskPolicyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_risk_policy(
    request: RiskPolicyCreateRequest,
    _principal: Principal = Depends(require_role(Role.TECHNICAL_OPERATOR)),
    session: AsyncSession = Depends(get_db_session),
) -> RiskPolicyResponse:
    """Register a new Risk Policy in DRAFT state (Operator only)."""
    service = RiskPolicyService(session)

    risk_bands = None
    if request.risk_bands is not None:
        risk_bands = tuple(
            RiskBandDefinition(
                band=RiskBand(b["band"]),
                min_score=int(b["min_score"]),
                max_score=int(b["max_score"]),
                description=b.get("description", ""),
            )
            for b in request.risk_bands
        )

    try:
        policy = await service.create_policy(
            policy_id=request.id,
            name=request.name,
            description=request.description,
            version=request.version,
            factor_weights=request.factor_weights,
            score_cap=request.score_cap,
            risk_bands=risk_bands,
            mandatory_review_rules=request.mandatory_review_rules,
            required_evidence_fields=request.required_evidence_fields,
        )
    except (ValueError, KeyError, ValidationError) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_RISK_POLICY", "message": str(err)},
        ) from err

    return RiskPolicyResponse.from_domain(policy)


@router.post(
    "/{policy_id}/ready",
    response_model=RiskPolicyResponse,
    status_code=status.HTTP_200_OK,
)
async def validate_and_mark_policy_ready(
    policy_id: str,
    _principal: Principal = Depends(require_role(Role.TECHNICAL_OPERATOR)),
    session: AsyncSession = Depends(get_db_session),
) -> RiskPolicyResponse:
    """Validate policy rules and advance DRAFT policy to READY state (Operator only)."""
    service = RiskPolicyService(session)
    try:
        policy = await service.validate_and_mark_ready(policy_id)
    except RiskPolicyLifecycleError as err:
        msg = str(err)
        if "not found" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": msg},
            ) from err
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_STATE_TRANSITION", "message": msg},
        ) from err

    return RiskPolicyResponse.from_domain(policy)


@router.post(
    "/{policy_id}/activate",
    response_model=RiskPolicyActivationResponse,
    status_code=status.HTTP_200_OK,
)
async def activate_risk_policy(
    policy_id: str,
    principal: Principal = Depends(require_role(Role.TECHNICAL_OPERATOR)),
    session: AsyncSession = Depends(get_db_session),
) -> RiskPolicyActivationResponse:
    """Atomically activate a READY policy and retire the previous ACTIVE policy (Operator only)."""
    service = RiskPolicyService(session)
    try:
        active, retired = await service.activate_policy(policy_id, principal.principal_id)
    except RiskPolicyLifecycleError as err:
        msg = str(err)
        if "not found" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": msg},
            ) from err
        if "already active" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "ALREADY_ACTIVE", "message": msg},
            ) from err
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_STATE_TRANSITION", "message": msg},
        ) from err

    active_resp = RiskPolicyResponse.from_domain(active)
    retired_resp = RiskPolicyResponse.from_domain(retired) if retired is not None else None

    return RiskPolicyActivationResponse(active_policy=active_resp, retired_policy=retired_resp)


@router.get("/active", response_model=RiskPolicyResponse)
async def get_active_risk_policy(
    _principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
) -> RiskPolicyResponse:
    """Retrieve the single active Risk Policy."""
    repo = RiskPolicyRepository(session)
    active = await repo.get_active_policy()
    if active is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NO_ACTIVE_POLICY", "message": "No active risk policy found"},
        )
    return RiskPolicyResponse.from_domain(active)


@router.get("", response_model=list[RiskPolicyResponse])
async def list_risk_policies(
    _principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
) -> list[RiskPolicyResponse]:
    """List all registered Risk Policies."""
    repo = RiskPolicyRepository(session)
    policies = await repo.list_policies()
    return [RiskPolicyResponse.from_domain(p) for p in policies]


@router.get("/{policy_id}", response_model=RiskPolicyResponse)
async def get_risk_policy(
    policy_id: str,
    _principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
) -> RiskPolicyResponse:
    """Retrieve a specific Risk Policy by its ID (including RETIRED policies)."""
    repo = RiskPolicyRepository(session)
    policy = await repo.get_policy(policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Risk policy '{policy_id}' not found"},
        )
    return RiskPolicyResponse.from_domain(policy)
