"""API route handlers for vehicle evidence and reviewer source observation inspection."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.adapters.mcp import McpAdapterError, VehicleMcpClientAdapter
from vehicle_risk_agent.api.deps import get_db_session, get_mcp_adapter, require_role
from vehicle_risk_agent.auth import Principal, Role
from vehicle_risk_agent.evidence.audit import (
    ObservationIdValidationError,
    SourceObservationAuditService,
    UnlinkedObservationError,
)
from vehicle_risk_agent.evidence.models import SourceObservationResponse
from vehicle_risk_agent.evidence.snapshot import VehicleEvidenceRepository

router = APIRouter(prefix="/api/v1/assessments", tags=["evidence-audit"])


@router.get(
    "/{assessment_id}/runs/{run_number}/evidence/observations/{observation_id}",
    response_model=SourceObservationResponse,
    status_code=status.HTTP_200_OK,
)
async def get_source_observation_audit(
    assessment_id: str,
    run_number: int,
    observation_id: str,
    principal: Principal = Depends(require_role(Role.REVIEWER)),
    session: AsyncSession = Depends(get_db_session),
    mcp_adapter: VehicleMcpClientAdapter = Depends(get_mcp_adapter),
) -> SourceObservationResponse:
    """Allow authorized reviewers to inspect exact source observation payloads linked to evidence."""
    repo = VehicleEvidenceRepository(session)
    snapshot = await repo.get_snapshot(assessment_id, run_number)

    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SNAPSHOT_NOT_FOUND",
                "message": f"Evidence snapshot not found for assessment {assessment_id} run {run_number}",
            },
        )

    audit_service = SourceObservationAuditService()
    try:
        obs = await audit_service.resolve_linked_observation(
            snapshot=snapshot,
            observation_id=observation_id,
            adapter=mcp_adapter,
        )
        return obs
    except (UnlinkedObservationError, ObservationIdValidationError, McpAdapterError) as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "OBSERVATION_NOT_FOUND",
                "message": f"Source observation '{observation_id}' not found in assessment run provenance",
            },
        ) from e
