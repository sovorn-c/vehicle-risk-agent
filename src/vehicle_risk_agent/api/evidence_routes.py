"""API route handlers for vehicle evidence and reviewer source observation inspection."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.adapters.mcp import McpAdapterError, VehicleMcpClientAdapter
from vehicle_risk_agent.api.deps import (
    get_db_session,
    get_mcp_adapter,
    get_settings,
    require_role,
)
from vehicle_risk_agent.auth import Principal, Role
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.evidence.audit import (
    CorruptedObservationError,
    ObservationIdValidationError,
    ProvenanceMismatchError,
    SourceObservationAuditService,
    UnlinkedObservationError,
)
from vehicle_risk_agent.evidence.models import SourceObservationResponse
from vehicle_risk_agent.evidence.snapshot import SnapshotIntegrityError, VehicleEvidenceRepository

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
    _principal: Principal = Depends(require_role(Role.REVIEWER)),
    session: AsyncSession = Depends(get_db_session),
    mcp_adapter: VehicleMcpClientAdapter = Depends(get_mcp_adapter),
    settings: Settings = Depends(get_settings),
) -> SourceObservationResponse:
    """Allow authorized reviewers to inspect exact source observation payloads."""
    repo = VehicleEvidenceRepository(
        session,
        integrity_secret=settings.snapshot_integrity_secret.get_secret_value(),
    )
    try:
        snapshot = await repo.get_snapshot(assessment_id, run_number)
    except SnapshotIntegrityError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "SNAPSHOT_INTEGRITY_ERROR",
                "message": "Evidence snapshot integrity verification failed",
            },
        ) from e

    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SNAPSHOT_NOT_FOUND",
                "message": (
                    f"Evidence snapshot not found for assessment {assessment_id} run {run_number}"
                ),
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
    except (UnlinkedObservationError, ObservationIdValidationError) as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "OBSERVATION_NOT_FOUND",
                "message": "Source observation was not found in assessment run provenance",
            },
        ) from e
    except McpAdapterError as e:
        error_status = {
            "INVALID_INPUT": status.HTTP_400_BAD_REQUEST,
            "VEHICLE_NOT_FOUND": status.HTTP_404_NOT_FOUND,
            "REVISION_NOT_FOUND": status.HTTP_404_NOT_FOUND,
            "OBSERVATION_NOT_FOUND": status.HTTP_404_NOT_FOUND,
            "PIPELINE_TIMEOUT": status.HTTP_503_SERVICE_UNAVAILABLE,
            "PIPELINE_UNAVAILABLE": status.HTTP_503_SERVICE_UNAVAILABLE,
            "PIPELINE_CONTRACT_ERROR": status.HTTP_502_BAD_GATEWAY,
            "INTERNAL_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        }[e.category.value]
        raise HTTPException(
            status_code=error_status,
            detail={"code": e.category.value, "message": e.message},
        ) from e
    except (CorruptedObservationError, ProvenanceMismatchError) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "OBSERVATION_INTEGRITY_ERROR",
                "message": "Source observation integrity verification failed",
            },
        ) from e
