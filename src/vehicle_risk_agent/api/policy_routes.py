"""FastAPI routes for Policy Sources, Snapshots, and Corpus management."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.api.deps import (
    get_current_principal,
    get_db_session,
    require_role,
)
from vehicle_risk_agent.api.policy_schemas import (
    PolicyPassageResponse,
    PolicySnapshotIngestRequest,
    PolicySnapshotResponse,
    PolicySourceCreateRequest,
    PolicySourceResponse,
)
from vehicle_risk_agent.auth import Principal, Role
from vehicle_risk_agent.persistence.policy_repository import PolicyRepository
from vehicle_risk_agent.policy.ingestion import (
    PolicyIngestionError,
    PolicyParser,
    ingest_policy_source,
)
from vehicle_risk_agent.policy.models import PolicySource

router = APIRouter(prefix="/api/v1/policy", tags=["policy"])


@router.post(
    "/sources",
    response_model=PolicySourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_policy_source(
    request: PolicySourceCreateRequest,
    principal: Principal = Depends(require_role(Role.POLICY_CORPUS_MAINTAINER)),
    session: AsyncSession = Depends(get_db_session),
) -> PolicySourceResponse:
    """Register or update an official Policy Source definition."""
    repo = PolicyRepository(session)
    source = PolicySource(
        id=request.id,
        title=request.title,
        issuing_authority=request.issuing_authority,
        jurisdiction=request.jurisdiction,
        canonical_origin=request.canonical_origin,
        authority_classification=request.authority_classification,
        reuse_terms=request.reuse_terms,
        expected_update_cadence=request.expected_update_cadence,
    )
    saved = await repo.create_or_update_source(source)
    return PolicySourceResponse(
        id=saved.id,
        title=saved.title,
        issuing_authority=saved.issuing_authority,
        jurisdiction=saved.jurisdiction,
        canonical_origin=saved.canonical_origin,
        authority_classification=str(saved.authority_classification),
        reuse_terms=saved.reuse_terms,
        expected_update_cadence=saved.expected_update_cadence,
        status=str(saved.status),
        created_at=saved.created_at,
        updated_at=saved.updated_at,
    )


@router.get("/sources", response_model=list[PolicySourceResponse])
async def list_policy_sources(
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
) -> list[PolicySourceResponse]:
    """List all registered Policy Sources."""
    repo = PolicyRepository(session)
    sources = await repo.list_sources()
    return [
        PolicySourceResponse(
            id=s.id,
            title=s.title,
            issuing_authority=s.issuing_authority,
            jurisdiction=s.jurisdiction,
            canonical_origin=s.canonical_origin,
            authority_classification=str(s.authority_classification),
            reuse_terms=s.reuse_terms,
            expected_update_cadence=s.expected_update_cadence,
            status=str(s.status),
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in sources
    ]


@router.get("/sources/{source_id}", response_model=PolicySourceResponse)
async def get_policy_source(
    source_id: str,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
) -> PolicySourceResponse:
    """Retrieve a specific Policy Source by its ID."""
    repo = PolicyRepository(session)
    source = await repo.get_source(source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Policy source {source_id} not found"},
        )
    return PolicySourceResponse(
        id=source.id,
        title=source.title,
        issuing_authority=source.issuing_authority,
        jurisdiction=source.jurisdiction,
        canonical_origin=source.canonical_origin,
        authority_classification=str(source.authority_classification),
        reuse_terms=source.reuse_terms,
        expected_update_cadence=source.expected_update_cadence,
        status=str(source.status),
        created_at=source.created_at,
        updated_at=source.updated_at,
    )


@router.post(
    "/sources/{source_id}/snapshots",
    response_model=PolicySnapshotResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_snapshot(
    source_id: str,
    request: PolicySnapshotIngestRequest,
    principal: Principal = Depends(require_role(Role.POLICY_CORPUS_MAINTAINER)),
    session: AsyncSession = Depends(get_db_session),
) -> PolicySnapshotResponse:
    """Ingest, validate, and parse a point-in-time snapshot for a policy source."""
    repo = PolicyRepository(session)
    source = await repo.get_source(source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Policy source {source_id} not found"},
        )

    try:
        snapshot = ingest_policy_source(
            source=source,
            raw_content=request.raw_content,
            parser=PolicyParser(),
            effective_date=request.effective_date,
            publication_date=request.publication_date,
        )
    except PolicyIngestionError as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "INVALID_POLICY_CONTENT", "message": str(err)},
        ) from err

    saved = await repo.create_snapshot(snapshot)
    return PolicySnapshotResponse(
        id=saved.id,
        source_id=saved.source_id,
        retrieved_at=saved.retrieved_at,
        effective_date=saved.effective_date,
        publication_date=saved.publication_date,
        content_hash=saved.content_hash,
        parser_version=saved.parser_version,
        validation_outcome=str(saved.validation_outcome),
        passages=[
            PolicyPassageResponse(
                id=p.id,
                snapshot_id=p.snapshot_id,
                source_id=p.source_id,
                section_identifier=p.section_identifier,
                heading=p.heading,
                text=p.text,
                sequence=p.sequence,
                content_hash=p.content_hash,
            )
            for p in saved.passages
        ],
    )


@router.get("/snapshots/{snapshot_id}", response_model=PolicySnapshotResponse)
async def get_policy_snapshot(
    snapshot_id: str,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
) -> PolicySnapshotResponse:
    """Retrieve a specific Policy Snapshot and its passages."""
    repo = PolicyRepository(session)
    snapshot = await repo.get_snapshot(snapshot_id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Policy snapshot {snapshot_id} not found"},
        )
    return PolicySnapshotResponse(
        id=snapshot.id,
        source_id=snapshot.source_id,
        retrieved_at=snapshot.retrieved_at,
        effective_date=snapshot.effective_date,
        publication_date=snapshot.publication_date,
        content_hash=snapshot.content_hash,
        parser_version=snapshot.parser_version,
        validation_outcome=str(snapshot.validation_outcome),
        passages=[
            PolicyPassageResponse(
                id=p.id,
                snapshot_id=p.snapshot_id,
                source_id=p.source_id,
                section_identifier=p.section_identifier,
                heading=p.heading,
                text=p.text,
                sequence=p.sequence,
                content_hash=p.content_hash,
            )
            for p in snapshot.passages
        ],
    )
