"""Immutable Vehicle Evidence Snapshot domain models and transactional repository."""

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    FieldConflict,
    ProvenanceLink,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.persistence.models import VehicleEvidenceSnapshotRecord


class VehicleEvidenceSnapshot(BaseModel):
    """Immutable point-in-time vehicle intelligence evidence captured for an Assessment Run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    assessment_id: str = Field(description="Unique assessment identifier")
    run_number: int = Field(ge=1, description="Assessment run sequence number")
    vin: str = Field(description="Canonical 17-character VIN")
    revision_id: str = Field(description="Unique upstream revision identifier")
    revision_number: int = Field(ge=1, description="Monotonic upstream revision number")
    material_hash: str = Field(min_length=64, max_length=64, description="Fingerprint of evidence")
    canonical_fields: dict[str, Any] = Field(description="Resolved canonical fields")
    field_provenance: dict[str, list[ProvenanceLink]] = Field(
        default_factory=dict, description="Lineage to all supporting source observations"
    )
    conflicts: list[FieldConflict] = Field(
        default_factory=list, description="Recorded field conflicts"
    )
    confidence: ConfidenceAssessment = Field(description="Evidence confidence assessment")
    as_of: datetime = Field(description="Upstream evaluation timestamp")
    published_at: datetime = Field(description="Upstream database publication timestamp")
    synthetic_notice: str | None = Field(
        default=None, description="Disclaimer notice when record contains synthetic demo data"
    )
    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def create_evidence_snapshot(
    assessment_id: str,
    run_number: int,
    revision: VehicleRevisionResponse,
    collected_at: datetime | None = None,
) -> VehicleEvidenceSnapshot:
    """Construct an immutable VehicleEvidenceSnapshot from an upstream VehicleRevisionResponse."""
    return VehicleEvidenceSnapshot(
        assessment_id=assessment_id,
        run_number=run_number,
        vin=revision.vin,
        revision_id=revision.revision_id,
        revision_number=revision.revision_number,
        material_hash=revision.material_hash,
        canonical_fields=dict(revision.canonical_fields),
        field_provenance=dict(revision.field_provenance),
        conflicts=list(revision.conflicts),
        confidence=revision.confidence,
        as_of=revision.as_of,
        published_at=revision.published_at,
        synthetic_notice=revision.synthetic_notice,
        collected_at=collected_at or datetime.now(UTC),
    )


class VehicleEvidenceRepository:
    """Provides transactional persistence for vehicle evidence snapshots."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_snapshot(self, snapshot: VehicleEvidenceSnapshot) -> None:
        """Persist or update an immutable evidence snapshot."""
        snapshot_json = snapshot.model_dump_json()
        record = VehicleEvidenceSnapshotRecord(
            assessment_id=snapshot.assessment_id,
            run_number=snapshot.run_number,
            vin=snapshot.vin,
            revision_id=snapshot.revision_id,
            revision_number=snapshot.revision_number,
            material_hash=snapshot.material_hash,
            snapshot_data_json=snapshot_json,
            collected_at=snapshot.collected_at,
        )
        self._session.add(record)
        await self._session.commit()

    async def get_snapshot(
        self, assessment_id: str, run_number: int
    ) -> VehicleEvidenceSnapshot | None:
        """Retrieve evidence snapshot by assessment ID and run number."""
        stmt = select(VehicleEvidenceSnapshotRecord).where(
            VehicleEvidenceSnapshotRecord.assessment_id == assessment_id,
            VehicleEvidenceSnapshotRecord.run_number == run_number,
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None

        data = json.loads(record.snapshot_data_json)
        return VehicleEvidenceSnapshot(**data)

    async def save_sufficiency_result(
        self, assessment_id: str, run_number: int, sufficiency: Any
    ) -> None:
        """Persist or update sufficiency evaluation result for an assessment run."""
        stmt = select(VehicleEvidenceSnapshotRecord).where(
            VehicleEvidenceSnapshotRecord.assessment_id == assessment_id,
            VehicleEvidenceSnapshotRecord.run_number == run_number,
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is not None:
            record.sufficiency_json = sufficiency.model_dump_json()
            await self._session.commit()

    async def get_sufficiency_result(
        self, assessment_id: str, run_number: int
    ) -> Any:
        """Retrieve sufficiency result for an assessment run."""
        stmt = select(VehicleEvidenceSnapshotRecord).where(
            VehicleEvidenceSnapshotRecord.assessment_id == assessment_id,
            VehicleEvidenceSnapshotRecord.run_number == run_number,
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None or record.sufficiency_json is None:
            return None

        from vehicle_risk_agent.evidence.sufficiency import EvidenceSufficiencyResult

        data = json.loads(record.sufficiency_json)
        return EvidenceSufficiencyResult(**data)

