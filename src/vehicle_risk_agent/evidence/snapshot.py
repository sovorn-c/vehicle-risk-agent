"""Immutable Vehicle Evidence Snapshot domain models and transactional repository."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    FieldConflict,
    FieldExplanationResult,
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
    material_hash: str = Field(
        pattern=r"^[a-fA-F0-9]{64}$",
        description="SHA-256 fingerprint of evidence material",
    )
    canonical_fields: dict[str, Any] = Field(description="Resolved canonical fields")
    field_provenance: dict[str, tuple[ProvenanceLink, ...]] = Field(
        default_factory=dict, description="Lineage to all supporting source observations"
    )
    conflicts: tuple[FieldConflict, ...] = Field(
        default_factory=tuple, description="Recorded field conflicts"
    )
    confidence: ConfidenceAssessment = Field(description="Evidence confidence assessment")
    as_of: datetime = Field(description="Upstream evaluation timestamp")
    published_at: datetime = Field(description="Upstream database publication timestamp")
    synthetic_notice: str | None = Field(
        default=None, description="Disclaimer notice when record contains synthetic demo data"
    )
    history: tuple[VehicleRevisionResponse, ...] = Field(
        default_factory=tuple,
        description="Preceding revision history collected for temporal depth",
    )
    field_explanations: dict[str, FieldExplanationResult] = Field(
        default_factory=dict,
        description="Parallel field-level explanations and confidence breakdowns",
    )
    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def create_evidence_snapshot(
    assessment_id: str,
    run_number: int,
    revision: VehicleRevisionResponse,
    history: tuple[VehicleRevisionResponse, ...] | list[VehicleRevisionResponse] = (),
    field_explanations: dict[str, FieldExplanationResult] | None = None,
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
        field_provenance={k: tuple(v) for k, v in revision.field_provenance.items()},
        conflicts=tuple(revision.conflicts),
        confidence=revision.confidence,
        as_of=revision.as_of,
        published_at=revision.published_at,
        synthetic_notice=revision.synthetic_notice,
        history=tuple(history),
        field_explanations=dict(field_explanations) if field_explanations else {},
        collected_at=collected_at or datetime.now(UTC),
    )


class VehicleEvidenceRepository:
    """Provides transactional persistence for vehicle evidence snapshots."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_snapshot(self, snapshot: VehicleEvidenceSnapshot) -> None:
        """Persist or update an immutable evidence snapshot idempotently."""
        snapshot_json = snapshot.model_dump_json()

        stmt = select(VehicleEvidenceSnapshotRecord).where(
            VehicleEvidenceSnapshotRecord.assessment_id == snapshot.assessment_id,
            VehicleEvidenceSnapshotRecord.run_number == snapshot.run_number,
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()

        if record is not None:
            record.vin = snapshot.vin
            record.revision_id = snapshot.revision_id
            record.revision_number = snapshot.revision_number
            record.material_hash = snapshot.material_hash
            record.snapshot_data_json = snapshot_json
            record.collected_at = snapshot.collected_at
        else:
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
        """Retrieve evidence snapshot by assessment ID and run number with hash verification."""
        stmt = select(VehicleEvidenceSnapshotRecord).where(
            VehicleEvidenceSnapshotRecord.assessment_id == assessment_id,
            VehicleEvidenceSnapshotRecord.run_number == run_number,
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None

        data = json.loads(record.snapshot_data_json)
        snapshot = VehicleEvidenceSnapshot(**data)

        # Authenticate that record database material_hash matches snapshot
        if snapshot.material_hash != record.material_hash:
            raise ValueError(
                f"Persisted evidence snapshot hash mismatch for {assessment_id}:{run_number}"
            )

        return snapshot

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

    async def get_sufficiency_result(self, assessment_id: str, run_number: int) -> Any:
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
