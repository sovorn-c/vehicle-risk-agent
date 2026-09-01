"""Immutable Vehicle Evidence Snapshot domain models and transactional repository."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.config import DEFAULT_SNAPSHOT_INTEGRITY_SECRET
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    FieldConflict,
    FieldExplanationResult,
    ProvenanceLink,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.persistence.models import VehicleEvidenceSnapshotRecord


class FrozenDict(dict[str, Any]):
    """JSON-compatible dictionary that rejects all in-place mutation."""

    def _raise_mutation(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("evidence snapshot is immutable")

    __setitem__ = _raise_mutation
    __delitem__ = _raise_mutation
    clear = _raise_mutation
    pop = _raise_mutation
    popitem = _raise_mutation  # type: ignore[assignment]
    setdefault = _raise_mutation
    update = _raise_mutation
    __ior__ = _raise_mutation  # type: ignore[assignment]


def _freeze_copy(value: Any) -> Any:
    """Clone and recursively freeze nested containers and Pydantic models."""
    if isinstance(value, BaseModel):
        copied = value.model_copy(deep=True)
        for field_name in type(copied).model_fields:
            object.__setattr__(
                copied,
                field_name,
                _freeze_copy(getattr(copied, field_name)),
            )
        return copied
    if isinstance(value, dict):
        return FrozenDict({key: _freeze_copy(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_copy(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_copy(item) for item in value)
    return value


class SnapshotIntegrityError(ValueError):
    """Raised when persisted evidence snapshot material or metadata is corrupted."""


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

    def model_post_init(self, __context: Any) -> None:
        """Clone and freeze every nested value after Pydantic validation."""
        for field_name in type(self).model_fields:
            object.__setattr__(self, field_name, _freeze_copy(getattr(self, field_name)))


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

    def __init__(
        self,
        session: AsyncSession,
        integrity_secret: str = DEFAULT_SNAPSHOT_INTEGRITY_SECRET,
    ) -> None:
        self._session = session
        self._integrity_secret = integrity_secret.encode("utf-8")

    def _integrity_hash(self, snapshot_json: str) -> str:
        return hmac.new(
            self._integrity_secret,
            snapshot_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def save_snapshot(self, snapshot: VehicleEvidenceSnapshot) -> None:
        """Persist one immutable snapshot, allowing only an identical retry."""
        snapshot_json = snapshot.model_dump_json()
        integrity_hash = self._integrity_hash(snapshot_json)
        values = {
            "assessment_id": snapshot.assessment_id,
            "run_number": snapshot.run_number,
            "vin": snapshot.vin,
            "revision_id": snapshot.revision_id,
            "revision_number": snapshot.revision_number,
            "material_hash": snapshot.material_hash,
            "snapshot_integrity_hash": integrity_hash,
            "snapshot_data_json": snapshot_json,
            "collected_at": snapshot.collected_at,
        }
        stmt = (
            insert(VehicleEvidenceSnapshotRecord)
            .values(values)
            .on_conflict_do_nothing(index_elements=["assessment_id", "run_number"])
        )
        await self._session.execute(stmt)

        result = await self._session.execute(
            select(VehicleEvidenceSnapshotRecord).where(
                VehicleEvidenceSnapshotRecord.assessment_id == snapshot.assessment_id,
                VehicleEvidenceSnapshotRecord.run_number == snapshot.run_number,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            raise SnapshotIntegrityError("Evidence snapshot could not be persisted")

        stored = self._verify_record(record)
        if stored != snapshot:
            raise ValueError(
                f"Evidence snapshot {snapshot.assessment_id}:{snapshot.run_number} is immutable"
            )
        await self._session.commit()

    def _verify_record(self, record: VehicleEvidenceSnapshotRecord) -> VehicleEvidenceSnapshot:
        """Verify serialized content, local integrity hash, and denormalized fields."""
        try:
            snapshot_json = record.snapshot_data_json
            computed_hash = self._integrity_hash(snapshot_json)
            if not hmac.compare_digest(record.snapshot_integrity_hash, computed_hash):
                raise SnapshotIntegrityError("Persisted evidence snapshot integrity check failed")
            snapshot = VehicleEvidenceSnapshot(**json.loads(snapshot_json))
        except SnapshotIntegrityError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SnapshotIntegrityError("Persisted evidence snapshot is invalid") from exc

        if (
            snapshot.assessment_id != record.assessment_id
            or snapshot.run_number != record.run_number
            or snapshot.vin != record.vin
            or snapshot.revision_id != record.revision_id
            or snapshot.revision_number != record.revision_number
            or snapshot.material_hash != record.material_hash
        ):
            raise SnapshotIntegrityError("Persisted evidence snapshot metadata check failed")
        return snapshot

    async def get_snapshot(
        self, assessment_id: str, run_number: int
    ) -> VehicleEvidenceSnapshot | None:
        """Retrieve and authenticate an immutable evidence snapshot."""
        stmt = select(VehicleEvidenceSnapshotRecord).where(
            VehicleEvidenceSnapshotRecord.assessment_id == assessment_id,
            VehicleEvidenceSnapshotRecord.run_number == run_number,
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return self._verify_record(record)

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
