"""Tests for resolving only provenance-linked bounded observation identifiers for reviewer audit."""

# story: e03s04

import hashlib
from datetime import UTC, datetime

import pytest

from vehicle_risk_agent.adapters.mcp import FakeVehicleMcpAdapter
from vehicle_risk_agent.evidence.audit import (
    ObservationIdValidationError,
    SourceObservationAuditService,
    UnlinkedObservationError,
)
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    ProvenanceLink,
    SourceObservationResponse,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.snapshot import create_evidence_snapshot


@pytest.fixture
def snapshot_with_provenance() -> tuple[VehicleRevisionResponse, SourceObservationResponse]:
    now = datetime.now(UTC)
    raw_json = '{"make": "TOYOTA", "model": "VITZ", "year": 2015}'
    obs = SourceObservationResponse(
        observation_id="obs-nzta-001",
        source_system="NZTA_MVR",
        source_record_id="rec-001",
        ingestion_run_id="ingest-001",
        raw_payload=raw_json,
        payload_hash_sha256=hashlib.sha256(raw_json.encode("utf-8")).hexdigest(),
        retrieved_at=now,
        synthetic=True,
    )
    p = ProvenanceLink(
        observation_id="obs-nzta-001",
        source_system="NZTA_MVR",
        source_record_id="rec-001",
        retrieved_at=now,
        synthetic=True,
    )
    rev = VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-001",
        revision_number=1,
        material_hash="a" * 64,
        canonical_fields={"make": "TOYOTA", "model": "VITZ", "year": 2015},
        field_provenance={"make": [p]},
        conflicts=[],
        confidence=ConfidenceAssessment(
            score=90,
            band=ConfidenceBand.HIGH,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="verified",
        ),
        as_of=now,
        published_at=now,
        synthetic_notice="SYNTHETIC DEMO NOTICE",
    )
    return rev, obs


@pytest.mark.asyncio
async def test_resolve_provenance_linked_observation_success(
    snapshot_with_provenance: tuple[VehicleRevisionResponse, SourceObservationResponse],
) -> None:
    """Reviewer can resolve an observation ID that is explicitly linked in snapshot provenance."""
    rev, obs = snapshot_with_provenance
    adapter = FakeVehicleMcpAdapter()
    adapter.seed_source_observation(obs)

    snapshot = create_evidence_snapshot("asmt-1", 1, rev)
    service = SourceObservationAuditService()

    resolved_obs = await service.resolve_linked_observation(
        snapshot=snapshot,
        observation_id="obs-nzta-001",
        adapter=adapter,
    )

    assert resolved_obs.observation_id == "obs-nzta-001"
    assert resolved_obs.source_system == "NZTA_MVR"
    assert resolved_obs.synthetic is True
    assert "TOYOTA" in resolved_obs.raw_payload


@pytest.mark.asyncio
async def test_resolve_unlinked_observation_rejected(
    snapshot_with_provenance: tuple[VehicleRevisionResponse, SourceObservationResponse],
) -> None:
    """Attempting to resolve an observation ID not in the snapshot provenance is rejected."""
    rev, obs = snapshot_with_provenance
    adapter = FakeVehicleMcpAdapter()
    adapter.seed_source_observation(obs)

    snapshot = create_evidence_snapshot("asmt-1", 1, rev)
    service = SourceObservationAuditService()

    with pytest.raises(UnlinkedObservationError):
        await service.resolve_linked_observation(
            snapshot=snapshot,
            observation_id="obs-unlinked-999",
            adapter=adapter,
        )


@pytest.mark.asyncio
async def test_resolve_malformed_observation_id_rejected(
    snapshot_with_provenance: tuple[VehicleRevisionResponse, SourceObservationResponse],
) -> None:
    """Invalid observation IDs (empty, oversized, or malicious characters) raise ValidationError."""
    rev, _ = snapshot_with_provenance
    adapter = FakeVehicleMcpAdapter()
    snapshot = create_evidence_snapshot("asmt-1", 1, rev)
    service = SourceObservationAuditService()

    with pytest.raises(ObservationIdValidationError):
        await service.resolve_linked_observation(
            snapshot=snapshot,
            observation_id="../invalid/path",
            adapter=adapter,
        )
