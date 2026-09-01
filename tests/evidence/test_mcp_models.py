"""Tests for strict local mirror Pydantic models for MCP vehicle intelligence evidence."""

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.evidence.models import (
    CandidateValue,
    ConfidenceAssessment,
    ConfidenceBand,
    ConflictState,
    FieldConflict,
    FieldExplanationResult,
    FieldOutcome,
    ProvenanceLink,
    SafeError,
    SafeErrorCategory,
    SourceObservationResponse,
    VehicleRevisionResponse,
)


def test_vehicle_revision_response_strict_mirror() -> None:
    """VehicleRevisionResponse validates canonical payload and rejects unexpected fields."""
    now = datetime.now(UTC)
    provenance = ProvenanceLink(
        observation_id="obs-nzta-001",
        source_system="NZTA_MVR",
        source_record_id="rec-001",
        retrieved_at=now,
        synthetic=True,
    )
    confidence = ConfidenceAssessment(
        score=85,
        band=ConfidenceBand.HIGH,
        field_scores={"make": 90, "model": 90, "year": 85},
        field_components={
            "make": {"authority": 40, "freshness": 20, "agreement": 20, "validation": 10}
        },
        rule_version="conf-v1",
        explanation="High authority government source with complete matching fields.",
    )

    revision = VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-001",
        revision_number=1,
        material_hash="a" * 64,
        canonical_fields={"make": "TOYOTA", "model": "COROLLA", "year": 2018},
        field_provenance={"make": [provenance]},
        conflicts=[],
        confidence=confidence,
        as_of=now,
        published_at=now,
        synthetic_notice="SYNTHETIC TEST DATA",
    )

    assert revision.vin == "7AT0BK00X00000001"
    assert revision.canonical_fields["make"] == "TOYOTA"
    assert revision.synthetic_notice == "SYNTHETIC TEST DATA"

    # Strictness: Extra field is forbidden
    with pytest.raises(ValidationError):
        VehicleRevisionResponse(
            vin="7AT0BK00X00000001",
            revision_id="rev-001",
            revision_number=1,
            material_hash="a" * 64,
            canonical_fields={"make": "TOYOTA"},
            field_provenance={},
            conflicts=[],
            confidence=confidence,
            as_of=now,
            published_at=now,
            extra_forbidden_field="invalid",  # type: ignore[call-arg]
        )


def test_field_explanation_result_strict_mirror() -> None:
    """FieldExplanationResult models individual field outcomes (RESOLVED, UNRESOLVED, ABSENT)."""
    now = datetime.now(UTC)
    provenance = ProvenanceLink(
        observation_id="obs-ppsr-001",
        source_system="PPSR",
        source_record_id="ppsr-rec-01",
        retrieved_at=now,
        synthetic=False,
    )

    resolved = FieldExplanationResult(
        vin="7AT0BK00X00000001",
        revision_number=1,
        field_name="ppsr_result",
        outcome=FieldOutcome.RESOLVED,
        value="NO_FINANCE_REGISTERED",
        provenance=[provenance],
        conflicts=[],
        confidence_score=95,
        confidence_band=ConfidenceBand.HIGH,
        field_confidence_score=95,
        field_components={"authority": 40, "agreement": 25, "freshness": 20, "validation": 10},
        available_fields=["ppsr_result", "stolen_status", "writeoff_status"],
        rationale="PPSR certificate indicates clean title with zero active security interests.",
    )

    assert resolved.outcome == FieldOutcome.RESOLVED
    assert resolved.value == "NO_FINANCE_REGISTERED"

    absent = FieldExplanationResult(
        vin="7AT0BK00X00000001",
        revision_number=1,
        field_name="odometer_reading",
        outcome=FieldOutcome.ABSENT,
        value=None,
        provenance=[],
        conflicts=[],
        available_fields=["make", "model"],
        rationale="Field not present in upstream evidence sources.",
    )

    assert absent.outcome == FieldOutcome.ABSENT
    assert absent.value is None


def test_field_conflict_and_candidate_values() -> None:
    """FieldConflict preserves competing candidate values, resolution state, and rule version."""
    now = datetime.now(UTC)
    p1 = ProvenanceLink(
        observation_id="obs-1",
        source_system="SYSTEM_A",
        source_record_id="r1",
        retrieved_at=now,
    )
    p2 = ProvenanceLink(
        observation_id="obs-2",
        source_system="SYSTEM_B",
        source_record_id="r2",
        retrieved_at=now,
    )

    cand1 = CandidateValue(field_name="year", value=2018, provenance=p1)
    cand2 = CandidateValue(field_name="year", value=2019, provenance=p2)

    conflict = FieldConflict(
        field_name="year",
        conflicting_candidates=[cand1, cand2],
        state=ConflictState.UNRESOLVED,
        winning_value=None,
        rule_version="conflict-v1",
        rationale="Discrepancy between registration doc and auction catalog.",
    )

    assert conflict.state == ConflictState.UNRESOLVED
    assert len(conflict.conflicting_candidates) == 2


def test_source_observation_response_and_safe_error() -> None:
    """SourceObservationResponse and SafeError enforce bounded contract types."""
    now = datetime.now(UTC)
    raw_json = '{"raw": "payload"}'
    obs = SourceObservationResponse(
        observation_id="obs-nzta-001",
        source_system="NZTA",
        source_record_id="12345",
        ingestion_run_id="run-001",
        raw_payload=raw_json,
        payload_hash_sha256=hashlib.sha256(raw_json.encode("utf-8")).hexdigest(),
        retrieved_at=now,
        synthetic=True,
    )

    assert obs.source_system == "NZTA"
    assert obs.synthetic is True

    err = SafeError(
        category=SafeErrorCategory.VEHICLE_NOT_FOUND,
        message="VIN 7AT0BK00X00000001 was not found in catalog.",
        retryable=False,
        remediation="Verify the 17-character VIN and retry with an existing vehicle.",
    )

    assert err.category == SafeErrorCategory.VEHICLE_NOT_FOUND
    assert not err.retryable
