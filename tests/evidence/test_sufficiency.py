"""Tests for evaluating Vehicle Evidence Sufficiency and required field completeness."""

from datetime import UTC, datetime

import pytest

from vehicle_risk_agent.evidence.models import (
    CandidateValue,
    ConfidenceAssessment,
    ConfidenceBand,
    ConflictState,
    FieldConflict,
    ProvenanceLink,
    SafeError,
    SafeErrorCategory,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.snapshot import create_evidence_snapshot
from vehicle_risk_agent.evidence.sufficiency import (
    EvidenceSufficiencyResult,
    MissingEvidenceReason,
    SufficiencyOutcome,
    evaluate_evidence_sufficiency,
)


@pytest.fixture
def base_revision() -> VehicleRevisionResponse:
    now = datetime.now(UTC)
    return VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-1",
        revision_number=1,
        material_hash="a" * 64,
        canonical_fields={
            "make": "TOYOTA",
            "model": "YARIS",
            "year": 2017,
            "ppsr_result": "NO_FINANCE_REGISTERED",
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        },
        field_provenance={},
        conflicts=[],
        confidence=ConfidenceAssessment(
            score=90,
            band=ConfidenceBand.HIGH,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="all good",
        ),
        as_of=now,
        published_at=now,
    )


def test_sufficiency_complete_when_all_three_required_fields_present(
    base_revision: VehicleRevisionResponse,
) -> None:
    """Sufficiency is COMPLETE when ppsr_result, stolen_status, and writeoff_status are resolved."""
    snapshot = create_evidence_snapshot("asmt-1", 1, base_revision)
    result = evaluate_evidence_sufficiency(snapshot)

    assert isinstance(result, EvidenceSufficiencyResult)
    assert result.outcome == SufficiencyOutcome.COMPLETE
    assert result.is_sufficient is True
    assert len(result.missing_findings) == 0


def test_sufficiency_incomplete_when_field_is_unknown(
    base_revision: VehicleRevisionResponse,
) -> None:
    """When a required field has value UNKNOWN, finding records reason UNKNOWN."""
    fields = dict(base_revision.canonical_fields)
    fields["stolen_status"] = "UNKNOWN"
    rev = base_revision.model_copy(update={"canonical_fields": fields})

    snapshot = create_evidence_snapshot("asmt-1", 1, rev)
    result = evaluate_evidence_sufficiency(snapshot)

    assert result.outcome == SufficiencyOutcome.INCOMPLETE
    assert result.is_sufficient is False
    assert len(result.missing_findings) == 1
    assert result.missing_findings[0].field_name == "stolen_status"
    assert result.missing_findings[0].reason == MissingEvidenceReason.UNKNOWN


def test_sufficiency_incomplete_when_field_is_absent(
    base_revision: VehicleRevisionResponse,
) -> None:
    """When a required field is absent from canonical fields, finding records reason ABSENT."""
    fields = dict(base_revision.canonical_fields)
    del fields["ppsr_result"]
    rev = base_revision.model_copy(update={"canonical_fields": fields})

    snapshot = create_evidence_snapshot("asmt-1", 1, rev)
    result = evaluate_evidence_sufficiency(snapshot)

    assert result.outcome == SufficiencyOutcome.INCOMPLETE
    assert result.is_sufficient is False
    assert any(
        f.field_name == "ppsr_result" and f.reason == MissingEvidenceReason.ABSENT
        for f in result.missing_findings
    )


def test_sufficiency_incomplete_when_field_has_unresolved_conflict(
    base_revision: VehicleRevisionResponse,
) -> None:
    """When a required field has an unresolved conflict, finding records reason UNRESOLVED_CONFLICT."""
    now = datetime.now(UTC)
    p = ProvenanceLink(
        observation_id="obs-1",
        source_system="SYS",
        source_record_id="r1",
        retrieved_at=now,
    )
    conflict = FieldConflict(
        field_name="writeoff_status",
        conflicting_candidates=[
            CandidateValue(field_name="writeoff_status", value="NOT_WRITTEN_OFF", provenance=p),
            CandidateValue(field_name="writeoff_status", value="WRITTEN_OFF", provenance=p),
        ],
        state=ConflictState.UNRESOLVED,
    )

    rev = base_revision.model_copy(update={"conflicts": [conflict]})
    snapshot = create_evidence_snapshot("asmt-1", 1, rev)
    result = evaluate_evidence_sufficiency(snapshot)

    assert result.outcome == SufficiencyOutcome.INCOMPLETE
    assert result.is_sufficient is False
    assert any(
        f.field_name == "writeoff_status"
        and f.reason == MissingEvidenceReason.UNRESOLVED_CONFLICT
        for f in result.missing_findings
    )


def test_sufficiency_unavailable_when_lookup_failed() -> None:
    """When snapshot is None due to upstream lookup failure, outcome is UNAVAILABLE."""
    err = SafeError(
        category=SafeErrorCategory.PIPELINE_TIMEOUT,
        message="Upstream MCP server timed out",
        retryable=True,
        remediation="Retry query",
    )
    result = evaluate_evidence_sufficiency(snapshot=None, failure_error=err)

    assert result.outcome == SufficiencyOutcome.UNAVAILABLE
    assert result.is_sufficient is False
    assert len(result.missing_findings) == 1
    assert result.missing_findings[0].reason == MissingEvidenceReason.LOOKUP_FAILED
