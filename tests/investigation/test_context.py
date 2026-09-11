"""Minimized proposal context tests."""

from datetime import UTC, datetime

from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    FieldExplanationResult,
    FieldOutcome,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.investigation.context import build_investigation_context


def _revision() -> VehicleRevisionResponse:
    return VehicleRevisionResponse(
        vin="1HGCM82633A004352",
        revision_id="rev-1",
        revision_number=3,
        material_hash="a" * 64,
        canonical_fields={"odometer_reading": 120000, "make": "Honda"},
        field_provenance={},
        confidence=ConfidenceAssessment(
            score=82,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="agreement",
        ),
        as_of=datetime.now(UTC),
        published_at=datetime.now(UTC),
    )


def test_context_contains_summaries_and_never_raw_payload() -> None:
    result = FieldExplanationResult(
        vin="1HGCM82633A004352",
        revision_number=3,
        field_name="odometer_reading",
        outcome=FieldOutcome.RESOLVED,
        value=120000,
        rationale="two sources agree",
    )
    context = build_investigation_context(
        revision=_revision(),
        questions=("Why is the odometer confidence low?",),
        evidence_targets=("odometer_reading",),
        prior_results=(result,),
    )
    assert context.evidence_targets == ("odometer_reading",)
    assert any("odometer_reading" in summary for summary in context.evidence_summaries)
    assert all("raw_payload" not in summary for summary in context.evidence_summaries)
    assert all(len(summary) <= 500 for summary in context.evidence_summaries)
