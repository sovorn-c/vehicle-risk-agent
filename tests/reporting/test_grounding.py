"""RED tests for claim grounding validation and bounded repair (e04s03-t02).

These tests drive the creation of:
  src/vehicle_risk_agent/reporting/grounding.py

The grounding validator checks every model draft claim against the pinned
evidence and policy citation allowlists from the run state.
"""

# story: e04s03

from __future__ import annotations

import pytest

from vehicle_risk_agent.reporting.grounding import GroundingContext
from vehicle_risk_agent.reporting.models import (
    ClaimReference,
    ReportDraft,
)
from vehicle_risk_agent.reporting.protocol import ReportDraftingContext
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    RiskBand,
    RiskResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_risk_result(incomplete: bool = False) -> RiskResult:
    from vehicle_risk_agent.evidence.sufficiency import (
        MissingEvidenceFinding,
        MissingEvidenceReason,
    )
    from vehicle_risk_agent.risk.models import RiskFactor, RiskFactorResult

    if incomplete:
        return RiskResult(
            id="rr-ground-inc",
            assessment_id="assess-ground-01",
            run_number=1,
            policy_id="nz-vehicle-risk-v1",
            policy_version="1.0",
            score=None,
            raw_score=None,
            band=None,
            outcome=AssessmentOutcome.INCOMPLETE,
            is_incomplete=True,
            factors=(),
            findings=(),
            missing_evidence=(
                MissingEvidenceFinding(
                    field_name="ppsr_interest",
                    reason=MissingEvidenceReason.LOOKUP_FAILED,
                    details="PPSR register unavailable",
                ),
            ),
            missing_findings=(),
            calculation_hash="inc-hash",
        )
    return RiskResult(
        id="rr-ground-01",
        assessment_id="assess-ground-01",
        run_number=1,
        policy_id="nz-vehicle-risk-v1",
        policy_version="1.0",
        score=45,
        raw_score=45,
        band=RiskBand.HIGH,
        outcome=AssessmentOutcome.SCORED,
        is_incomplete=False,
        factors=(
            RiskFactorResult(
                factor=RiskFactor.MATCH,
                triggered=True,
                weight=30,
                score_contribution=30,
                evidence_field="ppsr_interest",
                evidence_refs=("obs-match-001",),
                policy_citation_refs=("pol-match-001",),
            ),
            RiskFactorResult(
                factor=RiskFactor.LISTED,
                triggered=True,
                weight=45,
                score_contribution=45,
                evidence_field="stolen_vehicle",
                evidence_refs=("obs-listed-001",),
                policy_citation_refs=("pol-listed-001",),
            ),
        ),
        findings=(),
        missing_evidence=(),
        missing_findings=(),
        calculation_hash="def456",
    )


def _make_grounding_context(
    allowed_evidence_ids: tuple[str, ...] = ("obs-match-001", "obs-listed-001"),
    allowed_citation_ids: tuple[str, ...] = ("pol-match-001", "pol-listed-001"),
    risk_result: RiskResult | None = None,
) -> GroundingContext:
    from vehicle_risk_agent.reporting.grounding import GroundingContext

    return GroundingContext(
        assessment_id="assess-ground-01",
        allowed_evidence_ids=allowed_evidence_ids,
        allowed_citation_ids=allowed_citation_ids,
        risk_result=risk_result or _make_risk_result(),
    )


def _make_drafting_context(assessment_id: str = "assess-ground-01") -> ReportDraftingContext:
    """Build a minimal ReportDraftingContext with pinned evidence for grounding tests."""
    from vehicle_risk_agent.reporting.protocol import EvidenceItem

    return ReportDraftingContext(
        assessment_id=assessment_id,
        run_number=1,
        vehicle_id="VIN-GRND-001",
        vin="VIN-GRND-001",
        risk_result=_make_risk_result(),
        evidence_items=(
            EvidenceItem(
                field_name="ppsr_interest",
                value=True,
                observation_id="obs-match-001",
                source_system="PPSR",
            ),
            EvidenceItem(
                field_name="stolen_vehicle",
                value=True,
                observation_id="obs-listed-001",
                source_system="SVR",
            ),
        ),
    )


def _make_draft_with_claims(claims: tuple[ClaimReference, ...]) -> ReportDraft:
    """Build a minimal ReportDraft with custom claims in the executive summary."""
    import asyncio

    from vehicle_risk_agent.reporting.offline import OfflineReportDraftingAdapter

    ctx = _make_drafting_context()
    adapter = OfflineReportDraftingAdapter()

    loop = asyncio.new_event_loop()
    try:
        base_draft = loop.run_until_complete(adapter.draft_report(ctx))
    finally:
        loop.close()

    # Inject custom claims into the executive summary section
    new_exec = base_draft.sections.executive_summary.model_copy(update={"claims": claims})
    new_sections = base_draft.sections.model_copy(update={"executive_summary": new_exec})
    return base_draft.model_copy(update={"sections": new_sections})


# ---------------------------------------------------------------------------
# t02-a: GroundingContext instantiates correctly
# ---------------------------------------------------------------------------


def test_grounding_context_instantiation() -> None:
    """GroundingContext can be created with allowlists from pinned run state."""
    ctx = _make_grounding_context()
    assert "obs-match-001" in ctx.allowed_evidence_ids
    assert "pol-match-001" in ctx.allowed_citation_ids


# ---------------------------------------------------------------------------
# t02-b: Well-grounded draft passes validation
# ---------------------------------------------------------------------------


def test_grounding_validator_passes_fully_grounded_draft() -> None:
    """A draft whose claims only reference pinned evidence/citations passes."""
    from vehicle_risk_agent.reporting.grounding import GroundingValidator

    grounding_ctx = _make_grounding_context()
    validator = GroundingValidator()

    grounded_claim = ClaimReference(
        claim_id="claim-grounded-01",
        statement="Vehicle has PPSR interest recorded",
        evidence_refs=("obs-match-001",),
        policy_citation_refs=("pol-match-001",),
    )
    draft = _make_draft_with_claims((grounded_claim,))
    result = validator.validate(draft, grounding_ctx)

    assert result.is_valid
    assert result.repair_needed is False
    assert len(result.ungrounded_claims) == 0


# ---------------------------------------------------------------------------
# t02-c: Draft with invented evidence ref fails grounding
# ---------------------------------------------------------------------------


def test_grounding_validator_rejects_invented_evidence_ref() -> None:
    """A claim referencing an evidence ID not in the pinned set fails."""
    from vehicle_risk_agent.reporting.grounding import GroundingValidator

    grounding_ctx = _make_grounding_context()
    validator = GroundingValidator()

    bad_claim = ClaimReference(
        claim_id="claim-bad-001",
        statement="Invented evidence claim",
        evidence_refs=("obs-INVENTED-XYZ",),  # Not in allowlist
        policy_citation_refs=(),
    )
    draft = _make_draft_with_claims((bad_claim,))
    result = validator.validate(draft, grounding_ctx)

    assert result.is_valid is False
    assert result.repair_needed is True
    assert any(c.claim_id == "claim-bad-001" for c in result.ungrounded_claims)


# ---------------------------------------------------------------------------
# t02-d: Draft with invented citation ref fails grounding
# ---------------------------------------------------------------------------


def test_grounding_validator_rejects_invented_citation_ref() -> None:
    """A claim referencing a citation not in the allowlist fails grounding."""
    from vehicle_risk_agent.reporting.grounding import GroundingValidator

    grounding_ctx = _make_grounding_context()
    validator = GroundingValidator()

    bad_claim = ClaimReference(
        claim_id="claim-bad-cit-001",
        statement="Invented citation claim",
        evidence_refs=(),
        policy_citation_refs=("pol-INVENTED-ABC",),  # Not in allowlist
    )
    draft = _make_draft_with_claims((bad_claim,))
    result = validator.validate(draft, grounding_ctx)

    assert result.is_valid is False
    assert len(result.ungrounded_claims) >= 1


# ---------------------------------------------------------------------------
# t02-e: Repair attempt removes invented refs and text
# ---------------------------------------------------------------------------


def test_grounding_repair_strips_ungrounded_refs() -> None:
    """A single repair attempt produces a draft with ungrounded refs removed."""
    from vehicle_risk_agent.reporting.grounding import GroundingValidator

    grounding_ctx = _make_grounding_context()
    validator = GroundingValidator()

    bad_claim = ClaimReference(
        claim_id="claim-bad-repair-001",
        statement="Claim with an invented evidence ref",
        evidence_refs=("obs-INVENTED-XYZ",),
        policy_citation_refs=(),
    )
    draft = _make_draft_with_claims((bad_claim,))
    repaired = validator.repair(draft, grounding_ctx)

    # After repair: re-validate
    result = validator.validate(repaired, grounding_ctx)
    assert result.is_valid, "Repaired draft must pass grounding"


# ---------------------------------------------------------------------------
# t02-f: Repair cannot change score, band, or findings
# ---------------------------------------------------------------------------


def test_grounding_repair_does_not_alter_score_band_findings() -> None:
    """Repair is text-only — score, band, and findings are immutable."""
    from vehicle_risk_agent.reporting.grounding import GroundingValidator

    grounding_ctx = _make_grounding_context()
    validator = GroundingValidator()

    bad_claim = ClaimReference(
        claim_id="claim-bad-score-001",
        statement="Some claim",
        evidence_refs=("obs-INVENTED-XYZ",),
    )
    draft = _make_draft_with_claims((bad_claim,))
    repaired = validator.repair(draft, grounding_ctx)

    # Score and band must match original risk_result, not be altered
    orig = draft.sections.risk_score_and_band
    repaired_score_sec = repaired.sections.risk_score_and_band
    assert repaired_score_sec.score == orig.score
    assert repaired_score_sec.band == orig.band


# ---------------------------------------------------------------------------
# t02-g: Only one repair attempt is allowed
# ---------------------------------------------------------------------------


def test_grounding_validator_only_one_repair_attempt() -> None:
    """GroundingValidator tracks repair exhaustion and refuses a second repair."""
    from vehicle_risk_agent.reporting.grounding import GroundingValidator, RepairExhaustedError

    grounding_ctx = _make_grounding_context()
    validator = GroundingValidator()

    bad_claim = ClaimReference(
        claim_id="claim-bad-multi-001",
        statement="Bad claim",
        evidence_refs=("obs-INVENTED-XYZ",),
    )
    draft = _make_draft_with_claims((bad_claim,))

    # First repair is allowed
    repaired = validator.repair(draft, grounding_ctx)

    # Second repair must be rejected
    with pytest.raises(RepairExhaustedError):
        validator.repair(repaired, grounding_ctx)


# ---------------------------------------------------------------------------
# t02-h: GroundingResult is a strict Pydantic model
# ---------------------------------------------------------------------------


def test_grounding_result_is_immutable_pydantic_model() -> None:
    """GroundingResult rejects mutation after creation."""
    from pydantic import ValidationError

    from vehicle_risk_agent.reporting.grounding import GroundingResult

    result = GroundingResult(
        is_valid=True,
        repair_needed=False,
        ungrounded_claims=(),
        repair_count=0,
    )
    with pytest.raises((ValidationError, TypeError)):
        result.is_valid = False  # frozen=True raises ValidationError/TypeError at runtime
