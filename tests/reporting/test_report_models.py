"""Unit tests for Report Draft models, 9 canonical sections, claim citations, and notices."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.evidence.models import ConfidenceBand
from vehicle_risk_agent.evidence.sufficiency import MissingEvidenceFinding, MissingEvidenceReason
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.reporting.models import (
    AbstentionNotice,
    ClaimReference,
    ContributingFactorsSection,
    EvidenceSummarySection,
    ExecutiveSummarySection,
    LimitationsSection,
    MandatoryReviewSection,
    MissingEvidenceNotice,
    PolicyCitationsSection,
    ReportDraft,
    ReportDraftStatus,
    ReportSections,
    RiskScoreSection,
    SectionType,
    SyntheticNotice,
    SyntheticNoticeSection,
    VehicleIdentitySection,
    compute_draft_hash,
)
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    MandatoryFinding,
    RiskBand,
    RiskFactor,
    RiskFactorResult,
)


def _build_dummy_citation(passage_id: str = "snap-fta:p001") -> PolicyCitation:
    return PolicyCitation(
        passage_id=passage_id,
        snapshot_id="snap-fta",
        source_id="fta-1986",
        section_identifier="Section 9",
        heading="Misleading and Deceptive Conduct",
        source_title="Fair Trading Act 1986",
        canonical_origin="https://www.legislation.govt.nz/act/public/1986/0121/latest/DLM96439.html",
    )


def _build_sample_sections(
    outcome: AssessmentOutcome = AssessmentOutcome.SCORED,
    score: int | None = 30,
    band: RiskBand | None = RiskBand.MEDIUM,
    raw_score: int | None = 30,
    is_synthetic: bool = False,
    missing_findings: tuple[MissingEvidenceFinding, ...] = (),
) -> ReportSections:
    is_incomplete = outcome == AssessmentOutcome.INCOMPLETE

    exec_claims = (
        ClaimReference(
            claim_id="claim-exec-01",
            statement="PPSR registered security interest was detected.",
            evidence_refs=("obs-ppsr-001",),
            policy_citation_refs=("snap-ppsr:p001",),
            risk_factor_refs=(RiskFactor.MATCH,),
        ),
    )
    s1 = ExecutiveSummarySection(
        summary_text="Vehicle risk assessment draft.",
        outcome=outcome,
        key_findings=("Registered security interest detected on PPSR.",)
        if not is_incomplete
        else (),
        recommendation="Manual review required prior to release."
        if not is_incomplete
        else "Withheld pending missing evidence.",
        claims=exec_claims if not is_incomplete else (),
        evidence_refs=("obs-ppsr-001",) if not is_incomplete else (),
        policy_citation_refs=("snap-ppsr:p001",) if not is_incomplete else (),
        risk_factor_refs=(RiskFactor.MATCH,) if not is_incomplete else (),
    )

    s2 = VehicleIdentitySection(
        vin="7AT0BJ03X20000001",
        make="Toyota",
        model="Aqua",
        year=2015,
        plate="NZTEST1",
        confidence_score=95,
        confidence_band=ConfidenceBand.HIGH,
        evidence_refs=("obs-ident-001",),
    )

    s3 = RiskScoreSection(
        score=None if is_incomplete else score,
        band=None if is_incomplete else band,
        raw_score=None if is_incomplete else raw_score,
        policy_id="risk-policy-v1",
        policy_version="v1",
        calculation_hash="calc-hash-001",
        is_incomplete=is_incomplete,
        scoring_withheld_reason="Missing required evidence fields" if is_incomplete else None,
    )

    s4 = MandatoryReviewSection(
        findings=(
            MandatoryFinding(
                finding_id="finding-match-001",
                factor=RiskFactor.MATCH,
                title="Registered Security Interest Detected",
                description="PPSR record indicates active security interest.",
                evidence_field="ppsr_result",
                evidence_value="MATCH",
                weight=30,
                severity=RiskBand.MEDIUM,
                evidence_refs=("obs-ppsr-001",),
                policy_citation_refs=("snap-ppsr:p001",),
            ),
        )
        if not is_incomplete
        else (),
        evidence_refs=("obs-ppsr-001",) if not is_incomplete else (),
        policy_citation_refs=("snap-ppsr:p001",) if not is_incomplete else (),
        risk_factor_refs=(RiskFactor.MATCH,) if not is_incomplete else (),
    )

    s5 = ContributingFactorsSection(
        factor_breakdown=(
            RiskFactorResult(
                factor=RiskFactor.MATCH,
                weight=30,
                triggered=not is_incomplete,
                evidence_field="ppsr_result",
                evidence_value="MATCH" if not is_incomplete else None,
                rationale="Registered security interest identified on PPSR register",
                score_contribution=30 if not is_incomplete else 0,
                evidence_refs=("obs-ppsr-001",) if not is_incomplete else (),
                policy_citation_refs=("snap-ppsr:p001",) if not is_incomplete else (),
            ),
        ),
        triggered_factors=(RiskFactor.MATCH,) if not is_incomplete else (),
        risk_factor_refs=(RiskFactor.MATCH,) if not is_incomplete else (),
    )

    s6 = PolicyCitationsSection(
        citations=(_build_dummy_citation("snap-ppsr:p001"),),
        policy_citation_refs=("snap-ppsr:p001",),
    )

    s7 = EvidenceSummarySection(
        revision_id="rev-001",
        revision_number=1,
        material_hash="a" * 64,
        as_of=datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC),
        canonical_fields={"ppsr_result": "MATCH" if not is_incomplete else None},
        evidence_refs=("obs-ppsr-001", "obs-ident-001"),
    )

    missing_notices = tuple(
        MissingEvidenceNotice(
            field_name=mf.field_name,
            reason=mf.reason,
            details=mf.details,
        )
        for mf in missing_findings
    )
    s8 = LimitationsSection(
        missing_evidence_notices=missing_notices,
    )

    s9 = SyntheticNoticeSection(
        is_synthetic=is_synthetic,
        notice=SyntheticNotice(
            is_synthetic=True,
            notice_text="Assessment contains synthetic demonstration data.",
            synthetic_sources=("synth-obs-001",),
        )
        if is_synthetic
        else None,
        disclaimer_text="Synthetic demonstration data notice" if is_synthetic else None,
    )

    return ReportSections(
        executive_summary=s1,
        vehicle_identity=s2,
        risk_score_and_band=s3,
        mandatory_review_findings=s4,
        contributing_factors=s5,
        policy_citations=s6,
        evidence_summary=s7,
        limitations_and_missing_evidence=s8,
        synthetic_data_notice=s9,
    )


def test_valid_scored_report_draft_instantiation() -> None:
    """Verify standard SCORED ReportDraft instantiation with all 9 sections."""
    sections = _build_sample_sections(
        outcome=AssessmentOutcome.SCORED, score=30, band=RiskBand.MEDIUM
    )
    draft = ReportDraft(
        id="draft-001",
        assessment_id="asmt-001",
        run_number=1,
        risk_result_id="risk-res-001",
        status=ReportDraftStatus.DRAFT,
        outcome=AssessmentOutcome.SCORED,
        sections=sections,
    )

    assert draft.id == "draft-001"
    assert draft.assessment_id == "asmt-001"
    assert draft.run_number == 1
    assert draft.outcome == AssessmentOutcome.SCORED
    assert draft.score == 30
    assert draft.band == RiskBand.MEDIUM
    assert draft.vin == "7AT0BJ03X20000001"
    assert draft.policy_id == "risk-policy-v1"
    assert draft.policy_version == "v1"
    assert draft.draft_hash != ""
    assert len(draft.draft_hash) == 64

    # Verify 9 sections order and access
    sec_list = draft.sections.as_list()
    assert len(sec_list) == 9
    assert [s.section_type for s in sec_list] == [
        SectionType.EXECUTIVE_SUMMARY,
        SectionType.VEHICLE_IDENTITY,
        SectionType.RISK_SCORE_AND_BAND,
        SectionType.MANDATORY_REVIEW_FINDINGS,
        SectionType.CONTRIBUTING_FACTORS,
        SectionType.POLICY_CITATIONS,
        SectionType.EVIDENCE_SUMMARY,
        SectionType.LIMITATIONS_AND_MISSING_EVIDENCE,
        SectionType.SYNTHETIC_DATA_NOTICE,
    ]


def test_valid_incomplete_report_draft_instantiation() -> None:
    """Verify INCOMPLETE ReportDraft instantiation with withheld score and missing notices."""
    missing_findings = (
        MissingEvidenceFinding(
            field_name="ppsr_result",
            reason=MissingEvidenceReason.UNRESOLVED_CONFLICT,
            details="Conflicting PPSR registration status across registers",
        ),
    )
    sections = _build_sample_sections(
        outcome=AssessmentOutcome.INCOMPLETE,
        missing_findings=missing_findings,
    )
    draft = ReportDraft(
        id="draft-002",
        assessment_id="asmt-002",
        run_number=1,
        risk_result_id="risk-res-002",
        status=ReportDraftStatus.DRAFT,
        outcome=AssessmentOutcome.INCOMPLETE,
        sections=sections,
    )

    assert draft.outcome == AssessmentOutcome.INCOMPLETE
    assert draft.score is None
    assert draft.band is None
    assert draft.is_incomplete is True
    assert draft.sections.risk_score_and_band.is_incomplete is True
    assert len(draft.missing_evidence_notices) == 1
    assert draft.missing_evidence_notices[0].field_name == "ppsr_result"


def test_incomplete_report_with_numeric_score_raises_error() -> None:
    """Verify that an INCOMPLETE ReportDraft strictly rejects numeric scores or bands."""
    with pytest.raises(ValidationError):
        # Construct RiskScoreSection with is_incomplete=True but with score=30
        RiskScoreSection(
            score=30,
            band=RiskBand.MEDIUM,
            policy_id="risk-policy-v1",
            policy_version="v1",
            is_incomplete=True,
        )


def test_scored_report_without_score_or_band_raises_error() -> None:
    """Verify that a complete SCORED ReportDraft strictly requires both score and band."""
    with pytest.raises(ValidationError):
        RiskScoreSection(
            score=None,
            band=None,
            policy_id="risk-policy-v1",
            policy_version="v1",
            is_incomplete=False,
        )


def test_report_draft_outcome_mismatch_with_score_section_raises_error() -> None:
    """Verify that ReportDraft outcome must match RiskScoreSection completeness state."""
    # Scored score section
    scored_sections = _build_sample_sections(
        outcome=AssessmentOutcome.SCORED, score=30, band=RiskBand.MEDIUM
    )
    with pytest.raises(ValidationError, match="INCOMPLETE draft must contain is_incomplete=True"):
        ReportDraft(
            id="draft-err",
            assessment_id="asmt-err",
            run_number=1,
            outcome=AssessmentOutcome.INCOMPLETE,
            sections=scored_sections,
        )


def test_report_draft_hash_computation_and_tamper_detection() -> None:
    """Verify SHA-256 fingerprint computation and tamper rejection."""
    sections = _build_sample_sections(
        outcome=AssessmentOutcome.SCORED, score=30, band=RiskBand.MEDIUM
    )
    draft = ReportDraft(
        id="draft-003",
        assessment_id="asmt-003",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections,
    )

    computed = compute_draft_hash(
        assessment_id="asmt-003",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        risk_result_id=None,
        sections=sections,
    )
    assert draft.draft_hash == computed

    # If invalid declared hash is provided, validation fails
    with pytest.raises(ValidationError, match="Declared draft_hash"):
        ReportDraft(
            id="draft-003",
            assessment_id="asmt-003",
            run_number=1,
            outcome=AssessmentOutcome.SCORED,
            sections=sections,
            draft_hash="bad" * 16,
        )


def test_citation_and_claim_aggregation_properties() -> None:
    """Verify all_evidence_refs, all_policy_citation_refs, and all_claims rollups."""
    sections = _build_sample_sections(
        outcome=AssessmentOutcome.SCORED, score=30, band=RiskBand.MEDIUM
    )
    draft = ReportDraft(
        id="draft-004",
        assessment_id="asmt-004",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections,
    )

    assert "obs-ppsr-001" in draft.all_evidence_refs
    assert "obs-ident-001" in draft.all_evidence_refs
    assert "snap-ppsr:p001" in draft.all_policy_citation_refs
    assert "MATCH" in draft.all_risk_factor_refs
    assert len(draft.all_claims) >= 1
    assert draft.all_claims[0].claim_id == "claim-exec-01"


def test_synthetic_notice_inclusion() -> None:
    """Verify synthetic data notice section and property behavior."""
    sections = _build_sample_sections(
        outcome=AssessmentOutcome.SCORED,
        score=0,
        band=RiskBand.LOW,
        is_synthetic=True,
    )
    draft = ReportDraft(
        id="draft-005",
        assessment_id="asmt-005",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections,
    )

    assert draft.sections.synthetic_data_notice.is_synthetic is True
    assert draft.synthetic_notice is not None
    assert draft.synthetic_notice.is_synthetic is True
    assert "synthetic" in draft.synthetic_notice.notice_text.lower()


def test_abstention_notice_inclusion() -> None:
    """Verify policy abstention disclosure in PolicyCitationsSection and LimitationsSection."""
    abstention = AbstentionNotice(
        topic="Dispute Resolution Guidelines",
        reason="No policy passage met the 0.35 relevance threshold",
        impact="Citations omitted",
    )
    sections = _build_sample_sections(outcome=AssessmentOutcome.SCORED, score=0, band=RiskBand.LOW)
    # Update policy citations section with abstention
    policy_sec = PolicyCitationsSection(
        citations=(),
        has_abstention=True,
        abstention_notice=abstention,
    )
    lim_sec = LimitationsSection(
        abstention_notices=(abstention,),
    )
    new_sections = ReportSections(
        executive_summary=sections.executive_summary,
        vehicle_identity=sections.vehicle_identity,
        risk_score_and_band=sections.risk_score_and_band,
        mandatory_review_findings=sections.mandatory_review_findings,
        contributing_factors=sections.contributing_factors,
        policy_citations=policy_sec,
        evidence_summary=sections.evidence_summary,
        limitations_and_missing_evidence=lim_sec,
        synthetic_data_notice=sections.synthetic_data_notice,
    )
    draft = ReportDraft(
        id="draft-006",
        assessment_id="asmt-006",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=new_sections,
    )

    assert draft.abstention_notice is not None
    assert draft.abstention_notice.topic == "Dispute Resolution Guidelines"
    assert len(draft.sections.limitations_and_missing_evidence.abstention_notices) == 1


def test_immutability_of_report_models() -> None:
    """Verify that ReportDraft and sections reject in-place attribute mutations."""
    sections = _build_sample_sections(
        outcome=AssessmentOutcome.SCORED, score=30, band=RiskBand.MEDIUM
    )
    draft = ReportDraft(
        id="draft-007",
        assessment_id="asmt-007",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections,
    )

    with pytest.raises(ValidationError):
        draft.status = ReportDraftStatus.RELEASED

    with pytest.raises(ValidationError):
        draft.sections.risk_score_and_band.score = 50
