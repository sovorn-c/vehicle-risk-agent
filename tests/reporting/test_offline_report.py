from datetime import UTC, datetime
from typing import Any

import pytest

from vehicle_risk_agent.evidence.models import (
    CandidateValue,
    ConfidenceAssessment,
    ConfidenceBand,
    ConflictState,
    FieldConflict,
    ProvenanceLink,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.snapshot import (
    create_evidence_snapshot,
)
from vehicle_risk_agent.evidence.sufficiency import (
    evaluate_evidence_sufficiency,
)
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.reporting.models import (
    ReportDraft,
    ReportDraftStatus,
    SectionType,
)
from vehicle_risk_agent.reporting.offline import OfflineReportDraftingAdapter
from vehicle_risk_agent.reporting.protocol import (
    ReportDraftingContext,
    ReportDraftingProtocol,
)
from vehicle_risk_agent.risk.calculator import calculate_risk_result
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    RiskBand,
    RiskFactor,
    build_risk_policy_v1,
)


def _make_sample_citation(passage_id: str, heading: str, keyword: str) -> PolicyCitation:
    return PolicyCitation(
        passage_id=passage_id,
        snapshot_id="snap-nzta-001",
        source_id="nzta-vrmr-2011",
        section_identifier="Clause 4.2",
        heading=heading,
        source_title="Land Transport (Vehicle Risk) Rule 2011",
        canonical_origin=f"https://www.nzta.govt.nz/rules/{keyword}",
    )


def _make_sample_snapshot(
    vin: str = "7AT0BJ03X20000001",
    ppsr_result: str = "NO_MATCH",
    stolen_status: str = "NOT_STOLEN",
    writeoff_status: str = "NO_RECORD",
    is_synthetic: bool = False,
    conflicts: list[FieldConflict] | None = None,
) -> tuple[VehicleRevisionResponse, Any]:
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    provenance: dict[str, tuple[ProvenanceLink, ...] | list[ProvenanceLink]] = {
        "vin": [
            ProvenanceLink(
                observation_id="obs-nzta-vin",
                source_system="NZTA",
                source_record_id="rec-vin",
                retrieved_at=now,
                synthetic=is_synthetic,
            )
        ],
        "make": [
            ProvenanceLink(
                observation_id="obs-nzta-make",
                source_system="NZTA",
                source_record_id="rec-make",
                retrieved_at=now,
                synthetic=is_synthetic,
            )
        ],
        "model": [
            ProvenanceLink(
                observation_id="obs-nzta-model",
                source_system="NZTA",
                source_record_id="rec-model",
                retrieved_at=now,
                synthetic=is_synthetic,
            )
        ],
        "year": [
            ProvenanceLink(
                observation_id="obs-nzta-year",
                source_system="NZTA",
                source_record_id="rec-year",
                retrieved_at=now,
                synthetic=is_synthetic,
            )
        ],
        "plate": [
            ProvenanceLink(
                observation_id="obs-nzta-plate",
                source_system="NZTA",
                source_record_id="rec-plate",
                retrieved_at=now,
                synthetic=is_synthetic,
            )
        ],
        "ppsr_result": [
            ProvenanceLink(
                observation_id="obs-ppsr-001",
                source_system="PPSR",
                source_record_id="rec-ppsr",
                retrieved_at=now,
                synthetic=is_synthetic,
            )
        ],
        "stolen_status": [
            ProvenanceLink(
                observation_id="obs-police-001",
                source_system="POLICE",
                source_record_id="rec-police",
                retrieved_at=now,
                synthetic=is_synthetic,
            )
        ],
        "writeoff_status": [
            ProvenanceLink(
                observation_id="obs-nzta-writeoff",
                source_system="NZTA",
                source_record_id="rec-writeoff",
                retrieved_at=now,
                synthetic=is_synthetic,
            )
        ],
    }

    revision = VehicleRevisionResponse(
        vin=vin,
        revision_id="rev-001",
        revision_number=1,
        material_hash="b" * 64,
        canonical_fields={
            "vin": vin,
            "make": "Mazda",
            "model": "Demio",
            "year": 2018,
            "plate": "NZDEM1",
            "ppsr_result": ppsr_result,
            "stolen_status": stolen_status,
            "writeoff_status": writeoff_status,
        },
        field_provenance=provenance,
        conflicts=conflicts or [],
        confidence=ConfidenceAssessment(
            score=95,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="High confidence from official registers",
            field_scores={"vin": 100, "ppsr_result": 95},
        ),
        as_of=datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC),
        published_at=datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC),
        synthetic_notice="DEMO SYNTHETIC DATA" if is_synthetic else None,
    )
    snapshot = create_evidence_snapshot(
        assessment_id="asmt-test-01",
        run_number=1,
        revision=revision,
    )
    return revision, snapshot


@pytest.mark.asyncio
async def test_offline_draft_clean_vehicle_low_risk() -> None:
    """Verify clean vehicle drafts a complete 9-section report with score=0 and LOW band."""
    _, snapshot = _make_sample_snapshot()
    policy = build_risk_policy_v1()
    citations = (_make_sample_citation("snap-ppsr:p01", "PPSR Security Interest Rules", "ppsr"),)
    risk_result = calculate_risk_result(
        policy=policy,
        snapshot=snapshot,
        policy_citations=citations,
    )
    assert risk_result.score == 0
    assert risk_result.band == RiskBand.LOW
    assert len(risk_result.findings) == 0

    context = ReportDraftingContext(
        assessment_id="asmt-test-01",
        run_number=1,
        vehicle_id=snapshot.vin,
        risk_result=risk_result,
        evidence_snapshot=snapshot,
        policy_citations=citations,
    )

    adapter = OfflineReportDraftingAdapter()
    assert isinstance(adapter, ReportDraftingProtocol)

    draft = await adapter.draft_report(context)

    assert isinstance(draft, ReportDraft)
    assert draft.assessment_id == "asmt-test-01"
    assert draft.run_number == 1
    assert draft.outcome == AssessmentOutcome.SCORED
    assert draft.score == 0
    assert draft.band == RiskBand.LOW
    assert draft.status == ReportDraftStatus.DRAFT
    assert len(draft.draft_hash) == 64

    # Verify 9 sections are present and valid
    sec_list = draft.sections.as_list()
    assert len(sec_list) == 9

    # Section 1: Executive Summary
    s1 = draft.sections.executive_summary
    assert s1.section_type == SectionType.EXECUTIVE_SUMMARY
    assert s1.outcome == AssessmentOutcome.SCORED
    assert "Mazda Demio" in s1.summary_text
    assert "LOW" in s1.summary_text

    # Section 2: Vehicle Identity
    s2 = draft.sections.vehicle_identity
    assert s2.vin == snapshot.vin
    assert s2.make == "Mazda"
    assert s2.model == "Demio"
    assert s2.year == 2018
    assert s2.plate == "NZDEM1"

    # Section 3: Risk Score & Band
    s3 = draft.sections.risk_score_and_band
    assert s3.score == 0
    assert s3.band == RiskBand.LOW
    assert s3.is_incomplete is False

    # Section 4: Mandatory Review Findings
    s4 = draft.sections.mandatory_review_findings
    assert s4.findings_count == 0
    assert s4.has_mandatory_findings is False

    # Section 5: Contributing Factors
    s5 = draft.sections.contributing_factors
    assert len(s5.factor_breakdown) == 4
    assert len(s5.triggered_factors) == 0

    # Section 6: Policy Citations
    s6 = draft.sections.policy_citations
    assert len(s6.citations) == 1
    assert s6.has_abstention is False

    # Section 7: Evidence Summary
    s7 = draft.sections.evidence_summary
    assert s7.revision_id == "rev-001"
    assert s7.canonical_fields["ppsr_result"] == "NO_MATCH"

    # Section 8: Limitations & Missing Evidence
    s8 = draft.sections.limitations_and_missing_evidence
    assert s8.has_missing_evidence is False
    assert len(s8.standard_limitations) >= 3

    # Section 9: Synthetic Data Notice
    s9 = draft.sections.synthetic_data_notice
    assert s9.is_synthetic is False


@pytest.mark.asyncio
async def test_offline_draft_encumbered_vehicle_ppsr_match() -> None:
    """Verify vehicle with PPSR security match drafts report with findings and citations."""
    _, snapshot = _make_sample_snapshot(ppsr_result="MATCH")
    policy = build_risk_policy_v1()
    citations = (
        _make_sample_citation("snap-ppsr:p01", "PPSR Registered Financing Statements", "ppsr"),
    )
    risk_result = calculate_risk_result(
        policy=policy,
        snapshot=snapshot,
        policy_citations=citations,
    )
    assert risk_result.score == 30
    assert risk_result.band == RiskBand.MEDIUM
    assert len(risk_result.findings) == 1

    context = ReportDraftingContext(
        assessment_id="asmt-test-02",
        run_number=1,
        vehicle_id=snapshot.vin,
        risk_result=risk_result,
        evidence_snapshot=snapshot,
        policy_citations=citations,
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    assert draft.score == 30
    assert draft.band == RiskBand.MEDIUM
    assert draft.sections.mandatory_review_findings.has_mandatory_findings is True
    assert draft.sections.mandatory_review_findings.findings_count == 1
    finding = draft.sections.mandatory_review_findings.findings[0]
    assert finding.factor == RiskFactor.MATCH
    assert "obs-ppsr-001" in finding.evidence_refs
    assert "snap-ppsr:p01" in finding.policy_citation_refs

    # Check that finding is summarized in Executive Summary
    assert len(draft.sections.executive_summary.key_findings) == 1
    assert "Security Interest" in draft.sections.executive_summary.key_findings[0]


@pytest.mark.asyncio
async def test_offline_draft_stolen_vehicle() -> None:
    """Verify vehicle listed as stolen drafts report with HIGH risk and police citations."""
    _, snapshot = _make_sample_snapshot(stolen_status="LISTED")
    policy = build_risk_policy_v1()
    citations = (
        _make_sample_citation("snap-police:p01", "Police Stolen Vehicle Operations", "stolen"),
    )
    risk_result = calculate_risk_result(
        policy=policy,
        snapshot=snapshot,
        policy_citations=citations,
    )
    assert risk_result.score == 45
    assert risk_result.band == RiskBand.HIGH

    context = ReportDraftingContext(
        assessment_id="asmt-test-03",
        run_number=1,
        vehicle_id=snapshot.vin,
        risk_result=risk_result,
        evidence_snapshot=snapshot,
        policy_citations=citations,
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    assert draft.score == 45
    assert draft.band == RiskBand.HIGH
    assert RiskFactor.LISTED in draft.all_risk_factor_refs
    assert "obs-police-001" in draft.all_evidence_refs


@pytest.mark.asyncio
async def test_offline_draft_statutory_writeoff() -> None:
    """Verify statutory write-off vehicle drafts report with STATUTORY factor."""
    _, snapshot = _make_sample_snapshot(writeoff_status="STATUTORY")
    policy = build_risk_policy_v1()
    citations = (
        _make_sample_citation("snap-nzta:p01", "Statutory Damage and Deregistration", "statutory"),
    )
    risk_result = calculate_risk_result(
        policy=policy,
        snapshot=snapshot,
        policy_citations=citations,
    )
    assert risk_result.score == 40
    assert risk_result.band == RiskBand.HIGH

    context = ReportDraftingContext(
        assessment_id="asmt-test-04",
        run_number=1,
        vehicle_id=snapshot.vin,
        risk_result=risk_result,
        evidence_snapshot=snapshot,
        policy_citations=citations,
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    assert draft.score == 40
    assert draft.band == RiskBand.HIGH
    assert RiskFactor.STATUTORY in draft.all_risk_factor_refs


@pytest.mark.asyncio
async def test_offline_draft_compound_factors() -> None:
    """Verify compound adverse factors (MATCH + STATUTORY = 70 points, CRITICAL band)."""
    _, snapshot = _make_sample_snapshot(ppsr_result="MATCH", writeoff_status="STATUTORY")
    policy = build_risk_policy_v1()
    citations = (
        _make_sample_citation("snap-ppsr:p01", "PPSR Security Interests", "ppsr"),
        _make_sample_citation("snap-nzta:p01", "Statutory Write-off Provisions", "statutory"),
    )
    risk_result = calculate_risk_result(
        policy=policy,
        snapshot=snapshot,
        policy_citations=citations,
    )
    assert risk_result.score == 70
    assert risk_result.band == RiskBand.CRITICAL
    assert len(risk_result.findings) == 2

    context = ReportDraftingContext(
        assessment_id="asmt-test-05",
        run_number=1,
        vehicle_id=snapshot.vin,
        risk_result=risk_result,
        evidence_snapshot=snapshot,
        policy_citations=citations,
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    assert draft.score == 70
    assert draft.band == RiskBand.CRITICAL
    assert draft.sections.mandatory_review_findings.findings_count == 2


@pytest.mark.asyncio
async def test_offline_draft_incomplete_evidence() -> None:
    """Verify INCOMPLETE evidence yields abstention draft withholding scores and showing gaps."""
    # Construct a snapshot with missing ppsr_result (conflict or missing)
    conflict = FieldConflict(
        field_name="ppsr_result",
        conflicting_candidates=[
            CandidateValue(
                field_name="ppsr_result",
                value="MATCH",
                provenance=ProvenanceLink(
                    observation_id="obs-ppsr-a",
                    source_system="PPSR_A",
                    source_record_id="rec-a",
                    retrieved_at=datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC),
                ),
            ),
            CandidateValue(
                field_name="ppsr_result",
                value="NO_MATCH",
                provenance=ProvenanceLink(
                    observation_id="obs-ppsr-b",
                    source_system="PPSR_B",
                    source_record_id="rec-b",
                    retrieved_at=datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC),
                ),
            ),
        ],
        state=ConflictState.UNRESOLVED,
        winning_value=None,
        rule_version="v1",
        rationale="Equal confidence conflict unresolved",
    )
    _, snapshot = _make_sample_snapshot(ppsr_result=None, conflicts=[conflict])  # type: ignore[arg-type]
    sufficiency = evaluate_evidence_sufficiency(snapshot)
    policy = build_risk_policy_v1()
    risk_result = calculate_risk_result(
        policy=policy,
        snapshot=snapshot,
        sufficiency=sufficiency,
    )
    assert risk_result.is_incomplete is True
    assert risk_result.score is None
    assert risk_result.band is None

    context = ReportDraftingContext(
        assessment_id="asmt-test-06",
        run_number=1,
        vehicle_id=snapshot.vin,
        risk_result=risk_result,
        evidence_snapshot=snapshot,
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    assert draft.outcome == AssessmentOutcome.INCOMPLETE
    assert draft.score is None
    assert draft.band is None
    assert draft.is_incomplete is True
    assert "WITHHELD" in draft.sections.executive_summary.summary_text.upper()
    assert draft.sections.limitations_and_missing_evidence.has_missing_evidence is True
    assert len(draft.missing_evidence_notices) >= 1
    assert draft.missing_evidence_notices[0].field_name == "ppsr_result"


@pytest.mark.asyncio
async def test_offline_draft_synthetic_evidence_triggers_notice() -> None:
    """Verify synthetic provenance flag triggers SyntheticNotice and disclaimer section."""
    _, snapshot = _make_sample_snapshot(is_synthetic=True)
    policy = build_risk_policy_v1()
    risk_result = calculate_risk_result(policy=policy, snapshot=snapshot)

    context = ReportDraftingContext(
        assessment_id="asmt-test-07",
        run_number=1,
        vehicle_id=snapshot.vin,
        risk_result=risk_result,
        evidence_snapshot=snapshot,
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    assert draft.sections.synthetic_data_notice.is_synthetic is True
    assert draft.synthetic_notice is not None
    assert draft.synthetic_notice.is_synthetic is True
    assert "synthetic" in draft.synthetic_notice.notice_text.lower()


@pytest.mark.asyncio
async def test_offline_draft_policy_abstention_handling() -> None:
    """Verify report handles policy retrieval abstention gracefully."""
    _, snapshot = _make_sample_snapshot()
    policy = build_risk_policy_v1()
    risk_result = calculate_risk_result(policy=policy, snapshot=snapshot)

    context = ReportDraftingContext(
        assessment_id="asmt-test-08",
        run_number=1,
        vehicle_id=snapshot.vin,
        risk_result=risk_result,
        evidence_snapshot=snapshot,
        policy_citations=(),
        metadata={"is_abstention": True},
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    assert draft.sections.policy_citations.has_abstention is True
    assert draft.abstention_notice is not None


@pytest.mark.asyncio
async def test_offline_drafting_is_purely_deterministic() -> None:
    """Verify drafting multiple times on identical context produces identical draft_hash."""
    _, snapshot = _make_sample_snapshot(ppsr_result="MATCH")
    policy = build_risk_policy_v1()
    citations = (
        _make_sample_citation("snap-ppsr:p01", "PPSR Registered Financing Statements", "ppsr"),
    )
    risk_result = calculate_risk_result(
        policy=policy,
        snapshot=snapshot,
        policy_citations=citations,
    )
    context = ReportDraftingContext(
        assessment_id="asmt-test-09",
        run_number=1,
        vehicle_id=snapshot.vin,
        risk_result=risk_result,
        evidence_snapshot=snapshot,
        policy_citations=citations,
    )

    adapter = OfflineReportDraftingAdapter()
    draft1 = await adapter.draft_report(context)
    draft2 = await adapter.draft_report(context)

    assert draft1.draft_hash == draft2.draft_hash
    assert draft1.sections.as_list() == draft2.sections.as_list()
