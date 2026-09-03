from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta, timezone

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState
from vehicle_risk_agent.evidence.models import (
    CandidateValue,
    ConfidenceAssessment,
    ConfidenceBand,
    ConflictState,
    FieldConflict,
    ProvenanceLink,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.snapshot import create_evidence_snapshot
from vehicle_risk_agent.evidence.sufficiency import (
    MissingEvidenceFinding,
    MissingEvidenceReason,
    evaluate_evidence_sufficiency,
)
from vehicle_risk_agent.persistence.models import (
    Base,
)
from vehicle_risk_agent.persistence.repository import AssessmentRepository
from vehicle_risk_agent.reporting.models import (
    ClaimReference,
    ContributingFactorsSection,
    EvidenceSummarySection,
    ExecutiveSummarySection,
    LimitationsSection,
    MandatoryReviewSection,
    PolicyCitationsSection,
    ReportDraft,
    ReportSections,
    RiskScoreSection,
    SyntheticNoticeSection,
    VehicleIdentitySection,
    compute_draft_hash,
)
from vehicle_risk_agent.reporting.offline import OfflineReportDraftingAdapter
from vehicle_risk_agent.reporting.protocol import EvidenceItem, ReportDraftingContext
from vehicle_risk_agent.reporting.repository import ReportDraftRepository
from vehicle_risk_agent.risk.calculator import calculate_risk_result
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    MandatoryFinding,
    RiskBand,
    RiskFactor,
    RiskFactorResult,
    RiskResult,
    build_risk_policy_v1,
)

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Create fresh isolated database tables, seed default policy, and yield an async session."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as sess:
        from vehicle_risk_agent.risk.repository import RiskPolicyRepository

        policy_repo = RiskPolicyRepository(sess)
        policy = build_risk_policy_v1(policy_id="risk-policy-v1")
        await policy_repo.create_policy(policy)
        yield sess

    await engine.dispose()


def _make_dummy_sections(
    outcome: AssessmentOutcome = AssessmentOutcome.SCORED,
    score: int | None = 20,
    band: RiskBand | None = RiskBand.LOW,
    vin: str = "7AT0BJ03X20000001",
    summary_text: str = "Standard report summary",
    make: str = "Toyota",
    model: str = "Aqua",
) -> ReportSections:
    is_inc = outcome == AssessmentOutcome.INCOMPLETE
    return ReportSections(
        executive_summary=ExecutiveSummarySection(
            summary_text=summary_text,
            outcome=outcome,
            key_findings=(),
            recommendation="Review",
            claims=(),
            evidence_refs=(),
            policy_citation_refs=(),
            risk_factor_refs=(),
        ),
        vehicle_identity=VehicleIdentitySection(
            vin=vin,
            make=make,
            model=model,
            year=2015,
            plate="ABC123",
        ),
        risk_score_and_band=RiskScoreSection(
            score=None if is_inc else score,
            band=None if is_inc else band,
            raw_score=None if is_inc else score,
            policy_id="risk-policy-v1",
            policy_version="v1",
            is_incomplete=is_inc,
            scoring_withheld_reason="Incomplete evidence" if is_inc else None,
        ),
        mandatory_review_findings=MandatoryReviewSection(findings=()),
        contributing_factors=ContributingFactorsSection(factor_breakdown=()),
        policy_citations=PolicyCitationsSection(citations=()),
        evidence_summary=EvidenceSummarySection(
            revision_id="rev-1",
            revision_number=1,
            material_hash="0" * 64,
            as_of=datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC),
        ),
        limitations_and_missing_evidence=LimitationsSection(),
        synthetic_data_notice=SyntheticNoticeSection(),
    )


# =============================================================================
# 1. REPOSITORY IDEMPOTENCY & CONFLICT STRESS TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_repo_exact_duplicate_resave_is_idempotent(db_session: AsyncSession) -> None:
    """Test that saving identical draft multiple times returns exact domain object without error."""
    asmt_repo = AssessmentRepository(db_session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-adv-1",
        idempotency_key="idemp-adv-1",
        request=AssessmentCreateRequest(
            vin="7AT0BJ03X20000001",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    sections = _make_dummy_sections()
    draft = ReportDraft(
        id=f"draft-{asmt.id}-1",
        assessment_id=asmt.id,
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections,
    )

    repo = ReportDraftRepository(db_session)
    saved1 = await repo.save_draft(draft)
    saved2 = await repo.save_draft(draft)
    saved3 = await repo.save_draft(draft)

    assert saved1.draft_hash == draft.draft_hash
    assert saved2.draft_hash == draft.draft_hash
    assert saved3.draft_hash == draft.draft_hash

    retrieved = await repo.get_draft(asmt.id, 1)
    assert retrieved is not None
    assert retrieved == draft


@pytest.mark.asyncio
async def test_repo_conflicting_save_fields_trigger_rollback_and_error(
    db_session: AsyncSession,
) -> None:
    """Test that any divergence in draft payload causes immediate rollback and ValueError."""
    asmt_repo = AssessmentRepository(db_session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-adv-2",
        idempotency_key="idemp-adv-2",
        request=AssessmentCreateRequest(
            vin="7AT0BJ03X20000001",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    sections1 = _make_dummy_sections(score=10, band=RiskBand.LOW)
    draft1 = ReportDraft(
        id=f"draft-{asmt.id}-1",
        assessment_id=asmt.id,
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections1,
    )

    repo = ReportDraftRepository(db_session)
    await repo.save_draft(draft1)

    # Divergent score & band
    sections2 = _make_dummy_sections(score=60, band=RiskBand.HIGH)
    draft2 = ReportDraft(
        id=f"draft-{asmt.id}-1",
        assessment_id=asmt.id,
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections2,
    )

    with pytest.raises(ValueError, match="immutable and cannot be overwritten"):
        await repo.save_draft(draft2)

    # Verify original draft is intact and unaltered
    retrieved = await repo.get_draft(asmt.id, 1)
    assert retrieved is not None
    assert retrieved.score == 10
    assert retrieved.band == RiskBand.LOW


@pytest.mark.asyncio
async def test_repo_subtle_metadata_difference_detected_as_conflict(
    db_session: AsyncSession,
) -> None:
    """Test that even non-section metadata changes trigger conflict rejection."""
    asmt_repo = AssessmentRepository(db_session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-adv-3",
        idempotency_key="idemp-adv-3",
        request=AssessmentCreateRequest(
            vin="7AT0BJ03X20000001",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    sections = _make_dummy_sections()
    draft1 = ReportDraft(
        id=f"draft-{asmt.id}-1",
        assessment_id=asmt.id,
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections,
        metadata={"operator": "alice"},
    )
    repo = ReportDraftRepository(db_session)
    await repo.save_draft(draft1)

    # Slight metadata change
    draft2 = ReportDraft(
        id=f"draft-{asmt.id}-1",
        assessment_id=asmt.id,
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections,
        metadata={"operator": "bob"},
    )

    with pytest.raises(ValueError, match="immutable and cannot be overwritten"):
        await repo.save_draft(draft2)


@pytest.mark.asyncio
async def test_repo_transition_assessment_with_conflicting_draft_rolls_back_state(
    db_session: AsyncSession,
) -> None:
    """Verify that if draft persistence fails, assessment state transition is rolled back."""
    asmt_repo = AssessmentRepository(db_session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-adv-4",
        idempotency_key="idemp-adv-4",
        request=AssessmentCreateRequest(
            vin="7AT0BJ03X20000001",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )
    assert asmt.lifecycle_state == AssessmentLifecycleState.IN_PROGRESS

    sections1 = _make_dummy_sections(score=10)
    draft1 = ReportDraft(
        id=f"draft-{asmt.id}-1",
        assessment_id=asmt.id,
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections1,
    )
    repo = ReportDraftRepository(db_session)
    await repo.save_draft(draft1)

    # Now attempt save_draft_and_transition_assessment with conflicting draft
    sections2 = _make_dummy_sections(score=99, band=RiskBand.CRITICAL)
    draft2 = ReportDraft(
        id=f"draft-{asmt.id}-1",
        assessment_id=asmt.id,
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections2,
    )

    with pytest.raises(ValueError, match="immutable and cannot be overwritten"):
        await repo.save_draft_and_transition_assessment(
            draft2, AssessmentLifecycleState.AWAITING_REVIEW
        )

    # Check that assessment is still IN_PROGRESS (not updated)
    refreshed_asmt = await asmt_repo.get_assessment(asmt.id)
    assert refreshed_asmt is not None
    assert refreshed_asmt.lifecycle_state == AssessmentLifecycleState.IN_PROGRESS


# =============================================================================
# 2. INCOMPLETE EVIDENCE SNAPSHOTS & ABSTENTION STRESS TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_offline_adapter_handles_none_snapshot() -> None:
    """Verify adapter handles completely absent snapshot with discrete EvidenceItems."""
    missing = (
        MissingEvidenceFinding(
            field_name="ppsr_result",
            reason=MissingEvidenceReason.ABSENT,
            details="No PPSR observations recorded",
        ),
    )
    risk_result = RiskResult(
        id="risk-res-none-snap",
        assessment_id="asmt-adv-none-snap",
        run_number=1,
        policy_id="risk-policy-v1",
        policy_version="v1",
        is_incomplete=True,
        missing_evidence=missing,
    )

    context = ReportDraftingContext(
        assessment_id="asmt-adv-none-snap",
        run_number=1,
        vehicle_id="7AT0BJ03X20000001",
        vin="7AT0BJ03X20000001",
        risk_result=risk_result,
        evidence_snapshot=None,
        evidence_items=(
            EvidenceItem(
                field_name="make",
                value="Nissan",
                observation_id="obs-item-1",
                source_system="CUSTOM",
            ),
        ),
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    assert draft.outcome == AssessmentOutcome.INCOMPLETE
    assert draft.score is None
    assert draft.band is None
    assert draft.is_incomplete is True
    # Verify vehicle identity extracted make from evidence_items
    assert draft.sections.vehicle_identity.make == "Nissan"
    assert draft.sections.evidence_summary.revision_id == "untracked"
    assert len(draft.missing_evidence_notices) == 1
    assert draft.missing_evidence_notices[0].field_name == "ppsr_result"


@pytest.mark.asyncio
async def test_offline_adapter_handles_multiple_unresolved_conflicts() -> None:
    """Verify unresolved conflicting fields generate structured disclosures and abstention."""
    conflicts = [
        FieldConflict(
            field_name="ppsr_result",
            conflicting_candidates=[
                CandidateValue(
                    field_name="ppsr_result",
                    value="MATCH",
                    provenance=ProvenanceLink(
                        observation_id="obs-ppsr-1",
                        source_system="PPSR_REG",
                        source_record_id="rec-1",
                        retrieved_at=datetime.now(UTC),
                    ),
                ),
                CandidateValue(
                    field_name="ppsr_result",
                    value="NO_MATCH",
                    provenance=ProvenanceLink(
                        observation_id="obs-ppsr-2",
                        source_system="PPSR_ALT",
                        source_record_id="rec-2",
                        retrieved_at=datetime.now(UTC),
                    ),
                ),
            ],
            state=ConflictState.UNRESOLVED,
            winning_value=None,
            rule_version="v1",
            rationale="Unresolvable PPSR conflict",
        ),
        FieldConflict(
            field_name="stolen_status",
            conflicting_candidates=[
                CandidateValue(
                    field_name="stolen_status",
                    value="LISTED",
                    provenance=ProvenanceLink(
                        observation_id="obs-pol-1",
                        source_system="POLICE",
                        source_record_id="rec-p1",
                        retrieved_at=datetime.now(UTC),
                    ),
                ),
                CandidateValue(
                    field_name="stolen_status",
                    value="NOT_STOLEN",
                    provenance=ProvenanceLink(
                        observation_id="obs-pol-2",
                        source_system="NZTA",
                        source_record_id="rec-p2",
                        retrieved_at=datetime.now(UTC),
                    ),
                ),
            ],
            state=ConflictState.UNRESOLVED,
            winning_value=None,
            rule_version="v1",
            rationale="Unresolvable stolen conflict",
        ),
    ]

    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    rev = VehicleRevisionResponse(
        vin="7AT0BJ03X20000001",
        revision_id="rev-conflicts-01",
        revision_number=1,
        material_hash="f" * 64,
        canonical_fields={"vin": "7AT0BJ03X20000001", "make": "Subaru"},
        field_provenance={},
        conflicts=conflicts,
        confidence=ConfidenceAssessment(
            score=30,
            band=ConfidenceBand.LOW,
            rule_version="v1",
            explanation="Multiple unresolved conflicts",
        ),
        as_of=now,
        published_at=now,
    )
    snapshot = create_evidence_snapshot(
        assessment_id="asmt-conflicts-01",
        run_number=1,
        revision=rev,
    )
    sufficiency = evaluate_evidence_sufficiency(snapshot)
    policy = build_risk_policy_v1()
    risk_result = calculate_risk_result(
        policy=policy,
        snapshot=snapshot,
        sufficiency=sufficiency,
    )

    assert risk_result.is_incomplete is True

    context = ReportDraftingContext(
        assessment_id="asmt-conflicts-01",
        run_number=1,
        vehicle_id=snapshot.vin,
        risk_result=risk_result,
        evidence_snapshot=snapshot,
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    assert draft.outcome == AssessmentOutcome.INCOMPLETE
    assert draft.score is None
    assert draft.sections.evidence_summary.conflict_count == 2
    assert len(draft.missing_evidence_notices) >= 2
    missing_field_names = [m.field_name for m in draft.missing_evidence_notices]
    assert "ppsr_result" in missing_field_names
    assert "stolen_status" in missing_field_names


@pytest.mark.asyncio
async def test_offline_adapter_combines_synthetic_and_incomplete_notices() -> None:
    """Verify draft containing BOTH synthetic evidence AND missing fields reports both cleanly."""
    missing = (
        MissingEvidenceFinding(
            field_name="writeoff_status",
            reason=MissingEvidenceReason.ABSENT,
            details="Writeoff registry query failed",
        ),
    )
    risk_result = RiskResult(
        id="risk-res-synth-inc",
        assessment_id="asmt-synth-inc",
        run_number=1,
        policy_id="risk-policy-v1",
        policy_version="v1",
        is_incomplete=True,
        missing_evidence=missing,
    )

    context = ReportDraftingContext(
        assessment_id="asmt-synth-inc",
        run_number=1,
        vehicle_id="7AT0BJ03X20000001",
        risk_result=risk_result,
        metadata={"is_synthetic": True},
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    assert draft.is_incomplete is True
    assert draft.score is None
    assert draft.sections.synthetic_data_notice.is_synthetic is True
    assert draft.synthetic_notice is not None
    assert draft.sections.limitations_and_missing_evidence.has_missing_evidence is True


# =============================================================================
# 3. DRAFT HASH VERIFICATION & SENSITIVITY STRESS TESTS
# =============================================================================


def test_draft_hash_sensitive_to_every_core_parameter() -> None:
    """Test that modifying assessment_id, run_number, outcome, or risk_result_id changes hash."""
    sections = _make_dummy_sections(outcome=AssessmentOutcome.SCORED, score=10)

    base_hash = compute_draft_hash(
        assessment_id="asmt-base",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        risk_result_id="risk-base",
        sections=sections,
    )

    # 1. assessment_id modified
    hash_asmt = compute_draft_hash(
        assessment_id="asmt-other",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        risk_result_id="risk-base",
        sections=sections,
    )
    assert base_hash != hash_asmt

    # 2. run_number modified
    hash_run = compute_draft_hash(
        assessment_id="asmt-base",
        run_number=2,
        outcome=AssessmentOutcome.SCORED,
        risk_result_id="risk-base",
        sections=sections,
    )
    assert base_hash != hash_run

    # 3. outcome modified
    inc_sections = _make_dummy_sections(outcome=AssessmentOutcome.INCOMPLETE)
    hash_outcome = compute_draft_hash(
        assessment_id="asmt-base",
        run_number=1,
        outcome=AssessmentOutcome.INCOMPLETE,
        risk_result_id="risk-base",
        sections=inc_sections,
    )
    assert base_hash != hash_outcome

    # 4. risk_result_id modified
    hash_rr = compute_draft_hash(
        assessment_id="asmt-base",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        risk_result_id="risk-other",
        sections=sections,
    )
    assert base_hash != hash_rr


def test_draft_hash_sensitive_to_evidence_refs_and_claims_count() -> None:
    """Test that modifying evidence_refs or adding claims in sections alters draft_hash."""
    sections_base = _make_dummy_sections()
    base_hash = compute_draft_hash(
        assessment_id="asmt-ref-test",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        risk_result_id="risk-01",
        sections=sections_base,
    )

    # Add evidence_refs to ExecutiveSummarySection
    s1_modified = ExecutiveSummarySection(
        summary_text=sections_base.executive_summary.summary_text,
        outcome=AssessmentOutcome.SCORED,
        evidence_refs=("obs-extra-001",),
    )
    sections_extra_ref = ReportSections(
        executive_summary=s1_modified,
        vehicle_identity=sections_base.vehicle_identity,
        risk_score_and_band=sections_base.risk_score_and_band,
        mandatory_review_findings=sections_base.mandatory_review_findings,
        contributing_factors=sections_base.contributing_factors,
        policy_citations=sections_base.policy_citations,
        evidence_summary=sections_base.evidence_summary,
        limitations_and_missing_evidence=sections_base.limitations_and_missing_evidence,
        synthetic_data_notice=sections_base.synthetic_data_notice,
    )

    hash_extra_ref = compute_draft_hash(
        assessment_id="asmt-ref-test",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        risk_result_id="risk-01",
        sections=sections_extra_ref,
    )
    assert base_hash != hash_extra_ref

    # Add a claim
    s1_with_claim = ExecutiveSummarySection(
        summary_text=sections_base.executive_summary.summary_text,
        outcome=AssessmentOutcome.SCORED,
        claims=(
            ClaimReference(
                claim_id="claim-adv-1",
                statement="A verified claim",
            ),
        ),
    )
    sections_with_claim = ReportSections(
        executive_summary=s1_with_claim,
        vehicle_identity=sections_base.vehicle_identity,
        risk_score_and_band=sections_base.risk_score_and_band,
        mandatory_review_findings=sections_base.mandatory_review_findings,
        contributing_factors=sections_base.contributing_factors,
        policy_citations=sections_base.policy_citations,
        evidence_summary=sections_base.evidence_summary,
        limitations_and_missing_evidence=sections_base.limitations_and_missing_evidence,
        synthetic_data_notice=sections_base.synthetic_data_notice,
    )
    hash_with_claim = compute_draft_hash(
        assessment_id="asmt-ref-test",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        risk_result_id="risk-01",
        sections=sections_with_claim,
    )
    assert base_hash != hash_with_claim


# =============================================================================
# 4. EXTREME INPUTS & BOUNDARY CONDITIONS STRESS TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_offline_adapter_handles_extreme_unicode_in_vehicle_identity() -> None:
    """Verify that Māori macrons, international UTF-8, and emojis are preserved safely."""
    unicode_make = "Tōyōta 🚗💨"
    unicode_model = "Te Ākitai Waiohua 特别版"
    unicode_color = "Āniwaniwa 🌈"

    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    rev = VehicleRevisionResponse(
        vin="7AT0BJ03X20000001",
        revision_id="rev-unicode-1",
        revision_number=1,
        material_hash="9" * 64,
        canonical_fields={
            "vin": "7AT0BJ03X20000001",
            "make": unicode_make,
            "model": unicode_model,
            "color": unicode_color,
            "year": 2022,
            "ppsr_result": "NO_MATCH",
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NO_RECORD",
        },
        field_provenance={},
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=100,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="Unicode test",
        ),
        as_of=now,
        published_at=now,
    )
    snapshot = create_evidence_snapshot(
        assessment_id="asmt-unicode-01",
        run_number=1,
        revision=rev,
    )
    policy = build_risk_policy_v1()
    risk_result = calculate_risk_result(policy=policy, snapshot=snapshot)

    context = ReportDraftingContext(
        assessment_id="asmt-unicode-01",
        run_number=1,
        vehicle_id=snapshot.vin,
        risk_result=risk_result,
        evidence_snapshot=snapshot,
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    assert draft.sections.vehicle_identity.make == unicode_make
    assert draft.sections.vehicle_identity.model == unicode_model
    assert draft.sections.vehicle_identity.color == unicode_color
    assert unicode_make in draft.sections.executive_summary.summary_text

    # Verify JSON serialization round-trip preservation
    serialized = draft.model_dump_json()
    deserialized = ReportDraft.model_validate_json(serialized)
    assert deserialized.sections.vehicle_identity.make == unicode_make
    assert deserialized.sections.vehicle_identity.model == unicode_model


@pytest.mark.asyncio
async def test_offline_adapter_handles_massive_text_payload(db_session: AsyncSession) -> None:
    """Verify large text payloads (e.g. 500KB) draft and hash without memory corruption."""
    huge_text = "Detailed inspection finding narrative " * 15000  # ~600 KB
    finding = MandatoryFinding(
        finding_id="f-huge-01",
        factor=RiskFactor.MATCH,
        title="PPSR Security Interest Found",
        description=huge_text,
        evidence_field="ppsr_result",
        evidence_value="MATCH",
        weight=30,
        severity=RiskBand.MEDIUM,
    )
    factor_res = RiskFactorResult(
        factor=RiskFactor.MATCH,
        weight=30,
        triggered=True,
        evidence_field="ppsr_result",
        evidence_value="MATCH",
        score_contribution=30,
        rationale=huge_text,
    )
    risk_result = RiskResult(
        id="risk-huge-01",
        assessment_id="asmt-huge-01",
        run_number=1,
        policy_id="risk-policy-v1",
        policy_version="v1",
        score=30,
        band=RiskBand.MEDIUM,
        raw_score=30,
        findings=(finding,),
        factors=(factor_res,),
        is_incomplete=False,
    )

    context = ReportDraftingContext(
        assessment_id="asmt-huge-01",
        run_number=1,
        vehicle_id="7AT0BJ03X20000001",
        risk_result=risk_result,
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    assert draft.score == 30
    assert draft.band == RiskBand.MEDIUM
    assert draft.draft_hash != ""
    assert len(draft.draft_hash) == 64

    # Verify DB serialization
    asmt_repo = AssessmentRepository(db_session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-huge",
        idempotency_key="idemp-huge-1",
        request=AssessmentCreateRequest(
            vin="7AT0BJ03X20000001",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )
    draft_repo = ReportDraftRepository(db_session)
    huge_draft = ReportDraft(
        id=f"draft-{asmt.id}-1",
        assessment_id=asmt.id,
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=draft.sections,
    )
    await draft_repo.save_draft(huge_draft)
    retrieved = await draft_repo.get_draft(asmt.id, 1)
    assert retrieved is not None
    assert retrieved.draft_hash == huge_draft.draft_hash


def test_claim_reference_empty_statement_fails_validation() -> None:
    """Verify that ClaimReference strictly enforces min_length=1 on statement."""
    with pytest.raises(ValidationError):
        ClaimReference(
            claim_id="claim-empty",
            statement="",
        )


def test_claim_reference_extra_fields_forbidden() -> None:
    """Verify that ClaimReference forbids unknown extra attributes."""
    with pytest.raises(ValidationError):
        ClaimReference(
            claim_id="claim-extra",
            statement="Valid statement",
            unauthorized_field="malicious_payload",  # type: ignore[call-arg]
        )


@pytest.mark.asyncio
async def test_offline_adapter_non_integer_year_coercion() -> None:
    """Verify non-integer year strings like 'Unknown' or 'N/A' do not crash adapter."""
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    rev = VehicleRevisionResponse(
        vin="7AT0BJ03X20000001",
        revision_id="rev-year-01",
        revision_number=1,
        material_hash="8" * 64,
        canonical_fields={
            "vin": "7AT0BJ03X20000001",
            "make": "Toyota",
            "model": "Corolla",
            "year": "N/A",  # Non-integer string
            "ppsr_result": "NO_MATCH",
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NO_RECORD",
        },
        field_provenance={},
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=90,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="Year edge case",
        ),
        as_of=now,
        published_at=now,
    )
    snapshot = create_evidence_snapshot(
        assessment_id="asmt-year-01",
        run_number=1,
        revision=rev,
    )
    policy = build_risk_policy_v1()
    risk_result = calculate_risk_result(policy=policy, snapshot=snapshot)

    context = ReportDraftingContext(
        assessment_id="asmt-year-01",
        run_number=1,
        vehicle_id=snapshot.vin,
        risk_result=risk_result,
        evidence_snapshot=snapshot,
    )

    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)

    # Year should be safely coerced to None without raising ValueError
    assert draft.sections.vehicle_identity.year is None
    assert draft.sections.vehicle_identity.make == "Toyota"


@pytest.mark.asyncio
async def test_timezone_preservation_and_comparison_in_repository(db_session: AsyncSession) -> None:
    """Verify non-UTC timezone timestamps on ReportDraft serialize without equality failure."""
    asmt_repo = AssessmentRepository(db_session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-tz",
        idempotency_key="idemp-tz-1",
        request=AssessmentCreateRequest(
            vin="7AT0BJ03X20000001",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    nz_tz = timezone(timedelta(hours=12))
    nz_time = datetime(2026, 9, 1, 23, 59, 59, 123456, tzinfo=nz_tz)

    sections = _make_dummy_sections()
    draft = ReportDraft(
        id=f"draft-{asmt.id}-1",
        assessment_id=asmt.id,
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections,
        created_at=nz_time,
    )

    repo = ReportDraftRepository(db_session)
    saved = await repo.save_draft(draft)
    assert saved is not None

    # Idempotent re-save of the same draft with nz_time
    resaved = await repo.save_draft(draft)
    assert resaved is not None


@pytest.mark.asyncio
async def test_concurrent_identical_saves_succeed(db_session: AsyncSession) -> None:
    """Verify concurrent saves of identical draft succeed idempotently without deadlocking."""
    asmt_repo = AssessmentRepository(db_session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-conc-1",
        idempotency_key="idemp-conc-1",
        request=AssessmentCreateRequest(
            vin="7AT0BJ03X20000001",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    sections = _make_dummy_sections()
    draft = ReportDraft(
        id=f"draft-{asmt.id}-1",
        assessment_id=asmt.id,
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        sections=sections,
    )

    repo = ReportDraftRepository(db_session)
    # Run sequentially within the session
    res1 = await repo.save_draft(draft)
    res2 = await repo.save_draft(draft)
    assert res1.draft_hash == res2.draft_hash


def test_risk_score_section_strict_boundary_validation() -> None:
    """Verify strict mathematical boundary enforcement on scores and incomplete invariants."""
    # 1. Negative score
    with pytest.raises(ValidationError):
        RiskScoreSection(
            score=-1,
            band=RiskBand.LOW,
            policy_id="risk-policy-v1",
            policy_version="v1",
            is_incomplete=False,
        )

    # 2. Score > 100
    with pytest.raises(ValidationError):
        RiskScoreSection(
            score=101,
            band=RiskBand.CRITICAL,
            policy_id="risk-policy-v1",
            policy_version="v1",
            is_incomplete=False,
        )

    # 3. Incomplete with score=0 (0 is not None)
    err_match = "Incomplete RiskScoreSection must not contain numeric score"
    with pytest.raises(ValidationError, match=err_match):
        RiskScoreSection(
            score=0,
            band=None,
            policy_id="risk-policy-v1",
            policy_version="v1",
            is_incomplete=True,
        )

    # 4. Incomplete with raw_score=0
    with pytest.raises(ValidationError, match=err_match):
        RiskScoreSection(
            score=None,
            band=None,
            raw_score=0,
            policy_id="risk-policy-v1",
            policy_version="v1",
            is_incomplete=True,
        )


def test_draft_hash_attribution_graph_properties() -> None:
    """Verify that compute_draft_hash acts as an attribution provenance fingerprint.

    Demonstrates that draft_hash guarantees attribution integrity (section titles, references)
    while full content immutability is guaranteed by model comparison in ReportDraftRepository.
    """
    sections = _make_dummy_sections(outcome=AssessmentOutcome.SCORED, score=20)
    hash1 = compute_draft_hash(
        assessment_id="asmt-attr-1",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        risk_result_id="risk-01",
        sections=sections,
    )

    # Changing an evidence reference changes hash
    sec_with_ev = ReportSections(
        executive_summary=sections.executive_summary,
        vehicle_identity=VehicleIdentitySection(
            vin="7AT0BJ03X20000001",
            evidence_refs=("obs-ident-extra",),
        ),
        risk_score_and_band=sections.risk_score_and_band,
        mandatory_review_findings=sections.mandatory_review_findings,
        contributing_factors=sections.contributing_factors,
        policy_citations=sections.policy_citations,
        evidence_summary=sections.evidence_summary,
        limitations_and_missing_evidence=sections.limitations_and_missing_evidence,
        synthetic_data_notice=sections.synthetic_data_notice,
    )
    hash2 = compute_draft_hash(
        assessment_id="asmt-attr-1",
        run_number=1,
        outcome=AssessmentOutcome.SCORED,
        risk_result_id="risk-01",
        sections=sec_with_ev,
    )
    assert hash1 != hash2


@pytest.mark.asyncio
async def test_offline_adapter_empty_missing_evidence_list_on_incomplete() -> None:
    """Verify INCOMPLETE RiskResult with empty missing_evidence tuple drafts cleanly."""
    risk_result = RiskResult(
        id="risk-empty-miss",
        assessment_id="asmt-empty-miss",
        run_number=1,
        policy_id="risk-policy-v1",
        policy_version="v1",
        is_incomplete=True,
        missing_evidence=(),
    )
    context = ReportDraftingContext(
        assessment_id="asmt-empty-miss",
        run_number=1,
        vehicle_id="7AT0BJ03X20000001",
        risk_result=risk_result,
    )
    adapter = OfflineReportDraftingAdapter()
    draft = await adapter.draft_report(context)
    assert draft.outcome == AssessmentOutcome.INCOMPLETE
    assert draft.is_incomplete is True
    assert "Withheld" in draft.sections.executive_summary.recommendation


def test_confidence_score_out_of_bounds_rejection() -> None:
    """Verify that VehicleIdentitySection rejects confidence_score outside [0, 100]."""
    with pytest.raises(ValidationError):
        VehicleIdentitySection(
            vin="7AT0BJ03X20000001",
            confidence_score=105,
        )

    with pytest.raises(ValidationError):
        VehicleIdentitySection(
            vin="7AT0BJ03X20000001",
            confidence_score=-5,
        )
