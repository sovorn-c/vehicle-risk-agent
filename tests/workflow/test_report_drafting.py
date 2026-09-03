"""Integration tests for Report Draft database persistence and LangGraph workflow."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.adapters.mcp import FakeVehicleMcpAdapter
from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState, AssessmentRunPhase
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    ProvenanceLink,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.persistence.models import Base
from vehicle_risk_agent.persistence.repository import AssessmentRepository
from vehicle_risk_agent.reporting.models import (
    ContributingFactorsSection,
    EvidenceSummarySection,
    ExecutiveSummarySection,
    LimitationsSection,
    MandatoryReviewSection,
    PolicyCitationsSection,
    ReportDraft,
    ReportDraftStatus,
    ReportSections,
    RiskScoreSection,
    SyntheticNoticeSection,
    VehicleIdentitySection,
)
from vehicle_risk_agent.reporting.repository import ReportDraftRepository
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    MandatoryFinding,
    RiskBand,
    RiskFactor,
    RiskFactorResult,
    build_risk_policy_v1,
)
from vehicle_risk_agent.risk.repository import RiskPolicyRepository
from vehicle_risk_agent.risk.service import RiskPolicyService
from vehicle_risk_agent.workflow.runner import AssessmentWorkflowRunner
from vehicle_risk_agent.workflow.state import AssessmentGraphState

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Provide a fresh database session with all tables created."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as sess:
        yield sess

    await engine.dispose()


def _build_test_draft(
    assessment_id: str,
    run_number: int = 1,
    vin: str = "1HGCR2F85HA000000",
    outcome: AssessmentOutcome = AssessmentOutcome.SCORED,
    score: int | None = 30,
    band: RiskBand | None = RiskBand.MEDIUM,
) -> ReportDraft:
    is_incomplete = outcome == AssessmentOutcome.INCOMPLETE

    s1 = ExecutiveSummarySection(
        summary_text="Vehicle risk assessment draft.",
        outcome=outcome,
        key_findings=("Registered security interest detected.",) if not is_incomplete else (),
        recommendation=(
            "Manual review required prior to release." if not is_incomplete else "Withheld"
        ),
    )
    s2 = VehicleIdentitySection(
        vin=vin,
        make="HONDA",
        model="ACCORD",
        year=2017,
        plate="NZACC1",
    )
    s3 = RiskScoreSection(
        score=None if is_incomplete else score,
        band=None if is_incomplete else band,
        raw_score=None if is_incomplete else score,
        policy_id="risk-policy-v1",
        policy_version="v1",
        is_incomplete=is_incomplete,
    )
    s4 = MandatoryReviewSection(
        findings=(
            MandatoryFinding(
                finding_id="f-01",
                factor=RiskFactor.MATCH,
                title="PPSR Match",
                description="Security interest found",
                evidence_field="ppsr_result",
                evidence_value="MATCH",
                weight=30,
                severity=RiskBand.MEDIUM,
            ),
        )
        if not is_incomplete
        else (),
    )
    s5 = ContributingFactorsSection(
        factor_breakdown=(
            RiskFactorResult(
                factor=RiskFactor.MATCH,
                weight=30,
                triggered=not is_incomplete,
                evidence_field="ppsr_result",
                evidence_value="MATCH" if not is_incomplete else None,
                score_contribution=30 if not is_incomplete else 0,
            ),
        ),
        triggered_factors=(RiskFactor.MATCH,) if not is_incomplete else (),
    )
    s6 = PolicyCitationsSection(citations=())
    s7 = EvidenceSummarySection(
        revision_id="rev-01",
        revision_number=1,
        material_hash="c" * 64,
        as_of=datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC),
        canonical_fields={"vin": vin, "ppsr_result": "MATCH" if not is_incomplete else None},
    )
    s8 = LimitationsSection()
    s9 = SyntheticNoticeSection()

    sections = ReportSections(
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

    return ReportDraft(
        id=f"draft-{assessment_id}-{run_number}",
        assessment_id=assessment_id,
        run_number=run_number,
        risk_result_id="risk-res-01",
        status=ReportDraftStatus.DRAFT,
        outcome=outcome,
        sections=sections,
    )


@pytest.mark.asyncio
async def test_report_draft_repository_save_and_retrieve(session: AsyncSession) -> None:
    """Persist a ReportDraft and retrieve it by assessment_id and run_number."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-1",
        idempotency_key="idemp-draft-1",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)

    draft = _build_test_draft(assessment_id=asmt.id, run_number=1)
    draft_repo = ReportDraftRepository(session)

    await draft_repo.save_draft(draft)

    retrieved = await draft_repo.get_draft(asmt.id, 1)
    assert retrieved is not None
    assert retrieved.id == draft.id
    assert retrieved.assessment_id == asmt.id
    assert retrieved.run_number == 1
    assert retrieved.outcome == AssessmentOutcome.SCORED
    assert retrieved.score == 30
    assert retrieved.band == RiskBand.MEDIUM
    assert retrieved.draft_hash == draft.draft_hash
    assert len(retrieved.sections.as_list()) == 9


@pytest.mark.asyncio
async def test_report_draft_repository_idempotent_resave(session: AsyncSession) -> None:
    """Re-saving an identical ReportDraft succeeds idempotently."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-2",
        idempotency_key="idemp-draft-2",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)

    draft = _build_test_draft(assessment_id=asmt.id, run_number=1)
    draft_repo = ReportDraftRepository(session)

    await draft_repo.save_draft(draft)
    # Re-save identical draft
    await draft_repo.save_draft(draft)

    retrieved = await draft_repo.get_draft(asmt.id, 1)
    assert retrieved is not None
    assert retrieved.draft_hash == draft.draft_hash


@pytest.mark.asyncio
async def test_report_draft_repository_immutability_violation_raises(
    session: AsyncSession,
) -> None:
    """Attempting to overwrite an existing ReportDraft with different content raises ValueError."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-3",
        idempotency_key="idemp-draft-3",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)

    draft1 = _build_test_draft(assessment_id=asmt.id, run_number=1, score=30, band=RiskBand.MEDIUM)
    draft2 = _build_test_draft(
        assessment_id=asmt.id, run_number=1, score=70, band=RiskBand.CRITICAL
    )

    draft_repo = ReportDraftRepository(session)
    await draft_repo.save_draft(draft1)

    with pytest.raises(ValueError, match="immutable and cannot be overwritten"):
        await draft_repo.save_draft(draft2)


@pytest.mark.asyncio
async def test_save_draft_and_transition_assessment_state(session: AsyncSession) -> None:
    """Saving draft transitions Assessment aggregate lifecycle_state to AWAITING_REVIEW."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-4",
        idempotency_key="idemp-draft-4",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )
    assert asmt.lifecycle_state == AssessmentLifecycleState.IN_PROGRESS

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)

    draft = _build_test_draft(assessment_id=asmt.id, run_number=1)
    draft_repo = ReportDraftRepository(session)

    await draft_repo.save_draft_and_transition_assessment(
        draft, AssessmentLifecycleState.AWAITING_REVIEW
    )

    updated_asmt = await asmt_repo.get_assessment(asmt.id)
    assert updated_asmt is not None
    assert updated_asmt.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW


@pytest.mark.asyncio
async def test_workflow_e2e_persists_report_draft_and_transitions_assessment(
    session: AsyncSession,
) -> None:
    """Full workflow run creates RiskResult, drafts ReportDraft, and moves to AWAITING_REVIEW."""
    # 1. Create Assessment and seed Active Policy
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-wf-1",
        idempotency_key="idemp-wf-1",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    # 2. Setup MCP adapter with clean vehicle facts
    mcp_adapter = FakeVehicleMcpAdapter()
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    rev = VehicleRevisionResponse(
        vin="1HGCR2F85HA000000",
        revision_id="rev-wf-1",
        revision_number=1,
        material_hash="d" * 64,
        canonical_fields={
            "vin": "1HGCR2F85HA000000",
            "make": "HONDA",
            "model": "ACCORD",
            "year": 2017,
            "plate": "NZACC1",
            "ppsr_result": "NO_FINANCE_REGISTERED",
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        },
        field_provenance={
            "vin": [
                ProvenanceLink(
                    observation_id="obs-vin",
                    source_system="NZTA",
                    source_record_id="rec-vin",
                    retrieved_at=now,
                )
            ],
            "ppsr_result": [
                ProvenanceLink(
                    observation_id="obs-ppsr",
                    source_system="PPSR",
                    source_record_id="rec-ppsr",
                    retrieved_at=now,
                )
            ],
            "stolen_status": [
                ProvenanceLink(
                    observation_id="obs-police",
                    source_system="POLICE",
                    source_record_id="rec-police",
                    retrieved_at=now,
                )
            ],
            "writeoff_status": [
                ProvenanceLink(
                    observation_id="obs-writeoff",
                    source_system="NZTA",
                    source_record_id="rec-writeoff",
                    retrieved_at=now,
                )
            ],
        },
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=95,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="verified",
        ),
        as_of=now,
        published_at=now,
    )
    mcp_adapter.seed_vehicle(rev)

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=TEST_DB_URL)

    async with AssessmentWorkflowRunner.create(
        settings=settings,
        session_factory=session_factory,
        mcp_adapter=mcp_adapter,
    ) as runner:
        initial_state: AssessmentGraphState = {
            "assessment_id": asmt.id,
            "run_number": 1,
            "vin": "1HGCR2F85HA000000",
            "context": asmt.context,
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }

        result = await runner.run(initial_state=initial_state)

        assert result["phase"] == AssessmentRunPhase.COMPLETED
        assert result.get("report_draft") is not None
        draft: ReportDraft = result["report_draft"]
        assert draft.outcome == AssessmentOutcome.SCORED
        assert draft.score == 0
        assert draft.band == RiskBand.LOW

        # Verify DB persistence
        draft_repo = ReportDraftRepository(session)
        stored_draft = await draft_repo.get_draft(asmt.id, 1)
        assert stored_draft is not None
        assert stored_draft.draft_hash == draft.draft_hash

        # Verify Assessment aggregate lifecycle state updated
        updated_asmt = await asmt_repo.get_assessment(asmt.id)
        assert updated_asmt is not None
        assert updated_asmt.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW

    await engine.dispose()


@pytest.mark.asyncio
async def test_workflow_e2e_incomplete_evidence_persists_incomplete_draft_and_transitions(
    session: AsyncSession,
) -> None:
    """Incomplete evidence run generates draft and transitions Assessment to AWAITING_REVIEW."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-wf-2",
        idempotency_key="idemp-wf-2",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    # Setup MCP adapter with missing ppsr_result
    mcp_adapter = FakeVehicleMcpAdapter()
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    rev = VehicleRevisionResponse(
        vin="1HGCR2F85HA000000",
        revision_id="rev-wf-2",
        revision_number=1,
        material_hash="e" * 64,
        canonical_fields={
            "vin": "1HGCR2F85HA000000",
            "make": "HONDA",
            "model": "ACCORD",
            "year": 2017,
            # ppsr_result missing!
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        },
        field_provenance={},
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=70,
            band=ConfidenceBand.MEDIUM,
            rule_version="v1",
            explanation="partial",
        ),
        as_of=now,
        published_at=now,
    )
    mcp_adapter.seed_vehicle(rev)

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=TEST_DB_URL)

    async with AssessmentWorkflowRunner.create(
        settings=settings,
        session_factory=session_factory,
        mcp_adapter=mcp_adapter,
    ) as runner:
        initial_state: AssessmentGraphState = {
            "assessment_id": asmt.id,
            "run_number": 1,
            "vin": "1HGCR2F85HA000000",
            "context": asmt.context,
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }

        result = await runner.run(initial_state=initial_state)

        assert result["phase"] == AssessmentRunPhase.INCOMPLETE
        assert result.get("report_draft") is not None
        draft: ReportDraft = result["report_draft"]
        assert draft.outcome == AssessmentOutcome.INCOMPLETE
        assert draft.score is None
        assert draft.band is None

        # Verify DB persistence
        draft_repo = ReportDraftRepository(session)
        stored_draft = await draft_repo.get_draft(asmt.id, 1)
        assert stored_draft is not None
        assert stored_draft.draft_hash == draft.draft_hash

        # Verify Assessment aggregate lifecycle state updated
        updated_asmt = await asmt_repo.get_assessment(asmt.id)
        assert updated_asmt is not None
        assert updated_asmt.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW

    await engine.dispose()
