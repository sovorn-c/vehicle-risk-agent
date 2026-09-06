"""Integration tests for ordered Assessment history and audit projections (e05s03-t01)."""

# story: e05s03

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.domain.assessment import (
    AssessmentLifecycleState,
)
from vehicle_risk_agent.persistence.models import (
    Base,
    PolicyCorpusRecord,
    PolicySnapshotRecord,
    PolicySourceRecord,
)
from vehicle_risk_agent.persistence.repository import AssessmentRepository
from vehicle_risk_agent.policy.corpus_lifecycle import CorpusLifecycleManager
from vehicle_risk_agent.policy.corpus_models import CorpusLifecycleState, RetrievalConfiguration
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
from vehicle_risk_agent.review.models import (
    ApproveReportCommand,
    RejectReportCommand,
    ReportDisposition,
    RequestReinvestigationCommand,
    ReviewActionType,
)
from vehicle_risk_agent.review.service import ReviewDecisionService
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    RiskBand,
    RiskPolicyLifecycleState,
    build_risk_policy_v1,
)
from vehicle_risk_agent.risk.repository import RiskPolicyRepository

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


def _build_test_sections(
    incomplete: bool = False,
    policy_id: str = "risk-policy-v1",
    policy_version: str = "v1.0",
) -> ReportSections:
    return ReportSections(
        executive_summary=ExecutiveSummarySection(
            summary_text="Executive summary text",
            outcome=AssessmentOutcome.INCOMPLETE if incomplete else AssessmentOutcome.SCORED,
        ),
        vehicle_identity=VehicleIdentitySection(
            vin="1HGCR2F85HA000000",
            make="HONDA",
            model="ACCORD",
            year=2017,
        ),
        risk_score_and_band=RiskScoreSection(
            score=None if incomplete else 35,
            band=None if incomplete else RiskBand.LOW,
            raw_score=None if incomplete else 35,
            is_incomplete=incomplete,
            policy_id=policy_id,
            policy_version=policy_version,
        ),
        mandatory_review_findings=MandatoryReviewSection(),
        contributing_factors=ContributingFactorsSection(),
        policy_citations=PolicyCitationsSection(),
        evidence_summary=EvidenceSummarySection(),
        limitations_and_missing_evidence=LimitationsSection(),
        synthetic_data_notice=SyntheticNoticeSection(),
    )


async def _seed_active_corpus(session: AsyncSession, corpus_id: str = "corpus-v1") -> None:
    source = PolicySourceRecord(
        id=f"src-{corpus_id}",
        title="NZTA Safety Guide",
        issuing_authority="NZTA",
        jurisdiction="NZ",
        canonical_origin="https://example.com/source",
        authority_classification="GOVERNMENT",
        reuse_terms="OPEN",
        expected_update_cadence="ANNUAL",
        status="ACTIVE",
    )
    session.add(source)
    snap = PolicySnapshotRecord(
        id=f"snap-{corpus_id}",
        source_id=source.id,
        content_hash="b" * 64,
        raw_content="Policy content for tests",
        parser_version="v1",
        validation_outcome="VALID",
        metadata_json="{}",
    )
    session.add(snap)
    manager = CorpusLifecycleManager(session)
    await manager.create_corpus(
        corpus_id=corpus_id,
        name="Active Test Corpus",
        description="Corpus for testing",
        snapshot_ids=[snap.id],
        retrieval_config=RetrievalConfiguration(),
    )
    record = await session.get(PolicyCorpusRecord, corpus_id)
    assert record is not None
    record.lifecycle_state = CorpusLifecycleState.ACTIVE.value
    await session.commit()


async def _seed_active_risk_policy(
    session: AsyncSession, policy_id: str = "risk-policy-v1", version: str = "v1"
) -> None:
    repo = RiskPolicyRepository(session)
    active = await repo.get_active_policy()
    if active is not None and active.id != policy_id:
        retired = active.model_copy(update={"lifecycle_state": RiskPolicyLifecycleState.RETIRED})
        await repo.update_policy(retired)
    elif active is not None and active.id == policy_id:
        return

    policy = build_risk_policy_v1(policy_id=policy_id, version=version).model_copy(
        update={
            "lifecycle_state": RiskPolicyLifecycleState.ACTIVE,
        }
    )
    await repo.create_policy(policy)


async def _setup_assessment(
    session: AsyncSession, requester_id: str = "req-user-1"
) -> tuple[str, str]:
    """Create an assessment at run 1 with a draft in AWAITING_REVIEW."""
    await _seed_active_risk_policy(session, policy_id="risk-policy-v1", version="v1.0")

    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id=requester_id,
        idempotency_key=f"idemp-{uuid4()}",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(
                sale_type=SaleType.DEALER,
                intended_use="Commuting",
                questions=["Inspect vehicle history"],
            ),
        ),
    )

    draft = ReportDraft(
        id=f"draft-{uuid4()}",
        assessment_id=asmt.id,
        run_number=1,
        status=ReportDraftStatus.AWAITING_REVIEW,
        outcome=AssessmentOutcome.SCORED,
        sections=_build_test_sections(),
    )
    draft_repo = ReportDraftRepository(session)
    await draft_repo.save_draft_and_transition_assessment(
        draft, AssessmentLifecycleState.AWAITING_REVIEW
    )
    return asmt.id, draft.id


class TestAssessmentHistoryProjection:
    """Test suite for domain projection of Assessment history (e05s03-t01)."""

    @pytest.mark.asyncio
    async def test_history_for_single_run_awaiting_review(self, session: AsyncSession) -> None:
        asmt_id, draft_id = await _setup_assessment(session)

        service = ReviewDecisionService(session)
        history = await service.get_assessment_history(asmt_id)

        assert history is not None
        assert history.assessment_id == asmt_id
        assert history.vin == "1HGCR2F85HA000000"
        assert history.requester_id == "req-user-1"
        assert history.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW
        assert history.current_run_number == 1
        assert history.disposition == ReportDisposition.PENDING_REVIEW
        assert history.released_report is None

        assert len(history.runs) == 1
        run1 = history.runs[0]
        assert run1.run_number == 1
        assert run1.draft is not None
        assert run1.draft.id == draft_id
        assert run1.evidence_summary is not None
        assert run1.review_action is None

    @pytest.mark.asyncio
    async def test_history_for_reinvestigated_assessment(self, session: AsyncSession) -> None:
        await _seed_active_corpus(session, corpus_id="corpus-v1")
        asmt_id, draft_1_id = await _setup_assessment(session)

        service = ReviewDecisionService(session)
        # Reviewer requests reinvestigation for run 1
        cmd = RequestReinvestigationCommand(
            assessment_id=asmt_id,
            run_number=1,
            reviewer_id="rev-alice",
            idempotency_key="idemp-reinvest-1",
            rationale="Need odometer re-verification",
            questions=("Verify odometer records",),
            evidence_targets=("odometer_reading",),
        )
        await service.record_review_action(cmd)

        # Now simulate run 2 producing a draft
        draft_2 = ReportDraft(
            id=f"draft-{uuid4()}",
            assessment_id=asmt_id,
            run_number=2,
            status=ReportDraftStatus.AWAITING_REVIEW,
            outcome=AssessmentOutcome.SCORED,
            sections=_build_test_sections(),
        )
        draft_repo = ReportDraftRepository(session)
        await draft_repo.save_draft_and_transition_assessment(
            draft_2, AssessmentLifecycleState.AWAITING_REVIEW
        )

        history = await service.get_assessment_history(asmt_id)

        assert history is not None
        assert history.current_run_number == 2
        assert history.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW
        assert len(history.runs) == 2

        # Run 1 verification
        run1 = history.runs[0]
        assert run1.run_number == 1
        assert run1.draft is not None
        assert run1.draft.id == draft_1_id
        assert run1.review_action is not None
        assert run1.review_action.action_type == ReviewActionType.REQUEST_REINVESTIGATION
        assert run1.review_action.rationale == "Need odometer re-verification"
        assert run1.review_action.questions == ("Verify odometer records",)

        # Run 2 verification
        run2 = history.runs[1]
        assert run2.run_number == 2
        assert run2.draft is not None
        assert run2.draft.id == draft_2.id
        assert run2.pinned_versions is not None
        assert run2.pinned_versions.corpus_id == "corpus-v1"
        assert run2.review_action is None

    @pytest.mark.asyncio
    async def test_history_for_approved_assessment(self, session: AsyncSession) -> None:
        asmt_id, draft_id = await _setup_assessment(session)

        service = ReviewDecisionService(session)
        cmd = ApproveReportCommand(
            assessment_id=asmt_id,
            run_number=1,
            reviewer_id="rev-bob",
            idempotency_key="idemp-appr-1",
            notes="Vehicle passes risk requirements",
        )
        await service.record_review_action(cmd)

        history = await service.get_assessment_history(asmt_id)

        assert history is not None
        assert history.lifecycle_state == AssessmentLifecycleState.RELEASED
        assert history.disposition == ReportDisposition.RELEASED
        assert history.released_report is not None
        assert history.released_report.report_draft.id == draft_id
        assert history.released_report.review_action.action_type == ReviewActionType.APPROVE_REPORT
        assert history.released_report.review_action.notes == "Vehicle passes risk requirements"

        assert len(history.runs) == 1
        assert history.runs[0].review_action is not None
        assert history.runs[0].review_action.action_type == ReviewActionType.APPROVE_REPORT

    @pytest.mark.asyncio
    async def test_history_for_rejected_assessment(self, session: AsyncSession) -> None:
        asmt_id, draft_id = await _setup_assessment(session)

        service = ReviewDecisionService(session)
        cmd = RejectReportCommand(
            assessment_id=asmt_id,
            run_number=1,
            reviewer_id="rev-carol",
            idempotency_key="idemp-rej-1",
            rationale="Unacceptable fraud risk detected on VIN plate",
        )
        await service.record_review_action(cmd)

        history = await service.get_assessment_history(asmt_id)

        assert history is not None
        assert history.lifecycle_state == AssessmentLifecycleState.REJECTED
        assert history.disposition == ReportDisposition.REJECTED
        assert history.released_report is None

        assert len(history.runs) == 1
        assert history.runs[0].review_action is not None
        assert history.runs[0].review_action.action_type == ReviewActionType.REJECT_REPORT
        assert (
            history.runs[0].review_action.rationale
            == "Unacceptable fraud risk detected on VIN plate"
        )

    @pytest.mark.asyncio
    async def test_history_for_non_existent_assessment(self, session: AsyncSession) -> None:
        service = ReviewDecisionService(session)
        history = await service.get_assessment_history("non-existent-id")
        assert history is None

    @pytest.mark.asyncio
    async def test_history_projection_does_not_mutate_state(self, session: AsyncSession) -> None:
        asmt_id, _ = await _setup_assessment(session)

        service = ReviewDecisionService(session)
        h1 = await service.get_assessment_history(asmt_id)
        h2 = await service.get_assessment_history(asmt_id)

        assert h1 is not None
        assert h2 is not None
        assert h1 == h2
