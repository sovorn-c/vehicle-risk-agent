"""Integration tests for transactional Reinvestigation and run allocation (e05s02-t02)."""

# story: e05s02

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState, AssessmentRunPhase
from vehicle_risk_agent.domain.errors import IdempotencyConflictError
from vehicle_risk_agent.persistence.models import (
    AssessmentRecord,
    AssessmentRunRecord,
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
from vehicle_risk_agent.review.errors import (
    AssessmentNotReviewableError,
)
from vehicle_risk_agent.review.models import (
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
    # Activate
    record = (
        await session.execute(select(PolicyCorpusRecord).where(PolicyCorpusRecord.id == corpus_id))
    ).scalar_one()
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


async def _setup_reviewable_assessment(
    session: AsyncSession, run_number: int = 1
) -> tuple[str, str]:
    """Helper to create an Assessment in AWAITING_REVIEW with a ReportDraft."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="req-user-1",
        idempotency_key=f"idemp-{uuid4()}",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(
                sale_type=SaleType.DEALER,
                intended_use="Personal",
                questions=["Check odometer integrity"],
            ),
        ),
    )

    # If run_number > 1, update assessment and add runs
    if run_number > 1:
        asmt_rec = (
            await session.execute(select(AssessmentRecord).where(AssessmentRecord.id == asmt.id))
        ).scalar_one()
        asmt_rec.current_run_number = run_number
        for r in range(2, run_number + 1):
            session.add(
                AssessmentRunRecord(
                    id=str(uuid4()),
                    assessment_id=asmt.id,
                    run_number=r,
                    phase=AssessmentRunPhase.PENDING.value,
                )
            )
        await session.commit()

    policy_repo = RiskPolicyRepository(session)
    active_policy = await policy_repo.get_active_policy()
    if active_policy is None:
        await _seed_active_risk_policy(session, policy_id="risk-policy-v1", version="v1.0")
        active_policy = await policy_repo.get_active_policy()

    assert active_policy is not None

    draft = ReportDraft(
        id=f"draft-{uuid4()}",
        assessment_id=asmt.id,
        run_number=run_number,
        status=ReportDraftStatus.AWAITING_REVIEW,
        outcome=AssessmentOutcome.SCORED,
        sections=_build_test_sections(
            policy_id=active_policy.id,
            policy_version=active_policy.version,
        ),
    )
    draft_repo = ReportDraftRepository(session)
    await draft_repo.save_draft_and_transition_assessment(
        draft, AssessmentLifecycleState.AWAITING_REVIEW
    )
    return asmt.id, draft.id


class TestReinvestigationAllocation:
    """Integration test suite for atomicity, run allocation, and version pinning."""

    @pytest.mark.asyncio
    async def test_reinvestigation_from_run_1_allocates_run_2(self, session: AsyncSession) -> None:
        await _seed_active_corpus(session, corpus_id="corpus-initial")
        await _seed_active_risk_policy(session, policy_id="policy-initial", version="v1.0")

        asmt_id, _ = await _setup_reviewable_assessment(session, run_number=1)

        cmd = RequestReinvestigationCommand(
            assessment_id=asmt_id,
            run_number=1,
            reviewer_id="rev-alice",
            idempotency_key="idemp-reinvest-1",
            rationale="Discrepancy in reported odometer readings",
            questions=("Verify odometer against latest inspection records",),
            evidence_targets=("odometer_reading",),
        )

        service = ReviewDecisionService(session)
        result = await service.record_review_action(cmd)

        assert result.action.action_type == ReviewActionType.REQUEST_REINVESTIGATION
        assert result.disposition == ReportDisposition.REINVESTIGATION_REQUESTED
        assert result.assessment_state == AssessmentLifecycleState.IN_PROGRESS
        assert result.action.run_number == 1
        assert result.next_run_number == 2
        assert result.action.questions == ("Verify odometer against latest inspection records",)
        assert result.action.evidence_targets == ("odometer_reading",)

        # Version pinning verification
        assert result.pinned_versions is not None
        assert result.pinned_versions.corpus_id == "corpus-initial"
        assert result.pinned_versions.risk_policy_id == "policy-initial"
        assert result.pinned_versions.risk_policy_version == "v1.0"

        # Verify DB records
        asmt_rec = (
            await session.execute(select(AssessmentRecord).where(AssessmentRecord.id == asmt_id))
        ).scalar_one()
        assert asmt_rec.current_run_number == 2
        assert asmt_rec.lifecycle_state == AssessmentLifecycleState.IN_PROGRESS.value
        assert asmt_rec.vin == "1HGCR2F85HA000000"  # VIN unchanged

        runs = (
            (
                await session.execute(
                    select(AssessmentRunRecord)
                    .where(AssessmentRunRecord.assessment_id == asmt_id)
                    .order_by(AssessmentRunRecord.run_number)
                )
            )
            .scalars()
            .all()
        )
        assert len(runs) == 2
        assert runs[1].run_number == 2
        assert runs[1].phase == AssessmentRunPhase.PENDING.value

    @pytest.mark.asyncio
    async def test_reinvestigation_from_run_2_pins_new_active_version(
        self, session: AsyncSession
    ) -> None:
        await _seed_active_corpus(session, corpus_id="corpus-v1")
        await _seed_active_risk_policy(session, policy_id="policy-v1", version="v1.0")

        asmt_id, _ = await _setup_reviewable_assessment(session, run_number=2)

        # Now simulate a version activation change before allocating run 3
        await _seed_active_risk_policy(session, policy_id="policy-v2", version="v2.0")

        cmd = RequestReinvestigationCommand(
            assessment_id=asmt_id,
            run_number=2,
            reviewer_id="rev-bob",
            idempotency_key="idemp-reinvest-2",
            rationale="Need stolen status recheck",
            evidence_targets=("stolen_status",),
        )

        service = ReviewDecisionService(session)
        result = await service.record_review_action(cmd)

        assert result.next_run_number == 3
        assert result.assessment_state == AssessmentLifecycleState.IN_PROGRESS
        assert result.pinned_versions is not None
        assert result.pinned_versions.risk_policy_id == "policy-v2"
        assert result.pinned_versions.risk_policy_version == "v2.0"

        asmt_rec = (
            await session.execute(select(AssessmentRecord).where(AssessmentRecord.id == asmt_id))
        ).scalar_one()
        assert asmt_rec.current_run_number == 3

    @pytest.mark.asyncio
    async def test_exact_idempotent_replay_returns_same_result(self, session: AsyncSession) -> None:
        await _seed_active_corpus(session, corpus_id="corpus-idem")
        await _seed_active_risk_policy(session, policy_id="policy-idem", version="v1.0")

        asmt_id, _ = await _setup_reviewable_assessment(session, run_number=1)

        cmd = RequestReinvestigationCommand(
            assessment_id=asmt_id,
            run_number=1,
            reviewer_id="rev-alice",
            idempotency_key="idemp-replay-key",
            rationale="Double check writeoff history",
            evidence_targets=("writeoff_status",),
        )

        service = ReviewDecisionService(session)
        res1 = await service.record_review_action(cmd)
        res2 = await service.record_review_action(cmd)

        assert res1.action.id == res2.action.id
        assert res1.action.action_hash == res2.action.action_hash
        assert res2.next_run_number == 2

        # Verify no duplicate run was allocated
        runs = (
            (
                await session.execute(
                    select(AssessmentRunRecord).where(AssessmentRunRecord.assessment_id == asmt_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(runs) == 2

    @pytest.mark.asyncio
    async def test_idempotency_conflict_with_modified_payload(self, session: AsyncSession) -> None:
        await _seed_active_corpus(session, corpus_id="corpus-conflict")
        await _seed_active_risk_policy(session, policy_id="policy-conflict", version="v1.0")

        asmt_id, _ = await _setup_reviewable_assessment(session, run_number=1)

        cmd1 = RequestReinvestigationCommand(
            assessment_id=asmt_id,
            run_number=1,
            reviewer_id="rev-alice",
            idempotency_key="shared-idemp-key",
            rationale="Initial rationale",
            evidence_targets=("ppsr_result",),
        )
        cmd2 = RequestReinvestigationCommand(
            assessment_id=asmt_id,
            run_number=1,
            reviewer_id="rev-alice",
            idempotency_key="shared-idemp-key",
            rationale="Altered rationale for conflict test",
            evidence_targets=("stolen_status",),
        )

        service = ReviewDecisionService(session)
        await service.record_review_action(cmd1)

        with pytest.raises(IdempotencyConflictError):
            await service.record_review_action(cmd2)

    @pytest.mark.asyncio
    async def test_rejects_reinvestigation_when_not_awaiting_review(
        self, session: AsyncSession
    ) -> None:
        asmt_id, _ = await _setup_reviewable_assessment(session, run_number=1)
        asmt_rec = (
            await session.execute(select(AssessmentRecord).where(AssessmentRecord.id == asmt_id))
        ).scalar_one()
        asmt_rec.lifecycle_state = AssessmentLifecycleState.IN_PROGRESS.value
        await session.commit()

        cmd = RequestReinvestigationCommand(
            assessment_id=asmt_id,
            run_number=1,
            reviewer_id="rev-alice",
            idempotency_key="idemp-err",
            rationale="Premature review",
            questions=("Not yet ready?",),
        )

        service = ReviewDecisionService(session)
        with pytest.raises(AssessmentNotReviewableError):
            await service.record_review_action(cmd)
