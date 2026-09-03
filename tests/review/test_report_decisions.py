"""Integration tests for transactional Review Decisions, row locking, and disposition."""

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState
from vehicle_risk_agent.domain.errors import IdempotencyConflictError
from vehicle_risk_agent.persistence.models import Base
from vehicle_risk_agent.persistence.repository import AssessmentRepository
from vehicle_risk_agent.reporting.models import (
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
    SyntheticNoticeSection,
    VehicleIdentitySection,
)
from vehicle_risk_agent.reporting.repository import ReportDraftRepository
from vehicle_risk_agent.review.errors import (
    AssessmentNotFoundError,
    AssessmentNotReviewableError,
    DraftNotFoundError,
    ReviewActionConflictError,
)
from vehicle_risk_agent.review.models import (
    ApproveReportCommand,
    RejectReportCommand,
    ReportDisposition,
    ReviewActionType,
)
from vehicle_risk_agent.review.service import ReviewDecisionService
from vehicle_risk_agent.risk.models import AssessmentOutcome, RiskBand, build_risk_policy_v1
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


def _build_dummy_sections(incomplete: bool = False) -> ReportSections:
    """Helper to build valid ReportSections for testing."""
    return ReportSections(
        executive_summary=ExecutiveSummarySection(
            summary_text="Executive summary",
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
            scoring_withheld_reason="Missing odometer" if incomplete else None,
        ),
        mandatory_review_findings=MandatoryReviewSection(),
        contributing_factors=ContributingFactorsSection(),
        policy_citations=PolicyCitationsSection(),
        evidence_summary=EvidenceSummarySection(),
        limitations_and_missing_evidence=LimitationsSection(
            missing_evidence_notices=(
                (
                    MissingEvidenceNotice(
                        field_name="odometer_reading",
                        reason="MISSING",
                        details="No odometer reading found",
                    ),
                )
                if incomplete
                else ()
            )
        ),
        synthetic_data_notice=SyntheticNoticeSection(),
    )


async def _setup_assessment_with_draft(
    session: AsyncSession,
    incomplete: bool = False,
    state: AssessmentLifecycleState = AssessmentLifecycleState.AWAITING_REVIEW,
) -> tuple[str, int]:
    """Helper to seed an active policy, assessment, and report draft."""
    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1()
    await policy_repo.create_policy(policy)

    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="req-user-1",
        idempotency_key=f"idemp-asmt-{uuid4()}",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER, intended_use="Commuting"),
        ),
    )

    draft_repo = ReportDraftRepository(session)
    draft = ReportDraft(
        id=f"draft-{uuid4()}",
        assessment_id=asmt.id,
        run_number=1,
        status=ReportDraftStatus.AWAITING_REVIEW,
        outcome=AssessmentOutcome.INCOMPLETE if incomplete else AssessmentOutcome.SCORED,
        sections=_build_dummy_sections(incomplete=incomplete),
    )
    await draft_repo.save_draft_and_transition_assessment(draft, state=state)
    return asmt.id, 1


@pytest.mark.asyncio
async def test_approve_scored_report_draft_happy_path(session: AsyncSession) -> None:
    """A reviewer approves a scored draft: transitions to RELEASED and derives disposition."""
    assessment_id, run_number = await _setup_assessment_with_draft(session, incomplete=False)

    service = ReviewDecisionService(session)
    cmd = ApproveReportCommand(
        assessment_id=assessment_id,
        run_number=run_number,
        reviewer_id="reviewer-alice",
        idempotency_key="idemp-app-01",
        notes="Approved without conditions",
    )

    result = await service.record_review_action(cmd)

    assert result.action.action_type == ReviewActionType.APPROVE_REPORT
    assert result.action.disposition == ReportDisposition.RELEASED
    assert result.action.reviewer_id == "reviewer-alice"
    assert result.action.notes == "Approved without conditions"
    assert result.assessment_state == AssessmentLifecycleState.RELEASED
    assert result.released_report is not None
    assert result.released_report.outcome == AssessmentOutcome.SCORED
    assert result.released_report.score == 35
    assert result.released_report.band == RiskBand.LOW
    assert result.released_report.vin == "1HGCR2F85HA000000"

    # Verify draft in database was NOT mutated
    draft_repo = ReportDraftRepository(session)
    draft_after = await draft_repo.get_draft(assessment_id, run_number)
    assert draft_after is not None
    assert draft_after.status == ReportDraftStatus.AWAITING_REVIEW
    assert draft_after.outcome == AssessmentOutcome.SCORED

    # Verify assessment aggregate state in database is RELEASED
    asmt_repo = AssessmentRepository(session)
    asmt_after = await asmt_repo.get_assessment(assessment_id)
    assert asmt_after is not None
    assert asmt_after.lifecycle_state == AssessmentLifecycleState.RELEASED


@pytest.mark.asyncio
async def test_approve_incomplete_report_draft_with_acknowledgement(
    session: AsyncSession,
) -> None:
    """Reviewer approves an INCOMPLETE draft with explicit acknowledgement and rationale."""
    assessment_id, run_number = await _setup_assessment_with_draft(session, incomplete=True)

    service = ReviewDecisionService(session)
    cmd = ApproveReportCommand(
        assessment_id=assessment_id,
        run_number=run_number,
        reviewer_id="reviewer-bob",
        idempotency_key="idemp-app-02",
        draft_outcome=AssessmentOutcome.INCOMPLETE,
        acknowledge_missing_evidence=True,
        rationale="Odometer not required for stationary evaluation",
    )

    result = await service.record_review_action(cmd)

    assert result.action.action_type == ReviewActionType.APPROVE_REPORT
    assert result.action.disposition == ReportDisposition.RELEASED
    assert result.assessment_state == AssessmentLifecycleState.RELEASED
    assert result.released_report is not None
    assert result.released_report.outcome == AssessmentOutcome.INCOMPLETE
    assert result.released_report.score is None
    assert result.released_report.band is None
    assert result.released_report.is_incomplete is True
    assert len(result.released_report.missing_evidence_notices) == 1


@pytest.mark.asyncio
async def test_approve_incomplete_without_acknowledgement_fails_closed(
    session: AsyncSession,
) -> None:
    """Attempting to approve an INCOMPLETE draft without acknowledgement fails closed."""
    assessment_id, run_number = await _setup_assessment_with_draft(session, incomplete=True)

    service = ReviewDecisionService(session)
    # Passed with draft_outcome SCORED default to bypass command validator, but service checks draft
    cmd = ApproveReportCommand(
        assessment_id=assessment_id,
        run_number=run_number,
        reviewer_id="reviewer-charlie",
        idempotency_key="idemp-app-03",
        acknowledge_missing_evidence=False,
    )

    with pytest.raises(ValueError, match="acknowledgement"):
        await service.record_review_action(cmd)

    # Verify state remains AWAITING_REVIEW
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.get_assessment(assessment_id)
    assert asmt is not None
    assert asmt.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW


@pytest.mark.asyncio
async def test_reject_report_draft_happy_path(session: AsyncSession) -> None:
    """Reviewer rejects a report draft with valid bounded rationale: transitions to REJECTED."""
    assessment_id, run_number = await _setup_assessment_with_draft(session, incomplete=False)

    service = ReviewDecisionService(session)
    cmd = RejectReportCommand(
        assessment_id=assessment_id,
        run_number=run_number,
        reviewer_id="reviewer-diana",
        idempotency_key="idemp-rej-01",
        rationale="Unresolvable identity conflicts in vehicle chassis data",
    )

    result = await service.record_review_action(cmd)

    assert result.action.action_type == ReviewActionType.REJECT_REPORT
    assert result.action.disposition == ReportDisposition.REJECTED
    assert result.action.rationale == "Unresolvable identity conflicts in vehicle chassis data"
    assert result.assessment_state == AssessmentLifecycleState.REJECTED
    assert result.released_report is None

    # Verify assessment aggregate state in database is REJECTED
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.get_assessment(assessment_id)
    assert asmt is not None
    assert asmt.lifecycle_state == AssessmentLifecycleState.REJECTED


@pytest.mark.asyncio
async def test_review_action_not_reviewable_state_rejected(session: AsyncSession) -> None:
    """Review commands on assessments NOT in AWAITING_REVIEW are rejected."""
    # Seed assessment with state IN_PROGRESS
    assessment_id, run_number = await _setup_assessment_with_draft(
        session, incomplete=False, state=AssessmentLifecycleState.IN_PROGRESS
    )

    service = ReviewDecisionService(session)
    cmd = ApproveReportCommand(
        assessment_id=assessment_id,
        run_number=run_number,
        reviewer_id="reviewer-eve",
        idempotency_key="idemp-app-04",
    )

    with pytest.raises(AssessmentNotReviewableError):
        await service.record_review_action(cmd)


@pytest.mark.asyncio
async def test_review_action_missing_assessment_or_draft_rejected(
    session: AsyncSession,
) -> None:
    """Review commands with non-existent assessment or draft fail cleanly."""
    service = ReviewDecisionService(session)

    cmd1 = ApproveReportCommand(
        assessment_id="non-existent-asmt",
        run_number=1,
        reviewer_id="reviewer-eve",
        idempotency_key="idemp-app-05",
    )
    with pytest.raises(AssessmentNotFoundError):
        await service.record_review_action(cmd1)

    assessment_id, _ = await _setup_assessment_with_draft(session, incomplete=False)
    cmd2 = ApproveReportCommand(
        assessment_id=assessment_id,
        run_number=99,  # non-existent run
        reviewer_id="reviewer-eve",
        idempotency_key="idemp-app-06",
    )
    with pytest.raises(DraftNotFoundError):
        await service.record_review_action(cmd2)


@pytest.mark.asyncio
async def test_review_action_exactly_once_enforcement(session: AsyncSession) -> None:
    """A draft can be decided exactly once; subsequent decisions return conflict."""
    assessment_id, run_number = await _setup_assessment_with_draft(session, incomplete=False)

    service = ReviewDecisionService(session)
    cmd1 = ApproveReportCommand(
        assessment_id=assessment_id,
        run_number=run_number,
        reviewer_id="reviewer-frank",
        idempotency_key="idemp-01",
    )
    await service.record_review_action(cmd1)

    # Attempting second decision with new idempotency key
    cmd2 = RejectReportCommand(
        assessment_id=assessment_id,
        run_number=run_number,
        reviewer_id="reviewer-frank",
        idempotency_key="idemp-02",
        rationale="Trying to reject already released report",
    )
    with pytest.raises(ReviewActionConflictError):
        await service.record_review_action(cmd2)


@pytest.mark.asyncio
async def test_idempotent_command_replay(session: AsyncSession) -> None:
    """Exact command replay with same idempotency key returns original decision."""
    assessment_id, run_number = await _setup_assessment_with_draft(session, incomplete=False)

    service = ReviewDecisionService(session)
    cmd = ApproveReportCommand(
        assessment_id=assessment_id,
        run_number=run_number,
        reviewer_id="reviewer-grace",
        idempotency_key="idemp-grace-01",
        notes="Initial approval note",
    )

    first_result = await service.record_review_action(cmd)
    replay_result = await service.record_review_action(cmd)

    assert first_result.action.id == replay_result.action.id
    assert first_result.action.action_hash == replay_result.action.action_hash
    assert first_result.assessment_state == replay_result.assessment_state

    # Replay with modified payload raises IdempotencyConflictError
    conflicting_cmd = ApproveReportCommand(
        assessment_id=assessment_id,
        run_number=run_number,
        reviewer_id="reviewer-grace",
        idempotency_key="idemp-grace-01",
        notes="DIFFERENT NOTE",
    )
    with pytest.raises(IdempotencyConflictError):
        await service.record_review_action(conflicting_cmd)


@pytest.mark.asyncio
async def test_cross_type_command_action_type_mismatch_fails_closed(
    session: AsyncSession,
) -> None:
    """Service fails closed if command action_type does not match its class."""
    assessment_id, run_number = await _setup_assessment_with_draft(session, incomplete=False)
    service = ReviewDecisionService(session)

    cmd = RejectReportCommand(
        assessment_id=assessment_id,
        run_number=run_number,
        reviewer_id="reviewer-mallory",
        idempotency_key="idemp-mallory-01",
        rationale="Trying to act like an approval",
    )
    # Even if action_type was somehow set to APPROVE_REPORT on RejectReportCommand:
    object.__setattr__(cmd, "action_type", ReviewActionType.APPROVE_REPORT)
    with pytest.raises(ValueError, match="action_type"):
        await service.record_review_action(cmd)


@pytest.mark.asyncio
async def test_cross_type_reject_cannot_bypass_incomplete_draft_validation(
    session: AsyncSession,
) -> None:
    """A RejectReportCommand with APPROVE_REPORT cannot bypass incomplete-draft acknowledgement."""
    assessment_id, run_number = await _setup_assessment_with_draft(session, incomplete=True)
    service = ReviewDecisionService(session)

    cmd = RejectReportCommand(
        assessment_id=assessment_id,
        run_number=run_number,
        reviewer_id="reviewer-mallory",
        idempotency_key="idemp-mallory-02",
        rationale="Bypass attempt",
    )
    object.__setattr__(cmd, "action_type", ReviewActionType.APPROVE_REPORT)
    with pytest.raises(ValueError, match="action_type"):
        await service.record_review_action(cmd)
