"""API integration tests for Released Report reads and invariants (e05s03-t03)."""

# story: e05s03

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.api.deps import intake_rate_limiter
from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState
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
from vehicle_risk_agent.review.models import (
    ApproveReportCommand,
    RejectReportCommand,
)
from vehicle_risk_agent.review.service import ReviewDecisionService
from vehicle_risk_agent.risk.models import AssessmentOutcome, RiskBand, build_risk_policy_v1
from vehicle_risk_agent.risk.repository import RiskPolicyRepository

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def app_client() -> AsyncIterator[AsyncClient]:
    """Provide an AsyncClient for FastAPI application with test database."""
    intake_rate_limiter.reset()
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    settings = Settings(database_url=TEST_DB_URL)
    app = create_app(settings=settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await engine.dispose()
    intake_rate_limiter.reset()


def _build_test_sections(incomplete: bool = False) -> ReportSections:
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
            policy_id="risk-policy-v1",
            policy_version="v1.0",
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


async def _seed_assessment(requester_id: str, incomplete: bool = False) -> tuple[str, str]:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        policy_repo = RiskPolicyRepository(session)
        policy = build_risk_policy_v1(policy_id="risk-policy-v1", version="v1.0")
        await policy_repo.create_policy(policy)

        asmt_repo = AssessmentRepository(session)
        asmt = await asmt_repo.create_assessment(
            requester_id=requester_id,
            idempotency_key=f"idemp-{uuid4()}",
            request=AssessmentCreateRequest(
                vin="1HGCR2F85HA000000",
                context=AssessmentContext(sale_type=SaleType.DEALER, intended_use="Commuting"),
            ),
        )

        draft = ReportDraft(
            id=f"draft-{uuid4()}",
            assessment_id=asmt.id,
            run_number=1,
            status=ReportDraftStatus.AWAITING_REVIEW,
            outcome=AssessmentOutcome.INCOMPLETE if incomplete else AssessmentOutcome.SCORED,
            sections=_build_test_sections(incomplete=incomplete),
        )
        draft_repo = ReportDraftRepository(session)
        await draft_repo.save_draft_and_transition_assessment(
            draft, state=AssessmentLifecycleState.AWAITING_REVIEW
        )
        asmt_id = asmt.id
        draft_id = draft.id

    await engine.dispose()
    return asmt_id, draft_id


async def _approve_assessment(
    asmt_id: str,
    incomplete: bool = False,
    notes: str | None = None,
    rationale: str | None = None,
) -> None:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        service = ReviewDecisionService(session)
        cmd = ApproveReportCommand(
            assessment_id=asmt_id,
            run_number=1,
            reviewer_id="principal-reviewer-1",
            idempotency_key=f"idemp-appr-{uuid4()}",
            notes=notes,
            acknowledge_missing_evidence=incomplete,
            rationale=rationale if incomplete else None,
            draft_outcome=AssessmentOutcome.INCOMPLETE if incomplete else AssessmentOutcome.SCORED,
        )
        await service.record_review_action(cmd)
    await engine.dispose()


@pytest.mark.asyncio
async def test_unauthenticated_report_read_rejected(app_client: AsyncClient) -> None:
    asmt_id, _ = await _seed_assessment(requester_id="principal-requester-1")
    resp = await app_client.get(f"/api/v1/assessments/{asmt_id}/report")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_owner_reads_released_scored_report(app_client: AsyncClient) -> None:
    asmt_id, draft_id = await _seed_assessment(requester_id="principal-requester-1")
    await _approve_assessment(asmt_id, incomplete=False, notes="Approved scored report")

    resp = await app_client.get(
        f"/api/v1/assessments/{asmt_id}/report",
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Draft invariants preserved
    report_draft = data["report_draft"]
    assert report_draft["id"] == draft_id
    assert report_draft["outcome"] == "SCORED"
    assert report_draft["sections"]["risk_score_and_band"]["score"] == 35
    assert report_draft["sections"]["risk_score_and_band"]["band"] == "LOW"

    # Review metadata present
    review_action = data["review_action"]
    assert review_action["action_type"] == "APPROVE_REPORT"
    assert review_action["reviewer_id"] == "principal-reviewer-1"
    assert review_action["notes"] == "Approved scored report"
    assert "released_at" in data


@pytest.mark.asyncio
async def test_owner_reads_released_incomplete_report(app_client: AsyncClient) -> None:
    asmt_id, draft_id = await _seed_assessment(
        requester_id="principal-requester-1", incomplete=True
    )
    await _approve_assessment(
        asmt_id,
        incomplete=True,
        notes="Awaiting manual odometer inspection",
        rationale="Odometer missing but identity confirmed via chassis",
    )

    resp = await app_client.get(
        f"/api/v1/assessments/{asmt_id}/report",
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Incomplete invariants preserved
    report_draft = data["report_draft"]
    assert report_draft["id"] == draft_id
    assert report_draft["outcome"] == "INCOMPLETE"
    assert report_draft["sections"]["risk_score_and_band"]["score"] is None
    assert report_draft["sections"]["risk_score_and_band"]["band"] is None
    assert (
        len(
            report_draft["sections"]["limitations_and_missing_evidence"]["missing_evidence_notices"]
        )
        == 1
    )

    # Review metadata preserved
    review_action = data["review_action"]
    assert review_action["acknowledge_missing_evidence"] is True
    assert review_action["rationale"] == "Odometer missing but identity confirmed via chassis"


@pytest.mark.asyncio
async def test_cross_owner_report_read_forbidden(app_client: AsyncClient) -> None:
    asmt_id, _ = await _seed_assessment(requester_id="other-requester")
    await _approve_assessment(asmt_id)

    resp = await app_client.get(
        f"/api/v1/assessments/{asmt_id}/report",
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_reviewer_reads_released_report(app_client: AsyncClient) -> None:
    asmt_id, draft_id = await _seed_assessment(requester_id="other-requester")
    await _approve_assessment(asmt_id)

    resp = await app_client.get(
        f"/api/v1/assessments/{asmt_id}/report",
        headers={"Authorization": "Bearer dev-reviewer-token"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["report_draft"]["id"] == draft_id


@pytest.mark.asyncio
async def test_unauthorized_roles_forbidden(app_client: AsyncClient) -> None:
    asmt_id, _ = await _seed_assessment(requester_id="principal-requester-1")
    await _approve_assessment(asmt_id)

    resp = await app_client.get(
        f"/api/v1/assessments/{asmt_id}/report",
        headers={"Authorization": "Bearer dev-operator-token"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_report_not_released_returns_404(app_client: AsyncClient) -> None:
    # 1. In AWAITING_REVIEW (not approved)
    asmt_id, _ = await _seed_assessment(requester_id="principal-requester-1")
    resp = await app_client.get(
        f"/api/v1/assessments/{asmt_id}/report",
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "REPORT_NOT_RELEASED"

    # 2. In REJECTED state
    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        service = ReviewDecisionService(session)
        cmd = RejectReportCommand(
            assessment_id=asmt_id,
            run_number=1,
            reviewer_id="principal-reviewer-1",
            idempotency_key="idemp-rej-01",
            rationale="Rejected due to invalid documents",
        )
        await service.record_review_action(cmd)
    await engine.dispose()

    resp_rej = await app_client.get(
        f"/api/v1/assessments/{asmt_id}/report",
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    assert resp_rej.status_code == 404
    assert resp_rej.json()["error"]["code"] == "REPORT_NOT_RELEASED"


@pytest.mark.asyncio
async def test_non_existent_assessment_returns_404(app_client: AsyncClient) -> None:
    resp = await app_client.get(
        "/api/v1/assessments/non-existent-id/report",
        headers={"Authorization": "Bearer dev-reviewer-token"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"
