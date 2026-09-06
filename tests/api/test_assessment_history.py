"""API integration tests for Assessment history and authorization (e05s03-t02)."""

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
    PolicyCitationsSection,
    ReportDraft,
    ReportDraftStatus,
    ReportSections,
    RiskScoreSection,
    SyntheticNoticeSection,
    VehicleIdentitySection,
)
from vehicle_risk_agent.reporting.repository import ReportDraftRepository
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


def _build_dummy_sections() -> ReportSections:
    return ReportSections(
        executive_summary=ExecutiveSummarySection(
            summary_text="Executive summary text",
            outcome=AssessmentOutcome.SCORED,
        ),
        vehicle_identity=VehicleIdentitySection(
            vin="1HGCR2F85HA000000",
            make="HONDA",
            model="ACCORD",
            year=2017,
        ),
        risk_score_and_band=RiskScoreSection(
            score=35,
            band=RiskBand.LOW,
            raw_score=35,
            is_incomplete=False,
            policy_id="risk-policy-v1",
            policy_version="v1.0",
        ),
        mandatory_review_findings=MandatoryReviewSection(),
        contributing_factors=ContributingFactorsSection(),
        policy_citations=PolicyCitationsSection(),
        evidence_summary=EvidenceSummarySection(),
        limitations_and_missing_evidence=LimitationsSection(),
        synthetic_data_notice=SyntheticNoticeSection(),
    )


async def _seed_assessment_with_draft(requester_id: str) -> str:
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
            outcome=AssessmentOutcome.SCORED,
            sections=_build_dummy_sections(),
        )
        draft_repo = ReportDraftRepository(session)
        await draft_repo.save_draft_and_transition_assessment(
            draft, state=AssessmentLifecycleState.AWAITING_REVIEW
        )
        asmt_id = asmt.id

    await engine.dispose()
    return asmt_id


@pytest.mark.asyncio
async def test_unauthenticated_history_read_rejected(app_client: AsyncClient) -> None:
    asmt_id = await _seed_assessment_with_draft(requester_id="dev-requester")
    resp = await app_client.get(f"/api/v1/assessments/{asmt_id}/history")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_owner_authorized_history_read(app_client: AsyncClient) -> None:
    asmt_id = await _seed_assessment_with_draft(requester_id="principal-requester-1")
    resp = await app_client.get(
        f"/api/v1/assessments/{asmt_id}/history",
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["assessment_id"] == asmt_id
    assert data["requester_id"] == "principal-requester-1"
    assert data["vin"] == "1HGCR2F85HA000000"
    assert data["lifecycle_state"] == "AWAITING_REVIEW"
    assert data["disposition"] == "PENDING_REVIEW"
    assert len(data["runs"]) == 1
    assert data["runs"][0]["run_number"] == 1
    assert data["runs"][0]["draft"] is not None


@pytest.mark.asyncio
async def test_cross_owner_history_read_forbidden(app_client: AsyncClient) -> None:
    asmt_id = await _seed_assessment_with_draft(requester_id="other-requester")
    resp = await app_client.get(
        f"/api/v1/assessments/{asmt_id}/history",
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    assert resp.status_code == 403
    data = resp.json()
    assert data["error"]["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_reviewer_can_read_any_assessment_history(app_client: AsyncClient) -> None:
    asmt_id = await _seed_assessment_with_draft(requester_id="other-requester")
    resp = await app_client.get(
        f"/api/v1/assessments/{asmt_id}/history",
        headers={"Authorization": "Bearer dev-reviewer-token"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["assessment_id"] == asmt_id
    assert data["requester_id"] == "other-requester"


@pytest.mark.asyncio
async def test_other_roles_cannot_read_history(app_client: AsyncClient) -> None:
    asmt_id = await _seed_assessment_with_draft(requester_id="dev-requester")

    # Maintainer cannot read assessment history
    resp_maint = await app_client.get(
        f"/api/v1/assessments/{asmt_id}/history",
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    assert resp_maint.status_code == 403
    assert resp_maint.json()["error"]["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_non_existent_assessment_returns_404(app_client: AsyncClient) -> None:
    resp = await app_client.get(
        "/api/v1/assessments/non-existent-id/history",
        headers={"Authorization": "Bearer dev-reviewer-token"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"
