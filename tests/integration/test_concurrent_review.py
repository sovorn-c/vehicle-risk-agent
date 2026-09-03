"""Integration tests for concurrent review decisions, reviewer authorization, and safe conflicts."""

# story: e05s01

import asyncio
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


def _build_dummy_sections(incomplete: bool = False) -> ReportSections:
    """Helper to build valid ReportSections."""
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


async def _seed_reviewable_assessment(incomplete: bool = False) -> str:
    """Directly seed a reviewable Assessment in AWAITING_REVIEW with a ReportDraft."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
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
        await draft_repo.save_draft_and_transition_assessment(
            draft, state=AssessmentLifecycleState.AWAITING_REVIEW
        )
        assessment_id = asmt.id

    await engine.dispose()
    return assessment_id


@pytest.mark.asyncio
async def test_reviewer_authorization_enforced(app_client: AsyncClient) -> None:
    """Prove that only REVIEWER role can execute review actions."""
    assessment_id = await _seed_reviewable_assessment(incomplete=False)

    # 1. Unauthenticated request
    resp_no_auth = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/approve",
        json={"run_number": 1, "notes": "No auth test"},
        headers={"Idempotency-Key": "key-01"},
    )
    assert resp_no_auth.status_code == 401

    # 2. REQUESTER role is forbidden
    resp_requester = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/approve",
        json={"run_number": 1, "notes": "Requester trying to approve"},
        headers={
            "Authorization": "Bearer dev-requester-token",
            "Idempotency-Key": "key-02",
        },
    )
    assert resp_requester.status_code == 403

    # 3. Missing idempotency key
    resp_no_idemp = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/approve",
        json={"run_number": 1, "notes": "No idemp key"},
        headers={"Authorization": "Bearer dev-reviewer-token"},
    )
    assert resp_no_idemp.status_code == 400

    # 4. REVIEWER role is accepted
    resp_reviewer = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/approve",
        json={"run_number": 1, "notes": "Reviewer valid approval"},
        headers={
            "Authorization": "Bearer dev-reviewer-token",
            "Idempotency-Key": "key-03",
        },
    )
    assert resp_reviewer.status_code == 200
    data = resp_reviewer.json()
    assert data["disposition"] == "RELEASED"
    assert data["assessment_state"] == "RELEASED"
    assert data["action_type"] == "APPROVE_REPORT"


@pytest.mark.asyncio
async def test_reviewer_reject_requires_rationale(app_client: AsyncClient) -> None:
    """Prove that REJECT_REPORT requires non-empty rationale."""
    assessment_id = await _seed_reviewable_assessment(incomplete=False)

    # Missing rationale in request
    resp_no_rat = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/reject",
        json={"run_number": 1, "rationale": "   "},
        headers={
            "Authorization": "Bearer dev-reviewer-token",
            "Idempotency-Key": "key-rej-01",
        },
    )
    assert resp_no_rat.status_code == 422

    # Valid rationale
    resp_valid = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/reject",
        json={"run_number": 1, "rationale": "Evidence conflict in chassis numbers"},
        headers={
            "Authorization": "Bearer dev-reviewer-token",
            "Idempotency-Key": "key-rej-02",
        },
    )
    assert resp_valid.status_code == 200
    data = resp_valid.json()
    assert data["disposition"] == "REJECTED"
    assert data["assessment_state"] == "REJECTED"


@pytest.mark.asyncio
async def test_concurrent_review_actions_yield_one_winner(app_client: AsyncClient) -> None:
    """Prove that simultaneous review decisions produce one winner and safe conflicts."""
    assessment_id = await _seed_reviewable_assessment(incomplete=False)

    # 10 concurrent requests: 5 approvals and 5 rejections with unique idempotency keys
    headers_base = {"Authorization": "Bearer dev-reviewer-token"}

    async def _send_approve(idx: int) -> int:
        headers = {**headers_base, "Idempotency-Key": f"concurrent-app-{idx}"}
        payload = {"run_number": 1, "notes": f"Concurrent approve attempt {idx}"}
        resp = await app_client.post(
            f"/api/v1/assessments/{assessment_id}/review/approve",
            json=payload,
            headers=headers,
        )
        return resp.status_code

    async def _send_reject(idx: int) -> int:
        headers = {**headers_base, "Idempotency-Key": f"concurrent-rej-{idx}"}
        payload = {"run_number": 1, "rationale": f"Concurrent reject attempt {idx}"}
        resp = await app_client.post(
            f"/api/v1/assessments/{assessment_id}/review/reject",
            json=payload,
            headers=headers,
        )
        return resp.status_code

    tasks = []
    for i in range(5):
        tasks.append(_send_approve(i))
        tasks.append(_send_reject(i))

    results = await asyncio.gather(*tasks)

    # Exactly ONE winner exits 200; all 9 others return 409 Conflict
    winners = [code for code in results if code == 200]
    conflicts = [code for code in results if code == 409]

    assert len(winners) == 1, f"Expected exactly 1 winner, got {len(winners)}: {results}"
    assert len(conflicts) == 9, f"Expected 9 conflicts, got {len(conflicts)}: {results}"


@pytest.mark.asyncio
async def test_concurrent_idempotent_replay(app_client: AsyncClient) -> None:
    """Prove that concurrent identical replay requests with the same idempotency key all succeed."""
    assessment_id = await _seed_reviewable_assessment(incomplete=False)

    headers = {
        "Authorization": "Bearer dev-reviewer-token",
        "Idempotency-Key": "exact-replay-key",
    }
    payload = {"run_number": 1, "notes": "Identical replay note"}

    async def _send_req() -> tuple[int, str]:
        resp = await app_client.post(
            f"/api/v1/assessments/{assessment_id}/review/approve",
            json=payload,
            headers=headers,
        )
        return resp.status_code, resp.json()["action_id"]

    tasks = [_send_req() for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # All 5 return 200 and return the identical action_id
    status_codes = [r[0] for r in results]
    action_ids = [r[1] for r in results]

    assert all(code == 200 for code in status_codes)
    assert len(set(action_ids)) == 1, f"Expected 1 unique action_id, got {set(action_ids)}"
