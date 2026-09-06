"""Integration tests for concurrent reinvestigation, authorization, and 3-run limit (e05s02-t03)."""

# story: e05s02

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.api.deps import intake_rate_limiter
from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import (
    AssessmentLifecycleState,
    AssessmentRunPhase,
)
from vehicle_risk_agent.persistence.models import (
    AssessmentRecord,
    AssessmentRunRecord,
    Base,
    ReviewActionRecord,
)
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
    """Helper to build valid ReportSections."""
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


async def _seed_reviewable_assessment(run_number: int = 1) -> str:
    """Directly seed a reviewable Assessment in AWAITING_REVIEW with a ReportDraft."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        policy_repo = RiskPolicyRepository(session)
        policy = build_risk_policy_v1(policy_id="risk-policy-v1", version="v1.0")
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

        if run_number > 1:
            asmt_rec = (
                await session.execute(
                    select(AssessmentRecord).where(AssessmentRecord.id == asmt.id)
                )
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

        draft_repo = ReportDraftRepository(session)
        draft = ReportDraft(
            id=f"draft-{uuid4()}",
            assessment_id=asmt.id,
            run_number=run_number,
            status=ReportDraftStatus.AWAITING_REVIEW,
            outcome=AssessmentOutcome.SCORED,
            sections=_build_dummy_sections(),
        )
        await draft_repo.save_draft_and_transition_assessment(
            draft, state=AssessmentLifecycleState.AWAITING_REVIEW
        )
        assessment_id = asmt.id

    await engine.dispose()
    return assessment_id


@pytest.mark.asyncio
async def test_reviewer_authorization_for_reinvestigation(app_client: AsyncClient) -> None:
    """Prove that only REVIEWER role can request reinvestigation."""
    assessment_id = await _seed_reviewable_assessment(run_number=1)

    payload = {
        "run_number": 1,
        "rationale": "Odometer disparity requires inspection recheck",
        "questions": ["Verify odometer against inspection records"],
        "evidence_targets": ["odometer_reading"],
    }

    # 1. Unauthenticated request -> 401
    resp_no_auth = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/reinvestigate",
        json=payload,
        headers={"Idempotency-Key": "key-reinv-01"},
    )
    assert resp_no_auth.status_code == 401

    # 2. REQUESTER role is forbidden -> 403
    resp_requester = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/reinvestigate",
        json=payload,
        headers={
            "Authorization": "Bearer dev-requester-token",
            "Idempotency-Key": "key-reinv-02",
        },
    )
    assert resp_requester.status_code == 403

    # 3. Missing idempotency key -> 400
    resp_no_idemp = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/reinvestigate",
        json=payload,
        headers={"Authorization": "Bearer dev-reviewer-token"},
    )
    assert resp_no_idemp.status_code == 400

    # 4. REVIEWER role is accepted -> 200
    resp_reviewer = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/reinvestigate",
        json=payload,
        headers={
            "Authorization": "Bearer dev-reviewer-token",
            "Idempotency-Key": "key-reinv-03",
        },
    )
    assert resp_reviewer.status_code == 200
    data = resp_reviewer.json()
    assert data["action_type"] == "REQUEST_REINVESTIGATION"
    assert data["disposition"] == "REINVESTIGATION_REQUESTED"
    assert data["assessment_state"] == "IN_PROGRESS"
    assert data["run_number"] == 1
    assert data["next_run_number"] == 2
    assert data["questions"] == ["Verify odometer against inspection records"]
    assert data["evidence_targets"] == ["odometer_reading"]


@pytest.mark.asyncio
async def test_reinvestigation_validation_errors(app_client: AsyncClient) -> None:
    """Prove strict validation for reinvestigation parameters."""
    assessment_id = await _seed_reviewable_assessment(run_number=1)
    headers = {
        "Authorization": "Bearer dev-reviewer-token",
        "Idempotency-Key": "key-val-01",
    }

    # Blank rationale -> 422
    resp = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/reinvestigate",
        json={
            "run_number": 1,
            "rationale": "   ",
            "questions": ["Check odometer"],
        },
        headers=headers,
    )
    assert resp.status_code == 422

    # Neither questions nor targets -> 422
    resp = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/reinvestigate",
        json={
            "run_number": 1,
            "rationale": "Valid rationale",
            "questions": [],
            "evidence_targets": [],
        },
        headers={**headers, "Idempotency-Key": "key-val-02"},
    )
    assert resp.status_code == 422

    # Disallowed evidence target -> 422
    resp = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/reinvestigate",
        json={
            "run_number": 1,
            "rationale": "Valid rationale",
            "evidence_targets": ["invalid_target_field"],
        },
        headers={**headers, "Idempotency-Key": "key-val-03"},
    )
    assert resp.status_code == 422

    # More than 5 questions -> 422
    resp = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/reinvestigate",
        json={
            "run_number": 1,
            "rationale": "Valid rationale",
            "questions": [f"Question {i}" for i in range(6)],
        },
        headers={**headers, "Idempotency-Key": "key-val-04"},
    )
    assert resp.status_code == 422

    # Question exceeds 200 chars -> 422
    resp = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/reinvestigate",
        json={
            "run_number": 1,
            "rationale": "Valid rationale",
            "questions": ["Q" * 201],
        },
        headers={**headers, "Idempotency-Key": "key-val-05"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_concurrent_reinvestigation_requests_yield_one_winner(
    app_client: AsyncClient,
) -> None:
    """Prove that simultaneous reinvestigation requests produce at most one allocated next run."""
    assessment_id = await _seed_reviewable_assessment(run_number=1)

    headers_base = {"Authorization": "Bearer dev-reviewer-token"}

    async def _send_reinvestigate(idx: int) -> int:
        headers = {**headers_base, "Idempotency-Key": f"concurrent-reinv-{idx}"}
        payload = {
            "run_number": 1,
            "rationale": f"Concurrent reinvestigate attempt {idx}",
            "evidence_targets": ["odometer_reading"],
        }
        resp = await app_client.post(
            f"/api/v1/assessments/{assessment_id}/review/reinvestigate",
            json=payload,
            headers=headers,
        )
        return resp.status_code

    tasks = [_send_reinvestigate(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # Exactly 1 winner exits 200; all 9 others return 409 Conflict
    winners = [code for code in results if code == 200]
    conflicts = [code for code in results if code == 409]

    assert len(winners) == 1, f"Expected exactly 1 winner, got {len(winners)}: {results}"
    assert len(conflicts) == 9, f"Expected 9 conflicts, got {len(conflicts)}: {results}"

    # Verify database state
    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        asmt_rec = (
            await session.execute(
                select(AssessmentRecord).where(AssessmentRecord.id == assessment_id)
            )
        ).scalar_one()
        assert asmt_rec.current_run_number == 2
        assert asmt_rec.lifecycle_state == AssessmentLifecycleState.IN_PROGRESS.value

        runs = (
            (
                await session.execute(
                    select(AssessmentRunRecord)
                    .where(AssessmentRunRecord.assessment_id == assessment_id)
                    .order_by(AssessmentRunRecord.run_number)
                )
            )
            .scalars()
            .all()
        )
        assert len(runs) == 2
        assert runs[1].run_number == 2

        actions = (
            (
                await session.execute(
                    select(ReviewActionRecord).where(
                        ReviewActionRecord.assessment_id == assessment_id
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(actions) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_reinvestigation_limit_reached_on_fourth_run(
    app_client: AsyncClient,
) -> None:
    """Prove that requesting reinvestigation from run 3 is rejected with limit error."""
    assessment_id = await _seed_reviewable_assessment(run_number=3)

    headers = {
        "Authorization": "Bearer dev-reviewer-token",
        "Idempotency-Key": "key-limit-run-3",
    }
    payload = {
        "run_number": 3,
        "rationale": "Attempting 4th run reinvestigation",
        "evidence_targets": ["stolen_status"],
    }

    resp = await app_client.post(
        f"/api/v1/assessments/{assessment_id}/review/reinvestigate",
        json=payload,
        headers=headers,
    )
    assert resp.status_code == 409
    error_data = resp.json()
    assert error_data["error"]["code"] == "REINVESTIGATION_LIMIT_REACHED"

    # Verify no mutation to assessment, runs, or actions
    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        asmt_rec = (
            await session.execute(
                select(AssessmentRecord).where(AssessmentRecord.id == assessment_id)
            )
        ).scalar_one()
        assert asmt_rec.current_run_number == 3
        assert asmt_rec.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW.value

        runs = (
            (
                await session.execute(
                    select(AssessmentRunRecord).where(
                        AssessmentRunRecord.assessment_id == assessment_id
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(runs) == 3

        actions = (
            (
                await session.execute(
                    select(ReviewActionRecord).where(
                        ReviewActionRecord.assessment_id == assessment_id,
                        ReviewActionRecord.run_number == 3,
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(actions) == 0
    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_idempotent_reinvestigation_replay(
    app_client: AsyncClient,
) -> None:
    """Prove that concurrent identical reinvestigation requests succeed."""
    assessment_id = await _seed_reviewable_assessment(run_number=1)

    headers = {
        "Authorization": "Bearer dev-reviewer-token",
        "Idempotency-Key": "exact-reinvest-replay-key",
    }
    payload = {
        "run_number": 1,
        "rationale": "Identical replay note for reinvestigation",
        "evidence_targets": ["odometer_reading"],
    }

    async def _send_req() -> tuple[int, str, int]:
        resp = await app_client.post(
            f"/api/v1/assessments/{assessment_id}/review/reinvestigate",
            json=payload,
            headers=headers,
        )
        data = resp.json()
        return resp.status_code, data["action_id"], data["next_run_number"]

    tasks = [_send_req() for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # All 5 return 200 and return the identical action_id and next_run_number == 2
    status_codes = [r[0] for r in results]
    action_ids = [r[1] for r in results]
    next_runs = [r[2] for r in results]

    assert all(code == 200 for code in status_codes)
    assert len(set(action_ids)) == 1, f"Expected 1 unique action_id, got {set(action_ids)}"
    assert all(nr == 2 for nr in next_runs)
