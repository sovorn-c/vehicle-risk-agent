"""Integration tests for Risk Policy and Risk Result persistence, idempotency, and replay."""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.evidence.sufficiency import (
    MissingEvidenceFinding,
    MissingEvidenceReason,
)
from vehicle_risk_agent.persistence.models import Base
from vehicle_risk_agent.persistence.repository import AssessmentRepository
from vehicle_risk_agent.risk.models import (
    MandatoryFinding,
    RiskBand,
    RiskFactor,
    RiskFactorResult,
    RiskPolicyLifecycleState,
    RiskResult,
    build_risk_policy_v1,
)
from vehicle_risk_agent.risk.repository import (
    RiskPolicyRepository,
    RiskResultRepository,
)
from vehicle_risk_agent.risk.service import RiskPolicyService

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


@pytest.mark.asyncio
async def test_persist_and_retrieve_scored_risk_result(session: AsyncSession) -> None:
    """Persist a complete scored RiskResult and retrieve it with all fields intact."""
    # 1. Create Assessment and Policy
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-1",
        idempotency_key="idemp-key-1",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER, intended_use="Commuting"),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="policy-scored-1")
    await policy_repo.create_policy(policy)

    # 2. Construct and save RiskResult
    result_repo = RiskResultRepository(session)
    result = RiskResult(
        assessment_id=asmt.id,
        run_number=1,
        policy_id=policy.id,
        policy_version=policy.version,
        score=70,
        band=RiskBand.CRITICAL,
        raw_score=70,
        is_incomplete=False,
        factors=(
            RiskFactorResult(
                factor=RiskFactor.MATCH,
                weight=30,
                triggered=True,
                evidence_field="ppsr_result",
                evidence_value="MATCH",
                score_contribution=30,
            ),
            RiskFactorResult(
                factor=RiskFactor.STATUTORY,
                weight=40,
                triggered=True,
                evidence_field="writeoff_status",
                evidence_value="STATUTORY",
                score_contribution=40,
            ),
        ),
        findings=(
            MandatoryFinding(
                finding_id="f-1",
                factor=RiskFactor.MATCH,
                title="PPSR Match",
                description="Encumbered vehicle",
                evidence_field="ppsr_result",
                evidence_value="MATCH",
                weight=30,
            ),
            MandatoryFinding(
                finding_id="f-2",
                factor=RiskFactor.STATUTORY,
                title="Statutory Write-off",
                description="Non-repairable writeoff",
                evidence_field="writeoff_status",
                evidence_value="STATUTORY",
                weight=40,
            ),
        ),
        calculation_hash="b" * 64,
    )

    await result_repo.save_result(result)

    # 3. Retrieve and verify
    retrieved = await result_repo.get_result(asmt.id, 1)
    assert retrieved is not None
    assert retrieved.id == result.id
    assert retrieved.assessment_id == asmt.id
    assert retrieved.run_number == 1
    assert retrieved.score == 70
    assert retrieved.band == RiskBand.CRITICAL
    assert retrieved.raw_score == 70
    assert retrieved.is_incomplete is False
    assert len(retrieved.factors) == 2
    assert len(retrieved.findings) == 2
    assert retrieved.calculation_hash == "b" * 64


@pytest.mark.asyncio
async def test_persist_and_retrieve_incomplete_risk_result(session: AsyncSession) -> None:
    """Persist an incomplete RiskResult and verify numeric score/band are strictly None."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-1",
        idempotency_key="idemp-key-2",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER, intended_use="Commuting"),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="policy-incomp-1")
    await policy_repo.create_policy(policy)

    result_repo = RiskResultRepository(session)
    result = RiskResult(
        assessment_id=asmt.id,
        run_number=1,
        policy_id=policy.id,
        policy_version=policy.version,
        score=None,
        band=None,
        raw_score=None,
        is_incomplete=True,
        missing_evidence=(
            MissingEvidenceFinding(
                field_name="ppsr_result",
                reason=MissingEvidenceReason.UNKNOWN,
                details="PPSR lookup unavailable",
            ),
        ),
        calculation_hash="c" * 64,
    )

    await result_repo.save_result(result)

    retrieved = await result_repo.get_result(asmt.id, 1)
    assert retrieved is not None
    assert retrieved.is_incomplete is True
    assert retrieved.score is None
    assert retrieved.band is None
    assert retrieved.raw_score is None
    assert len(retrieved.missing_findings) == 1
    assert retrieved.missing_findings[0].field_name == "ppsr_result"


@pytest.mark.asyncio
async def test_risk_result_save_idempotency(session: AsyncSession) -> None:
    """Re-saving the exact same RiskResult succeeds idempotently without error."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-1",
        idempotency_key="idemp-key-3",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER, intended_use="Commuting"),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="policy-idem-1")
    await policy_repo.create_policy(policy)

    result_repo = RiskResultRepository(session)
    result = RiskResult(
        assessment_id=asmt.id,
        run_number=1,
        policy_id=policy.id,
        policy_version=policy.version,
        score=0,
        band=RiskBand.LOW,
        raw_score=0,
        is_incomplete=False,
        calculation_hash="d" * 64,
    )

    # Save once
    await result_repo.save_result(result)
    # Save identical second time
    await result_repo.save_result(result)

    retrieved = await result_repo.get_result(asmt.id, 1)
    assert retrieved is not None
    assert retrieved.score == 0


@pytest.mark.asyncio
async def test_risk_result_immutability_violation_raises(session: AsyncSession) -> None:
    """Attempting to overwrite a persisted RiskResult with conflicting score raises ValueError."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-1",
        idempotency_key="idemp-key-4",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER, intended_use="Commuting"),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="policy-immut-1")
    await policy_repo.create_policy(policy)

    result_repo = RiskResultRepository(session)
    result1 = RiskResult(
        assessment_id=asmt.id,
        run_number=1,
        policy_id=policy.id,
        policy_version=policy.version,
        score=20,
        band=RiskBand.MEDIUM,
        raw_score=20,
        is_incomplete=False,
        calculation_hash="e" * 64,
    )
    await result_repo.save_result(result1)

    # Mutated result with different score
    result2 = RiskResult(
        assessment_id=asmt.id,
        run_number=1,
        policy_id=policy.id,
        policy_version=policy.version,
        score=40,
        band=RiskBand.HIGH,
        raw_score=40,
        is_incomplete=False,
        calculation_hash="f" * 64,
    )

    with pytest.raises(ValueError, match="is immutable and cannot be overwritten"):
        await result_repo.save_result(result2)


@pytest.mark.asyncio
async def test_replayability_after_policy_retirement(session: AsyncSession) -> None:
    """Ensure historical assessments retain pinned policy version after policy is retired."""
    # 1. Setup policy service and create 2 policies
    service = RiskPolicyService(session)
    policy1 = await service.create_policy(
        policy_id="policy-v1-replay", name="Policy v1", description="Baseline"
    )
    await service.validate_and_mark_ready(policy1.id)
    await service.activate_policy(policy1.id, operator_id="op-1")

    # 2. Record assessment 1 using Policy 1
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="user-1",
        idempotency_key="idemp-key-5",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER, intended_use="Commuting"),
        ),
    )

    result_repo = RiskResultRepository(session)
    result = RiskResult(
        assessment_id=asmt.id,
        run_number=1,
        policy_id=policy1.id,
        policy_version=policy1.version,
        score=30,
        band=RiskBand.MEDIUM,
        raw_score=30,
        is_incomplete=False,
        calculation_hash="1" * 64,
    )
    await result_repo.save_result(result)

    # 3. Create Policy 2, validate and activate it (retiring Policy 1)
    policy2 = await service.create_policy(
        policy_id="policy-v2-replay", name="Policy v2", description="Updated"
    )
    await service.validate_and_mark_ready(policy2.id)
    active_p2, retired_p1 = await service.activate_policy(policy2.id, operator_id="op-1")

    assert active_p2.id == "policy-v2-replay"
    assert active_p2.lifecycle_state == RiskPolicyLifecycleState.ACTIVE
    assert retired_p1 is not None
    assert retired_p1.id == "policy-v1-replay"
    assert retired_p1.lifecycle_state == RiskPolicyLifecycleState.RETIRED

    # 4. Retrieve stored assessment result: it remains tied to policy-v1-replay
    stored_res = await result_repo.get_result(asmt.id, 1)
    assert stored_res is not None
    assert stored_res.policy_id == "policy-v1-replay"

    # 5. Fetch retired policy by ID for historical audit replay
    policy_repo = RiskPolicyRepository(session)
    pinned_policy = await policy_repo.get_policy(stored_res.policy_id)
    assert pinned_policy is not None
    assert pinned_policy.lifecycle_state == RiskPolicyLifecycleState.RETIRED
    assert pinned_policy.retired_at is not None


@pytest.mark.asyncio
async def test_activate_policy_with_lexicographically_earlier_id(session: AsyncSession) -> None:
    """Activating a policy with lower ID than active policy succeeds without UniqueViolation."""
    service = RiskPolicyService(session)
    # Create and activate policy with higher lexicographical ID (pol-zzz)
    pol_z = await service.create_policy(
        policy_id="pol-zzz", name="Policy Z", description="Active Policy Z"
    )
    await service.validate_and_mark_ready(pol_z.id)
    active_z, _ = await service.activate_policy(pol_z.id, operator_id="op-1")
    assert active_z.id == "pol-zzz"
    assert active_z.lifecycle_state == RiskPolicyLifecycleState.ACTIVE

    # Create and activate policy with lower lexicographical ID (pol-aaa)
    pol_a = await service.create_policy(
        policy_id="pol-aaa", name="Policy A", description="Target Policy A"
    )
    await service.validate_and_mark_ready(pol_a.id)
    active_a, retired_z = await service.activate_policy(pol_a.id, operator_id="op-1")

    assert active_a.id == "pol-aaa"
    assert active_a.lifecycle_state == RiskPolicyLifecycleState.ACTIVE
    assert retired_z is not None
    assert retired_z.id == "pol-zzz"
    assert retired_z.lifecycle_state == RiskPolicyLifecycleState.RETIRED
