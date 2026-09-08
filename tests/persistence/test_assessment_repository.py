"""Tests for transactional Assessment and Assessment Run persistence with idempotency."""

# story: e01s03

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState, AssessmentRunPhase
from vehicle_risk_agent.domain.errors import IdempotencyConflictError
from vehicle_risk_agent.persistence.models import Base
from vehicle_risk_agent.persistence.repository import AssessmentRepository

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Provide a database session connected to local PostgreSQL with freshly created tables."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as sess:
        yield sess

    await engine.dispose()


@pytest.mark.asyncio
async def test_create_and_get_assessment(session: AsyncSession) -> None:
    """Verify creating an Assessment atomically allocates Run 1 in PENDING state."""
    repo = AssessmentRepository(session)
    req = AssessmentCreateRequest(
        vin="1HGCR2F85HA000000",
        context=AssessmentContext(
            sale_type=SaleType.DEALER,
            intended_use="Private buyer commuter",
            questions=["Any stolen record?"],
        ),
    )

    assessment = await repo.create_assessment(
        requester_id="principal-requester-1",
        idempotency_key="idemp-key-1",
        request=req,
    )

    assert assessment.id is not None
    assert assessment.requester_id == "principal-requester-1"
    assert assessment.vin == "1HGCR2F85HA000000"
    assert assessment.lifecycle_state == AssessmentLifecycleState.IN_PROGRESS
    assert assessment.current_run_number == 1
    assert len(assessment.runs) == 1
    assert assessment.runs[0].run_number == 1
    assert assessment.runs[0].phase == AssessmentRunPhase.PENDING

    # Retrieve by ID
    retrieved = await repo.get_assessment(assessment.id)
    assert retrieved is not None
    assert retrieved.id == assessment.id
    assert retrieved.vin == assessment.vin
    assert retrieved.current_run_number == 1


@pytest.mark.asyncio
async def test_idempotent_replay_returns_existing_record(session: AsyncSession) -> None:
    """Verify exact command replay with identical payload returns the existing Assessment."""
    repo = AssessmentRepository(session)
    req = AssessmentCreateRequest(
        vin="1HGCR2F85HA000000",
        context=AssessmentContext(sale_type=SaleType.PRIVATE),
    )

    first = await repo.create_assessment(
        requester_id="principal-requester-1",
        idempotency_key="idemp-key-replay",
        request=req,
    )

    second = await repo.create_assessment(
        requester_id="principal-requester-1",
        idempotency_key="idemp-key-replay",
        request=req,
    )

    assert first.id == second.id
    assert second.current_run_number == 1


@pytest.mark.asyncio
async def test_idempotent_key_reuse_with_different_payload_raises_conflict(
    session: AsyncSession,
) -> None:
    """Verify key reuse with different VIN or context raises IdempotencyConflictError."""
    repo = AssessmentRepository(session)
    req1 = AssessmentCreateRequest(
        vin="1HGCR2F85HA000000",
        context=AssessmentContext(sale_type=SaleType.PRIVATE),
    )
    req2 = AssessmentCreateRequest(
        vin="1HGCR2F8XHA000008",
        context=AssessmentContext(sale_type=SaleType.DEALER),
    )

    await repo.create_assessment(
        requester_id="principal-requester-1",
        idempotency_key="idemp-conflict-key",
        request=req1,
    )

    with pytest.raises(IdempotencyConflictError):
        await repo.create_assessment(
            requester_id="principal-requester-1",
            idempotency_key="idemp-conflict-key",
            request=req2,
        )
