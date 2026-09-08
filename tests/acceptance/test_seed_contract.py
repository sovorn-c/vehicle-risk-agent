"""Acceptance tests verifying database migrations and deterministic, idempotent seeding."""

from collections.abc import AsyncIterator
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.auth import Role, authenticate_bearer_token
from vehicle_risk_agent.cli.seed import seed_database
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.evaluation.matrix import load_evaluation_matrix
from vehicle_risk_agent.persistence.models import (
    Base,
    PolicyCorpusRecord,
    PolicyPassageRecord,
    PolicySourceRecord,
    RiskPolicyRecord,
)

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def clean_engine() -> AsyncIterator[AsyncEngine]:
    """Provide a clean PostgreSQL database for testing migrations and seeds."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_seed_database_idempotent(clean_engine: AsyncEngine) -> None:
    """Seeding must create migrations and seeds, and second execution must be idempotent."""
    settings = Settings(database_url=TEST_DB_URL)

    # First seed run
    result1 = await seed_database(database_url=TEST_DB_URL, settings=settings)
    assert result1["status"] in ("seeded", "ok")

    # Second seed run (must not fail or duplicate active policies)
    result2 = await seed_database(database_url=TEST_DB_URL, settings=settings)
    assert result2["status"] in ("seeded", "ok")

    # Verify database state
    session_factory = async_sessionmaker(clean_engine, expire_on_commit=False)
    async with session_factory() as session:
        # Exactly one active risk policy
        rp_stmt = select(RiskPolicyRecord).where(RiskPolicyRecord.lifecycle_state == "ACTIVE")
        active_rps = (await session.execute(rp_stmt)).scalars().all()
        assert len(active_rps) == 1
        assert active_rps[0].version == "v1"

        # Exactly one active policy corpus
        pc_stmt = select(PolicyCorpusRecord).where(PolicyCorpusRecord.lifecycle_state == "ACTIVE")
        active_pcs = (await session.execute(pc_stmt)).scalars().all()
        assert len(active_pcs) == 1

        # Policy sources and passages exist
        sources = (await session.execute(select(PolicySourceRecord))).scalars().all()
        assert len(sources) >= 1

        passages = (await session.execute(select(PolicyPassageRecord))).scalars().all()
        assert len(passages) >= 1


@pytest.mark.asyncio
async def test_principals_and_evaluation_matrix_available() -> None:
    """Principals for all four roles must authenticate, and evaluation scenarios must load."""
    settings = Settings(database_url=TEST_DB_URL)

    req_p = authenticate_bearer_token("dev-requester-token", settings)
    assert req_p is not None and req_p.role == Role.REQUESTER

    rev_p = authenticate_bearer_token("dev-reviewer-token", settings)
    assert rev_p is not None and rev_p.role == Role.REVIEWER

    op_p = authenticate_bearer_token("dev-operator-token", settings)
    assert op_p is not None and op_p.role == Role.TECHNICAL_OPERATOR

    maint_p = authenticate_bearer_token("dev-maintainer-token", settings)
    assert maint_p is not None and maint_p.role == Role.POLICY_CORPUS_MAINTAINER

    scenarios = load_evaluation_matrix()
    assert len(scenarios) == 30
