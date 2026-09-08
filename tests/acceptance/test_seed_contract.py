"""Acceptance tests verifying database migrations and deterministic, idempotent seeding."""

# story: e07s02

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from vehicle_risk_agent.auth import Role, authenticate_bearer_token
from vehicle_risk_agent.cli.seed import seed_database
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.evaluation.matrix import get_evaluation_matrix
from vehicle_risk_agent.persistence.models import (
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
    reset_sql = (
        "DROP SCHEMA public CASCADE; CREATE SCHEMA public; CREATE EXTENSION IF NOT EXISTS vector;"
    )
    async with engine.begin() as conn:
        await conn.execute(sa.text(reset_sql))
    yield engine
    async with engine.begin() as conn:
        await conn.execute(sa.text(reset_sql))
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
    assert result2["principals"] == {
        "requester": "principal-requester-1",
        "reviewer": "principal-reviewer-1",
        "operator": "principal-operator-1",
        "maintainer": "principal-maintainer-1",
    }
    assert result2["evaluation_scenarios"] == 30

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

        # Manifest hash must not be overwritten by placeholder snapshot ID
        active_pc = active_pcs[0]
        from vehicle_risk_agent.policy.corpus_models import (
            RetrievalConfiguration,
            compute_manifest_hash,
        )

        placeholder_hash = compute_manifest_hash(
            corpus_id=active_pc.id,
            snapshot_ids=["nz-fta-1986-snap1"],
            retrieval_config=RetrievalConfiguration(),
        )
        assert active_pc.manifest_hash != placeholder_hash, (
            "Active corpus manifest_hash must not be overwritten with placeholder snapshot ID"
        )


@pytest.mark.asyncio
async def test_seed_database_raises_on_migration_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alembic migration failure must not be masked by seed_database."""

    def _failing_migration(_url: str) -> None:
        raise RuntimeError("Simulated Alembic migration failure")

    monkeypatch.setattr("vehicle_risk_agent.cli.seed.run_migrations", _failing_migration)
    settings = Settings(database_url=TEST_DB_URL)
    with pytest.raises(RuntimeError, match="Simulated Alembic migration failure"):
        await seed_database(database_url=TEST_DB_URL, settings=settings)


@pytest.mark.asyncio
async def test_principals_and_evaluation_matrix_available() -> None:
    """Principals for all four roles must authenticate, and evaluation scenarios must load."""
    settings = Settings(database_url=TEST_DB_URL)

    req_p = authenticate_bearer_token("dev-requester-token", settings)
    assert req_p is not None
    assert req_p.role == Role.REQUESTER

    rev_p = authenticate_bearer_token("dev-reviewer-token", settings)
    assert rev_p is not None
    assert rev_p.role == Role.REVIEWER

    op_p = authenticate_bearer_token("dev-operator-token", settings)
    assert op_p is not None
    assert op_p.role == Role.TECHNICAL_OPERATOR

    maint_p = authenticate_bearer_token("dev-maintainer-token", settings)
    assert maint_p is not None
    assert maint_p.role == Role.POLICY_CORPUS_MAINTAINER

    scenarios = get_evaluation_matrix()
    assert len(scenarios) == 30
