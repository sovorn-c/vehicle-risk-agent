"""Tests for PolicyRepository CRUD operations, foreign key integrity, and retention constraints."""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.persistence.models import (
    Base,
    PolicyCorpusRecord,
    PolicyCorpusSnapshotRecord,
    PolicySnapshotRecord,
    PolicySourceRecord,
)
from vehicle_risk_agent.persistence.policy_repository import PolicyRepository
from vehicle_risk_agent.policy.models import AuthorityClassification, PolicySource

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def session_factory() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_policy_repository_retention_blocks_deleting_referenced_snapshot(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Deleting a PolicySource referenced by an active corpus fails with IntegrityError."""
    async with session_factory() as session:
        repo = PolicyRepository(session)
        source = PolicySource(
            id="nz-fta-1986",
            title="Fair Trading Act 1986",
            issuing_authority="NZ Parliament",
            jurisdiction="NZ",
            canonical_origin="https://legislation.govt.nz/fta",
            authority_classification=AuthorityClassification.PRIMARY_LEGISLATION,
            reuse_terms="Open Government Licence NZ",
            expected_update_cadence="annual",
        )
        await repo.create_or_update_source(source)

        # Create raw snapshot record
        now = datetime.now(UTC)
        snap_rec = PolicySnapshotRecord(
            id="nz-fta-1986:snap1",
            source_id="nz-fta-1986",
            raw_content="# Content",
            content_hash="a" * 64,
            parser_version="v1",
            validation_outcome="VALID",
            retrieved_at=now,
            created_at=now,
        )
        session.add(snap_rec)

        # Create corpus referencing snapshot
        corpus_rec = PolicyCorpusRecord(
            id="corpus-locked",
            name="Locked Corpus",
            description="Active corpus",
            lifecycle_state="ACTIVE",
            retrieval_config_json="{}",
            manifest_hash="b" * 64,
            created_at=now,
        )
        session.add(corpus_rec)
        await session.flush()

        assoc = PolicyCorpusSnapshotRecord(
            corpus_id="corpus-locked",
            snapshot_id="nz-fta-1986:snap1",
        )
        session.add(assoc)
        await session.commit()

    # Now attempt to delete the source
    async with session_factory() as session:
        source_rec = await session.get(PolicySourceRecord, "nz-fta-1986")
        assert source_rec is not None
        await session.delete(source_rec)
        with pytest.raises(IntegrityError):
            await session.commit()
