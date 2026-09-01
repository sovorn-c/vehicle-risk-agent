"""Tests for PolicyRepository CRUD operations, foreign key integrity, and retention constraints."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from vehicle_risk_agent.persistence.models import (
    Base,
    PolicyCorpusRecord,
    PolicyCorpusSnapshotRecord,
    PolicyPassageRecord,
    PolicySnapshotRecord,
    PolicySourceRecord,
)
from vehicle_risk_agent.persistence.policy_repository import PolicyRepository
from vehicle_risk_agent.policy.ingestion import ingest_policy_source
from vehicle_risk_agent.policy.models import AuthorityClassification, PolicySource
from vehicle_risk_agent.retrieval.adapters import FakeEmbeddingAdapter

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


@pytest.mark.asyncio
async def test_deleting_referenced_snapshot_through_orm_is_restricted(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """ORM deletion cannot cascade away a corpus's snapshot association."""
    now = datetime.now(UTC)
    async with session_factory() as session:
        session.add(
            PolicySourceRecord(
                id="snapshot-source",
                title="Source",
                issuing_authority="NZ",
                jurisdiction="NZ",
                canonical_origin="https://example.test/source",
                authority_classification="OFFICIAL_GUIDANCE",
                reuse_terms="Open",
                expected_update_cadence="annual",
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            PolicySnapshotRecord(
                id="snapshot-to-protect",
                source_id="snapshot-source",
                raw_content="content",
                content_hash="c" * 64,
                parser_version="v1",
                validation_outcome="VALID",
                retrieved_at=now,
                created_at=now,
            )
        )
        session.add(
            PolicyCorpusRecord(
                id="corpus-protecting-snapshot",
                name="Corpus",
                description="Description",
                lifecycle_state="ACTIVE",
                retrieval_config_json="{}",
                manifest_hash="d" * 64,
                created_at=now,
            )
        )
        await session.flush()
        session.add(
            PolicyCorpusSnapshotRecord(
                corpus_id="corpus-protecting-snapshot",
                snapshot_id="snapshot-to-protect",
            )
        )
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(
            select(PolicySnapshotRecord)
            .options(selectinload(PolicySnapshotRecord.corpora_associations))
            .where(PolicySnapshotRecord.id == "snapshot-to-protect")
        )
        snapshot = result.scalar_one()
        await session.delete(snapshot)
        with pytest.raises(IntegrityError):
            await session.commit()


@pytest.mark.asyncio
async def test_concurrent_identical_snapshot_ingestion_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Concurrent ingestion of identical content returns one durable snapshot."""
    source = PolicySource(
        id="concurrent-source",
        title="Concurrent Source",
        issuing_authority="NZ",
        jurisdiction="NZ",
        canonical_origin="https://example.test/concurrent",
        authority_classification=AuthorityClassification.OFFICIAL_GUIDANCE,
        reuse_terms="Open",
        expected_update_cadence="annual",
    )
    snapshot = ingest_policy_source(source, "# Guide\n## Section 1\nContent")

    async with session_factory() as session:
        await PolicyRepository(session).create_or_update_source(source)

    async def ingest_once() -> str:
        async with session_factory() as session:
            saved = await PolicyRepository(session).create_snapshot(snapshot)
            return saved.id

    results = await asyncio.gather(ingest_once(), ingest_once())
    assert list(results) == [snapshot.id, snapshot.id]

    async with session_factory() as session:
        count = await session.scalar(
            sa.select(sa.func.count())
            .select_from(PolicySnapshotRecord)
            .where(PolicySnapshotRecord.source_id == source.id)
        )
        assert count == 1


@pytest.mark.asyncio
async def test_snapshot_ingestion_persists_embeddings_when_configured(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Configured ingestion stores vectors for every persisted passage."""
    source = PolicySource(
        id="embedding-source",
        title="Embedding Source",
        issuing_authority="NZ",
        jurisdiction="NZ",
        canonical_origin="https://example.test/embedding",
        authority_classification=AuthorityClassification.OFFICIAL_GUIDANCE,
        reuse_terms="Open",
        expected_update_cadence="annual",
    )
    snapshot = ingest_policy_source(source, "# Guide\n## Section 1\nContent")

    async with session_factory() as session:
        repo = PolicyRepository(session, embedder=FakeEmbeddingAdapter())
        await repo.create_or_update_source(source)
        await repo.create_snapshot(snapshot)

        result = await session.execute(
            select(PolicyPassageRecord).where(PolicyPassageRecord.snapshot_id == snapshot.id)
        )
        passages = result.scalars().all()
        assert passages
        assert all(p.embedding is not None for p in passages)
        assert all(len(p.embedding or []) == 384 for p in passages)
