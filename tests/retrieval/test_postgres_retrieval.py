"""Integration tests for PostgreSQL pgvector dense search and tsvector full-text search."""

import hashlib
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.persistence.models import (
    Base,
    PolicyPassageRecord,
    PolicySnapshotRecord,
    PolicySourceRecord,
)
from vehicle_risk_agent.retrieval.adapters import FakeEmbeddingAdapter, FakeRerankerAdapter
from vehicle_risk_agent.retrieval.postgres_index import PostgresPolicyIndex
from vehicle_risk_agent.retrieval.service import HybridRetrievalService

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


@pytest_asyncio.fixture
async def seeded_postgres_passages(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[str]:
    """Seed PostgreSQL with test policy passages with dense vectors."""
    embedder = FakeEmbeddingAdapter(dimensions=384)
    now = datetime.now(UTC)

    async with session_factory() as session:
        src = PolicySourceRecord(
            id="nz-legislation-fta-1986",
            title="Fair Trading Act 1986",
            issuing_authority="NZ Parliament",
            jurisdiction="NZ",
            canonical_origin="https://legislation.govt.nz/fta",
            authority_classification="PRIMARY_LEGISLATION",
            reuse_terms="Open",
            expected_update_cadence="annual",
            created_at=now,
        )
        src2 = PolicySourceRecord(
            id="ppsr-guide",
            title="PPSR Guide",
            issuing_authority="MBIE",
            jurisdiction="NZ",
            canonical_origin="https://ppsr.govt.nz",
            authority_classification="REGULATORY_GUIDANCE",
            reuse_terms="Open",
            expected_update_cadence="annual",
            created_at=now,
        )
        session.add_all([src, src2])

        snap = PolicySnapshotRecord(
            id="snap-1",
            source_id="nz-legislation-fta-1986",
            raw_content="Content",
            content_hash="a" * 64,
            parser_version="v1",
            validation_outcome="VALID",
            retrieved_at=now,
            created_at=now,
        )
        snap2 = PolicySnapshotRecord(
            id="snap-2",
            source_id="ppsr-guide",
            raw_content="Content",
            content_hash="b" * 64,
            parser_version="v1",
            validation_outcome="VALID",
            retrieved_at=now,
            created_at=now,
        )
        session.add_all([snap, snap2])
        await session.flush()

        # Seed passages with vectors
        p1_text = "No person shall, in trade, engage in misleading or deceptive conduct."
        p2_text = "A registered security interest allows a secured creditor to repossess."

        v1 = await embedder.embed_query(f"Misleading conduct\n{p1_text}")
        v2 = await embedder.embed_query(f"Security interest repossession\n{p2_text}")

        pass1 = PolicyPassageRecord(
            id="snap-1:p001",
            snapshot_id="snap-1",
            source_id="nz-legislation-fta-1986",
            section_identifier="Section 9",
            heading="Misleading conduct",
            text=p1_text,
            sequence=1,
            char_offset_start=0,
            char_offset_end=len(p1_text),
            content_hash=hashlib.sha256(p1_text.encode()).hexdigest(),
            embedding=v1,
        )
        pass2 = PolicyPassageRecord(
            id="snap-2:p001",
            snapshot_id="snap-2",
            source_id="ppsr-guide",
            section_identifier="Section 1",
            heading="Security interest repossession",
            text=p2_text,
            sequence=1,
            char_offset_start=0,
            char_offset_end=len(p2_text),
            content_hash=hashlib.sha256(p2_text.encode()).hexdigest(),
            embedding=v2,
        )
        session.add_all([pass1, pass2])
        await session.commit()

    return ["snap-1", "snap-2"]


@pytest.mark.asyncio
async def test_postgres_dense_pgvector_search(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_postgres_passages: list[str],
) -> None:
    """PostgresPolicyIndex searches pgvector embeddings using cosine distance."""
    embedder = FakeEmbeddingAdapter(dimensions=384)
    async with session_factory() as session:
        index = PostgresPolicyIndex(
            session=session,
            embedder=embedder,
            snapshot_ids=seeded_postgres_passages,
        )
        results = await index.search_dense(query="creditor repossess motor vehicle", top_k=5)

        assert len(results) >= 1
        assert results[0].passage_id == "snap-2:p001"
        assert results[0].score > 0.0


@pytest.mark.asyncio
async def test_postgres_index_resolves_authoritative_citation_metadata(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_postgres_passages: list[str],
) -> None:
    """Postgres-backed retrieval resolves citations from the source registry."""
    embedder = FakeEmbeddingAdapter(dimensions=384)
    async with session_factory() as session:
        index = PostgresPolicyIndex(
            session=session,
            embedder=embedder,
            snapshot_ids=seeded_postgres_passages,
        )
        service = HybridRetrievalService(
            index=index,
            reranker=FakeRerankerAdapter(),
        )
        result = await service.retrieve(query="creditor repossess motor vehicle")

    assert result.citations
    assert result.citations[0].source_id == "ppsr-guide"
    assert result.citations[0].source_title == "PPSR Guide"
    assert result.citations[0].canonical_origin == "https://ppsr.govt.nz"


@pytest.mark.asyncio
async def test_postgres_keyword_tsvector_search(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_postgres_passages: list[str],
) -> None:
    """PostgresPolicyIndex searches PostgreSQL full-text tsvector/tsquery."""
    embedder = FakeEmbeddingAdapter(dimensions=384)
    async with session_factory() as session:
        index = PostgresPolicyIndex(
            session=session,
            embedder=embedder,
            snapshot_ids=seeded_postgres_passages,
        )
        results = await index.search_keyword(query="misleading deceptive", top_k=5)

        assert len(results) >= 1
        assert results[0].passage_id == "snap-1:p001"
        assert "Misleading" in results[0].passage.heading
