"""Tests for Policy Corpus state transitions, validation, and single-active-corpus invariant."""

# story: e02s02

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.persistence.models import Base
from vehicle_risk_agent.persistence.policy_repository import PolicyRepository
from vehicle_risk_agent.policy.corpus_lifecycle import (
    CorpusLifecycleError,
    CorpusLifecycleManager,
)
from vehicle_risk_agent.policy.corpus_models import (
    CorpusLifecycleState,
)
from vehicle_risk_agent.policy.ingestion import PolicyParser, ingest_policy_source
from vehicle_risk_agent.policy.models import (
    AuthorityClassification,
    PolicySource,
)
from vehicle_risk_agent.retrieval.adapters import FakeEmbeddingAdapter

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def session_factory() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        import sqlalchemy as sa

        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def seeded_snapshots(session_factory: async_sessionmaker[AsyncSession]) -> list[str]:
    async with session_factory() as session:
        repo = PolicyRepository(session, embedder=FakeEmbeddingAdapter(dimensions=384))
        source = PolicySource(
            id="nz-legislation-fta-1986",
            title="Fair Trading Act 1986",
            issuing_authority="Parliament of NZ",
            jurisdiction="NZ",
            canonical_origin="https://example.com",
            authority_classification=AuthorityClassification.PRIMARY_LEGISLATION,
            reuse_terms="CC BY 4.0",
            expected_update_cadence="ADHOC",
        )
        await repo.create_or_update_source(source)
        snap = ingest_policy_source(
            source=source,
            raw_content="# FTA 1986\n## Section 9: Misleading conduct\nProhibited.",
            parser=PolicyParser(),
        )
        saved_snap = await repo.create_snapshot(snap)
        return [saved_snap.id]


@pytest.mark.asyncio
async def test_corpus_lifecycle_transitions_draft_to_ready_to_active(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_snapshots: list[str],
) -> None:
    """A valid corpus moves from DRAFT -> READY -> ACTIVE atomically."""
    async with session_factory() as session:
        manager = CorpusLifecycleManager(session)

        # 1. Create DRAFT
        draft = await manager.create_corpus(
            corpus_id="corpus-2026-v1",
            name="NZ Vehicle Risk Baseline 2026",
            description="Initial policy corpus",
            snapshot_ids=seeded_snapshots,
        )
        assert draft.lifecycle_state == CorpusLifecycleState.DRAFT

        # 2. Mark READY
        ready = await manager.validate_and_mark_ready("corpus-2026-v1")
        assert ready.lifecycle_state == CorpusLifecycleState.READY

        # 3. Activate
        active, prior = await manager.activate_corpus("corpus-2026-v1", principal_id="maintainer-1")
        assert active.lifecycle_state == CorpusLifecycleState.ACTIVE
        assert active.activated_at is not None
        assert prior is None

        # Verify active corpus query
        current_active = await manager.get_active_corpus()
        assert current_active is not None
        assert current_active.id == "corpus-2026-v1"


@pytest.mark.asyncio
async def test_activating_new_corpus_retires_previous_active(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_snapshots: list[str],
) -> None:
    """Activating a second corpus retires the first active corpus in the same transaction."""
    async with session_factory() as session:
        manager = CorpusLifecycleManager(session)

        # Setup and activate corpus 1
        await manager.create_corpus("corpus-v1", "Corpus 1", "Desc", seeded_snapshots)
        await manager.validate_and_mark_ready("corpus-v1")
        await manager.activate_corpus("corpus-v1", "maintainer-1")

        # Setup corpus 2
        await manager.create_corpus("corpus-v2", "Corpus 2", "Desc", seeded_snapshots)
        await manager.validate_and_mark_ready("corpus-v2")

        # Activate corpus 2
        active_2, retired_1 = await manager.activate_corpus("corpus-v2", "maintainer-1")
        assert active_2.id == "corpus-v2"
        assert active_2.lifecycle_state == CorpusLifecycleState.ACTIVE
        assert retired_1 is not None
        assert retired_1.id == "corpus-v1"
        assert retired_1.lifecycle_state == CorpusLifecycleState.RETIRED
        assert retired_1.retired_at is not None

        # Exactly one ACTIVE corpus exists
        active = await manager.get_active_corpus()
        assert active is not None
        assert active.id == "corpus-v2"

        # Old corpus is RETIRED and inspectable for replay
        c1 = await manager.get_corpus("corpus-v1")
        assert c1 is not None
        assert c1.lifecycle_state == CorpusLifecycleState.RETIRED


@pytest.mark.asyncio
async def test_invalid_transitions_rejected(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_snapshots: list[str],
) -> None:
    """Cannot activate DRAFT directly or activate non-existent corpus."""
    async with session_factory() as session:
        manager = CorpusLifecycleManager(session)
        await manager.create_corpus("corpus-draft", "Draft", "Desc", seeded_snapshots)

        # Direct activation of DRAFT fails
        with pytest.raises(CorpusLifecycleError, match="must be in READY state"):
            await manager.activate_corpus("corpus-draft", "maintainer-1")

        # Non-existent corpus fails
        with pytest.raises(CorpusLifecycleError, match="not found"):
            await manager.validate_and_mark_ready("missing-corpus")


@pytest.mark.asyncio
async def test_concurrent_activation_enforces_single_active_invariant(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_snapshots: list[str],
) -> None:
    """Concurrent activation across distinct sessions results in exactly one active corpus."""
    import asyncio

    from sqlalchemy import select

    from vehicle_risk_agent.persistence.models import PolicyCorpusRecord

    # Create 5 ready corpora
    async with session_factory() as session:
        mgr = CorpusLifecycleManager(session)
        for i in range(1, 6):
            cid = f"corpus-race-{i}"
            await mgr.create_corpus(cid, f"Corpus {i}", "Desc", seeded_snapshots)
            await mgr.validate_and_mark_ready(cid)

    async def activate_in_session(cid: str) -> None:
        async with session_factory() as s:
            m = CorpusLifecycleManager(s)
            await m.activate_corpus(cid, f"maintainer-{cid}")

    # Fire 5 concurrent activations
    tasks = [activate_in_session(f"corpus-race-{i}") for i in range(1, 6)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Verify at least one succeeded
    succeeded = [r for r in results if not isinstance(r, Exception)]
    assert len(succeeded) >= 1

    # Verify strictly at most 1 active row exists in the database
    async with session_factory() as session:
        active_stmt = select(PolicyCorpusRecord).where(
            PolicyCorpusRecord.lifecycle_state == CorpusLifecycleState.ACTIVE.value
        )
        active_res = await session.execute(active_stmt)
        active_rows = active_res.scalars().all()
        assert len(active_rows) == 1


@pytest.mark.asyncio
async def test_validate_and_mark_ready_rejects_missing_vectors(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A corpus containing snapshots with missing passage vectors fails validation."""
    async with session_factory() as session:
        repo = PolicyRepository(session)  # no embedder -> missing vectors
        source = PolicySource(
            id="nz-fta-no-vec",
            title="Fair Trading Act 1986",
            issuing_authority="Parliament of NZ",
            jurisdiction="NZ",
            canonical_origin="https://example.com",
            authority_classification=AuthorityClassification.PRIMARY_LEGISLATION,
            reuse_terms="CC BY 4.0",
            expected_update_cadence="ADHOC",
        )
        await repo.create_or_update_source(source)
        snap = ingest_policy_source(
            source=source,
            raw_content="# FTA\n## S9\nText without vectors",
            parser=PolicyParser(),
        )
        saved_snap = await repo.create_snapshot(snap)

        mgr = CorpusLifecycleManager(session)
        await mgr.create_corpus("corpus-no-vec", "No Vec", "Desc", [saved_snap.id])
        with pytest.raises(CorpusLifecycleError, match="missing embedding vector"):
            await mgr.validate_and_mark_ready("corpus-no-vec")


@pytest.mark.asyncio
async def test_validate_and_mark_ready_rejects_zero_placeholder_vectors(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A corpus containing all-zero placeholder vectors fails validation."""
    from sqlalchemy import select

    from vehicle_risk_agent.persistence.models import PolicyPassageRecord

    async with session_factory() as session:
        repo = PolicyRepository(session, embedder=FakeEmbeddingAdapter(dimensions=384))
        source = PolicySource(
            id="nz-fta-zero-vec",
            title="Fair Trading Act 1986",
            issuing_authority="Parliament of NZ",
            jurisdiction="NZ",
            canonical_origin="https://example.com",
            authority_classification=AuthorityClassification.PRIMARY_LEGISLATION,
            reuse_terms="CC BY 4.0",
            expected_update_cadence="ADHOC",
        )
        await repo.create_or_update_source(source)
        snap = ingest_policy_source(
            source=source,
            raw_content="# FTA\n## S9\nText with zero vector",
            parser=PolicyParser(),
        )
        saved_snap = await repo.create_snapshot(snap)

        # Mutate the passage embedding to all zeros
        stmt = select(PolicyPassageRecord).where(PolicyPassageRecord.snapshot_id == saved_snap.id)
        res = await session.execute(stmt)
        p = res.scalar_one()
        p.embedding = [0.0] * 384
        await session.commit()

        mgr = CorpusLifecycleManager(session)
        await mgr.create_corpus("corpus-zero-vec", "Zero Vec", "Desc", [saved_snap.id])
        with pytest.raises(CorpusLifecycleError, match="placeholder zero embedding vector"):
            await mgr.validate_and_mark_ready("corpus-zero-vec")


@pytest.mark.asyncio
async def test_validate_and_mark_ready_rejects_wrong_dimension_vectors(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_snapshots: list[str],
) -> None:
    """A corpus configured with incompatible vector dimensions fails validation."""
    from vehicle_risk_agent.policy.corpus_models import RetrievalConfiguration

    async with session_factory() as session:
        mgr = CorpusLifecycleManager(session)
        config = RetrievalConfiguration(embedding_dimensions=512)
        await mgr.create_corpus(
            "corpus-wrong-dim",
            "Wrong Dim",
            "Desc",
            seeded_snapshots,
            retrieval_config=config,
        )
        with pytest.raises(CorpusLifecycleError, match="embedding dimension"):
            await mgr.validate_and_mark_ready("corpus-wrong-dim")


@pytest.mark.asyncio
async def test_validate_and_mark_ready_rejects_mismatched_profile_revision(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_snapshots: list[str],
) -> None:
    """A corpus whose configured revision differs from snapshot metadata fails validation."""
    from vehicle_risk_agent.policy.corpus_models import RetrievalConfiguration

    async with session_factory() as session:
        mgr = CorpusLifecycleManager(session)
        config = RetrievalConfiguration(embedding_revision="incompatible-pinned-rev")
        await mgr.create_corpus(
            "corpus-wrong-rev",
            "Wrong Rev",
            "Desc",
            seeded_snapshots,
            retrieval_config=config,
        )
        with pytest.raises(CorpusLifecycleError, match="does not match corpus revision"):
            await mgr.validate_and_mark_ready("corpus-wrong-rev")
