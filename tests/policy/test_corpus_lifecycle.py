"""Tests for Policy Corpus lifecycle state transitions, validation, and single-active-corpus transactional invariant."""

from datetime import UTC, datetime

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
    RetrievalConfiguration,
    build_corpus_manifest,
)
from vehicle_risk_agent.policy.ingestion import PolicyParser, ingest_policy_source
from vehicle_risk_agent.policy.models import (
    AuthorityClassification,
    PolicySource,
    SourceStatus,
)

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def session_factory() -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def seeded_snapshots(session_factory: async_sessionmaker[AsyncSession]) -> list[str]:
    async with session_factory() as session:
        repo = PolicyRepository(session)
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
