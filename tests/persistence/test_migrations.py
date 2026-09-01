"""Integration tests for Alembic migration upgrade and downgrade cycles."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
import sqlalchemy as sa
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from alembic import command

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def clean_engine() -> AsyncIterator[AsyncEngine]:
    """Provide a clean PostgreSQL database with all tables dropped."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.execute(sa.text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    yield engine
    async with engine.begin() as conn:
        await conn.execute(sa.text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    await engine.dispose()


@pytest.mark.asyncio
async def test_alembic_upgrade_and_downgrade(clean_engine: AsyncEngine) -> None:
    """Verify that alembic upgrade creates all assessment and policy tables with pgvector."""
    root_dir = Path(__file__).parent.parent.parent
    alembic_ini = root_dir / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini))
    alembic_cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)

    # Run upgrade head in thread to avoid event loop collision with env.py
    await asyncio.to_thread(command.upgrade, alembic_cfg, "head")

    # Inspect tables
    def inspect_tables(conn: sa.Connection) -> list[str]:
        insp = inspect(conn)
        assert insp is not None
        return list(insp.get_table_names())

    def inspect_columns(conn: sa.Connection, table_name: str) -> list[str]:
        insp = inspect(conn)
        assert insp is not None
        return [str(col["name"]) for col in insp.get_columns(table_name)]

    def inspect_unique_indexes(conn: sa.Connection, table_name: str) -> list[str]:
        insp = inspect(conn)
        assert insp is not None
        indexes = insp.get_indexes(table_name)
        return [str(idx["name"]) for idx in indexes if idx.get("name") is not None]

    async with clean_engine.connect() as conn:
        table_names = await conn.run_sync(inspect_tables)
        assert "assessments" in table_names
        assert "assessment_runs" in table_names
        assert "idempotency_keys" in table_names
        assert "workflow_events" in table_names
        assert "policy_sources" in table_names
        assert "policy_snapshots" in table_names
        assert "policy_passages" in table_names
        assert "policy_corpora" in table_names
        assert "policy_corpus_snapshots" in table_names

        snapshot_cols = await conn.run_sync(lambda c: inspect_columns(c, "policy_snapshots"))
        assert "metadata_json" in snapshot_cols

        passage_cols = await conn.run_sync(lambda c: inspect_columns(c, "policy_passages"))
        assert {"id", "snapshot_id", "source_id", "text", "embedding"}.issubset(set(passage_cols))

        def inspect_constraints(conn: sa.Connection, table_name: str) -> list[str]:
            insp = inspect(conn)
            assert insp is not None
            return [str(item["name"]) for item in insp.get_unique_constraints(table_name)]

        snapshot_constraints = await conn.run_sync(
            lambda c: inspect_constraints(c, "policy_snapshots")
        )
        assert "uq_snapshot_source_hash" in snapshot_constraints

        passage_constraints = await conn.run_sync(
            lambda c: inspect_constraints(c, "policy_passages")
        )
        assert "uq_passage_snapshot_seq" in passage_constraints

        corpora_indexes = await conn.run_sync(lambda c: inspect_unique_indexes(c, "policy_corpora"))
        assert "uq_policy_corpora_single_active" in corpora_indexes

    # Run downgrade base
    await asyncio.to_thread(command.downgrade, alembic_cfg, "base")

    async with clean_engine.connect() as conn:
        table_names_after = await conn.run_sync(inspect_tables)
        assert "policy_corpora" not in table_names_after
        assert "policy_passages" not in table_names_after
        assert "policy_snapshots" not in table_names_after
        assert "policy_sources" not in table_names_after
        assert "workflow_events" not in table_names_after
        assert "assessments" not in table_names_after
