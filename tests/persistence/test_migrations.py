"""Integration tests for Alembic migration upgrade and downgrade cycles."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from vehicle_risk_agent.persistence.models import Base

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def clean_engine() -> AsyncIterator[AsyncEngine]:
    """Provide a clean PostgreSQL database with all tables dropped."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_alembic_upgrade_and_downgrade(clean_engine: AsyncEngine) -> None:
    """Verify that alembic upgrade creates all tables including workflow_events and downgrade drops them."""
    root_dir = Path(__file__).parent.parent.parent
    alembic_ini = root_dir / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini))
    alembic_cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)

    # Run upgrade head in thread to avoid event loop collision with env.py
    await asyncio.to_thread(command.upgrade, alembic_cfg, "head")

    # Inspect tables
    def inspect_tables(conn: object) -> list[str]:
        insp = inspect(conn)
        return insp.get_table_names()

    def inspect_workflow_events_columns(conn: object) -> list[str]:
        insp = inspect(conn)
        return [col["name"] for col in insp.get_columns("workflow_events")]

    async with clean_engine.connect() as conn:
        table_names = await conn.run_sync(inspect_tables)
        assert "assessments" in table_names
        assert "assessment_runs" in table_names
        assert "idempotency_keys" in table_names
        assert "workflow_events" in table_names

        columns = await conn.run_sync(inspect_workflow_events_columns)
        expected_cols = {
            "id",
            "assessment_id",
            "run_number",
            "sequence",
            "phase",
            "safe_message",
            "timestamp",
        }
        assert expected_cols.issubset(set(columns))

    # Run downgrade base
    await asyncio.to_thread(command.downgrade, alembic_cfg, "base")

    async with clean_engine.connect() as conn:
        table_names_after = await conn.run_sync(inspect_tables)
        assert "workflow_events" not in table_names_after
        assert "assessments" not in table_names_after
        assert "assessment_runs" not in table_names_after
        assert "idempotency_keys" not in table_names_after
