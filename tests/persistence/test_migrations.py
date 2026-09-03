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
        assert "vehicle_evidence_snapshots" in table_names
        assert "risk_policies" in table_names
        assert "risk_results" in table_names
        assert "report_drafts" in table_names
        assert "review_actions" in table_names

        snapshot_cols = await conn.run_sync(lambda c: inspect_columns(c, "policy_snapshots"))
        assert "metadata_json" in snapshot_cols

        evidence_snapshot_cols = await conn.run_sync(
            lambda c: inspect_columns(c, "vehicle_evidence_snapshots")
        )
        assert "snapshot_integrity_hash" in evidence_snapshot_cols

        risk_policy_cols = await conn.run_sync(lambda c: inspect_columns(c, "risk_policies"))
        assert {
            "id",
            "version",
            "name",
            "lifecycle_state",
            "factor_weights_json",
            "score_cap",
            "risk_bands_json",
        }.issubset(set(risk_policy_cols))

        risk_result_cols = await conn.run_sync(lambda c: inspect_columns(c, "risk_results"))
        assert {
            "id",
            "assessment_id",
            "run_number",
            "policy_id",
            "score",
            "band",
            "result_data_json",
        }.issubset(set(risk_result_cols))

        report_draft_cols = await conn.run_sync(lambda c: inspect_columns(c, "report_drafts"))
        assert {
            "id",
            "assessment_id",
            "run_number",
            "vehicle_id",
            "policy_id",
            "policy_version",
            "outcome",
            "score",
            "band",
            "draft_hash",
            "draft_data_json",
            "created_at",
        }.issubset(set(report_draft_cols))

        review_action_cols = await conn.run_sync(lambda c: inspect_columns(c, "review_actions"))
        assert {
            "id",
            "assessment_id",
            "run_number",
            "reviewer_id",
            "action_type",
            "disposition",
            "idempotency_key",
            "rationale",
            "notes",
            "acknowledge_missing_evidence",
            "action_hash",
            "created_at",
        }.issubset(set(review_action_cols))

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

        risk_result_constraints = await conn.run_sync(
            lambda c: inspect_constraints(c, "risk_results")
        )
        assert "uq_risk_results_run_id" in risk_result_constraints

        report_draft_constraints = await conn.run_sync(
            lambda c: inspect_constraints(c, "report_drafts")
        )
        assert "uq_report_draft_run" in report_draft_constraints

        review_action_constraints = await conn.run_sync(
            lambda c: inspect_constraints(c, "review_actions")
        )
        assert "uq_review_action_draft_run" in review_action_constraints

        corpora_indexes = await conn.run_sync(lambda c: inspect_unique_indexes(c, "policy_corpora"))
        assert "uq_policy_corpora_single_active" in corpora_indexes

        risk_policy_indexes = await conn.run_sync(
            lambda c: inspect_unique_indexes(c, "risk_policies")
        )
        assert "uq_risk_policies_single_active" in risk_policy_indexes

    # Run downgrade base
    await asyncio.to_thread(command.downgrade, alembic_cfg, "base")

    async with clean_engine.connect() as conn:
        table_names_after = await conn.run_sync(inspect_tables)
        assert "review_actions" not in table_names_after
        assert "report_drafts" not in table_names_after
        assert "risk_results" not in table_names_after
        assert "risk_policies" not in table_names_after
        assert "policy_corpora" not in table_names_after
        assert "policy_passages" not in table_names_after
        assert "policy_snapshots" not in table_names_after
        assert "policy_sources" not in table_names_after
        assert "workflow_events" not in table_names_after
        assert "assessments" not in table_names_after
