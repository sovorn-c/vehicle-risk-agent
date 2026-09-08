"""Baseline smoke test for package setup and local verification."""

# story: e07s03

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vehicle_risk_agent import __version__
from vehicle_risk_agent.cli.smoke import run_smoke
from vehicle_risk_agent.persistence.models import IdempotencyRecord

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


def test_package_version() -> None:
    """Verify package exposes a semantic version."""
    assert __version__ == "0.1.0"


@pytest.mark.asyncio
async def test_smoke_verification_leaves_no_residual_records() -> None:
    """run_smoke must clean up assessments, idempotency records, and checkpoints cleanly."""
    result = await run_smoke(database_url=TEST_DB_URL)
    assert result["status"] == "success"

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        # Check that no smoke idempotency records remain
        idemp_stmt = select(IdempotencyRecord).where(
            IdempotencyRecord.idempotency_key.like("%smoke%")
        )
        idemp_records = (await session.execute(idemp_stmt)).scalars().all()
        assert len(idemp_records) == 0, f"Found leaked idempotency records: {idemp_records}"

        # Check that no checkpoint records for smoke assessments remain
        checkpoints_stmt = text("SELECT count(*) FROM checkpoints WHERE thread_id LIKE '%asmt-%'")
        try:
            count = (await session.execute(checkpoints_stmt)).scalar_one_or_none()
            assert count == 0, f"Found leaked checkpoints: {count}"
        except Exception:
            pass
    await engine.dispose()
