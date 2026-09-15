"""Root pytest configuration and common database fixtures."""

import os

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from tests.database import TEST_DB_URL, ensure_test_database


def pytest_sessionstart(session: pytest.Session) -> None:
    """Create the isolated test database before database fixtures run."""
    _ = session
    ensure_test_database()
    os.environ["DATABASE_URL"] = TEST_DB_URL


@pytest_asyncio.fixture(autouse=True)
async def ensure_pgvector_extension() -> None:
    """Ensure pgvector extension is available in the test database for all tests."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))
    await engine.dispose()
