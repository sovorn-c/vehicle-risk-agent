"""Contracts for keeping destructive tests away from the live database."""

import os

import pytest
from sqlalchemy.engine import make_url

from tests import database
from tests.database import TEST_DB_URL


def test_tests_use_a_dedicated_database() -> None:
    """Test cleanup must target a dedicated database, never the runtime database."""
    database = make_url(TEST_DB_URL).database
    assert database is not None
    assert database.endswith("_test")
    assert database != "postgres"
    assert os.environ["DATABASE_URL"] == TEST_DB_URL


def test_test_database_setup_rejects_the_runtime_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misconfigured destructive test target must fail before connecting."""
    monkeypatch.setattr(
        database,
        "TEST_DB_URL",
        "postgresql+psycopg://postgres:postgres@localhost:54329/postgres",
    )

    with pytest.raises(RuntimeError, match=r"must name a \*_test database"):
        database.ensure_test_database()


def test_test_database_setup_rejects_a_matching_runtime_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured runtime database name cannot equal the test database name."""
    monkeypatch.setenv("DATABASE_URL", TEST_DB_URL)

    with pytest.raises(RuntimeError, match="must not be the configured runtime database"):
        database.ensure_test_database()
