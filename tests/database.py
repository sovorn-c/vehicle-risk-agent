"""Database setup shared by PostgreSQL integration tests."""

from __future__ import annotations

import os

import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url

from vehicle_risk_agent.config import Settings

_DEFAULT_TEST_DB_URL = (
    "postgresql+psycopg://postgres:postgres@localhost:54329/vehicle_risk_agent_test"
)
TEST_DB_URL = os.environ.get("VEHICLE_RISK_AGENT_TEST_DATABASE_URL", _DEFAULT_TEST_DB_URL)


def ensure_test_database() -> None:
    """Create the dedicated test database and reject the runtime database."""
    test_url = make_url(TEST_DB_URL)
    test_database = test_url.database
    if not test_database or test_database == "postgres" or not test_database.endswith("_test"):
        raise RuntimeError("VEHICLE_RISK_AGENT_TEST_DATABASE_URL must name a *_test database")

    runtime_url_value = os.environ.get("DATABASE_URL") or Settings().database_url
    if runtime_url_value:
        runtime_url = make_url(runtime_url_value)
        if runtime_url.database == test_database:
            raise RuntimeError("test database must not be the configured runtime database")

    admin_url = test_url.set(drivername="postgresql", database="postgres")
    with psycopg.connect(
        admin_url.render_as_string(hide_password=False),
        autocommit=True,
    ) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (test_database,),
        ).fetchone()
        if exists is None:
            connection.execute(
                sql.SQL("CREATE DATABASE {}\n").format(sql.Identifier(test_database))
            )
