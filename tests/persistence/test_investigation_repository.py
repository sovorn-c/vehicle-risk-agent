"""Durable investigation reservation and crash-recovery contracts."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.investigation.budget import InvestigationLimits
from vehicle_risk_agent.investigation.repository import (
    InvestigationLedgerRepository,
    InvestigationLedgerStatus,
)
from vehicle_risk_agent.persistence.models import Base

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_reservation_commits_before_io_and_completed_result_replays(
    session: AsyncSession,
) -> None:
    repo = InvestigationLedgerRepository(session)
    ledger = await repo.ensure_ledger(
        assessment_id="asmt-ledger-1",
        run_number=1,
        vin="1HGCR2F85HA000000",
        pins={"model_version": "claude-sonnet-4-6", "prompt_version": "e11-v1"},
        limits=InvestigationLimits.first_slice(),
    )
    assert ledger.status == InvestigationLedgerStatus.READY

    reservation = await repo.reserve_action(
        assessment_id="asmt-ledger-1",
        run_number=1,
        request_hash="a" * 64,
        projected_cost=0.01,
    )
    assert reservation.status == InvestigationLedgerStatus.ACTION_IN_FLIGHT
    assert reservation.supplementary_attempts == 1

    completed = await repo.complete_action(
        assessment_id="asmt-ledger-1",
        run_number=1,
        request_hash="a" * 64,
        actual_cost=0.02,
        result_summary="field explanation available",
        references=("obs-1",),
    )
    assert completed.status == InvestigationLedgerStatus.COMPLETED
    replay = await repo.reserve_action(
        assessment_id="asmt-ledger-1",
        run_number=1,
        request_hash="a" * 64,
        projected_cost=0.01,
    )
    assert replay.status == InvestigationLedgerStatus.COMPLETED
    assert replay.references == ("obs-1",)


@pytest.mark.asyncio
async def test_inflight_recovery_is_terminal_and_does_not_repeat_io(
    session: AsyncSession,
) -> None:
    repo = InvestigationLedgerRepository(session)
    await repo.ensure_ledger(
        assessment_id="asmt-ledger-2",
        run_number=1,
        vin="1HGCR2F85HA000000",
        pins={},
        limits=InvestigationLimits.first_slice(),
        now=datetime.now(UTC) - timedelta(seconds=2),
    )
    await repo.reserve_action(
        assessment_id="asmt-ledger-2",
        run_number=1,
        request_hash="b" * 64,
        projected_cost=0.01,
    )
    recovered = await repo.recover_inflight("asmt-ledger-2", 1)
    assert recovered.status == InvestigationLedgerStatus.INDETERMINATE
    assert recovered.current_request_hash == "b" * 64