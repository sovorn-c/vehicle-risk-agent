"""Durable investigation reservation and crash-recovery contracts."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tests.database import TEST_DB_URL
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.investigation.budget import InvestigationLimits
from vehicle_risk_agent.investigation.models import (
    InvestigationAction,
    InvestigationLimitation,
    InvestigationResult,
    VehicleHistoryResult,
)
from vehicle_risk_agent.investigation.repository import (
    InvestigationLedgerRepository,
    InvestigationLedgerStatus,
)
from vehicle_risk_agent.persistence.models import AssessmentRecord, Base
from vehicle_risk_agent.policy.models import PolicyCitation


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add_all(
            [
                AssessmentRecord(
                    id="asmt-ledger-1",
                    requester_id="tester",
                    vin="1HGCR2F85HA000000",
                    context_json="{}",
                ),
                AssessmentRecord(
                    id="asmt-ledger-2",
                    requester_id="tester",
                    vin="1HGCR2F85HA000000",
                    context_json="{}",
                ),
            ]
        )
        await session.commit()
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

    revision = VehicleRevisionResponse(
        vin="1HGCR2F85HA000000",
        revision_id="rev-1",
        revision_number=1,
        material_hash="a" * 64,
        canonical_fields={"make": "Honda"},
        confidence=ConfidenceAssessment(
            score=90,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="verified",
        ),
        as_of=datetime.now(UTC),
        published_at=datetime.now(UTC),
    )
    investigation_result = InvestigationResult(
        action=InvestigationAction.GET_VEHICLE_HISTORY,
        summary="history returned",
        references=("rev-1",),
        evidence_result=VehicleHistoryResult(
            vin=revision.vin,
            revisions=(revision,),
        ),
        policy_citations=(
            PolicyCitation(
                source_id="source-1",
                snapshot_id="snapshot-1",
                passage_id="passage-1",
                section_identifier="section-1",
                heading="Vehicle history",
                source_title="Vehicle policy",
                canonical_origin="https://example.test/policy",
            ),
        ),
        completed=True,
        dispatched=True,
    )
    completed = await repo.complete_action(
        assessment_id="asmt-ledger-1",
        run_number=1,
        request_hash="a" * 64,
        actual_cost=0.02,
        result_summary="field explanation available",
        references=("obs-1",),
        result=investigation_result,
    )
    assert completed.status == InvestigationLedgerStatus.COMPLETED
    replay = await repo.reserve_action(
        assessment_id="asmt-ledger-1",
        run_number=1,
        request_hash="a" * 64,
        projected_cost=0.01,
    )
    assert replay.status == InvestigationLedgerStatus.COMPLETED
    assert replay.result == investigation_result
    assert replay.result is not None
    assert replay.result.evidence_result == investigation_result.evidence_result
    assert replay.result.policy_citations == investigation_result.policy_citations


@pytest.mark.asyncio
async def test_failed_result_replays_with_limitation_and_incomplete_state(
    session: AsyncSession,
) -> None:
    repo = InvestigationLedgerRepository(session)
    await repo.ensure_ledger(
        assessment_id="asmt-ledger-2",
        run_number=1,
        vin="1HGCR2F85HA000000",
        pins={},
        limits=InvestigationLimits.first_slice(),
    )
    await repo.reserve_action(
        assessment_id="asmt-ledger-2",
        run_number=1,
        request_hash="d" * 64,
        projected_cost=0.01,
    )
    investigation_result = InvestigationResult(
        action=InvestigationAction.GET_VEHICLE_HISTORY,
        summary="history unavailable",
        limitation=InvestigationLimitation(
            code="UPSTREAM_UNAVAILABLE",
            message="The upstream history call was unavailable.",
        ),
        completed=False,
        dispatched=True,
    )

    completed = await repo.complete_action(
        assessment_id="asmt-ledger-2",
        run_number=1,
        request_hash="d" * 64,
        actual_cost=0.01,
        result_summary=investigation_result.summary,
        references=(),
        result=investigation_result,
    )

    assert completed.status == InvestigationLedgerStatus.INDETERMINATE
    replay = await repo.get_ledger("asmt-ledger-2", 1)
    assert replay is not None
    assert replay.status == InvestigationLedgerStatus.INDETERMINATE
    assert replay.result == investigation_result


@pytest.mark.asyncio
async def test_proposal_and_retry_reservations_are_monotonic(session: AsyncSession) -> None:
    repo = InvestigationLedgerRepository(session)
    await repo.ensure_ledger(
        assessment_id="asmt-ledger-2",
        run_number=1,
        vin="1HGCR2F85HA000000",
        pins={},
        limits=InvestigationLimits.final(),
    )
    proposal = await repo.reserve_proposal("asmt-ledger-2", 1)
    assert proposal.status == InvestigationLedgerStatus.COUNTING
    ready = await repo.complete_proposal("asmt-ledger-2", 1, 10, 5)
    assert ready.status == InvestigationLedgerStatus.READY
    action = await repo.reserve_action("asmt-ledger-2", 1, "c" * 64, 0.01)
    assert action.supplementary_attempts == 1
    retry = await repo.reserve_retry("asmt-ledger-2", 1, "c" * 64, 0.01)
    assert retry.supplementary_attempts == 2
    assert retry.supplementary_retries == 1


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
