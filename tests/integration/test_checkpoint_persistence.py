"""Integration tests for LangGraph PostgreSQL checkpoint persistence with runner."""

from datetime import UTC, datetime

import pytest
from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import create_async_engine

from vehicle_risk_agent.adapters.mcp import (
    FakeVehicleMcpAdapter,
    StreamableHttpVehicleMcpAdapter,
)
from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.persistence.models import Base
from vehicle_risk_agent.workflow.runner import AssessmentWorkflowRunner
from vehicle_risk_agent.workflow.state import AssessmentGraphState

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest.fixture
def fake_mcp_adapter() -> FakeVehicleMcpAdapter:
    adapter = FakeVehicleMcpAdapter()
    now = datetime.now(UTC)
    rev = VehicleRevisionResponse(
        vin="1HGCR2F85HA000000",
        revision_id="rev-001",
        revision_number=1,
        material_hash="a" * 64,
        canonical_fields={
            "make": "HONDA",
            "model": "ACCORD",
            "year": 2017,
            "ppsr_result": "NO_FINANCE_REGISTERED",
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        },
        field_provenance={},
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=90,
            band=ConfidenceBand.HIGH,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="verified",
        ),
        as_of=now,
        published_at=now,
    )
    adapter.seed_vehicle(rev)
    return adapter


@pytest.mark.asyncio
async def test_checkpoint_persistence_under_stable_run_identifier(
    fake_mcp_adapter: FakeVehicleMcpAdapter,
) -> None:
    """Verify workflow state is saved in PostgreSQL under thread_id = assessment_id:run_number."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

    settings = Settings(database_url=TEST_DB_URL)
    async with AssessmentWorkflowRunner.create(settings, mcp_adapter=fake_mcp_adapter) as runner:
        thread_id = runner.get_thread_id("asmt-runner-001", 1)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "mcp_adapter": fake_mcp_adapter,
            }
        }

        initial_state: AssessmentGraphState = {
            "assessment_id": "asmt-runner-001",
            "run_number": 1,
            "vin": "1HGCR2F85HA000000",
            "context": AssessmentContext(sale_type=SaleType.DEALER),
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }

        result = await runner.app.ainvoke(initial_state, config=config)
        assert result["phase"] == AssessmentRunPhase.COMPLETED

        # Inspect checkpoint saved in database
        saved_state = await runner.app.aget_state(config)
        assert saved_state is not None
        assert saved_state.values["phase"] == AssessmentRunPhase.COMPLETED
        assert saved_state.values["assessment_id"] == "asmt-runner-001"


@pytest.mark.asyncio
async def test_runner_run_automatically_injects_thread_and_persists_events_to_event_store(
    fake_mcp_adapter: FakeVehicleMcpAdapter,
) -> None:
    """Verify AssessmentWorkflowRunner.run automatically manages thread_id and EventStore."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from vehicle_risk_agent.events.broadcaster import ProgressEventBroadcaster
    from vehicle_risk_agent.persistence.event_store import EventStore

    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    settings = Settings(database_url=TEST_DB_URL)
    broadcaster = ProgressEventBroadcaster()
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    from vehicle_risk_agent.api.models import AssessmentCreateRequest
    from vehicle_risk_agent.persistence.repository import AssessmentRepository

    async with session_factory() as sess:
        repo = AssessmentRepository(sess)
        req = AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        )
        asmt = await repo.create_assessment("req-01", "key-auto-01", req)
        assessment_id = asmt.id

    async with AssessmentWorkflowRunner.create(
        settings,
        session_factory=session_factory,
        broadcaster=broadcaster,
        mcp_adapter=fake_mcp_adapter,
    ) as runner:
        initial_state: AssessmentGraphState = {
            "assessment_id": assessment_id,
            "run_number": 1,
            "vin": "1HGCR2F85HA000000",
            "context": AssessmentContext(sale_type=SaleType.DEALER),
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }

        # Caller calls runner.run directly without manual RunnableConfig or EventStore plumbing
        result = await runner.run(initial_state)
        assert result["phase"] == AssessmentRunPhase.COMPLETED

        # Check that EventStore in PostgreSQL automatically received all 6 workflow events
        async with session_factory() as sess:
            store = EventStore(sess, broadcaster=broadcaster)
            events = await store.get_events(assessment_id)
            assert len(events) == 6
            assert events[0].phase == AssessmentRunPhase.COLLECTING_EVIDENCE
            assert events[-1].phase == AssessmentRunPhase.COMPLETED

    await engine.dispose()


@pytest.mark.asyncio
async def test_workflow_runner_uses_configured_mcp_adapter() -> None:
    """Runner construction wires the real MCP adapter when configured."""
    settings = Settings(database_url=TEST_DB_URL, mcp_server_url="http://mcp:8000/mcp")
    async with AssessmentWorkflowRunner.create(settings) as runner:
        assert isinstance(runner.mcp_adapter, StreamableHttpVehicleMcpAdapter)
