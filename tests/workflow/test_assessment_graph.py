"""Tests for typed LangGraph Assessment graph state, explicit phase transitions, and reducers."""

import pytest

from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.workflow.graph import build_assessment_graph
from vehicle_risk_agent.workflow.state import AssessmentGraphState


@pytest.mark.asyncio
async def test_graph_phase_transitions_happy_path() -> None:
    """Verify graph advances through the explicit phases sequentially."""
    graph = build_assessment_graph()
    app = graph.compile()

    initial_state: AssessmentGraphState = {
        "assessment_id": "asmt-001",
        "run_number": 1,
        "vin": "1HGCR2F85HA000000",
        "context": AssessmentContext(sale_type=SaleType.DEALER),
        "phase": AssessmentRunPhase.PENDING,
        "visited_phases": [AssessmentRunPhase.PENDING],
        "events": [],
    }

    result = await app.ainvoke(initial_state)

    assert result["phase"] == AssessmentRunPhase.COMPLETED
    expected_sequence = [
        AssessmentRunPhase.PENDING,
        AssessmentRunPhase.COLLECTING_EVIDENCE,
        AssessmentRunPhase.RETRIEVING_POLICY,
        AssessmentRunPhase.EVALUATING_RISK,
        AssessmentRunPhase.DRAFTING_REPORT,
        AssessmentRunPhase.COMPLETED,
    ]
    assert result["visited_phases"] == expected_sequence


@pytest.mark.asyncio
async def test_deterministic_reducers_prevent_duplicate_phases() -> None:
    """Verify reducer preserves phase progression without duplication."""
    graph = build_assessment_graph()
    app = graph.compile()

    initial_state: AssessmentGraphState = {
        "assessment_id": "asmt-002",
        "run_number": 1,
        "vin": "1HGCR2F85HA000000",
        "context": AssessmentContext(sale_type=SaleType.PRIVATE),
        "phase": AssessmentRunPhase.PENDING,
        "visited_phases": [AssessmentRunPhase.PENDING],
        "events": [],
    }

    result = await app.ainvoke(initial_state)
    phases = result["visited_phases"]
    # Check that visited_phases contains no consecutive duplicate entries
    for i in range(len(phases) - 1):
        assert phases[i] != phases[i + 1]


@pytest.mark.asyncio
async def test_graph_execution_emits_progress_events_to_state() -> None:
    """Verify graph execution emits sequential WorkflowProgressEvents into state."""
    graph = build_assessment_graph()
    app = graph.compile()

    initial_state: AssessmentGraphState = {
        "assessment_id": "asmt-events-001",
        "run_number": 1,
        "vin": "1HGCR2F85HA000000",
        "context": AssessmentContext(sale_type=SaleType.DEALER),
        "phase": AssessmentRunPhase.PENDING,
        "visited_phases": [AssessmentRunPhase.PENDING],
        "events": [],
    }

    result = await app.ainvoke(initial_state)
    events = result["events"]
    assert len(events) == 5

    expected_phases = [
        AssessmentRunPhase.COLLECTING_EVIDENCE,
        AssessmentRunPhase.RETRIEVING_POLICY,
        AssessmentRunPhase.EVALUATING_RISK,
        AssessmentRunPhase.DRAFTING_REPORT,
        AssessmentRunPhase.COMPLETED,
    ]
    for i, (evt, expected_phase) in enumerate(zip(events, expected_phases, strict=True)):
        assert evt.sequence == i + 1
        assert evt.phase == expected_phase
        assert evt.assessment_id == "asmt-events-001"
        assert evt.run_number == 1
        assert len(evt.safe_message) > 0


@pytest.mark.asyncio
async def test_graph_execution_persists_events_to_event_store_when_configured() -> None:
    """Verify graph execution automatically persists events to EventStore and broadcaster."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from vehicle_risk_agent.api.models import AssessmentCreateRequest
    from vehicle_risk_agent.events.broadcaster import ProgressEventBroadcaster
    from vehicle_risk_agent.persistence.event_store import EventStore
    from vehicle_risk_agent.persistence.models import Base
    from vehicle_risk_agent.persistence.repository import AssessmentRepository

    db_url = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"
    engine = create_async_engine(db_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    broadcaster = ProgressEventBroadcaster()
    from sqlalchemy.ext.asyncio import async_sessionmaker

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as sess:
        repo = AssessmentRepository(sess)
        req = AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        )
        asmt = await repo.create_assessment("req-01", "key-01", req)
        assessment_id = asmt.id

        store = EventStore(sess, broadcaster=broadcaster)

        graph = build_assessment_graph()
        app = graph.compile()

        initial_state: AssessmentGraphState = {
            "assessment_id": assessment_id,
            "run_number": 1,
            "vin": "1HGCR2F85HA000000",
            "context": AssessmentContext(sale_type=SaleType.DEALER),
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }

        # Run graph with EventStore in configurable
        result = await app.ainvoke(initial_state, config={"configurable": {"event_store": store}})
        assert result["phase"] == AssessmentRunPhase.COMPLETED

        # Check that EventStore in Postgres received all 5 events
        stored_events = await store.get_events(assessment_id)
        assert len(stored_events) == 5
        assert stored_events[-1].phase == AssessmentRunPhase.COMPLETED

    await engine.dispose()
