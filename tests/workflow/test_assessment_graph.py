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
