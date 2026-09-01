"""Tests for workflow routing of incomplete evidence and lookup failures."""

from datetime import UTC, datetime

import pytest

from vehicle_risk_agent.adapters.mcp import FakeVehicleMcpAdapter
from vehicle_risk_agent.api.models import AssessmentContext
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.sufficiency import SufficiencyOutcome
from vehicle_risk_agent.workflow.graph import build_assessment_graph
from vehicle_risk_agent.workflow.runner import AssessmentRunner
from vehicle_risk_agent.workflow.state import AssessmentGraphState


@pytest.fixture
def sample_complete_revision() -> VehicleRevisionResponse:
    now = datetime.now(UTC)
    return VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-1",
        revision_number=1,
        material_hash="a" * 64,
        canonical_fields={
            "make": "TOYOTA",
            "model": "YARIS",
            "year": 2017,
            "ppsr_result": "NO_FINANCE_REGISTERED",
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        },
        field_provenance={},
        conflicts=[],
        confidence=ConfidenceAssessment(
            score=90,
            band=ConfidenceBand.HIGH,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="ok",
        ),
        as_of=now,
        published_at=now,
    )


@pytest.mark.asyncio
async def test_workflow_routes_complete_evidence_to_completed(
    sample_complete_revision: VehicleRevisionResponse,
) -> None:
    """When all Required Evidence is complete, workflow continues to COMPLETED."""
    mcp_adapter = FakeVehicleMcpAdapter()
    mcp_adapter.seed_vehicle(sample_complete_revision)

    runner = AssessmentRunner(mcp_adapter=mcp_adapter)
    final_state = await runner.run(
        assessment_id="asmt-complete",
        run_number=1,
        vin="7AT0BK00X00000001",
        context=AssessmentContext(intent="purchase_risk"),
    )

    assert final_state["phase"] == AssessmentRunPhase.COMPLETED
    assert final_state["evidence_snapshot"] is not None
    assert final_state["sufficiency_result"] is not None
    assert final_state["sufficiency_result"].outcome == SufficiencyOutcome.COMPLETE


@pytest.mark.asyncio
async def test_workflow_routes_incomplete_evidence_to_incomplete_phase(
    sample_complete_revision: VehicleRevisionResponse,
) -> None:
    """When Required Evidence has missing fields (e.g. unknown stolen_status), route to INCOMPLETE."""
    fields = dict(sample_complete_revision.canonical_fields)
    fields["stolen_status"] = "UNKNOWN"
    rev = sample_complete_revision.model_copy(update={"canonical_fields": fields})

    mcp_adapter = FakeVehicleMcpAdapter()
    mcp_adapter.seed_vehicle(rev)

    runner = AssessmentRunner(mcp_adapter=mcp_adapter)
    final_state = await runner.run(
        assessment_id="asmt-incomplete",
        run_number=1,
        vin="7AT0BK00X00000001",
        context=AssessmentContext(intent="purchase_risk"),
    )

    assert final_state["phase"] == AssessmentRunPhase.INCOMPLETE
    assert final_state["evidence_snapshot"] is not None
    assert final_state["sufficiency_result"] is not None
    assert final_state["sufficiency_result"].outcome == SufficiencyOutcome.INCOMPLETE
    assert len(final_state["sufficiency_result"].missing_findings) >= 1


@pytest.mark.asyncio
async def test_workflow_routes_mcp_lookup_failure_to_failed_phase() -> None:
    """When MCP vehicle lookup fails (vehicle not found or timeout), route to FAILED."""
    mcp_adapter = FakeVehicleMcpAdapter()  # Empty adapter, lookup will fail

    runner = AssessmentRunner(mcp_adapter=mcp_adapter)
    final_state = await runner.run(
        assessment_id="asmt-not-found",
        run_number=1,
        vin="7AT0BK00X00000099",
        context=AssessmentContext(intent="purchase_risk"),
    )

    assert final_state["phase"] == AssessmentRunPhase.FAILED
    assert final_state.get("mcp_error") is not None
