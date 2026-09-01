"""Typed LangGraph state and deterministic reducers."""

from typing import Annotated

from typing_extensions import TypedDict

from vehicle_risk_agent.api.models import AssessmentContext
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.domain.events import WorkflowProgressEvent
from vehicle_risk_agent.evidence.models import SafeError
from vehicle_risk_agent.evidence.snapshot import VehicleEvidenceSnapshot
from vehicle_risk_agent.evidence.sufficiency import EvidenceSufficiencyResult


def reduce_visited_phases(
    current: list[AssessmentRunPhase], new: list[AssessmentRunPhase]
) -> list[AssessmentRunPhase]:
    """Deterministically append new phases while discarding immediate consecutive duplicates."""
    result = list(current)
    for phase in new:
        if not result or result[-1] != phase:
            result.append(phase)
    return result


def reduce_events(
    current: list[WorkflowProgressEvent], new: list[WorkflowProgressEvent]
) -> list[WorkflowProgressEvent]:
    """Deterministically combine event lists without sequence collisions."""
    result = list(current)
    seen_ids = {e.event_id for e in result}
    for e in new:
        if e.event_id not in seen_ids:
            result.append(e)
            seen_ids.add(e.event_id)
    return result


class AssessmentGraphState(TypedDict, total=False):
    """Typed state dictionary for LangGraph assessment workflow."""

    assessment_id: str
    run_number: int
    vin: str
    context: AssessmentContext
    phase: AssessmentRunPhase
    visited_phases: Annotated[list[AssessmentRunPhase], reduce_visited_phases]
    events: Annotated[list[WorkflowProgressEvent], reduce_events]
    evidence_snapshot: VehicleEvidenceSnapshot | None
    sufficiency_result: EvidenceSufficiencyResult | None
    mcp_error: SafeError | None

