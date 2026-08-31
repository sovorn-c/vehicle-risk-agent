"""LangGraph workflow definition for Assessment Runs."""

from langgraph.graph import END, START, StateGraph

from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.workflow.state import AssessmentGraphState


async def node_collecting_evidence(_state: AssessmentGraphState) -> dict[str, object]:
    """Execute evidence collection phase."""
    return {
        "phase": AssessmentRunPhase.COLLECTING_EVIDENCE,
        "visited_phases": [AssessmentRunPhase.COLLECTING_EVIDENCE],
    }


async def node_retrieving_policy(_state: AssessmentGraphState) -> dict[str, object]:
    """Execute policy retrieval phase."""
    return {
        "phase": AssessmentRunPhase.RETRIEVING_POLICY,
        "visited_phases": [AssessmentRunPhase.RETRIEVING_POLICY],
    }


async def node_evaluating_risk(_state: AssessmentGraphState) -> dict[str, object]:
    """Execute deterministic risk evaluation phase."""
    return {
        "phase": AssessmentRunPhase.EVALUATING_RISK,
        "visited_phases": [AssessmentRunPhase.EVALUATING_RISK],
    }


async def node_drafting_report(_state: AssessmentGraphState) -> dict[str, object]:
    """Execute report drafting phase."""
    return {
        "phase": AssessmentRunPhase.DRAFTING_REPORT,
        "visited_phases": [AssessmentRunPhase.DRAFTING_REPORT],
    }


async def node_complete(_state: AssessmentGraphState) -> dict[str, object]:
    """Finalize assessment run."""
    return {
        "phase": AssessmentRunPhase.COMPLETED,
        "visited_phases": [AssessmentRunPhase.COMPLETED],
    }


def build_assessment_graph() -> StateGraph:  # type: ignore[type-arg]
    """Construct the StateGraph defining the assessment workflow."""
    builder = StateGraph(AssessmentGraphState)

    builder.add_node("collecting_evidence", node_collecting_evidence)  # type: ignore[call-overload]
    builder.add_node("retrieving_policy", node_retrieving_policy)  # type: ignore[call-overload]
    builder.add_node("evaluating_risk", node_evaluating_risk)  # type: ignore[call-overload]
    builder.add_node("drafting_report", node_drafting_report)  # type: ignore[call-overload]
    builder.add_node("complete", node_complete)  # type: ignore[call-overload]

    builder.add_edge(START, "collecting_evidence")
    builder.add_edge("collecting_evidence", "retrieving_policy")
    builder.add_edge("retrieving_policy", "evaluating_risk")
    builder.add_edge("evaluating_risk", "drafting_report")
    builder.add_edge("drafting_report", "complete")
    builder.add_edge("complete", END)

    return builder
