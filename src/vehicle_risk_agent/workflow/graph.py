"""LangGraph workflow definition for Assessment Runs."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.domain.events import WorkflowProgressEvent
from vehicle_risk_agent.workflow.state import AssessmentGraphState


async def _emit_progress(
    state: AssessmentGraphState,
    phase: AssessmentRunPhase,
    safe_message: str,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Construct a progress event and optionally persist to EventStore."""
    seq = len(state.get("events", [])) + 1
    evt = WorkflowProgressEvent(
        event_id=str(uuid4()),
        sequence=seq,
        assessment_id=state["assessment_id"],
        run_number=state["run_number"],
        phase=phase,
        safe_message=safe_message,
        timestamp=datetime.now(UTC),
    )
    if config:
        configurable = config.get("configurable", {})
        store = configurable.get("event_store")
        if store is not None:
            await store.append_event(evt)

    return {
        "phase": phase,
        "visited_phases": [phase],
        "events": [evt],
    }


async def node_collecting_evidence(
    state: AssessmentGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Execute evidence collection phase."""
    return await _emit_progress(
        state,
        AssessmentRunPhase.COLLECTING_EVIDENCE,
        "Collecting vehicle facts and history from MCP tools",
        config,
    )


async def node_retrieving_policy(
    state: AssessmentGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Execute policy retrieval phase."""
    return await _emit_progress(
        state,
        AssessmentRunPhase.RETRIEVING_POLICY,
        "Retrieving policy citations and rules",
        config,
    )


async def node_evaluating_risk(
    state: AssessmentGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Execute deterministic risk evaluation phase."""
    return await _emit_progress(
        state,
        AssessmentRunPhase.EVALUATING_RISK,
        "Evaluating deterministic risk factors",
        config,
    )


async def node_drafting_report(
    state: AssessmentGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Execute report drafting phase."""
    return await _emit_progress(
        state,
        AssessmentRunPhase.DRAFTING_REPORT,
        "Drafting risk assessment report",
        config,
    )


async def node_complete(
    state: AssessmentGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Finalize assessment run."""
    return await _emit_progress(
        state,
        AssessmentRunPhase.COMPLETED,
        "Assessment workflow completed successfully",
        config,
    )


def build_assessment_graph() -> StateGraph:  # type: ignore[type-arg]
    """Construct the StateGraph defining the assessment workflow."""
    builder = StateGraph(AssessmentGraphState)

    builder.add_node("collecting_evidence", node_collecting_evidence)
    builder.add_node("retrieving_policy", node_retrieving_policy)
    builder.add_node("evaluating_risk", node_evaluating_risk)
    builder.add_node("drafting_report", node_drafting_report)
    builder.add_node("complete", node_complete)

    builder.add_edge(START, "collecting_evidence")
    builder.add_edge("collecting_evidence", "retrieving_policy")
    builder.add_edge("retrieving_policy", "evaluating_risk")
    builder.add_edge("evaluating_risk", "drafting_report")
    builder.add_edge("drafting_report", "complete")
    builder.add_edge("complete", END)

    return builder
