"""LangGraph workflow definition for Assessment Runs."""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from vehicle_risk_agent.adapters.mcp import McpAdapterError
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.domain.events import WorkflowProgressEvent
from vehicle_risk_agent.evidence.history import collect_vehicle_history
from vehicle_risk_agent.evidence.models import (
    FieldExplanationResult,
    SafeError,
    SafeErrorCategory,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.parallel import explain_fields_in_parallel
from vehicle_risk_agent.evidence.snapshot import create_evidence_snapshot
from vehicle_risk_agent.evidence.sufficiency import (
    REQUIRED_EVIDENCE_FIELDS,
    SufficiencyOutcome,
    evaluate_evidence_sufficiency,
)
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
    """Execute evidence collection phase through MCP adapter."""
    progress = await _emit_progress(
        state,
        AssessmentRunPhase.COLLECTING_EVIDENCE,
        "Collecting vehicle facts and history from MCP tools",
        config,
    )

    configurable = config.get("configurable", {}) if config else {}
    mcp_adapter = configurable.get("mcp_adapter")

    if mcp_adapter is None:
        safe_err = SafeError(
            category=SafeErrorCategory.PIPELINE_UNAVAILABLE,
            message="Vehicle intelligence MCP adapter is not configured or unavailable",
            retryable=True,
            remediation=(
                "Ensure MCP client adapter is initialized and provided in runner configuration."
            ),
        )
        return {
            **progress,
            "phase": AssessmentRunPhase.FAILED,
            "mcp_error": safe_err,
        }

    try:
        revision: VehicleRevisionResponse = await mcp_adapter.lookup_vehicle(state["vin"])

        history: tuple[VehicleRevisionResponse, ...] = ()
        if revision.revision_number > 1:
            history_list = await collect_vehicle_history(mcp_adapter, current_revision=revision)
            history = tuple(history_list)

        field_explanations: dict[str, FieldExplanationResult] = {}
        try:
            field_explanations = await explain_fields_in_parallel(
                mcp_adapter, state["vin"], REQUIRED_EVIDENCE_FIELDS
            )
        except Exception:
            # Fallback if field explanation is partially unavailable
            field_explanations = {}

        snapshot = create_evidence_snapshot(
            assessment_id=state["assessment_id"],
            run_number=state["run_number"],
            revision=revision,
            history=history,
            field_explanations=field_explanations,
        )

        evidence_repo = configurable.get("evidence_repo")
        if evidence_repo is not None:
            await evidence_repo.save_snapshot(snapshot)

        return {
            **progress,
            "evidence_snapshot": snapshot,
        }
    except McpAdapterError as e:
        safe_err = SafeError(
            category=e.category,
            message=e.message,
            retryable=e.retryable,
            remediation=e.remediation,
        )
        return {
            **progress,
            "phase": AssessmentRunPhase.FAILED,
            "mcp_error": safe_err,
        }
    except Exception:
        safe_err = SafeError(
            category=SafeErrorCategory.INTERNAL_ERROR,
            message="Unexpected error collecting vehicle evidence",
            retryable=False,
            remediation="Contact support or retry the assessment.",
        )
        return {
            **progress,
            "phase": AssessmentRunPhase.FAILED,
            "mcp_error": safe_err,
        }


async def node_evaluating_sufficiency(
    state: AssessmentGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Evaluate evidence sufficiency against risk policy rules."""
    progress = await _emit_progress(
        state,
        AssessmentRunPhase.EVALUATING_SUFFICIENCY,
        "Evaluating evidence sufficiency against required policy fields",
        config,
    )
    snapshot = state.get("evidence_snapshot")
    mcp_error = state.get("mcp_error")
    sufficiency_result = evaluate_evidence_sufficiency(snapshot, failure_error=mcp_error)

    configurable = config.get("configurable", {}) if config else {}
    evidence_repo = configurable.get("evidence_repo")
    if evidence_repo is not None and snapshot is not None:
        await evidence_repo.save_sufficiency_result(
            state["assessment_id"], state["run_number"], sufficiency_result
        )

    return {
        **progress,
        "sufficiency_result": sufficiency_result,
    }


async def node_incomplete(
    state: AssessmentGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Handle incomplete evidence outcome without generating risk scores."""
    result = state.get("sufficiency_result")
    missing_count = len(result.missing_findings) if result else 0
    return await _emit_progress(
        state,
        AssessmentRunPhase.INCOMPLETE,
        f"Assessment withheld scoring due to {missing_count} incomplete required evidence fields",
        config,
    )


async def node_failed(
    state: AssessmentGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Handle failed assessment run with safe, sanitized message."""
    err = state.get("mcp_error")
    msg = err.message if err else "Assessment run failed"
    return await _emit_progress(
        state,
        AssessmentRunPhase.FAILED,
        msg,
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


def route_after_evidence(
    state: AssessmentGraphState,
) -> Literal["node_failed", "node_evaluating_sufficiency"]:
    """Route to failed node if evidence lookup failed, otherwise evaluate sufficiency."""
    if state.get("phase") == AssessmentRunPhase.FAILED:
        return "node_failed"
    return "node_evaluating_sufficiency"


def route_after_sufficiency(
    state: AssessmentGraphState,
) -> Literal["node_incomplete", "node_failed", "node_retrieving_policy"]:
    """Route strictly based on evidence sufficiency outcome."""
    result = state.get("sufficiency_result")
    if result is None or result.outcome == SufficiencyOutcome.UNAVAILABLE:
        return "node_failed"
    if result.outcome == SufficiencyOutcome.INCOMPLETE:
        return "node_incomplete"
    if result.outcome == SufficiencyOutcome.COMPLETE:
        return "node_retrieving_policy"
    return "node_failed"


def build_assessment_graph() -> StateGraph:  # type: ignore[type-arg]
    """Construct the StateGraph defining the assessment workflow."""
    builder = StateGraph(AssessmentGraphState)

    builder.add_node("collecting_evidence", node_collecting_evidence)
    builder.add_node("evaluating_sufficiency", node_evaluating_sufficiency)
    builder.add_node("incomplete", node_incomplete)
    builder.add_node("failed", node_failed)
    builder.add_node("retrieving_policy", node_retrieving_policy)
    builder.add_node("evaluating_risk", node_evaluating_risk)
    builder.add_node("drafting_report", node_drafting_report)
    builder.add_node("complete", node_complete)

    builder.add_edge(START, "collecting_evidence")
    builder.add_conditional_edges(
        "collecting_evidence",
        route_after_evidence,
        {
            "node_failed": "failed",
            "node_evaluating_sufficiency": "evaluating_sufficiency",
        },
    )
    builder.add_conditional_edges(
        "evaluating_sufficiency",
        route_after_sufficiency,
        {
            "node_incomplete": "incomplete",
            "node_failed": "failed",
            "node_retrieving_policy": "retrieving_policy",
        },
    )
    builder.add_edge("incomplete", END)
    builder.add_edge("failed", END)
    builder.add_edge("retrieving_policy", "evaluating_risk")
    builder.add_edge("evaluating_risk", "drafting_report")
    builder.add_edge("drafting_report", "complete")
    builder.add_edge("complete", END)

    return builder
