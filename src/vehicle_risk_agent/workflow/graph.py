"""LangGraph workflow definition for Assessment Runs."""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from vehicle_risk_agent.adapters.mcp import McpAdapterError
from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState, AssessmentRunPhase
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
from vehicle_risk_agent.observability.telemetry import (
    record_model_tokens,
    trace_boundary,
)
from vehicle_risk_agent.reporting.offline import OfflineReportDraftingAdapter
from vehicle_risk_agent.reporting.protocol import ReportDraftingContext
from vehicle_risk_agent.risk.calculator import calculate_risk_result
from vehicle_risk_agent.risk.models import RiskPolicy, build_risk_policy_v1
from vehicle_risk_agent.risk.repository import RiskPolicyRepository
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
        assessment_repo = configurable.get("assessment_repo")
        if assessment_repo is not None:
            await assessment_repo.update_run_phase(
                state["assessment_id"], state["run_number"], phase
            )

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

        field_explanations: dict[str, FieldExplanationResult] = await explain_fields_in_parallel(
            mcp_adapter, state["vin"], REQUIRED_EVIDENCE_FIELDS
        )

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


async def _resolve_policy(configurable: dict[str, Any]) -> RiskPolicy:
    """Retrieve active RiskPolicy from repository, or ensure default v1 exists in database."""
    policy_repo: RiskPolicyRepository | None = configurable.get("policy_repo")
    if policy_repo is None:
        return build_risk_policy_v1()

    active = await policy_repo.get_active_policy()
    if active is not None:
        return active

    existing = await policy_repo.get_policy("risk-policy-v1")
    if existing is not None:
        return existing

    default_policy = build_risk_policy_v1("risk-policy-v1")
    return await policy_repo.create_policy(default_policy)


async def node_incomplete(
    state: AssessmentGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Handle incomplete evidence outcome without generating risk scores."""
    result = state.get("sufficiency_result")
    missing_count = len(result.missing_findings) if result else 0
    progress = await _emit_progress(
        state,
        AssessmentRunPhase.INCOMPLETE,
        f"Assessment withheld scoring due to {missing_count} incomplete required evidence fields",
        config,
    )

    configurable = config.get("configurable", {}) if config else {}
    policy = await _resolve_policy(configurable)

    snapshot = state.get("evidence_snapshot")
    policy_citations = state.get("policy_citations") or ()

    risk_result = calculate_risk_result(
        policy=policy,
        snapshot=snapshot,
        sufficiency=result,
        assessment_id=state["assessment_id"],
        run_number=state["run_number"],
        policy_citations=policy_citations,
    )

    risk_repo = configurable.get("risk_repo")
    if risk_repo is not None:
        await risk_repo.save_result(risk_result)

    draft_context = ReportDraftingContext(
        assessment_id=state["assessment_id"],
        run_number=state["run_number"],
        vehicle_id=state.get("vin", ""),
        vin=state.get("vin", ""),
        risk_result=risk_result,
        evidence_snapshot=snapshot,
        policy_citations=tuple(policy_citations),
    )
    drafter = configurable.get("drafting_adapter") or OfflineReportDraftingAdapter()
    with trace_boundary(
        "model.draft_report",
        boundary="model",
        assessment_id=state["assessment_id"],
        run_number=state["run_number"],
        model=type(drafter).__name__,
    ):
        draft = await drafter.draft_report(draft_context)
        # Offline drafting has no provider usage; provider adapters record actual usage.
        record_model_tokens(0, 0, model=type(drafter).__name__)

    draft_repo = configurable.get("draft_repo")
    if draft_repo is not None:
        await draft_repo.save_draft_and_transition_assessment(
            draft, AssessmentLifecycleState.AWAITING_REVIEW
        )

    return {
        **progress,
        "phase": AssessmentRunPhase.INCOMPLETE,
        "risk_result": risk_result,
        "report_draft": draft,
    }


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
    progress = await _emit_progress(
        state,
        AssessmentRunPhase.RETRIEVING_POLICY,
        "Retrieving policy citations and rules",
        config,
    )
    configurable = config.get("configurable", {}) if config else {}
    citations_override = configurable.get("policy_citations")
    with trace_boundary(
        "retrieval.policy",
        boundary="retrieval",
        assessment_id=state["assessment_id"],
        run_number=state["run_number"],
    ):
        if citations_override is not None:
            citations = tuple(citations_override)
        else:
            retrieval_service = configurable.get("retrieval_service")
            if retrieval_service is None:
                citations = ()
            else:
                questions = state["context"].questions
                query = "vehicle sale consumer protection" + (
                    " " + " ".join(questions) if questions else ""
                )
                retrieval_result = await retrieval_service.retrieve(query)
                citations = tuple(retrieval_result.citations)
    return {
        **progress,
        "policy_citations": citations,
    }


async def node_evaluating_risk(
    state: AssessmentGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Execute deterministic risk evaluation phase."""
    progress = await _emit_progress(
        state,
        AssessmentRunPhase.EVALUATING_RISK,
        "Evaluating deterministic risk factors",
        config,
    )

    configurable = config.get("configurable", {}) if config else {}
    policy = await _resolve_policy(configurable)

    snapshot = state.get("evidence_snapshot")
    sufficiency = state.get("sufficiency_result")
    policy_citations = state.get("policy_citations") or ()

    risk_result = calculate_risk_result(
        policy=policy,
        snapshot=snapshot,
        sufficiency=sufficiency,
        assessment_id=state["assessment_id"],
        run_number=state["run_number"],
        policy_citations=policy_citations,
    )

    risk_repo = configurable.get("risk_repo")
    if risk_repo is not None:
        await risk_repo.save_result(risk_result)

    return {
        **progress,
        "risk_result": risk_result,
    }


async def node_drafting_report(
    state: AssessmentGraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Execute report drafting phase."""
    progress = await _emit_progress(
        state,
        AssessmentRunPhase.DRAFTING_REPORT,
        "Drafting risk assessment report",
        config,
    )

    snapshot = state.get("evidence_snapshot")
    risk_result = state.get("risk_result")
    policy_citations = state.get("policy_citations") or ()

    if risk_result is None:
        raise ValueError("Cannot draft report without risk_result")

    draft_context = ReportDraftingContext(
        assessment_id=state["assessment_id"],
        run_number=state["run_number"],
        vehicle_id=state.get("vin", ""),
        vin=state.get("vin", ""),
        risk_result=risk_result,
        evidence_snapshot=snapshot,
        policy_citations=tuple(policy_citations),
    )
    configurable = config.get("configurable", {}) if config else {}
    drafter = configurable.get("drafting_adapter") or OfflineReportDraftingAdapter()
    with trace_boundary(
        "model.draft_report",
        boundary="model",
        assessment_id=state["assessment_id"],
        run_number=state["run_number"],
        model=type(drafter).__name__,
    ):
        draft = await drafter.draft_report(draft_context)
        # Offline drafting has no provider usage; provider adapters record actual usage.
        record_model_tokens(0, 0, model=type(drafter).__name__)
    draft_repo = configurable.get("draft_repo")
    if draft_repo is not None:
        await draft_repo.save_draft_and_transition_assessment(
            draft, AssessmentLifecycleState.AWAITING_REVIEW
        )

    return {
        **progress,
        "report_draft": draft,
    }


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
