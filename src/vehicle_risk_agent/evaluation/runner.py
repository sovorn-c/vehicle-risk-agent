"""Scenario runner for deterministic offline evaluation through fake adapters."""

# story: e06s01
# story: e06s02

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from vehicle_risk_agent.adapters.mcp import (
    FakeVehicleMcpAdapter,
    McpAdapterError,
    VehicleMcpClientAdapter,
)
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.domain.events import WorkflowProgressEvent
from vehicle_risk_agent.evaluation.models import EvaluationProvenance, EvaluationScenario
from vehicle_risk_agent.evidence.models import (
    FieldExplanationResult,
    SafeError,
    SourceObservationResponse,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.sufficiency import EvidenceSufficiencyResult
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.reporting.models import ReportDraft
from vehicle_risk_agent.reporting.offline import OfflineReportDraftingAdapter
from vehicle_risk_agent.risk.models import RiskResult
from vehicle_risk_agent.workflow.graph import build_assessment_graph
from vehicle_risk_agent.workflow.state import AssessmentGraphState

DEFAULT_PINNED_CLOCK = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)


class FailingMcpAdapter(VehicleMcpClientAdapter):
    """Simulated MCP adapter that unconditionally raises a configured SafeError."""

    def __init__(self, safe_error: SafeError) -> None:
        self.safe_error = safe_error

    async def lookup_vehicle(self, vin: str) -> VehicleRevisionResponse:
        del vin
        raise McpAdapterError(
            category=self.safe_error.category,
            message=self.safe_error.message,
            retryable=self.safe_error.retryable,
            remediation=self.safe_error.remediation,
        )

    async def get_vehicle_revision(self, vin: str, revision_number: int) -> VehicleRevisionResponse:
        del revision_number
        return await self.lookup_vehicle(vin)

    async def explain_vehicle_field(self, vin: str, field_name: str) -> FieldExplanationResult:
        del vin, field_name
        raise McpAdapterError(
            category=self.safe_error.category,
            message=self.safe_error.message,
            retryable=self.safe_error.retryable,
            remediation=self.safe_error.remediation,
        )

    async def get_vehicle_history(
        self, vin: str, limit: int = 20, before_revision: int | None = None
    ) -> list[VehicleRevisionResponse]:
        del vin, limit, before_revision
        return []

    async def get_source_observation(self, observation_id: str) -> SourceObservationResponse:
        del observation_id
        raise McpAdapterError(
            category=self.safe_error.category,
            message=self.safe_error.message,
            retryable=self.safe_error.retryable,
            remediation=self.safe_error.remediation,
        )


class ScenarioExecutionResult(BaseModel):
    """Traceable, immutable result of executing an evaluation scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario: EvaluationScenario
    provenance: EvaluationProvenance
    timestamp: datetime
    final_phase: AssessmentRunPhase
    sufficiency_result: EvidenceSufficiencyResult | None = None
    risk_result: RiskResult | None = None
    policy_citations: tuple[PolicyCitation, ...] = ()
    report_draft: ReportDraft | None = None
    mcp_error: SafeError | None = None
    events: tuple[WorkflowProgressEvent, ...] = ()


class ScenarioRunner:
    """Orchestrates deterministic scenario execution against the LangGraph workflow."""

    def __init__(
        self,
        clock: Callable[[], datetime] | None = None,
        default_provenance: EvaluationProvenance | None = None,
    ) -> None:
        self.clock = clock or (lambda: DEFAULT_PINNED_CLOCK)
        self.default_provenance = default_provenance or EvaluationProvenance(
            scenario_version="1.0.0",
            corpus_version="corpus-2026.1",
            risk_policy_version="risk-policy-v1",
            provider="fake",
            prompt_version="prompt-v1",
            grader_version="grader-v1",
            code_version="0.1.0",
        )
        self._graph = build_assessment_graph()
        self._app = self._graph.compile()

    def _build_mcp_adapter(self, scenario: EvaluationScenario) -> VehicleMcpClientAdapter:
        """Configure fake MCP adapter based on scenario specifications."""
        if scenario.mock_mcp_error is not None:
            return FailingMcpAdapter(scenario.mock_mcp_error)

        fake_mcp = FakeVehicleMcpAdapter()
        for rev in sorted(scenario.mock_vehicle_revisions, key=lambda r: r.revision_number):
            fake_mcp.seed_vehicle(rev)
        for explanation in scenario.mock_field_explanations.values():
            fake_mcp.seed_field_explanation(explanation)
        return fake_mcp

    async def run_scenario(
        self,
        scenario: EvaluationScenario,
        overrides: dict[str, Any] | None = None,
    ) -> ScenarioExecutionResult:
        """Execute a single evaluation scenario end-to-end with pinned clock and adapters."""
        current_time = self.clock()
        provenance = EvaluationProvenance(
            scenario_version=scenario.version,
            corpus_version=self.default_provenance.corpus_version,
            risk_policy_version=self.default_provenance.risk_policy_version,
            provider=self.default_provenance.provider,
            prompt_version=self.default_provenance.prompt_version,
            grader_version=self.default_provenance.grader_version,
            code_version=self.default_provenance.code_version,
        )

        mcp_adapter = self._build_mcp_adapter(scenario)
        configurable: dict[str, Any] = {
            "mcp_adapter": mcp_adapter,
            "policy_citations": scenario.mock_citations,
            "drafting_adapter": OfflineReportDraftingAdapter(),
        }
        if overrides:
            configurable.update(overrides)

        initial_state: AssessmentGraphState = {
            "assessment_id": f"asmt-eval-{scenario.scenario_id}",
            "run_number": 1,
            "vin": scenario.vin,
            "context": scenario.context,
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }

        final_state: dict[str, Any] = await self._app.ainvoke(
            initial_state,
            config={"configurable": configurable},
        )

        return ScenarioExecutionResult(
            scenario=scenario,
            provenance=provenance,
            timestamp=current_time,
            final_phase=final_state["phase"],
            sufficiency_result=final_state.get("sufficiency_result"),
            risk_result=final_state.get("risk_result"),
            policy_citations=tuple(final_state.get("policy_citations", ())),
            report_draft=final_state.get("report_draft"),
            mcp_error=final_state.get("mcp_error"),
            events=tuple(final_state.get("events", ())),
        )

    async def run_all(
        self,
        scenarios: list[EvaluationScenario],
    ) -> list[ScenarioExecutionResult]:
        """Run multiple evaluation scenarios sequentially and deterministically."""
        results: list[ScenarioExecutionResult] = []
        for scenario in scenarios:
            res = await self.run_scenario(scenario)
            results.append(res)
        return results
