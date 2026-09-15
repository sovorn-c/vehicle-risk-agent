"""Deterministic labelled scenarios and graders for live bounded investigation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vehicle_risk_agent.evidence.models import FieldExplanationResult, VehicleRevisionResponse
from vehicle_risk_agent.investigation.models import VehicleHistoryResult


class InvestigationScenario(BaseModel):
    """One intent-labelled scenario with action-level grading expectations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    intent: str = Field(min_length=1, max_length=200)
    questions: tuple[str, ...] = Field(default_factory=tuple)
    evidence_targets: tuple[str, ...] = Field(default_factory=tuple)
    expected_action: str
    expected_field: str | None = None
    expected_revision_number: int | None = Field(default=None, ge=1)
    expected_query: str | None = None
    expected_dispatched: bool = True
    expected_outcome: str = "SCORED"
    synthetic_notice: str = (
        "Synthetic demonstration scenario; live provider output is not production evidence."
    )


def e11_scenarios() -> tuple[InvestigationScenario, ...]:
    """Return the fixed three-scenario e11 acceptance set."""
    return (
        InvestigationScenario(
            scenario_id="odometer-explanation",
            intent="Resolve an odometer discrepancy before review.",
            questions=("Explain the unresolved odometer reading discrepancy.",),
            evidence_targets=("odometer_reading",),
            expected_action="explain_vehicle_field",
            expected_field="odometer_reading",
        ),
        InvestigationScenario(
            scenario_id="required-evidence-policy",
            intent="Find the pinned policy passage for an evidence question.",
            questions=(
                "Which policy rule applies when required vehicle evidence remains unresolved?",
            ),
            expected_action="search_policy",
            expected_query="required evidence vehicle sale",
        ),
        InvestigationScenario(
            scenario_id="identity-already-answered",
            intent="No discrepancy remains and no supplementary call is needed.",
            questions=("Do the vehicle make and model match the current record?",),
            evidence_targets=("make", "model"),
            expected_action="NO_ACTION",
            expected_dispatched=False,
        ),
    )


def grade_investigation_scenario(
    scenario: InvestigationScenario,
    *,
    observed_action: str | None,
    dispatched: bool,
    references: tuple[str, ...] = (),
    citation_references: tuple[str, ...] = (),
    completed: bool = True,
    evidence_result: Any | None = None,
    observed_query: str | None = None,
) -> bool:
    """Grade stable actions, typed results, and bounded references."""
    if observed_action != scenario.expected_action:
        return False
    if dispatched != scenario.expected_dispatched:
        return False
    if scenario.expected_action == "NO_ACTION":
        return completed and not references and not citation_references
    if not completed:
        return False
    if scenario.expected_action == "search_policy":
        if not citation_references:
            return False
        return scenario.expected_query is None or (
            observed_query is not None
            and scenario.expected_query.casefold() in observed_query.casefold()
        )
    if scenario.expected_action == "explain_vehicle_field":
        return isinstance(evidence_result, FieldExplanationResult) and (
            scenario.expected_field is None or evidence_result.field_name == scenario.expected_field
        )
    if scenario.expected_action == "get_vehicle_history":
        return isinstance(evidence_result, VehicleHistoryResult) and bool(evidence_result.revisions)
    if scenario.expected_action == "get_vehicle_revision":
        return isinstance(evidence_result, VehicleRevisionResponse) and (
            scenario.expected_revision_number is None
            or evidence_result.revision_number == scenario.expected_revision_number
        )
    return False


# Small aliases keep the evaluator API explicit for callers and tests.
def build_e11_scenarios() -> tuple[InvestigationScenario, ...]:
    return e11_scenarios()


def build_e11_investigation_scenarios() -> tuple[InvestigationScenario, ...]:
    return e11_scenarios()


def grade_e11_scenario(scenario: InvestigationScenario, result: Any) -> bool:
    return grade_scenario(scenario, result)


def grade_scenario(scenario: InvestigationScenario, result: Any) -> bool:
    """Grade an InvestigationResult-like object without trusting arbitrary fields."""
    return grade_investigation_scenario(
        scenario,
        observed_action=(
            result.action.value
            if result.action
            else ("NO_ACTION" if not result.dispatched and result.completed else None)
        ),
        dispatched=bool(result.dispatched),
        references=tuple(result.references),
        citation_references=tuple(citation.passage_id for citation in result.policy_citations),
        completed=bool(result.completed),
        evidence_result=result.evidence_result,
        observed_query=getattr(result, "observed_query", None),
    )
