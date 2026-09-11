"""Workflow integration contracts for optional investigation phase."""

from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.evidence.sufficiency import EvidenceSufficiencyResult, SufficiencyOutcome
from vehicle_risk_agent.workflow.graph import route_after_sufficiency


def test_complete_mandatory_evidence_enters_investigation_only_when_enabled() -> None:
    sufficient = EvidenceSufficiencyResult(
        outcome=SufficiencyOutcome.COMPLETE,
        is_sufficient=True,
    )
    assert route_after_sufficiency({"sufficiency_result": sufficient}) == "node_retrieving_policy"
    assert (
        route_after_sufficiency(
            {
                "sufficiency_result": sufficient,
                "investigation_enabled": True,
            }
        )
        == "node_investigating"
    )
    assert AssessmentRunPhase.INVESTIGATING.value == "INVESTIGATING"
