"""Tests for attributable vehicle-history projection into report drafting."""

from datetime import UTC, datetime
from typing import Any

import pytest

from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.snapshot import create_evidence_snapshot
from vehicle_risk_agent.investigation.models import (
    InvestigationAction,
    InvestigationResult,
    VehicleHistoryResult,
)
from vehicle_risk_agent.reporting.protocol import ReportDraftingContext
from vehicle_risk_agent.risk.models import AssessmentOutcome, RiskBand, RiskResult
from vehicle_risk_agent.workflow.graph import node_drafting_report


def _revision(
    revision_id: str,
    revision_number: int,
    *,
    ppsr_result: str,
) -> VehicleRevisionResponse:
    timestamp = datetime(2026, 9, revision_number, 12, 0, tzinfo=UTC)
    return VehicleRevisionResponse(
        vin="1HGCM82633A004352",
        revision_id=revision_id,
        revision_number=revision_number,
        material_hash=str(revision_number) * 64,
        canonical_fields={"make": "HONDA", "ppsr_result": ppsr_result},
        confidence=ConfidenceAssessment(
            score=80,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="verified",
        ),
        as_of=timestamp,
        published_at=timestamp,
    )


@pytest.mark.asyncio
async def test_history_revisions_reach_drafting_as_bounded_attributable_items() -> None:
    current = _revision("rev-current", 3, ppsr_result="MATCH")
    snapshot_history = _revision("rev-1", 1, ppsr_result="NO_MATCH")
    supplementary_history = _revision("rev-2", 2, ppsr_result="PENDING")
    snapshot = create_evidence_snapshot(
        "assessment-history",
        1,
        current,
        history=(snapshot_history,),
    )

    class CapturingDrafter:
        context: ReportDraftingContext | None = None

        async def draft_report(self, context: ReportDraftingContext) -> Any:
            self.context = context
            return object()

    drafter = CapturingDrafter()
    result = await node_drafting_report(
        {
            "assessment_id": "assessment-history",
            "run_number": 1,
            "vin": current.vin,
            "context": AssessmentContext(sale_type=SaleType.DEALER),
            "phase": AssessmentRunPhase.EVALUATING_RISK,
            "visited_phases": [],
            "events": [],
            "evidence_snapshot": snapshot,
            "risk_result": RiskResult(
                assessment_id="assessment-history",
                run_number=1,
                policy_id="risk-policy-v1",
                policy_version="v1",
                score=30,
                band=RiskBand.MEDIUM,
                raw_score=30,
                outcome=AssessmentOutcome.SCORED,
            ),
            "investigation_result": InvestigationResult(
                action=InvestigationAction.GET_VEHICLE_HISTORY,
                summary="History returned.",
                references=("rev-2",),
                evidence_result=VehicleHistoryResult(
                    vin=current.vin,
                    revisions=(supplementary_history,),
                ),
                completed=True,
                dispatched=True,
            ),
        },
        {"configurable": {"drafting_adapter": drafter}},
    )

    assert result["report_draft"] is not None
    captured_context = drafter.context
    assert captured_context is not None
    items = {item.observation_id: item for item in captured_context.evidence_items}
    assert set(items) == {"rev-1", "rev-2"}
    assert "revision=2" in items["rev-2"].value
    assert "ppsr_result=PENDING->MATCH" in items["rev-2"].value
    assert "published_at=2026-09-02T12:00:00+00:00" in items["rev-2"].value
    assert all("raw_payload" not in item.model_dump_json() for item in items.values())

    from vehicle_risk_agent.adapters.anthropic_drafting import AnthropicDraftingAdapter

    prompt = AnthropicDraftingAdapter()._build_user_prompt(captured_context)
    assert "rev-2" in prompt
    assert "ppsr_result=PENDING->MATCH" in prompt
