"""Workflow integration contracts for optional investigation phase."""

from datetime import UTC, datetime
from types import SimpleNamespace
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
from vehicle_risk_agent.evidence.sufficiency import EvidenceSufficiencyResult, SufficiencyOutcome
from vehicle_risk_agent.investigation.budget import InvestigationLimits
from vehicle_risk_agent.investigation.models import (
    InvestigationAction,
    InvestigationContext,
    InvestigationResult,
    NoActionProposal,
    ProviderProposalResult,
    ProviderUsage,
    VehicleHistoryResult,
)
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.workflow.graph import (
    node_investigating,
    node_retrieving_policy,
    route_after_sufficiency,
)


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


@pytest.mark.asyncio
async def test_investigation_uses_configured_targets_and_ledger_limits() -> None:
    revision = VehicleRevisionResponse(
        vin="1HGCM82633A004352",
        revision_id="rev-1",
        revision_number=3,
        material_hash="a" * 64,
        canonical_fields={"body_type": "SEDAN", "make": "Honda"},
        field_provenance={},
        confidence=ConfidenceAssessment(
            score=82,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="agreement",
        ),
        as_of=datetime.now(UTC),
        published_at=datetime.now(UTC),
    )
    snapshot = create_evidence_snapshot("assessment-1", 1, revision)

    class Provider:
        context: InvestigationContext | None = None

        async def propose(self, context: InvestigationContext) -> ProviderProposalResult:
            self.context = context
            return ProviderProposalResult(
                proposal=NoActionProposal(),
                usage=ProviderUsage(input_tokens=10, output_tokens=5),
            )

    provider = Provider()
    ledger_state = SimpleNamespace(status="READY", limits=InvestigationLimits.final())

    class Ledger:
        async def get_ledger(self, *_args: Any) -> Any:
            return ledger_state

        async def reserve_proposal(self, *_args: Any) -> Any:
            return SimpleNamespace(status="COUNTING")

        async def complete_proposal(self, *_args: Any) -> Any:
            return SimpleNamespace(status="READY")

        async def complete_no_action(self, *_args: Any) -> Any:
            return ledger_state

    class Dispatcher:
        async def dispatch(self, *_args: Any, **_kwargs: Any) -> InvestigationResult:
            return InvestigationResult(
                summary="No supplementary investigation was requested.",
                completed=True,
                dispatched=False,
            )

    result = await node_investigating(
        {
            "assessment_id": "assessment-1",
            "run_number": 1,
            "vin": revision.vin,
            "context": AssessmentContext(sale_type=SaleType.DEALER, questions=["Explain this."]),
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
            "evidence_snapshot": snapshot,
        },
        {
            "configurable": {
                "investigation_provider": provider,
                "investigation_ledger": Ledger(),
                "investigation_dispatcher": Dispatcher(),
                "investigation_targets": ("odometer_reading",),
            }
        },
    )

    assert provider.context is not None
    assert provider.context.evidence_targets == ("odometer_reading",)
    assert result["investigation_result"] is not None
    assert result["investigation_usage"] == ProviderUsage(input_tokens=10, output_tokens=5)


@pytest.mark.asyncio
async def test_node_retrieving_policy_uses_investigation_citations_directly() -> None:
    investigation_citation = PolicyCitation(
        passage_id="snap-virm:p002",
        snapshot_id="snap-virm",
        source_id="virm",
        section_identifier="2-4",
        heading="Statutory write-offs",
        source_title="NZTA VIRM",
        canonical_origin="https://virm.nzta.govt.nz",
    )

    class TrackingRetrievalService:
        called: bool = False

        async def retrieve(self, query: str) -> Any:
            del query
            self.called = True
            return SimpleNamespace(citations=())

    service = TrackingRetrievalService()
    result = await node_retrieving_policy(
        {
            "assessment_id": "asmt-1",
            "run_number": 1,
            "vin": "1HGCM82633A004352",
            "context": AssessmentContext(
                sale_type=SaleType.DEALER,
                questions=["Which Consumer Information Notice or write-off guidance applies?"],
            ),
            "phase": AssessmentRunPhase.INVESTIGATING,
            "visited_phases": [],
            "events": [],
            "investigation_result": InvestigationResult(
                summary="Investigation policy found.",
                policy_citations=(investigation_citation,),
                completed=True,
                dispatched=True,
            ),
        },
        {"configurable": {"retrieval_service": service}},
    )

    assert result["policy_citations"] == (investigation_citation,)
    assert service.called is False


@pytest.mark.asyncio
async def test_completed_ledger_replays_typed_result_without_provider_call() -> None:
    revision = VehicleRevisionResponse(
        vin="1HGCM82633A004352",
        revision_id="rev-history-1",
        revision_number=1,
        material_hash="b" * 64,
        canonical_fields={"make": "Honda"},
        confidence=ConfidenceAssessment(
            score=90,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="verified",
        ),
        as_of=datetime.now(UTC),
        published_at=datetime.now(UTC),
    )
    expected = InvestigationResult(
        action=InvestigationAction.GET_VEHICLE_HISTORY,
        summary="history returned",
        references=(revision.revision_id,),
        evidence_result=VehicleHistoryResult(vin=revision.vin, revisions=(revision,)),
        completed=True,
        dispatched=True,
    )
    snapshot = create_evidence_snapshot("assessment-replay", 1, revision)

    class Provider:
        async def propose(self, _context: InvestigationContext) -> ProviderProposalResult:
            raise AssertionError("replayed result must not call provider")

    class Ledger:
        async def get_ledger(self, *_args: Any) -> Any:
            return SimpleNamespace(
                status="COMPLETED",
                result=expected,
                limits=InvestigationLimits.final(),
            )

    class Dispatcher:
        async def dispatch(self, *_args: Any, **_kwargs: Any) -> InvestigationResult:
            raise AssertionError("replayed result must not call dispatcher")

    result = await node_investigating(
        {
            "assessment_id": "assessment-replay",
            "run_number": 1,
            "vin": revision.vin,
            "context": AssessmentContext(sale_type=SaleType.DEALER, questions=["Find history."]),
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
            "evidence_snapshot": snapshot,
        },
        {
            "configurable": {
                "investigation_provider": Provider(),
                "investigation_ledger": Ledger(),
                "investigation_dispatcher": Dispatcher(),
            }
        },
    )

    assert result["investigation_result"] == expected
