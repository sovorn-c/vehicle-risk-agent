"""Dispatcher tests prove proposals do not gain execution authority."""

from datetime import UTC, datetime
from typing import Any

import pytest

from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    FieldExplanationResult,
    FieldOutcome,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.investigation.dispatcher import InvestigationDispatcher
from vehicle_risk_agent.investigation.models import (
    InvestigationProposal,
    VehicleHistoryResult,
)
from vehicle_risk_agent.policy.models import PolicyCitation


class FakeVehicleClient:
    def __init__(self, revision: VehicleRevisionResponse) -> None:
        self.revision = revision
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def explain_vehicle_field(self, vin: str, field_name: str) -> Any:
        self.calls.append(("explain", (vin, field_name)))
        return {"not": "a validated result"}

    async def get_vehicle_history(
        self, vin: str, limit: int = 20, before_revision: int | None = None
    ) -> Any:
        self.calls.append(("history", (vin, limit, before_revision)))
        return [self.revision]

    async def get_vehicle_revision(self, vin: str, revision_number: int) -> Any:
        self.calls.append(("revision", (vin, revision_number)))
        return self.revision


class FakePolicyRetriever:
    async def retrieve(self, _query: str) -> Any:
        return type(
            "Retrieved",
            (),
            {
                "citations": [
                    PolicyCitation(
                        passage_id="passage-1",
                        snapshot_id="snapshot-1",
                        source_id="source-1",
                        section_identifier="1",
                        heading="Evidence",
                        source_title="Official policy",
                        canonical_origin="https://example.test/policy",
                    )
                ]
            },
        )()


def revision() -> VehicleRevisionResponse:
    return VehicleRevisionResponse(
        vin="1HGCM82633A004352",
        revision_id="rev-1",
        revision_number=3,
        material_hash="a" * 64,
        canonical_fields={"odometer_reading": 120000},
        confidence=ConfidenceAssessment(
            score=80,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="agreement",
        ),
        as_of=datetime.now(UTC),
        published_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_dispatcher_owns_vin_and_tool_arguments() -> None:
    vehicle = FakeVehicleClient(revision())
    dispatcher = InvestigationDispatcher(vehicle=vehicle, policy=FakePolicyRetriever())
    proposal = InvestigationProposal.from_tool_input(
        {
            "kind": "REQUEST",
            "action": "get_vehicle_history",
            "arguments": {},
        }
    )
    result = await dispatcher.dispatch("1HGCM82633A004352", proposal)
    assert result.completed is True
    assert result.references == ("rev-1",)
    assert vehicle.calls == [("history", ("1HGCM82633A004352", 5, None))]


@pytest.mark.asyncio
async def test_history_dispatch_returns_bounded_typed_revisions() -> None:
    class HistoryVehicle(FakeVehicleClient):
        async def get_vehicle_history(
            self, vin: str, limit: int = 20, before_revision: int | None = None
        ) -> Any:
            self.calls.append(("history", (vin, limit, before_revision)))
            return [
                self.revision.model_copy(
                    update={
                        "revision_id": "rev-1",
                        "revision_number": 1,
                        "material_hash": "b" * 64,
                    }
                ),
                self.revision.model_copy(
                    update={
                        "revision_id": "rev-2",
                        "revision_number": 2,
                        "material_hash": "c" * 64,
                    }
                ),
            ]

    vehicle = HistoryVehicle(revision())
    dispatcher = InvestigationDispatcher(vehicle=vehicle, policy=FakePolicyRetriever())
    proposal = InvestigationProposal.from_tool_input(
        {
            "kind": "REQUEST",
            "action": "get_vehicle_history",
            "arguments": {},
        }
    )

    result = await dispatcher.dispatch("1HGCM82633A004352", proposal)

    assert isinstance(result.evidence_result, VehicleHistoryResult)
    assert tuple(item.revision_number for item in result.evidence_result.revisions) == (2, 1)
    assert result.references == ("rev-2", "rev-1")


@pytest.mark.asyncio
async def test_history_dispatch_rejects_duplicate_revisions() -> None:
    class InvalidHistoryVehicle(FakeVehicleClient):
        async def get_vehicle_history(
            self, vin: str, limit: int = 20, before_revision: int | None = None
        ) -> Any:
            self.calls.append(("history", (vin, limit, before_revision)))
            return [
                self.revision.model_copy(update={"revision_id": "rev-old-1", "revision_number": 1}),
                self.revision.model_copy(update={"revision_id": "rev-old-2", "revision_number": 1}),
            ]

    vehicle = InvalidHistoryVehicle(revision())
    dispatcher = InvestigationDispatcher(vehicle=vehicle, policy=FakePolicyRetriever())
    proposal = InvestigationProposal.from_tool_input(
        {
            "kind": "REQUEST",
            "action": "get_vehicle_history",
            "arguments": {},
        }
    )

    result = await dispatcher.dispatch(
        "1HGCM82633A004352",
        proposal,
    )

    assert result.completed is False
    assert result.limitation is not None
    assert result.limitation.code == "INVALID_RESULT"


@pytest.mark.asyncio
async def test_history_dispatch_rejects_current_or_future_revisions() -> None:
    vehicle = FakeVehicleClient(revision())
    dispatcher = InvestigationDispatcher(vehicle=vehicle, policy=FakePolicyRetriever())
    proposal = InvestigationProposal.from_tool_input(
        {
            "kind": "REQUEST",
            "action": "get_vehicle_history",
            "arguments": {},
        }
    )

    result = await dispatcher.dispatch(
        "1HGCM82633A004352",
        proposal,
        current_revision=3,
    )

    assert result.completed is False
    assert result.limitation is not None
    assert result.limitation.code == "INVALID_RESULT"


@pytest.mark.asyncio
async def test_dispatcher_maps_odometer_target_to_mcp_canonical_field() -> None:
    class AliasVehicle(FakeVehicleClient):
        async def explain_vehicle_field(self, vin: str, field_name: str) -> Any:
            self.calls.append(("explain", (vin, field_name)))
            return FieldExplanationResult(
                vin=vin,
                revision_number=3,
                field_name=field_name,
                outcome=FieldOutcome.RESOLVED,
                value=52300,
                confidence_score=80,
                confidence_band=ConfidenceBand.HIGH,
                rationale="resolved",
            )

    vehicle = AliasVehicle(revision())
    dispatcher = InvestigationDispatcher(vehicle=vehicle, policy=FakePolicyRetriever())
    proposal = InvestigationProposal.from_tool_input(
        {
            "kind": "REQUEST",
            "action": "explain_vehicle_field",
            "arguments": {"field_name": "odometer_reading"},
        }
    )

    result = await dispatcher.dispatch("1HGCM82633A004352", proposal)

    assert result.completed is True
    evidence_result = result.evidence_result
    assert isinstance(evidence_result, FieldExplanationResult)
    assert evidence_result.field_name == "odometer_reading"
    assert vehicle.calls == [("explain", ("1HGCM82633A004352", "odometer_km"))]


@pytest.mark.asyncio
async def test_dispatcher_returns_no_action_without_external_calls() -> None:
    vehicle = FakeVehicleClient(revision())
    dispatcher = InvestigationDispatcher(vehicle=vehicle, policy=FakePolicyRetriever())
    proposal = InvestigationProposal.from_tool_input({"kind": "NO_ACTION"})
    result = await dispatcher.dispatch("1HGCM82633A004352", proposal)
    assert result.dispatched is False
    assert vehicle.calls == []
    assert "evidence_result" not in result.safe_metadata()
