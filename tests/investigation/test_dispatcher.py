"""Dispatcher tests prove proposals do not gain execution authority."""

from datetime import UTC, datetime

import pytest

from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.investigation.dispatcher import InvestigationDispatcher
from vehicle_risk_agent.investigation.models import InvestigationProposal
from vehicle_risk_agent.policy.models import PolicyCitation


class FakeVehicleClient:
    def __init__(self, revision: VehicleRevisionResponse) -> None:
        self.revision = revision
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def explain_vehicle_field(self, vin: str, field_name: str):
        self.calls.append(("explain", (vin, field_name)))
        return {"not": "a validated result"}

    async def get_vehicle_history(self, vin: str, limit: int = 20, before_revision=None):
        self.calls.append(("history", (vin, limit, before_revision)))
        return [self.revision]

    async def get_vehicle_revision(self, vin: str, revision_number: int):
        self.calls.append(("revision", (vin, revision_number)))
        return self.revision


class FakePolicyRetriever:
    async def retrieve(self, query: str):
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
    assert vehicle.calls == [("history", ("1HGCM82633A004352", 20, None))]


@pytest.mark.asyncio
async def test_dispatcher_returns_no_action_without_external_calls() -> None:
    vehicle = FakeVehicleClient(revision())
    dispatcher = InvestigationDispatcher(vehicle=vehicle, policy=FakePolicyRetriever())
    proposal = InvestigationProposal.from_tool_input({"kind": "NO_ACTION"})
    result = await dispatcher.dispatch("1HGCM82633A004352", proposal)
    assert result.dispatched is False
    assert vehicle.calls == []