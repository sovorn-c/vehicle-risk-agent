"""Tests for retrieving newest-first vehicle revision history only when temporal depth exists."""

from datetime import UTC, datetime

import pytest

from vehicle_risk_agent.adapters.mcp import FakeVehicleMcpAdapter
from vehicle_risk_agent.evidence.history import collect_vehicle_history
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    ProvenanceLink,
    VehicleRevisionResponse,
)


@pytest.fixture
def base_revisions() -> tuple[VehicleRevisionResponse, VehicleRevisionResponse]:
    now = datetime.now(UTC)
    p1 = ProvenanceLink(
        observation_id="obs-1",
        source_system="SYS_1",
        source_record_id="r1",
        retrieved_at=now,
    )
    p2 = ProvenanceLink(
        observation_id="obs-2",
        source_system="SYS_2",
        source_record_id="r2",
        retrieved_at=now,
    )

    rev1 = VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-001",
        revision_number=1,
        material_hash="1" * 64,
        canonical_fields={"make": "TOYOTA", "year": 2018},
        field_provenance={"make": [p1]},
        conflicts=[],
        confidence=ConfidenceAssessment(
            score=80,
            band=ConfidenceBand.HIGH,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="Initial revision",
        ),
        as_of=now,
        published_at=now,
    )

    rev2 = VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-002",
        revision_number=2,
        material_hash="2" * 64,
        canonical_fields={"make": "TOYOTA", "year": 2018, "stolen_status": "NOT_STOLEN"},
        field_provenance={"make": [p1], "stolen_status": [p2]},
        conflicts=[],
        confidence=ConfidenceAssessment(
            score=90,
            band=ConfidenceBand.HIGH,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="Updated revision with stolen status",
        ),
        as_of=now,
        published_at=now,
    )

    return rev1, rev2


@pytest.mark.asyncio
async def test_collect_vehicle_history_skips_when_revision_is_first(
    base_revisions: tuple[VehicleRevisionResponse, VehicleRevisionResponse],
) -> None:
    """When revision_number is 1, no history call is made and empty list is returned."""
    rev1, _ = base_revisions
    adapter = FakeVehicleMcpAdapter()
    adapter.seed_vehicle(rev1)

    history = await collect_vehicle_history(adapter, rev1)
    assert history == []


@pytest.mark.asyncio
async def test_collect_vehicle_history_returns_newest_first_prior_revisions(
    base_revisions: tuple[VehicleRevisionResponse, VehicleRevisionResponse],
) -> None:
    """When revision_number > 1, prior revisions are returned sorted newest-first."""
    rev1, rev2 = base_revisions
    adapter = FakeVehicleMcpAdapter()
    adapter.seed_vehicle(rev1)
    adapter.seed_vehicle(rev2)

    history = await collect_vehicle_history(adapter, rev2)
    assert len(history) == 1
    assert history[0].revision_number == 1
    assert history[0].revision_id == "rev-001"
    assert history[0].field_provenance["make"][0].observation_id == "obs-1"
