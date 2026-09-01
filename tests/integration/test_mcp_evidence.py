"""Integration tests for typed asynchronous MCP Vehicle Intelligence adapter."""

from datetime import UTC, datetime

import pytest

from vehicle_risk_agent.adapters.mcp import (
    FakeVehicleMcpAdapter,
    McpAdapterError,
)
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    FieldExplanationResult,
    FieldOutcome,
    ProvenanceLink,
    SafeErrorCategory,
    VehicleRevisionResponse,
)


@pytest.fixture
def sample_vehicle_revision() -> VehicleRevisionResponse:
    """Fixture providing a valid canonical VehicleRevisionResponse."""
    now = datetime.now(UTC)
    provenance = ProvenanceLink(
        observation_id="obs-nzta-001",
        source_system="NZTA_MVR",
        source_record_id="rec-001",
        retrieved_at=now,
        synthetic=True,
    )
    confidence = ConfidenceAssessment(
        score=90,
        band=ConfidenceBand.HIGH,
        field_scores={"make": 95, "model": 95, "year": 90},
        field_components={
            "make": {"authority": 40, "freshness": 25, "agreement": 20, "validation": 10}
        },
        rule_version="conf-v1",
        explanation="Official NZTA register record.",
    )
    return VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-001",
        revision_number=1,
        material_hash="a" * 64,
        canonical_fields={
            "make": "NISSAN",
            "model": "LEAF",
            "year": 2020,
            "ppsr_result": "NO_FINANCE_REGISTERED",
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        },
        field_provenance={"make": [provenance]},
        conflicts=[],
        confidence=confidence,
        as_of=now,
        published_at=now,
        synthetic_notice="SYNTHETIC DEMO DATA",
    )


@pytest.mark.asyncio
async def test_mcp_adapter_lookup_vehicle_success(
    sample_vehicle_revision: VehicleRevisionResponse,
) -> None:
    """lookup_vehicle returns typed VehicleRevisionResponse on successful lookup."""
    adapter = FakeVehicleMcpAdapter()
    adapter.seed_vehicle(sample_vehicle_revision)

    revision = await adapter.lookup_vehicle("7AT0BK00X00000001")
    assert isinstance(revision, VehicleRevisionResponse)
    assert revision.vin == "7AT0BK00X00000001"
    assert revision.canonical_fields["make"] == "NISSAN"
    assert revision.canonical_fields["ppsr_result"] == "NO_FINANCE_REGISTERED"


@pytest.mark.asyncio
async def test_mcp_adapter_lookup_vehicle_not_found() -> None:
    """lookup_vehicle raises McpAdapterError with VEHICLE_NOT_FOUND when vehicle does not exist."""
    adapter = FakeVehicleMcpAdapter()

    with pytest.raises(McpAdapterError) as exc_info:
        await adapter.lookup_vehicle("7AT0BK00X00000099")

    assert exc_info.value.category == SafeErrorCategory.VEHICLE_NOT_FOUND
    assert not exc_info.value.retryable


@pytest.mark.asyncio
async def test_mcp_adapter_explain_vehicle_field() -> None:
    """explain_vehicle_field returns deterministic FieldExplanationResult."""
    adapter = FakeVehicleMcpAdapter()
    explanation = FieldExplanationResult(
        vin="7AT0BK00X00000001",
        revision_number=1,
        field_name="ppsr_result",
        outcome=FieldOutcome.RESOLVED,
        value="NO_FINANCE_REGISTERED",
        provenance=[],
        conflicts=[],
        available_fields=["ppsr_result"],
        rationale="PPSR certificate indicates clean title.",
    )
    adapter.seed_field_explanation(explanation)

    result = await adapter.explain_vehicle_field("7AT0BK00X00000001", "ppsr_result")
    assert isinstance(result, FieldExplanationResult)
    assert result.outcome == FieldOutcome.RESOLVED
    assert result.value == "NO_FINANCE_REGISTERED"


@pytest.mark.asyncio
async def test_mcp_adapter_timeout_and_retries() -> None:
    """Transient network/adapter errors are retried with exponential backoff up to max attempts."""
    adapter = FakeVehicleMcpAdapter(max_retries=2, timeout_seconds=1.0)
    adapter.simulate_transient_failures("7AT0BK00X00000001", failure_count=1)
    now = datetime.now(UTC)
    adapter.seed_vehicle(
        VehicleRevisionResponse(
            vin="7AT0BK00X00000001",
            revision_id="rev-1",
            revision_number=1,
            material_hash="c" * 64,
            canonical_fields={"make": "MAZDA"},
            field_provenance={},
            conflicts=[],
            confidence=ConfidenceAssessment(
                score=80,
                band=ConfidenceBand.HIGH,
                field_scores={},
                field_components={},
                rule_version="v1",
                explanation="ok",
            ),
            as_of=now,
            published_at=now,
        )
    )

    # Should succeed on second attempt after 1 failure
    rev = await adapter.lookup_vehicle("7AT0BK00X00000001")
    assert rev.canonical_fields["make"] == "MAZDA"
