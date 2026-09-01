"""Tests proving parallel completion order cannot alter snapshot, sufficiency, or events."""

import asyncio
from datetime import UTC, datetime

import pytest

from vehicle_risk_agent.adapters.mcp import FakeVehicleMcpAdapter
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    FieldExplanationResult,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.parallel import explain_fields_in_parallel
from vehicle_risk_agent.evidence.snapshot import create_evidence_snapshot
from vehicle_risk_agent.evidence.sufficiency import evaluate_evidence_sufficiency


class DelayedVehicleMcpAdapter(FakeVehicleMcpAdapter):
    """Adapter with controllable per-field simulated latency to test race/completion ordering."""

    def __init__(self, field_delays: dict[str, float]) -> None:
        super().__init__()
        self.field_delays = field_delays

    async def explain_vehicle_field(self, vin: str, field_name: str) -> FieldExplanationResult:
        delay = self.field_delays.get(field_name, 0.0)
        if delay > 0:
            await asyncio.sleep(delay)
        return await super().explain_vehicle_field(vin, field_name)


@pytest.mark.asyncio
async def test_parallel_field_explanations_ordering_invariance() -> None:
    """Proves parallel completion order does not affect the resulting ordered map."""
    fields = ["writeoff_status", "ppsr_result", "stolen_status"]

    # Run 1: Delay ppsr_result longest
    adapter1 = DelayedVehicleMcpAdapter({"ppsr_result": 0.04, "stolen_status": 0.01})
    adapter1.seed_vehicle(
        VehicleRevisionResponse(
            vin="7AT0BK00X00000001",
            revision_id="rev-1",
            revision_number=1,
            material_hash="a" * 64,
            canonical_fields={
                "ppsr_result": "NO_FINANCE_REGISTERED",
                "stolen_status": "NOT_STOLEN",
                "writeoff_status": "NOT_WRITTEN_OFF",
            },
            confidence=ConfidenceAssessment(
                score=90,
                band=ConfidenceBand.HIGH,
                field_scores={},
                field_components={},
                rule_version="v1",
                explanation="ok",
            ),
            as_of=datetime.now(UTC),
            published_at=datetime.now(UTC),
        )
    )

    # Run 2: Delay writeoff_status longest
    adapter2 = DelayedVehicleMcpAdapter({"writeoff_status": 0.04, "ppsr_result": 0.01})
    adapter2.seed_vehicle(
        VehicleRevisionResponse(
            vin="7AT0BK00X00000001",
            revision_id="rev-1",
            revision_number=1,
            material_hash="a" * 64,
            canonical_fields={
                "ppsr_result": "NO_FINANCE_REGISTERED",
                "stolen_status": "NOT_STOLEN",
                "writeoff_status": "NOT_WRITTEN_OFF",
            },
            confidence=ConfidenceAssessment(
                score=90,
                band=ConfidenceBand.HIGH,
                field_scores={},
                field_components={},
                rule_version="v1",
                explanation="ok",
            ),
            as_of=datetime.now(UTC),
            published_at=datetime.now(UTC),
        )
    )

    res1 = await explain_fields_in_parallel(adapter1, "7AT0BK00X00000001", fields)
    res2 = await explain_fields_in_parallel(adapter2, "7AT0BK00X00000001", fields)

    # Dictionary keys and values must match in deterministic sorted order
    assert list(res1.keys()) == list(res2.keys())
    assert list(res1.keys()) == ["ppsr_result", "stolen_status", "writeoff_status"]
    assert res1["ppsr_result"] == res2["ppsr_result"]
    assert res1["stolen_status"] == res2["stolen_status"]
    assert res1["writeoff_status"] == res2["writeoff_status"]


@pytest.mark.asyncio
async def test_sufficiency_ordering_invariance_across_input_order() -> None:
    """Sufficiency evaluation produces identically ordered findings regardless of input map."""
    now = datetime.now(UTC)
    rev1 = VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-1",
        revision_number=1,
        material_hash="b" * 64,
        canonical_fields={"writeoff_status": "UNKNOWN", "ppsr_result": "UNKNOWN"},
        confidence=ConfidenceAssessment(
            score=50,
            band=ConfidenceBand.LOW,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="partial",
        ),
        as_of=now,
        published_at=now,
    )
    rev2 = VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-1",
        revision_number=1,
        material_hash="b" * 64,
        canonical_fields={"ppsr_result": "UNKNOWN", "writeoff_status": "UNKNOWN"},
        confidence=ConfidenceAssessment(
            score=50,
            band=ConfidenceBand.LOW,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="partial",
        ),
        as_of=now,
        published_at=now,
    )

    snap1 = create_evidence_snapshot("asmt-1", 1, rev1)
    snap2 = create_evidence_snapshot("asmt-1", 1, rev2)

    suff1 = evaluate_evidence_sufficiency(snap1)
    suff2 = evaluate_evidence_sufficiency(snap2)

    assert suff1.model_dump() == suff2.model_dump()
    assert [f.field_name for f in suff1.missing_findings] == [
        f.field_name for f in suff2.missing_findings
    ]
