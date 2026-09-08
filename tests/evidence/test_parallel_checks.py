"""Tests for parallel field explanations, fan-out, and deterministic keyed merge."""

# story: e03s03

import pytest

from vehicle_risk_agent.adapters.mcp import FakeVehicleMcpAdapter
from vehicle_risk_agent.evidence.models import (
    FieldExplanationResult,
    FieldOutcome,
)
from vehicle_risk_agent.evidence.parallel import (
    DuplicateEvidenceKeyError,
    explain_fields_in_parallel,
    merge_field_explanations,
)


@pytest.mark.asyncio
async def test_explain_fields_in_parallel_success() -> None:
    """Parallel fan-out collects explanations for multiple fields concurrently."""
    adapter = FakeVehicleMcpAdapter()
    adapter.seed_field_explanation(
        FieldExplanationResult(
            vin="7AT0BK00X00000001",
            revision_number=1,
            field_name="ppsr_result",
            outcome=FieldOutcome.RESOLVED,
            value="NO_FINANCE_REGISTERED",
            provenance=[],
            conflicts=[],
            available_fields=["ppsr_result"],
        )
    )
    adapter.seed_field_explanation(
        FieldExplanationResult(
            vin="7AT0BK00X00000001",
            revision_number=1,
            field_name="stolen_status",
            outcome=FieldOutcome.RESOLVED,
            value="NOT_STOLEN",
            provenance=[],
            conflicts=[],
            available_fields=["stolen_status"],
        )
    )
    adapter.seed_field_explanation(
        FieldExplanationResult(
            vin="7AT0BK00X00000001",
            revision_number=1,
            field_name="writeoff_status",
            outcome=FieldOutcome.RESOLVED,
            value="NOT_WRITTEN_OFF",
            provenance=[],
            conflicts=[],
            available_fields=["writeoff_status"],
        )
    )

    results = await explain_fields_in_parallel(
        adapter,
        vin="7AT0BK00X00000001",
        field_names=["ppsr_result", "stolen_status", "writeoff_status"],
    )

    assert len(results) == 3
    assert results["ppsr_result"].value == "NO_FINANCE_REGISTERED"
    assert results["stolen_status"].value == "NOT_STOLEN"
    assert results["writeoff_status"].value == "NOT_WRITTEN_OFF"


def test_merge_field_explanations_rejects_duplicate_conflicting_keys() -> None:
    """Merging collections with duplicate keys raises DuplicateEvidenceKeyError."""
    exp1 = FieldExplanationResult(
        vin="7AT0BK00X00000001",
        revision_number=1,
        field_name="ppsr_result",
        outcome=FieldOutcome.RESOLVED,
        value="NO_FINANCE_REGISTERED",
        provenance=[],
        conflicts=[],
        available_fields=["ppsr_result"],
    )
    exp2 = FieldExplanationResult(
        vin="7AT0BK00X00000001",
        revision_number=1,
        field_name="ppsr_result",
        outcome=FieldOutcome.RESOLVED,
        value="FINANCE_ACTIVE",
        provenance=[],
        conflicts=[],
        available_fields=["ppsr_result"],
    )

    current = {"ppsr_result": exp1}
    incoming = {"ppsr_result": exp2}

    with pytest.raises(DuplicateEvidenceKeyError):
        merge_field_explanations(current, incoming)
