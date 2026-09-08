"""Parallel asynchronous fan-out and deterministic keyed merge for evidence checks."""

# story: e03s03

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vehicle_risk_agent.evidence.models import FieldExplanationResult

if TYPE_CHECKING:
    from vehicle_risk_agent.adapters.mcp import VehicleMcpClientAdapter


class DuplicateEvidenceKeyError(Exception):
    """Raised when duplicate or conflicting keys are merged into parallel evidence state."""


def merge_field_explanations(
    current: dict[str, FieldExplanationResult],
    incoming: dict[str, FieldExplanationResult],
) -> dict[str, FieldExplanationResult]:
    """Merge incoming field explanations into current state with duplicate-key rejection."""
    for key in incoming:
        if key in current:
            raise DuplicateEvidenceKeyError(
                f"Duplicate evidence key '{key}' rejected during parallel merge"
            )

    merged = {**current, **incoming}
    # Sort deterministically by key
    return {k: merged[k] for k in sorted(merged.keys())}


async def explain_fields_in_parallel(
    adapter: VehicleMcpClientAdapter,
    vin: str,
    field_names: Sequence[str],
) -> dict[str, FieldExplanationResult]:
    """Execute independent field explanations concurrently across parallel tasks."""
    clean_vin = vin.strip().upper()
    clean_fields = [f.strip().lower() for f in field_names]

    tasks = [adapter.explain_vehicle_field(clean_vin, f) for f in clean_fields]
    results = await asyncio.gather(*tasks)

    keyed_results: dict[str, FieldExplanationResult] = {}
    for explanation in results:
        field = explanation.field_name
        if field in keyed_results:
            raise DuplicateEvidenceKeyError(
                f"Duplicate field explanation returned for key '{field}'"
            )
        keyed_results[field] = explanation

    return {k: keyed_results[k] for k in sorted(keyed_results.keys())}
