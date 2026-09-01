"""Temporal vehicle revision history acquisition preserving revision identity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vehicle_risk_agent.evidence.models import VehicleRevisionResponse

if TYPE_CHECKING:
    from vehicle_risk_agent.adapters.mcp import VehicleMcpClientAdapter


async def collect_vehicle_history(
    adapter: VehicleMcpClientAdapter,
    current_revision: VehicleRevisionResponse,
    limit: int = 20,
) -> list[VehicleRevisionResponse]:
    """Retrieve newest-first prior revisions only when temporal depth exists."""
    if current_revision.revision_number <= 1:
        return []

    revisions = await adapter.get_vehicle_history(
        vin=current_revision.vin,
        limit=limit,
        before_revision=current_revision.revision_number,
    )
    return sorted(revisions, key=lambda r: r.revision_number, reverse=True)
