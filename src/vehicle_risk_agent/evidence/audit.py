"""Reviewer audit service for inspecting exact provenance-linked source observations."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from vehicle_risk_agent.evidence.models import SourceObservationResponse
from vehicle_risk_agent.evidence.snapshot import VehicleEvidenceSnapshot

if TYPE_CHECKING:
    from vehicle_risk_agent.adapters.mcp import VehicleMcpClientAdapter

OBSERVATION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\:]{1,128}$")


class ObservationIdValidationError(Exception):
    """Raised when an observation identifier format or length is invalid."""


class UnlinkedObservationError(Exception):
    """Raised when an observation ID is not explicitly linked in the snapshot provenance."""


class SourceObservationAuditService:
    """Service allowing authorized reviewers to inspect exact source evidence."""

    @staticmethod
    def validate_observation_id(observation_id: str) -> str:
        """Enforce strict bounded observation identifier format."""
        if not observation_id or not isinstance(observation_id, str):
            raise ObservationIdValidationError("Observation ID must be a non-empty string")
        clean_id = observation_id.strip()
        if not OBSERVATION_ID_PATTERN.match(clean_id):
            raise ObservationIdValidationError(
                f"Observation ID '{clean_id}' must match bounded pattern [a-zA-Z0-9_-:]{{1,128}}"
            )
        return clean_id

    async def resolve_linked_observation(
        self,
        snapshot: VehicleEvidenceSnapshot,
        observation_id: str,
        adapter: VehicleMcpClientAdapter,
    ) -> SourceObservationResponse:
        """Resolve exact source observation only if linked in snapshot provenance."""
        valid_id = self.validate_observation_id(observation_id)

        # Collect all observation IDs referenced in snapshot provenance
        linked_ids: set[str] = set()
        for prov_list in snapshot.field_provenance.values():
            for p in prov_list:
                linked_ids.add(p.observation_id)

        for conflict in snapshot.conflicts:
            for cand in conflict.conflicting_candidates:
                linked_ids.add(cand.provenance.observation_id)

        if valid_id not in linked_ids:
            raise UnlinkedObservationError(
                f"Observation ID '{valid_id}' is not linked in snapshot "
                f"provenance for VIN {snapshot.vin}"
            )

        return await adapter.get_source_observation(valid_id)
