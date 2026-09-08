"""Reviewer audit service for inspecting exact provenance-linked source observations."""

# story: e03s04

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from vehicle_risk_agent.evidence.models import ProvenanceLink, SourceObservationResponse
from vehicle_risk_agent.evidence.snapshot import VehicleEvidenceSnapshot

if TYPE_CHECKING:
    from vehicle_risk_agent.adapters.mcp import VehicleMcpClientAdapter

OBSERVATION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\:]{1,128}$")


class ObservationIdValidationError(Exception):
    """Raised when an observation identifier format or length is invalid."""


class UnlinkedObservationError(Exception):
    """Raised when an observation ID is not explicitly linked in the snapshot provenance."""


class CorruptedObservationError(Exception):
    """Raised when a retrieved source observation payload hash does not match content."""


class ProvenanceMismatchError(Exception):
    """Raised when an observation's metadata differs from its linked provenance."""


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

        # Collect complete provenance links, not only IDs, for metadata verification.
        linked_provenance: list[ProvenanceLink] = []

        for prov_list in snapshot.field_provenance.values():
            linked_provenance.extend(prov_list)
        for conflict in snapshot.conflicts:
            linked_provenance.extend(cand.provenance for cand in conflict.conflicting_candidates)
        for hist_rev in snapshot.history:
            for hist_prov_list in hist_rev.field_provenance.values():
                linked_provenance.extend(hist_prov_list)
            for conflict in hist_rev.conflicts:
                linked_provenance.extend(
                    cand.provenance for cand in conflict.conflicting_candidates
                )
        for explanation in snapshot.field_explanations.values():
            linked_provenance.extend(explanation.provenance)
            for conflict in explanation.conflicts:
                linked_provenance.extend(
                    cand.provenance for cand in conflict.conflicting_candidates
                )

        matching_links = [p for p in linked_provenance if p.observation_id == valid_id]
        if not matching_links:
            raise UnlinkedObservationError(
                f"Observation ID '{valid_id}' is not linked in snapshot "
                f"provenance for VIN {snapshot.vin}"
            )

        obs = await adapter.get_source_observation(valid_id)

        # Verify the adapter returned the exact requested observation.
        if obs.observation_id != valid_id:
            raise ProvenanceMismatchError(
                "Observation identifier does not match the requested link"
            )

        # Verify identity metadata before exposing the exact payload to a reviewer.
        if any(
            (
                obs.source_system != link.source_system
                or obs.source_record_id != link.source_record_id
                or obs.retrieved_at != link.retrieved_at
                or obs.synthetic != link.synthetic
            )
            for link in matching_links
        ):
            raise ProvenanceMismatchError("Observation metadata does not match snapshot provenance")

        # Verify cryptographic integrity.
        computed_hash = hashlib.sha256(obs.raw_payload.encode("utf-8")).hexdigest()
        if obs.payload_hash_sha256.lower() != computed_hash.lower():
            raise CorruptedObservationError("Observation payload integrity verification failed")

        return obs
