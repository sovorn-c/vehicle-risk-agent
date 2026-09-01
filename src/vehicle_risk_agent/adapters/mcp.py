"""Typed asynchronous MCP adapter for Vehicle Intelligence server tools."""

import asyncio
import random
from typing import Any, Protocol

from vehicle_risk_agent.evidence.models import (
    FieldExplanationResult,
    SafeErrorCategory,
    SourceObservationResponse,
    VehicleRevisionResponse,
)


class McpAdapterError(Exception):
    """Raised when an MCP tool invocation fails or returns a safe error."""

    def __init__(
        self,
        category: SafeErrorCategory,
        message: str,
        retryable: bool = False,
        remediation: str = "",
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.retryable = retryable
        self.remediation = remediation


class VehicleMcpClientAdapter(Protocol):
    """Protocol for asynchronous Vehicle Intelligence MCP client adapters."""

    async def lookup_vehicle(self, vin: str) -> VehicleRevisionResponse:
        """Lookup canonical vehicle revision by VIN."""
        ...

    async def explain_vehicle_field(
        self, vin: str, field_name: str
    ) -> FieldExplanationResult:
        """Explain one vehicle field outcome, value, and conflict status."""
        ...

    async def get_vehicle_history(
        self, vin: str, limit: int = 20, before_revision: int | None = None
    ) -> list[VehicleRevisionResponse]:
        """Retrieve revision history for a vehicle."""
        ...

    async def get_source_observation(
        self, observation_id: str
    ) -> SourceObservationResponse:
        """Retrieve exact source observation by identifier."""
        ...


class FakeVehicleMcpAdapter:
    """Deterministic offline MCP adapter for unit tests, workflow simulation, and grading."""

    def __init__(
        self,
        max_retries: int = 3,
        timeout_seconds: float = 5.0,
        initial_backoff: float = 0.05,
    ) -> None:
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.initial_backoff = initial_backoff
        self._vehicles: dict[str, VehicleRevisionResponse] = {}
        self._explanations: dict[tuple[str, str], FieldExplanationResult] = {}
        self._observations: dict[str, SourceObservationResponse] = {}
        self._history: dict[str, list[VehicleRevisionResponse]] = {}
        self._transient_failures: dict[str, int] = {}

    def seed_vehicle(self, revision: VehicleRevisionResponse) -> None:
        """Seed a vehicle revision in fake store."""
        self._vehicles[revision.vin] = revision
        if revision.vin not in self._history:
            self._history[revision.vin] = [revision]

    def seed_field_explanation(self, explanation: FieldExplanationResult) -> None:
        """Seed a field explanation in fake store."""
        self._explanations[(explanation.vin, explanation.field_name)] = explanation

    def seed_source_observation(self, observation: SourceObservationResponse) -> None:
        """Seed a source observation in fake store."""
        self._observations[observation.observation_id] = observation

    def simulate_transient_failures(self, vin: str, failure_count: int = 1) -> None:
        """Simulate transient failures before succeeding on a VIN."""
        self._transient_failures[vin] = failure_count

    async def _execute_with_retry(self, operation: Any, vin: str) -> Any:
        """Execute operation with bounded retry and exponential backoff."""
        attempts = 0
        backoff = self.initial_backoff

        while True:
            attempts += 1
            if vin in self._transient_failures and self._transient_failures[vin] > 0:
                self._transient_failures[vin] -= 1
                if attempts <= self.max_retries:
                    jitter = random.uniform(0.8, 1.2)
                    await asyncio.sleep(backoff * jitter)
                    backoff *= 2.0
                    continue
                raise McpAdapterError(
                    category=SafeErrorCategory.PIPELINE_TIMEOUT,
                    message="Exceeded retry attempts for vehicle lookup",
                    retryable=True,
                    remediation="Retry when upstream pipeline recovers.",
                )

            return operation()

    async def lookup_vehicle(self, vin: str) -> VehicleRevisionResponse:
        """Lookup vehicle revision with simulated retry behavior."""
        def op() -> VehicleRevisionResponse:
            clean_vin = vin.strip().upper()
            if clean_vin not in self._vehicles:
                raise McpAdapterError(
                    category=SafeErrorCategory.VEHICLE_NOT_FOUND,
                    message=f"Vehicle with VIN {clean_vin} not found in catalog",
                    retryable=False,
                    remediation="Check VIN and verify vehicle exists.",
                )
            return self._vehicles[clean_vin]

        return await self._execute_with_retry(op, vin.strip().upper())  # type: ignore[no-any-return]

    async def explain_vehicle_field(
        self, vin: str, field_name: str
    ) -> FieldExplanationResult:
        """Explain field from fake store or synthesize default resolved/absent explanation."""
        clean_vin = vin.strip().upper()
        clean_field = field_name.strip().lower()

        key = (clean_vin, clean_field)
        if key in self._explanations:
            return self._explanations[key]

        if clean_vin in self._vehicles:
            veh = self._vehicles[clean_vin]
            val = veh.canonical_fields.get(clean_field)
            outcome = FieldOutcome.RESOLVED if val is not None else FieldOutcome.ABSENT
            return FieldExplanationResult(
                vin=clean_vin,
                revision_number=veh.revision_number,
                field_name=clean_field,
                outcome=outcome,
                value=val,
                provenance=veh.field_provenance.get(clean_field, []),
                conflicts=[c for c in veh.conflicts if c.field_name == clean_field],
                confidence_score=veh.confidence.score,
                confidence_band=veh.confidence.band,
                available_fields=sorted(list(veh.canonical_fields.keys())),
                rationale="Resolved from canonical fields.",
                synthetic_notice=veh.synthetic_notice,
            )

        raise McpAdapterError(
            category=SafeErrorCategory.VEHICLE_NOT_FOUND,
            message=f"Vehicle with VIN {clean_vin} not found in catalog",
            retryable=False,
        )

    async def get_vehicle_history(
        self, vin: str, limit: int = 20, before_revision: int | None = None
    ) -> list[VehicleRevisionResponse]:
        """Retrieve revisions history."""
        clean_vin = vin.strip().upper()
        if clean_vin not in self._history:
            return []
        revisions = self._history[clean_vin]
        if before_revision is not None:
            revisions = [r for r in revisions if r.revision_number < before_revision]
        # Sort newest-first
        sorted_revs = sorted(revisions, key=lambda r: r.revision_number, reverse=True)
        return sorted_revs[:limit]

    async def get_source_observation(
        self, observation_id: str
    ) -> SourceObservationResponse:
        """Retrieve source observation by identifier."""
        clean_id = observation_id.strip()
        if clean_id not in self._observations:
            raise McpAdapterError(
                category=SafeErrorCategory.OBSERVATION_NOT_FOUND,
                message=f"Source observation {clean_id} not found",
                retryable=False,
            )
        return self._observations[clean_id]
