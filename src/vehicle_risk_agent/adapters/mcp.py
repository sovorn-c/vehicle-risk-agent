"""Typed asynchronous MCP adapters for Vehicle Intelligence server tools."""

# story: e07s01

import asyncio
import json
import random
from typing import Any, NoReturn, Protocol, TypeVar

from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ValidationError

from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.evidence.models import (
    FieldExplanationResult,
    FieldOutcome,
    SafeError,
    SafeErrorCategory,
    SourceObservationResponse,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.observability.telemetry import trace_boundary

_SAFE_MESSAGES: dict[SafeErrorCategory, tuple[str, str]] = {
    SafeErrorCategory.INVALID_INPUT: (
        "The MCP request was invalid.",
        "Verify the request values and try again.",
    ),
    SafeErrorCategory.VEHICLE_NOT_FOUND: (
        "No vehicle evidence was found.",
        "Verify the VIN and upstream availability.",
    ),
    SafeErrorCategory.REVISION_NOT_FOUND: (
        "The requested vehicle revision was not found.",
        "Request a revision present in the vehicle history.",
    ),
    SafeErrorCategory.OBSERVATION_NOT_FOUND: (
        "The requested source observation was not found.",
        "Use an observation identifier from snapshot provenance.",
    ),
    SafeErrorCategory.PIPELINE_TIMEOUT: (
        "The vehicle intelligence service timed out.",
        "Retry the assessment after a short delay.",
    ),
    SafeErrorCategory.PIPELINE_UNAVAILABLE: (
        "The vehicle intelligence service is unavailable.",
        "Check MCP service availability and retry the assessment.",
    ),
    SafeErrorCategory.PIPELINE_CONTRACT_ERROR: (
        "The vehicle intelligence response was invalid.",
        "Verify MCP server and client contract compatibility.",
    ),
    SafeErrorCategory.INTERNAL_ERROR: (
        "An internal MCP processing error occurred.",
        "Inspect service diagnostics and retry the assessment.",
    ),
}


class McpAdapterError(Exception):
    """Raised when an MCP tool invocation fails or returns a safe error."""

    def __init__(
        self,
        category: SafeErrorCategory,
        message: str,
        retryable: bool = False,
        remediation: str = "",
    ) -> None:
        del message, remediation
        try:
            safe_category = SafeErrorCategory(category)
        except (TypeError, ValueError):
            safe_category = SafeErrorCategory.INTERNAL_ERROR
        safe_message, safe_remediation = _SAFE_MESSAGES[safe_category]
        super().__init__(safe_message)
        self.category = safe_category
        self.message = safe_message
        self.retryable = retryable
        self.remediation = safe_remediation


class VehicleMcpClientAdapter(Protocol):
    """Protocol for asynchronous Vehicle Intelligence MCP client adapters."""

    async def lookup_vehicle(self, vin: str) -> VehicleRevisionResponse:
        """Lookup canonical vehicle revision by VIN."""
        ...

    async def get_vehicle_revision(self, vin: str, revision_number: int) -> VehicleRevisionResponse:
        """Retrieve exact canonical revision of a vehicle."""
        ...

    async def explain_vehicle_field(self, vin: str, field_name: str) -> FieldExplanationResult:
        """Explain one vehicle field outcome, value, and conflict status."""
        ...

    async def get_vehicle_history(
        self, vin: str, limit: int = 20, before_revision: int | None = None
    ) -> list[VehicleRevisionResponse]:
        """Retrieve revision history for a vehicle."""
        ...

    async def get_source_observation(self, observation_id: str) -> SourceObservationResponse:
        """Retrieve exact source observation by identifier."""
        ...


class UnavailableVehicleMcpAdapter:
    """Fail-closed adapter used when no MCP endpoint is configured."""

    @staticmethod
    def _raise_unavailable() -> NoReturn:
        raise McpAdapterError(
            category=SafeErrorCategory.PIPELINE_UNAVAILABLE,
            message="MCP endpoint is not configured",
            retryable=True,
            remediation="Configure the vehicle intelligence MCP server endpoint.",
        )

    async def lookup_vehicle(self, _vin: str) -> VehicleRevisionResponse:
        self._raise_unavailable()

    async def get_vehicle_revision(
        self, _vin: str, _revision_number: int
    ) -> VehicleRevisionResponse:
        self._raise_unavailable()

    async def explain_vehicle_field(self, _vin: str, _field_name: str) -> FieldExplanationResult:
        self._raise_unavailable()

    async def get_vehicle_history(
        self, _vin: str, _limit: int = 20, _before_revision: int | None = None
    ) -> list[VehicleRevisionResponse]:
        self._raise_unavailable()

    async def get_source_observation(self, _observation_id: str) -> SourceObservationResponse:
        self._raise_unavailable()


ModelT = TypeVar("ModelT", bound=BaseModel)


class StreamableHttpVehicleMcpAdapter:
    """Official MCP client adapter for the Vehicle Intelligence Streamable HTTP server."""

    def __init__(
        self,
        server_url: str,
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
        initial_backoff: float = 0.05,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    async def _call_once(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        with trace_boundary("mcp.call_tool", boundary="mcp", tool=tool_name):
            async with (
                streamable_http_client(self.server_url) as streams,
                ClientSession(
                    streams[0],
                    streams[1],
                    read_timeout_seconds=self.timeout_seconds,
                ) as session,
            ):
                await session.initialize()
                return await session.call_tool(
                    tool_name,
                    arguments=arguments,
                    read_timeout_seconds=self.timeout_seconds,
                )

    @staticmethod
    def _text_content(result: types.CallToolResult) -> str:
        return "\n".join(
            item.text for item in result.content if isinstance(item, types.TextContent)
        )

    @classmethod
    def _decode_result(cls, result: Any) -> Any:
        if not isinstance(result, types.CallToolResult):
            raise McpAdapterError(
                category=SafeErrorCategory.PIPELINE_CONTRACT_ERROR,
                message="Unexpected MCP result type",
            )
        if result.is_error:
            try:
                error = json.loads(cls._text_content(result))
                safe_error = SafeError.model_validate(error)
                raise McpAdapterError(
                    category=safe_error.category,
                    message=safe_error.message,
                    retryable=safe_error.retryable,
                    remediation=safe_error.remediation,
                )
            except McpAdapterError:
                raise
            except (ValueError, TypeError, ValidationError):
                raise McpAdapterError(
                    category=SafeErrorCategory.INTERNAL_ERROR,
                    message="Malformed MCP tool error",
                ) from None

        if result.structured_content is not None:
            return result.structured_content
        try:
            return json.loads(cls._text_content(result))
        except (ValueError, TypeError):
            raise McpAdapterError(
                category=SafeErrorCategory.PIPELINE_CONTRACT_ERROR,
                message="MCP result did not contain structured JSON",
            ) from None

    async def _call_with_retry(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        backoff = self.initial_backoff
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._call_once(tool_name, arguments), timeout=self.timeout_seconds
                )
                return self._decode_result(result)
            except asyncio.CancelledError:
                raise
            except McpAdapterError as exc:
                if not exc.retryable or attempt == self.max_retries:
                    raise
            except TimeoutError as exc:
                if attempt == self.max_retries:
                    raise McpAdapterError(
                        category=SafeErrorCategory.PIPELINE_TIMEOUT,
                        message="MCP call timed out",
                        retryable=True,
                    ) from exc
            except Exception as exc:
                if attempt == self.max_retries:
                    raise McpAdapterError(
                        category=SafeErrorCategory.PIPELINE_UNAVAILABLE,
                        message="MCP transport failed",
                        retryable=True,
                    ) from exc

            await asyncio.sleep(backoff + random.uniform(0.0, backoff * 0.1))
            backoff *= 2.0

        raise AssertionError("MCP retry loop exhausted without a result")

    @staticmethod
    def _require_contract(condition: bool) -> None:
        if not condition:
            raise McpAdapterError(
                category=SafeErrorCategory.PIPELINE_CONTRACT_ERROR,
                message="MCP result does not match the request",
            )

    @staticmethod
    def _validate(model_type: type[ModelT], payload: Any) -> ModelT:
        try:
            return model_type.model_validate(payload)
        except ValidationError as exc:
            raise McpAdapterError(
                category=SafeErrorCategory.PIPELINE_CONTRACT_ERROR,
                message="MCP result failed local contract validation",
            ) from exc

    async def lookup_vehicle(self, vin: str) -> VehicleRevisionResponse:
        clean_vin = vin.strip().upper()
        payload = await self._call_with_retry("lookup_vehicle", {"vin": clean_vin})
        revision = self._validate(VehicleRevisionResponse, payload)
        self._require_contract(revision.vin == clean_vin)
        return revision

    async def get_vehicle_revision(self, vin: str, revision_number: int) -> VehicleRevisionResponse:
        clean_vin = vin.strip().upper()
        payload = await self._call_with_retry(
            "get_vehicle_revision", {"vin": clean_vin, "revision_number": revision_number}
        )
        revision = self._validate(VehicleRevisionResponse, payload)
        self._require_contract(
            revision.vin == clean_vin and revision.revision_number == revision_number
        )
        return revision

    async def explain_vehicle_field(self, vin: str, field_name: str) -> FieldExplanationResult:
        clean_vin = vin.strip().upper()
        clean_field = field_name.strip().lower()
        payload = await self._call_with_retry(
            "explain_vehicle_field", {"vin": clean_vin, "field_name": clean_field}
        )
        explanation = self._validate(FieldExplanationResult, payload)
        self._require_contract(
            explanation.vin == clean_vin and explanation.field_name == clean_field
        )
        return explanation

    async def get_vehicle_history(
        self, vin: str, limit: int = 20, before_revision: int | None = None
    ) -> list[VehicleRevisionResponse]:
        clean_vin = vin.strip().upper()
        arguments: dict[str, Any] = {"vin": clean_vin, "limit": limit}
        if before_revision is not None:
            arguments["before_revision"] = before_revision
        revisions: list[VehicleRevisionResponse] = []
        payload = await self._call_with_retry("get_vehicle_history", arguments)
        if not isinstance(payload, list):
            raise McpAdapterError(
                category=SafeErrorCategory.PIPELINE_CONTRACT_ERROR,
                message="MCP history result is not a list",
            )
        for item in payload:
            revision = self._validate(VehicleRevisionResponse, item)
            revisions.append(revision)
        self._require_contract(all(revision.vin == clean_vin for revision in revisions))
        return revisions

    async def get_source_observation(self, observation_id: str) -> SourceObservationResponse:
        clean_id = observation_id.strip()
        payload = await self._call_with_retry(
            "get_source_observation", {"observation_id": clean_id}
        )
        observation = self._validate(SourceObservationResponse, payload)
        self._require_contract(observation.observation_id == clean_id)
        return observation


def create_mcp_adapter(settings: Settings) -> VehicleMcpClientAdapter:
    """Build the configured MCP adapter without silently falling back to fake evidence."""
    if settings.mcp_server_url:
        return StreamableHttpVehicleMcpAdapter(
            server_url=settings.mcp_server_url,
            timeout_seconds=settings.mcp_timeout_seconds,
            max_retries=settings.mcp_max_retries,
            initial_backoff=settings.mcp_initial_backoff,
        )
    return UnavailableVehicleMcpAdapter()


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
        self._revisions: dict[tuple[str, int], VehicleRevisionResponse] = {}
        self._explanations: dict[tuple[str, str], FieldExplanationResult] = {}
        self._observations: dict[str, SourceObservationResponse] = {}
        self._history: dict[str, list[VehicleRevisionResponse]] = {}
        self._transient_failures: dict[str, int] = {}

    def seed_vehicle(self, revision: VehicleRevisionResponse) -> None:
        """Seed a vehicle revision in fake store."""
        clean_vin = revision.vin.strip().upper()
        self._vehicles[clean_vin] = revision
        self._revisions[(clean_vin, revision.revision_number)] = revision
        if clean_vin not in self._history:
            self._history[clean_vin] = [revision]
        elif revision not in self._history[clean_vin]:
            self._history[clean_vin].append(revision)
            self._history[clean_vin].sort(key=lambda r: r.revision_number, reverse=True)

    def seed_field_explanation(self, explanation: FieldExplanationResult) -> None:
        """Seed a field explanation in fake store."""
        clean_vin = explanation.vin.strip().upper()
        clean_field = explanation.field_name.strip().lower()
        self._explanations[(clean_vin, clean_field)] = explanation

    def seed_source_observation(self, observation: SourceObservationResponse) -> None:
        """Seed a source observation in fake store."""
        self._observations[observation.observation_id] = observation

    def simulate_transient_failures(self, vin: str, failure_count: int = 1) -> None:
        """Simulate transient failures before succeeding on a VIN."""
        clean_vin = vin.strip().upper()
        self._transient_failures[clean_vin] = failure_count

    async def _execute_with_retry(self, operation: Any, vin: str) -> Any:
        """Execute operation with bounded retry and exponential backoff."""
        clean_vin = vin.strip().upper()
        attempts = 0
        backoff = self.initial_backoff

        while True:
            attempts += 1
            if clean_vin in self._transient_failures and self._transient_failures[clean_vin] > 0:
                self._transient_failures[clean_vin] -= 1
                if attempts <= self.max_retries:
                    jitter = random.uniform(0.005, 0.015)
                    await asyncio.sleep(min(backoff + jitter, self.timeout_seconds))
                    backoff *= 2.0
                    continue
                raise McpAdapterError(
                    category=SafeErrorCategory.PIPELINE_TIMEOUT,
                    message=(
                        f"Vehicle intelligence pipeline timed out after {self.max_retries} attempts"
                    ),
                    retryable=True,
                    remediation="Retry assessment intake after upstream pipeline recovers.",
                )

            return operation()

    async def lookup_vehicle(self, vin: str) -> VehicleRevisionResponse:
        """Lookup vehicle revision with simulated retry behavior."""
        clean_vin = vin.strip().upper()

        def op() -> VehicleRevisionResponse:
            if clean_vin not in self._vehicles:
                raise McpAdapterError(
                    category=SafeErrorCategory.VEHICLE_NOT_FOUND,
                    message=f"No vehicle intelligence record found for VIN: {clean_vin}",
                    retryable=False,
                    remediation=(
                        "Verify VIN format and ensure vehicle has been ingested by pipeline."
                    ),
                )
            return self._vehicles[clean_vin]

        return await self._execute_with_retry(op, clean_vin)  # type: ignore[no-any-return]

    async def get_vehicle_revision(self, vin: str, revision_number: int) -> VehicleRevisionResponse:
        """Retrieve exact canonical revision of a vehicle."""
        clean_vin = vin.strip().upper()
        key = (clean_vin, revision_number)
        if key in self._revisions:
            return self._revisions[key]

        raise McpAdapterError(
            category=SafeErrorCategory.REVISION_NOT_FOUND,
            message=f"Revision {revision_number} not found for VIN: {clean_vin}",
            retryable=False,
            remediation="Request valid revision number within known vehicle history range.",
        )

    async def explain_vehicle_field(self, vin: str, field_name: str) -> FieldExplanationResult:
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
            prov = tuple(veh.field_provenance.get(clean_field, ()))
            conflicts = tuple(c for c in veh.conflicts if c.field_name == clean_field)
            return FieldExplanationResult(
                vin=clean_vin,
                revision_number=veh.revision_number,
                field_name=clean_field,
                outcome=outcome,
                value=val,
                provenance=prov,
                conflicts=conflicts,
                confidence_score=veh.confidence.score,
                confidence_band=veh.confidence.band,
                available_fields=tuple(sorted(veh.canonical_fields.keys())),
                rationale="Resolved from canonical fields.",
                synthetic_notice=veh.synthetic_notice,
            )

        raise McpAdapterError(
            category=SafeErrorCategory.VEHICLE_NOT_FOUND,
            message=f"Cannot explain field '{clean_field}': vehicle {clean_vin} not found",
            retryable=False,
            remediation="Verify VIN exists before requesting field explanation.",
        )

    async def get_vehicle_history(
        self, vin: str, limit: int = 20, before_revision: int | None = None
    ) -> list[VehicleRevisionResponse]:
        """Retrieve simulated newest-first vehicle revision history."""
        clean_vin = vin.strip().upper()
        if clean_vin not in self._history:
            return []

        revs = self._history[clean_vin]
        if before_revision is not None:
            revs = [r for r in revs if r.revision_number < before_revision]

        revs_sorted = sorted(revs, key=lambda r: r.revision_number, reverse=True)
        return revs_sorted[:limit]

    async def get_source_observation(self, observation_id: str) -> SourceObservationResponse:
        """Retrieve source observation by identifier."""
        clean_id = observation_id.strip()
        if clean_id in self._observations:
            return self._observations[clean_id]

        raise McpAdapterError(
            category=SafeErrorCategory.OBSERVATION_NOT_FOUND,
            message=f"Source observation not found: {clean_id}",
            retryable=False,
            remediation="Ensure observation identifier matches an active provenance link.",
        )
