"""Graph-owned execution of strictly typed supplementary proposals."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import ValidationError

from vehicle_risk_agent.evidence.models import (
    FieldExplanationResult,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.investigation.models import (
    InvestigationAction,
    InvestigationLimitation,
    InvestigationProposal,
    InvestigationResult,
    ProposalValue,
    VehicleHistoryResult,
)
from vehicle_risk_agent.policy.models import PolicyCitation

_VEHICLE_FIELD_ALIASES = {"odometer_reading": "odometer_km"}


class InvestigationVehicleClient(Protocol):
    """Read-only vehicle calls available to the dispatcher."""

    async def explain_vehicle_field(self, vin: str, field_name: str) -> Any: ...

    async def get_vehicle_history(
        self, vin: str, limit: int = 20, before_revision: int | None = None
    ) -> Any: ...

    async def get_vehicle_revision(self, vin: str, revision_number: int) -> Any: ...


class InvestigationPolicyRetriever(Protocol):
    """Pinned policy retrieval boundary supplied by the graph."""

    async def retrieve(self, query: str) -> Any: ...


class InvestigationDispatcher:
    """Translate proposals into bounded calls using graph-owned vehicle identity."""

    def __init__(
        self,
        vehicle: InvestigationVehicleClient,
        policy: InvestigationPolicyRetriever,
        history_limit: int = 5,
    ) -> None:
        self.vehicle = vehicle
        self.policy = policy
        self.history_limit = min(max(history_limit, 1), 5)

    async def dispatch(
        self,
        vin: str,
        proposal: ProposalValue | InvestigationProposal,
        current_revision: int | None = None,
    ) -> InvestigationResult:
        """Execute only the proposal's allow-listed action, never model arguments for VIN."""
        typed = (
            InvestigationProposal.from_tool_input(proposal.model_dump())
            if isinstance(proposal, InvestigationProposal)
            else proposal
        )
        if typed.kind == "NO_ACTION":
            return InvestigationResult(
                summary="No supplementary investigation was requested.",
                completed=True,
                dispatched=False,
            )

        assert typed.action is not None
        try:
            if typed.action == InvestigationAction.EXPLAIN_VEHICLE_FIELD:
                field_arguments = typed.arguments
                assert field_arguments is not None
                field_name = field_arguments.field_name
                upstream_field_name = _VEHICLE_FIELD_ALIASES.get(field_name, field_name)
                bounded_call = getattr(self.vehicle, "bounded_investigation_call", None)
                if bounded_call is not None:
                    raw = await bounded_call(
                        "explain_vehicle_field",
                        {"vin": vin, "field_name": upstream_field_name},
                        max_retries=0,
                    )
                else:
                    raw = await self.vehicle.explain_vehicle_field(vin, upstream_field_name)
                result = FieldExplanationResult.model_validate(raw)
                if result.vin != vin or result.field_name != upstream_field_name:
                    raise ValueError("vehicle response does not match request")
                if upstream_field_name != field_name:
                    result = result.model_copy(update={"field_name": field_name})
                return InvestigationResult(
                    action=typed.action,
                    summary=f"Supplementary explanation returned for {field_name}.",
                    references=tuple(link.observation_id for link in result.provenance[:20]),
                    evidence_result=result,
                    completed=True,
                    dispatched=True,
                )

            if typed.action == InvestigationAction.GET_VEHICLE_HISTORY:
                bounded_call = getattr(self.vehicle, "bounded_investigation_call", None)
                if bounded_call is not None:
                    history_arguments: dict[str, Any] = {"vin": vin, "limit": self.history_limit}
                    if current_revision is not None:
                        history_arguments["before_revision"] = current_revision
                    raw_history = await bounded_call(
                        "get_vehicle_history", history_arguments, max_retries=0
                    )
                else:
                    raw_history = await self.vehicle.get_vehicle_history(
                        vin,
                        limit=self.history_limit,
                        before_revision=current_revision,
                    )
                revisions = [VehicleRevisionResponse.model_validate(item) for item in raw_history]
                if current_revision is not None and any(
                    item.revision_number >= current_revision for item in revisions
                ):
                    raise ValueError("vehicle history response contains a non-historical revision")
                revision_numbers = [item.revision_number for item in revisions]
                if len(revision_numbers) != len(set(revision_numbers)):
                    raise ValueError("vehicle history response contains duplicate revisions")
                history = tuple(
                    sorted(
                        revisions,
                        key=lambda item: item.revision_number,
                        reverse=True,
                    )[: self.history_limit]
                )
                if any(item.vin != vin for item in history):
                    raise ValueError("vehicle history response does not match request")
                history_result = VehicleHistoryResult(vin=vin, revisions=history)
                return InvestigationResult(
                    action=typed.action,
                    summary=f"Supplementary history returned {len(history)} revisions.",
                    references=tuple(item.revision_id for item in history[:20]),
                    evidence_result=history_result,
                    completed=True,
                    dispatched=True,
                )

            if typed.action == InvestigationAction.GET_VEHICLE_REVISION:
                revision_arguments = typed.arguments
                assert revision_arguments is not None
                revision_number = revision_arguments.revision_number
                bounded_call = getattr(self.vehicle, "bounded_investigation_call", None)
                if bounded_call is not None:
                    raw_revision = await bounded_call(
                        "get_vehicle_revision",
                        {"vin": vin, "revision_number": revision_number},
                        max_retries=0,
                    )
                else:
                    raw_revision = await self.vehicle.get_vehicle_revision(vin, revision_number)
                revision = VehicleRevisionResponse.model_validate(raw_revision)
                if revision.vin != vin or revision.revision_number != revision_number:
                    raise ValueError("vehicle revision response does not match request")
                return InvestigationResult(
                    action=typed.action,
                    summary=f"Supplementary revision {revision_number} returned.",
                    references=(revision.revision_id, revision.material_hash),
                    evidence_result=revision,
                    completed=True,
                    dispatched=True,
                )

            search_arguments = typed.arguments
            assert search_arguments is not None
            query = search_arguments.query
            retrieved = await self.policy.retrieve(query)
            citations = tuple(
                PolicyCitation.model_validate(item) for item in retrieved.citations[:20]
            )
            return InvestigationResult(
                action=typed.action,
                summary=f"Supplementary policy search returned {len(citations)} citations.",
                references=tuple(citation.passage_id for citation in citations),
                policy_citations=citations,
                observed_query=query,
                completed=True,
                dispatched=True,
            )
        except (ValidationError, ValueError, TypeError, AttributeError):
            return self._limited_result(typed.action, "INVALID_RESULT")
        except Exception:
            return self._limited_result(typed.action, "UPSTREAM_UNAVAILABLE")

    @staticmethod
    def _limited_result(action: InvestigationAction, code: str) -> InvestigationResult:
        return InvestigationResult(
            action=action,
            summary="Supplementary investigation was not completed.",
            limitation=InvestigationLimitation(
                code=code,
                message="The supplementary result was unavailable or failed validation.",
            ),
            completed=False,
            dispatched=True,
        )
