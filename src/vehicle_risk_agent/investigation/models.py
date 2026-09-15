"""Strict models for generated investigation proposals and safe results."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vehicle_risk_agent.evidence.models import FieldExplanationResult, VehicleRevisionResponse
from vehicle_risk_agent.policy.models import PolicyCitation

ALLOWED_EVIDENCE_TARGETS: frozenset[str] = frozenset(
    {
        "ppsr_result",
        "stolen_status",
        "writeoff_status",
        "odometer_reading",
        "registration_status",
        "ownership_history",
        "wof_status",
        "safety_recall",
        "inspection_history",
        "make",
        "model",
        "year",
        "plate",
        "is_commercial",
        "vehicle_usage",
        "damage_history",
        "fuel_type",
    }
)


class InvestigationAction(StrEnum):
    """Read-only actions available to the investigation dispatcher."""

    EXPLAIN_VEHICLE_FIELD = "explain_vehicle_field"
    GET_VEHICLE_HISTORY = "get_vehicle_history"
    GET_VEHICLE_REVISION = "get_vehicle_revision"
    SEARCH_POLICY = "search_policy"


class InvestigationKind(StrEnum):
    """Top-level generated proposal kinds."""

    NO_ACTION = "NO_ACTION"
    REQUEST = "REQUEST"


class NoActionProposal(BaseModel):
    """Proposal that deliberately performs no supplementary call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["NO_ACTION"] = "NO_ACTION"


class ExplainVehicleFieldArguments(BaseModel):
    """Arguments for a graph-owned VIN field explanation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: str = Field(min_length=1)

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_EVIDENCE_TARGETS:
            raise ValueError("field_name is not an allowed evidence target")
        return normalized


class EmptyArguments(BaseModel):
    """Empty object required for history requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RevisionArguments(BaseModel):
    """Arguments for a bounded exact revision lookup."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision_number: int = Field(ge=1, le=1000)


class SearchPolicyArguments(BaseModel):
    """Bounded policy search query; corpus and top-k remain graph-owned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=200)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query cannot be blank")
        return stripped


class ExplainVehicleFieldProposal(BaseModel):
    """Typed field explanation request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["REQUEST"] = "REQUEST"
    action: Literal[InvestigationAction.EXPLAIN_VEHICLE_FIELD] = (
        InvestigationAction.EXPLAIN_VEHICLE_FIELD
    )
    arguments: ExplainVehicleFieldArguments


class GetVehicleHistoryProposal(BaseModel):
    """Typed history request with no model-owned paging arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["REQUEST"] = "REQUEST"
    action: Literal[InvestigationAction.GET_VEHICLE_HISTORY] = (
        InvestigationAction.GET_VEHICLE_HISTORY
    )
    arguments: EmptyArguments


class GetVehicleRevisionProposal(BaseModel):
    """Typed exact revision request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["REQUEST"] = "REQUEST"
    action: Literal[InvestigationAction.GET_VEHICLE_REVISION] = (
        InvestigationAction.GET_VEHICLE_REVISION
    )
    arguments: RevisionArguments


class SearchPolicyProposal(BaseModel):
    """Typed pinned-corpus policy search request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["REQUEST"] = "REQUEST"
    action: Literal[InvestigationAction.SEARCH_POLICY] = InvestigationAction.SEARCH_POLICY
    arguments: SearchPolicyArguments


class InvestigationProposal(BaseModel):
    """Strict proposal object with action-specific nested validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: InvestigationKind
    action: InvestigationAction | None = None
    arguments: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError("proposal must be an object")
        kind = value.get("kind")
        if kind == InvestigationKind.NO_ACTION.value and set(value) != {"kind"}:
            raise ValueError("NO_ACTION accepts exactly the kind field")
        if kind == InvestigationKind.REQUEST.value and set(value) != {
            "kind",
            "action",
            "arguments",
        }:
            raise ValueError("REQUEST requires exactly kind, action, and arguments")
        return value

    @model_validator(mode="after")
    def validate_variant(self) -> InvestigationProposal:
        if self.kind == InvestigationKind.NO_ACTION:
            if self.action is not None or self.arguments is not None:
                raise ValueError("NO_ACTION accepts no action or arguments")
            return self
        if self.action is None or self.arguments is None:
            raise ValueError("REQUEST requires one action and non-null arguments")
        parsed: BaseModel
        if self.action == InvestigationAction.EXPLAIN_VEHICLE_FIELD:
            parsed = ExplainVehicleFieldArguments.model_validate(self.arguments)
        elif self.action == InvestigationAction.GET_VEHICLE_HISTORY:
            parsed = EmptyArguments.model_validate(self.arguments)
        elif self.action == InvestigationAction.GET_VEHICLE_REVISION:
            parsed = RevisionArguments.model_validate(self.arguments)
        elif self.action == InvestigationAction.SEARCH_POLICY:
            parsed = SearchPolicyArguments.model_validate(self.arguments)
        else:
            raise ValueError("unsupported investigation action")
        object.__setattr__(self, "arguments", parsed.model_dump())
        return self

    @classmethod
    def from_tool_input(
        cls, value: Any
    ) -> (
        NoActionProposal
        | ExplainVehicleFieldProposal
        | GetVehicleHistoryProposal
        | GetVehicleRevisionProposal
        | SearchPolicyProposal
    ):
        """Validate one exact proposal object and return its typed variant."""
        parsed = cls.model_validate(value)
        if parsed.kind == InvestigationKind.NO_ACTION:
            return NoActionProposal()
        assert parsed.action is not None
        assert parsed.arguments is not None
        payload = {
            "kind": parsed.kind.value,
            "action": parsed.action.value,
            "arguments": parsed.arguments,
        }
        if parsed.action == InvestigationAction.EXPLAIN_VEHICLE_FIELD:
            return ExplainVehicleFieldProposal.model_validate(payload)
        if parsed.action == InvestigationAction.GET_VEHICLE_HISTORY:
            return GetVehicleHistoryProposal.model_validate(payload)
        if parsed.action == InvestigationAction.GET_VEHICLE_REVISION:
            return GetVehicleRevisionProposal.model_validate(payload)
        return SearchPolicyProposal.model_validate(payload)

    @property
    def typed_arguments(self) -> BaseModel | None:
        """Return the action-specific strict argument model."""
        if self.action == InvestigationAction.EXPLAIN_VEHICLE_FIELD:
            return ExplainVehicleFieldArguments.model_validate(self.arguments)
        if self.action == InvestigationAction.GET_VEHICLE_HISTORY:
            return EmptyArguments.model_validate(self.arguments or {})
        if self.action == InvestigationAction.GET_VEHICLE_REVISION:
            return RevisionArguments.model_validate(self.arguments)
        if self.action == InvestigationAction.SEARCH_POLICY:
            return SearchPolicyArguments.model_validate(self.arguments)
        return None


class ProviderReadiness(BaseModel):
    """Safe preflight result for paid investigation execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    model: str
    strict_schema: bool
    token_counting: bool
    max_retries: int = Field(ge=0)
    failure_code: str | None = None


class ProviderUsage(BaseModel):
    """Provider-reported token usage required for budget accounting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ProviderToolBlock(BaseModel):
    """Minimal provider content block accepted at the trust boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["tool_use"]
    name: str
    input: dict[str, Any]


class VehicleHistoryResult(BaseModel):
    """Validated, bounded history returned by supplementary investigation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vin: str = Field(min_length=17, max_length=17)
    revisions: tuple[VehicleRevisionResponse, ...] = Field(default_factory=tuple, max_length=5)

    @field_validator("revisions", mode="before")
    @classmethod
    def coerce_revisions_to_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_revisions(self) -> VehicleHistoryResult:
        if any(revision.vin != self.vin for revision in self.revisions):
            raise ValueError("vehicle history response does not match request")
        revision_numbers = tuple(revision.revision_number for revision in self.revisions)
        if len(revision_numbers) != len(set(revision_numbers)):
            raise ValueError("vehicle history response contains duplicate revisions")
        return self


type ProposalValue = (
    NoActionProposal
    | ExplainVehicleFieldProposal
    | GetVehicleHistoryProposal
    | GetVehicleRevisionProposal
    | SearchPolicyProposal
)


class SubmitInvestigationToolResponse(BaseModel):
    """Validated response from the forced single proposal tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stop_reason: Literal["tool_use"]
    content: tuple[ProviderToolBlock, ...]
    usage: ProviderUsage
    proposal: ProposalValue

    @model_validator(mode="after")
    def validate_single_forced_block(self) -> SubmitInvestigationToolResponse:
        if len(self.content) != 1:
            raise ValueError("provider response must contain exactly one content block")
        block = self.content[0]
        if block.name != "submit_investigation_proposal":
            raise ValueError("provider response must contain the forced investigation tool")
        return self

    @classmethod
    def validate_provider_payload(cls, payload: Any) -> SubmitInvestigationToolResponse:
        """Reject every response shape except one named strict tool block."""
        if not isinstance(payload, dict):
            raise ValueError("provider response must be an object")
        raw_content = payload.get("content")
        if not isinstance(raw_content, list):
            raw_content = []
        block_input = (
            raw_content[0].get("input")
            if len(raw_content) == 1 and isinstance(raw_content[0], dict)
            else {}
        )
        return cls.model_validate(
            {
                "stop_reason": payload.get("stop_reason"),
                "content": raw_content,
                "usage": payload.get("usage"),
                "proposal": InvestigationProposal.from_tool_input(block_input),
            }
        )


class InvestigationContext(BaseModel):
    """Minimized, bounded context supplied to a proposal provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    questions: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    evidence_targets: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    evidence_summaries: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    prior_result_summaries: tuple[str, ...] = Field(default_factory=tuple, max_length=3)
    stable_references: tuple[str, ...] = Field(default_factory=tuple, max_length=50)

    @field_validator(
        "questions",
        "evidence_targets",
        "evidence_summaries",
        "prior_result_summaries",
        "stable_references",
        mode="before",
    )
    @classmethod
    def coerce_sequences(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if any(not item or len(item) > 200 for item in cleaned):
            raise ValueError("questions must be non-empty and at most 200 characters")
        return cleaned

    @field_validator("evidence_targets")
    @classmethod
    def validate_targets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip().lower() for item in value)
        if any(item not in ALLOWED_EVIDENCE_TARGETS for item in cleaned):
            raise ValueError("evidence_targets contains a prohibited field")
        return cleaned

    @field_validator("evidence_summaries", "prior_result_summaries")
    @classmethod
    def validate_summaries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if any(not item or len(item) > 500 for item in cleaned):
            raise ValueError("summaries must be non-empty and at most 500 characters")
        if any("raw_payload" in item.lower() for item in cleaned):
            raise ValueError("raw observations are not allowed in proposal context")
        return cleaned

    @field_validator("stable_references")
    @classmethod
    def validate_references(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if any(not item or len(item) > 128 for item in cleaned):
            raise ValueError("references must be bounded")
        return cleaned


class InvestigationLimitation(BaseModel):
    """Sanitized reason why supplementary investigation did not complete."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=500)


class InvestigationResult(BaseModel):
    """Safe additive result returned by the dispatcher."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: InvestigationAction | None = None
    summary: str = Field(min_length=1, max_length=500)
    references: tuple[str, ...] = Field(default_factory=tuple)
    evidence_result: (
        FieldExplanationResult | VehicleRevisionResponse | VehicleHistoryResult | None
    ) = None
    policy_citations: tuple[PolicyCitation, ...] = Field(default_factory=tuple)
    observed_query: str | None = Field(default=None, max_length=200)
    limitation: InvestigationLimitation | None = None
    completed: bool = True
    dispatched: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def safe_metadata(self) -> dict[str, Any]:
        """Return a bounded report projection without raw upstream payloads."""
        return {
            "action": self.action.value if self.action else None,
            "summary": self.summary,
            "references": list(self.references[:50]),
            "policy_citation_refs": [
                citation.passage_id for citation in self.policy_citations[:20]
            ],
            "observed_query": self.observed_query,
            "completed": self.completed,
            "dispatched": self.dispatched,
            "limitation": self.limitation.model_dump() if self.limitation else None,
        }


class ProviderProposalResult(BaseModel):
    """Provider proposal plus usage needed by the durable ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal: ProposalValue
    usage: ProviderUsage


def proposal_json_schema() -> dict[str, Any]:
    """Return the exact five-variant provider-compatible strict schema."""
    target_enum = sorted(ALLOWED_EVIDENCE_TARGETS)
    variants: list[dict[str, Any]] = [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind"],
            "properties": {"kind": {"const": "NO_ACTION", "type": "string"}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "action", "arguments"],
            "properties": {
                "kind": {"const": "REQUEST", "type": "string"},
                "action": {
                    "const": InvestigationAction.EXPLAIN_VEHICLE_FIELD.value,
                    "type": "string",
                },
                "arguments": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["field_name"],
                    "properties": {"field_name": {"type": "string", "enum": target_enum}},
                },
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "action", "arguments"],
            "properties": {
                "kind": {"const": "REQUEST", "type": "string"},
                "action": {
                    "const": InvestigationAction.GET_VEHICLE_HISTORY.value,
                    "type": "string",
                },
                "arguments": {"type": "object", "additionalProperties": False},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "action", "arguments"],
            "properties": {
                "kind": {"const": "REQUEST", "type": "string"},
                "action": {
                    "const": InvestigationAction.GET_VEHICLE_REVISION.value,
                    "type": "string",
                },
                "arguments": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["revision_number"],
                    "properties": {
                        "revision_number": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                        }
                    },
                },
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "action", "arguments"],
            "properties": {
                "kind": {"const": "REQUEST", "type": "string"},
                "action": {"const": InvestigationAction.SEARCH_POLICY.value, "type": "string"},
                "arguments": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["query"],
                    "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 200}},
                },
            },
        },
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "oneOf": variants,
    }
