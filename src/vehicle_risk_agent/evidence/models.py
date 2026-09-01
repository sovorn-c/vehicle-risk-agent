"""Strict local mirror Pydantic models for upstream Vehicle Intelligence MCP contracts."""

import re
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

VIN_PATTERN = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")


class FieldOutcome(StrEnum):
    """Deterministic field explanation outcome."""

    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    ABSENT = "ABSENT"


class ConfidenceBand(StrEnum):
    """Calibrated tier of confidence rating."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ConfidenceAssessment(BaseModel):
    """Reproducible assessment of canonical evidence strength."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: int = Field(description="Integer confidence score from 0 through 100")
    band: ConfidenceBand = Field(description="Confidence band rating")
    field_scores: dict[str, int] = Field(
        default_factory=dict, description="Per-field weighted confidence scores"
    )
    field_components: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Detailed authority, agreement, freshness, and validation breakdowns",
    )
    rule_version: str = Field(description="Version of confidence calculation rule")
    explanation: str = Field(description="Human-readable explanation of score factors")


class ProvenanceLink(BaseModel):
    """Immutable trace pointing back to the exact source observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observation_id: str = Field(description="Unique observation identifier")
    source_system: str = Field(description="Originating source system")
    source_record_id: str = Field(description="Record ID in source")
    retrieved_at: datetime = Field(description="Timestamp observation was retrieved from source")
    synthetic: bool = Field(
        default=False, description="Flag indicating synthetic demonstration source"
    )


class CandidateValue(BaseModel):
    """Normalized field value proposed by one source observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_name: str = Field(description="Canonical field name")
    value: Any = Field(description="Extracted attribute value")
    provenance: ProvenanceLink = Field(description="Lineage to source observation")


class ConflictState(StrEnum):
    """Lifecycle state of a detected field conflict."""

    DETECTED = "DETECTED"
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"


class FieldConflict(BaseModel):
    """Recorded disagreement between credible incompatible candidate values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_name: str = Field(description="Target canonical field")
    conflicting_candidates: list[CandidateValue] = Field(
        description="All competing candidate values"
    )
    state: ConflictState = Field(
        default=ConflictState.DETECTED, description="Current conflict resolution state"
    )
    winning_value: Any | None = Field(
        default=None, description="Winning candidate value if resolved"
    )
    rule_version: str = Field(default="", description="Version of resolution rule applied")
    rationale: str = Field(default="", description="Explanation of resolution decision")


class VehicleRevisionResponse(BaseModel):
    """Canonical vehicle revision representation validated at the upstream MCP boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vin: str = Field(description="Canonical 17-character VIN")
    revision_id: str = Field(description="Unique revision identifier")
    revision_number: int = Field(description="Monotonic revision number")
    material_hash: str = Field(description="SHA-256 fingerprint of canonical material")
    canonical_fields: dict[str, Any] = Field(description="Resolved canonical fields")
    field_provenance: dict[str, list[ProvenanceLink]] = Field(
        default_factory=dict, description="Lineage to all supporting source observations"
    )
    conflicts: list[FieldConflict] = Field(
        default_factory=list, description="Recorded field conflicts"
    )
    confidence: ConfidenceAssessment = Field(description="Confidence assessment")
    as_of: datetime = Field(description="Evaluation timestamp")
    published_at: datetime = Field(description="Database publication timestamp")
    synthetic_notice: str | None = Field(
        default=None,
        description="Disclaimer notice when record contains synthetic demonstration data",
    )

    @field_validator("vin")
    @classmethod
    def validate_vin(cls, v: str) -> str:
        if not VIN_PATTERN.match(v):
            raise ValueError(
                "VIN must be exactly 17 ASCII alphanumeric characters excluding letters I, O, and Q"
            )
        return v


class FieldExplanationResult(BaseModel):
    """Deterministic projection of one vehicle field's current evidence state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vin: str = Field(description="Canonical 17-character VIN")
    revision_number: int = Field(description="Canonical revision number evaluated")
    field_name: str = Field(description="Evaluated field name")
    outcome: FieldOutcome = Field(description="RESOLVED, UNRESOLVED, or ABSENT outcome")
    value: Any | None = Field(default=None, description="Resolved canonical value if present")
    provenance: list[ProvenanceLink] = Field(
        default_factory=list, description="Lineage to supporting source observations"
    )
    conflicts: list[FieldConflict] = Field(
        default_factory=list, description="Recorded field conflicts if any"
    )
    confidence_score: int | None = Field(
        default=None, description="Overall revision confidence score"
    )
    confidence_band: ConfidenceBand | None = Field(
        default=None, description="Overall revision confidence band"
    )
    field_confidence_score: int | None = Field(
        default=None, description="Per-field confidence score if evaluated"
    )
    field_components: dict[str, int] | None = Field(
        default=None, description="Per-field confidence score component breakdown"
    )
    available_fields: list[str] = Field(
        default_factory=list,
        description="Sorted available canonical and conflicting field names",
    )
    rationale: str | None = Field(
        default=None, description="Human-readable explanation of outcome or conflict rationale"
    )
    synthetic_notice: str | None = Field(
        default=None,
        description="Disclaimer notice when record contains synthetic demonstration data",
    )


class SourceObservationResponse(BaseModel):
    """Exact immutable source observation containing raw source evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observation_id: str = Field(description="Unique observation identifier")
    source_system: str = Field(description="Source system name")
    source_record_id: str = Field(description="Source-native record identifier")
    ingestion_run_id: str = Field(description="Ingestion run identifier")
    raw_payload: str = Field(description="Exact raw payload string captured from source")
    payload_hash_sha256: str = Field(description="SHA-256 fingerprint of the raw payload")
    retrieved_at: datetime = Field(description="Timestamp when source evidence was retrieved")
    synthetic: bool = Field(
        description="Flag indicating if observation contains demonstration data"
    )


class SafeErrorCategory(StrEnum):
    """Stable public error categories defined in ubiquitous language."""

    INVALID_INPUT = "INVALID_INPUT"
    VEHICLE_NOT_FOUND = "VEHICLE_NOT_FOUND"
    REVISION_NOT_FOUND = "REVISION_NOT_FOUND"
    OBSERVATION_NOT_FOUND = "OBSERVATION_NOT_FOUND"
    PIPELINE_TIMEOUT = "PIPELINE_TIMEOUT"
    PIPELINE_UNAVAILABLE = "PIPELINE_UNAVAILABLE"
    PIPELINE_CONTRACT_ERROR = "PIPELINE_CONTRACT_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class SafeError(BaseModel):
    """Standardized error structure returned in tool error messages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: SafeErrorCategory = Field(description="Stable safe error category")
    message: str = Field(description="Safe error explanation with no secrets or stack traces")
    retryable: bool = Field(description="Indicates whether client retry could succeed")
    remediation: str = Field(description="Actionable guidance to resolve the issue")
