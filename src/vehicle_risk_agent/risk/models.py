"""Domain models for Risk Policy, Factors, Bands, Mandatory Findings, and Results."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vehicle_risk_agent.evidence.sufficiency import MissingEvidenceFinding


class RiskPolicyLifecycleState(StrEnum):
    """Lifecycle state of a Risk Policy."""

    DRAFT = "DRAFT"
    READY = "READY"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


class RiskFactor(StrEnum):
    """Canonical risk factors evaluated against vehicle evidence."""

    MATCH = "MATCH"  # PPSR registered security interest match (+30)
    LISTED = "LISTED"  # Stolen vehicle register listed (+45)
    REPAIRABLE = "REPAIRABLE"  # NZTA repairable write-off status (+20)
    STATUTORY = "STATUTORY"  # NZTA statutory/non-repairable write-off or deregistration (+40)


class RiskBand(StrEnum):
    """Calibrated risk severity tier."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AssessmentOutcome(StrEnum):
    """Categorical outcome of risk assessment calculation."""

    SCORED = "SCORED"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"


class RiskBandDefinition(BaseModel):
    """Definition of a score interval for a risk band."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    band: RiskBand
    min_score: int = Field(ge=0, le=100)
    max_score: int = Field(ge=0, le=100)
    description: str = Field(default="")

    @model_validator(mode="after")
    def validate_interval(self) -> RiskBandDefinition:
        if self.min_score > self.max_score:
            raise ValueError(f"min_score {self.min_score} cannot exceed max_score {self.max_score}")
        return self


class MandatoryFinding(BaseModel):
    """Attributable finding generated for positive risk factors requiring human review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_id: str = Field(description="Unique attributable finding identifier")
    factor: RiskFactor = Field(description="Triggered risk factor")
    title: str = Field(description="Clear title of the mandatory finding")
    description: str = Field(description="Detailed explanation of the risk condition")
    evidence_field: str = Field(description="Canonical evidence field that triggered the finding")
    evidence_value: Any = Field(description="Observed canonical evidence value")
    weight: int = Field(ge=0, le=100, description="Risk weight points contributed")
    severity: RiskBand | None = Field(default=None, description="Severity rating of finding")
    evidence_refs: tuple[str, ...] = Field(
        default_factory=tuple, description="Supporting observation IDs from vehicle evidence"
    )
    policy_citation_refs: tuple[str, ...] = Field(
        default_factory=tuple, description="Attributable policy passage citation IDs"
    )


class RiskFactorResult(BaseModel):
    """Outcome of evaluating one risk factor against canonical evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factor: RiskFactor
    weight: int = Field(ge=0, le=100)
    triggered: bool
    evidence_field: str
    evidence_value: Any | None = None
    rationale: str = Field(default="")
    score_contribution: int = Field(default=0, ge=0)
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    policy_citation_refs: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def populate_score_contribution(self) -> RiskFactorResult:
        if self.triggered and self.score_contribution == 0 and self.weight > 0:
            object.__setattr__(self, "score_contribution", self.weight)
        elif not self.triggered and self.score_contribution != 0:
            object.__setattr__(self, "score_contribution", 0)
        return self


DEFAULT_POLICY_V1_FACTOR_WEIGHTS: dict[RiskFactor, int] = {
    RiskFactor.MATCH: 30,
    RiskFactor.LISTED: 45,
    RiskFactor.REPAIRABLE: 20,
    RiskFactor.STATUTORY: 40,
}

DEFAULT_POLICY_V1_BANDS: tuple[RiskBandDefinition, ...] = (
    RiskBandDefinition(
        band=RiskBand.LOW,
        min_score=0,
        max_score=19,
        description="Low vehicle risk; no adverse register entries",
    ),
    RiskBandDefinition(
        band=RiskBand.MEDIUM,
        min_score=20,
        max_score=39,
        description="Moderate risk; repairable write-off or minor factor detected",
    ),
    RiskBandDefinition(
        band=RiskBand.HIGH,
        min_score=40,
        max_score=69,
        description="High risk; registered security interest or statutory write-off detected",
    ),
    RiskBandDefinition(
        band=RiskBand.CRITICAL,
        min_score=70,
        max_score=100,
        description=(
            "Critical risk; active stolen listing or multiple compound severe adverse factors"
        ),
    ),
)

DEFAULT_REQUIRED_EVIDENCE_FIELDS: tuple[str, ...] = (
    "ppsr_result",
    "stolen_status",
    "writeoff_status",
)


def compute_policy_hash(
    policy_id: str,
    version: str,
    factor_weights: dict[RiskFactor, int],
    score_cap: int,
    risk_bands: tuple[RiskBandDefinition, ...],
    required_evidence_fields: tuple[str, ...],
) -> str:
    """Compute deterministic SHA-256 fingerprint for a policy's rules."""
    payload = {
        "id": policy_id,
        "version": version,
        "factor_weights": {
            str(k.value if isinstance(k, RiskFactor) else k): v
            for k, v in sorted(factor_weights.items(), key=lambda x: str(x[0]))
        },
        "score_cap": score_cap,
        "risk_bands": [
            {
                "band": str(b.band.value if isinstance(b.band, RiskBand) else b.band),
                "min_score": b.min_score,
                "max_score": b.max_score,
            }
            for b in sorted(risk_bands, key=lambda b: b.min_score)
        ],
        "required_evidence_fields": sorted(required_evidence_fields),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RiskPolicy(BaseModel):
    """Authoritative, immutable Risk Policy definition with versioned factor weights and bands."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=128)
    version: str = Field(default="v1", min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1)
    lifecycle_state: RiskPolicyLifecycleState = RiskPolicyLifecycleState.DRAFT
    factor_weights: dict[RiskFactor, int] = Field(
        default_factory=lambda: dict(DEFAULT_POLICY_V1_FACTOR_WEIGHTS)
    )
    score_cap: int = Field(default=100, ge=1, le=100)
    risk_bands: tuple[RiskBandDefinition, ...] = Field(
        default_factory=lambda: tuple(DEFAULT_POLICY_V1_BANDS)
    )
    mandatory_review_rules: tuple[str, ...] = Field(default=("EVERY_POSITIVE_FACTOR",))
    required_evidence_fields: tuple[str, ...] = Field(
        default_factory=lambda: tuple(DEFAULT_REQUIRED_EVIDENCE_FIELDS)
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    activated_at: datetime | None = None
    activated_by: str | None = None
    retired_at: datetime | None = None
    policy_hash: str = Field(default="")

    @field_validator("factor_weights")
    @classmethod
    def validate_factor_weights(cls, v: dict[RiskFactor, int]) -> dict[RiskFactor, int]:
        for factor, weight in v.items():
            if not (0 <= weight <= 100):
                raise ValueError(f"Weight for {factor} must be between 0 and 100")
        return v

    @model_validator(mode="after")
    def validate_policy_and_compute_hash(self) -> RiskPolicy:
        # Validate bands are contiguous and cover 0 to 100
        sorted_bands = sorted(self.risk_bands, key=lambda b: b.min_score)
        if not sorted_bands:
            raise ValueError("risk_bands cannot be empty")
        if sorted_bands[0].min_score != 0:
            raise ValueError("First risk band must start at min_score 0")
        if sorted_bands[-1].max_score != 100:
            raise ValueError("Last risk band must end at max_score 100")
        for i in range(len(sorted_bands) - 1):
            if sorted_bands[i].max_score + 1 != sorted_bands[i + 1].min_score:
                raise ValueError(
                    f"Risk bands have gap or overlap between {sorted_bands[i].max_score} "
                    f"and {sorted_bands[i + 1].min_score}"
                )

        computed = compute_policy_hash(
            policy_id=self.id,
            version=self.version,
            factor_weights=self.factor_weights,
            score_cap=self.score_cap,
            risk_bands=self.risk_bands,
            required_evidence_fields=self.required_evidence_fields,
        )
        if not self.policy_hash:
            object.__setattr__(self, "policy_hash", computed)
        elif self.policy_hash != computed:
            raise ValueError(
                f"Declared policy_hash {self.policy_hash} does not match computed {computed}"
            )
        return self


def build_risk_policy_v1(
    policy_id: str = "risk-policy-v1",
    version: str = "v1",
    factor_weights: dict[RiskFactor, int] | None = None,
    score_cap: int = 100,
    risk_bands: tuple[RiskBandDefinition, ...] | None = None,
) -> RiskPolicy:
    """Factory helper to construct canonical Risk Policy v1."""
    weights = dict(DEFAULT_POLICY_V1_FACTOR_WEIGHTS) if factor_weights is None else factor_weights
    bands = tuple(DEFAULT_POLICY_V1_BANDS) if risk_bands is None else risk_bands
    return RiskPolicy(
        id=policy_id,
        version=version,
        name="New Zealand Vehicle Risk Policy v1",
        description=(
            "Authoritative baseline policy for NZ vehicle risk scoring "
            "with PPSR, Stolen, and Writeoff factors."
        ),
        lifecycle_state=RiskPolicyLifecycleState.DRAFT,
        factor_weights=weights,
        score_cap=score_cap,
        risk_bands=bands,
        mandatory_review_rules=("EVERY_POSITIVE_FACTOR",),
        required_evidence_fields=tuple(DEFAULT_REQUIRED_EVIDENCE_FIELDS),
    )


def create_default_policy_v1(policy_id: str = "risk-policy-v1") -> RiskPolicy:
    """Alias for build_risk_policy_v1."""
    return build_risk_policy_v1(policy_id=policy_id)


class RiskResult(BaseModel):
    """Immutable calculated risk outcome for an assessment run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    assessment_id: str = Field(description="Associated assessment identifier")
    run_number: int = Field(ge=1, description="Assessment run sequence number")
    policy_id: str = Field(description="Policy ID used for calculation")
    policy_version: str = Field(description="Policy version evaluated")
    score: int | None = Field(
        default=None, ge=0, le=100, description="Calculated score (None if incomplete)"
    )
    band: RiskBand | None = Field(
        default=None, description="Assigned risk band (None if incomplete)"
    )
    raw_score: int | None = Field(default=None, description="Uncapped raw score summation")
    is_incomplete: bool = Field(
        default=False, description="True if calculation was withheld due to incomplete evidence"
    )
    outcome: AssessmentOutcome = Field(
        default=AssessmentOutcome.SCORED, description="SCORED or INCOMPLETE outcome"
    )
    factors: tuple[RiskFactorResult, ...] = Field(
        default_factory=tuple, description="Evaluation of all evaluated factors"
    )
    findings: tuple[MandatoryFinding, ...] = Field(
        default_factory=tuple, description="Mandatory-Review findings generated"
    )
    missing_evidence: tuple[MissingEvidenceFinding, ...] = Field(
        default_factory=tuple, description="Missing evidence findings if incomplete"
    )
    missing_findings: tuple[MissingEvidenceFinding, ...] = Field(
        default_factory=tuple, description="Alias for missing_evidence"
    )
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    calculation_hash: str = Field(default="", description="SHA-256 calculation fingerprint")

    @model_validator(mode="after")
    def validate_completeness_invariants(self) -> RiskResult:
        if self.is_incomplete:
            if self.score is not None or self.band is not None or self.raw_score is not None:
                raise ValueError(
                    "Incomplete risk results must not contain numeric scores, "
                    "raw_score, or risk bands"
                )
            if self.outcome != AssessmentOutcome.INCOMPLETE:
                object.__setattr__(self, "outcome", AssessmentOutcome.INCOMPLETE)
        else:
            if self.score is None or self.band is None:
                raise ValueError(
                    "Complete risk results must contain both numeric score and risk band"
                )
            if self.outcome != AssessmentOutcome.SCORED:
                object.__setattr__(self, "outcome", AssessmentOutcome.SCORED)

        # Synchronize missing_evidence and missing_findings
        if self.missing_evidence and not self.missing_findings:
            object.__setattr__(self, "missing_findings", self.missing_evidence)
        elif self.missing_findings and not self.missing_evidence:
            object.__setattr__(self, "missing_evidence", self.missing_findings)

        return self
