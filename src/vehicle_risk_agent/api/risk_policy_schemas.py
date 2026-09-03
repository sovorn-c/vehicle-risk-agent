"""Pydantic schemas for Risk Policy REST API endpoints."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vehicle_risk_agent.risk.models import (
    RiskFactor,
    RiskPolicy,
)


class RiskBandDefinitionResponse(BaseModel):
    """Response view of a risk band score interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    band: str
    min_score: int
    max_score: int
    description: str


class RiskPolicyCreateRequest(BaseModel):
    """Request payload to register a new Risk Policy."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1)
    version: str = Field(default="v1", min_length=1, max_length=32)
    factor_weights: dict[RiskFactor, int] | None = None
    score_cap: int = Field(default=100, ge=1, le=100)
    risk_bands: list[dict[str, Any]] | None = None
    mandatory_review_rules: list[str] | None = None
    required_evidence_fields: list[str] | None = None


class RiskPolicyResponse(BaseModel):
    """Serialized representation of a Risk Policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    version: str
    name: str
    description: str
    lifecycle_state: str
    factor_weights: dict[str, int]
    score_cap: int
    risk_bands: list[RiskBandDefinitionResponse]
    mandatory_review_rules: list[str]
    required_evidence_fields: list[str]
    policy_hash: str
    created_at: datetime
    updated_at: datetime
    activated_at: datetime | None = None
    activated_by: str | None = None
    retired_at: datetime | None = None

    @classmethod
    def from_domain(cls, policy: RiskPolicy) -> "RiskPolicyResponse":
        return cls(
            id=policy.id,
            version=policy.version,
            name=policy.name,
            description=policy.description,
            lifecycle_state=policy.lifecycle_state.value,
            factor_weights={k.value: v for k, v in policy.factor_weights.items()},
            score_cap=policy.score_cap,
            risk_bands=[
                RiskBandDefinitionResponse(
                    band=b.band.value,
                    min_score=b.min_score,
                    max_score=b.max_score,
                    description=b.description,
                )
                for b in policy.risk_bands
            ],
            mandatory_review_rules=list(policy.mandatory_review_rules),
            required_evidence_fields=list(policy.required_evidence_fields),
            policy_hash=policy.policy_hash,
            created_at=policy.created_at,
            updated_at=policy.updated_at,
            activated_at=policy.activated_at,
            activated_by=policy.activated_by,
            retired_at=policy.retired_at,
        )


class RiskPolicyActivationResponse(BaseModel):
    """Response outcome of policy activation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    active_policy: RiskPolicyResponse
    retired_policy: RiskPolicyResponse | None = None
