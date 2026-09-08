"""Versioned Evaluation Scenario, Run, Grader Result, and threshold contracts."""

# story: e06s01
# story: e06s02
# story: e06s03
# story: e06s04

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vehicle_risk_agent.api.models import AssessmentContext
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.evidence.models import (
    FieldExplanationResult,
    SafeError,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.sufficiency import SufficiencyOutcome
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.risk.models import AssessmentOutcome, RiskBand


class ScenarioCategory(StrEnum):
    """Categories for evaluation scenarios."""

    CLEAN = "CLEAN"
    RISKY = "RISKY"
    INCOMPLETE = "INCOMPLETE"
    CONFLICT = "CONFLICT"
    TEMPORAL = "TEMPORAL"
    POLICY_ABSTENTION = "POLICY_ABSTENTION"
    ADVERSARIAL_AUTH = "ADVERSARIAL_AUTH"
    ADVERSARIAL_IDEMPOTENCY = "ADVERSARIAL_IDEMPOTENCY"
    ADVERSARIAL_CONCURRENCY = "ADVERSARIAL_CONCURRENCY"
    ADVERSARIAL_TIMEOUT = "ADVERSARIAL_TIMEOUT"
    ADVERSARIAL_CONTRACT_DRIFT = "ADVERSARIAL_CONTRACT_DRIFT"
    ADVERSARIAL_PROMPT_INJECTION = "ADVERSARIAL_PROMPT_INJECTION"
    ADVERSARIAL_DATA_LEAKAGE = "ADVERSARIAL_DATA_LEAKAGE"


class EvaluationProvenance(BaseModel):
    """Immutable provenance metadata for evaluation runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_version: str
    corpus_version: str
    risk_policy_version: str
    provider: str
    prompt_version: str
    grader_version: str
    code_version: str

    def compute_hash(self) -> str:
        """Compute SHA-256 hex digest over canonical provenance elements."""
        payload = json.dumps(
            {
                "scenario_version": self.scenario_version,
                "corpus_version": self.corpus_version,
                "risk_policy_version": self.risk_policy_version,
                "provider": self.provider,
                "prompt_version": self.prompt_version,
                "grader_version": self.grader_version,
                "code_version": self.code_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_provenance_hash(prov: EvaluationProvenance) -> str:
    """Compute SHA-256 hex digest of EvaluationProvenance."""
    return prov.compute_hash()


class ExpectedEvaluationLabels(BaseModel):
    """Expected domain labels for an evaluation scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sufficiency_outcome: SufficiencyOutcome | None = None
    assessment_outcome: AssessmentOutcome | None = None
    risk_band: RiskBand | None = None
    min_risk_score: float | None = None
    max_risk_score: float | None = None
    required_factor_ids: tuple[str, ...] = ()
    expected_citations: tuple[str, ...] = ()
    should_abstain: bool | None = None
    expected_phase: AssessmentRunPhase | None = None


class ProhibitedEvaluationLabels(BaseModel):
    """Prohibited domain labels, phrases, and outcomes for an evaluation scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prohibited_phrases: tuple[str, ...] = ()
    prohibited_factor_ids: tuple[str, ...] = ()
    prohibited_outcomes: tuple[AssessmentOutcome, ...] = ()


class EvaluationScenario(BaseModel):
    """Versioned, immutable evaluation scenario specification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    version: str
    title: str
    description: str
    category: ScenarioCategory
    vin: str
    context: AssessmentContext
    mock_vehicle_revisions: tuple[VehicleRevisionResponse, ...] = ()
    mock_field_explanations: dict[str, FieldExplanationResult] = Field(default_factory=dict)
    mock_mcp_error: SafeError | None = None
    mock_citations: tuple[PolicyCitation, ...] = ()
    mock_draft_override: str | None = None
    expected_labels: ExpectedEvaluationLabels
    prohibited_labels: ProhibitedEvaluationLabels


class GraderResult(BaseModel):
    """Evaluation result from an individual domain grader."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grader_name: str
    passed: bool
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    details: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    prohibited_violations: tuple[str, ...] = ()


class EvaluationThresholds(BaseModel):
    """Quality gate threshold contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_recall: float = Field(default=0.90, ge=0.0, le=1.0)
    min_precision: float = Field(default=0.70, ge=0.0, le=1.0)
    min_mrr: float = Field(default=0.80, ge=0.0, le=1.0)
    min_abstention_accuracy: float = Field(default=0.95, ge=0.0, le=1.0)
    min_citation_grounding: float = Field(default=1.00, ge=0.0, le=1.0)
    max_p95_latency_ms: float = Field(default=30000.0, ge=0.0)


class EvaluationRun(BaseModel):
    """Terminal evaluation execution result with immutable provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    scenario_id: str
    scenario_version: str
    provenance: EvaluationProvenance
    timestamp: datetime
    grader_results: tuple[GraderResult, ...]
    passed: bool
    run_hash: str = ""

    @model_validator(mode="before")
    @classmethod
    def calculate_run_hash(cls, data: Any) -> Any:
        """Ensure run_hash is deterministically computed if missing."""
        if isinstance(data, dict) and not data.get("run_hash"):
            prov = data.get("provenance")
            prov_hash = prov.compute_hash() if isinstance(prov, EvaluationProvenance) else ""
            graders_summary = [
                f"{g.grader_name}:{g.passed}:{g.score}"
                for g in data.get("grader_results", ())
                if isinstance(g, GraderResult)
            ]
            graders_str = ",".join(graders_summary)
            hash_input = (
                f"{data.get('scenario_id')}:{data.get('scenario_version')}:"
                f"{prov_hash}:{data.get('passed')}:{graders_str}"
            )
            data["run_hash"] = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        return data
