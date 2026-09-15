"""Credential-gated live evaluation runner, cost tracking, and release verdict."""

# story: e06s04

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vehicle_risk_agent.evaluation.comparative import ComparativeReport
from vehicle_risk_agent.evaluation.provenance import resolve_source_commit


class LiveEvaluationConsentError(Exception):
    """Raised when live evaluation is attempted without explicit provider consent opt-in."""


class LiveEvaluationCredentialsError(Exception):
    """Raised when required provider API keys are missing or invalid."""


class LiveEvaluationBudgetError(Exception):
    """Raised when evaluation budget bounds are exceeded or invalid."""


class LiveEvaluationCorpusError(Exception):
    """Raised when active neural policy corpus is missing, invalid, or unready."""


_SENTINEL = object()
_E12_MAX_INPUT_TOKENS = 16_384
_PROVIDER_CREDENTIALS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class ModelPricingConfig(BaseModel):
    """Pricing configuration per million input/output tokens for cost estimation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "claude-sonnet-4-6"
    input_token_cost_per_million: float = 3.00
    output_token_cost_per_million: float = 15.00
    provenance: str = "anthropic-published-2026.1"
    source_url: str = "https://docs.anthropic.com/en/docs/about-claude/pricing"
    effective_at: str | None = None
    checked_at: str = "2026-09-11"


class LiveScenarioMetrics(BaseModel):
    """Per-scenario live evaluation execution metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    provider: str = "anthropic"
    draft_latency_seconds: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    quality_passed: bool
    deterministic_risk_passed: bool = False
    retrieval_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_mrr: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_applicable: bool = False
    claim_support: float = Field(default=0.0, ge=0.0, le=1.0)
    claim_support_applicable: bool = False
    citation_grounding: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_grounding_applicable: bool = False
    abstention_correct: bool = False
    abstention_applicable: bool = False
    useful_tool_selection: float = Field(default=0.0, ge=0.0, le=1.0)
    useful_tool_selection_applicable: bool = False
    deterministic_risk_applicable: bool = False
    retrieved_passage_ids: tuple[str, ...] = ()
    retrieved_citation_ids: tuple[str, ...] = ()
    missed_findings: int = Field(default=0, ge=0)
    false_positive_citations: int = Field(default=0, ge=0)
    unauthorized_dispatches: int = Field(default=0, ge=0)
    automatic_approval: bool = False
    outcome: str = "SCORED"
    failure_reason: str | None = None
    expected_action: str | None = None
    observed_action: str | None = None
    dispatched: bool | None = None
    result_references: tuple[str, ...] = ()
    citation_references: tuple[str, ...] = ()
    synthetic_notice: str | None = None
    suite_version: str = "general-v1"
    scenario_version: str = "general-v1"
    proposal_valid: bool | None = None
    provider_marker: bool = False
    mcp_marker: bool = False
    proposal_rounds: int = 0
    supplementary_attempts: int = 0
    supplementary_retries: int = 0
    termination_reason: str | None = None
    report_draft_id: str | None = None
    report_status: str | None = None
    assessment_state: str | None = None


class LiveEvaluationRecord(BaseModel):
    """Immutable, machine-readable record of a live evaluation run and release verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    created_at: str
    model_version: str
    provider: str = "anthropic"
    prompt_version: str
    corpus_version: str
    index_version: str = "pgvector-hnsw-v1"
    policy_version: str
    grader_version: str
    code_version: str
    suite_version: str = "general-v1"
    pricing_provenance: str = "anthropic-published-2026.1"
    input_price_per_million: float = 3.0
    output_price_per_million: float = 15.0
    pricing_source_url: str = "https://docs.anthropic.com/en/docs/about-claude/pricing"
    pricing_effective_at: str | None = None
    pricing_checked_at: str = "2026-09-11"
    execution_mode: str = "LIVE"
    total_scenarios: int
    passed_scenarios: int
    p95_draft_latency_seconds: float
    p50_draft_latency_seconds: float
    max_draft_latency_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float
    p95_latency_passed: bool
    quality_passed: bool
    release_verdict: str
    verdict_passed: bool
    scenarios: tuple[LiveScenarioMetrics, ...]
    run_hash: str

    def save_to_file(self, path: str | os.PathLike[str]) -> None:
        """Save the immutable redacted record as formatted JSON."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.model_dump_json(indent=2)
        target_path.write_text(content, encoding="utf-8")

    @classmethod
    def load_from_file(cls, path: str | os.PathLike[str]) -> LiveEvaluationRecord:
        """Load and validate an immutable evidence record from file."""
        target_path = Path(path)
        content = target_path.read_text(encoding="utf-8")
        return cls.model_validate_json(content)

    @classmethod
    def from_scenario_metrics(
        cls,
        scenario_metrics: list[LiveScenarioMetrics],
        model_version: str = "claude-sonnet-4-6",
        prompt_version: str = "prompt-2026.1",
        corpus_version: str = "corpus-2026.1",
        index_version: str = "pgvector-hnsw-v1",
        policy_version: str = "nz-vehicle-risk-v1",
        grader_version: str = "grader-2026.1",
        code_version: str | None = None,
        pricing_provenance: str = "anthropic-published-2026.1",
        execution_mode: str = "LIVE",
        suite_version: str = "general-v1",
        input_price_per_million: float = 3.0,
        output_price_per_million: float = 15.0,
        pricing_source_url: str = "https://docs.anthropic.com/en/docs/about-claude/pricing",
        pricing_effective_at: str | None = None,
        pricing_checked_at: str = "2026-09-11",
        p95_latency_threshold: float = 30.0,
        offline_control: bool = False,
        provider: str = "anthropic",
    ) -> LiveEvaluationRecord:
        """Aggregate per-scenario metrics into an immutable record with release verdict."""
        resolved_code_version = code_version or resolve_source_commit()
        latencies = [m.draft_latency_seconds for m in scenario_metrics]
        p95_latency = compute_p95_latency(latencies)
        p50_latency = compute_p50_latency(latencies)
        max_latency = max(latencies) if latencies else 0.0

        total_input = sum(m.input_tokens for m in scenario_metrics)
        total_output = sum(m.output_tokens for m in scenario_metrics)
        total_cost = round(sum(m.estimated_cost_usd for m in scenario_metrics), 6)

        total_scenarios = len(scenario_metrics)
        passed_scenarios = sum(1 for m in scenario_metrics if m.quality_passed)

        p95_passed = p95_latency <= p95_latency_threshold
        quality_passed = passed_scenarios == total_scenarios if total_scenarios > 0 else True
        verdict_passed = p95_passed and quality_passed
        release_verdict = "BLOCKED" if offline_control else ("PASS" if verdict_passed else "FAIL")
        if offline_control:
            verdict_passed = False

        now = datetime.datetime.now(datetime.UTC).isoformat()
        record_id = f"live-eval-{now[:10]}-{abs(hash(now)) % 100000:05d}"

        hash_payload = {
            "record_id": record_id,
            "model_version": model_version,
            "provider": provider,
            "prompt_version": prompt_version,
            "corpus_version": corpus_version,
            "index_version": index_version,
            "policy_version": policy_version,
            "grader_version": grader_version,
            "code_version": resolved_code_version,
            "suite_version": suite_version,
            "pricing_provenance": pricing_provenance,
            "input_price_per_million": input_price_per_million,
            "output_price_per_million": output_price_per_million,
            "pricing_source_url": pricing_source_url,
            "pricing_effective_at": pricing_effective_at,
            "pricing_checked_at": pricing_checked_at,
            "execution_mode": execution_mode,
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "p95_draft_latency_seconds": p95_latency,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_estimated_cost_usd": total_cost,
            "release_verdict": release_verdict,
            "offline_control": offline_control,
            "scenarios": [metric.model_dump(mode="json") for metric in scenario_metrics],
        }
        run_hash = hashlib.sha256(
            json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        return cls(
            record_id=record_id,
            created_at=now,
            model_version=model_version,
            provider=provider,
            prompt_version=prompt_version,
            corpus_version=corpus_version,
            index_version=index_version,
            policy_version=policy_version,
            grader_version=grader_version,
            code_version=resolved_code_version,
            suite_version=suite_version,
            pricing_provenance=pricing_provenance,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            pricing_source_url=pricing_source_url,
            pricing_effective_at=pricing_effective_at,
            pricing_checked_at=pricing_checked_at,
            execution_mode=execution_mode,
            total_scenarios=total_scenarios,
            passed_scenarios=passed_scenarios,
            p95_draft_latency_seconds=p95_latency,
            p50_draft_latency_seconds=p50_latency,
            max_draft_latency_seconds=round(max_latency, 3),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_estimated_cost_usd=total_cost,
            p95_latency_passed=p95_passed,
            quality_passed=quality_passed,
            release_verdict=release_verdict,
            verdict_passed=verdict_passed,
            scenarios=tuple(scenario_metrics),
            run_hash=run_hash,
        )


class LiveEvaluationConfig(BaseModel):
    """Configuration for credential-gated live evaluation runs."""

    model_config = ConfigDict(extra="forbid")

    enable_live_eval: bool = False
    api_key: str = Field(default="", exclude=True)
    provider: Literal["anthropic", "gemini"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    p95_latency_threshold: float = 30.0
    timeout_seconds: float = 30.0
    max_output_tokens: int = 2048
    max_repairs: int = 1
    max_budget_usd: float = 5.0
    max_scenarios: int | None = None
    output_file: str | None = None
    require_neural_corpus: bool = True
    suite: str = "general"
    mcp_server_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_e12_provider(cls, values: Any) -> Any:
        if isinstance(values, dict) and values.get("suite") == "e12-comparative":
            values = dict(values)
            provider = values.setdefault("provider", "gemini")
            if provider == "gemini":
                values.setdefault("model", "gemini-3.1-flash-lite")
        return values

    def __repr__(self) -> str:
        return (
            f"LiveEvaluationConfig(enable_live_eval={self.enable_live_eval}, "
            f"model={self.model!r}, p95_latency_threshold={self.p95_latency_threshold})"
        )

    def __str__(self) -> str:
        return self.__repr__()


def calculate_estimated_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: ModelPricingConfig | None = None,
) -> float:
    """Compute dollar cost from input and output tokens against pricing metadata."""
    cfg = pricing or ModelPricingConfig()
    input_cost = (input_tokens * cfg.input_token_cost_per_million) / 1_000_000.0
    output_cost = (output_tokens * cfg.output_token_cost_per_million) / 1_000_000.0
    return round(input_cost + output_cost, 6)


def _e12_worst_case_call_cost(
    pricing: ModelPricingConfig,
    max_output_tokens: int,
) -> float:
    """Reserve a bounded input/output cost before an E12 provider call."""
    return calculate_estimated_cost(
        input_tokens=_E12_MAX_INPUT_TOKENS,
        output_tokens=max_output_tokens,
        pricing=pricing,
    )


def compute_p95_latency(latencies: list[float]) -> float:
    """Compute the 95th percentile latency from observed durations."""
    if not latencies:
        return 0.0
    sorted_l = sorted(latencies)
    idx = min(len(sorted_l) - 1, int(math.ceil(0.95 * len(sorted_l))))
    return round(sorted_l[idx], 3)


def compute_p50_latency(latencies: list[float]) -> float:
    """Compute the median (50th percentile) latency from observed durations."""
    if not latencies:
        return 0.0
    sorted_l = sorted(latencies)
    idx = int(math.ceil(0.50 * len(sorted_l))) - 1
    return round(sorted_l[max(0, idx)], 3)


def _default_provider_pricing(provider: str, model: str) -> ModelPricingConfig:
    if provider == "gemini":
        return ModelPricingConfig(
            model=model,
            input_token_cost_per_million=0.25,
            output_token_cost_per_million=1.50,
            provenance="google-gemini-paid-2026.2",
            source_url="https://ai.google.dev/gemini-api/docs/pricing",
            checked_at="2026-09-12",
        )
    return ModelPricingConfig(model=model)


def _e12_anthropic_pricing(model: str, checked_at: str) -> ModelPricingConfig:
    """Return the frozen Anthropic alternate pricing used only by E12."""
    default = _default_provider_pricing("anthropic", model)
    return default.model_copy(
        update={
            "input_token_cost_per_million": 3.00,
            "output_token_cost_per_million": 15.00,
            "source_url": "https://platform.claude.com/docs/en/about-claude/pricing",
            "checked_at": checked_at,
        }
    )


def _build_default_offline_mcp_adapter() -> Any:
    from datetime import UTC, datetime

    from vehicle_risk_agent.adapters.mcp import FakeVehicleMcpAdapter
    from vehicle_risk_agent.evidence.models import (
        ConfidenceAssessment,
        ConfidenceBand,
        ProvenanceLink,
        VehicleRevisionResponse,
    )

    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    clean_rev = VehicleRevisionResponse(
        vin="1HGCR2F85HA000000",
        revision_id="rev-clean-01",
        revision_number=1,
        material_hash="e" * 64,
        canonical_fields={
            "vin": "1HGCR2F85HA000000",
            "make": "HONDA",
            "model": "ACCORD",
            "year": 2017,
            "plate": "NZACC1",
            "ppsr_result": "NO_FINANCE_REGISTERED",
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        },
        field_provenance={
            "vin": [
                ProvenanceLink(
                    observation_id="obs-vin",
                    source_system="NZTA",
                    source_record_id="rec-vin",
                    retrieved_at=now,
                )
            ],
            "ppsr_result": [
                ProvenanceLink(
                    observation_id="obs-ppsr",
                    source_system="PPSR",
                    source_record_id="rec-ppsr",
                    retrieved_at=now,
                )
            ],
            "stolen_status": [
                ProvenanceLink(
                    observation_id="obs-police",
                    source_system="POLICE",
                    source_record_id="rec-police",
                    retrieved_at=now,
                )
            ],
            "writeoff_status": [
                ProvenanceLink(
                    observation_id="obs-writeoff",
                    source_system="NZTA",
                    source_record_id="rec-writeoff",
                    retrieved_at=now,
                )
            ],
        },
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=95,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="Clean verified record",
        ),
        as_of=now,
        published_at=now,
    )
    inc_rev = VehicleRevisionResponse(
        vin="JM0BL10F000000000",
        revision_id="rev-inc-01",
        revision_number=1,
        material_hash="f" * 64,
        canonical_fields={
            "vin": "JM0BL10F000000000",
            "make": "MAZDA",
            "model": "AXELA",
            "year": 2016,
            "plate": "AXL100",
        },
        field_provenance={
            "vin": [
                ProvenanceLink(
                    observation_id="obs-vin",
                    source_system="NZTA",
                    source_record_id="rec-vin",
                    retrieved_at=now,
                )
            ],
        },
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=50,
            band=ConfidenceBand.MEDIUM,
            rule_version="v1",
            explanation="Partial unverified record",
        ),
        as_of=now,
        published_at=now,
    )
    adapter = FakeVehicleMcpAdapter()
    adapter.seed_vehicle(clean_rev)
    adapter.seed_vehicle(inc_rev)
    return adapter


class LiveEvaluationRunner:
    """Coordinates credential validation, scenario execution, and release verdict."""

    def __init__(
        self,
        config: LiveEvaluationConfig | None = None,
        pricing: ModelPricingConfig | None = None,
        active_corpus: Any = None,
    ) -> None:
        self.config = config or LiveEvaluationConfig()
        if pricing is None and self.config.suite == "e12-comparative":
            from vehicle_risk_agent.evaluation.comparative import load_e12_evaluation_config

            e12_config = load_e12_evaluation_config()
            if self.config.provider == e12_config.provider:
                pricing = ModelPricingConfig(
                    model=e12_config.model,
                    input_token_cost_per_million=e12_config.pricing_input_usd_per_million,
                    output_token_cost_per_million=e12_config.pricing_output_usd_per_million,
                    provenance=_default_provider_pricing(
                        e12_config.provider, e12_config.model
                    ).provenance,
                    source_url=e12_config.pricing_source_url,
                    checked_at=e12_config.pricing_checked_at,
                )
            elif self.config.provider == "anthropic":
                pricing = _e12_anthropic_pricing(self.config.model, e12_config.pricing_checked_at)
            else:
                pricing = _default_provider_pricing(self.config.provider, self.config.model)
        self.pricing = pricing or _default_provider_pricing(self.config.provider, self.config.model)
        self.active_corpus = active_corpus
        self._allow_custom_investigation = False
        self._comparative_metrics = False

    def validate_readiness(self, active_corpus: Any = _SENTINEL) -> None:
        """Verify explicit consent, environment credentials, suite caps, and corpus readiness."""
        investigation_suite = self.config.suite in {"e11-investigation", "e12-comparative"}
        if (
            self.config.provider == "gemini"
            and self.config.suite
            not in {
                "e11-investigation",
                "e12-comparative",
            }
            and not self._comparative_metrics
        ):
            raise LiveEvaluationBudgetError(
                "Live evaluation refused: Gemini provider is supported only for e11-investigation "
                "and e12-comparative."
            )
        if investigation_suite and self.config.api_key:
            credential_name = _PROVIDER_CREDENTIALS[self.config.provider]
            raise LiveEvaluationCredentialsError(
                f"Live evaluation refused: {self.config.suite} accepts {credential_name} "
                "from the environment only."
            )
        if not self.config.enable_live_eval:
            raise LiveEvaluationConsentError(
                "Live evaluation refused: explicit provider consent and opt-in flag "
                "--enable-live-eval required for paid API execution."
            )
        credential_name = _PROVIDER_CREDENTIALS[self.config.provider]
        key = (
            os.environ.get(credential_name, "")
            if investigation_suite
            else self.config.api_key or os.environ.get(credential_name, "")
        )
        if not key or not key.strip():
            raise LiveEvaluationCredentialsError(
                f"Live evaluation refused: missing required {credential_name}."
            )
        if investigation_suite and not (
            self.config.mcp_server_url or os.environ.get("MCP_SERVER_URL")
        ):
            raise LiveEvaluationCredentialsError(
                f"Live evaluation refused: MCP_SERVER_URL is required for {self.config.suite}."
            )
        expected_model = (
            "gemini-3.1-flash-lite" if self.config.provider == "gemini" else "claude-sonnet-4-6"
        )
        expected_pricing = _default_provider_pricing(self.config.provider, expected_model)
        pricing_matches = (
            self.config.model == expected_model
            and self.pricing.model == expected_pricing.model
            and self.pricing.input_token_cost_per_million
            == expected_pricing.input_token_cost_per_million
            and self.pricing.output_token_cost_per_million
            == expected_pricing.output_token_cost_per_million
            and self.pricing.provenance == expected_pricing.provenance
            and self.pricing.source_url == expected_pricing.source_url
            and self.pricing.effective_at == expected_pricing.effective_at
            and self.pricing.checked_at == expected_pricing.checked_at
        )
        if self.config.suite == "e11-investigation" and not pricing_matches:
            raise LiveEvaluationBudgetError(
                f"Live evaluation refused: {self.config.suite} pricing is pinned to "
                f"{expected_model}."
            )
        if self.config.suite == "e12-comparative":
            from vehicle_risk_agent.evaluation.comparative import load_e12_evaluation_config

            e12_config = load_e12_evaluation_config()
            if self.config.provider == e12_config.provider:
                expected_model = e12_config.model
                expected_pricing = ModelPricingConfig(
                    model=e12_config.model,
                    input_token_cost_per_million=e12_config.pricing_input_usd_per_million,
                    output_token_cost_per_million=e12_config.pricing_output_usd_per_million,
                    provenance=self.pricing.provenance,
                    source_url=e12_config.pricing_source_url,
                    checked_at=e12_config.pricing_checked_at,
                )
            else:
                expected_pricing = _e12_anthropic_pricing(
                    expected_model, e12_config.pricing_checked_at
                )
            pricing_matches = (
                self.config.model == expected_model
                and self.pricing.model == expected_pricing.model
                and self.pricing.input_token_cost_per_million
                == expected_pricing.input_token_cost_per_million
                and self.pricing.output_token_cost_per_million
                == expected_pricing.output_token_cost_per_million
                and self.pricing.source_url == expected_pricing.source_url
                and self.pricing.checked_at == expected_pricing.checked_at
            )
            if not pricing_matches:
                raise LiveEvaluationBudgetError(
                    "Live evaluation refused: e12 pricing/model provenance does not match "
                    "the selected provider contract."
                )
        budget_cap = (
            15.0
            if self.config.suite == "e12-comparative"
            else (3.0 if self.config.suite == "e11-investigation" else 5.0)
        )
        if self.config.max_budget_usd <= 0.0 or self.config.max_budget_usd > budget_cap:
            raise LiveEvaluationBudgetError(
                f"Live evaluation refused: max budget ${self.config.max_budget_usd:.2f} "
                f"must be positive and at most ${budget_cap:.2f}."
            )
        scenario_cap = 4 if self.config.suite == "e12-comparative" else 3
        if self.config.max_scenarios is not None:
            if self.config.suite == "e12-comparative" and self.config.max_scenarios != scenario_cap:
                raise LiveEvaluationBudgetError(
                    "Live evaluation refused: e12-comparative requires exactly "
                    f"{scenario_cap} comparable scenarios."
                )
            if self.config.max_scenarios <= 0 or self.config.max_scenarios > scenario_cap:
                raise LiveEvaluationBudgetError(
                    f"Live evaluation refused: max scenarios {self.config.max_scenarios} "
                    f"must be between 1 and {scenario_cap}."
                )

        corpus_to_check = active_corpus if active_corpus is not _SENTINEL else self.active_corpus
        if corpus_to_check is None:
            raise LiveEvaluationCorpusError(
                "Live evaluation refused: active neural policy corpus is required."
            )
        else:
            state = getattr(corpus_to_check, "lifecycle_state", None)
            state_str = state.value if state is not None and hasattr(state, "value") else str(state)
            if state_str not in ("ACTIVE", "READY"):
                raise LiveEvaluationCorpusError(
                    f"Live evaluation refused: active corpus lifecycle state '{state_str}' "
                    "must be ACTIVE or READY."
                )
            retrieval_cfg = getattr(corpus_to_check, "retrieval_config", None)
            profile = getattr(retrieval_cfg, "profile", None) if retrieval_cfg else None
            if profile != "neural":
                raise LiveEvaluationCorpusError(
                    f"Live evaluation refused: active corpus retrieval profile '{profile}' "
                    "must be 'neural'."
                )

    def validate_scenarios(self, scenarios: list[Any]) -> None:
        """Verify scenario batch size conforms to the active suite's bounded limits."""
        cap = 4 if self.config.suite == "e12-comparative" else 3
        if len(scenarios) > cap:
            raise LiveEvaluationBudgetError(
                f"Live evaluation refused: scenario count {len(scenarios)} exceeds limit of {cap}."
            )
        if self.config.max_scenarios is not None and len(scenarios) > self.config.max_scenarios:
            raise LiveEvaluationBudgetError(
                f"Live evaluation refused: scenario count {len(scenarios)} "
                f"exceeds configured max of {self.config.max_scenarios}."
            )

    async def run_bounded_acceptance(
        self,
        session_factory: Any | None = None,
        mcp_adapter: Any | None = None,
        drafting_adapter: Any | None = None,
        settings: Any | None = None,
        scenarios: list[dict[str, Any]] | None = None,
    ) -> LiveEvaluationRecord | ComparativeReport:
        """Execute bounded acceptance scenarios through the normal workflow and emit evidence."""
        if self.config.suite == "e12-comparative":
            if not self.config.enable_live_eval:
                from vehicle_risk_agent.evaluation.comparative import (
                    build_offline_comparative_report,
                    load_e12_evaluation_config,
                )

                report = await build_offline_comparative_report(load_e12_evaluation_config())
                if self.config.output_file:
                    report.save_to_file(self.config.output_file)
                return report
            return await self._run_e12_live_comparative(
                session_factory=session_factory,
                mcp_adapter=mcp_adapter,
                drafting_adapter=drafting_adapter,
                settings=settings,
            )
        import time
        from uuid import uuid4

        from pydantic import SecretStr
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from vehicle_risk_agent.adapters.anthropic_drafting import AnthropicDraftingAdapter
        from vehicle_risk_agent.adapters.gemini_drafting import GeminiDraftingAdapter
        from vehicle_risk_agent.adapters.investigation import (
            AnthropicInvestigationAdapter,
            GeminiInvestigationAdapter,
        )
        from vehicle_risk_agent.adapters.mcp import StreamableHttpVehicleMcpAdapter
        from vehicle_risk_agent.api.models import (
            AssessmentContext,
            AssessmentCreateRequest,
            SaleType,
        )
        from vehicle_risk_agent.config import Settings
        from vehicle_risk_agent.domain.assessment import (
            AssessmentLifecycleState,
            AssessmentRunPhase,
        )
        from vehicle_risk_agent.evaluation.investigation import (
            e11_scenarios,
            grade_scenario,
        )
        from vehicle_risk_agent.investigation.repository import InvestigationLedgerRepository
        from vehicle_risk_agent.persistence.repository import AssessmentRepository
        from vehicle_risk_agent.reporting.models import ReportDraftStatus
        from vehicle_risk_agent.reporting.offline import OfflineReportDraftingAdapter
        from vehicle_risk_agent.reporting.repository import ReportDraftRepository
        from vehicle_risk_agent.workflow.runner import AssessmentWorkflowRunner
        from vehicle_risk_agent.workflow.state import AssessmentGraphState

        if (
            self.config.enable_live_eval
            and self.active_corpus is None
            and session_factory is not None
        ):
            try:
                from vehicle_risk_agent.policy.corpus_lifecycle import CorpusLifecycleManager

                async with session_factory() as session:
                    db_corpus = await CorpusLifecycleManager(session).get_active_corpus()
                    if db_corpus is not None:
                        self.active_corpus = db_corpus
            except Exception:
                pass
        scenarios_to_run = scenarios
        if scenarios_to_run is None and self.config.suite == "e11-investigation":
            scenarios_to_run = [
                {
                    **scenario.model_dump(),
                    "vin": "1HGCR2F85HA000000",
                    "sale_type": SaleType.DEALER,
                }
                for scenario in e11_scenarios()
            ]
        if scenarios_to_run is None:
            scenarios_to_run = [
                {
                    "scenario_id": "scn-01-scored",
                    "vin": "1HGCR2F85HA000000",
                    "sale_type": SaleType.DEALER,
                    "expected_outcome": "SCORED",
                },
                {
                    "scenario_id": "scn-02-incomplete",
                    "vin": "JM0BL10F000000000",
                    "sale_type": SaleType.PRIVATE,
                    "expected_outcome": "WITHHELD",
                },
                {
                    "scenario_id": "scn-03-abstention",
                    "vin": "1HGCR2F85HA000000",
                    "sale_type": SaleType.AUCTION,
                    "questions": ["What is the salvage liability for maritime cargo containers?"],
                    "expected_outcome": "SCORED",
                },
            ]
            if self.config.max_scenarios is not None:
                scenarios_to_run = scenarios_to_run[: self.config.max_scenarios]

        self.validate_scenarios(scenarios_to_run)
        if self.config.suite == "e11-investigation" and not self._allow_custom_investigation:
            expected_ids = tuple(scenario.scenario_id for scenario in e11_scenarios())
            actual_ids = tuple(scenario.get("scenario_id") for scenario in scenarios_to_run)
            if actual_ids != expected_ids:
                raise LiveEvaluationBudgetError(
                    "Live evaluation refused: e11-investigation requires its fixed "
                    "labelled scenarios."
                )

        if self.config.suite == "e11-investigation" and not self.config.enable_live_eval:
            metrics = [
                LiveScenarioMetrics(
                    scenario_id=scenario["scenario_id"],
                    provider=self.config.provider,
                    draft_latency_seconds=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    estimated_cost_usd=0.0,
                    quality_passed=False,
                    outcome="BLOCKED",
                    failure_reason="OFFLINE_CONTROL_NOT_LIVE_EVIDENCE",
                    expected_action=scenario.get("expected_action"),
                    synthetic_notice=scenario.get("synthetic_notice"),
                    suite_version="e11-live-v1",
                    scenario_version="e11-live-v1",
                    termination_reason="OFFLINE_CONTROL_NOT_LIVE_EVIDENCE",
                )
                for scenario in scenarios_to_run
            ]
            record = LiveEvaluationRecord.from_scenario_metrics(
                metrics,
                model_version="offline-deterministic",
                provider=self.config.provider,
                prompt_version="offline-v1",
                corpus_version="corpus-offline",
                policy_version="policy-offline",
                grader_version="grader-e11-v1",
                code_version=resolve_source_commit(),
                pricing_provenance=self.pricing.provenance,
                execution_mode="OFFLINE",
                p95_latency_threshold=self.config.p95_latency_threshold,
                offline_control=True,
                suite_version="e11-live-v1",
            )
            if self.config.output_file:
                record.save_to_file(self.config.output_file)
            return record

        live_drafting = self.config.enable_live_eval and self.config.suite != "e11-investigation"
        settings_live_drafting = live_drafting and self.config.provider == "anthropic"
        effective_settings: Settings = settings or Settings(
            retrieval_mode="live" if self.config.enable_live_eval else "offline",
            drafting_mode="live" if settings_live_drafting else "offline",
            enable_live_drafting=settings_live_drafting,
            anthropic_api_key=(
                SecretStr(self.config.api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
                if settings_live_drafting and self.config.provider == "anthropic"
                else None
            ),
            mcp_server_url=self.config.mcp_server_url or os.environ.get("MCP_SERVER_URL"),
        )

        if session_factory is None:
            engine = create_async_engine(effective_settings.database_url, echo=False)
            async with engine.begin() as conn:
                from vehicle_risk_agent.persistence.models import Base

                await conn.run_sync(Base.metadata.create_all)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)

        if (
            self.config.enable_live_eval
            and self.active_corpus is None
            and session_factory is not None
        ):
            try:
                from vehicle_risk_agent.policy.corpus_lifecycle import CorpusLifecycleManager

                async with session_factory() as session:
                    self.active_corpus = await CorpusLifecycleManager(session).get_active_corpus()
            except Exception:
                pass
        if self.config.enable_live_eval:
            self.validate_readiness()

        effective_mcp = mcp_adapter
        if effective_mcp is None:
            if self.config.enable_live_eval:
                effective_mcp = StreamableHttpVehicleMcpAdapter(
                    server_url=effective_settings.mcp_server_url or "http://localhost:8000/mcp",
                    timeout_seconds=int(self.config.timeout_seconds),
                )
            else:
                effective_mcp = _build_default_offline_mcp_adapter()

        investigation_provider = None
        if self.config.suite == "e11-investigation" and self.config.enable_live_eval:
            provider_type = (
                GeminiInvestigationAdapter
                if self.config.provider == "gemini"
                else AnthropicInvestigationAdapter
            )
            credential_name = _PROVIDER_CREDENTIALS[self.config.provider]
            investigation_provider = provider_type(
                model=self.config.model,
                timeout_seconds=self.config.timeout_seconds,
                max_input_tokens=3072,
                max_output_tokens=512,
                api_key=os.environ.get(credential_name, ""),
            )
            readiness = await investigation_provider.readiness()
            if not readiness.ready:
                raise LiveEvaluationCredentialsError(
                    "Live evaluation refused: investigation provider readiness failed."
                )

        effective_drafter = drafting_adapter
        if effective_drafter is None:
            if self.config.enable_live_eval and self.config.suite != "e11-investigation":
                credential_name = _PROVIDER_CREDENTIALS[self.config.provider]
                key = self.config.api_key or os.environ.get(credential_name, "")
                drafter_type = (
                    GeminiDraftingAdapter
                    if self.config.provider == "gemini"
                    else AnthropicDraftingAdapter
                )
                effective_drafter = drafter_type(
                    model=self.config.model,
                    max_tokens=self.config.max_output_tokens,
                    timeout_seconds=int(self.config.timeout_seconds),
                    api_key=key,
                )
            else:
                effective_drafter = OfflineReportDraftingAdapter()

        scenario_metrics: list[LiveScenarioMetrics] = []
        cumulative_cost = 0.0

        for scn in scenarios_to_run:
            scenario_id = scn["scenario_id"]
            vin = scn["vin"]
            sale_type = scn.get("sale_type", SaleType.DEALER)
            questions = scn.get("questions", [])
            expected_outcome = scn.get("expected_outcome", "SCORED")

            start_t = time.monotonic()
            outcome = expected_outcome
            failure_reason: str | None = None
            quality_passed = False
            deterministic_risk_passed = False
            retrieval_relevance = 0.0
            retrieval_precision = 0.0
            retrieval_mrr = 0.0
            retrieval_applicable = False
            claim_support = 0.0
            claim_support_applicable = False
            citation_grounding = 0.0
            citation_grounding_applicable = False
            abstention_correct = False
            abstention_applicable = False
            useful_tool_selection = 0.0
            useful_tool_selection_applicable = False
            deterministic_risk_applicable = False
            retrieved_passage_ids: tuple[str, ...] = ()
            retrieved_citation_ids: tuple[str, ...] = ()
            missed_findings = 0
            false_positive_citations = 0
            unauthorized_dispatches = 0
            automatic_approval = False
            input_tokens = 0
            output_tokens = 0
            expected_action = scn.get("expected_action")
            observed_action: str | None = None
            dispatched: bool | None = None
            result_references: tuple[str, ...] = ()
            citation_references: tuple[str, ...] = ()
            synthetic_notice = scn.get("synthetic_notice")
            proposal_valid: bool | None = None
            provider_marker = False
            mcp_marker = False
            proposal_rounds = 0
            supplementary_attempts = 0
            supplementary_retries = 0
            termination_reason: str | None = None
            report_draft_id: str | None = None
            report_status: str | None = None
            assessment_state: str | None = None
            investigation_ledger = None

            try:
                async with session_factory() as session:
                    asmt_repo = AssessmentRepository(session)
                    asmt = await asmt_repo.create_assessment(
                        requester_id="eval-operator",
                        idempotency_key=f"eval-{scenario_id}-{uuid4()}",
                        request=AssessmentCreateRequest(
                            vin=vin,
                            context=AssessmentContext(
                                sale_type=sale_type,
                                questions=questions,
                            ),
                        ),
                    )

                async with AssessmentWorkflowRunner.create(
                    settings=effective_settings,
                    session_factory=session_factory,
                    mcp_adapter=effective_mcp,
                    drafting_adapter=effective_drafter,
                    investigation_provider=investigation_provider,
                ) as wf_runner:
                    initial_state: AssessmentGraphState = {
                        "assessment_id": asmt.id,
                        "run_number": 1,
                        "vin": vin,
                        "context": asmt.context,
                        "phase": AssessmentRunPhase.PENDING,
                        "visited_phases": [AssessmentRunPhase.PENDING],
                        "events": [],
                    }
                    final_state = await wf_runner.run(
                        initial_state=initial_state,
                        config={
                            "configurable": {
                                "investigation_targets": tuple(scn.get("evidence_targets", ()))
                            }
                        },
                    )

                async with session_factory() as session:
                    asmt_repo = AssessmentRepository(session)
                    draft_repo = ReportDraftRepository(session)
                    persisted_asmt = await asmt_repo.get_assessment(asmt.id)
                    persisted_draft = await draft_repo.get_draft(asmt.id, run_number=1)
                    investigation_ledger = await InvestigationLedgerRepository(session).get_ledger(
                        asmt.id, 1
                    )

                is_awaiting_review = (
                    persisted_asmt is not None
                    and persisted_asmt.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW
                )
                if persisted_asmt is not None:
                    assessment_state = persisted_asmt.lifecycle_state.value
                if persisted_draft is not None:
                    report_draft_id = persisted_draft.id
                    report_status = persisted_draft.status.value

                workflow_citations = tuple(final_state.get("policy_citations") or ())
                retrieved_passage_ids = tuple(
                    citation.passage_id for citation in workflow_citations
                )
                retrieved_citation_ids = retrieved_passage_ids
                citation_references = retrieved_citation_ids
                investigation_result = final_state.get("investigation_result")
                if self.config.suite == "e11-investigation":
                    from vehicle_risk_agent.evaluation.investigation import (
                        InvestigationScenario,
                        grade_scenario,
                    )

                    usage = final_state.get("investigation_usage")
                    if usage is not None:
                        input_tokens = usage.input_tokens
                        output_tokens = usage.output_tokens
                    proposal_valid = (
                        investigation_result is not None
                        and usage is not None
                        and investigation_provider is not None
                    )
                    provider_marker = investigation_provider is not None
                    mcp_marker = (
                        self.config.enable_live_eval
                        and type(effective_mcp).__name__ == "StreamableHttpVehicleMcpAdapter"
                    )
                    if investigation_result is not None:
                        observed_action = (
                            investigation_result.action.value
                            if investigation_result.action is not None
                            else (
                                "NO_ACTION"
                                if not investigation_result.dispatched
                                and investigation_result.completed
                                else None
                            )
                        )
                        dispatched = investigation_result.dispatched
                        result_references = tuple(investigation_result.references)
                        if not citation_references:
                            citation_references = tuple(
                                citation.passage_id
                                for citation in investigation_result.policy_citations
                            )
                            retrieved_passage_ids = citation_references
                            retrieved_citation_ids = citation_references
                    quality_passed = (
                        investigation_result is not None
                        and grade_scenario(
                            InvestigationScenario.model_validate(
                                {
                                    key: value
                                    for key, value in scn.items()
                                    if key in InvestigationScenario.model_fields
                                }
                            ),
                            investigation_result,
                        )
                        and is_awaiting_review
                    )
                    outcome = "SCORED" if quality_passed else "FAILED"
                    if investigation_ledger is not None:
                        proposal_rounds = investigation_ledger.proposal_rounds
                        supplementary_attempts = investigation_ledger.supplementary_attempts
                        supplementary_retries = investigation_ledger.supplementary_retries
                        termination_reason = investigation_ledger.status.value
                    if investigation_result is not None and termination_reason is None:
                        proposal_rounds = 1
                        supplementary_attempts = int(investigation_result.dispatched)
                    if not quality_passed:
                        termination_reason = "INVESTIGATION_SCENARIO_FAILED"
                        failure_reason = "INVESTIGATION_SCENARIO_FAILED"
                    elif termination_reason is None:
                        termination_reason = "COMPLETED"
                elif final_state.get("phase") == AssessmentRunPhase.FAILED:
                    outcome = "FAILED"
                    failure_reason = "WORKFLOW_FAILED"
                    quality_passed = False
                elif persisted_draft is not None:
                    if persisted_draft.metadata:
                        input_tokens = int(persisted_draft.metadata.get("input_tokens", 0))
                        output_tokens = int(persisted_draft.metadata.get("output_tokens", 0))
                    provider_marker = (
                        self.config.enable_live_eval
                        and getattr(effective_drafter, "provider", None) == self.config.provider
                    )
                    mcp_marker = self.config.enable_live_eval and isinstance(
                        effective_mcp, StreamableHttpVehicleMcpAdapter
                    )
                    if self._comparative_metrics:
                        from vehicle_risk_agent.evaluation.comparative import (
                            deterministic_labels_match,
                            get_e12_retrieval_label,
                            measure_report_draft_labels,
                        )

                        draft_measurement = measure_report_draft_labels(
                            scn.get("expected_labels"),
                            expected_action or "",
                            persisted_draft,
                            retrieval_label=get_e12_retrieval_label(scn.get("retrieval_query_id")),
                            retrieved_passage_ids=retrieved_passage_ids,
                            retrieved_citation_ids=retrieved_citation_ids,
                        )
                        retrieval_applicable = bool(draft_measurement["retrieval_applicable"])
                        claim_support_applicable = bool(
                            draft_measurement["claim_support_applicable"]
                        )
                        citation_grounding_applicable = bool(
                            draft_measurement["citation_grounding_applicable"]
                        )
                        abstention_applicable = bool(draft_measurement["abstention_applicable"])
                        deterministic_risk_applicable = True
                        useful_tool_selection_applicable = False
                        deterministic_risk_passed = deterministic_labels_match(
                            scn.get("expected_labels"), persisted_draft
                        )
                        retrieval_relevance = float(draft_measurement["retrieval_relevance"])
                        retrieval_precision = float(draft_measurement["retrieval_precision"])
                        retrieval_mrr = float(draft_measurement["retrieval_mrr"])
                        claim_support = float(draft_measurement["claim_support"])
                        citation_grounding = float(draft_measurement["citation_grounding"])
                        abstention_correct = bool(draft_measurement["abstention_correct"])
                        useful_tool_selection = float(draft_measurement["useful_tool_selection"])
                        missed_findings = int(draft_measurement["missed_findings"])
                        false_positive_citations = int(
                            draft_measurement["false_positive_citations"]
                        )
                    automatic_approval = (
                        assessment_state == AssessmentLifecycleState.RELEASED.value
                        or report_status == ReportDraftStatus.RELEASED.value
                    )

                    if expected_outcome in {"WITHHELD", "INCOMPLETE"}:
                        quality_passed = (
                            is_awaiting_review
                            and persisted_draft.status == ReportDraftStatus.DRAFT
                            and persisted_draft.score is None
                        )
                        outcome = expected_outcome
                    else:
                        quality_passed = (
                            is_awaiting_review
                            and persisted_draft.status == ReportDraftStatus.DRAFT
                            and persisted_draft.score is not None
                        )
                        outcome = "SCORED"
                else:
                    outcome = "FAILED"
                    failure_reason = "MISSING_PERSISTED_DRAFT"
                    quality_passed = False

            except Exception:
                outcome = "FAILED"
                failure_reason = "SCENARIO_EXECUTION_FAILED"
                termination_reason = "SCENARIO_EXECUTION_FAILED"
                quality_passed = False

            if termination_reason is None:
                termination_reason = "COMPLETED" if quality_passed else "SCENARIO_FAILED"
            latency = round(time.monotonic() - start_t, 3)
            cost = calculate_estimated_cost(input_tokens, output_tokens, self.pricing)
            cumulative_cost += cost

            if cumulative_cost > self.config.max_budget_usd:
                raise LiveEvaluationBudgetError(
                    f"Live evaluation budget exceeded: ${cumulative_cost:.4f} > "
                    f"${self.config.max_budget_usd:.2f}"
                )

            scenario_metrics.append(
                LiveScenarioMetrics(
                    scenario_id=scenario_id,
                    provider=self.config.provider,
                    draft_latency_seconds=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_cost_usd=cost,
                    quality_passed=quality_passed,
                    deterministic_risk_passed=deterministic_risk_passed,
                    retrieval_relevance=retrieval_relevance,
                    retrieval_precision=retrieval_precision,
                    retrieval_mrr=retrieval_mrr,
                    retrieval_applicable=retrieval_applicable,
                    claim_support=claim_support,
                    claim_support_applicable=claim_support_applicable,
                    citation_grounding=citation_grounding,
                    citation_grounding_applicable=citation_grounding_applicable,
                    abstention_correct=abstention_correct,
                    abstention_applicable=abstention_applicable,
                    useful_tool_selection=useful_tool_selection,
                    useful_tool_selection_applicable=useful_tool_selection_applicable,
                    deterministic_risk_applicable=deterministic_risk_applicable,
                    retrieved_passage_ids=retrieved_passage_ids,
                    retrieved_citation_ids=retrieved_citation_ids,
                    missed_findings=missed_findings,
                    false_positive_citations=false_positive_citations,
                    unauthorized_dispatches=unauthorized_dispatches,
                    automatic_approval=automatic_approval,
                    outcome=outcome,
                    failure_reason=failure_reason,
                    expected_action=expected_action,
                    observed_action=observed_action,
                    dispatched=dispatched,
                    result_references=result_references,
                    citation_references=citation_references,
                    synthetic_notice=synthetic_notice,
                    suite_version="e11-live-v1"
                    if self.config.suite == "e11-investigation"
                    else "general-v1",
                    scenario_version="e11-live-v1"
                    if self.config.suite == "e11-investigation"
                    else "general-v1",
                    proposal_valid=proposal_valid,
                    provider_marker=provider_marker,
                    mcp_marker=mcp_marker,
                    proposal_rounds=proposal_rounds,
                    supplementary_attempts=supplementary_attempts,
                    supplementary_retries=supplementary_retries,
                    termination_reason=termination_reason,
                    report_draft_id=report_draft_id,
                    report_status=report_status,
                    assessment_state=assessment_state,
                )
            )

        if self.config.suite == "e11-investigation":
            bounded_metrics: list[LiveScenarioMetrics] = []
            for metric in scenario_metrics:
                complete = (
                    metric.quality_passed
                    and metric.provider_marker
                    and metric.mcp_marker
                    and metric.proposal_valid is True
                    and metric.assessment_state == AssessmentLifecycleState.AWAITING_REVIEW.value
                    and metric.report_status == ReportDraftStatus.DRAFT.value
                    and metric.unauthorized_dispatches == 0
                    and not metric.automatic_approval
                    and metric.input_tokens <= 3072
                    and metric.output_tokens <= 512
                )
                bounded_metrics.append(
                    metric.model_copy(
                        update={
                            "quality_passed": complete,
                            "failure_reason": metric.failure_reason
                            if complete
                            else (metric.failure_reason or "INCOMPLETE_LIVE_EVIDENCE"),
                        }
                    )
                )
            scenario_metrics = bounded_metrics

        execution_mode = "LIVE" if self.config.enable_live_eval else "OFFLINE"
        record = LiveEvaluationRecord.from_scenario_metrics(
            scenario_metrics=scenario_metrics,
            model_version=self.config.model
            if self.config.enable_live_eval
            else "offline-deterministic",
            provider=self.config.provider,
            prompt_version="prompt-2026.1" if self.config.enable_live_eval else "offline-v1",
            corpus_version="corpus-2026.1" if self.config.enable_live_eval else "corpus-offline",
            index_version="pgvector-hnsw-v1",
            policy_version="nz-vehicle-risk-v1",
            grader_version="grader-2026.1",
            code_version=resolve_source_commit(),
            pricing_provenance=self.pricing.provenance,
            execution_mode=execution_mode,
            p95_latency_threshold=self.config.p95_latency_threshold,
            suite_version=(
                "e11-live-v1" if self.config.suite == "e11-investigation" else "general-v1"
            ),
            input_price_per_million=self.pricing.input_token_cost_per_million,
            output_price_per_million=self.pricing.output_token_cost_per_million,
            pricing_source_url=self.pricing.source_url,
            pricing_effective_at=self.pricing.effective_at,
            pricing_checked_at=self.pricing.checked_at,
        )

        if self.config.output_file:
            record.save_to_file(self.config.output_file)

        return record

    async def _run_e12_live_comparative(
        self,
        session_factory: Any | None = None,
        mcp_adapter: Any | None = None,
        drafting_adapter: Any | None = None,
        settings: Any | None = None,
    ) -> ComparativeReport:
        """Run the four frozen inputs in both live modes at three repeats."""
        if drafting_adapter is not None and getattr(drafting_adapter, "provider", None) != (
            self.config.provider
        ):
            raise LiveEvaluationBudgetError(
                "Live evaluation refused: injected drafting adapter does not match "
                "the selected provider."
            )
        from vehicle_risk_agent.evaluation.comparative import (
            ComparativeMetric,
            ComparativeMode,
            build_comparative_report,
            build_offline_comparative_report,
            get_e12_held_out_scenarios,
            get_e12_retrieval_label,
            load_e12_evaluation_config,
            load_semantic_judgments,
        )

        self.validate_readiness()
        cfg = load_e12_evaluation_config()
        run_cfg = cfg.model_copy(
            update={
                "provider": self.config.provider,
                "model": self.config.model,
                "pricing_input_usd_per_million": self.pricing.input_token_cost_per_million,
                "pricing_output_usd_per_million": self.pricing.output_token_cost_per_million,
                "pricing_source_url": self.pricing.source_url,
                "pricing_checked_at": self.pricing.checked_at,
            }
        )
        scenario_by_id = {item.scenario_id: item for item in get_e12_held_out_scenarios(run_cfg)}
        offline_report = await build_offline_comparative_report(run_cfg)
        metrics: list[ComparativeMetric] = list(offline_report.metrics)
        cumulative_cost = 0.0
        reserved_cost = 0.0
        worst_case_call_cost = _e12_worst_case_call_cost(
            self.pricing,
            self.config.max_output_tokens,
        )

        def reserve_provider_call() -> None:
            nonlocal reserved_cost
            projected_cost = reserved_cost + worst_case_call_cost
            if projected_cost > self.config.max_budget_usd:
                raise LiveEvaluationBudgetError(
                    "Live evaluation budget reservation exceeded before provider call."
                )
            reserved_cost = projected_cost

        for repeat in range(1, cfg.repeats + 1):
            for overlay in cfg.comparable_shared_inputs:
                scenario = scenario_by_id[overlay.scenario_id]
                common_payload = {
                    "scenario_id": scenario.scenario_id,
                    "vin": scenario.vin,
                    "sale_type": scenario.context.sale_type,
                    "questions": [overlay.investigation_question],
                    "expected_outcome": (
                        scenario.expected_labels.assessment_outcome.value
                        if scenario.expected_labels.assessment_outcome is not None
                        else "SCORED"
                    ),
                    "expected_action": overlay.expected_action,
                    "expected_field": overlay.expected_field,
                    "retrieval_query_id": overlay.retrieval_query_id,
                    "expected_labels": scenario.expected_labels.model_dump(),
                }
                draft_runner = LiveEvaluationRunner(
                    config=self.config.model_copy(
                        update={
                            "suite": "general",
                            "max_scenarios": 1,
                            "max_budget_usd": 5.0,
                        }
                    ),
                    active_corpus=self.active_corpus,
                )
                draft_runner._comparative_metrics = True
                reserve_provider_call()
                draft_record = await draft_runner.run_bounded_acceptance(
                    session_factory=session_factory,
                    mcp_adapter=mcp_adapter,
                    drafting_adapter=drafting_adapter,
                    settings=settings,
                    scenarios=[common_payload],
                )
                if isinstance(draft_record, ComparativeReport):
                    raise LiveEvaluationBudgetError(
                        "unexpected comparative report from drafting mode"
                    )
                draft_metric = draft_record.scenarios[0]
                draft_cost = draft_metric.estimated_cost_usd
                cumulative_cost += draft_cost
                if cumulative_cost > self.config.max_budget_usd:
                    raise LiveEvaluationBudgetError(
                        f"Live evaluation budget exceeded: ${cumulative_cost:.4f} > "
                        f"${self.config.max_budget_usd:.2f}"
                    )
                metrics.append(
                    ComparativeMetric(
                        scenario_id=scenario.scenario_id,
                        provider=draft_metric.provider,
                        mode=ComparativeMode.LIVE_DRAFTING,
                        repeat=repeat,
                        quality_passed=draft_metric.quality_passed,
                        deterministic_risk_passed=draft_metric.deterministic_risk_passed,
                        retrieval_relevance=draft_metric.retrieval_relevance,
                        retrieval_precision=draft_metric.retrieval_precision,
                        retrieval_mrr=draft_metric.retrieval_mrr,
                        retrieval_applicable=draft_metric.retrieval_applicable,
                        citation_grounding=draft_metric.citation_grounding,
                        citation_grounding_applicable=draft_metric.citation_grounding_applicable,
                        claim_support=draft_metric.claim_support,
                        claim_support_applicable=draft_metric.claim_support_applicable,
                        abstention_correct=draft_metric.abstention_correct,
                        abstention_applicable=draft_metric.abstention_applicable,
                        useful_tool_selection=draft_metric.useful_tool_selection,
                        useful_tool_selection_applicable=draft_metric.useful_tool_selection_applicable,
                        deterministic_risk_applicable=draft_metric.deterministic_risk_applicable,
                        retrieved_passage_ids=draft_metric.retrieved_passage_ids,
                        retrieved_citation_ids=draft_metric.retrieved_citation_ids,
                        draft_latency_seconds=draft_metric.draft_latency_seconds,
                        input_tokens=draft_metric.input_tokens,
                        output_tokens=draft_metric.output_tokens,
                        estimated_cost_usd=draft_cost,
                        missed_findings=draft_metric.missed_findings,
                        false_positive_citations=draft_metric.false_positive_citations,
                        unauthorized_dispatches=draft_metric.unauthorized_dispatches,
                        automatic_approval=draft_metric.automatic_approval,
                        provider_marker=draft_metric.provider_marker,
                        mcp_marker=draft_metric.mcp_marker,
                        usage_known=draft_metric.input_tokens > 0
                        and draft_metric.output_tokens > 0,
                        failure_reason=draft_metric.failure_reason,
                    )
                )
                investigation_payload = {
                    **common_payload,
                    "intent": overlay.investigation_question,
                    "evidence_targets": (
                        [overlay.expected_field] if overlay.expected_field is not None else []
                    ),
                    "expected_dispatched": overlay.expected_action != "NO_ACTION",
                    "expected_query": (
                        overlay.investigation_question
                        if overlay.expected_action == "search_policy"
                        else None
                    ),
                }
                investigation_runner = LiveEvaluationRunner(
                    config=self.config.model_copy(
                        update={
                            "suite": "e11-investigation",
                            "max_scenarios": 1,
                            "max_budget_usd": 3.0,
                        }
                    ),
                    active_corpus=self.active_corpus,
                )
                investigation_runner._allow_custom_investigation = True
                investigation_runner._comparative_metrics = True
                reserve_provider_call()
                investigation_record = await investigation_runner.run_bounded_acceptance(
                    session_factory=session_factory,
                    mcp_adapter=mcp_adapter,
                    drafting_adapter=drafting_adapter,
                    settings=settings,
                    scenarios=[investigation_payload],
                )
                if isinstance(investigation_record, ComparativeReport):
                    raise LiveEvaluationBudgetError(
                        "unexpected comparative report from investigation mode"
                    )
                investigation_metric = investigation_record.scenarios[0]
                cumulative_cost += investigation_metric.estimated_cost_usd
                if cumulative_cost > self.config.max_budget_usd:
                    raise LiveEvaluationBudgetError(
                        f"Live evaluation budget exceeded: ${cumulative_cost:.4f} > "
                        f"${self.config.max_budget_usd:.2f}"
                    )
                allowed_actions = {
                    "NO_ACTION",
                    "explain_vehicle_field",
                    "get_vehicle_history",
                    "get_vehicle_revision",
                    "search_policy",
                }
                action_matches = investigation_metric.observed_action == overlay.expected_action
                retrieval_label = get_e12_retrieval_label(overlay.retrieval_query_id)
                citation_matches = retrieval_label is None or set(
                    investigation_metric.citation_references
                ) == set(retrieval_label.required_citation_ids)
                investigation_quality_passed = (
                    investigation_metric.quality_passed and action_matches and citation_matches
                )
                unauthorized_dispatches = int(
                    investigation_metric.dispatched is True
                    and investigation_metric.observed_action not in allowed_actions
                )
                metrics.append(
                    ComparativeMetric(
                        scenario_id=scenario.scenario_id,
                        provider=investigation_metric.provider,
                        mode=ComparativeMode.LIVE_INVESTIGATION,
                        repeat=repeat,
                        quality_passed=investigation_quality_passed,
                        deterministic_risk_passed=investigation_metric.deterministic_risk_passed,
                        retrieval_relevance=investigation_metric.retrieval_relevance,
                        retrieval_precision=investigation_metric.retrieval_precision,
                        retrieval_mrr=investigation_metric.retrieval_mrr,
                        retrieval_applicable=False,
                        citation_grounding=investigation_metric.citation_grounding,
                        citation_grounding_applicable=False,
                        claim_support=investigation_metric.claim_support,
                        claim_support_applicable=False,
                        abstention_correct=investigation_metric.abstention_correct,
                        abstention_applicable=False,
                        useful_tool_selection=1.0 if action_matches else 0.0,
                        useful_tool_selection_applicable=True,
                        deterministic_risk_applicable=False,
                        retrieved_passage_ids=investigation_metric.retrieved_passage_ids,
                        retrieved_citation_ids=investigation_metric.retrieved_citation_ids,
                        draft_latency_seconds=investigation_metric.draft_latency_seconds,
                        input_tokens=investigation_metric.input_tokens,
                        output_tokens=investigation_metric.output_tokens,
                        estimated_cost_usd=investigation_metric.estimated_cost_usd,
                        missed_findings=investigation_metric.missed_findings,
                        false_positive_citations=investigation_metric.false_positive_citations,
                        unauthorized_dispatches=unauthorized_dispatches,
                        automatic_approval=investigation_metric.automatic_approval,
                        provider_marker=investigation_metric.provider_marker,
                        mcp_marker=investigation_metric.mcp_marker,
                        usage_known=investigation_metric.input_tokens > 0
                        and investigation_metric.output_tokens > 0,
                        failure_reason=(
                            investigation_metric.failure_reason
                            if investigation_quality_passed
                            else "INVESTIGATION_EXPECTATION_FAILED"
                        ),
                    )
                )
                if cumulative_cost > self.config.max_budget_usd:
                    raise LiveEvaluationBudgetError(
                        f"Live evaluation budget exceeded: ${cumulative_cost:.4f} > "
                        f"${self.config.max_budget_usd:.2f}"
                    )
        report = build_comparative_report(
            run_cfg,
            metrics,
            semantic_judgments=load_semantic_judgments(),
            execution_mode="LIVE",
        )
        if self.config.output_file:
            report.save_to_file(self.config.output_file)
        return report

    def __repr__(self) -> str:
        return f"LiveEvaluationRunner(model={self.config.model!r})"

    def __str__(self) -> str:
        return self.__repr__()


def build_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for live evaluation."""
    parser = argparse.ArgumentParser(
        prog="vehicle_risk_agent.evaluation.live",
        description=(
            "Credential-gated live evaluation and release verdict for the "
            "Vehicle Risk Assessment Agent."
        ),
    )
    parser.add_argument(
        "--enable-live-eval",
        action="store_true",
        default=False,
        help="Explicit opt-in consent for paid live evaluation execution.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="general",
        choices=("general", "e11-investigation", "e12-comparative"),
        help="Evaluation suite to run.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help=(
            "Legacy general-suite key override; "
            "investigation suites use environment credentials only."
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=("anthropic", "gemini"),
        default=None,
        help="Provider (E12 defaults to Gemini; other suites default to Anthropic).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Provider model (defaults to claude-sonnet-4-6 or gemini-3.1-flash-lite).",
    )
    parser.add_argument(
        "--p95-latency-threshold",
        type=float,
        default=30.0,
        help="p95 draft latency threshold in seconds (default: 30.0s).",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=5.0,
        help="Maximum spend limit in USD (default: 5.00).",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional cap on unique scenarios to evaluate (max 3, or 4 for e12-comparative).",
    )
    parser.add_argument(
        "--require-neural-corpus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require active neural policy corpus during readiness check (default: True).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to save machine-readable evaluation record JSON.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Run deterministic offline control scenarios without paid provider calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate credentials and configuration without making paid API calls.",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    """Command-line interface entry point for live evaluation."""
    import asyncio

    parser = build_parser()
    parsed = parser.parse_args(args)
    provider = cast(
        Literal["anthropic", "gemini"],
        parsed.provider or ("gemini" if parsed.suite == "e12-comparative" else "anthropic"),
    )
    model = parsed.model or (
        "gemini-3.1-flash-lite" if provider == "gemini" else "claude-sonnet-4-6"
    )

    credential_name = _PROVIDER_CREDENTIALS[provider]
    if parsed.suite in {"e11-investigation", "e12-comparative"} and parsed.api_key:
        if parsed.suite == "e12-comparative" and parsed.output_file:
            from vehicle_risk_agent.evaluation.comparative import build_blocked_report

            build_blocked_report(reason="ENVIRONMENT_ONLY_CREDENTIALS").save_to_file(
                parsed.output_file
            )
        sys.stderr.write(
            f"Live evaluation refused: {parsed.suite} accepts {credential_name} only.\n"
        )
        return 1

    if parsed.offline:
        sys.stdout.write(
            "Offline evaluation starting (deterministic control mode, not live evidence)...\n"
        )
        config = LiveEvaluationConfig(
            enable_live_eval=False,
            provider=provider,
            model=model,
            output_file=parsed.output_file,
            max_scenarios=parsed.max_scenarios,
            max_budget_usd=parsed.max_budget_usd,
            suite=parsed.suite,
        )
        runner = LiveEvaluationRunner(config=config)
        try:
            record = asyncio.run(runner.run_bounded_acceptance())
            if isinstance(record, ComparativeReport):
                passed = sum(
                    metric.quality_passed
                    for metric in record.metrics
                    if metric.mode.value == "OFFLINE_BASELINE"
                )
                total = record.offline_runs
            else:
                passed = record.passed_scenarios
                total = record.total_scenarios
            sys.stdout.write(
                f"Offline evaluation complete: {passed}/{total} passed. "
                f"Verdict: {record.release_verdict}.\n"
            )
            return 0 if record.verdict_passed else 1
        except Exception:
            sys.stderr.write("Offline evaluation failed: OFFLINE_EVALUATION_FAILED.\n")
            return 1

    config = LiveEvaluationConfig(
        enable_live_eval=parsed.enable_live_eval,
        api_key=parsed.api_key,
        provider=provider,
        model=model,
        p95_latency_threshold=parsed.p95_latency_threshold,
        max_budget_usd=parsed.max_budget_usd,
        max_scenarios=parsed.max_scenarios,
        output_file=parsed.output_file,
        require_neural_corpus=parsed.require_neural_corpus,
        suite=parsed.suite,
        mcp_server_url=os.environ.get("MCP_SERVER_URL"),
    )
    active_corpus = None
    if parsed.enable_live_eval and parsed.require_neural_corpus:
        try:
            from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

            from vehicle_risk_agent.config import Settings
            from vehicle_risk_agent.policy.corpus_lifecycle import CorpusLifecycleManager

            async def _resolve_corpus() -> Any:
                settings = Settings()
                engine = create_async_engine(settings.database_url, echo=False)
                sm = async_sessionmaker(engine, expire_on_commit=False)
                async with sm() as sess:
                    return await CorpusLifecycleManager(sess).get_active_corpus()

            active_corpus = asyncio.run(_resolve_corpus())
        except Exception:
            active_corpus = None

    runner = LiveEvaluationRunner(config=config, active_corpus=active_corpus)

    try:
        runner.validate_readiness()
    except (
        LiveEvaluationConsentError,
        LiveEvaluationCredentialsError,
        LiveEvaluationBudgetError,
        LiveEvaluationCorpusError,
    ) as exc:
        if parsed.suite == "e12-comparative" and parsed.output_file:
            from vehicle_risk_agent.evaluation.comparative import build_blocked_report

            build_blocked_report(reason=type(exc).__name__).save_to_file(parsed.output_file)
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    if parsed.dry_run:
        sys.stdout.write(
            f"Live evaluation configuration for {config.provider}/{config.model} and "
            "credentials validated successfully (dry run).\n"
        )
        return 0

    sys.stdout.write(
        f"Live evaluation starting for model {config.model}. "
        f"p95 threshold: {config.p95_latency_threshold}s, "
        f"budget cap: ${config.max_budget_usd:.2f}...\n"
    )
    try:
        record = asyncio.run(runner.run_bounded_acceptance())
        if isinstance(record, ComparativeReport):
            sys.stdout.write(
                f"Live comparative evaluation complete: {record.live_runs} live runs, "
                f"p95 latency: {record.p95_latency_seconds}s, "
                f"total cost: ${record.total_estimated_cost_usd:.4f}. "
                f"Verdict: {record.release_verdict}.\n"
            )
        else:
            sys.stdout.write(
                f"Live evaluation complete: {record.passed_scenarios}/"
                f"{record.total_scenarios} passed. "
                f"p95 latency: {record.p95_draft_latency_seconds}s, "
                f"total cost: ${record.total_estimated_cost_usd:.4f}. "
                f"Verdict: {record.release_verdict}.\n"
            )
        return 0 if record.verdict_passed else 1
    except Exception:
        if parsed.suite == "e12-comparative" and parsed.output_file:
            from vehicle_risk_agent.evaluation.comparative import build_blocked_report

            build_blocked_report(reason="LIVE_EXECUTION_FAILED").save_to_file(parsed.output_file)
        sys.stderr.write("Live evaluation failed: LIVE_EXECUTION_FAILED.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
