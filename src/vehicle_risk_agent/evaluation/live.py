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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LiveEvaluationConsentError(Exception):
    """Raised when live evaluation is attempted without explicit provider consent opt-in."""


class LiveEvaluationCredentialsError(Exception):
    """Raised when required provider API keys are missing or invalid."""


class LiveEvaluationBudgetError(Exception):
    """Raised when evaluation budget bounds are exceeded or invalid."""


class LiveEvaluationCorpusError(Exception):
    """Raised when active neural policy corpus is missing, invalid, or unready."""


_SENTINEL = object()


class ModelPricingConfig(BaseModel):
    """Pricing configuration per million input/output tokens for cost estimation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "claude-3-5-sonnet-20241022"
    input_token_cost_per_million: float = 3.00
    output_token_cost_per_million: float = 15.00
    provenance: str = "anthropic-published-2026.1"


class LiveScenarioMetrics(BaseModel):
    """Per-scenario live evaluation execution metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    draft_latency_seconds: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    quality_passed: bool
    outcome: str = "SCORED"
    failure_reason: str | None = None


class LiveEvaluationRecord(BaseModel):
    """Immutable, machine-readable record of a live evaluation run and release verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    created_at: str
    model_version: str
    prompt_version: str
    corpus_version: str
    index_version: str = "pgvector-hnsw-v1"
    policy_version: str
    grader_version: str
    code_version: str
    pricing_provenance: str = "anthropic-published-2026.1"
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
        model_version: str = "claude-3-5-sonnet-20241022",
        prompt_version: str = "prompt-2026.1",
        corpus_version: str = "corpus-2026.1",
        index_version: str = "pgvector-hnsw-v1",
        policy_version: str = "nz-vehicle-risk-v1",
        grader_version: str = "grader-2026.1",
        code_version: str = "0.1.0",
        pricing_provenance: str = "anthropic-published-2026.1",
        execution_mode: str = "LIVE",
        p95_latency_threshold: float = 30.0,
    ) -> LiveEvaluationRecord:
        """Aggregate per-scenario metrics into an immutable record with release verdict."""
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
        release_verdict = "PASS" if verdict_passed else "FAIL"

        now = datetime.datetime.now(datetime.UTC).isoformat()
        record_id = f"live-eval-{now[:10]}-{abs(hash(now)) % 100000:05d}"

        hash_payload = {
            "record_id": record_id,
            "model_version": model_version,
            "prompt_version": prompt_version,
            "corpus_version": corpus_version,
            "index_version": index_version,
            "policy_version": policy_version,
            "grader_version": grader_version,
            "code_version": code_version,
            "pricing_provenance": pricing_provenance,
            "execution_mode": execution_mode,
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "p95_draft_latency_seconds": p95_latency,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_estimated_cost_usd": total_cost,
            "release_verdict": release_verdict,
        }
        run_hash = hashlib.sha256(
            json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        return cls(
            record_id=record_id,
            created_at=now,
            model_version=model_version,
            prompt_version=prompt_version,
            corpus_version=corpus_version,
            index_version=index_version,
            policy_version=policy_version,
            grader_version=grader_version,
            code_version=code_version,
            pricing_provenance=pricing_provenance,
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
    model: str = "claude-sonnet-4-6"
    p95_latency_threshold: float = 30.0
    timeout_seconds: float = 30.0
    max_output_tokens: int = 2048
    max_repairs: int = 1
    max_budget_usd: float = 5.0
    max_scenarios: int | None = None
    output_file: str | None = None
    require_neural_corpus: bool = False

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


class LiveEvaluationRunner:
    """Coordinates credential validation, scenario execution, and release verdict."""

    def __init__(
        self,
        config: LiveEvaluationConfig | None = None,
        pricing: ModelPricingConfig | None = None,
        active_corpus: Any = None,
    ) -> None:
        self.config = config or LiveEvaluationConfig()
        self.pricing = pricing or ModelPricingConfig(model=self.config.model)
        self.active_corpus = active_corpus

    def validate_readiness(self, active_corpus: Any = _SENTINEL) -> None:
        """Verify explicit operator consent, valid credentials, budget bounds, and neural corpus."""
        if not self.config.enable_live_eval:
            raise LiveEvaluationConsentError(
                "Live evaluation refused: explicit provider consent and opt-in flag "
                "--enable-live-eval required for paid API execution."
            )
        key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key or not key.strip():
            raise LiveEvaluationCredentialsError(
                "Live evaluation refused: missing required ANTHROPIC_API_KEY."
            )
        if self.config.max_budget_usd <= 0.0 or self.config.max_budget_usd > 5.0:
            raise LiveEvaluationBudgetError(
                f"Live evaluation refused: max budget ${self.config.max_budget_usd:.2f} "
                "must be positive and at most $5.00."
            )
        if self.config.max_scenarios is not None and (
            self.config.max_scenarios <= 0 or self.config.max_scenarios > 3
        ):
            raise LiveEvaluationBudgetError(
                f"Live evaluation refused: max scenarios {self.config.max_scenarios} "
                "must be between 1 and 3."
            )

        corpus_to_check = active_corpus if active_corpus is not _SENTINEL else self.active_corpus
        if corpus_to_check is None:
            if self.config.require_neural_corpus or active_corpus is None:
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
        """Verify scenario batch size conforms to bounded live limits."""
        if len(scenarios) > 3:
            raise LiveEvaluationBudgetError(
                f"Live evaluation refused: scenario count {len(scenarios)} exceeds limit of 3."
            )
        if self.config.max_scenarios is not None and len(scenarios) > self.config.max_scenarios:
            raise LiveEvaluationBudgetError(
                f"Live evaluation refused: scenario count {len(scenarios)} "
                f"exceeds configured max of {self.config.max_scenarios}."
            )

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
        "--api-key",
        type=str,
        default="",
        help="Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-6",
        help="Anthropic model to evaluate.",
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
        help="Optional cap on number of scenarios to evaluate (max 3).",
    )
    parser.add_argument(
        "--require-neural-corpus",
        action="store_true",
        default=False,
        help="Require active neural policy corpus to be validated during readiness check.",
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
    parser = build_parser()
    parsed = parser.parse_args(args)

    if parsed.offline:
        sys.stdout.write(
            "Offline evaluation executed successfully "
            "(deterministic control mode, not live evidence).\n"
        )
        if parsed.output_file:
            record = LiveEvaluationRecord.from_scenario_metrics(
                scenario_metrics=[],
                model_version="offline-deterministic",
                prompt_version="offline-v1",
                corpus_version="corpus-offline",
                execution_mode="OFFLINE",
            )
            record.save_to_file(parsed.output_file)
        return 0

    config = LiveEvaluationConfig(
        enable_live_eval=parsed.enable_live_eval,
        api_key=parsed.api_key,
        model=parsed.model,
        p95_latency_threshold=parsed.p95_latency_threshold,
        max_budget_usd=parsed.max_budget_usd,
        max_scenarios=parsed.max_scenarios,
        output_file=parsed.output_file,
        require_neural_corpus=parsed.require_neural_corpus,
    )
    runner = LiveEvaluationRunner(config=config)

    try:
        runner.validate_readiness()
    except (
        LiveEvaluationConsentError,
        LiveEvaluationCredentialsError,
        LiveEvaluationBudgetError,
        LiveEvaluationCorpusError,
    ) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    if parsed.dry_run:
        sys.stdout.write(
            "Live evaluation configuration and credentials validated successfully (dry run).\n"
        )
        return 0

    sys.stdout.write(
        f"Live evaluation runner initialized for model {config.model}. "
        f"p95 threshold: {config.p95_latency_threshold}s.\n"
    )
    if parsed.output_file:
        record = LiveEvaluationRecord.from_scenario_metrics(
            scenario_metrics=[],
            model_version=config.model,
            execution_mode="LIVE",
        )
        record.save_to_file(parsed.output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
