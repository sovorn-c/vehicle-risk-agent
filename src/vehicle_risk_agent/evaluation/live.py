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

from pydantic import BaseModel, ConfigDict, Field


class LiveEvaluationConsentError(Exception):
    """Raised when live evaluation is attempted without explicit provider consent opt-in."""


class LiveEvaluationCredentialsError(Exception):
    """Raised when required provider API keys are missing or invalid."""


class ModelPricingConfig(BaseModel):
    """Pricing configuration per million input/output tokens for cost estimation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "claude-3-5-sonnet-20241022"
    input_token_cost_per_million: float = 3.00
    output_token_cost_per_million: float = 15.00


class LiveScenarioMetrics(BaseModel):
    """Per-scenario live evaluation execution metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    draft_latency_seconds: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    quality_passed: bool


class LiveEvaluationRecord(BaseModel):
    """Immutable, machine-readable record of a live evaluation run and release verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    created_at: str
    model_version: str
    prompt_version: str
    corpus_version: str
    policy_version: str
    grader_version: str
    code_version: str
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

    @classmethod
    def from_scenario_metrics(
        cls,
        scenario_metrics: list[LiveScenarioMetrics],
        model_version: str = "claude-3-5-sonnet-20241022",
        prompt_version: str = "prompt-2026.1",
        corpus_version: str = "corpus-2026.1",
        policy_version: str = "nz-vehicle-risk-v1",
        grader_version: str = "grader-2026.1",
        code_version: str = "0.1.0",
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
            "policy_version": policy_version,
            "grader_version": grader_version,
            "code_version": code_version,
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
            policy_version=policy_version,
            grader_version=grader_version,
            code_version=code_version,
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
    model: str = "claude-3-5-sonnet-20241022"
    p95_latency_threshold: float = 30.0
    max_budget_usd: float = 5.0
    max_scenarios: int | None = None
    output_file: str | None = None

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
    ) -> None:
        self.config = config or LiveEvaluationConfig()
        self.pricing = pricing or ModelPricingConfig(model=self.config.model)

    def validate_readiness(self) -> None:
        """Verify explicit operator consent and valid credentials before execution."""
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
        default="claude-3-5-sonnet-20241022",
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
        help="Maximum spend limit in USD (default: .00).",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional cap on number of scenarios to evaluate.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to save machine-readable evaluation record JSON.",
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

    config = LiveEvaluationConfig(
        enable_live_eval=parsed.enable_live_eval,
        api_key=parsed.api_key,
        model=parsed.model,
        p95_latency_threshold=parsed.p95_latency_threshold,
        max_budget_usd=parsed.max_budget_usd,
        max_scenarios=parsed.max_scenarios,
        output_file=parsed.output_file,
    )
    runner = LiveEvaluationRunner(config=config)

    try:
        runner.validate_readiness()
    except (LiveEvaluationConsentError, LiveEvaluationCredentialsError) as exc:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
