"""Tests for live evaluation record emission, pricing, and latency thresholds."""

# story: e06s04
# task: e06s04-t02

import pytest

from vehicle_risk_agent.evaluation.live import (
    LiveEvaluationRecord,
    LiveScenarioMetrics,
    ModelPricingConfig,
    calculate_estimated_cost,
    compute_p95_latency,
)


def test_estimated_cost_calculation() -> None:
    """Cost calculation accurately applies configured per-million token pricing."""
    pricing = ModelPricingConfig(
        model="claude-3-5-sonnet-20241022",
        input_token_cost_per_million=3.00,
        output_token_cost_per_million=15.00,
    )
    cost = calculate_estimated_cost(
        input_tokens=10_000,
        output_tokens=2_000,
        pricing=pricing,
    )
    # 10k * /M = zsh.030
    # 2k * /M = zsh.030
    # Total = zsh.060
    assert pytest.approx(cost, 0.0001) == 0.0600


def test_p95_latency_computation() -> None:
    """p95 latency is correctly calculated from sorted sample latencies."""
    latencies = [1.0] * 19 + [35.0]
    p95 = compute_p95_latency(latencies)
    assert p95 == 35.0


def test_live_eval_record_contains_version_stamps_and_metrics() -> None:
    """LiveEvaluationRecord contains required version stamps, metrics, and verdict."""
    scenario_metrics = [
        LiveScenarioMetrics(
            scenario_id="scn-01",
            draft_latency_seconds=2.5,
            input_tokens=1200,
            output_tokens=350,
            estimated_cost_usd=0.00885,
            quality_passed=True,
        ),
        LiveScenarioMetrics(
            scenario_id="scn-02",
            draft_latency_seconds=3.1,
            input_tokens=1100,
            output_tokens=400,
            estimated_cost_usd=0.0093,
            quality_passed=True,
        ),
    ]

    record = LiveEvaluationRecord.from_scenario_metrics(
        scenario_metrics=scenario_metrics,
        model_version="claude-3-5-sonnet-20241022",
        prompt_version="prompt-2026.1",
        corpus_version="corpus-2026.1",
        policy_version="nz-vehicle-risk-v1",
        grader_version="grader-2026.1",
        code_version="0.1.0",
        p95_latency_threshold=30.0,
    )

    assert record.total_scenarios == 2
    assert record.passed_scenarios == 2
    assert record.p95_draft_latency_seconds <= 30.0
    assert record.p95_latency_passed is True
    assert record.release_verdict == "PASS"
    assert record.verdict_passed is True
    assert record.model_version == "claude-3-5-sonnet-20241022"
    assert record.prompt_version == "prompt-2026.1"
    assert record.corpus_version == "corpus-2026.1"
    assert record.policy_version == "nz-vehicle-risk-v1"
    assert record.grader_version == "grader-2026.1"
    assert record.code_version == "0.1.0"


def test_release_verdict_fails_when_p95_latency_exceeds_threshold() -> None:
    """Release verdict is FAIL when p95 draft latency exceeds 30 seconds."""
    scenario_metrics = [
        LiveScenarioMetrics(
            scenario_id=f"scn-{i:02d}",
            draft_latency_seconds=32.0 if i == 20 else 5.0,
            input_tokens=1000,
            output_tokens=200,
            estimated_cost_usd=0.006,
            quality_passed=True,
        )
        for i in range(1, 21)
    ]

    record = LiveEvaluationRecord.from_scenario_metrics(
        scenario_metrics=scenario_metrics,
        model_version="claude-3-5-sonnet-20241022",
        prompt_version="prompt-2026.1",
        corpus_version="corpus-2026.1",
        policy_version="nz-vehicle-risk-v1",
        grader_version="grader-2026.1",
        code_version="0.1.0",
        p95_latency_threshold=30.0,
    )

    assert record.p95_draft_latency_seconds > 30.0
    assert record.p95_latency_passed is False
    assert record.release_verdict == "FAIL"
    assert record.verdict_passed is False


def test_security_isolation_no_prompts_or_secrets_in_record() -> None:
    """LiveEvaluationRecord never serializes prompts, messages, or credentials."""
    record = LiveEvaluationRecord.from_scenario_metrics(
        scenario_metrics=[
            LiveScenarioMetrics(
                scenario_id="scn-01",
                draft_latency_seconds=1.5,
                input_tokens=500,
                output_tokens=150,
                estimated_cost_usd=0.00375,
                quality_passed=True,
            )
        ],
        model_version="claude-3-5-sonnet-20241022",
        prompt_version="prompt-2026.1",
        corpus_version="corpus-2026.1",
        policy_version="nz-vehicle-risk-v1",
        grader_version="grader-2026.1",
        code_version="0.1.0",
        p95_latency_threshold=30.0,
    )

    dump = record.model_dump_json()
    for sensitive in (
        "sk-ant",
        "system_prompt",
        "user_prompt",
        "messages",
        "raw_prompt",
        "password",
        "secret",
    ):
        assert sensitive not in dump.lower()
    # Confirm prompt content is not stored, only prompt_version
    assert "prompt_version" in record.model_dump()
    assert "prompt" not in record.model_dump()
