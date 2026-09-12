"""Contracts for the e12 comparative report and fail-closed verdict."""

import asyncio

from vehicle_risk_agent.evaluation.comparative import (
    ComparativeMetric,
    ComparativeMode,
    SemanticJudgment,
    build_comparative_report,
    build_offline_comparative_report,
    load_e12_evaluation_config,
    semantic_gate_passes,
)


def test_offline_report_covers_all_held_out_scenarios_and_is_blocked() -> None:
    report = asyncio.run(build_offline_comparative_report())

    assert report.total_runs == 30
    assert report.offline_runs == 30
    assert report.live_runs == 0
    assert report.execution_mode == "OFFLINE"
    assert report.release_verdict == "BLOCKED"
    assert report.verdict_passed is False
    assert len(report.held_out_scenario_ids) == 30
    assert report.synthetic_vehicle_evidence is True


def test_semantic_gate_requires_eight_first_repeat_human_judgments() -> None:
    config = load_e12_evaluation_config()
    assert semantic_gate_passes([], config) is False
    judgments = [
        SemanticJudgment(
            scenario_id=overlay.scenario_id,
            mode=mode,
            repeat=1,
            claim_support=1.0,
            missed_findings=0,
            false_positive_citations=0,
            reviewer_id="operator",
        )
        for overlay in config.comparable_shared_inputs
        for mode in (ComparativeMode.LIVE_DRAFTING, ComparativeMode.LIVE_INVESTIGATION)
    ]
    assert len(judgments) == 8
    assert semantic_gate_passes(judgments, config) is True


def test_live_report_cannot_pass_without_semantic_judgments() -> None:
    config = load_e12_evaluation_config()
    metrics = [
        ComparativeMetric(
            scenario_id=overlay.scenario_id,
            mode=mode,
            repeat=repeat,
            quality_passed=True,
            deterministic_risk_passed=True,
            retrieval_relevance=1.0,
            citation_grounding=1.0,
            abstention_correct=True,
            useful_tool_selection=1.0,
            draft_latency_seconds=1.0,
            input_tokens=10,
            output_tokens=10,
            estimated_cost_usd=0.001,
            provider_marker=True,
            mcp_marker=True,
        )
        for overlay in config.comparable_shared_inputs
        for mode in (ComparativeMode.LIVE_DRAFTING, ComparativeMode.LIVE_INVESTIGATION)
        for repeat in range(1, 4)
    ]
    report = build_comparative_report(config, metrics, execution_mode="LIVE")
    assert report.semantic_gate_passed is False
    assert report.release_verdict == "BLOCKED"
    assert report.verdict_passed is False
