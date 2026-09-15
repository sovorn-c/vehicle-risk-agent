"""Contracts for the e12 comparative report and fail-closed verdict."""

import asyncio
from types import SimpleNamespace
from typing import Any

from vehicle_risk_agent.evaluation.comparative import (
    ComparativeMetric,
    ComparativeMode,
    SemanticJudgment,
    build_comparative_report,
    build_offline_comparative_report,
    deterministic_labels_match,
    get_e12_held_out_scenarios,
    load_e12_evaluation_config,
    measure_report_draft_labels,
    semantic_gate_passes,
)
from vehicle_risk_agent.evaluation.retrieval import RetrievalQueryLabel


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


def test_unlabelled_report_metrics_are_not_treated_as_quality_failures() -> None:
    config = load_e12_evaluation_config()
    scenario = next(
        item
        for item in get_e12_held_out_scenarios(config)
        if item.scenario_id == "sc-risk-statutory-04"
    )
    claim = SimpleNamespace(evidence_refs=("obs-1",), policy_citation_refs=(), risk_factor_refs=())
    draft = SimpleNamespace(
        outcome="SCORED",
        band="HIGH",
        score=40,
        is_incomplete=False,
        all_risk_factor_refs=("STATUTORY",),
        all_policy_citation_refs=(),
        all_claims=(claim,),
    )

    metrics = measure_report_draft_labels(
        scenario.expected_labels.model_dump(), "search_policy", draft
    )

    assert deterministic_labels_match(scenario.expected_labels.model_dump(), draft) is True
    assert metrics["claim_support"] == 1.0
    assert metrics["citation_grounding"] == 1.0
    assert metrics["citation_grounding_applicable"] is False
    assert metrics["retrieval_relevance"] == 0.0
    assert metrics["retrieval_applicable"] is False
    assert metrics["missed_findings"] == 0


def test_ranked_retrieval_and_policy_claim_grounding_use_applicable_labels() -> None:
    label = RetrievalQueryLabel(
        query_id="q-test",
        query="statutory write-off",
        relevant_passage_ids=("gold",),
        required_citation_ids=("gold",),
    )
    policy_claim = SimpleNamespace(
        evidence_refs=(), policy_citation_refs=("gold",), risk_factor_refs=()
    )
    evidence_claim = SimpleNamespace(
        evidence_refs=("obs-1",), policy_citation_refs=(), risk_factor_refs=()
    )
    draft = SimpleNamespace(
        all_policy_citation_refs=("gold",),
        all_claims=(policy_claim, evidence_claim),
        all_risk_factor_refs=(),
    )

    metrics = measure_report_draft_labels(
        {"expected_citations": ("gold",)},
        "search_policy",
        draft,
        retrieval_label=label,
        retrieved_passage_ids=["wrong", "gold"],
        retrieved_citation_ids=["gold"],
    )

    assert metrics["retrieval_relevance"] == 1.0
    assert metrics["retrieval_precision"] == 0.5
    assert metrics["retrieval_mrr"] == 0.5
    assert metrics["citation_grounding"] == 1.0
    assert metrics["citation_grounding_applicable"] is True


def test_report_excludes_inapplicable_investigation_metrics() -> None:
    config = load_e12_evaluation_config()
    common: dict[str, Any] = {
        "quality_passed": True,
        "deterministic_risk_passed": True,
        "retrieval_relevance": 1.0,
        "retrieval_precision": 1.0,
        "retrieval_mrr": 1.0,
        "citation_grounding": 1.0,
        "claim_support": 1.0,
        "abstention_correct": True,
        "draft_latency_seconds": 1.0,
        "input_tokens": 10,
        "output_tokens": 10,
        "estimated_cost_usd": 0.001,
        "provider_marker": True,
        "mcp_marker": True,
        "usage_known": True,
    }
    metrics = [
        ComparativeMetric(
            scenario_id="sc-clean-01",
            mode=ComparativeMode.OFFLINE_BASELINE,
            repeat=0,
            **common,
            useful_tool_selection=0.0,
            retrieval_applicable=False,
            citation_grounding_applicable=False,
            abstention_applicable=False,
            useful_tool_selection_applicable=False,
        ),
        ComparativeMetric(
            scenario_id="sc-clean-01",
            mode=ComparativeMode.LIVE_DRAFTING,
            repeat=1,
            **common,
            useful_tool_selection=0.0,
            useful_tool_selection_applicable=False,
        ),
        ComparativeMetric(
            scenario_id="sc-clean-01",
            mode=ComparativeMode.LIVE_INVESTIGATION,
            repeat=1,
            **{
                **common,
                "retrieval_relevance": 0.0,
                "retrieval_precision": 0.0,
                "retrieval_mrr": 0.0,
                "citation_grounding": 0.0,
                "claim_support": 0.0,
                "abstention_correct": False,
            },
            deterministic_risk_applicable=False,
            retrieval_applicable=False,
            citation_grounding_applicable=False,
            abstention_applicable=False,
            claim_support_applicable=False,
            useful_tool_selection=1.0,
            useful_tool_selection_applicable=True,
        ),
    ]
    report = build_comparative_report(config, metrics, execution_mode="OFFLINE")

    assert report.retrieval_relevance == 1.0
    assert report.retrieval_precision == 1.0
    assert report.retrieval_mrr == 1.0
    assert report.citation_grounding == 1.0
    assert report.abstention_accuracy == 1.0
    assert report.useful_tool_selection == 1.0


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
