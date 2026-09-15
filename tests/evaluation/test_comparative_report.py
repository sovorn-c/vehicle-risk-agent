"""Contracts for the e12 comparative report and fail-closed verdict."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vehicle_risk_agent.evaluation.comparative import (
    ComparativeMetric,
    ComparativeMode,
    ComparativeReport,
    SemanticJudgment,
    build_blocked_report,
    build_comparative_report,
    build_offline_comparative_report,
    deterministic_labels_match,
    get_e12_held_out_scenarios,
    load_e12_evaluation_config,
    measure_report_draft_labels,
    semantic_gate_passes,
)
from vehicle_risk_agent.evaluation.provenance import sha256_bytes
from vehicle_risk_agent.evaluation.retrieval import RetrievalQueryLabel


def _complete_live_metric(
    scenario_id: str,
    mode: ComparativeMode,
    repeat: int,
    **overrides: Any,
) -> ComparativeMetric:
    values: dict[str, Any] = {
        "scenario_id": scenario_id,
        "mode": mode,
        "repeat": repeat,
        "quality_passed": True,
        "deterministic_risk_passed": True,
        "retrieval_relevance": 1.0,
        "retrieval_precision": 1.0,
        "retrieval_mrr": 1.0,
        "retrieval_applicable": False,
        "citation_grounding": 1.0,
        "citation_grounding_applicable": False,
        "claim_support": 1.0,
        "claim_support_applicable": False,
        "abstention_correct": True,
        "abstention_applicable": False,
        "useful_tool_selection": 1.0,
        "useful_tool_selection_applicable": False,
        "deterministic_risk_applicable": False,
        "draft_latency_seconds": 1.0,
        "input_tokens": 10,
        "output_tokens": 10,
        "estimated_cost_usd": 0.001,
        "provider_marker": True,
        "mcp_marker": True,
        "usage_known": True,
    }
    values.update(overrides)
    return ComparativeMetric(**values)


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


def test_prerequisite_blocked_report_can_be_published(tmp_path: Path) -> None:
    report_path = tmp_path / "blocked-report.json"
    report = build_blocked_report(reason="LiveEvaluationCorpusError")
    report.save_to_file(report_path)

    assert ComparativeReport.load_from_file(report_path) == report


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
            evaluation_input_sha256="a" * 64,
        )
        for overlay in config.comparable_shared_inputs
        for mode in (ComparativeMode.LIVE_DRAFTING, ComparativeMode.LIVE_INVESTIGATION)
    ]
    assert len(judgments) == 8
    assert semantic_gate_passes(judgments, config, evaluation_input_sha256="a" * 64) is True


def test_semantic_gate_rejects_judgments_bound_to_another_input() -> None:
    config = load_e12_evaluation_config()
    judgments = [
        SemanticJudgment(
            scenario_id=overlay.scenario_id,
            mode=mode,
            repeat=1,
            claim_support=1.0,
            missed_findings=0,
            false_positive_citations=0,
            reviewer_id="operator",
            evaluation_input_sha256="a" * 64,
        )
        for overlay in config.comparable_shared_inputs
        for mode in (ComparativeMode.LIVE_DRAFTING, ComparativeMode.LIVE_INVESTIGATION)
    ]

    assert (
        semantic_gate_passes(
            judgments,
            config,
            evaluation_input_sha256="a" * 64,
        )
        is True
    )
    assert (
        semantic_gate_passes(
            judgments,
            config,
            evaluation_input_sha256="b" * 64,
        )
        is False
    )


def test_report_hashes_supplied_judgment_values() -> None:
    config = load_e12_evaluation_config()
    judgments = (
        SemanticJudgment(
            scenario_id="sc-clean-01",
            mode=ComparativeMode.LIVE_DRAFTING,
            repeat=1,
            claim_support=1.0,
            missed_findings=0,
            false_positive_citations=0,
            reviewer_id="operator",
        ),
    )

    report = build_comparative_report(
        config,
        (),
        semantic_judgments=judgments,
        execution_mode="BLOCKED",
    )
    expected_digest = sha256_bytes(
        json.dumps(
            [item.model_dump(mode="json") for item in judgments],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    assert report.judgments_sha256 == expected_digest


def test_report_rejects_fabricated_provenance_inputs() -> None:
    config = load_e12_evaluation_config()

    with pytest.raises(ValueError, match="config digest"):
        build_comparative_report(config, (), config_sha256="a" * 64)
    with pytest.raises(ValueError, match="judgment digest"):
        build_comparative_report(config, (), judgments_sha256="b" * 64)
    with pytest.raises(ValueError, match="source commit"):
        build_comparative_report(config, (), source_commit="c" * 40)


def test_report_rejects_modified_frozen_threshold_configuration() -> None:
    config = load_e12_evaluation_config()
    modified = config.model_copy(
        update={
            "thresholds": config.thresholds.model_copy(
                update={"min_claim_support": 1.0}
            )
        }
    )

    with pytest.raises(ValueError, match="frozen"):
        build_comparative_report(modified, (), execution_mode="LIVE")


def test_duplicate_applicable_metric_rows_block_live_verdict() -> None:
    config = load_e12_evaluation_config()
    metrics = [
        _complete_live_metric(
            scenario_id,
            ComparativeMode.OFFLINE_BASELINE,
            0,
            claim_support_applicable=True,
            deterministic_risk_applicable=True,
        )
        for scenario_id in config.held_out_scenario_ids
    ]
    for overlay in config.comparable_shared_inputs:
        for repeat in range(1, config.repeats + 1):
            retrieval_applicable = overlay.retrieval_query_id is not None
            metrics.append(
                _complete_live_metric(
                    overlay.scenario_id,
                    ComparativeMode.LIVE_DRAFTING,
                    repeat,
                    claim_support_applicable=True,
                    deterministic_risk_applicable=True,
                    retrieval_applicable=retrieval_applicable,
                    citation_grounding_applicable=retrieval_applicable,
                    abstention_applicable=retrieval_applicable,
                )
            )
            metrics.append(
                _complete_live_metric(
                    overlay.scenario_id,
                    ComparativeMode.LIVE_INVESTIGATION,
                    repeat,
                    deterministic_risk_passed=False,
                    useful_tool_selection_applicable=True,
                )
            )

    metrics.append(
        _complete_live_metric(
            config.held_out_scenario_ids[0],
            ComparativeMode.OFFLINE_BASELINE,
            0,
            claim_support_applicable=True,
            deterministic_risk_applicable=True,
        )
    )
    preliminary = build_comparative_report(config, metrics, execution_mode="LIVE")
    binding = [
        SemanticJudgment(
            scenario_id=overlay.scenario_id,
            mode=mode,
            repeat=1,
            claim_support=1.0,
            missed_findings=0,
            false_positive_citations=0,
            reviewer_id="operator",
            evaluation_input_sha256=preliminary.evaluation_input_sha256,
        )
        for overlay in config.comparable_shared_inputs
        for mode in (ComparativeMode.LIVE_DRAFTING, ComparativeMode.LIVE_INVESTIGATION)
    ]
    report = build_comparative_report(
        config,
        metrics,
        semantic_judgments=binding,
        execution_mode="LIVE",
        source_commit=preliminary.source_commit,
        config_sha256=preliminary.config_sha256,
    )

    assert report.verdict_passed is False


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


def test_missing_ranked_retrieval_does_not_fallback_to_report_citations() -> None:
    label = RetrievalQueryLabel(
        query_id="q-test",
        query="statutory write-off",
        relevant_passage_ids=("gold",),
        required_citation_ids=("gold",),
    )
    claim = SimpleNamespace(
        evidence_refs=(),
        policy_citation_refs=("gold",),
        risk_factor_refs=(),
    )
    draft = SimpleNamespace(
        all_policy_citation_refs=("gold",),
        all_claims=(claim,),
        all_risk_factor_refs=(),
    )

    metrics = measure_report_draft_labels(
        {},
        "search_policy",
        draft,
        retrieval_label=label,
        retrieved_passage_ids=[],
        retrieved_citation_ids=[],
    )

    assert metrics["retrieval_relevance"] == 0.0
    assert metrics["retrieval_precision"] == 0.0
    assert metrics["citation_grounding"] == 0.0
    assert metrics["false_positive_citations"] == 0


def test_no_answer_with_extra_citation_counts_as_false_positive() -> None:
    label = RetrievalQueryLabel(
        query_id="q-no-answer",
        query="unsupported topic",
        is_no_answer=True,
    )
    claim = SimpleNamespace(
        evidence_refs=(),
        policy_citation_refs=("wrong",),
        risk_factor_refs=(),
    )
    draft = SimpleNamespace(
        all_policy_citation_refs=("wrong",),
        all_claims=(claim,),
        all_risk_factor_refs=(),
    )

    metrics = measure_report_draft_labels(
        {},
        "search_policy",
        draft,
        retrieval_label=label,
        retrieved_passage_ids=["wrong"],
        retrieved_citation_ids=["wrong"],
    )

    assert metrics["false_positive_citations"] == 1


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


def test_live_claim_support_excludes_offline_baseline_rows() -> None:
    config = load_e12_evaluation_config()
    metrics = [
        _complete_live_metric(
            "sc-clean-01",
            ComparativeMode.OFFLINE_BASELINE,
            0,
            claim_support=0.0,
            claim_support_applicable=True,
        ),
        _complete_live_metric(
            "sc-clean-01",
            ComparativeMode.LIVE_DRAFTING,
            1,
            claim_support=1.0,
            claim_support_applicable=True,
        ),
    ]

    report = build_comparative_report(config, metrics, execution_mode="LIVE")

    assert report.claim_support == 1.0


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
