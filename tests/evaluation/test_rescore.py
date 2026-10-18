"""Tests for no-network re-score command and split semantic gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from vehicle_risk_agent.evaluation.comparative import (
    ComparativeMetric,
    ComparativeMode,
    ComparativeReport,
    SemanticJudgment,
    build_blocked_report,
    build_comparative_report,
    load_e12_evaluation_config,
)
from vehicle_risk_agent.evaluation.publication import validate_publication
from vehicle_risk_agent.evaluation.rescore import rescore_report


def _complete_live_metrics(config: Any) -> list[ComparativeMetric]:
    def make_metric(
        scenario_id: str, mode: ComparativeMode, repeat: int, **overrides: Any
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
            "provider": "gemini",
            "provider_marker": True,
            "mcp_marker": True,
            "usage_known": True,
        }
        values.update(overrides)
        return ComparativeMetric(**values)

    metrics = [
        make_metric(
            sid,
            ComparativeMode.OFFLINE_BASELINE,
            0,
            claim_support_applicable=True,
            deterministic_risk_applicable=True,
        )
        for sid in config.held_out_scenario_ids
    ]

    for overlay in config.comparable_shared_inputs:
        for repeat in range(1, config.repeats + 1):
            retrieval_app = overlay.retrieval_query_id is not None
            metrics.append(
                make_metric(
                    overlay.scenario_id,
                    ComparativeMode.LIVE_DRAFTING,
                    repeat,
                    claim_support_applicable=True,
                    deterministic_risk_applicable=True,
                    retrieval_applicable=retrieval_app,
                    citation_grounding_applicable=retrieval_app,
                    abstention_applicable=retrieval_app,
                )
            )
            metrics.append(
                make_metric(
                    overlay.scenario_id,
                    ComparativeMode.LIVE_INVESTIGATION,
                    repeat,
                    useful_tool_selection_applicable=True,
                    deterministic_risk_applicable=False,
                )
            )
    return metrics


def _write_judgments_file(path: Path, judgments: list[dict[str, Any]]) -> None:
    path.write_text(yaml.safe_dump({"judgments": judgments}), encoding="utf-8")


def _valid_judgments(
    report: ComparativeReport, config: Any, **overrides: Any
) -> list[dict[str, Any]]:
    base = {
        "claim_support": 1.0,
        "missed_findings": 0,
        "false_positive_citations": 0,
        "reviewer_id": "operator",
        "evaluation_input_sha256": report.evaluation_input_sha256,
    }
    base.update(overrides)
    return [
        {
            "scenario_id": overlay.scenario_id,
            "mode": mode.value,
            "repeat": 1,
            **base,
        }
        for overlay in config.comparable_shared_inputs
        for mode in (ComparativeMode.LIVE_DRAFTING, ComparativeMode.LIVE_INVESTIGATION)
    ]


def test_rescoring_live_report_with_passing_judgments_gives_pass(tmp_path: Path) -> None:
    config = load_e12_evaluation_config()
    metrics = _complete_live_metrics(config)
    unbound_report = build_comparative_report(config, metrics, execution_mode="LIVE")
    assert unbound_report.release_verdict == "BLOCKED"
    assert unbound_report.verdict_passed is False
    assert unbound_report.semantic_status == "UNBOUND"

    report_path = tmp_path / "live-report.json"
    unbound_report.save_to_file(report_path)

    judgments_path = tmp_path / "judgments.yaml"
    _write_judgments_file(judgments_path, _valid_judgments(unbound_report, config))

    publication_path = tmp_path / "live-report.publication.json"
    rescored_report, publication = rescore_report(
        report_path=report_path,
        judgments_path=judgments_path,
        publication_file=publication_path,
    )

    assert rescored_report.release_verdict == "PASS"
    assert rescored_report.verdict_passed is True
    assert rescored_report.semantic_status == "PASSED"
    assert rescored_report.evaluation_input_sha256 == unbound_report.evaluation_input_sha256
    assert rescored_report.run_hash != unbound_report.run_hash

    # Publication matches
    assert publication is not None
    assert publication.release_verdict == "PASS"
    assert publication.verdict_passed is True
    assert publication.evaluation_input_sha256 == unbound_report.evaluation_input_sha256
    validate_publication(report_path, publication_path)


def test_hash_mismatch_gives_refusal_or_blocked(tmp_path: Path) -> None:
    config = load_e12_evaluation_config()
    metrics = _complete_live_metrics(config)
    report = build_comparative_report(config, metrics, execution_mode="LIVE")

    report_path = tmp_path / "live-report.json"
    report.save_to_file(report_path)

    judgments_path = tmp_path / "mismatched-judgments.yaml"
    # Provide hash mismatch in judgments
    mismatched_judgments = _valid_judgments(report, config, evaluation_input_sha256="0" * 64)
    _write_judgments_file(judgments_path, mismatched_judgments)

    # Re-score entry point refuses hash mismatch
    with pytest.raises(ValueError, match="evaluation_input_sha256"):
        rescore_report(report_path=report_path, judgments_path=judgments_path)

    # Calling build_comparative_report directly yields BLOCKED
    loaded_judgments = [SemanticJudgment.model_validate(item) for item in mismatched_judgments]
    direct_report = build_comparative_report(
        config,
        metrics,
        semantic_judgments=loaded_judgments,
        execution_mode="LIVE",
        source_commit=report.source_commit,
        config_sha256=report.config_sha256,
    )
    assert direct_report.release_verdict == "BLOCKED"
    assert direct_report.verdict_passed is False
    assert direct_report.semantic_status == "UNBOUND"


def test_fewer_than_eight_or_wrong_set_gives_blocked() -> None:
    config = load_e12_evaluation_config()
    metrics = _complete_live_metrics(config)
    report = build_comparative_report(config, metrics, execution_mode="LIVE")

    # Fewer than 8 judgments
    partial_judgments = _valid_judgments(report, config)[:7]
    judgments_objs = [SemanticJudgment.model_validate(item) for item in partial_judgments]

    partial_report = build_comparative_report(
        config,
        metrics,
        semantic_judgments=judgments_objs,
        execution_mode="LIVE",
        source_commit=report.source_commit,
        config_sha256=report.config_sha256,
    )
    assert partial_report.release_verdict == "BLOCKED"
    assert partial_report.verdict_passed is False
    assert partial_report.semantic_status == "UNBOUND"

    # Wrong scenario/mode set
    wrong_set = _valid_judgments(report, config)
    wrong_set[0]["scenario_id"] = "sc-wrong-id"
    wrong_objs = [SemanticJudgment.model_validate(item) for item in wrong_set]
    wrong_report = build_comparative_report(
        config,
        metrics,
        semantic_judgments=wrong_objs,
        execution_mode="LIVE",
        source_commit=report.source_commit,
        config_sha256=report.config_sha256,
    )
    assert wrong_report.release_verdict == "BLOCKED"
    assert wrong_report.verdict_passed is False
    assert wrong_report.semantic_status == "UNBOUND"


@pytest.mark.parametrize(
    ("override", "field_name"),
    [
        ({"missed_findings": 1}, "missed_findings"),
        ({"false_positive_citations": 1}, "false_positive_citations"),
        ({"claim_support": 0.8}, "claim_support"),
    ],
)
def test_bound_negative_judgments_give_fail(
    tmp_path: Path, override: dict[str, Any], field_name: str
) -> None:
    config = load_e12_evaluation_config()
    metrics = _complete_live_metrics(config)
    report = build_comparative_report(config, metrics, execution_mode="LIVE")

    report_path = tmp_path / f"live-report-{field_name}.json"
    report.save_to_file(report_path)

    # Create bound judgments but with negative outcome on first item
    bad_judgments = _valid_judgments(report, config)
    bad_judgments[0].update(override)

    judgments_path = tmp_path / f"judgments-{field_name}.yaml"
    _write_judgments_file(judgments_path, bad_judgments)

    rescored, publication = rescore_report(
        report_path=report_path,
        judgments_path=judgments_path,
    )
    assert rescored.release_verdict == "FAIL"
    assert rescored.verdict_passed is False
    assert rescored.semantic_status == "BELOW_THRESHOLD"


def test_non_live_and_blocker_reports_cannot_be_rescored(tmp_path: Path) -> None:
    config = load_e12_evaluation_config()

    # Blocker report
    blocker = build_blocked_report(config, reason="LiveEvaluationCorpusError")
    blocker_path = tmp_path / "blocker.json"
    blocker.save_to_file(blocker_path)

    empty_judgments = tmp_path / "empty-judgments.yaml"
    _write_judgments_file(empty_judgments, [])

    with pytest.raises(ValueError, match="blocker_reason"):
        rescore_report(report_path=blocker_path, judgments_path=empty_judgments)

    # Offline report
    offline = build_comparative_report(config, (), execution_mode="OFFLINE")
    offline_path = tmp_path / "offline.json"
    offline.save_to_file(offline_path)

    with pytest.raises(ValueError, match="execution_mode"):
        rescore_report(report_path=offline_path, judgments_path=empty_judgments)


def test_rescoring_is_deterministic(tmp_path: Path) -> None:
    config = load_e12_evaluation_config()
    metrics = _complete_live_metrics(config)
    report = build_comparative_report(config, metrics, execution_mode="LIVE")

    report_path = tmp_path / "live-report.json"
    report.save_to_file(report_path)

    judgments_path = tmp_path / "judgments.yaml"
    _write_judgments_file(judgments_path, _valid_judgments(report, config))

    res1, _ = rescore_report(
        report_path=report_path,
        judgments_path=judgments_path,
        output_file=tmp_path / "r1.json",
    )
    res2, _ = rescore_report(
        report_path=report_path,
        judgments_path=judgments_path,
        output_file=tmp_path / "r2.json",
    )

    assert res1.evaluation_input_sha256 == report.evaluation_input_sha256
    assert res2.evaluation_input_sha256 == report.evaluation_input_sha256
    assert res1.release_verdict == res2.release_verdict == "PASS"
    assert res1.semantic_status == res2.semantic_status == "PASSED"
    assert res1.judgments_sha256 == res2.judgments_sha256
