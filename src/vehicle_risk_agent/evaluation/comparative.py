"""Held-out comparative evaluation contracts for e12."""

# story: e12s01

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vehicle_risk_agent.evaluation.graders import CompositeDomainGrader
from vehicle_risk_agent.evaluation.matrix import get_evaluation_matrix
from vehicle_risk_agent.evaluation.models import EvaluationScenario
from vehicle_risk_agent.evaluation.runner import ScenarioRunner


class ComparativeMode(StrEnum):
    """Execution modes included in the comparative report."""

    OFFLINE_BASELINE = "OFFLINE_BASELINE"
    LIVE_DRAFTING = "LIVE_DRAFTING"
    LIVE_INVESTIGATION = "LIVE_INVESTIGATION"


class ComparativeThresholds(BaseModel):
    """Frozen thresholds that determine comparative quality gates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_recall_at_5: float = Field(ge=0.0, le=1.0)
    min_precision_at_5: float = Field(ge=0.0, le=1.0)
    min_mrr: float = Field(ge=0.0, le=1.0)
    min_abstention_accuracy: float = Field(ge=0.0, le=1.0)
    min_citation_grounding: float = Field(ge=0.0, le=1.0)
    deterministic_risk_match: float = Field(ge=0.0, le=1.0)
    unauthorized_dispatches: int = Field(ge=0)
    automatic_approvals: int = Field(ge=0)
    min_useful_tool_selection: float = Field(ge=0.0, le=1.0)
    semantic_claim_support: float = Field(ge=0.0, le=1.0)
    semantic_missed_findings: int = Field(ge=0)
    semantic_false_positive_citations: int = Field(ge=0)
    max_p95_latency_seconds: float = Field(gt=0.0)


class InvestigationOverlay(BaseModel):
    """Frozen question and expected action for a comparable investigation input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    live_drafting: bool
    investigation_question: str = Field(min_length=1, max_length=500)
    expected_action: str = Field(min_length=1)
    expected_field: str | None = None


class E12EvaluationConfig(BaseModel):
    """Validated representation of ``specs/evaluation/e12-eval-v1.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1"
    config_id: str = Field(alias="id", min_length=1)
    frozen_at: str
    suite: str
    model: str
    pricing_input_usd_per_million: float = Field(gt=0.0)
    pricing_output_usd_per_million: float = Field(gt=0.0)
    pricing_source_url: str
    pricing_checked_at: str
    repeats: int = Field(ge=1)
    live_report_p95_seconds: float = Field(gt=0.0)
    live_suite_cost_usd: float = Field(gt=0.0)
    live_unique_comparable_scenarios: int = Field(ge=1)
    live_run_count: int = Field(ge=1)
    live_modes: tuple[ComparativeMode, ...] = (
        ComparativeMode.OFFLINE_BASELINE,
        ComparativeMode.LIVE_DRAFTING,
        ComparativeMode.LIVE_INVESTIGATION,
    )
    semantic_judgments_required: int = Field(ge=1)
    held_out_scenario_ids: tuple[str, ...]
    tuning_excluded_scenario_ids: tuple[str, ...]
    comparable_shared_inputs: tuple[InvestigationOverlay, ...]
    thresholds: ComparativeThresholds
    deterministic_grader_version: str
    retrieval_grader_version: str
    investigation_grader_version: str
    semantic_judge: str
    synthetic_vehicle_evidence: bool
    restricted_register_access: bool
    e11_gemini_live_validation: str

    @model_validator(mode="after")
    def validate_counts(self) -> E12EvaluationConfig:
        if len(self.held_out_scenario_ids) != 30:
            raise ValueError("e12-eval-v1 must contain exactly 30 held-out scenario IDs")
        if len(self.comparable_shared_inputs) != self.live_unique_comparable_scenarios:
            raise ValueError("comparable input count must match live_unique_comparable_scenarios")
        if self.live_run_count != self.live_unique_comparable_scenarios * self.repeats * 2:
            raise ValueError("live_run_count must cover both live modes at every repeat")
        if self.suite != "e12-comparative":
            raise ValueError("e12 config must declare suite e12-comparative")
        if self.live_modes != tuple(ComparativeMode):
            raise ValueError("e12 config must declare all three comparative modes")
        return self

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> E12EvaluationConfig:
        """Convert the authored nested YAML shape into a strict model."""
        pricing = data.get("pricing", {})
        held_out = data.get("held_out", {})
        excluded = data.get("tuning_excluded", {})
        shared = data.get("comparable_shared_inputs", {})
        scorers = data.get("scorers", {})
        disclosure = data.get("disclosure", {})
        thresholds = data.get("thresholds", {})
        return cls.model_validate(
            {
                "schema_version": str(data.get("schema_version", "1")),
                "id": data["id"],
                "frozen_at": data["frozen_at"],
                "suite": data["suite"],
                "model": data["model"],
                "pricing_input_usd_per_million": pricing["input_usd_per_million_tokens"],
                "pricing_output_usd_per_million": pricing["output_usd_per_million_tokens"],
                "pricing_source_url": pricing["source_url"],
                "pricing_checked_at": pricing["checked_at"],
                "repeats": data["repeats"],
                "live_report_p95_seconds": data["live_report_p95_seconds"],
                "live_suite_cost_usd": data["live_suite_cost_usd"],
                "live_unique_comparable_scenarios": data["live_unique_comparable_scenarios"],
                "live_run_count": data["live_run_count"],
                "live_modes": tuple(
                    data.get(
                        "live_modes", ("OFFLINE_BASELINE", "LIVE_DRAFTING", "LIVE_INVESTIGATION")
                    )
                ),
                "semantic_judgments_required": data["semantic_judgments_required"],
                "held_out_scenario_ids": tuple(held_out["scenario_ids"]),
                "tuning_excluded_scenario_ids": tuple(
                    excluded.get("e10_live_general", ()) + excluded.get("e11_live_v1", ())
                ),
                "comparable_shared_inputs": tuple(shared["scenarios"]),
                "thresholds": thresholds,
                "deterministic_grader_version": scorers["deterministic_grader_version"],
                "retrieval_grader_version": scorers["retrieval_grader_version"],
                "investigation_grader_version": scorers["investigation_grader_version"],
                "semantic_judge": scorers["semantic_judge"],
                "synthetic_vehicle_evidence": disclosure["synthetic_vehicle_evidence"],
                "restricted_register_access": disclosure["restricted_register_access"],
                "e11_gemini_live_validation": disclosure["e11_gemini_live_validation"],
            }
        )


# Public aliases make the authored contract discoverable without exposing the YAML shape.
ComparativeEvaluationConfig = E12EvaluationConfig


def load_e12_evaluation_config(
    path: str | Path = "specs/evaluation/e12-eval-v1.yaml",
) -> E12EvaluationConfig:
    """Load and validate the frozen e12 comparative configuration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("e12 evaluation configuration must be a YAML object")
    return E12EvaluationConfig.from_mapping(raw)


def validate_e12_split(
    config: E12EvaluationConfig,
    scenarios: Sequence[EvaluationScenario] | None = None,
) -> None:
    """Reject held-out contamination, missing IDs, or copied e11 wording."""
    matrix = list(scenarios or get_evaluation_matrix())
    by_id = {scenario.scenario_id: scenario for scenario in matrix}
    missing = set(config.held_out_scenario_ids) - set(by_id)
    if missing:
        raise ValueError(f"held-out scenarios are missing from e06 matrix: {sorted(missing)}")
    if set(config.held_out_scenario_ids) & set(config.tuning_excluded_scenario_ids):
        raise ValueError("held-out and tuning-excluded scenario IDs overlap")
    for overlay in config.comparable_shared_inputs:
        if overlay.scenario_id not in config.held_out_scenario_ids:
            raise ValueError(f"comparable input {overlay.scenario_id} is not held out")
        lowered = overlay.investigation_question.lower()
        if any(
            term in lowered
            for term in ("odometer discrepancy", "required vehicle evidence remains unresolved")
        ):
            raise ValueError("e12 investigation overlay copies e11-live-v1 wording")
        expected_actions = {
            "sc-clean-01": ("NO_ACTION", None),
            "sc-risk-statutory-04": ("search_policy", None),
            "sc-conflict-ppsr-01": ("explain_vehicle_field", "ppsr_result"),
            "sc-temporal-multi-rev-02": ("get_vehicle_history", None),
        }
        expected = expected_actions.get(overlay.scenario_id)
        if expected is None or (overlay.expected_action, overlay.expected_field) != expected:
            raise ValueError(f"unexpected e12 overlay contract for {overlay.scenario_id}")
        if not overlay.live_drafting:
            raise ValueError(f"e12 overlay {overlay.scenario_id} must be live comparable")


def get_e12_held_out_scenarios(
    config: E12EvaluationConfig | None = None,
    scenarios: Sequence[EvaluationScenario] | None = None,
) -> tuple[EvaluationScenario, ...]:
    """Return the frozen 30-scenario held-out set in declared order."""
    cfg = config or load_e12_evaluation_config()
    source = list(scenarios or get_evaluation_matrix())
    validate_e12_split(cfg, source)
    by_id = {scenario.scenario_id: scenario for scenario in source}
    return tuple(by_id[scenario_id] for scenario_id in cfg.held_out_scenario_ids)


class SemanticJudgment(BaseModel):
    """Human-reviewed semantic judgment for one first-repeat live result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    mode: ComparativeMode
    repeat: int = Field(ge=1)
    claim_support: float = Field(ge=0.0, le=1.0)
    missed_findings: int = Field(ge=0)
    false_positive_citations: int = Field(ge=0)
    reviewer_id: str = Field(min_length=1)


class ComparativeMetric(BaseModel):
    """Sanitized metric row for one scenario/mode/repeat."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    mode: ComparativeMode
    repeat: int = Field(ge=0)
    quality_passed: bool
    deterministic_risk_passed: bool
    retrieval_relevance: float = Field(ge=0.0, le=1.0)
    retrieval_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_mrr: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_grounding: float = Field(ge=0.0, le=1.0)
    claim_support: float = Field(default=0.0, ge=0.0, le=1.0)
    abstention_correct: bool
    useful_tool_selection: float = Field(ge=0.0, le=1.0)
    draft_latency_seconds: float = Field(ge=0.0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)
    missed_findings: int = Field(default=0, ge=0)
    false_positive_citations: int = Field(default=0, ge=0)
    unauthorized_dispatches: int = Field(default=0, ge=0)
    automatic_approval: bool = False
    provider_marker: bool = False
    mcp_marker: bool = False
    usage_known: bool = False
    failure_reason: str | None = None


class ComparativeReport(BaseModel):
    """Immutable, redacted three-mode comparative evaluation artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: str
    created_at: str
    config_id: str
    config_hash: str
    model: str
    execution_mode: str
    repeats: int
    pricing_input_usd_per_million: float
    pricing_output_usd_per_million: float
    pricing_source_url: str
    pricing_checked_at: str
    live_suite_cost_usd: float
    thresholds: ComparativeThresholds
    held_out_scenario_ids: tuple[str, ...]
    modes: tuple[ComparativeMode, ...]
    total_runs: int
    offline_runs: int
    live_runs: int
    metrics: tuple[ComparativeMetric, ...]
    semantic_judgments: tuple[SemanticJudgment, ...] = ()
    semantic_judgments_required: int
    semantic_gate_passed: bool
    deterministic_risk_match: float
    retrieval_relevance: float
    retrieval_precision: float
    retrieval_mrr: float
    claim_support: float
    citation_grounding: float
    abstention_accuracy: float
    useful_tool_selection: float
    missed_findings: int
    false_positive_citations: int
    unauthorized_dispatches: int
    automatic_approvals: int
    p95_latency_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float
    release_verdict: str
    verdict_passed: bool
    blocker_reason: str | None = None
    synthetic_vehicle_evidence: bool
    restricted_register_access: bool
    e11_gemini_live_validation: str
    run_hash: str

    @property
    def suite_version(self) -> str:
        """Expose the version shape shared with the legacy evaluation record."""
        return self.config_id

    @property
    def total_scenarios(self) -> int:
        """Expose held-out scenario coverage for shared runner callers."""
        return len(self.held_out_scenario_ids)

    @property
    def scenarios(self) -> tuple[Any, ...]:
        """Expose metric rows for callers that consume either evaluation record."""
        return self.metrics

    def save_to_file(self, path: str | Path) -> None:
        """Write the sanitized report as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_from_file(cls, path: str | Path) -> ComparativeReport:
        """Load and validate a report artifact."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return round(ordered[index], 3)


def load_semantic_judgments(
    path: str | Path = "specs/verifications/e12-semantic-judgments.yaml",
) -> tuple[SemanticJudgment, ...]:
    """Load operator judgments, returning no evidence when the file is absent."""
    target = Path(path)
    if not target.is_file():
        return ()
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    values = raw.get("judgments", ()) if isinstance(raw, dict) else raw
    if not isinstance(values, list):
        raise ValueError("semantic judgment artifact must contain a judgments list")
    return tuple(SemanticJudgment.model_validate(value) for value in values)


def semantic_gate_passes(
    judgments: Sequence[SemanticJudgment],
    config: E12EvaluationConfig,
) -> bool:
    """Require exactly the declared first-repeat human judgments for live PASS."""
    expected = {
        (overlay.scenario_id, mode, 1)
        for overlay in config.comparable_shared_inputs
        for mode in (ComparativeMode.LIVE_DRAFTING, ComparativeMode.LIVE_INVESTIGATION)
    }
    actual = {(item.scenario_id, item.mode, item.repeat) for item in judgments}
    if len(judgments) != config.semantic_judgments_required or actual != expected:
        return False
    return all(
        item.claim_support >= config.thresholds.semantic_claim_support
        and item.missed_findings <= config.thresholds.semantic_missed_findings
        and item.false_positive_citations <= config.thresholds.semantic_false_positive_citations
        for item in judgments
    )


def _report_hash_payload(report: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _label(labels: Any, name: str, default: Any = None) -> Any:
    return labels.get(name, default) if isinstance(labels, dict) else getattr(labels, name, default)


def _label_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def deterministic_labels_match(labels: Any, draft: Any | None) -> bool:
    """Compare only graph-owned risk outputs with frozen scenario labels."""
    if draft is None or labels is None:
        return False
    sufficiency = _label(labels, "sufficiency_outcome")
    if sufficiency is not None:
        is_incomplete = bool(getattr(draft, "is_incomplete", False))
        expected_incomplete = _label_value(sufficiency) == "INCOMPLETE"
        if is_incomplete != expected_incomplete:
            return False
    assessment_outcome = _label(labels, "assessment_outcome")
    if assessment_outcome is not None and _label_value(draft.outcome) != _label_value(
        assessment_outcome
    ):
        return False
    risk_band = _label(labels, "risk_band")
    if risk_band is not None and _label_value(draft.band) != _label_value(risk_band):
        return False
    min_score = _label(labels, "min_risk_score")
    if min_score is not None and (draft.score is None or draft.score < min_score):
        return False
    max_score = _label(labels, "max_risk_score")
    if max_score is not None and (draft.score is None or draft.score > max_score):
        return False
    required_factor_ids = _label(labels, "required_factor_ids", ())
    if not set(required_factor_ids).issubset(set(draft.all_risk_factor_refs)):
        return False
    citation_refs = set(draft.all_policy_citation_refs)
    expected_citations = _label(labels, "expected_citations", ())
    if not set(expected_citations).issubset(citation_refs):
        return False
    return not (_label(labels, "should_abstain") is True and citation_refs)


def deterministic_draft_matches(
    scenario: EvaluationScenario,
    draft: Any | None,
) -> bool:
    """Compare a draft with the frozen scenario's deterministic labels."""
    return deterministic_labels_match(scenario.expected_labels, draft)


def measure_report_draft(
    scenario: EvaluationScenario,
    expected_action: str,
    draft: Any | None,
) -> dict[str, float | int | bool]:
    """Measure observable draft references without turning unavailable gold labels into passes."""
    return measure_report_draft_labels(scenario.expected_labels, expected_action, draft)


def measure_report_draft_labels(
    labels: Any,
    expected_action: str,
    draft: Any | None,
) -> dict[str, float | int | bool]:
    """Measure a report against a frozen expected-label projection."""
    if draft is None or labels is None:
        return {
            "retrieval_relevance": 0.0,
            "retrieval_precision": 0.0,
            "retrieval_mrr": 0.0,
            "claim_support": 0.0,
            "citation_grounding": 0.0,
            "abstention_correct": False,
            "useful_tool_selection": 0.0,
            "missed_findings": len(_label(labels, "required_factor_ids", ())),
            "false_positive_citations": 0,
        }
    observed_refs = tuple(draft.all_policy_citation_refs)
    observed = set(observed_refs)
    expected = set(_label(labels, "expected_citations", ()))
    policy_expected = expected_action == "search_policy"
    relevant = observed & expected
    if expected:
        retrieval_relevance = len(relevant) / len(expected)
        retrieval_precision = len(relevant) / len(observed) if observed else 0.0
        first_match = next(
            (index + 1 for index, ref in enumerate(observed_refs) if ref in expected), None
        )
        retrieval_mrr = 1.0 / first_match if first_match is not None else 0.0
    else:
        retrieval_relevance = float(bool(observed) == policy_expected)
        retrieval_precision = retrieval_relevance
        retrieval_mrr = retrieval_relevance
    claims = tuple(draft.all_claims)
    supported_claims = sum(
        bool(claim.evidence_refs or claim.policy_citation_refs or claim.risk_factor_refs)
        for claim in claims
    )
    grounded_claims = sum(
        bool(claim.policy_citation_refs) and set(claim.policy_citation_refs).issubset(observed)
        for claim in claims
    )
    claim_support = supported_claims / len(claims) if claims else 0.0
    citation_grounding = grounded_claims / len(claims) if claims else 0.0
    return {
        "retrieval_relevance": retrieval_relevance,
        "retrieval_precision": retrieval_precision,
        "retrieval_mrr": retrieval_mrr,
        "claim_support": claim_support,
        "citation_grounding": citation_grounding,
        "abstention_correct": bool(observed) == policy_expected,
        "useful_tool_selection": retrieval_relevance,
        "missed_findings": len(
            set(_label(labels, "required_factor_ids", ())) - set(draft.all_risk_factor_refs)
        ),
        "false_positive_citations": len(observed - expected) if expected else 0,
    }


def build_comparative_report(
    config: E12EvaluationConfig,
    metrics: Sequence[ComparativeMetric],
    *,
    semantic_judgments: Sequence[SemanticJudgment] = (),
    execution_mode: str = "LIVE",
    blocker_reason: str | None = None,
) -> ComparativeReport:
    """Aggregate metrics and apply the fail-closed comparative verdict rules."""
    rows = tuple(metrics)
    deterministic = [row for row in rows if row.mode == ComparativeMode.OFFLINE_BASELINE]
    live = [row for row in rows if row.mode != ComparativeMode.OFFLINE_BASELINE]
    all_latencies = [row.draft_latency_seconds for row in live]
    count = len(deterministic)
    deterministic_match = (
        sum(row.deterministic_risk_passed for row in deterministic) / count if count else 0.0
    )
    retrieval = sum(row.retrieval_relevance for row in rows) / len(rows) if rows else 0.0
    precision = sum(row.retrieval_precision for row in rows) / len(rows) if rows else 0.0
    mrr = sum(row.retrieval_mrr for row in rows) / len(rows) if rows else 0.0
    claim_support = sum(row.claim_support for row in rows) / len(rows) if rows else 0.0
    citations = sum(row.citation_grounding for row in rows) / len(rows) if rows else 0.0
    abstention = sum(row.abstention_correct for row in rows) / len(rows) if rows else 0.0
    tool_selection = sum(row.useful_tool_selection for row in live) / len(live) if live else 0.0
    semantic = tuple(semantic_judgments)
    semantic_passed = semantic_gate_passes(semantic, config) if live else False
    quality = bool(rows) and all(row.quality_passed for row in rows)
    expected_live_ids = {
        (overlay.scenario_id, mode, repeat)
        for overlay in config.comparable_shared_inputs
        for mode in (ComparativeMode.LIVE_DRAFTING, ComparativeMode.LIVE_INVESTIGATION)
        for repeat in range(1, config.repeats + 1)
    }
    actual_live_ids = {(row.scenario_id, row.mode, row.repeat) for row in live}
    threshold_passed = (
        deterministic_match >= config.thresholds.deterministic_risk_match
        and retrieval >= config.thresholds.min_recall_at_5
        and precision >= config.thresholds.min_precision_at_5
        and mrr >= config.thresholds.min_mrr
        and citations >= config.thresholds.min_citation_grounding
        and abstention >= config.thresholds.min_abstention_accuracy
        and tool_selection >= config.thresholds.min_useful_tool_selection
        and sum(row.unauthorized_dispatches for row in rows)
        == config.thresholds.unauthorized_dispatches
        and sum(row.automatic_approval for row in rows) == config.thresholds.automatic_approvals
        and _p95(all_latencies) <= config.thresholds.max_p95_latency_seconds
        and (execution_mode != "LIVE" or actual_live_ids == expected_live_ids)
        and (execution_mode != "LIVE" or len(live) == config.live_run_count)
        and (execution_mode != "LIVE" or all(row.provider_marker for row in live))
        and (execution_mode != "LIVE" or all(row.mcp_marker for row in live))
        and (execution_mode != "LIVE" or all(row.usage_known for row in live))
        and (
            execution_mode != "LIVE"
            or sum(row.estimated_cost_usd for row in live) <= config.live_suite_cost_usd
        )
    )
    verdict_passed = (
        execution_mode == "LIVE"
        and blocker_reason is None
        and quality
        and threshold_passed
        and semantic_passed
    )
    verdict = (
        "PASS"
        if verdict_passed
        else (
            "BLOCKED"
            if execution_mode != "LIVE" or blocker_reason is not None or not semantic_passed
            else "FAIL"
        )
    )
    now = datetime.now(UTC).isoformat()
    report_id = f"e12-comparative-{now[:10]}-{abs(hash(now)) % 100000:05d}"
    base = {
        "report_id": report_id,
        "created_at": now,
        "config_id": config.config_id,
        "config_hash": _report_hash_payload(config.model_dump(mode="json", by_alias=True)),
        "model": config.model,
        "execution_mode": execution_mode,
        "repeats": config.repeats,
        "pricing_input_usd_per_million": config.pricing_input_usd_per_million,
        "pricing_output_usd_per_million": config.pricing_output_usd_per_million,
        "pricing_source_url": config.pricing_source_url,
        "pricing_checked_at": config.pricing_checked_at,
        "live_suite_cost_usd": config.live_suite_cost_usd,
        "thresholds": config.thresholds,
        "held_out_scenario_ids": config.held_out_scenario_ids,
        "modes": config.live_modes,
        "total_runs": len(rows),
        "offline_runs": len(deterministic),
        "live_runs": len(live),
        "metrics": rows,
        "semantic_judgments": semantic,
        "semantic_judgments_required": config.semantic_judgments_required,
        "semantic_gate_passed": semantic_passed,
        "deterministic_risk_match": round(deterministic_match, 4),
        "retrieval_relevance": round(retrieval, 4),
        "retrieval_precision": round(precision, 4),
        "retrieval_mrr": round(mrr, 4),
        "claim_support": round(claim_support, 4),
        "citation_grounding": round(citations, 4),
        "abstention_accuracy": round(abstention, 4),
        "useful_tool_selection": round(tool_selection, 4),
        "missed_findings": sum(row.missed_findings for row in rows),
        "false_positive_citations": sum(row.false_positive_citations for row in rows),
        "unauthorized_dispatches": sum(row.unauthorized_dispatches for row in rows),
        "automatic_approvals": sum(row.automatic_approval for row in rows),
        "p95_latency_seconds": _p95(all_latencies),
        "total_input_tokens": sum(row.input_tokens for row in rows),
        "total_output_tokens": sum(row.output_tokens for row in rows),
        "total_estimated_cost_usd": round(sum(row.estimated_cost_usd for row in rows), 6),
        "release_verdict": verdict,
        "verdict_passed": verdict_passed,
        "blocker_reason": blocker_reason,
        "synthetic_vehicle_evidence": config.synthetic_vehicle_evidence,
        "restricted_register_access": config.restricted_register_access,
        "e11_gemini_live_validation": config.e11_gemini_live_validation,
    }
    hash_base = {
        **base,
        "metrics": [item.model_dump(mode="json") for item in rows],
        "semantic_judgments": [item.model_dump(mode="json") for item in semantic],
        "thresholds": config.thresholds.model_dump(mode="json"),
    }
    return ComparativeReport.model_validate({**base, "run_hash": _report_hash_payload(hash_base)})


async def build_offline_comparative_report(
    config: E12EvaluationConfig | None = None,
    scenarios: Sequence[EvaluationScenario] | None = None,
) -> ComparativeReport:
    """Execute all 30 held-out scenarios through the deterministic workflow."""
    cfg = config or load_e12_evaluation_config()
    held_out = get_e12_held_out_scenarios(cfg, scenarios)
    runner = ScenarioRunner()
    grader = CompositeDomainGrader()
    metrics: list[ComparativeMetric] = []
    for scenario in held_out:
        result = await runner.run_scenario(scenario)
        evaluation = grader.evaluate(scenario, result)
        citation_grade = next(
            (item for item in evaluation.grader_results if item.grader_name == "citations_grader"),
            None,
        )
        retrieval_score = citation_grade.score if citation_grade is not None else 0.0
        draft_measurement = measure_report_draft(
            scenario,
            "search_policy" if scenario.expected_labels.expected_citations else "",
            result.report_draft,
        )
        risk_grader_names = {
            "evidence_state_grader",
            "outcome_grader",
            "risk_band_grader",
            "score_threshold_grader",
            "risk_factors_grader",
        }
        deterministic_risk_passed = all(
            item.passed
            for item in evaluation.grader_results
            if item.grader_name in risk_grader_names
        )
        metrics.append(
            ComparativeMetric(
                scenario_id=scenario.scenario_id,
                mode=ComparativeMode.OFFLINE_BASELINE,
                repeat=0,
                quality_passed=evaluation.passed,
                deterministic_risk_passed=deterministic_risk_passed,
                retrieval_relevance=retrieval_score,
                retrieval_precision=retrieval_score,
                retrieval_mrr=retrieval_score,
                citation_grounding=float(draft_measurement["citation_grounding"]),
                claim_support=float(draft_measurement["claim_support"]),
                abstention_correct=retrieval_score == 1.0,
                useful_tool_selection=0.0,
                draft_latency_seconds=0.0,
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=0.0,
                missed_findings=int(draft_measurement["missed_findings"]),
                false_positive_citations=int(draft_measurement["false_positive_citations"]),
                usage_known=True,
            )
        )
    return build_comparative_report(cfg, metrics, execution_mode="OFFLINE")


def build_blocked_report(
    config: E12EvaluationConfig | None = None,
    *,
    reason: str = "LIVE_PREREQUISITE_UNAVAILABLE",
) -> ComparativeReport:
    """Create a sanitized non-passing artifact when live setup cannot start."""
    cfg = config or load_e12_evaluation_config()
    return build_comparative_report(cfg, (), execution_mode="BLOCKED", blocker_reason=reason)


# Compatibility aliases for callers that prefer an explicit e12 name.
E12ComparativeReport = ComparativeReport
run_offline_comparative = build_offline_comparative_report
