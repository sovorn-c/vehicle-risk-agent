"""Deterministic domain graders for evidence, outcome, score, citations, and prohibited claims."""

# story: e06s01
# story: e06s02
# story: e06s03

from __future__ import annotations

import re
from typing import Protocol

from vehicle_risk_agent.evaluation.models import (
    EvaluationRun,
    EvaluationScenario,
    GraderResult,
)
from vehicle_risk_agent.evaluation.runner import ScenarioExecutionResult
from vehicle_risk_agent.risk.models import AssessmentOutcome

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"bearer\s+[a-zA-Z0-9_.-]{20,}", re.IGNORECASE),
    re.compile(r"password\s*=\s*[^\s,;]+", re.IGNORECASE),
    re.compile(r"-----BEGIN (RSA|EC|OPENSSH|PRIVATE) KEY-----", re.IGNORECASE),
)


class DomainGrader(Protocol):
    """Protocol for domain-specific evaluation graders."""

    def grade(
        self, scenario: EvaluationScenario, result: ScenarioExecutionResult
    ) -> GraderResult: ...


class EvidenceStateGrader:
    """Grades evidence sufficiency outcome and completeness."""

    def grade(self, scenario: EvaluationScenario, result: ScenarioExecutionResult) -> GraderResult:
        expected = scenario.expected_labels.sufficiency_outcome
        if expected is None:
            return GraderResult(
                grader_name="evidence_state_grader",
                passed=True,
                details="No expected sufficiency outcome specified",
            )

        actual = (
            result.sufficiency_result.outcome if result.sufficiency_result is not None else None
        )
        passed = actual == expected
        details = (
            f"Sufficiency matches expected {expected.value}"
            if passed
            else f"Sufficiency mismatch: expected {expected.value}, got {actual}"
        )
        return GraderResult(
            grader_name="evidence_state_grader",
            passed=passed,
            score=1.0 if passed else 0.0,
            details=details,
        )


class OutcomeGrader:
    """Grades assessment outcome against expected and prohibited outcomes."""

    def grade(self, scenario: EvaluationScenario, result: ScenarioExecutionResult) -> GraderResult:
        actual_outcome = (
            result.risk_result.outcome
            if result.risk_result is not None
            else (AssessmentOutcome.FAILED if result.mcp_error is not None else None)
        )

        # Prohibited outcomes check
        if actual_outcome in scenario.prohibited_labels.prohibited_outcomes:
            return GraderResult(
                grader_name="outcome_grader",
                passed=False,
                score=0.0,
                details=f"Prohibited outcome {actual_outcome.value} was produced",
            )

        expected = scenario.expected_labels.assessment_outcome
        if expected is None:
            return GraderResult(
                grader_name="outcome_grader",
                passed=True,
                details="No expected outcome specified",
            )

        passed = actual_outcome == expected
        details = (
            f"Outcome matches expected {expected.value}"
            if passed
            else f"Expected outcome {expected.value}, got {actual_outcome}"
        )
        return GraderResult(
            grader_name="outcome_grader",
            passed=passed,
            score=1.0 if passed else 0.0,
            details=details,
        )


class RiskBandGrader:
    """Grades calculated risk band against expected band."""

    def grade(self, scenario: EvaluationScenario, result: ScenarioExecutionResult) -> GraderResult:
        expected = scenario.expected_labels.risk_band
        if expected is None:
            return GraderResult(
                grader_name="risk_band_grader",
                passed=True,
                details="No expected risk band specified",
            )

        actual = result.risk_result.band if result.risk_result is not None else None
        passed = actual == expected
        details = (
            f"Risk band matches expected {expected.value}"
            if passed
            else f"Risk band mismatch: expected {expected.value}, got {actual}"
        )
        return GraderResult(
            grader_name="risk_band_grader",
            passed=passed,
            score=1.0 if passed else 0.0,
            details=details,
        )


class ScoreThresholdGrader:
    """Grades deterministic risk score against min and max bounds."""

    def grade(self, scenario: EvaluationScenario, result: ScenarioExecutionResult) -> GraderResult:
        min_score = scenario.expected_labels.min_risk_score
        max_score = scenario.expected_labels.max_risk_score
        if min_score is None and max_score is None:
            return GraderResult(
                grader_name="score_threshold_grader",
                passed=True,
                details="No score threshold bounds specified",
            )

        if result.risk_result is None or result.risk_result.score is None:
            return GraderResult(
                grader_name="score_threshold_grader",
                passed=False,
                score=0.0,
                details="No score calculated for scenario with required bounds",
            )

        actual_score = float(result.risk_result.score)
        passed = True
        reasons: list[str] = []
        if min_score is not None and actual_score < min_score:
            passed = False
            reasons.append(f"score {actual_score} < min {min_score}")
        if max_score is not None and actual_score > max_score:
            passed = False
            reasons.append(f"score {actual_score} > max {max_score}")

        details = "Score satisfies bounds" if passed else "; ".join(reasons)
        return GraderResult(
            grader_name="score_threshold_grader",
            passed=passed,
            score=1.0 if passed else 0.0,
            details=details,
            metrics={"actual_score": actual_score},
        )


class RiskFactorsGrader:
    """Grades active and detected risk factor evaluations."""

    def grade(self, scenario: EvaluationScenario, result: ScenarioExecutionResult) -> GraderResult:
        detected_factors: set[str] = set()
        if result.risk_result is not None:
            for f_res in result.risk_result.factors:
                if f_res.triggered:
                    detected_factors.add(str(f_res.factor.value))

        required = set(scenario.expected_labels.required_factor_ids)
        prohibited = set(scenario.prohibited_labels.prohibited_factor_ids)

        missing_required = required - detected_factors
        found_prohibited = prohibited & detected_factors

        passed = not missing_required and not found_prohibited
        details_list: list[str] = []
        if missing_required:
            details_list.append(f"Missing required factors: {sorted(missing_required)}")
        if found_prohibited:
            details_list.append(f"Found prohibited factors: {sorted(found_prohibited)}")
        if passed:
            details_list.append("Factor evaluations match expectations")

        return GraderResult(
            grader_name="risk_factors_grader",
            passed=passed,
            score=1.0 if passed else 0.0,
            details="; ".join(details_list),
        )


class CitationsGrader:
    """Grades policy citations, expected passage identifiers, and abstention."""

    def grade(self, scenario: EvaluationScenario, result: ScenarioExecutionResult) -> GraderResult:
        citations = result.policy_citations
        citation_ids = {c.passage_id for c in citations} | {c.section_identifier for c in citations}

        if scenario.expected_labels.should_abstain:
            passed = len(citations) == 0
            return GraderResult(
                grader_name="citations_grader",
                passed=passed,
                score=1.0 if passed else 0.0,
                details=(
                    "Abstention confirmed (no citations)"
                    if passed
                    else "Expected abstention but citations were returned"
                ),
            )

        expected_citations = set(scenario.expected_labels.expected_citations)
        missing = expected_citations - citation_ids
        passed = not missing
        details = (
            "All expected citations present"
            if passed
            else f"Missing expected citations: {sorted(missing)}"
        )
        return GraderResult(
            grader_name="citations_grader",
            passed=passed,
            score=1.0 if passed else 0.0,
            details=details,
        )


class WorkflowPhaseGrader:
    """Grades terminal workflow phase."""

    def grade(self, scenario: EvaluationScenario, result: ScenarioExecutionResult) -> GraderResult:
        expected = scenario.expected_labels.expected_phase
        if expected is None:
            return GraderResult(
                grader_name="workflow_phase_grader",
                passed=True,
                details="No expected phase specified",
            )

        passed = result.final_phase == expected
        details = (
            f"Final phase matches {expected.value}"
            if passed
            else f"Phase mismatch: expected {expected.value}, got {result.final_phase.value}"
        )
        return GraderResult(
            grader_name="workflow_phase_grader",
            passed=passed,
            score=1.0 if passed else 0.0,
            details=details,
        )


class ProhibitedClaimGrader:
    """Detects prohibited phrases, false claims, and secret/credential leakage in reports."""

    def grade(self, scenario: EvaluationScenario, result: ScenarioExecutionResult) -> GraderResult:
        text_corpus = self._extract_text(result)
        violations: list[str] = []

        # 1. Prohibited phrases check
        for phrase in scenario.prohibited_labels.prohibited_phrases:
            if phrase.lower() in text_corpus.lower():
                violations.append(f"Prohibited phrase detected: '{phrase}'")

        # 2. Secret / credential leakage check
        for pattern in _SECRET_PATTERNS:
            match = pattern.search(text_corpus)
            if match:
                prefix = match.group(0)[:8]
                violations.append(f"Secret or credential pattern matched: '{prefix}...'")

        passed = len(violations) == 0
        details = (
            "No prohibited claims or credentials detected"
            if passed
            else f"Found {len(violations)} violations"
        )
        return GraderResult(
            grader_name="prohibited_claim_grader",
            passed=passed,
            score=1.0 if passed else 0.0,
            details=details,
            prohibited_violations=tuple(violations),
        )

    def _extract_text(self, result: ScenarioExecutionResult) -> str:
        parts: list[str] = []
        if result.report_draft is not None:
            for section in result.report_draft.sections.as_list():
                parts.append(section.model_dump_json())
        if result.mcp_error is not None:
            parts.extend([result.mcp_error.message, result.mcp_error.remediation])
        return "\n".join(parts)


class CompositeDomainGrader:
    """Aggregates all deterministic domain graders into an immutable EvaluationRun."""

    def __init__(self, graders: tuple[DomainGrader, ...] | None = None) -> None:
        self.graders = graders or (
            EvidenceStateGrader(),
            OutcomeGrader(),
            RiskBandGrader(),
            ScoreThresholdGrader(),
            RiskFactorsGrader(),
            CitationsGrader(),
            WorkflowPhaseGrader(),
            ProhibitedClaimGrader(),
        )

    def evaluate(
        self, scenario: EvaluationScenario, result: ScenarioExecutionResult
    ) -> EvaluationRun:
        grader_results: list[GraderResult] = []
        for g in self.graders:
            res = g.grade(scenario, result)
            grader_results.append(res)

        overall_passed = all(r.passed for r in grader_results)
        return EvaluationRun(
            run_id=f"run-{scenario.scenario_id}-{scenario.version}",
            scenario_id=scenario.scenario_id,
            scenario_version=scenario.version,
            provenance=result.provenance,
            timestamp=result.timestamp,
            grader_results=tuple(grader_results),
            passed=overall_passed,
        )
