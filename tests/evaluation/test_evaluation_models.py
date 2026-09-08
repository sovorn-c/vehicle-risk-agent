"""Tests for versioned Evaluation Scenario, Run, Grader Result, and threshold contracts."""

# story: e06s01

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.evaluation.models import (
    EvaluationProvenance,
    EvaluationRun,
    EvaluationScenario,
    EvaluationThresholds,
    ExpectedEvaluationLabels,
    GraderResult,
    ProhibitedEvaluationLabels,
    ScenarioCategory,
    compute_provenance_hash,
)
from vehicle_risk_agent.evidence.sufficiency import SufficiencyOutcome
from vehicle_risk_agent.risk.models import AssessmentOutcome, RiskBand


def test_evaluation_provenance_immutability_and_hash() -> None:
    prov = EvaluationProvenance(
        scenario_version="1.0.0",
        corpus_version="corpus-2026.1",
        risk_policy_version="risk-policy-v1",
        provider="fake",
        prompt_version="prompt-v1",
        grader_version="grader-v1",
        code_version="0.1.0",
    )
    # Immutable
    with pytest.raises(ValidationError):
        prov.scenario_version = "2.0.0"

    # Forbid extra fields
    with pytest.raises(ValidationError):
        EvaluationProvenance(
            scenario_version="1.0.0",
            corpus_version="corpus-2026.1",
            risk_policy_version="risk-policy-v1",
            provider="fake",
            prompt_version="prompt-v1",
            grader_version="grader-v1",
            code_version="0.1.0",
            unexpected_field="disallowed",  # type: ignore[call-arg]
        )

    # Hash determinism
    h1 = prov.compute_hash()
    h2 = compute_provenance_hash(prov)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest


def test_evaluation_thresholds_defaults_and_validation() -> None:
    thresholds = EvaluationThresholds()
    assert thresholds.min_recall == 0.90
    assert thresholds.min_precision == 0.70
    assert thresholds.min_mrr == 0.80
    assert thresholds.min_abstention_accuracy == 0.95
    assert thresholds.min_citation_grounding == 1.00
    assert thresholds.max_p95_latency_ms == 30000.0

    # Custom valid thresholds
    custom = EvaluationThresholds(min_recall=0.95, min_precision=0.85)
    assert custom.min_recall == 0.95
    assert custom.min_precision == 0.85

    # Boundary invalidations
    with pytest.raises(ValidationError):
        EvaluationThresholds(min_recall=1.5)
    with pytest.raises(ValidationError):
        EvaluationThresholds(min_recall=-0.1)


def test_evaluation_scenario_creation_and_rejection() -> None:
    context = AssessmentContext(
        sale_type=SaleType.DEALER,
        intended_use="Personal family commuting",
        questions=["Are there any safety recalls?"],
    )
    scenario = EvaluationScenario(
        scenario_id="sc-clean-01",
        version="1.0.0",
        title="Clean Seed Vehicle Assessment",
        description="Verify clean vehicle results in LOW risk band with complete sufficiency.",
        category=ScenarioCategory.CLEAN,
        vin="7AT0BJ03X20000001",
        context=context,
        expected_labels=ExpectedEvaluationLabels(
            sufficiency_outcome=SufficiencyOutcome.COMPLETE,
            assessment_outcome=AssessmentOutcome.SCORED,
            risk_band=RiskBand.LOW,
            min_risk_score=0.0,
            max_risk_score=35.0,
        ),
        prohibited_labels=ProhibitedEvaluationLabels(
            prohibited_phrases=("GUARANTEED SAFE", "OFFICIAL POLICE CLEARANCE"),
            prohibited_outcomes=(AssessmentOutcome.FAILED,),
        ),
    )
    assert scenario.scenario_id == "sc-clean-01"
    assert scenario.category == ScenarioCategory.CLEAN
    assert scenario.expected_labels.risk_band == RiskBand.LOW

    # Reject extra fields
    with pytest.raises(ValidationError):
        EvaluationScenario(
            scenario_id="sc-clean-01",
            version="1.0.0",
            title="Clean",
            description="Desc",
            category=ScenarioCategory.CLEAN,
            vin="7AT0BJ03X20000001",
            context=context,
            expected_labels=ExpectedEvaluationLabels(),
            prohibited_labels=ProhibitedEvaluationLabels(),
            unknown_arg="invalid",  # type: ignore[call-arg]
        )


def test_grader_result_and_evaluation_run() -> None:
    prov = EvaluationProvenance(
        scenario_version="1.0.0",
        corpus_version="corpus-2026.1",
        risk_policy_version="risk-policy-v1",
        provider="fake",
        prompt_version="prompt-v1",
        grader_version="grader-v1",
        code_version="0.1.0",
    )
    grader1 = GraderResult(
        grader_name="sufficiency_grader",
        passed=True,
        score=1.0,
        details="Sufficiency matches expected COMPLETE",
    )
    grader2 = GraderResult(
        grader_name="prohibited_claim_grader",
        passed=True,
        score=1.0,
        details="No prohibited phrases found",
    )

    now = datetime.now(UTC)
    run = EvaluationRun(
        run_id="run-eval-001",
        scenario_id="sc-clean-01",
        scenario_version="1.0.0",
        provenance=prov,
        timestamp=now,
        grader_results=(grader1, grader2),
        passed=True,
    )
    assert run.passed is True
    assert len(run.grader_results) == 2
    assert run.run_hash != ""
    assert len(run.run_hash) == 64

    # Run is frozen
    with pytest.raises(ValidationError):
        run.passed = False
