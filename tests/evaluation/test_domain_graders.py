"""Tests for deterministic domain graders and security checks."""

# story: e06s01

from datetime import UTC, datetime

import pytest

from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.evaluation.graders import (
    CitationsGrader,
    CompositeDomainGrader,
    EvidenceStateGrader,
    OutcomeGrader,
    ProhibitedClaimGrader,
    RiskBandGrader,
    ScoreThresholdGrader,
    WorkflowPhaseGrader,
)
from vehicle_risk_agent.evaluation.models import (
    EvaluationRun,
    EvaluationScenario,
    ExpectedEvaluationLabels,
    ProhibitedEvaluationLabels,
    ScenarioCategory,
)
from vehicle_risk_agent.evaluation.runner import ScenarioExecutionResult, ScenarioRunner
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.sufficiency import SufficiencyOutcome
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    RiskBand,
)


@pytest.fixture
def clean_scenario_and_result() -> tuple[EvaluationScenario, ScenarioExecutionResult]:
    now = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)
    rev = VehicleRevisionResponse(
        vin="7AT0BJ03X20000001",
        revision_id="rev-clean-01",
        revision_number=1,
        material_hash="a" * 64,
        canonical_fields={
            "make": "TOYOTA",
            "model": "COROLLA",
            "year": 2020,
            "ppsr_result": "NO_FINANCE_REGISTERED",
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        },
        field_provenance={},
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=95,
            band=ConfidenceBand.HIGH,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="Clean verified record",
        ),
        as_of=now,
        published_at=now,
    )
    citation = PolicyCitation(
        source_id="src-wof",
        snapshot_id="snap-wof-01",
        passage_id="sec-wof-clean",
        section_identifier="Section 3.1",
        heading="Roadworthiness Standards",
        source_title="Warrant of Fitness Overview",
        canonical_origin="NZTA Vehicle Inspection Requirements Manual",
    )
    scenario = EvaluationScenario(
        scenario_id="sc-clean-01",
        version="1.0.0",
        title="Clean Seed Vehicle Assessment",
        description="Verify clean vehicle results in LOW risk band with complete sufficiency.",
        category=ScenarioCategory.CLEAN,
        vin="7AT0BJ03X20000001",
        context=AssessmentContext(sale_type=SaleType.DEALER),
        mock_vehicle_revisions=(rev,),
        mock_citations=(citation,),
        expected_labels=ExpectedEvaluationLabels(
            sufficiency_outcome=SufficiencyOutcome.COMPLETE,
            assessment_outcome=AssessmentOutcome.SCORED,
            risk_band=RiskBand.LOW,
            min_risk_score=0.0,
            max_risk_score=20.0,
            expected_citations=("sec-wof-clean",),
            expected_phase=AssessmentRunPhase.COMPLETED,
        ),
        prohibited_labels=ProhibitedEvaluationLabels(
            prohibited_phrases=("GUARANTEED SAFE", "OFFICIAL POLICE CLEARANCE"),
            prohibited_outcomes=(AssessmentOutcome.FAILED,),
        ),
    )

    runner = ScenarioRunner(clock=lambda: now)
    # Synchronously run
    import asyncio

    exec_result = asyncio.run(runner.run_scenario(scenario))
    return scenario, exec_result


def test_individual_graders_on_clean_result(
    clean_scenario_and_result: tuple[EvaluationScenario, ScenarioExecutionResult],
) -> None:
    scenario, result = clean_scenario_and_result

    g_evidence = EvidenceStateGrader().grade(scenario, result)
    assert g_evidence.passed is True

    g_outcome = OutcomeGrader().grade(scenario, result)
    assert g_outcome.passed is True

    g_band = RiskBandGrader().grade(scenario, result)
    assert g_band.passed is True

    g_score = ScoreThresholdGrader().grade(scenario, result)
    assert g_score.passed is True

    g_citations = CitationsGrader().grade(scenario, result)
    assert g_citations.passed is True

    g_prohibited = ProhibitedClaimGrader().grade(scenario, result)
    assert g_prohibited.passed is True

    g_phase = WorkflowPhaseGrader().grade(scenario, result)
    assert g_phase.passed is True


def test_outcome_grader_detects_prohibited_and_mismatched_outcome(
    clean_scenario_and_result: tuple[EvaluationScenario, ScenarioExecutionResult],
) -> None:
    scenario, result = clean_scenario_and_result

    # Mutate scenario expectation to FAILED
    bad_scenario = scenario.model_copy(
        update={
            "expected_labels": scenario.expected_labels.model_copy(
                update={"assessment_outcome": AssessmentOutcome.FAILED}
            )
        }
    )
    res = OutcomeGrader().grade(bad_scenario, result)
    assert res.passed is False
    assert "Expected outcome FAILED" in res.details

    # Prohibited outcome check
    prohibited_scenario = scenario.model_copy(
        update={
            "prohibited_labels": scenario.prohibited_labels.model_copy(
                update={"prohibited_outcomes": (AssessmentOutcome.SCORED,)}
            )
        }
    )
    res2 = OutcomeGrader().grade(prohibited_scenario, result)
    assert res2.passed is False
    assert "Prohibited outcome SCORED was produced" in res2.details


def test_score_and_band_graders_fail_on_out_of_bounds(
    clean_scenario_and_result: tuple[EvaluationScenario, ScenarioExecutionResult],
) -> None:
    scenario, result = clean_scenario_and_result

    # Band mismatch
    bad_band_scenario = scenario.model_copy(
        update={
            "expected_labels": scenario.expected_labels.model_copy(
                update={"risk_band": RiskBand.HIGH}
            )
        }
    )
    res_band = RiskBandGrader().grade(bad_band_scenario, result)
    assert res_band.passed is False

    # Score threshold out of bounds
    bad_score_scenario = scenario.model_copy(
        update={
            "expected_labels": scenario.expected_labels.model_copy(
                update={"min_risk_score": 50.0, "max_risk_score": 100.0}
            )
        }
    )
    res_score = ScoreThresholdGrader().grade(bad_score_scenario, result)
    assert res_score.passed is False


def test_prohibited_claim_grader_detects_forbidden_text_and_secrets(
    clean_scenario_and_result: tuple[EvaluationScenario, ScenarioExecutionResult],
) -> None:
    scenario, result = clean_scenario_and_result
    assert result.report_draft is not None

    bad_summary = result.report_draft.sections.executive_summary.model_copy(
        update={
            "summary_text": (
                "This vehicle is GUARANTEED SAFE. "
                "Secret token: sk-ant-api03-abcdef1234567890abcdef1234567890"
            )
        }
    )
    bad_sections = result.report_draft.sections.model_copy(
        update={"executive_summary": bad_summary}
    )
    bad_draft = result.report_draft.model_copy(update={"sections": bad_sections})
    bad_result = result.model_copy(update={"report_draft": bad_draft})

    grader = ProhibitedClaimGrader()
    res = grader.grade(scenario, bad_result)
    assert res.passed is False
    assert len(res.prohibited_violations) >= 2
    assert any("GUARANTEED SAFE" in v for v in res.prohibited_violations)
    has_secret = any(
        "secret" in v.lower() or "credential" in v.lower() for v in res.prohibited_violations
    )
    assert has_secret


def test_composite_domain_grader_builds_evaluation_run(
    clean_scenario_and_result: tuple[EvaluationScenario, ScenarioExecutionResult],
) -> None:
    scenario, result = clean_scenario_and_result

    composite = CompositeDomainGrader()
    eval_run = composite.evaluate(scenario, result)

    assert isinstance(eval_run, EvaluationRun)
    assert eval_run.scenario_id == scenario.scenario_id
    assert eval_run.passed is True
    assert len(eval_run.grader_results) >= 5
    assert all(g.passed for g in eval_run.grader_results)
    assert len(eval_run.run_hash) == 64
