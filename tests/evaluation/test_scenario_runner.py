"""Tests for ScenarioRunner running assessments through fake adapters with pinned clock."""

# story: e06s01

from datetime import UTC, datetime

import pytest

from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.evaluation.models import (
    EvaluationScenario,
    ExpectedEvaluationLabels,
    ProhibitedEvaluationLabels,
    ScenarioCategory,
)
from vehicle_risk_agent.evaluation.runner import ScenarioExecutionResult, ScenarioRunner
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    SafeError,
    SafeErrorCategory,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.sufficiency import SufficiencyOutcome
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.risk.models import AssessmentOutcome, RiskBand


@pytest.fixture
def clean_scenario() -> EvaluationScenario:
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
        source_id="src-warrant-of-fitness",
        snapshot_id="snap-wof-01",
        passage_id="sec-wof-clean",
        section_identifier="Section 3.1",
        heading="Roadworthiness Standards",
        source_title="Warrant of Fitness Overview",
        canonical_origin="NZTA Vehicle Inspection Requirements Manual",
    )
    return EvaluationScenario(
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
            max_risk_score=35.0,
        ),
        prohibited_labels=ProhibitedEvaluationLabels(
            prohibited_phrases=("GUARANTEED SAFE",),
            prohibited_outcomes=(AssessmentOutcome.FAILED,),
        ),
    )


@pytest.fixture
def incomplete_scenario() -> EvaluationScenario:
    now = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)
    rev = VehicleRevisionResponse(
        vin="7AT0BJ03X20000002",
        revision_id="rev-inc-01",
        revision_number=1,
        material_hash="b" * 64,
        canonical_fields={
            "make": "NISSAN",
            "model": "LEAF",
            "year": 2018,
            "ppsr_result": "NO_FINANCE_REGISTERED",
        },
        field_provenance={},
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=50,
            band=ConfidenceBand.MEDIUM,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="Partial record",
        ),
        as_of=now,
        published_at=now,
    )
    return EvaluationScenario(
        scenario_id="sc-inc-01",
        version="1.0.0",
        title="Incomplete Vehicle Assessment",
        description="Missing required evidence results in INCOMPLETE outcome without score.",
        category=ScenarioCategory.INCOMPLETE,
        vin="7AT0BJ03X20000002",
        context=AssessmentContext(sale_type=SaleType.PRIVATE),
        mock_vehicle_revisions=(rev,),
        expected_labels=ExpectedEvaluationLabels(
            sufficiency_outcome=SufficiencyOutcome.INCOMPLETE,
            assessment_outcome=AssessmentOutcome.INCOMPLETE,
        ),
        prohibited_labels=ProhibitedEvaluationLabels(
            prohibited_outcomes=(AssessmentOutcome.SCORED,),
        ),
    )


@pytest.fixture
def failed_mcp_scenario() -> EvaluationScenario:
    return EvaluationScenario(
        scenario_id="sc-fail-01",
        version="1.0.0",
        title="Pipeline Failure Scenario",
        description="Pipeline failure routes to FAILED phase safely.",
        category=ScenarioCategory.ADVERSARIAL_TIMEOUT,
        vin="7AT0BJ03X20000003",
        context=AssessmentContext(sale_type=SaleType.DEALER),
        mock_mcp_error=SafeError(
            category=SafeErrorCategory.PIPELINE_TIMEOUT,
            message="The vehicle intelligence service timed out.",
            retryable=True,
            remediation="Retry assessment intake after upstream pipeline recovers.",
        ),
        expected_labels=ExpectedEvaluationLabels(
            expected_phase=AssessmentRunPhase.FAILED,
            assessment_outcome=AssessmentOutcome.FAILED,
        ),
        prohibited_labels=ProhibitedEvaluationLabels(
            prohibited_outcomes=(AssessmentOutcome.SCORED,),
        ),
    )


@pytest.mark.asyncio
async def test_scenario_runner_clean_execution(clean_scenario: EvaluationScenario) -> None:
    pinned_clock = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)
    runner = ScenarioRunner(clock=lambda: pinned_clock)

    result = await runner.run_scenario(clean_scenario)
    assert isinstance(result, ScenarioExecutionResult)
    assert result.scenario.scenario_id == "sc-clean-01"
    assert result.final_phase == AssessmentRunPhase.COMPLETED
    assert result.sufficiency_result is not None
    assert result.sufficiency_result.outcome == SufficiencyOutcome.COMPLETE
    assert result.risk_result is not None
    assert result.risk_result.band == RiskBand.LOW
    assert result.risk_result.outcome == AssessmentOutcome.SCORED
    assert result.report_draft is not None
    assert len(result.policy_citations) == 1
    assert result.policy_citations[0].passage_id == "sec-wof-clean"
    assert result.provenance.provider == "fake"
    assert result.timestamp == pinned_clock


@pytest.mark.asyncio
async def test_scenario_runner_incomplete_execution(
    incomplete_scenario: EvaluationScenario,
) -> None:
    pinned_clock = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)
    runner = ScenarioRunner(clock=lambda: pinned_clock)

    result = await runner.run_scenario(incomplete_scenario)
    assert result.final_phase == AssessmentRunPhase.INCOMPLETE
    assert result.sufficiency_result is not None
    assert result.sufficiency_result.outcome == SufficiencyOutcome.INCOMPLETE
    assert result.risk_result is not None
    assert result.risk_result.outcome == AssessmentOutcome.INCOMPLETE


@pytest.mark.asyncio
async def test_scenario_runner_failed_execution(failed_mcp_scenario: EvaluationScenario) -> None:
    pinned_clock = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)
    runner = ScenarioRunner(clock=lambda: pinned_clock)

    result = await runner.run_scenario(failed_mcp_scenario)
    assert result.final_phase == AssessmentRunPhase.FAILED
    assert result.mcp_error is not None
    assert result.mcp_error.category == SafeErrorCategory.PIPELINE_TIMEOUT


@pytest.mark.asyncio
async def test_scenario_runner_pinned_determinism(clean_scenario: EvaluationScenario) -> None:
    pinned_clock = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)
    runner = ScenarioRunner(clock=lambda: pinned_clock)

    res1 = await runner.run_scenario(clean_scenario)
    res2 = await runner.run_scenario(clean_scenario)

    assert res1.final_phase == res2.final_phase
    assert res1.risk_result is not None
    assert res2.risk_result is not None
    assert res1.risk_result.score == res2.risk_result.score
    assert res1.risk_result.calculation_hash == res2.risk_result.calculation_hash
