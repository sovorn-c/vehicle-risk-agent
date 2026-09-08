"""Tests for 30+ labelled normal and adversarial evaluation scenarios and meta-gates."""

# story: e06s02

from datetime import UTC, datetime

import pytest

from vehicle_risk_agent.evaluation.graders import CompositeDomainGrader
from vehicle_risk_agent.evaluation.matrix import get_evaluation_matrix
from vehicle_risk_agent.evaluation.models import EvaluationScenario, ScenarioCategory
from vehicle_risk_agent.evaluation.runner import ScenarioRunner


@pytest.mark.asyncio
async def test_labelled_matrix_domain_execution() -> None:
    """Verify all domain scenarios pass."""
    matrix = get_evaluation_matrix()
    domain_categories = {
        ScenarioCategory.CLEAN,
        ScenarioCategory.RISKY,
        ScenarioCategory.INCOMPLETE,
        ScenarioCategory.CONFLICT,
        ScenarioCategory.TEMPORAL,
        ScenarioCategory.POLICY_ABSTENTION,
    }
    domain_scenarios = [s for s in matrix if s.category in domain_categories]
    assert len(domain_scenarios) >= 15

    pinned_clock = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)
    runner = ScenarioRunner(clock=lambda: pinned_clock)
    grader = CompositeDomainGrader()

    for sc in domain_scenarios:
        exec_res = await runner.run_scenario(sc)
        eval_run = grader.evaluate(sc, exec_res)
        failed_graders = [g for g in eval_run.grader_results if not g.passed]
        assert eval_run.passed is True, f"Scenario {sc.scenario_id} failed: {failed_graders}"


@pytest.mark.asyncio
async def test_labelled_matrix_adversarial_execution() -> None:
    """Verify adversarial scenarios pass."""
    matrix = get_evaluation_matrix()
    adversarial_categories = {
        ScenarioCategory.ADVERSARIAL_AUTH,
        ScenarioCategory.ADVERSARIAL_IDEMPOTENCY,
        ScenarioCategory.ADVERSARIAL_CONCURRENCY,
        ScenarioCategory.ADVERSARIAL_TIMEOUT,
        ScenarioCategory.ADVERSARIAL_CONTRACT_DRIFT,
        ScenarioCategory.ADVERSARIAL_PROMPT_INJECTION,
        ScenarioCategory.ADVERSARIAL_DATA_LEAKAGE,
    }
    adversarial_scenarios = [s for s in matrix if s.category in adversarial_categories]
    assert len(adversarial_scenarios) >= 14

    pinned_clock = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)
    runner = ScenarioRunner(clock=lambda: pinned_clock)
    grader = CompositeDomainGrader()

    for sc in adversarial_scenarios:
        exec_res = await runner.run_scenario(sc)
        eval_run = grader.evaluate(sc, exec_res)
        failed_graders = [g for g in eval_run.grader_results if not g.passed]
        assert eval_run.passed is True, f"Adversarial {sc.scenario_id} failed: {failed_graders}"


def test_labelled_matrix_meta_completeness_and_uniqueness() -> None:
    """Verify matrix completeness: at least 30 scenarios, unique IDs, versioned, category minima."""
    matrix = get_evaluation_matrix()
    assert len(matrix) >= 30, f"Expected at least 30 scenarios, got {len(matrix)}"

    # Unique scenario IDs
    scenario_ids = [s.scenario_id for s in matrix]
    assert len(scenario_ids) == len(set(scenario_ids)), "Duplicate scenario_id detected"

    # Unique (scenario_id, version) pairs
    id_version_pairs = [(s.scenario_id, s.version) for s in matrix]
    assert len(id_version_pairs) == len(set(id_version_pairs)), "Duplicate (id, version) detected"

    # Category coverage
    categories_present = {s.category for s in matrix}
    all_categories = set(ScenarioCategory)
    missing_cats = all_categories - categories_present
    assert categories_present == all_categories, f"Missing categories: {missing_cats}"

    # Category minima: at least 2 scenarios per category
    for cat in ScenarioCategory:
        count = sum(1 for s in matrix if s.category == cat)
        assert count >= 2, f"Category {cat.value} has only {count} scenarios (minimum 2)"

    # Validate every scenario structure
    for sc in matrix:
        assert isinstance(sc, EvaluationScenario)
        assert len(sc.scenario_id) > 0
        assert len(sc.version) > 0
        assert len(sc.title) > 0
        assert len(sc.description) > 0
        assert len(sc.vin) == 17
        assert sc.context is not None
        assert sc.expected_labels is not None
        assert sc.prohibited_labels is not None


@pytest.mark.asyncio
async def test_labelled_matrix_meta_repeatability() -> None:
    """Verify deterministic repeatability across multiple full runs."""
    matrix = get_evaluation_matrix()
    sample = matrix[:5]

    pinned_clock = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)
    runner = ScenarioRunner(clock=lambda: pinned_clock)
    grader = CompositeDomainGrader()

    runs_1 = [grader.evaluate(s, await runner.run_scenario(s)) for s in sample]
    runs_2 = [grader.evaluate(s, await runner.run_scenario(s)) for s in sample]

    for r1, r2 in zip(runs_1, runs_2, strict=True):
        assert r1.scenario_id == r2.scenario_id
        assert r1.passed == r2.passed
        assert r1.run_hash == r2.run_hash
