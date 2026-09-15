"""Deterministic contracts for the bounded e11 live-investigation suite."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from vehicle_risk_agent.evaluation.investigation import (
    InvestigationScenario,
    grade_scenario,
    e11_scenarios,
)
from vehicle_risk_agent.investigation.models import InvestigationAction
from vehicle_risk_agent.evaluation.live import (
    LiveEvaluationBudgetError,
    LiveEvaluationConfig,
    LiveEvaluationCredentialsError,
    LiveEvaluationRunner,
)


def test_e11_scenarios_are_exactly_labelled() -> None:
    scenarios = e11_scenarios()
    assert [scenario.scenario_id for scenario in scenarios] == [
        "odometer-explanation",
        "required-evidence-policy",
        "identity-already-answered",
    ]
    assert scenarios[0].expected_action == "explain_vehicle_field"
    assert scenarios[0].expected_field == "odometer_reading"
    assert scenarios[1].expected_action == "search_policy"
    assert scenarios[2].expected_action == "NO_ACTION"
    assert scenarios[2].expected_dispatched is False


def test_investigation_grading_rejects_wrong_field_result() -> None:
    scenario = InvestigationScenario(
        scenario_id="conflict",
        intent="Explain the PPSR conflict.",
        evidence_targets=("ppsr_result",),
        expected_action="explain_vehicle_field",
        expected_field="ppsr_result",
    )
    result = SimpleNamespace(
        action=InvestigationAction.EXPLAIN_VEHICLE_FIELD,
        dispatched=True,
        completed=True,
        references=("obs-1",),
        policy_citations=(),
        evidence_result=SimpleNamespace(field_name="stolen_status"),
    )

    assert grade_scenario(scenario, result) is False


def test_investigation_grading_rejects_untyped_history_result() -> None:
    scenario = InvestigationScenario(
        scenario_id="history",
        intent="Find older revisions.",
        expected_action="get_vehicle_history",
    )
    result = SimpleNamespace(
        action=InvestigationAction.GET_VEHICLE_HISTORY,
        dispatched=True,
        completed=True,
        references=("rev-1",),
        policy_citations=(),
        evidence_result=None,
    )

    assert grade_scenario(scenario, result) is False


def test_e11_offline_control_is_blocked_and_redacted(tmp_path: Path) -> None:
    output = tmp_path / "e11-offline.json"
    runner = LiveEvaluationRunner(
        LiveEvaluationConfig(
            suite="e11-investigation",
            output_file=str(output),
            max_budget_usd=3.0,
            max_scenarios=3,
        )
    )

    import asyncio

    record = asyncio.run(runner.run_bounded_acceptance())
    assert record.execution_mode == "OFFLINE"
    assert record.release_verdict == "BLOCKED"
    assert record.verdict_passed is False
    assert record.suite_version == "e11-live-v1"
    assert output.exists()
    assert "ANTHROPIC" not in output.read_text()


def test_e11_requires_environment_credentials_and_endpoint() -> None:
    runner = LiveEvaluationRunner(
        LiveEvaluationConfig(
            enable_live_eval=True,
            suite="e11-investigation",
            api_key="",
            max_budget_usd=3.0,
        )
    )
    with pytest.raises(LiveEvaluationCredentialsError, match="ANTHROPIC_API_KEY"):
        runner.validate_readiness()


def test_e11_gemini_accepts_environment_credentials_and_pinned_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("MCP_SERVER_URL", "http://localhost:8080/mcp")
    runner = LiveEvaluationRunner(
        LiveEvaluationConfig(
            enable_live_eval=True,
            suite="e11-investigation",
            provider="gemini",
            model="gemini-3.1-flash-lite",
            max_budget_usd=3.0,
        ),
        active_corpus=type(
            "Corpus",
            (),
            {
                "lifecycle_state": "ACTIVE",
                "retrieval_config": type("Retrieval", (), {"profile": "neural"})(),
            },
        )(),
    )

    runner.validate_readiness()
    assert runner.pricing.model == "gemini-3.1-flash-lite"
    assert runner.pricing.input_token_cost_per_million == 0.25
    assert runner.pricing.output_token_cost_per_million == 1.50


def test_e11_gemini_rejects_cli_key_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    runner = LiveEvaluationRunner(
        LiveEvaluationConfig(
            enable_live_eval=True,
            suite="e11-investigation",
            provider="gemini",
            model="gemini-3.1-flash-lite",
            api_key="must-not-be-used",
        )
    )
    with pytest.raises(LiveEvaluationCredentialsError, match="GEMINI_API_KEY"):
        runner.validate_readiness()


def test_e11_gemini_rejects_mismatched_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("MCP_SERVER_URL", "http://localhost:8080/mcp")
    runner = LiveEvaluationRunner(
        LiveEvaluationConfig(
            enable_live_eval=True,
            suite="e11-investigation",
            provider="gemini",
            model="gemini-2.0-flash",
        ),
        active_corpus=type(
            "Corpus",
            (),
            {
                "lifecycle_state": "ACTIVE",
                "retrieval_config": type("Retrieval", (), {"profile": "neural"})(),
            },
        )(),
    )
    with pytest.raises(LiveEvaluationBudgetError, match="pinned"):
        runner.validate_readiness()


def test_non_e11_suites_reject_gemini_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    with pytest.raises(LiveEvaluationBudgetError, match="e11-investigation"):
        LiveEvaluationRunner(
            LiveEvaluationConfig(
                enable_live_eval=True,
                suite="general",
                provider="gemini",
                model="gemini-3.1-flash-lite",
            )
        ).validate_readiness()


def test_e11_rejects_mismatched_pricing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = LiveEvaluationRunner(
        LiveEvaluationConfig(
            enable_live_eval=True,
            suite="e11-investigation",
            max_budget_usd=3.0,
        ),
        active_corpus=type(
            "Corpus",
            (),
            {
                "lifecycle_state": "ACTIVE",
                "retrieval_config": type("Retrieval", (), {"profile": "neural"})(),
            },
        )(),
    )
    with pytest.raises(LiveEvaluationCredentialsError):
        runner.validate_readiness()
    monkeypatch.setenv("MCP_SERVER_URL", "http://localhost:8080/mcp")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    with pytest.raises(LiveEvaluationBudgetError, match="pinned"):
        LiveEvaluationRunner(
            LiveEvaluationConfig(
                enable_live_eval=True,
                suite="e11-investigation",
                max_budget_usd=3.0,
                model="other-model",
            ),
            active_corpus=runner.active_corpus,
        ).validate_readiness()
