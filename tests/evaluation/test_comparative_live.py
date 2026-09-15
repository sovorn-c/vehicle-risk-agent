"""Contracts for e12 comparative live-suite limits and readiness."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vehicle_risk_agent.evaluation.live import (
    LiveEvaluationBudgetError,
    LiveEvaluationConfig,
    LiveEvaluationCredentialsError,
    LiveEvaluationRunner,
)


@pytest.fixture
def active_corpus() -> SimpleNamespace:
    return SimpleNamespace(
        lifecycle_state="ACTIVE",
        retrieval_config=SimpleNamespace(profile="neural"),
    )


def test_e12_readiness_uses_environment_only_key_and_local_caps(
    active_corpus: SimpleNamespace,
) -> None:
    config = LiveEvaluationConfig(
        suite="e12-comparative",
        enable_live_eval=True,
        api_key="not-allowed",
        max_budget_usd=15.0,
        max_scenarios=4,
    )
    runner = LiveEvaluationRunner(config=config, active_corpus=active_corpus)
    with pytest.raises(LiveEvaluationCredentialsError, match="environment"):
        runner.validate_readiness()

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test", "MCP_SERVER_URL": "http://mcp"}):
        LiveEvaluationRunner(
            config.model_copy(update={"api_key": None}), active_corpus=active_corpus
        ).validate_readiness()


def test_e12_defaults_to_gemini_and_allows_explicit_anthropic_alternate(
    active_corpus: SimpleNamespace,
) -> None:
    with patch.dict(
        "os.environ",
        {
            "GEMINI_API_KEY": "gemini-test",
            "ANTHROPIC_API_KEY": "anthropic-test",
            "MCP_SERVER_URL": "http://mcp",
        },
    ):
        gemini = LiveEvaluationRunner(
            LiveEvaluationConfig(
                suite="e12-comparative",
                enable_live_eval=True,
                provider="gemini",
                model="gemini-3.1-flash-lite",
                max_budget_usd=15.0,
                max_scenarios=4,
            ),
            active_corpus=active_corpus,
        )
        gemini.validate_readiness()
        assert gemini.pricing.input_token_cost_per_million == 0.25
        assert gemini.pricing.output_token_cost_per_million == 1.5

        anthropic = LiveEvaluationRunner(
            LiveEvaluationConfig(
                suite="e12-comparative",
                enable_live_eval=True,
                provider="anthropic",
                model="claude-sonnet-4-6",
                max_budget_usd=15.0,
                max_scenarios=4,
            ),
            active_corpus=active_corpus,
        )
        anthropic.validate_readiness()
        assert anthropic.pricing.input_token_cost_per_million == 3.0
        assert anthropic.pricing.output_token_cost_per_million == 15.0


def test_e12_rejects_caps_above_suite_limits(active_corpus: SimpleNamespace) -> None:
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test", "MCP_SERVER_URL": "http://mcp"}):
        with pytest.raises(LiveEvaluationBudgetError, match="15.00"):
            LiveEvaluationRunner(
                LiveEvaluationConfig(
                    suite="e12-comparative",
                    enable_live_eval=True,
                    max_budget_usd=15.01,
                    max_scenarios=4,
                ),
                active_corpus=active_corpus,
            ).validate_readiness()
        with pytest.raises(LiveEvaluationBudgetError, match="4"):
            LiveEvaluationRunner(
                LiveEvaluationConfig(
                    suite="e12-comparative",
                    enable_live_eval=True,
                    max_budget_usd=15.0,
                    max_scenarios=5,
                ),
                active_corpus=active_corpus,
            ).validate_readiness()
