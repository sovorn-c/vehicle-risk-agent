"""Tests for credential-gated live evaluation consent and configuration contract."""

# story: e06s04
# task: e06s04-t01

import os
from unittest.mock import patch

import pytest

from vehicle_risk_agent.evaluation.live import (
    LiveEvaluationConfig,
    LiveEvaluationConsentError,
    LiveEvaluationCredentialsError,
    LiveEvaluationRunner,
)


def test_refuses_without_explicit_opt_in() -> None:
    """Runner strictly refuses execution without explicit provider opt-in consent."""
    config = LiveEvaluationConfig(
        enable_live_eval=False,
        api_key="mock-key-123",
    )
    runner = LiveEvaluationRunner(config=config)

    with pytest.raises(LiveEvaluationConsentError, match="consent.*required"):
        runner.validate_readiness()


def test_refuses_without_api_key() -> None:
    """Runner strictly refuses execution when Anthropic API key is missing or empty."""
    with patch.dict(os.environ, {}, clear=True):
        config = LiveEvaluationConfig(
            enable_live_eval=True,
            api_key="",
        )
        runner = LiveEvaluationRunner(config=config)

        with pytest.raises(LiveEvaluationCredentialsError, match="ANTHROPIC_API_KEY"):
            runner.validate_readiness()


def test_accepts_when_consent_and_api_key_present() -> None:
    """Runner validates readiness successfully when consent and credentials are provided."""
    config = LiveEvaluationConfig(
        enable_live_eval=True,
        api_key="sk-ant-test-valid-key",
    )
    runner = LiveEvaluationRunner(config=config)
    # Does not raise
    runner.validate_readiness()


def test_security_isolation_no_key_leaked_in_repr_or_errors() -> None:
    """Credentials are never exposed in exceptions, repr, str, or serialization."""
    sensitive_key = "sk-ant-api03-super-secret-production-key"
    config = LiveEvaluationConfig(
        enable_live_eval=True,
        api_key=sensitive_key,
    )
    runner = LiveEvaluationRunner(config=config)

    assert sensitive_key not in repr(config)
    assert sensitive_key not in str(config)
    assert sensitive_key not in repr(runner)
    assert sensitive_key not in str(runner)

    dump = config.model_dump_json()
    assert sensitive_key not in dump


def test_live_evaluation_config_provider_bounds() -> None:
    """Config enforces claude-sonnet-4-6 default, 2048 token cap, 30s timeout, 1 repair."""
    config = LiveEvaluationConfig()
    assert config.model == "claude-sonnet-4-6"
    assert config.max_output_tokens == 2048
    assert config.timeout_seconds == 30.0
    assert config.max_repairs == 1
    assert config.max_budget_usd == 5.0
