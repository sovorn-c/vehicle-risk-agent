"""Tests for credential-gated live evaluation consent and configuration contract."""

# story: e06s04
# task: e06s04-t01

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from vehicle_risk_agent.evaluation.live import (
    LiveEvaluationBudgetError,
    LiveEvaluationConfig,
    LiveEvaluationConsentError,
    LiveEvaluationCorpusError,
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


def test_refuses_when_active_corpus_missing_by_default() -> None:
    """Runner strictly refuses live readiness when active neural corpus is missing by default."""
    config = LiveEvaluationConfig(
        enable_live_eval=True,
        api_key="sk-ant-test-valid-key",
    )
    runner = LiveEvaluationRunner(config=config)
    with pytest.raises(LiveEvaluationCorpusError, match="active neural policy corpus is required"):
        runner.validate_readiness()


def test_accepts_when_consent_api_key_and_neural_corpus_present() -> None:
    """Runner validates readiness when consent, key, and neural corpus are provided."""
    from types import SimpleNamespace

    valid_corpus = SimpleNamespace(
        lifecycle_state="ACTIVE",
        retrieval_config=SimpleNamespace(profile="neural"),
    )
    config = LiveEvaluationConfig(
        enable_live_eval=True,
        api_key="sk-ant-test-valid-key",
    )
    runner = LiveEvaluationRunner(config=config, active_corpus=valid_corpus)
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
    assert config.require_neural_corpus is True


def test_refuses_budget_exceeding_limit() -> None:
    """Runner refuses execution when configured budget exceeds $5.00 or is non-positive."""
    over_budget = LiveEvaluationConfig(
        enable_live_eval=True,
        api_key="sk-ant-test-key",
        max_budget_usd=5.01,
    )
    with pytest.raises(LiveEvaluationBudgetError, match=r"budget.*at most \$5\.00"):
        LiveEvaluationRunner(config=over_budget).validate_readiness()

    zero_budget = LiveEvaluationConfig(
        enable_live_eval=True,
        api_key="sk-ant-test-key",
        max_budget_usd=0.0,
    )
    with pytest.raises(LiveEvaluationBudgetError, match=r"budget.*must be positive"):
        LiveEvaluationRunner(config=zero_budget).validate_readiness()


def test_refuses_scenarios_exceeding_limit() -> None:
    """Runner refuses execution when max_scenarios exceeds 3 or is non-positive."""
    too_many = LiveEvaluationConfig(
        enable_live_eval=True,
        api_key="sk-ant-test-key",
        max_scenarios=4,
    )
    with pytest.raises(LiveEvaluationBudgetError, match="max scenarios.*between 1 and 3"):
        LiveEvaluationRunner(config=too_many).validate_readiness()

    zero_scenarios = LiveEvaluationConfig(
        enable_live_eval=True,
        api_key="sk-ant-test-key",
        max_scenarios=0,
    )
    with pytest.raises(LiveEvaluationBudgetError, match="max scenarios.*between 1 and 3"):
        LiveEvaluationRunner(config=zero_scenarios).validate_readiness()


def test_refuses_scenario_batch_exceeding_cap() -> None:
    """Runner refuses scenario batches larger than 3 or exceeding configured limit."""
    runner = LiveEvaluationRunner(
        config=LiveEvaluationConfig(
            enable_live_eval=True,
            api_key="sk-ant-test-key",
            max_scenarios=2,
        )
    )
    with pytest.raises(LiveEvaluationBudgetError, match="exceeds limit of 3"):
        runner.validate_scenarios(["s1", "s2", "s3", "s4"])

    with pytest.raises(LiveEvaluationBudgetError, match="exceeds configured max of 2"):
        runner.validate_scenarios(["s1", "s2", "s3"])

    # Within bounds does not raise
    runner.validate_scenarios(["s1", "s2"])


def test_refuses_when_active_corpus_missing() -> None:
    """Runner refuses readiness when neural corpus is missing by default."""
    # Omitted active_corpus with default config
    runner = LiveEvaluationRunner(
        config=LiveEvaluationConfig(enable_live_eval=True, api_key="sk-ant-test-key")
    )
    with pytest.raises(LiveEvaluationCorpusError, match="active neural policy corpus is required"):
        runner.validate_readiness()

    # Explicit None passed to validate_readiness
    with pytest.raises(LiveEvaluationCorpusError, match="active neural policy corpus is required"):
        runner.validate_readiness(active_corpus=None)

    # Config with explicit require_neural_corpus=False allows execution without corpus
    permissive_runner = LiveEvaluationRunner(
        config=LiveEvaluationConfig(
            enable_live_eval=True,
            api_key="sk-ant-test-key",
            require_neural_corpus=False,
        )
    )
    permissive_runner.validate_readiness()


def test_refuses_when_active_corpus_unready() -> None:
    """Runner refuses readiness when active corpus is in DRAFT or RETIRED state."""
    from types import SimpleNamespace

    draft_corpus = SimpleNamespace(
        lifecycle_state="DRAFT",
        retrieval_config=SimpleNamespace(profile="neural"),
    )
    runner = LiveEvaluationRunner(
        config=LiveEvaluationConfig(enable_live_eval=True, api_key="sk-ant-test-key"),
        active_corpus=draft_corpus,
    )
    with pytest.raises(LiveEvaluationCorpusError, match="must be ACTIVE or READY"):
        runner.validate_readiness()


def test_refuses_when_active_corpus_not_neural() -> None:
    """Runner refuses readiness when active corpus retrieval profile is not neural."""
    from types import SimpleNamespace

    offline_corpus = SimpleNamespace(
        lifecycle_state="ACTIVE",
        retrieval_config=SimpleNamespace(profile="offline"),
    )
    runner = LiveEvaluationRunner(
        config=LiveEvaluationConfig(enable_live_eval=True, api_key="sk-ant-test-key"),
        active_corpus=offline_corpus,
    )
    with pytest.raises(LiveEvaluationCorpusError, match=r"retrieval profile.*must be 'neural'"):
        runner.validate_readiness()


def test_accepts_when_all_live_prerequisites_valid() -> None:
    """Runner validates when consent, key, budget <= $5, <= 3 scenarios, and neural corpus valid."""
    from types import SimpleNamespace

    valid_corpus = SimpleNamespace(
        lifecycle_state="ACTIVE",
        retrieval_config=SimpleNamespace(profile="neural"),
    )
    runner = LiveEvaluationRunner(
        config=LiveEvaluationConfig(
            enable_live_eval=True,
            api_key="sk-ant-test-key",
            max_budget_usd=5.0,
            max_scenarios=3,
            require_neural_corpus=True,
        ),
        active_corpus=valid_corpus,
    )
    # Does not raise
    runner.validate_readiness()


def test_cli_main_offline_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI --offline executes deterministically without credentials and records OFFLINE mode."""

    from vehicle_risk_agent.evaluation.live import LiveEvaluationRecord, main

    out_file = Path(str(tmp_path)) / "offline_evidence.json"
    code = main(["--offline", "--output-file", str(out_file)])
    assert code == 0
    captured = capsys.readouterr()
    assert "deterministic control mode, not live evidence" in captured.out

    assert out_file.exists()
    record = LiveEvaluationRecord.load_from_file(out_file)
    assert record.execution_mode == "OFFLINE"


def test_cli_main_refuses_without_consent(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI exits with error code 1 when live consent is missing."""
    from vehicle_risk_agent.evaluation.live import main

    code = main([])
    assert code == 1
    captured = capsys.readouterr()
    assert "Live evaluation refused" in captured.err


def test_cli_main_refuses_when_active_corpus_missing(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI exits with error code 1 when active neural corpus is missing."""
    from vehicle_risk_agent.evaluation.live import main

    code = main(["--enable-live-eval", "--api-key", "sk-ant-test-key", "--dry-run"])
    assert code == 1
    captured = capsys.readouterr()
    assert "active neural policy corpus is required" in captured.err


def test_cli_main_dry_run_success(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI --dry-run validates credentials when require-neural-corpus is disabled."""
    from vehicle_risk_agent.evaluation.live import main

    code = main(
        [
            "--enable-live-eval",
            "--api-key",
            "sk-ant-test-key",
            "--no-require-neural-corpus",
            "--dry-run",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "validated successfully (dry run)" in captured.out


def test_cli_main_dry_run_with_active_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI --dry-run validates configuration successfully when active neural corpus is present."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from vehicle_risk_agent.evaluation.live import main

    valid_corpus = SimpleNamespace(
        lifecycle_state="ACTIVE",
        retrieval_config=SimpleNamespace(profile="neural"),
    )
    with patch(
        "vehicle_risk_agent.policy.corpus_lifecycle.CorpusLifecycleManager.get_active_corpus",
        new_callable=AsyncMock,
    ) as mock_get:
        mock_get.return_value = valid_corpus
        code = main(["--enable-live-eval", "--api-key", "sk-ant-test-key", "--dry-run"])
        assert code == 0
        captured = capsys.readouterr()
        assert "validated successfully (dry run)" in captured.out
