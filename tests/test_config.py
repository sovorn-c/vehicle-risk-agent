"""Tests for vehicle-risk-agent configuration and role Principal mapping."""

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.auth import Role, authenticate_bearer_token
from vehicle_risk_agent.config import Settings


def test_settings_load_defaults() -> None:
    """Verify default settings provide necessary local development values."""
    settings = Settings()
    assert settings.environment in ("development", "test", "production")
    assert settings.database_url.startswith("postgresql")
    assert settings.requester_token
    assert settings.reviewer_token
    assert settings.operator_token
    assert settings.maintainer_token


def test_settings_reject_extra_fields() -> None:
    """Verify strict settings reject unexpected configuration fields."""
    with pytest.raises(ValidationError):
        Settings(_extra_forbidden_key="unexpected")  # type: ignore[call-arg]


def test_principal_role_mapping_from_token() -> None:
    """Verify known local tokens map to principals with mutually exclusive roles."""
    settings = Settings()

    requester = authenticate_bearer_token(settings.requester_token, settings)
    assert requester is not None
    assert requester.role == Role.REQUESTER
    assert isinstance(requester.principal_id, str)
    assert len(requester.principal_id) > 0

    reviewer = authenticate_bearer_token(settings.reviewer_token, settings)
    assert reviewer is not None
    assert reviewer.role == Role.REVIEWER

    operator = authenticate_bearer_token(settings.operator_token, settings)
    assert operator is not None
    assert operator.role == Role.TECHNICAL_OPERATOR

    maintainer = authenticate_bearer_token(settings.maintainer_token, settings)
    assert maintainer is not None
    assert maintainer.role == Role.POLICY_CORPUS_MAINTAINER


def test_principal_role_mapping_unknown_token() -> None:
    """Verify unknown or invalid token returns None (fails closed)."""
    settings = Settings()
    assert authenticate_bearer_token("invalid-secret-token", settings) is None
    assert authenticate_bearer_token("", settings) is None


def test_mcp_server_url_normalizes_to_mcp_endpoint() -> None:
    """Bare base URL must normalize to the streamable /mcp endpoint path."""
    settings = Settings(mcp_server_url="http://localhost:8080")
    assert settings.mcp_server_url == "http://localhost:8080/mcp"

    settings_already_mcp = Settings(mcp_server_url="http://localhost:8080/mcp")
    assert settings_already_mcp.mcp_server_url == "http://localhost:8080/mcp"


def test_settings_drafting_configuration_defaults() -> None:
    """Settings provides safe offline defaults and claude-sonnet-4-6 provider config."""
    settings = Settings()
    assert settings.drafting_mode == "offline"
    assert settings.drafting_model == "claude-sonnet-4-6"
    assert settings.drafting_max_tokens == 2048
    assert settings.drafting_timeout_seconds == 30.0
    assert settings.drafting_max_repairs == 1
    assert settings.live_budget_usd == 5.0
    assert settings.enable_live_drafting is False


def test_settings_drafting_mode_live_validation() -> None:
    """Live drafting requires explicit opt-in and an API key; invalid modes are rejected."""
    from pydantic import SecretStr

    # Invalid mode
    with pytest.raises(ValidationError):
        Settings(drafting_mode="unsupported_mode")

    # Live mode without opt-in
    with pytest.raises(ValidationError, match="requires explicit opt-in"):
        Settings(drafting_mode="live")

    # Live mode with opt-in but no API key
    with pytest.raises(ValidationError, match="requires anthropic_api_key"):
        Settings(drafting_mode="live", enable_live_drafting=True)

    # Valid live configuration
    valid_live = Settings(
        drafting_mode="live",
        enable_live_drafting=True,
        anthropic_api_key=SecretStr("sk-ant-api03-test-valid-key"),
    )
    assert valid_live.drafting_mode == "live"
    assert valid_live.enable_live_drafting is True
