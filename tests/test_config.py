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
