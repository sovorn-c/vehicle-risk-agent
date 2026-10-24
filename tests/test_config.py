"""Tests for vehicle-risk-agent configuration and role Principal mapping."""

import pytest
from pydantic import SecretStr, ValidationError

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


def test_settings_gemini_api_key_field() -> None:
    """Settings must accept gemini_api_key without validation error."""
    settings = Settings(gemini_api_key=SecretStr("test-key-123"))
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "test-key-123"

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


def test_live_retrieval_uses_configured_model_revisions() -> None:
    """Policy ingestion and retrieval use the operator's model revision pins."""
    from vehicle_risk_agent.api.app import create_app

    app = create_app(
        Settings(
            retrieval_mode="live",
            embedding_revision="embedding-pinned-revision",
            reranker_revision="reranker-pinned-revision",
        )
    )
    assert app.state.embedding_adapter.revision == "embedding-pinned-revision"
    assert app.state.reranker_adapter.revision == "reranker-pinned-revision"


_AZURE_ENDPOINT = "https://oai-example.openai.azure.com"


def _azure_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "drafting_mode": "azure_openai",
        "enable_live_drafting": True,
        "azure_openai_endpoint": _AZURE_ENDPOINT,
        "azure_openai_deployment": "chat",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type,call-arg]


def test_azure_openai_mode_loads_with_endpoint_and_deployment() -> None:
    settings = _azure_settings()

    assert settings.drafting_mode == "azure_openai"
    assert settings.azure_openai_endpoint == _AZURE_ENDPOINT
    assert settings.azure_openai_deployment == "chat"
    assert settings.azure_client_id is None


def test_azure_openai_mode_requires_live_drafting_opt_in() -> None:
    with pytest.raises(ValidationError, match="ENABLE_LIVE_DRAFTING"):
        _azure_settings(enable_live_drafting=False)


@pytest.mark.parametrize("endpoint", [None, "", "   "])
def test_azure_openai_mode_requires_endpoint(endpoint: str | None) -> None:
    with pytest.raises(ValidationError, match="AZURE_OPENAI_ENDPOINT"):
        _azure_settings(azure_openai_endpoint=endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://oai-example.openai.azure.com",
        "https://user:pass@oai-example.openai.azure.com",
        "https://oai-example.openai.azure.com/?api-version=1",
        "https://oai-example.openai.azure.com/#frag",
        "https://oai-example.openai.azure.com/openai/v1",
        "oai-example.openai.azure.com",
    ],
)
def test_azure_openai_mode_rejects_unsafe_endpoint(endpoint: str) -> None:
    with pytest.raises(ValidationError, match="AZURE_OPENAI_ENDPOINT"):
        _azure_settings(azure_openai_endpoint=endpoint)


@pytest.mark.parametrize("deployment", [None, "", "  "])
def test_azure_openai_mode_requires_deployment(deployment: str | None) -> None:
    with pytest.raises(ValidationError, match="AZURE_OPENAI_DEPLOYMENT"):
        _azure_settings(azure_openai_deployment=deployment)


_PRODUCTION_VALUES: dict[str, object] = {
    "environment": "production",
    "mcp_server_url": "http://ca-mcp/mcp",
    "requester_token": "prod-requester-token-value",
    "reviewer_token": "prod-reviewer-token-value",
    "operator_token": "prod-operator-token-value",
    "maintainer_token": "prod-maintainer-token-value",
    "snapshot_integrity_secret": "prod-snapshot-integrity-secret-0123456789",
}


def test_azure_openai_mode_requires_client_id_in_production() -> None:
    with pytest.raises(ValidationError, match="AZURE_CLIENT_ID"):
        _azure_settings(**_PRODUCTION_VALUES)

    settings = _azure_settings(
        **_PRODUCTION_VALUES, azure_client_id="00000000-0000-0000-0000-000000000001"
    )
    assert settings.azure_client_id == "00000000-0000-0000-0000-000000000001"


def test_settings_have_no_azure_openai_api_key_field() -> None:
    with pytest.raises(ValidationError):
        _azure_settings(azure_openai_api_key="key")


# Mirrors vehicle-intelligence-platform terraform/stack/apps.tf local.agent_env (lines 23-37).
_TERRAFORM_AGENT_ENV = {
    "ENVIRONMENT": "production",
    "MCP_SERVER_URL": "http://ca-mcp/mcp",
    "DRAFTING_MODE": "azure_openai",
    "ENABLE_LIVE_DRAFTING": "true",
    "AZURE_CLIENT_ID": "00000000-0000-0000-0000-000000000001",
    "AZURE_OPENAI_ENDPOINT": "https://oai-example.openai.azure.com/",
    "AZURE_OPENAI_DEPLOYMENT": "chat",
    "DATABASE_URL": "postgresql+psycopg://agent:secret@psql.example:5432/agent",
    "REQUESTER_TOKEN": "kv-requester-token-value",
    "REVIEWER_TOKEN": "kv-reviewer-token-value",
    "OPERATOR_TOKEN": "kv-operator-token-value",
    "MAINTAINER_TOKEN": "kv-maintainer-token-value",
    "SNAPSHOT_INTEGRITY_SECRET": "kv-snapshot-integrity-secret-0123456789",
}


def test_settings_load_from_terraform_agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings load from exactly the env set ca-agent receives from Terraform.

    Source: vehicle-intelligence-platform terraform/stack/apps.tf local.agent_env,
    lines 23-37. A field production requires that Terraform does not set would crash
    ca-agent, job-agent-seed, and job-agent-postdeploy-eval at startup.
    """
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)
    for name, value in _TERRAFORM_AGENT_ENV.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.environment == "production"
    assert settings.drafting_mode == "azure_openai"
    assert settings.azure_openai_endpoint == "https://oai-example.openai.azure.com"
    assert settings.azure_openai_deployment == "chat"
    assert settings.mcp_server_url == "http://ca-mcp/mcp"
