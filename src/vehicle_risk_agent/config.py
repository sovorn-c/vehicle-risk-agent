"""Configuration settings for vehicle-risk-agent."""

from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SNAPSHOT_INTEGRITY_SECRET = "dev-snapshot-integrity-secret-v1-32chars"


class Settings(BaseSettings):
    """Application configuration with strict validation."""

    model_config = SettingsConfigDict(
        extra="forbid",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"
    requester_token: str = "dev-requester-token"
    reviewer_token: str = "dev-reviewer-token"
    operator_token: str = "dev-operator-token"
    maintainer_token: str = "dev-maintainer-token"
    sse_heartbeat_interval_seconds: float = Field(default=15.0, gt=0)
    mcp_server_url: str | None = None
    mcp_timeout_seconds: float = Field(default=5.0, gt=0)
    mcp_max_retries: int = Field(default=3, ge=0, le=10)
    mcp_initial_backoff: float = Field(default=0.05, gt=0)
    snapshot_integrity_secret: SecretStr = SecretStr(DEFAULT_SNAPSHOT_INTEGRITY_SECRET)
    retrieval_mode: str = "offline"
    drafting_mode: str = "offline"
    drafting_model: str = "claude-sonnet-4-6"
    drafting_max_tokens: int = Field(default=2048, gt=0, le=4096)
    drafting_timeout_seconds: float = Field(default=30.0, gt=0)
    drafting_max_repairs: int = Field(default=1, ge=0, le=2)
    anthropic_api_key: SecretStr | None = None
    live_budget_usd: float = Field(default=5.0, gt=0)
    enable_live_drafting: bool = False

    @field_validator("retrieval_mode")
    @classmethod
    def validate_retrieval_mode(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"offline", "live"}:
            raise ValueError("retrieval_mode must be 'offline' or 'live'")
        return cleaned

    @field_validator("drafting_mode")
    @classmethod
    def validate_drafting_mode(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"offline", "live"}:
            raise ValueError("drafting_mode must be 'offline' or 'live'")
        return cleaned

    @field_validator("mcp_server_url")
    @classmethod
    def validate_mcp_server_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("mcp_server_url must not be blank")
        parsed = urlsplit(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("mcp_server_url must be an HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("mcp_server_url must not contain credentials")
        path = parsed.path.rstrip("/") or "/mcp"
        if path != "/mcp":
            raise ValueError("mcp_server_url must target the /mcp endpoint")
        return urlunsplit((parsed.scheme, parsed.netloc, "/mcp", parsed.query, ""))

    @field_validator("snapshot_integrity_secret")
    @classmethod
    def validate_snapshot_integrity_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError("snapshot_integrity_secret must contain at least 32 characters")
        return value

    @model_validator(mode="after")
    def validate_production_configuration(self) -> "Settings":
        if self.environment.lower() == "production":
            if not self.mcp_server_url:
                raise ValueError("mcp_server_url is required in production")
            if (
                self.snapshot_integrity_secret.get_secret_value()
                == DEFAULT_SNAPSHOT_INTEGRITY_SECRET
            ):
                raise ValueError("snapshot_integrity_secret must be overridden in production")
            if any(
                token.startswith("dev-")
                for token in (
                    self.requester_token,
                    self.reviewer_token,
                    self.operator_token,
                    self.maintainer_token,
                )
            ):
                raise ValueError("development bearer tokens are not allowed in production")
        if self.drafting_mode == "live":
            if not self.enable_live_drafting:
                raise ValueError(
                    "drafting_mode='live' requires explicit opt-in enable_live_drafting=True"
                )
            if not self.anthropic_api_key or not self.anthropic_api_key.get_secret_value().strip():
                raise ValueError("drafting_mode='live' requires anthropic_api_key")
        return self
