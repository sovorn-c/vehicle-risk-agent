"""Configuration settings for vehicle-risk-agent."""

from urllib.parse import urlsplit

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

    @field_validator("mcp_server_url")
    @classmethod
    def validate_mcp_server_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("mcp_server_url must not be blank")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("mcp_server_url must be an HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("mcp_server_url must not contain credentials")
        return value

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
        return self
