"""Configuration settings for vehicle-risk-agent."""

from pydantic_settings import BaseSettings, SettingsConfigDict


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
