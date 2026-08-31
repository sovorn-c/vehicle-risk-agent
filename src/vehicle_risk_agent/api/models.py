"""Pydantic models for assessment intake API boundaries."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vehicle_risk_agent.domain.vin import validate_vin


class SaleType(StrEnum):
    """Permitted sale context types."""

    DEALER = "DEALER"
    PRIVATE = "PRIVATE"
    AUCTION = "AUCTION"


class AssessmentContext(BaseModel):
    """Bounded assessment context information."""

    model_config = ConfigDict(extra="forbid")

    sale_type: SaleType
    intended_use: str | None = Field(default=None, max_length=500)
    questions: list[str] = Field(default_factory=list)

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions: list[str]) -> list[str]:
        """Enforce maximum of 5 questions and 200 characters per question."""
        if len(questions) > 5:
            raise ValueError("Assessment context permits at most 5 questions")
        for q in questions:
            if len(q) > 200:
                raise ValueError(
                    f"Each assessment question must be at most 200 characters (got {len(q)})"
                )
        return questions


class AssessmentCreateRequest(BaseModel):
    """Strict request payload for creating an Assessment."""

    model_config = ConfigDict(extra="forbid")

    vin: str
    context: AssessmentContext

    @field_validator("vin")
    @classmethod
    def validate_vin_field(cls, value: str) -> str:
        """Validate VIN against ISO 3779 checksum and forbidden character rules."""
        result = validate_vin(value)
        if not result.is_valid or result.normalized_vin is None:
            raise ValueError(result.error_reason or "Invalid VIN")
        return result.normalized_vin
