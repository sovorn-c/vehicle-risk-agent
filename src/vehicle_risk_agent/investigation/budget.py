"""Hard limits and deterministic budget calculations for investigation."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class InvestigationLimits(BaseModel):
    """Bounded, persisted limits for one assessment run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_duration_seconds: int = Field(ge=1, le=900, default=300)
    max_proposal_rounds: int = Field(ge=0, le=5, default=1)
    max_supplementary_attempts: int = Field(ge=0, le=10, default=3)
    max_supplementary_retries: int = Field(ge=0, le=3, default=0)
    max_input_tokens: int = Field(ge=1, le=100_000, default=3072)
    max_output_tokens: int = Field(ge=1, le=20_000, default=512)
    max_cost_usd: Decimal = Field(gt=Decimal("0"), le=Decimal("100"), default=Decimal("0.20"))

    @classmethod
    def first_slice(cls) -> InvestigationLimits:
        return cls(
            max_duration_seconds=300,
            max_proposal_rounds=1,
            max_supplementary_attempts=3,
            max_supplementary_retries=0,
            max_input_tokens=3072,
            max_output_tokens=512,
            max_cost_usd=Decimal("0.20"),
        )

    def projected_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Estimate standard Sonnet pricing for a bounded reservation."""
        return (
            Decimal(input_tokens) * Decimal("3") + Decimal(output_tokens) * Decimal("15")
        ) / Decimal(1_000_000)

    def within_cost(self, cost: Decimal) -> bool:
        return cost <= self.max_cost_usd
