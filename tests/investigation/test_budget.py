"""Final bounded investigation budget contracts."""

from decimal import Decimal

from vehicle_risk_agent.investigation.budget import InvestigationLimits


def test_final_investigation_limits_are_monotonic_and_priced() -> None:
    limits = InvestigationLimits.final()
    assert limits.max_proposal_rounds == 2
    assert limits.max_supplementary_attempts == 3
    assert limits.max_supplementary_retries == 2
    assert limits.max_input_tokens == 3072
    assert limits.max_output_tokens == 512
    assert limits.max_model_tokens == 7168
    assert limits.max_duration_seconds == 30
    assert limits.max_cost_usd == Decimal("1.00")
    assert limits.projected_cost(3072, 512) == Decimal("0.016896")
    assert limits.within_cost(Decimal("1.00"))
    assert not limits.within_cost(Decimal("1.01"))
