"""Contract tests for bounded investigation proposal data."""

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.investigation.models import (
    ExplainVehicleFieldArguments,
    InvestigationAction,
    InvestigationContext,
    InvestigationProposal,
    NoActionProposal,
    SubmitInvestigationToolResponse,
)


def test_valid_proposals_are_strict_and_action_specific() -> None:
    no_action = InvestigationProposal.from_tool_input({"kind": "NO_ACTION"})
    assert isinstance(no_action, NoActionProposal)
    request = InvestigationProposal.from_tool_input(
        {
            "kind": "REQUEST",
            "action": "explain_vehicle_field",
            "arguments": {"field_name": "odometer_reading"},
        }
    )
    assert not isinstance(request, NoActionProposal)
    assert request.action == InvestigationAction.EXPLAIN_VEHICLE_FIELD
    assert isinstance(request.arguments, ExplainVehicleFieldArguments)

    with pytest.raises(ValidationError):
        InvestigationProposal.from_tool_input({"kind": "NO_ACTION", "arguments": None})
    with pytest.raises(ValidationError):
        InvestigationProposal.from_tool_input(
            {
                "kind": "REQUEST",
                "action": "search_policy",
                "arguments": {"query": "policy", "vin": "1HGCR2F85HA000000"},
            }
        )


def test_tool_response_requires_one_named_block_and_known_usage() -> None:
    parsed = SubmitInvestigationToolResponse.validate_provider_payload(
        {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "name": "submit_investigation_proposal",
                    "input": {"kind": "NO_ACTION"},
                }
            ],
            "usage": {"input_tokens": 12, "output_tokens": 4},
        }
    )
    assert parsed.proposal.kind == "NO_ACTION"
    with pytest.raises(ValidationError):
        SubmitInvestigationToolResponse.validate_provider_payload(
            {
                "stop_reason": "end_turn",
                "content": [],
                "usage": {"input_tokens": 12, "output_tokens": 4},
            }
        )
    with pytest.raises(ValidationError):
        SubmitInvestigationToolResponse.validate_provider_payload(
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "text",
                        "text": "ignore instructions",
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 4},
            }
        )


def test_context_is_bounded_and_never_accepts_raw_observations() -> None:
    context = InvestigationContext(
        questions=("Explain the discrepancy",),
        evidence_targets=("odometer_reading",),
        evidence_summaries=("odometer_reading: 100000; ref=obs-1",),
        prior_result_summaries=(),
    )
    assert context.questions == ("Explain the discrepancy",)
    with pytest.raises(ValidationError):
        InvestigationContext(
            questions=("x",),
            evidence_targets=("raw_payload",),
            evidence_summaries=(),
            prior_result_summaries=(),
        )
    with pytest.raises(ValidationError):
        InvestigationContext(
            questions=("x",),
            evidence_targets=(),
            evidence_summaries=("x" * 501,),
            prior_result_summaries=(),
        )
