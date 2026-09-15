"""Provider boundary tests for forced strict investigation proposals."""

import pytest

from vehicle_risk_agent.investigation.models import (
    InvestigationAction,
    InvestigationContext,
    InvestigationProposal,
    ProviderUsage,
    SearchPolicyProposal,
    proposal_json_schema,
)
from vehicle_risk_agent.investigation.protocol import FakeInvestigationProvider


@pytest.mark.asyncio
async def test_fake_provider_returns_typed_proposal_and_usage() -> None:
    provider = FakeInvestigationProvider(
        InvestigationProposal.from_tool_input(
            {
                "kind": "REQUEST",
                "action": "search_policy",
                "arguments": {"query": "required evidence"},
            }
        ),
        usage=ProviderUsage(input_tokens=10, output_tokens=5),
    )
    result = await provider.propose(InvestigationContext(questions=("question",)))
    assert isinstance(result.proposal, SearchPolicyProposal)
    assert result.proposal.action == InvestigationAction.SEARCH_POLICY
    assert result.usage.input_tokens == 10


def test_schema_has_exactly_five_strict_variants() -> None:
    schema = proposal_json_schema()
    assert len(schema["oneOf"]) == 5
    assert all(item["additionalProperties"] is False for item in schema["oneOf"])
    assert schema["oneOf"][0]["properties"]["kind"]["const"] == "NO_ACTION"
