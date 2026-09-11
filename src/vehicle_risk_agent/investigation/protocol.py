"""Provider protocol and deterministic fake for bounded investigation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from vehicle_risk_agent.investigation.models import (
    InvestigationContext,
    InvestigationProposal,
    ProposalValue,
    ProviderProposalResult,
    ProviderUsage,
)


class InvestigationProvider(Protocol):
    """Narrow boundary used by the graph; providers cannot execute actions."""

    async def propose(
        self, context: InvestigationContext, timeout_seconds: float = 30.0
    ) -> ProviderProposalResult:
        """Return one strictly validated proposal and provider usage."""
        ...

    async def count_input_tokens(self, context: InvestigationContext) -> int:
        """Count provider input tokens before a paid proposal call."""
        ...


class FakeInvestigationProvider:
    """Deterministic provider used by offline tests and workflow fixtures."""

    def __init__(
        self,
        proposal: ProposalValue | InvestigationProposal,
        usage: ProviderUsage | None = None,
        proposals: Iterable[ProposalValue] | None = None,
    ) -> None:
        if isinstance(proposal, InvestigationProposal):
            proposal = InvestigationProposal.from_tool_input(proposal.model_dump())
        self._proposals = list(proposals or ())
        self._proposals.insert(0, proposal)
        self._usage = usage or ProviderUsage(input_tokens=1, output_tokens=1)
        self.calls = 0
        self.contexts: list[InvestigationContext] = []

    async def propose(
        self, context: InvestigationContext, timeout_seconds: float = 30.0
    ) -> ProviderProposalResult:
        del timeout_seconds
        self.calls += 1
        self.contexts.append(context)
        proposal = self._proposals[min(self.calls - 1, len(self._proposals) - 1)]
        return ProviderProposalResult(proposal=proposal, usage=self._usage)

    async def count_input_tokens(self, context: InvestigationContext) -> int:
        values = (*context.questions, *context.evidence_summaries)
        return sum(len(value) for value in values) // 4 + 1
