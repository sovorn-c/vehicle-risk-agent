"""Anthropic strict-tool adapter for bounded investigation proposals."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from vehicle_risk_agent.investigation.models import (
    InvestigationContext,
    ProviderProposalResult,
    SubmitInvestigationToolResponse,
    proposal_json_schema,
)
from vehicle_risk_agent.investigation.protocol import InvestigationProvider


class InvestigationProviderError(Exception):
    """Safe provider boundary failure with no raw provider details."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(f"Investigation provider failed: {category}")


class AnthropicInvestigationAdapter(InvestigationProvider):
    """Use one forced strict tool block and disable provider retries."""

    tool_name = "submit_investigation_proposal"
    model = "claude-sonnet-4-6"
    max_retries = 0

    def __init__(
        self,
        model: str = model,
        timeout_seconds: float = 30.0,
        max_input_tokens: int = 3072,
        max_output_tokens: int = 512,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self._api_key = api_key
        self._client = client

    async def count_input_tokens(self, context: InvestigationContext) -> int:
        """Ask the official SDK for token count before paid execution."""
        client = await self._get_client()
        try:
            response = await client.messages.count_tokens(
                model=self.model,
                system=self._system_prompt(),
                messages=[{"role": "user", "content": self._user_prompt(context)}],
                tools=[self._tool_definition()],
            )
            count = getattr(response, "input_tokens", None)
            if count is None and isinstance(response, dict):
                count = response.get("input_tokens")
            if not isinstance(count, int):
                raise InvestigationProviderError("UNKNOWN_USAGE")
            return count
        except InvestigationProviderError:
            raise
        except Exception as exc:
            raise InvestigationProviderError("TOKEN_COUNT_UNAVAILABLE") from exc

    async def propose(
        self, context: InvestigationContext, timeout_seconds: float = 30.0
    ) -> ProviderProposalResult:
        """Make one no-retry strict-tool request and validate the response."""
        input_tokens = await self.count_input_tokens(context)
        if input_tokens > self.max_input_tokens:
            raise InvestigationProviderError("INPUT_TOKEN_LIMIT")
        client = await self._get_client()
        try:
            response = await asyncio.wait_for(
                client.messages.create(
                    model=self.model,
                    max_tokens=self.max_output_tokens,
                    system=self._system_prompt(),
                    messages=[{"role": "user", "content": self._user_prompt(context)}],
                    tools=[self._tool_definition()],
                    tool_choice={
                        "type": "tool",
                        "name": self.tool_name,
                        "disable_parallel_tool_use": True,
                    },
                ),
                timeout=min(timeout_seconds, self.timeout_seconds),
            )
        except TimeoutError as exc:
            raise InvestigationProviderError("TIMEOUT") from exc
        except Exception as exc:
            raise InvestigationProviderError("PROVIDER_UNAVAILABLE") from exc

        payload = self._response_payload(response)
        try:
            validated = SubmitInvestigationToolResponse.validate_provider_payload(payload)
        except Exception as exc:
            raise InvestigationProviderError("INVALID_PROPOSAL") from exc
        usage = validated.usage
        if (
            usage.input_tokens > self.max_input_tokens
            or usage.output_tokens > self.max_output_tokens
        ):
            raise InvestigationProviderError("TOKEN_LIMIT")
        return ProviderProposalResult(proposal=validated.proposal, usage=usage)

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise InvestigationProviderError("CREDENTIALS_UNAVAILABLE")
        try:
            import anthropic  # type: ignore[import-not-found]

            self._client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)
            return self._client
        except InvestigationProviderError:
            raise
        except Exception as exc:
            raise InvestigationProviderError("PROVIDER_UNAVAILABLE") from exc

    @classmethod
    def _tool_definition(cls) -> dict[str, Any]:
        return {
            "name": cls.tool_name,
            "description": "Submit one bounded read-only investigation proposal.",
            "strict": True,
            "input_schema": proposal_json_schema(),
        }

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Choose exactly one safe investigation proposal. Never provide prose, VINs, URLs, "
            "paging, corpus IDs, scores, transitions, or approval actions."
        )

    @staticmethod
    def _user_prompt(context: InvestigationContext) -> str:
        return context.model_dump_json()

    @staticmethod
    def _response_payload(response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            return response.model_dump()
        content = getattr(response, "content", None)
        usage = getattr(response, "usage", None)
        stop_reason = getattr(response, "stop_reason", None)
        blocks = [
            block.model_dump() if hasattr(block, "model_dump") else block for block in content or []
        ]
        usage_data = usage.model_dump() if hasattr(usage, "model_dump") else usage
        return {"content": blocks, "usage": usage_data, "stop_reason": stop_reason}
