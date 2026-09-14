"""Strict-tool adapters for bounded investigation proposals."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from collections.abc import Mapping
from typing import Any, cast

from vehicle_risk_agent.investigation.models import (
    ALLOWED_EVIDENCE_TARGETS,
    InvestigationAction,
    InvestigationContext,
    ProviderProposalResult,
    ProviderReadiness,
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

    provider = "anthropic"
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

    async def readiness(self) -> ProviderReadiness:
        """Check configuration and strict-call guarantees without a paid request."""
        has_credentials = bool(self._api_key or os.environ.get("ANTHROPIC_API_KEY"))
        if self._client is None:
            try:
                importlib.import_module("anthropic")
            except Exception:
                return ProviderReadiness(
                    ready=False,
                    model=self.model,
                    strict_schema=True,
                    token_counting=True,
                    max_retries=self.max_retries,
                    failure_code="SDK_UNAVAILABLE",
                )
        if not has_credentials and self._client is None:
            return ProviderReadiness(
                ready=False,
                model=self.model,
                strict_schema=True,
                token_counting=True,
                max_retries=self.max_retries,
                failure_code="CREDENTIALS_UNAVAILABLE",
            )
        return ProviderReadiness(
            ready=True,
            model=self.model,
            strict_schema=True,
            token_counting=True,
            max_retries=self.max_retries,
        )

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
            anthropic = cast(Any, importlib.import_module("anthropic"))

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
            return cast(dict[str, Any], response)
        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        content = getattr(response, "content", None)
        usage = getattr(response, "usage", None)
        stop_reason = getattr(response, "stop_reason", None)
        blocks = [
            block.model_dump() if hasattr(block, "model_dump") else block for block in content or []
        ]
        usage_data = (
            usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else usage
        )
        return {"content": blocks, "usage": usage_data, "stop_reason": stop_reason}


class GeminiInvestigationAdapter(InvestigationProvider):
    """Use Gemini's async function-calling API for one strict proposal."""

    provider = "gemini"
    tool_name = "submit_investigation_proposal"
    model = "gemini-3.1-flash-lite"
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

    async def readiness(self) -> ProviderReadiness:
        """Check credentials and SDK availability without a paid request."""
        has_credentials = bool((self._api_key or os.environ.get("GEMINI_API_KEY", "")).strip())
        if not has_credentials and self._client is None:
            return ProviderReadiness(
                ready=False,
                model=self.model,
                strict_schema=True,
                token_counting=True,
                max_retries=self.max_retries,
                failure_code="CREDENTIALS_UNAVAILABLE",
            )
        if self._client is None:
            try:
                importlib.import_module("google.genai")
            except Exception:
                return ProviderReadiness(
                    ready=False,
                    model=self.model,
                    strict_schema=True,
                    token_counting=True,
                    max_retries=self.max_retries,
                    failure_code="SDK_UNAVAILABLE",
                )
        return ProviderReadiness(
            ready=True,
            model=self.model,
            strict_schema=True,
            token_counting=True,
            max_retries=self.max_retries,
        )

    async def count_input_tokens(self, context: InvestigationContext) -> int:
        """Count the minimized prompt and exact tool schema before generation."""
        client = await self._get_client()
        try:
            response = await asyncio.wait_for(
                client.aio.models.count_tokens(
                    model=self.model,
                    contents=self._count_prompt(context),
                    config=self._count_config(),
                ),
                timeout=max(0.001, self.timeout_seconds),
            )
        except TimeoutError as exc:
            raise InvestigationProviderError("TOKEN_COUNT_UNAVAILABLE") from exc
        except InvestigationProviderError:
            raise
        except Exception as exc:
            raise InvestigationProviderError(
                self._provider_failure_category(exc, "TOKEN_COUNT_UNAVAILABLE")
            ) from exc
        count = self._field(response, "total_tokens", "totalTokens")
        if not self._known_int(count):
            raise InvestigationProviderError("UNKNOWN_USAGE")
        return cast(int, count)

    async def propose(
        self, context: InvestigationContext, timeout_seconds: float = 30.0
    ) -> ProviderProposalResult:
        """Make one no-retry function-call request and validate its proposal."""
        input_tokens = await self.count_input_tokens(context)
        if input_tokens > self.max_input_tokens:
            raise InvestigationProviderError("INPUT_TOKEN_LIMIT")
        client = await self._get_client()
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self.model,
                    contents=self._user_prompt(context),
                    config=self._generate_config(),
                ),
                timeout=max(0.001, min(timeout_seconds, self.timeout_seconds)),
            )
        except TimeoutError as exc:
            raise InvestigationProviderError("TIMEOUT") from exc
        except Exception as exc:
            raise InvestigationProviderError(
                self._provider_failure_category(exc, "PROVIDER_UNAVAILABLE")
            ) from exc

        try:
            payload = self._response_payload(response)
            validated = SubmitInvestigationToolResponse.validate_provider_payload(payload)
        except InvestigationProviderError:
            raise
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
        api_key = self._api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key.strip():
            raise InvestigationProviderError("CREDENTIALS_UNAVAILABLE")
        try:
            genai = cast(Any, importlib.import_module("google.genai"))
            types = cast(Any, importlib.import_module("google.genai.types"))

            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=max(1, int(self.timeout_seconds * 1000)),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            return self._client
        except Exception as exc:
            raise InvestigationProviderError("SDK_UNAVAILABLE") from exc

    @classmethod
    def _tool_definition(cls) -> dict[str, Any]:
        return {
            "name": cls.tool_name,
            "description": "Submit one bounded read-only investigation proposal.",
            "parameters_json_schema": cls._gemini_parameters_schema(),
        }

    @staticmethod
    def _gemini_parameters_schema() -> dict[str, Any]:
        """Flatten the union for Gemini's function-declaration schema subset."""
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind"],
            "properties": {
                "kind": {"type": "string", "enum": ["NO_ACTION", "REQUEST"]},
                "action": {
                    "type": "string",
                    "enum": [action.value for action in InvestigationAction],
                },
                "arguments": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "field_name": {
                            "type": "string",
                            "enum": sorted(ALLOWED_EVIDENCE_TARGETS),
                        },
                        "revision_number": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                        },
                        "query": {"type": "string", "minLength": 1, "maxLength": 200},
                    },
                },
            },
        }

    @classmethod
    def _tools(cls) -> list[dict[str, Any]]:
        return [{"function_declarations": [cls._tool_definition()]}]

    @classmethod
    def _count_config(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def _count_prompt(cls, context: InvestigationContext) -> str:
        schema_text = json.dumps(cls._tools(), sort_keys=True, separators=(",", ":"))
        return f"{cls._system_prompt()}\n\n{cls._user_prompt(context)}\n\n{schema_text}"

    def _generate_config(self) -> dict[str, Any]:
        return {
            "system_instruction": self._system_prompt(),
            "tools": self._tools(),
            "tool_config": {
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": [self.tool_name],
                }
            },
            "automatic_function_calling": {"disable": True},
            "max_output_tokens": self.max_output_tokens,
            "temperature": 0.0,
        }

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Choose exactly one safe investigation proposal. If the question asks to explain a "
            "vehicle field, use explain_vehicle_field for the matching evidence target. If it "
            "asks whether present target fields match the current record and there are no "
            "conflicts, use NO_ACTION. If it asks for older revision history or revisions before "
            "the current record, use get_vehicle_history. If it asks for a policy rule or passage, "
            "use search_policy with a query derived from the policy question. Use NO_ACTION when "
            "complete evidence already answers the question and no discrepancy remains. Request "
            "one read-only action only when supplementary evidence or policy is needed. Never "
            "provide prose, VINs, URLs, paging, corpus IDs, scores, transitions, or approval "
            "actions."
        )

    @staticmethod
    def _user_prompt(context: InvestigationContext) -> str:
        return context.model_dump_json()

    @classmethod
    def _response_payload(cls, response: Any) -> dict[str, Any]:
        candidates = cls._field(response, "candidates")
        if not isinstance(candidates, (list, tuple)):
            candidates = []
        blocks: list[dict[str, Any]] = []
        for candidate in candidates:
            content = cls._field(candidate, "content")
            parts = cls._field(content, "parts")
            if not isinstance(parts, (list, tuple)):
                continue
            for part in parts:
                function_call = cls._field(part, "function_call", "functionCall")
                if function_call is not None:
                    args = cls._field(function_call, "args")
                    input_args = dict(args) if isinstance(args, Mapping) else {}
                    if (
                        input_args.get("kind") == "REQUEST"
                        and input_args.get("action")
                        == InvestigationAction.GET_VEHICLE_HISTORY.value
                    ):
                        # Gemini may omit an empty object for a no-argument action.
                        input_args.setdefault("arguments", {})
                    blocks.append(
                        {
                            "type": "tool_use",
                            "name": cls._field(function_call, "name"),
                            "input": cast(dict[str, Any], input_args),
                        }
                    )
                elif cls._field(part, "thought"):
                    continue
                elif cls._field(part, "text") is not None:
                    blocks.append({"type": "text", "text": cls._field(part, "text")})
                else:
                    blocks.append({"type": "unknown"})
        usage = cls._usage_data(cls._field(response, "usage_metadata", "usageMetadata"))
        return {
            "stop_reason": "tool_use"
            if len(blocks) == 1 and blocks[0].get("type") == "tool_use"
            else None,
            "content": blocks,
            "usage": usage,
        }

    @classmethod
    def _usage_data(cls, usage: Any) -> dict[str, int]:
        if usage is None:
            raise InvestigationProviderError("UNKNOWN_USAGE")
        prompt = cls._field(usage, "prompt_token_count", "promptTokenCount")
        tool_use = cls._field(usage, "tool_use_prompt_token_count", "toolUsePromptTokenCount")
        response = cls._field(usage, "response_token_count", "responseTokenCount")
        if response is None:
            response = cls._field(usage, "candidates_token_count", "candidatesTokenCount")
        thoughts = cls._field(usage, "thoughts_token_count", "thoughtsTokenCount")
        if not cls._known_int(prompt) or not cls._known_int(response):
            raise InvestigationProviderError("UNKNOWN_USAGE")
        if tool_use is None:
            tool_use = 0
        if thoughts is None:
            thoughts = 0
        if not cls._known_int(tool_use) or not cls._known_int(thoughts):
            raise InvestigationProviderError("UNKNOWN_USAGE")
        return {
            "input_tokens": prompt + tool_use,
            "output_tokens": response + thoughts,
        }

    @staticmethod
    def _field(value: Any, *names: str) -> Any:
        for name in names:
            if isinstance(value, Mapping) and name in value:
                return value[name]
            if value is not None and hasattr(value, name):
                return getattr(value, name)
        return None

    @staticmethod
    def _known_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    @staticmethod
    def _provider_failure_category(error: Exception, fallback: str) -> str:
        status = getattr(error, "status_code", None) or getattr(error, "code", None)
        return "MODEL_UNAVAILABLE" if status in (404, "404") else fallback
