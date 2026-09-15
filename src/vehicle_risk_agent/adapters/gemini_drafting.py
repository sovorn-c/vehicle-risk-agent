"""Gemini paid-API report drafting adapter for bounded live evaluation."""

from __future__ import annotations

import asyncio
import importlib
import os
from collections.abc import Mapping
from typing import Any, cast

from vehicle_risk_agent.adapters.anthropic_drafting import (
    AnthropicDraftingAdapter,
    DraftingFailureError,
)
from vehicle_risk_agent.observability.telemetry import record_model_tokens
from vehicle_risk_agent.reporting.models import ReportDraft
from vehicle_risk_agent.reporting.protocol import ReportDraftingContext


class GeminiDraftingAdapter(AnthropicDraftingAdapter):
    """Draft a grounded report with Gemini's structured JSON response mode."""

    provider = "gemini"
    model = "gemini-3.1-flash-lite"
    max_retries = 0

    def __init__(
        self,
        model: str = model,
        max_tokens: int = 2048,
        timeout_seconds: int = 30,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )
        self._client = client

    async def _safe_api_call(self, context: ReportDraftingContext) -> Any:
        """Wrap one Gemini request with a bounded timeout and safe failures."""
        try:
            return await asyncio.wait_for(
                self._call_gemini_api(context), timeout=float(self.timeout_seconds)
            )
        except TimeoutError:
            raise DraftingFailureError("TIMEOUT") from None
        except DraftingFailureError:
            raise
        except Exception:
            raise DraftingFailureError("PROVIDER_UNAVAILABLE") from None

    async def _call_gemini_api(self, context: ReportDraftingContext) -> dict[str, Any]:
        """Call Gemini's async SDK and return only text plus known usage."""
        client = await self._get_client()
        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=self._build_user_prompt(context),
                config={
                    "system_instruction": self._build_system_prompt(context),
                    "response_mime_type": "application/json",
                    "response_schema": self._output_schema(),
                    "max_output_tokens": self.max_tokens,
                    "temperature": 0.0,
                },
            )
        except Exception as exc:
            raise DraftingFailureError("PROVIDER_UNAVAILABLE") from exc

        usage = self._usage_data(self._field(response, "usage_metadata", "usageMetadata"))
        if usage["output_tokens"] > self.max_tokens:
            raise DraftingFailureError("TOKEN_LIMIT")
        text = self._field(response, "text")
        if not isinstance(text, str) or not text.strip():
            candidates = self._field(response, "candidates")
            text = self._candidate_text(candidates)
        if not isinstance(text, str) or not text.strip():
            raise DraftingFailureError("INVALID_OUTPUT")
        record_model_tokens(usage["input_tokens"], usage["output_tokens"], model=self.model)
        return {
            "text": text,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key.strip():
            raise DraftingFailureError("PROVIDER_UNAVAILABLE")
        try:
            genai = cast(Any, importlib.import_module("google.genai"))
            types = cast(Any, importlib.import_module("google.genai.types"))
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=max(1, self.timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            return self._client
        except Exception as exc:
            raise DraftingFailureError("PROVIDER_UNAVAILABLE") from exc

    @staticmethod
    def _output_schema() -> dict[str, Any]:
        """Return a Developer API-compatible schema; Pydantic remains strict locally."""
        # The Developer API accepts a smaller JSON-Schema subset than the
        # Pydantic schema.  Unknown fields and constraints are still rejected
        # by _ModelOutputSchema after the response is received.
        return {
            "type": "object",
            "properties": {
                "assessment_id": {"type": "string"},
                "run_number": {"type": "integer"},
                "outcome": {"type": "string", "enum": ["SCORED", "INCOMPLETE"]},
                "sections": {"type": "object"},
                "claim_refs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id": {"type": "string"},
                            "statement": {"type": "string"},
                            "evidence_refs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "policy_citation_refs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "risk_factor_refs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["claim_id", "statement"],
                    },
                },
            },
            "required": ["assessment_id", "outcome"],
        }

    @staticmethod
    def _field(value: Any, *names: str) -> Any:
        for name in names:
            if isinstance(value, Mapping) and name in value:
                return value[name]
            if value is not None and hasattr(value, name):
                return getattr(value, name)
        return None

    @classmethod
    def _usage_data(cls, usage: Any) -> dict[str, int]:
        if usage is None:
            raise DraftingFailureError("UNKNOWN_USAGE")
        prompt = cls._field(usage, "prompt_token_count", "promptTokenCount")
        response = cls._field(usage, "response_token_count", "responseTokenCount")
        if response is None:
            response = cls._field(usage, "candidates_token_count", "candidatesTokenCount")
        thoughts = cls._field(usage, "thoughts_token_count", "thoughtsTokenCount") or 0
        if not all(cls._known_int(value) for value in (prompt, response, thoughts)):
            raise DraftingFailureError("UNKNOWN_USAGE")
        return {"input_tokens": cast(int, prompt), "output_tokens": cast(int, response) + thoughts}

    @staticmethod
    def _known_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    @classmethod
    def _candidate_text(cls, candidates: Any) -> str:
        if not isinstance(candidates, (list, tuple)):
            return ""
        for candidate in candidates:
            content = cls._field(candidate, "content")
            parts = cls._field(content, "parts")
            if not isinstance(parts, (list, tuple)):
                continue
            for part in parts:
                text = cls._field(part, "text")
                if isinstance(text, str) and text.strip():
                    return text
        return ""

    async def _assemble_draft(
        self,
        context: ReportDraftingContext,
        validated: Any,
        elapsed_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> ReportDraft:
        """Reuse deterministic assembly and relabel telemetry for Gemini pricing."""
        draft = await super()._assemble_draft(
            context,
            validated,
            elapsed_ms=elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        metadata = {
            **draft.metadata,
            "adapter_id": "gemini-v1",
            "provider": self.provider,
            "model": self.model,
            "estimated_cost_usd": (input_tokens * 0.25 + output_tokens * 1.5) / 1_000_000,
            "pricing_provenance": f"model={self.model}, input=$0.25/M, output=$1.5/M",
        }
        return draft.model_copy(update={"metadata": metadata})

    def __repr__(self) -> str:
        return (
            f"GeminiDraftingAdapter(model={self.model!r}, "
            f"max_tokens={self.max_tokens}, timeout_seconds={self.timeout_seconds})"
        )

    def __str__(self) -> str:
        return self.__repr__()
