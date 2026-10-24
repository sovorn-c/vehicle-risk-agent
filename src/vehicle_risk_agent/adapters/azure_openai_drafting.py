"""Azure OpenAI report-drafting adapter using the v1 API and Microsoft Entra ID.

Authentication is keyless: a managed-identity (or developer) credential issues
bearer tokens on demand. No token is stored on the adapter, logged, or included
in errors. SDK imports are lazy so the core install does not need the ``azure``
extra.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from vehicle_risk_agent.adapters.anthropic_drafting import (
    AnthropicDraftingAdapter,
    DraftingFailureError,
)
from vehicle_risk_agent.observability.telemetry import record_model_tokens
from vehicle_risk_agent.reporting.models import ReportDraft
from vehicle_risk_agent.reporting.protocol import ReportDraftingContext

if TYPE_CHECKING:
    import httpx2
    from azure.core.credentials_async import AsyncTokenCredential
    from openai import AsyncOpenAI

AZURE_OPENAI_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


class AzureOpenAIDraftingAdapter(AnthropicDraftingAdapter):
    """Draft a grounded report with one Azure OpenAI chat completion (JSON mode)."""

    provider = "azure_openai"
    max_retries = 0

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        client_id: str | None = None,
        max_tokens: int = 2048,
        timeout_seconds: int = 30,
        credential: AsyncTokenCredential | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        super().__init__(model=deployment, max_tokens=max_tokens, timeout_seconds=timeout_seconds)
        self._base_url = f"{endpoint.rstrip('/')}/openai/v1/"
        self._client_id = client_id
        self._credential = credential
        self._http_client = http_client
        self._client: AsyncOpenAI | None = None

    async def _safe_api_call(self, context: ReportDraftingContext) -> Any:
        """Wrap one Azure request with a bounded timeout and body-free failures."""
        try:
            return await asyncio.wait_for(
                self._call_azure_openai_api(context), timeout=float(self.timeout_seconds)
            )
        except TimeoutError:
            raise DraftingFailureError("TIMEOUT") from None
        except DraftingFailureError:
            raise
        except Exception:
            raise DraftingFailureError("PROVIDER_UNAVAILABLE") from None

    async def _call_azure_openai_api(self, context: ReportDraftingContext) -> dict[str, Any]:
        """Send one chat completion and return only text plus known usage."""
        import openai  # noqa: PLC0415

        client = self._get_client()
        try:
            completion = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(context)},
                    {"role": "user", "content": self._build_user_prompt(context)},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=self.max_tokens,
            )
        except openai.APITimeoutError:
            raise DraftingFailureError("TIMEOUT") from None
        except Exception:
            raise DraftingFailureError("PROVIDER_UNAVAILABLE") from None

        usage = completion.usage
        if usage is None or not all(
            self._known_int(value) for value in (usage.prompt_tokens, usage.completion_tokens)
        ):
            raise DraftingFailureError("UNKNOWN_USAGE")
        if not completion.choices:
            raise DraftingFailureError("INVALID_OUTPUT")
        choice = completion.choices[0]
        if choice.finish_reason == "length":
            raise DraftingFailureError("TOKEN_LIMIT")
        text = choice.message.content
        if not isinstance(text, str) or not text.strip():
            raise DraftingFailureError("INVALID_OUTPUT")
        record_model_tokens(usage.prompt_tokens, usage.completion_tokens, model=self.model)
        return {
            "text": text,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }

    def _get_client(self) -> AsyncOpenAI:
        """Build the client lazily so construction performs no I/O or credential lookup."""
        if self._client is not None:
            return self._client
        from azure.identity.aio import (  # noqa: PLC0415
            DefaultAzureCredential,
            get_bearer_token_provider,
        )
        from openai import AsyncOpenAI  # noqa: PLC0415

        if self._credential is None:
            self._credential = DefaultAzureCredential(managed_identity_client_id=self._client_id)
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=get_bearer_token_provider(self._credential, AZURE_OPENAI_TOKEN_SCOPE),
            max_retries=0,
            timeout=float(self.timeout_seconds),
            http_client=self._http_client,
        )
        return self._client

    @staticmethod
    def _known_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    async def _assemble_draft(
        self,
        context: ReportDraftingContext,
        validated: Any,
        elapsed_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> ReportDraft:
        """Reuse deterministic assembly; drop pricing because Azure prices are not configured."""
        draft = await super()._assemble_draft(
            context,
            validated,
            elapsed_ms=elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        metadata = {
            key: value
            for key, value in draft.metadata.items()
            if key not in {"estimated_cost_usd", "pricing_provenance"}
        }
        metadata.update(
            {"adapter_id": "azure-openai-v1", "provider": self.provider, "model": self.model}
        )
        return draft.model_copy(update={"metadata": metadata})

    def __repr__(self) -> str:
        return (
            f"AzureOpenAIDraftingAdapter(provider={self.provider!r}, deployment={self.model!r}, "
            f"max_tokens={self.max_tokens}, timeout_seconds={self.timeout_seconds})"
        )

    def __str__(self) -> str:
        return self.__repr__()
