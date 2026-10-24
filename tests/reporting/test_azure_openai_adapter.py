"""Contract tests for the Azure OpenAI (Entra ID, v1 API) report-drafting adapter."""

import json
from collections.abc import Callable
from types import TracebackType
from typing import Any

import httpx2
import pytest
from azure.core.credentials import AccessToken

from vehicle_risk_agent.adapters.anthropic_drafting import DraftingFailureError
from vehicle_risk_agent.adapters.azure_openai_drafting import (
    AZURE_OPENAI_TOKEN_SCOPE,
    AzureOpenAIDraftingAdapter,
)
from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.reporting.models import ReportDraft
from vehicle_risk_agent.reporting.protocol import EvidenceItem, ReportDraftingContext
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    RiskBand,
    RiskFactor,
    RiskFactorResult,
    RiskResult,
)

ENDPOINT = "https://aoai-test.openai.azure.com"
DEPLOYMENT = "chat"
FAKE_TOKEN = "fake-entra-token-value"
MODEL_JSON = (
    '{"assessment_id":"assess-azure-01","run_number":1,"outcome":"SCORED",'
    '"sections":{},"claim_refs":[]}'
)


def _context() -> ReportDraftingContext:
    result = RiskResult(
        id="rr-azure-01",
        assessment_id="assess-azure-01",
        run_number=1,
        policy_id="risk-policy-v1",
        policy_version="v1",
        score=35,
        raw_score=35,
        band=RiskBand.MEDIUM,
        outcome=AssessmentOutcome.SCORED,
        is_incomplete=False,
        factors=(
            RiskFactorResult(
                factor=RiskFactor.LISTED,
                triggered=True,
                weight=35,
                score_contribution=35,
                evidence_field="listed",
                evidence_refs=("obs-001",),
                policy_citation_refs=(),
            ),
        ),
        findings=(),
        missing_evidence=(),
        missing_findings=(),
        calculation_hash="a" * 64,
    )
    return ReportDraftingContext(
        assessment_id="assess-azure-01",
        run_number=1,
        vehicle_id="VIN-AZURE-001",
        vin="VIN-AZURE-001",
        risk_result=result,
        evidence_items=(
            EvidenceItem(
                field_name="listed",
                value=True,
                observation_id="obs-001",
                source_system="MCP",
            ),
        ),
        drafter_id="azure-openai-v1",
    )


class _FakeCredential:
    """Async credential double recording requested scopes; never a real token."""

    def __init__(self) -> None:
        self.scopes: list[tuple[str, ...]] = []

    async def get_token(
        self,
        *scopes: str,
        claims: str | None = None,
        tenant_id: str | None = None,
        enable_cae: bool = False,
        **kwargs: Any,
    ) -> AccessToken:
        del claims, tenant_id, enable_cae, kwargs
        self.scopes.append(scopes)
        return AccessToken(FAKE_TOKEN, 4_102_444_800)

    async def close(self) -> None:
        return None

    async def __aenter__(self) -> "_FakeCredential":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        await self.close()


def _completion(
    content: str | None = MODEL_JSON,
    *,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-5.4-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return body


_USAGE = {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55}


class _Recorder:
    def __init__(self, respond: Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.requests: list[httpx2.Request] = []
        self._respond = respond

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self._respond(request)


def _adapter(
    respond: Callable[[httpx2.Request], httpx2.Response],
) -> tuple[AzureOpenAIDraftingAdapter, _Recorder, _FakeCredential]:
    recorder = _Recorder(respond)
    credential = _FakeCredential()
    adapter = AzureOpenAIDraftingAdapter(
        endpoint=ENDPOINT,
        deployment=DEPLOYMENT,
        max_tokens=1024,
        timeout_seconds=10,
        credential=credential,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)),
    )
    return adapter, recorder, credential


def _ok(_request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json=_completion(usage=_USAGE))


@pytest.mark.asyncio
async def test_request_uses_v1_endpoint_entra_bearer_and_deployment() -> None:
    adapter, recorder, credential = _adapter(_ok)

    await adapter.draft_report(_context())

    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{ENDPOINT}/openai/v1/chat/completions"
    assert "api-version" not in request.url.params
    assert request.headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
    assert "api-key" not in request.headers
    body = json.loads(request.content)
    assert body["model"] == DEPLOYMENT
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_completion_tokens"] == 1024
    assert "temperature" not in body
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert credential.scopes == [(AZURE_OPENAI_TOKEN_SCOPE,)]


def test_token_scope_is_cognitive_services_default() -> None:
    assert AZURE_OPENAI_TOKEN_SCOPE == "https://cognitiveservices.azure.com/.default"


@pytest.mark.asyncio
async def test_valid_response_yields_grounded_draft_with_known_usage() -> None:
    adapter, _, _ = _adapter(_ok)

    draft = await adapter.draft_report(_context())

    assert isinstance(draft, ReportDraft)
    assert ReportDraft.model_validate_json(draft.model_dump_json()) == draft
    assert draft.risk_result_id == "rr-azure-01"
    assert draft.metadata["adapter_id"] == "azure-openai-v1"
    assert draft.metadata["provider"] == "azure_openai"
    assert draft.metadata["model"] == DEPLOYMENT
    assert draft.metadata["input_tokens"] == 40
    assert draft.metadata["output_tokens"] == 15
    # Azure pricing metadata is out of scope; never report another provider's prices.
    assert "estimated_cost_usd" not in draft.metadata
    assert "pricing_provenance" not in draft.metadata


@pytest.mark.parametrize(
    ("respond", "category"),
    [
        (lambda _: httpx2.Response(200, json=_completion()), "UNKNOWN_USAGE"),
        (
            lambda _: httpx2.Response(200, json=_completion(finish_reason="length", usage=_USAGE)),
            "TOKEN_LIMIT",
        ),
        (
            lambda _: httpx2.Response(200, json=_completion(content="", usage=_USAGE)),
            "INVALID_OUTPUT",
        ),
        (
            lambda _: httpx2.Response(200, json=_completion(content=None, usage=_USAGE)),
            "INVALID_OUTPUT",
        ),
    ],
)
@pytest.mark.asyncio
async def test_response_failures_map_to_safe_categories(
    respond: Callable[[httpx2.Request], httpx2.Response], category: str
) -> None:
    adapter, recorder, _ = _adapter(respond)

    with pytest.raises(DraftingFailureError, match=category):
        await adapter.draft_report(_context())
    assert len(recorder.requests) == 1


@pytest.mark.asyncio
async def test_transport_timeout_maps_to_timeout() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=request)

    adapter, recorder, _ = _adapter(respond)

    with pytest.raises(DraftingFailureError, match="TIMEOUT"):
        await adapter.draft_report(_context())
    assert len(recorder.requests) == 1


@pytest.mark.parametrize("status", [401, 403, 429, 500])
@pytest.mark.asyncio
async def test_http_errors_map_to_provider_unavailable_without_body_or_retry(status: int) -> None:
    secret_body = "upstream-secret-error-body"
    adapter, recorder, _ = _adapter(
        lambda _: httpx2.Response(status, json={"error": {"message": secret_body}})
    )

    with pytest.raises(DraftingFailureError) as excinfo:
        await adapter.draft_report(_context())

    assert excinfo.value.safe_category == "PROVIDER_UNAVAILABLE"
    assert excinfo.value.__cause__ is None
    assert secret_body not in str(excinfo.value)
    assert secret_body not in repr(excinfo.value)
    assert len(recorder.requests) == 1


def test_adapter_is_single_attempt_and_repr_hides_secrets() -> None:
    adapter, _, _ = _adapter(_ok)

    assert adapter.max_retries == 0
    assert adapter.provider == "azure_openai"
    assert adapter.model == DEPLOYMENT
    text = repr(adapter)
    assert FAKE_TOKEN not in text
    assert DEPLOYMENT in text
    assert str(adapter) == text


def test_create_app_wires_azure_adapter_without_network() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        mcp_server_url=None,
        drafting_mode="azure_openai",
        enable_live_drafting=True,
        azure_openai_endpoint=ENDPOINT,
        azure_openai_deployment=DEPLOYMENT,
        drafting_max_tokens=1536,
        drafting_timeout_seconds=12.0,
    )

    app = create_app(settings=settings)

    adapter = app.state.drafting_adapter
    assert isinstance(adapter, AzureOpenAIDraftingAdapter)
    assert adapter.model == DEPLOYMENT
    assert adapter.max_tokens == 1536
    assert adapter.timeout_seconds == 12


@pytest.mark.asyncio
async def test_managed_identity_credential_builds_without_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Container Apps managed-identity path must construct a real async credential."""
    monkeypatch.setenv("IDENTITY_ENDPOINT", "http://127.0.0.1:9/msi/token")
    monkeypatch.setenv("IDENTITY_HEADER", "test-identity-header")
    adapter = AzureOpenAIDraftingAdapter(
        endpoint=ENDPOINT,
        deployment=DEPLOYMENT,
        client_id="00000000-0000-0000-0000-000000000000",
    )

    adapter._get_client()

    await adapter.aclose()


@pytest.mark.asyncio
async def test_aclose_leaves_injected_credential_to_its_owner() -> None:
    closed: list[bool] = []

    class _TrackedCredential(_FakeCredential):
        async def close(self) -> None:
            closed.append(True)

    adapter = AzureOpenAIDraftingAdapter(
        endpoint=ENDPOINT,
        deployment=DEPLOYMENT,
        credential=_TrackedCredential(),
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(_ok)),
    )
    await adapter.draft_report(_context())

    await adapter.aclose()

    assert closed == []
