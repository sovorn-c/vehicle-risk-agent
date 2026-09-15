"""Contract tests for the official Gemini bounded investigation adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vehicle_risk_agent.adapters.investigation import (
    GeminiInvestigationAdapter,
    InvestigationProviderError,
)
from vehicle_risk_agent.investigation.models import (
    InvestigationAction,
    InvestigationContext,
    SearchPolicyProposal,
)


def _response(
    *,
    args: dict[str, object] | None = None,
    name: str = "submit_investigation_proposal",
    parts: list[object] | None = None,
    usage: object | None = None,
) -> SimpleNamespace:
    function_call = SimpleNamespace(name=name, args=args or {"kind": "NO_ACTION"})
    response_parts = parts or [
        SimpleNamespace(function_call=function_call, text=None, thought=False)
    ]
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=response_parts))],
        usage_metadata=usage
        or SimpleNamespace(
            prompt_token_count=24,
            tool_use_prompt_token_count=2,
            response_token_count=8,
            thoughts_token_count=3,
            total_token_count=37,
        ),
    )


def _client(response: object | None = None, *, input_tokens: int = 26) -> SimpleNamespace:
    calls: list[str] = []

    async def count_tokens(**_kwargs: object) -> SimpleNamespace:
        calls.append("count")
        return SimpleNamespace(total_tokens=input_tokens)

    async def generate_content(**_kwargs: object) -> object:
        calls.append("generate")
        return response or _response()

    models = SimpleNamespace(
        count_tokens=AsyncMock(side_effect=count_tokens),
        generate_content=AsyncMock(side_effect=generate_content),
    )
    return SimpleNamespace(aio=SimpleNamespace(models=models), calls=calls)


def test_gemini_defaults_to_pinned_model() -> None:
    assert GeminiInvestigationAdapter().model == "gemini-3.1-flash-lite"


@pytest.mark.asyncio
async def test_gemini_returns_typed_proposal_and_provider_usage() -> None:
    client = _client(
        _response(
            args={
                "kind": "REQUEST",
                "action": "search_policy",
                "arguments": {"query": "required evidence"},
            }
        )
    )
    adapter = GeminiInvestigationAdapter(api_key="test-key", client=client)

    result = await adapter.propose(InvestigationContext(questions=("Which policy applies?",)))

    assert isinstance(result.proposal, SearchPolicyProposal)
    assert result.proposal.action == InvestigationAction.SEARCH_POLICY
    assert result.usage.input_tokens == 26
    assert result.usage.output_tokens == 11
    assert client.calls == ["count", "generate"]


@pytest.mark.asyncio
async def test_gemini_accepts_omitted_empty_history_arguments() -> None:
    client = _client(
        _response(
            args={
                "kind": "REQUEST",
                "action": InvestigationAction.GET_VEHICLE_HISTORY.value,
            }
        )
    )
    result = await GeminiInvestigationAdapter(api_key="test-key", client=client).propose(
        InvestigationContext(questions=("What older revision history exists?",))
    )

    assert result.proposal.kind == "REQUEST"
    assert result.proposal.action == InvestigationAction.GET_VEHICLE_HISTORY


@pytest.mark.asyncio
async def test_gemini_uses_one_strict_allowed_function_without_execution() -> None:
    client = _client()
    adapter = GeminiInvestigationAdapter(api_key="test-key", client=client)

    await adapter.propose(InvestigationContext(questions=("Explain the discrepancy",)))

    count_call = client.aio.models.count_tokens.await_args.kwargs
    generate_call = client.aio.models.generate_content.await_args.kwargs
    assert count_call["config"] == {}
    assert "Choose exactly one safe investigation proposal" in count_call["contents"]
    assert "get_vehicle_history" in count_call["contents"]
    assert "submit_investigation_proposal" in count_call["contents"]
    tool = generate_call["config"]["tools"][0]
    declaration = tool["function_declarations"][0]
    assert declaration["name"] == "submit_investigation_proposal"
    assert declaration["parameters_json_schema"]["required"] == ["kind"]
    assert declaration["parameters_json_schema"]["properties"]["kind"]["enum"] == [
        "NO_ACTION",
        "REQUEST",
    ]
    assert "oneOf" not in declaration["parameters_json_schema"]
    function_config = generate_call["config"]["tool_config"]["function_calling_config"]
    assert function_config == {
        "mode": "ANY",
        "allowed_function_names": ["submit_investigation_proposal"],
    }
    assert generate_call["config"]["automatic_function_calling"] == {"disable": True}
    assert generate_call["config"]["max_output_tokens"] == 512


@pytest.mark.asyncio
async def test_gemini_rejects_text_and_multiple_function_calls() -> None:
    text_client = _client(
        _response(
            parts=[
                SimpleNamespace(function_call=None, text="I will investigate.", thought=False),
            ]
        )
    )
    with pytest.raises(InvestigationProviderError, match="INVALID_PROPOSAL"):
        await GeminiInvestigationAdapter(api_key="test-key", client=text_client).propose(
            InvestigationContext(questions=("question",))
        )

    multiple_client = _client(
        _response(
            parts=[
                SimpleNamespace(
                    function_call=SimpleNamespace(
                        name="submit_investigation_proposal", args={"kind": "NO_ACTION"}
                    ),
                    text=None,
                    thought=False,
                ),
                SimpleNamespace(
                    function_call=SimpleNamespace(
                        name="submit_investigation_proposal", args={"kind": "NO_ACTION"}
                    ),
                    text=None,
                    thought=False,
                ),
            ]
        )
    )
    with pytest.raises(InvestigationProviderError, match="INVALID_PROPOSAL"):
        await GeminiInvestigationAdapter(api_key="test-key", client=multiple_client).propose(
            InvestigationContext(questions=("question",))
        )


@pytest.mark.asyncio
async def test_gemini_rejects_unknown_usage_and_input_limit() -> None:
    unknown_usage = _client(_response(usage=SimpleNamespace(prompt_token_count=10)))
    with pytest.raises(InvestigationProviderError, match="UNKNOWN_USAGE"):
        await GeminiInvestigationAdapter(api_key="test-key", client=unknown_usage).propose(
            InvestigationContext(questions=("question",))
        )
    assert unknown_usage.aio.models.generate_content.await_count == 1

    too_large = _client(input_tokens=3073)
    with pytest.raises(InvestigationProviderError, match="INPUT_TOKEN_LIMIT"):
        await GeminiInvestigationAdapter(api_key="test-key", client=too_large).propose(
            InvestigationContext(questions=("question",))
        )
    assert too_large.aio.models.generate_content.await_count == 0


@pytest.mark.asyncio
async def test_gemini_maps_timeout_and_provider_errors_safely() -> None:
    async def fail(**_kwargs: object) -> object:
        raise RuntimeError("secret provider response")

    client = _client()
    client.aio.models.generate_content = AsyncMock(side_effect=fail)
    with pytest.raises(InvestigationProviderError, match="PROVIDER_UNAVAILABLE") as exc_info:
        await GeminiInvestigationAdapter(api_key="test-key", client=client).propose(
            InvestigationContext(questions=("question",))
        )
    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_gemini_readiness_requires_sdk_and_environment_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    adapter = GeminiInvestigationAdapter()
    readiness = await adapter.readiness()
    assert readiness.ready is False
    assert readiness.failure_code == "CREDENTIALS_UNAVAILABLE"
    assert readiness.max_retries == 0


@pytest.mark.asyncio
async def test_gemini_readiness_rejects_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError()),
    )
    readiness = await GeminiInvestigationAdapter().readiness()
    assert readiness.ready is False
    assert readiness.failure_code == "SDK_UNAVAILABLE"


@pytest.mark.asyncio
async def test_gemini_maps_model_not_found_without_provider_details() -> None:
    class ModelNotFoundError(Exception):
        status_code = 404

    client = _client()
    client.aio.models.count_tokens = AsyncMock(side_effect=ModelNotFoundError("private detail"))
    with pytest.raises(InvestigationProviderError, match="MODEL_UNAVAILABLE") as exc_info:
        await GeminiInvestigationAdapter(api_key="test-key", client=client).propose(
            InvestigationContext(questions=("question",))
        )
    assert "private" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_gemini_timeout_is_safe() -> None:
    async def never_finishes(**_kwargs: object) -> object:
        await __import__("asyncio").sleep(1)
        return _response()

    client = _client()
    client.aio.models.generate_content = AsyncMock(side_effect=never_finishes)
    with pytest.raises(InvestigationProviderError, match="TIMEOUT"):
        await GeminiInvestigationAdapter(
            api_key="test-key", client=client, timeout_seconds=0.001
        ).propose(InvestigationContext(questions=("question",)))
