"""Contract tests for the bounded postdeploy live check."""

# story: e13s04

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from types import TracebackType
from typing import Any

import httpx2
import pytest
from azure.core.credentials import AccessToken

from vehicle_risk_agent.adapters.anthropic_drafting import DraftingFailureError
from vehicle_risk_agent.adapters.azure_openai_drafting import AzureOpenAIDraftingAdapter
from vehicle_risk_agent.adapters.mcp import McpAdapterError
from vehicle_risk_agent.cli import postdeploy_eval
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.evidence.models import SafeErrorCategory
from vehicle_risk_agent.reporting.models import ReportDraft
from vehicle_risk_agent.reporting.protocol import ReportDraftingContext

KEYS = {"check", "status", "latency_ms", "error_category"}


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "drafting_mode": "azure_openai",
        "enable_live_drafting": True,
        "azure_openai_endpoint": "https://aoai-test.openai.azure.com",
        "azure_openai_deployment": "chat",
        "drafting_max_tokens": 1500,
        "mcp_server_url": "http://mcp:8000/mcp",
    }
    values.update(overrides)
    return Settings(**values)


class _Conn:
    def __init__(self, fail: bool) -> None:
        self.fail = fail

    async def __aenter__(self) -> _Conn:
        if self.fail:
            raise OSError("db password=hunter2 unreachable")
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def execute(self, _statement: Any) -> None:
        return None


class _Engine:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.disposed = False

    def connect(self) -> _Conn:
        return _Conn(self.fail)

    async def dispose(self) -> None:
        self.disposed = True


class _Mcp:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.vins: list[str] = []

    async def lookup_vehicle(self, vin: str) -> Any:
        self.vins.append(vin)
        if self.error is not None:
            raise self.error
        return object()


class _Drafter:
    def __init__(self, error: Exception | None = None, delay: float = 0.0) -> None:
        self.error = error
        self.delay = delay
        self.contexts: list[ReportDraftingContext] = []
        self.closed = False

    async def draft_report(self, context: ReportDraftingContext) -> Any:
        self.contexts.append(context)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return object()

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_drafter_is_closed_after_run() -> None:
    harness = _Harness()

    assert await harness.run(_settings()) == 0

    assert harness.drafter.closed


@pytest.mark.asyncio
async def test_unusable_database_url_exits_2_without_traceback() -> None:
    lines: list[dict[str, Any]] = []

    def broken_engine(_url: str) -> Any:
        raise ModuleNotFoundError("No module named 'psycopg2'")

    exit_code = await postdeploy_eval.run_checks(
        _settings(),
        engine_factory=broken_engine,
        emit=lambda line: lines.append(json.loads(line)),
    )

    assert exit_code == 2
    assert lines == [
        {"check": "config", "status": "fail", "latency_ms": 0, "error_category": "CONFIG_INVALID"}
    ]


def test_module_entrypoint_forwards_arguments() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vehicle_risk_agent.cli.postdeploy_eval", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 2
    assert json.loads(result.stdout.strip().splitlines()[-1])["error_category"] == "USAGE"


class _Harness:
    def __init__(
        self,
        *,
        engine: _Engine | None = None,
        mcp: _Mcp | None = None,
        drafter: _Drafter | None = None,
    ) -> None:
        self.engine = engine or _Engine()
        self.mcp = mcp or _Mcp()
        self.drafter = drafter or _Drafter()
        self.lines: list[dict[str, Any]] = []

    async def run(self, settings: Settings, **kwargs: Any) -> int:
        return await postdeploy_eval.run_checks(
            settings,
            engine_factory=lambda _url: self.engine,
            mcp_factory=lambda _s: self.mcp,
            drafter_factory=lambda _s: self.drafter,
            emit=lambda line: self.lines.append(json.loads(line)),
            **kwargs,
        )

    def by_check(self) -> dict[str, dict[str, Any]]:
        return {line["check"]: line for line in self.lines}


@pytest.mark.asyncio
async def test_all_checks_pass_emit_four_json_lines() -> None:
    harness = _Harness()

    assert await harness.run(_settings()) == 0

    assert [line["check"] for line in harness.lines] == ["database", "mcp", "drafting", "summary"]
    assert all(set(line) == KEYS for line in harness.lines)
    assert all(line["status"] == "pass" for line in harness.lines)
    assert all(line["error_category"] is None for line in harness.lines)
    assert harness.mcp.vins == [postdeploy_eval.CANARY_VIN]
    assert harness.drafter.contexts == [postdeploy_eval.CANARY_CONTEXT]
    assert harness.engine.disposed


@pytest.mark.asyncio
async def test_database_failure_skips_mcp_and_drafting() -> None:
    harness = _Harness(engine=_Engine(fail=True))

    assert await harness.run(_settings()) == 1

    checks = harness.by_check()
    assert checks["database"]["status"] == "fail"
    assert checks["database"]["error_category"] == "DATABASE_UNAVAILABLE"
    assert checks["summary"]["status"] == "fail"
    assert harness.mcp.vins == []
    assert harness.drafter.contexts == []
    assert "hunter2" not in json.dumps(harness.lines)
    assert harness.engine.disposed


@pytest.mark.asyncio
async def test_mcp_failure_skips_paid_drafting_call() -> None:
    error = McpAdapterError(SafeErrorCategory.PIPELINE_UNAVAILABLE, "raw upstream text")
    harness = _Harness(mcp=_Mcp(error=error))

    assert await harness.run(_settings()) == 1

    checks = harness.by_check()
    assert checks["mcp"]["status"] == "fail"
    assert checks["mcp"]["error_category"] == SafeErrorCategory.PIPELINE_UNAVAILABLE.value
    assert harness.drafter.contexts == []


@pytest.mark.asyncio
async def test_drafting_failure_reports_safe_category() -> None:
    harness = _Harness(drafter=_Drafter(error=DraftingFailureError("TOKEN_LIMIT")))

    assert await harness.run(_settings()) == 1

    checks = harness.by_check()
    assert checks["drafting"]["status"] == "fail"
    assert checks["drafting"]["error_category"] == "TOKEN_LIMIT"
    assert len(harness.drafter.contexts) == 1


@pytest.mark.asyncio
async def test_overall_timeout_fails_with_timeout_category() -> None:
    harness = _Harness(drafter=_Drafter(delay=5.0))

    exit_code = await harness.run(_settings(), overall_timeout_seconds=0.2)

    assert exit_code == 1
    checks = harness.by_check()
    assert checks["drafting"]["error_category"] == "TIMEOUT"
    assert checks["summary"]["status"] == "fail"


def test_default_drafter_is_bounded_azure_adapter() -> None:
    settings = _settings(drafting_timeout_seconds=12.0)

    drafter = postdeploy_eval.build_drafter(settings)

    assert isinstance(drafter, AzureOpenAIDraftingAdapter)
    assert drafter.max_tokens == settings.drafting_max_tokens
    assert drafter.max_retries == 0


def test_offline_mode_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DRAFTING_MODE", "offline")

    assert postdeploy_eval.main([]) == 2
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["error_category"] == "WRONG_DRAFTING_MODE"


def test_invalid_settings_exit_2_without_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = "dev-sentinel-secret-token-value"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("REQUESTER_TOKEN", sentinel)

    assert postdeploy_eval.main([]) == 2
    out = capsys.readouterr()
    assert sentinel not in out.out + out.err
    line = json.loads(out.out.strip().splitlines()[-1])
    assert line["error_category"] == "CONFIG_INVALID"


class _FakeCredential:
    async def get_token(
        self,
        *scopes: str,
        claims: str | None = None,
        tenant_id: str | None = None,
        enable_cae: bool = False,
        **kwargs: Any,
    ) -> AccessToken:
        del scopes, claims, tenant_id, enable_cae, kwargs
        return AccessToken("fake", 4_102_444_800)

    async def close(self) -> None:
        return None

    async def __aenter__(self) -> _FakeCredential:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        return None


@pytest.mark.asyncio
async def test_canary_context_is_synthetic_and_groundable() -> None:
    context = postdeploy_eval.CANARY_CONTEXT
    evidence_ids = [item.observation_id for item in context.evidence_items]
    assert context.metadata["synthetic"] is True
    assert context.policy_citations == ()
    model_output = {
        "assessment_id": context.assessment_id,
        "run_number": 1,
        "outcome": "SCORED",
        "sections": {},
        "claim_refs": [
            {
                "claim_id": "canary-claim-01",
                "statement": "Synthetic canary evidence shows no register listing.",
                "evidence_refs": evidence_ids,
            }
        ],
    }
    completion = {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(model_output)},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    adapter = AzureOpenAIDraftingAdapter(
        endpoint="https://aoai-test.openai.azure.com",
        deployment="chat",
        credential=_FakeCredential(),
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(lambda _r: httpx2.Response(200, json=completion))
        ),
    )

    draft = await adapter.draft_report(context)

    assert isinstance(draft, ReportDraft)
    claim_ids = [c.claim_id for c in draft.sections.executive_summary.claims]
    assert "canary-claim-01" in claim_ids
