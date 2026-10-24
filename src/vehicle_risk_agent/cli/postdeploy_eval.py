"""Bounded postdeploy live check for the deployed agent environment.

Runs three checks in order, each with its own timeout and a 120 s overall cap:
database ``SELECT 1``, one real MCP ``lookup_vehicle``, and exactly one Azure
OpenAI draft of a fixed synthetic canary context. The canary is never persisted
and carries no policy citations. A failed check skips the later ones, so
database or MCP failures never trigger the paid drafting call.

Output is one JSON line per check plus a summary line with the keys
``check``, ``status``, ``latency_ms`` and ``error_category``. Prompts, tokens,
secrets, exception text and stack traces are never printed.

Exit codes: 0 all pass, 1 any check failed, 2 configuration invalid or the
drafting mode is not ``azure_openai``.
"""

# story: e13s04

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from vehicle_risk_agent.adapters.anthropic_drafting import DraftingFailureError
from vehicle_risk_agent.adapters.azure_openai_drafting import AzureOpenAIDraftingAdapter
from vehicle_risk_agent.adapters.mcp import McpAdapterError, create_mcp_adapter
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.reporting.protocol import EvidenceItem, ReportDraftingContext
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    RiskBand,
    RiskFactor,
    RiskFactorResult,
    RiskResult,
)

# [UNVERIFIED] until Phase 4: in pipeline phase2 fixtures, not checked against the Azure seed.
CANARY_VIN = "1HGCR2F85HA000000"
DATABASE_TIMEOUT_SECONDS = 10.0
MCP_TIMEOUT_SECONDS = 30.0
OVERALL_TIMEOUT_SECONDS = 120.0

EXIT_PASS = 0
EXIT_CHECK_FAILED = 1
EXIT_CONFIG_INVALID = 2

CANARY_CONTEXT = ReportDraftingContext(
    assessment_id="postdeploy-canary",
    run_number=1,
    vehicle_id="SYNTHETIC-CANARY-VIN",
    vin="SYNTHETIC-CANARY-VIN",
    risk_result=RiskResult(
        id="rr-postdeploy-canary",
        assessment_id="postdeploy-canary",
        run_number=1,
        policy_id="postdeploy-canary-policy",
        policy_version="canary",
        score=0,
        raw_score=0,
        band=RiskBand.LOW,
        outcome=AssessmentOutcome.SCORED,
        is_incomplete=False,
        factors=(
            RiskFactorResult(
                factor=RiskFactor.MATCH,
                triggered=False,
                weight=30,
                score_contribution=0,
                evidence_field="ppsr_interest",
                evidence_refs=("obs-canary-ppsr",),
                policy_citation_refs=(),
            ),
            RiskFactorResult(
                factor=RiskFactor.LISTED,
                triggered=False,
                weight=45,
                score_contribution=0,
                evidence_field="stolen_vehicle",
                evidence_refs=("obs-canary-svr",),
                policy_citation_refs=(),
            ),
        ),
        findings=(),
        missing_evidence=(),
        missing_findings=(),
        calculation_hash="0" * 64,
    ),
    evidence_items=(
        EvidenceItem(
            field_name="ppsr_interest",
            value=False,
            observation_id="obs-canary-ppsr",
            source_system="SYNTHETIC",
        ),
        EvidenceItem(
            field_name="stolen_vehicle",
            value=False,
            observation_id="obs-canary-svr",
            source_system="SYNTHETIC",
        ),
    ),
    drafter_id="azure-openai-v1",
    metadata={"synthetic": True, "purpose": "postdeploy-canary"},
)

Emit = Callable[[str], None]


def build_drafter(settings: Settings) -> AzureOpenAIDraftingAdapter:
    """Build the production Azure drafting adapter from validated settings."""
    assert settings.azure_openai_endpoint is not None
    assert settings.azure_openai_deployment is not None
    return AzureOpenAIDraftingAdapter(
        endpoint=settings.azure_openai_endpoint,
        deployment=settings.azure_openai_deployment,
        client_id=settings.azure_client_id,
        max_tokens=settings.drafting_max_tokens,
        timeout_seconds=int(settings.drafting_timeout_seconds),
    )


def _line(check: str, status: str, latency_ms: int, error_category: str | None) -> str:
    return json.dumps(
        {
            "check": check,
            "status": status,
            "latency_ms": latency_ms,
            "error_category": error_category,
        }
    )


def _error_category(check: str, exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "TIMEOUT"
    if isinstance(exc, DraftingFailureError):
        return exc.safe_category
    if isinstance(exc, McpAdapterError):
        return exc.category.value
    if check == "database":
        return "DATABASE_UNAVAILABLE"
    return "UNEXPECTED_ERROR"


async def run_checks(
    settings: Settings,
    *,
    engine_factory: Callable[[str], Any] = create_async_engine,
    mcp_factory: Callable[[Settings], Any] = create_mcp_adapter,
    drafter_factory: Callable[[Settings], Any] = build_drafter,
    emit: Emit = print,
    overall_timeout_seconds: float = OVERALL_TIMEOUT_SECONDS,
) -> int:
    """Run database, MCP and drafting checks in order; return the exit code."""
    try:
        engine = engine_factory(settings.database_url)
    except Exception:
        emit(_line("config", "fail", 0, "CONFIG_INVALID"))
        return EXIT_CONFIG_INVALID
    drafters: list[Any] = []

    async def database() -> None:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def mcp() -> None:
        await mcp_factory(settings).lookup_vehicle(CANARY_VIN)

    async def drafting() -> None:
        drafter = drafter_factory(settings)
        drafters.append(drafter)
        await drafter.draft_report(CANARY_CONTEXT)

    checks: list[tuple[str, Callable[[], Awaitable[None]], float]] = [
        ("database", database, DATABASE_TIMEOUT_SECONDS),
        ("mcp", mcp, MCP_TIMEOUT_SECONDS),
        ("drafting", drafting, settings.drafting_timeout_seconds + 5.0),
    ]
    run_start = time.monotonic()
    passed = True
    current = checks[0][0]
    check_start = run_start
    try:
        async with asyncio.timeout(overall_timeout_seconds):
            for name, check, timeout in checks:
                current, check_start = name, time.monotonic()
                try:
                    await asyncio.wait_for(check(), timeout=timeout)
                except Exception as exc:
                    emit(_line(name, "fail", _elapsed_ms(check_start), _error_category(name, exc)))
                    passed = False
                    break
                emit(_line(name, "pass", _elapsed_ms(check_start), None))
    except TimeoutError:
        emit(_line(current, "fail", _elapsed_ms(check_start), "TIMEOUT"))
        passed = False
    finally:
        await engine.dispose()
        for drafter in drafters:
            aclose = getattr(drafter, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()

    emit(
        _line(
            "summary",
            "pass" if passed else "fail",
            _elapsed_ms(run_start),
            None if passed else "CHECK_FAILED",
        )
    )
    return EXIT_PASS if passed else EXIT_CHECK_FAILED


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def main(argv: Sequence[str] | None = None, emit: Emit = print) -> int:
    """Load settings from the environment and run the bounded checks."""
    if argv:
        emit(_line("config", "fail", 0, "USAGE"))
        return EXIT_CONFIG_INVALID
    try:
        settings = Settings()
    except Exception:
        emit(_line("config", "fail", 0, "CONFIG_INVALID"))
        return EXIT_CONFIG_INVALID
    if settings.drafting_mode != "azure_openai":
        emit(_line("config", "fail", 0, "WRONG_DRAFTING_MODE"))
        return EXIT_CONFIG_INVALID
    return asyncio.run(run_checks(settings, emit=emit))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
