"""Contract tests for Gemini's bounded report-drafting adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vehicle_risk_agent.adapters.anthropic_drafting import DraftingFailureError
from vehicle_risk_agent.adapters.gemini_drafting import GeminiDraftingAdapter
from vehicle_risk_agent.reporting.models import ReportDraft
from vehicle_risk_agent.reporting.protocol import EvidenceItem, ReportDraftingContext
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    RiskBand,
    RiskFactor,
    RiskFactorResult,
    RiskResult,
)


def _context() -> ReportDraftingContext:
    result = RiskResult(
        id="rr-gemini-01",
        assessment_id="assess-gemini-01",
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
        assessment_id="assess-gemini-01",
        run_number=1,
        vehicle_id="VIN-GEMINI-001",
        vin="VIN-GEMINI-001",
        risk_result=result,
        evidence_items=(
            EvidenceItem(
                field_name="listed",
                value=True,
                observation_id="obs-001",
                source_system="MCP",
            ),
        ),
        drafter_id="gemini-v1",
    )


def _client() -> SimpleNamespace:
    response = SimpleNamespace(
        text='{"assessment_id":"assess-gemini-01","run_number":1,"outcome":"SCORED","sections":{},"claim_refs":[]}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=30,
            response_token_count=12,
            thoughts_token_count=2,
        ),
    )
    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=AsyncMock(return_value=response))
        )
    )


def test_gemini_drafting_defaults_to_pinned_model() -> None:
    assert GeminiDraftingAdapter().model == "gemini-3.1-flash-lite"


@pytest.mark.asyncio
async def test_gemini_drafting_returns_grounded_report_and_known_usage() -> None:
    client = _client()
    draft = await GeminiDraftingAdapter(api_key="ignored", client=client).draft_report(_context())

    assert isinstance(draft, ReportDraft)
    assert draft.metadata["provider"] == "gemini"
    assert draft.metadata["adapter_id"] == "gemini-v1"
    assert draft.metadata["input_tokens"] == 30
    assert draft.metadata["output_tokens"] == 14
    assert ReportDraft.model_validate_json(draft.model_dump_json()) == draft
    call = client.aio.models.generate_content.await_args.kwargs
    assert call["model"] == "gemini-3.1-flash-lite"
    assert call["config"]["response_mime_type"] == "application/json"
    assert call["config"]["max_output_tokens"] == 2048

    def has_additional_properties(value: object) -> bool:
        if isinstance(value, dict):
            return "additionalProperties" in value or any(
                has_additional_properties(child) for child in value.values()
            )
        if isinstance(value, list):
            return any(has_additional_properties(child) for child in value)
        return False

    assert not has_additional_properties(call["config"]["response_schema"])


@pytest.mark.asyncio
async def test_gemini_drafting_rejects_unknown_usage() -> None:
    client = _client()
    client.aio.models.generate_content.return_value.usage_metadata = SimpleNamespace(
        prompt_token_count=30
    )

    with pytest.raises(DraftingFailureError, match="UNKNOWN_USAGE"):
        await GeminiDraftingAdapter(api_key="ignored", client=client).draft_report(_context())
