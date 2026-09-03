"""RED tests for the Anthropic paid-API drafting adapter (e04s03-t01).

These tests drive the creation of:
  src/vehicle_risk_agent/adapters/anthropic_drafting.py

All Anthropic API calls are mocked — no live network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from vehicle_risk_agent.reporting.protocol import (
    EvidenceItem,
    ReportDraftingContext,
)
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    RiskBand,
    RiskResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _scored_risk_result() -> RiskResult:
    """Minimal SCORED RiskResult for adapter tests."""
    from vehicle_risk_agent.risk.models import RiskFactor, RiskFactorResult

    return RiskResult(
        id="rr-adapter-01",
        assessment_id="assess-adapter-01",
        run_number=1,
        policy_id="nz-vehicle-risk-v1",
        policy_version="1.0",
        score=35,
        raw_score=35,
        band=RiskBand.MEDIUM,
        outcome=AssessmentOutcome.SCORED,
        is_incomplete=False,
        factors=(
            RiskFactorResult(
                factor=RiskFactor.LISTED,
                triggered=True,
                weight=45,
                score_contribution=45,
                evidence_field="stolen_vehicle",
                evidence_refs=("obs-001",),
                policy_citation_refs=("pol-listed-001",),
            ),
        ),
        findings=(),
        missing_evidence=(),
        missing_findings=(),
        calculation_hash="abc123",
    )


@pytest.fixture
def drafting_context(scored_risk_result: RiskResult) -> ReportDraftingContext:
    return ReportDraftingContext(
        assessment_id="assess-adapter-01",
        run_number=1,
        vehicle_id="VIN-ADAPTER-001",
        vin="VIN-ADAPTER-001",
        risk_result=scored_risk_result,
        evidence_items=(
            EvidenceItem(
                field_name="ppsr_interest",
                value=True,
                observation_id="obs-001",
                source_system="PPSR",
            ),
        ),
        drafter_id="anthropic-v1",
    )


@pytest.fixture
def scored_risk_result() -> RiskResult:
    return _scored_risk_result()


# ---------------------------------------------------------------------------
# t01-a: Adapter can be instantiated with config
# ---------------------------------------------------------------------------


def test_anthropic_adapter_instantiation() -> None:
    """Adapter can be instantiated with model, token, and timeout config."""
    from vehicle_risk_agent.adapters.anthropic_drafting import AnthropicDraftingAdapter

    adapter = AnthropicDraftingAdapter(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        timeout_seconds=30,
    )
    assert adapter.model == "claude-3-5-sonnet-20241022"
    assert adapter.max_tokens == 4096
    assert adapter.timeout_seconds == 30


def test_anthropic_adapter_default_config() -> None:
    """Adapter uses safe defaults when not configured."""
    from vehicle_risk_agent.adapters.anthropic_drafting import AnthropicDraftingAdapter

    adapter = AnthropicDraftingAdapter()
    assert adapter.max_tokens <= 16384
    assert adapter.timeout_seconds <= 60


# ---------------------------------------------------------------------------
# t01-b: Adapter implements ReportDraftingProtocol
# ---------------------------------------------------------------------------


def test_anthropic_adapter_implements_protocol() -> None:
    """Adapter satisfies the ReportDraftingProtocol structural check."""
    from vehicle_risk_agent.adapters.anthropic_drafting import AnthropicDraftingAdapter
    from vehicle_risk_agent.reporting.protocol import ReportDraftingProtocol

    adapter = AnthropicDraftingAdapter()
    assert isinstance(adapter, ReportDraftingProtocol)


# ---------------------------------------------------------------------------
# t01-c: Happy path — valid structured response produces a ReportDraft
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_adapter_happy_path(
    drafting_context: ReportDraftingContext,
) -> None:
    """A valid Anthropic structured response is parsed into a ReportDraft."""
    from vehicle_risk_agent.adapters.anthropic_drafting import AnthropicDraftingAdapter
    from vehicle_risk_agent.reporting.models import ReportDraft, ReportDraftStatus

    adapter = AnthropicDraftingAdapter(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        timeout_seconds=30,
    )

    mock_response = _build_valid_structured_response(drafting_context)

    with patch.object(adapter, "_call_anthropic_api", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_response
        draft = await adapter.draft_report(drafting_context)

    assert isinstance(draft, ReportDraft)
    assert draft.assessment_id == drafting_context.assessment_id
    assert draft.status in (ReportDraftStatus.DRAFT, ReportDraftStatus.VALIDATED)
    assert draft.outcome == AssessmentOutcome.SCORED
    # Score and band are preserved from risk_result — not from LLM output
    assert draft.sections.risk_score_and_band.score == 35
    assert draft.sections.risk_score_and_band.band == RiskBand.MEDIUM


# ---------------------------------------------------------------------------
# t01-d: Token and timeout bounds are enforced
# ---------------------------------------------------------------------------


def test_anthropic_adapter_rejects_token_limit_above_max() -> None:
    """Adapter rejects configuration exceeding hard token limit."""
    from vehicle_risk_agent.adapters.anthropic_drafting import AnthropicDraftingAdapter

    with pytest.raises((ValueError, TypeError)):
        AnthropicDraftingAdapter(max_tokens=100_000)  # Exceeds hard cap


def test_anthropic_adapter_rejects_timeout_above_max() -> None:
    """Adapter rejects timeout exceeding hard ceiling."""
    from vehicle_risk_agent.adapters.anthropic_drafting import AnthropicDraftingAdapter

    with pytest.raises((ValueError, TypeError)):
        AnthropicDraftingAdapter(timeout_seconds=3600)  # Exceeds hard cap


# ---------------------------------------------------------------------------
# t01-e: Credentials are never exposed
# ---------------------------------------------------------------------------


def test_anthropic_adapter_does_not_expose_api_key() -> None:
    """API key is not exposed in repr, str, or model dump."""
    from vehicle_risk_agent.adapters.anthropic_drafting import AnthropicDraftingAdapter

    adapter = AnthropicDraftingAdapter()
    repr_str = repr(adapter)
    str_str = str(adapter)
    assert "sk-ant-" not in repr_str
    assert "sk-ant-" not in str_str
    # Even if an env var were present
    assert "ANTHROPIC_API_KEY" not in repr_str


# ---------------------------------------------------------------------------
# t01-f: Telemetry metadata is included without request body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_adapter_includes_telemetry_without_prompt(
    drafting_context: ReportDraftingContext,
) -> None:
    """Draft metadata includes adapter_id, model, elapsed_ms — never the prompt text."""
    from vehicle_risk_agent.adapters.anthropic_drafting import AnthropicDraftingAdapter

    adapter = AnthropicDraftingAdapter(max_tokens=4096, timeout_seconds=30)
    mock_response = _build_valid_structured_response(drafting_context)

    with patch.object(adapter, "_call_anthropic_api", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_response
        draft = await adapter.draft_report(drafting_context)

    meta = draft.metadata
    assert "adapter_id" in meta
    assert "model" in meta
    assert "elapsed_ms" in meta
    # Prompt text must NOT be stored
    assert "prompt" not in meta
    assert "messages" not in meta
    assert "system" not in meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_valid_structured_response(ctx: ReportDraftingContext) -> dict[str, Any]:
    """Build a minimal valid structured adapter response dict."""
    return {
        "assessment_id": ctx.assessment_id,
        "run_number": ctx.run_number,
        "outcome": "SCORED",
        "sections": {
            "executive_summary": f"Vehicle {ctx.vin} assessed as MEDIUM risk (score 35).",
            "risk_narrative": "One adverse PPSR security interest was detected.",
        },
        "claim_refs": [
            {
                "claim_id": "claim-adapt-001",
                "statement": "PPSR security interest detected",
                "evidence_refs": ["obs-001"],
                # No policy_citation_refs — context has no policy_citations in allowlist
                "policy_citation_refs": [],
                "risk_factor_refs": ["LISTED"],
            }
        ],
    }
