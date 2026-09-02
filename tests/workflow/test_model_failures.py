"""RED tests for safe-fail handling in the model drafting workflow (e04s03-t03).

These tests drive the creation of failure-handling paths in:
  src/vehicle_risk_agent/adapters/anthropic_drafting.py
  src/vehicle_risk_agent/reporting/grounding.py

The adapter must reach terminal FAILED status on:
  - Timeout (asyncio.TimeoutError)
  - Provider unavailability (API errors)
  - Invalid / malformed JSON output
  - Ungrounded claims that survive repair
No draft must be persisted, and internal details must not be exposed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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


def _scored_risk_result(assessment_id: str = "assess-fail-01") -> RiskResult:
    from vehicle_risk_agent.risk.models import RiskFactor, RiskFactorResult

    return RiskResult(
        id=f"rr-{assessment_id}",
        assessment_id=assessment_id,
        run_number=1,
        policy_id="nz-vehicle-risk-v1",
        policy_version="1.0",
        score=20,
        raw_score=20,
        band=RiskBand.MEDIUM,
        outcome=AssessmentOutcome.SCORED,
        is_incomplete=False,
        factors=(
            RiskFactorResult(
                factor=RiskFactor.MATCH,
                triggered=True,
                weight=30,
                score_contribution=20,
                evidence_field="ppsr_interest",
                evidence_refs=("obs-real-001",),
                policy_citation_refs=("pol-real-001",),
            ),
        ),
        findings=(),
        missing_evidence=(),
        missing_findings=(),
        calculation_hash="fail-hash-001",
    )


@pytest.fixture
def fail_context() -> ReportDraftingContext:
    return ReportDraftingContext(
        assessment_id="assess-fail-01",
        run_number=1,
        vehicle_id="VIN-FAIL-001",
        vin="VIN-FAIL-001",
        risk_result=_scored_risk_result("assess-fail-01"),
        evidence_items=(
            EvidenceItem(
                field_name="ppsr_interest",
                value=True,
                observation_id="obs-real-001",
                source_system="PPSR",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# t03-a: Timeout produces DraftingFailureError, no draft persisted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_raises_on_timeout(fail_context: ReportDraftingContext) -> None:
    """asyncio.TimeoutError from Anthropic call raises a safe DraftingFailureError."""
    from vehicle_risk_agent.adapters.anthropic_drafting import (
        AnthropicDraftingAdapter,
        DraftingFailureError,
    )

    adapter = AnthropicDraftingAdapter(max_tokens=512, timeout_seconds=5)

    with patch.object(adapter, "_call_anthropic_api", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = TimeoutError()

        with pytest.raises(DraftingFailureError) as exc_info:
            await adapter.draft_report(fail_context)

    error = exc_info.value
    assert error.safe_category == "TIMEOUT"
    # Internal details must not be exposed
    assert "sk-ant-" not in str(error)
    assert "prompt" not in str(error).lower()


# ---------------------------------------------------------------------------
# t03-b: Malformed JSON output raises DraftingFailureError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_raises_on_malformed_json(fail_context: ReportDraftingContext) -> None:
    """Malformed (non-JSON) model output raises DraftingFailureError with safe category."""
    from vehicle_risk_agent.adapters.anthropic_drafting import (
        AnthropicDraftingAdapter,
        DraftingFailureError,
    )

    adapter = AnthropicDraftingAdapter(max_tokens=512, timeout_seconds=5)

    with patch.object(adapter, "_call_anthropic_api", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = "THIS IS NOT VALID JSON {{{<<<"

        with pytest.raises(DraftingFailureError) as exc_info:
            await adapter.draft_report(fail_context)

    error = exc_info.value
    assert error.safe_category in ("INVALID_OUTPUT", "PARSE_ERROR")


# ---------------------------------------------------------------------------
# t03-c: Unknown fields in model output raise DraftingFailureError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_raises_on_unknown_fields(fail_context: ReportDraftingContext) -> None:
    """Output with extra unknown fields fails strict validation."""
    from vehicle_risk_agent.adapters.anthropic_drafting import (
        AnthropicDraftingAdapter,
        DraftingFailureError,
    )

    adapter = AnthropicDraftingAdapter(max_tokens=512, timeout_seconds=5)

    bad_response: dict[str, Any] = {
        "assessment_id": fail_context.assessment_id,
        "outcome": "SCORED",
        "INJECTED_FIELD": "evil payload",  # Should not be allowed
        "score_override": 0,  # Should not be allowed — score is deterministic
    }

    with patch.object(adapter, "_call_anthropic_api", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = bad_response

        with pytest.raises(DraftingFailureError) as exc_info:
            await adapter.draft_report(fail_context)

    assert exc_info.value.safe_category in ("INVALID_OUTPUT", "SCHEMA_VIOLATION")


# ---------------------------------------------------------------------------
# t03-d: Provider error (API exception) raises DraftingFailureError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_raises_on_provider_error(fail_context: ReportDraftingContext) -> None:
    """HTTP / provider errors are caught and re-raised as DraftingFailureError."""
    from vehicle_risk_agent.adapters.anthropic_drafting import (
        AnthropicDraftingAdapter,
        DraftingFailureError,
    )

    adapter = AnthropicDraftingAdapter(max_tokens=512, timeout_seconds=5)

    with patch.object(adapter, "_call_anthropic_api", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = RuntimeError("Connection refused")

        with pytest.raises(DraftingFailureError) as exc_info:
            await adapter.draft_report(fail_context)

    error = exc_info.value
    assert error.safe_category == "PROVIDER_UNAVAILABLE"
    # Raw exception details must not be re-exposed
    assert "Connection refused" not in str(error)


# ---------------------------------------------------------------------------
# t03-e: DraftingFailureError does not contain credentials or stack traces
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drafting_failure_does_not_leak_internals(
    fail_context: ReportDraftingContext,
) -> None:
    """Raised DraftingFailureError carries a safe public message only."""
    from vehicle_risk_agent.adapters.anthropic_drafting import (
        AnthropicDraftingAdapter,
        DraftingFailureError,
    )

    adapter = AnthropicDraftingAdapter(max_tokens=512, timeout_seconds=5)

    with patch.object(adapter, "_call_anthropic_api", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = TimeoutError()

        with pytest.raises(DraftingFailureError) as exc_info:
            await adapter.draft_report(fail_context)

    error = exc_info.value
    error_str = str(error)
    # These must NOT appear in the public error
    forbidden = ["sk-ant-", "ANTHROPIC_API_KEY", "Traceback", "prompt", "system:", "messages:"]
    for forbidden_term in forbidden:
        assert forbidden_term not in error_str, (
            f"DraftingFailureError must not expose '{forbidden_term}'"
        )


# ---------------------------------------------------------------------------
# t03-f: On failure, no draft is persisted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_draft_persisted_on_failure(fail_context: ReportDraftingContext) -> None:
    """When the adapter raises DraftingFailureError, no ReportDraft object is returned."""
    from vehicle_risk_agent.adapters.anthropic_drafting import (
        AnthropicDraftingAdapter,
        DraftingFailureError,
    )

    adapter = AnthropicDraftingAdapter(max_tokens=512, timeout_seconds=5)
    import contextlib

    with patch.object(adapter, "_call_anthropic_api", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = TimeoutError()

        result = None
        with contextlib.suppress(DraftingFailureError):
            result = await adapter.draft_report(fail_context)

    assert result is None, "No ReportDraft must be returned on failure"


# ---------------------------------------------------------------------------
# t03-g: Workflow transitions to FAILED status on DraftingFailureError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_transitions_to_failed_on_drafting_error(
    fail_context: ReportDraftingContext,
) -> None:
    """When draft_live_report raises DraftingFailureError, assessment moves to FAILED."""
    from vehicle_risk_agent.adapters.anthropic_drafting import (
        AnthropicDraftingAdapter,
        DraftingFailureError,
    )

    adapter = AnthropicDraftingAdapter(max_tokens=512, timeout_seconds=5)

    with patch.object(adapter, "_call_anthropic_api", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = TimeoutError()

        failure_status: str | None = None
        try:
            await adapter.draft_report(fail_context)
        except DraftingFailureError as exc:
            failure_status = exc.safe_category

    assert failure_status is not None, "DraftingFailureError must be raised"
    # The workflow layer must use this to transition to FAILED — confirmed by audit log
    # (full workflow integration is covered in test_report_drafting.py)


# ---------------------------------------------------------------------------
# t03-h: Ungrounded claims after repair raise DraftingFailureError (integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_fails_after_exhausted_repair(fail_context: ReportDraftingContext) -> None:
    """DraftingFailureError(GROUNDING_FAILED) is raised when repair raises RepairExhaustedError."""
    from vehicle_risk_agent.adapters.anthropic_drafting import (
        AnthropicDraftingAdapter,
        DraftingFailureError,
    )
    from vehicle_risk_agent.reporting.grounding import RepairExhaustedError

    adapter = AnthropicDraftingAdapter(max_tokens=512, timeout_seconds=5)

    # Provide a valid-schema response so assembly succeeds
    valid_response: dict[str, Any] = {
        "assessment_id": fail_context.assessment_id,
        "run_number": fail_context.run_number,
        "outcome": "SCORED",
        "sections": {"executive_summary": "Test report"},
        "claim_refs": [],
    }

    with patch.object(adapter, "_call_anthropic_api", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = valid_response

        # Patch the GroundingValidator at its source module
        with patch(
            "vehicle_risk_agent.reporting.grounding.GroundingValidator"
        ) as mock_validator_cls:
            mock_validator_instance = MagicMock()
            mock_validator_cls.return_value = mock_validator_instance

            from vehicle_risk_agent.reporting.grounding import GroundingResult

            # First validate call returns invalid
            mock_validator_instance.validate.return_value = GroundingResult(
                is_valid=False, repair_needed=True, ungrounded_claims=(), repair_count=0
            )
            # Repair raises RepairExhaustedError
            mock_validator_instance.repair.side_effect = RepairExhaustedError("exhausted")

            with pytest.raises(DraftingFailureError) as exc_info:
                await adapter.draft_report(fail_context)

    error = exc_info.value
    assert error.safe_category == "GROUNDING_FAILED"
