"""Anthropic paid-API drafting adapter with strict bounds and safe-fail semantics.

Design constraints (e04s03):
- No live API calls in tests — caller injects _call_anthropic_api mock point.
- Credentials sourced exclusively from environment; never logged or returned.
- max_tokens hard-capped at 16 384; timeout hard-capped at 120 s.
- All model output validated against a strict Pydantic schema before use.
- Score, band, and findings are ALWAYS sourced from the deterministic risk_result,
  never from the model response.
- Telemetry metadata included (adapter_id, model, elapsed_ms); prompt excluded.
- DraftingFailureError raised on timeout, parse error, schema violation, provider
  error, or grounding exhaustion — never bubbling raw exceptions or credentials.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vehicle_risk_agent.reporting.models import (
    ClaimReference,
    ReportDraft,
)
from vehicle_risk_agent.reporting.protocol import (
    ReportDraftingContext,
    ReportDraftingProtocol,
)

# Hard bounds enforced at construction time
_MAX_TOKENS_HARD_CAP: int = 16_384
_MAX_TIMEOUT_HARD_CAP: int = 120


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class DraftingFailureError(Exception):
    """Raised when the Anthropic adapter cannot produce a valid, grounded draft.

    Carries only a safe, opaque category string.  Internal details (API key,
    prompt text, raw stack traces, provider error bodies) are never included.
    """

    def __init__(self, safe_category: str) -> None:
        self.safe_category = safe_category
        # Public message is deliberately vague
        super().__init__(f"Drafting failed: {safe_category}")

    def __repr__(self) -> str:
        return f"DraftingFailureError(safe_category={self.safe_category!r})"


# ---------------------------------------------------------------------------
# Strict schema for model output
# ---------------------------------------------------------------------------


class _ClaimRefSchema(BaseModel):
    """Strict schema for a claim reference inside model output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    statement: str = Field(min_length=1, max_length=2048)
    evidence_refs: list[str] = Field(default_factory=list, max_length=50)
    policy_citation_refs: list[str] = Field(default_factory=list, max_length=50)
    risk_factor_refs: list[str] = Field(default_factory=list, max_length=20)


class _ModelOutputSchema(BaseModel):
    """Strict schema for the structured JSON output from the Anthropic model.

    Unknown fields are forbidden.  Score, band, and score-altering fields are
    explicitly disallowed — they are always sourced from the deterministic result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_id: str
    run_number: int = Field(default=1, ge=1)
    outcome: str = Field(pattern="^(SCORED|INCOMPLETE)$")
    sections: dict[str, str] = Field(default_factory=dict, max_length=20)
    claim_refs: list[_ClaimRefSchema] = Field(default_factory=list, max_length=200)

    @field_validator("sections", mode="before")
    @classmethod
    def _reject_score_override(cls, v: Any) -> Any:
        if isinstance(v, dict):
            forbidden_keys = {"score_override", "band_override", "score", "band"}
            if forbidden_keys & set(v.keys()):
                raise ValueError("Model output must not contain score/band override fields")
        return v


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class AnthropicDraftingAdapter(ReportDraftingProtocol):
    """Anthropic Messages API adapter producing strict, grounded report drafts.

    All Anthropic API calls are routed through ``_call_anthropic_api`` so that
    tests can patch it without ever constructing a live client.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
        timeout_seconds: int = 30,
    ) -> None:
        if max_tokens > _MAX_TOKENS_HARD_CAP:
            raise ValueError(f"max_tokens={max_tokens} exceeds hard cap {_MAX_TOKENS_HARD_CAP}")
        if timeout_seconds > _MAX_TIMEOUT_HARD_CAP:
            raise ValueError(
                f"timeout_seconds={timeout_seconds} exceeds hard cap {_MAX_TIMEOUT_HARD_CAP}"
            )
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # ReportDraftingProtocol implementation
    # ------------------------------------------------------------------

    async def draft_report(self, context: ReportDraftingContext) -> ReportDraft:
        """Draft a structured, grounded report using the Anthropic API.

        Raises:
            DraftingFailureError: On timeout, parse error, schema violation,
                provider error, or grounding exhaustion after one repair attempt.
        """
        start = time.monotonic()

        # 1. Call the API (mockable boundary)
        raw = await self._safe_api_call(context)

        # 2. Parse and validate the model output schema
        validated = self._parse_and_validate(raw)

        # 3. Build a ReportDraft — score/band sourced from risk_result only
        draft = await self._assemble_draft(
            context, validated, elapsed_ms=int((time.monotonic() - start) * 1000)
        )

        # 4. Grounding validation + bounded repair
        draft = self._ground_and_repair(draft, context)

        return draft

    # ------------------------------------------------------------------
    # Internal — each step is independently testable via patch
    # ------------------------------------------------------------------

    async def _safe_api_call(self, context: ReportDraftingContext) -> Any:
        """Wrap _call_anthropic_api with timeout and provider-error mapping."""
        try:
            return await asyncio.wait_for(
                self._call_anthropic_api(context),
                timeout=float(self.timeout_seconds),
            )
        except TimeoutError:
            raise DraftingFailureError("TIMEOUT") from None
        except Exception:
            raise DraftingFailureError("PROVIDER_UNAVAILABLE") from None

    async def _call_anthropic_api(self, context: ReportDraftingContext) -> Any:
        """Send a structured prompt to the Anthropic API and return raw output.

        This method is the patch target for tests.  In production it would
        create an ``AsyncAnthropic`` client from environment credentials.
        Credentials are read inside this method and never stored on ``self``.
        """
        # Production path: credentials read from environment at call time.
        # In test mode this method is replaced by a mock.
        import os  # noqa: PLC0415

        try:
            import anthropic  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as err:
            raise DraftingFailureError("PROVIDER_UNAVAILABLE") from err

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise DraftingFailureError("PROVIDER_UNAVAILABLE")

        client = anthropic.AsyncAnthropic(api_key=api_key)
        system_prompt = self._build_system_prompt(context)
        user_prompt = self._build_user_prompt(context)
        try:
            message = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:
            raise DraftingFailureError("PROVIDER_UNAVAILABLE") from exc

        # Extract text block
        text = message.content[0].text if message.content else ""
        return text

    def _extract_allowed_evidence_ids(self, context: ReportDraftingContext) -> tuple[str, ...]:
        """Extract all valid observation IDs from items, snapshot provenance, and risk result."""
        evidence_ids: set[str] = set()
        for item in context.evidence_items:
            if item.observation_id:
                evidence_ids.add(item.observation_id)
        if context.evidence_snapshot is not None and context.evidence_snapshot.field_provenance:
            for prov_list in context.evidence_snapshot.field_provenance.values():
                for p in prov_list:
                    if p.observation_id:
                        evidence_ids.add(p.observation_id)
        for factor in context.risk_result.factors:
            evidence_ids.update(factor.evidence_refs)
        for finding in context.risk_result.findings:
            evidence_ids.update(finding.evidence_refs)
        return tuple(sorted(evidence_ids))

    def _build_system_prompt(self, context: ReportDraftingContext) -> str:
        """Build a grounding-aware system prompt.  Never stored or returned."""
        allowed_ev = list(self._extract_allowed_evidence_ids(context))
        return (
            "You are a vehicle risk report drafting assistant. "
            "Your output must be valid JSON matching the required schema. "
            "You MUST only reference these evidence IDs: "
            f"{allowed_ev}. "
            "Do NOT include score_override, band_override, or any field not in the schema."
        )

    def _build_user_prompt(self, context: ReportDraftingContext) -> str:
        """Build the user prompt.  Never stored or returned."""
        return (
            f"Draft a structured risk report for assessment {context.assessment_id}, "
            f"run {context.run_number}. Vehicle: {context.vin}. "
            "Return valid JSON with keys: assessment_id, run_number, outcome, sections, claim_refs."
        )

    def _parse_and_validate(self, raw: Any) -> _ModelOutputSchema:
        """Parse raw model output into the strict schema.

        Raises:
            DraftingFailureError: On JSON parse failure or schema violation.
        """
        # raw may be a string (JSON) or already a dict (from mock)
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                raise DraftingFailureError("PARSE_ERROR") from None
        elif isinstance(raw, dict):
            data = raw
        else:
            raise DraftingFailureError("INVALID_OUTPUT")

        try:
            return _ModelOutputSchema.model_validate(data)
        except Exception:
            raise DraftingFailureError("SCHEMA_VIOLATION") from None

    async def _assemble_draft(
        self,
        context: ReportDraftingContext,
        validated: _ModelOutputSchema,
        elapsed_ms: int,
    ) -> ReportDraft:
        """Build a ReportDraft using the offline adapter for structure integrity.

        Score, band, and findings are always sourced from context.risk_result.
        The model's narrative text supplements section content only.
        """
        from vehicle_risk_agent.reporting.offline import (
            OfflineReportDraftingAdapter,  # noqa: PLC0415
        )

        # Generate the full structural draft via the offline adapter (await — no new loop)
        offline_adapter = OfflineReportDraftingAdapter(drafter_version="anthropic-v1")
        base_draft = await offline_adapter.draft_report(context)

        # Convert model claim_refs to ClaimReference objects
        model_claims: list[ClaimReference] = []
        for cr in validated.claim_refs:
            model_claims.append(
                ClaimReference(
                    claim_id=cr.claim_id,
                    statement=cr.statement,
                    evidence_refs=tuple(cr.evidence_refs),
                    policy_citation_refs=tuple(cr.policy_citation_refs),
                    risk_factor_refs=tuple(cr.risk_factor_refs),
                )
            )

        # Augment executive summary with model's claim refs (merged, not replaced)
        combined_claims = base_draft.sections.executive_summary.claims + tuple(model_claims)
        new_exec = base_draft.sections.executive_summary.model_copy(
            update={"claims": combined_claims}
        )
        new_sections = base_draft.sections.model_copy(update={"executive_summary": new_exec})

        # Metadata: telemetry only — never prompt, credentials, or raw response
        metadata = {
            "adapter_id": "anthropic-v1",
            "model": self.model,
            "elapsed_ms": elapsed_ms,
            **context.metadata,
        }

        return base_draft.model_copy(
            update={
                "sections": new_sections,
                "metadata": metadata,
            }
        )

    def _ground_and_repair(self, draft: ReportDraft, context: ReportDraftingContext) -> ReportDraft:
        """Validate grounding and attempt at most one repair.

        Raises:
            DraftingFailureError: If claims remain ungrounded after repair.
        """
        from vehicle_risk_agent.reporting.grounding import (  # noqa: PLC0415
            GroundingContext,
            GroundingValidator,
            RepairExhaustedError,
        )

        # Build the grounding context from pinned run state.
        # evidence_ids: from items, snapshot provenance, and risk_result
        allowed_evidence_ids = self._extract_allowed_evidence_ids(context)

        # citation_ids: from explicit policy_citations + any refs already in risk_result
        # (risk_result factor/finding citation refs are part of pinned run state)
        citation_ids: set[str] = set()
        for c in context.policy_citations:
            citation_ids.add(c.passage_id)
        for factor in context.risk_result.factors:
            citation_ids.update(factor.policy_citation_refs)
        for finding in context.risk_result.findings:
            citation_ids.update(finding.policy_citation_refs)
        allowed_citation_ids = tuple(sorted(citation_ids))

        grounding_ctx = GroundingContext(
            assessment_id=context.assessment_id,
            allowed_evidence_ids=allowed_evidence_ids,
            allowed_citation_ids=allowed_citation_ids,
            risk_result=context.risk_result,
        )

        validator = GroundingValidator()
        result = validator.validate(draft, grounding_ctx)

        if result.is_valid:
            return draft

        # One bounded repair attempt
        try:
            repaired = validator.repair(draft, grounding_ctx)
        except RepairExhaustedError:
            raise DraftingFailureError("GROUNDING_FAILED") from None

        # Validate again after repair
        result2 = validator.validate(repaired, grounding_ctx)
        if not result2.is_valid:
            raise DraftingFailureError("UNGROUNDED_CLAIMS")

        return repaired

    # ------------------------------------------------------------------
    # Dunder — credentials must never appear
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AnthropicDraftingAdapter("
            f"model={self.model!r}, "
            f"max_tokens={self.max_tokens}, "
            f"timeout_seconds={self.timeout_seconds})"
        )

    def __str__(self) -> str:
        return self.__repr__()
