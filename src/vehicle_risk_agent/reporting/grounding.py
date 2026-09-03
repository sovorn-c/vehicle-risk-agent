"""Claim grounding validation and bounded text-only repair for report drafts.

Design constraints (e04s03):
- GroundingContext wraps the pinned run-state allowlists.
- GroundingValidator.validate() scans every ClaimReference across all sections.
- GroundingValidator.repair() strips ungrounded evidence_refs and
  policy_citation_refs from claims — it CANNOT change score, band, or findings.
- Only one repair attempt is permitted; RepairExhaustedError is raised on a
  second attempt.
- GroundingResult is a frozen Pydantic model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

# ClaimReference, RiskResult, and ReportSections are imported at runtime so
# Pydantic can resolve type annotations and mypy can type-check correctly.
from vehicle_risk_agent.reporting.models import ClaimReference, ReportSections
from vehicle_risk_agent.risk.models import RiskResult

if TYPE_CHECKING:
    from vehicle_risk_agent.reporting.models import ReportDraft


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class RepairExhaustedError(Exception):
    """Raised when a second repair attempt is made on the same validator instance."""


# ---------------------------------------------------------------------------
# Grounding context — immutable input from pinned run state
# ---------------------------------------------------------------------------


class GroundingContext(BaseModel):
    """Immutable allowlist derived from pinned run-state evidence and citations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    assessment_id: str
    allowed_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    allowed_citation_ids: tuple[str, ...] = Field(default_factory=tuple)
    risk_result: RiskResult


# ---------------------------------------------------------------------------
# Grounding result
# ---------------------------------------------------------------------------


class GroundingResult(BaseModel):
    """Immutable result of a grounding validation pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool
    repair_needed: bool
    ungrounded_claims: tuple[ClaimReference, ...] = Field(default_factory=tuple)
    repair_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Grounding validator
# ---------------------------------------------------------------------------


class GroundingValidator:
    """Validates and repairs claim grounding in a ReportDraft.

    Each instance tracks repair exhaustion.  After one repair attempt,
    ``RepairExhaustedError`` is raised if repair is called again.
    """

    def __init__(self) -> None:
        self._repair_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(self, draft: ReportDraft, context: GroundingContext) -> GroundingResult:
        """Validate every claim reference in the draft against the allowlists.

        Returns a GroundingResult indicating whether the draft is valid and
        which claims (if any) are ungrounded.
        """
        allowed_ev = set(context.allowed_evidence_ids)
        allowed_cit = set(context.allowed_citation_ids)

        ungrounded: list[ClaimReference] = []
        for claim in self._all_claims(draft):
            if self._claim_is_ungrounded(claim, allowed_ev, allowed_cit):
                ungrounded.append(claim)

        is_valid = len(ungrounded) == 0
        return GroundingResult(
            is_valid=is_valid,
            repair_needed=not is_valid,
            ungrounded_claims=tuple(ungrounded),
            repair_count=self._repair_count,
        )

    def repair(self, draft: ReportDraft, context: GroundingContext) -> ReportDraft:
        """Strip ungrounded evidence_refs and policy_citation_refs from all claims.

        - Score, band, and findings sections are NEVER modified.
        - Only evidence_refs and policy_citation_refs are stripped from claims.
        - Raises RepairExhaustedError if called a second time.
        """
        if self._repair_count >= 1:
            raise RepairExhaustedError(
                "Only one repair attempt is permitted per GroundingValidator instance."
            )
        self._repair_count += 1

        allowed_ev = set(context.allowed_evidence_ids)
        allowed_cit = set(context.allowed_citation_ids)

        new_sections = self._repair_sections(draft.sections, allowed_ev, allowed_cit)
        return draft.model_copy(update={"sections": new_sections})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _all_claims(self, draft: ReportDraft) -> list[ClaimReference]:
        """Collect all ClaimReference objects from every section of the draft."""

        sections = draft.sections
        claims: list[ClaimReference] = []

        def _extend(section_claims: tuple[ClaimReference, ...] | None) -> None:
            if section_claims:
                claims.extend(section_claims)

        _extend(getattr(sections.executive_summary, "claims", None))
        _extend(getattr(sections.vehicle_identity, "claims", None))
        _extend(getattr(sections.risk_score_and_band, "claims", None))
        _extend(getattr(sections.mandatory_review_findings, "claims", None))
        _extend(getattr(sections.contributing_factors, "claims", None))
        _extend(getattr(sections.policy_citations, "claims", None))
        _extend(getattr(sections.evidence_summary, "claims", None))
        _extend(getattr(sections.limitations_and_missing_evidence, "claims", None))
        _extend(getattr(sections.synthetic_data_notice, "claims", None))
        return claims

    def _claim_is_ungrounded(
        self,
        claim: ClaimReference,
        allowed_ev: set[str],
        allowed_cit: set[str],
    ) -> bool:
        """Return True if the claim references any ID outside the allowlists."""
        if any(ref and ref not in allowed_ev for ref in claim.evidence_refs):
            return True
        return any(ref and ref not in allowed_cit for ref in claim.policy_citation_refs)

    def _repair_sections(
        self,
        sections: ReportSections,
        allowed_ev: set[str],
        allowed_cit: set[str],
    ) -> ReportSections:
        """Return a copy of sections with ungrounded refs stripped from claims.

        Score, band, and findings sections are returned unchanged.
        """

        def _repair_claims(
            claims: tuple[ClaimReference, ...] | None,
        ) -> tuple[ClaimReference, ...]:
            if not claims:
                return ()
            return tuple(self._repair_claim(c, allowed_ev, allowed_cit) for c in claims)

        def _repair_section(section: BaseModel | None) -> BaseModel | None:
            """Repair claims within a section object if it has a 'claims' field."""
            if section is None:
                return section
            if not hasattr(section, "claims"):
                return section
            old_claims = getattr(section, "claims", None) or ()
            new_claims = _repair_claims(old_claims)
            if new_claims == old_claims:
                return section
            return section.model_copy(update={"claims": new_claims})

        # Repair narrative sections only — risk_score_and_band and
        # mandatory_review_findings are immutable (score/band/findings)
        return sections.model_copy(
            update={
                "executive_summary": _repair_section(sections.executive_summary),
                "vehicle_identity": _repair_section(sections.vehicle_identity),
                "policy_citations": _repair_section(sections.policy_citations),
                "evidence_summary": _repair_section(sections.evidence_summary),
                "limitations_and_missing_evidence": _repair_section(
                    sections.limitations_and_missing_evidence
                ),
                "synthetic_data_notice": _repair_section(sections.synthetic_data_notice),
                "contributing_factors": _repair_section(sections.contributing_factors),
                # risk_score_and_band — NEVER modified (score/band immutable)
                # mandatory_review_findings — NEVER modified (findings immutable)
            }
        )

    def _repair_claim(
        self,
        claim: ClaimReference,
        allowed_ev: set[str],
        allowed_cit: set[str],
    ) -> ClaimReference:
        """Return a copy of the claim with ungrounded refs stripped."""
        new_ev = tuple(r for r in claim.evidence_refs if r in allowed_ev)
        new_cit = tuple(r for r in claim.policy_citation_refs if r in allowed_cit)
        if new_ev == claim.evidence_refs and new_cit == claim.policy_citation_refs:
            return claim
        return claim.model_copy(update={"evidence_refs": new_ev, "policy_citation_refs": new_cit})
