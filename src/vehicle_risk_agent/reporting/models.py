"""Domain models for Report Drafts, 9 Standard Sections, Claim Citations, and Notices."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vehicle_risk_agent.evidence.models import ConfidenceBand, FieldConflict
from vehicle_risk_agent.evidence.sufficiency import MissingEvidenceReason
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    MandatoryFinding,
    RiskBand,
    RiskFactor,
    RiskFactorResult,
)


class ReportDraftStatus(StrEnum):
    """Lifecycle status of a Report Draft."""

    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    RELEASED = "RELEASED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class SectionType(StrEnum):
    """Canonical identifier for each of the 9 required report sections."""

    EXECUTIVE_SUMMARY = "EXECUTIVE_SUMMARY"
    VEHICLE_IDENTITY = "VEHICLE_IDENTITY"
    RISK_SCORE_AND_BAND = "RISK_SCORE_AND_BAND"
    MANDATORY_REVIEW_FINDINGS = "MANDATORY_REVIEW_FINDINGS"
    CONTRIBUTING_FACTORS = "CONTRIBUTING_FACTORS"
    POLICY_CITATIONS = "POLICY_CITATIONS"
    EVIDENCE_SUMMARY = "EVIDENCE_SUMMARY"
    LIMITATIONS_AND_MISSING_EVIDENCE = "LIMITATIONS_AND_MISSING_EVIDENCE"
    SYNTHETIC_DATA_NOTICE = "SYNTHETIC_DATA_NOTICE"


# Alias for compatibility
ReportSectionType = SectionType


class ClaimReference(BaseModel):
    """Fine-grained attributable citation attached to an analytical claim or assertion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(default_factory=lambda: f"claim-{uuid4()}")
    statement: str = Field(min_length=1, description="Attributable statement or assertion")
    evidence_refs: tuple[str, ...] = Field(
        default_factory=tuple, description="Source observation IDs or canonical field keys"
    )
    policy_citation_refs: tuple[str, ...] = Field(
        default_factory=tuple, description="Policy passage IDs or citation IDs"
    )
    risk_factor_refs: tuple[str, ...] = Field(
        default_factory=tuple, description="Risk factor identifiers (e.g. MATCH, LISTED)"
    )

    @field_validator("risk_factor_refs", mode="before")
    @classmethod
    def normalize_risk_factors(cls, v: Any) -> tuple[str, ...]:
        if isinstance(v, (list, tuple, set)):
            return tuple(item.value if isinstance(item, RiskFactor) else str(item) for item in v)
        return ()


# Alias for compatibility
ReportClaim = ClaimReference


class MissingEvidenceNotice(BaseModel):
    """Explicit disclosure of missing or unverified required evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: str = Field(description="Target required evidence field")
    reason: str = Field(description="Specific cause for missing evidence")
    details: str = Field(description="Detailed explanation of missing data condition")
    impact: str = Field(
        default="Scoring withheld due to incomplete required evidence",
        description="Operational effect on assessment outcome",
    )

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, v: Any) -> str:
        if isinstance(v, MissingEvidenceReason):
            return v.value
        return str(v)


class SyntheticNotice(BaseModel):
    """Notice indicating presence of demonstration or synthetic data sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    is_synthetic: bool = Field(default=True, description="Flag indicating synthetic provenance")
    notice_text: str = Field(
        default=(
            "This assessment contains synthetic demonstration data and must not be "
            "used for production credit or underwriting decisions."
        ),
        description="Disclaimer text",
    )
    synthetic_sources: tuple[str, ...] = Field(
        default_factory=tuple, description="Identifiers of synthetic observation sources"
    )
    synthetic_fields: tuple[str, ...] = Field(
        default_factory=tuple, description="Identifiers of fields containing synthetic data"
    )


class AbstentionNotice(BaseModel):
    """Explicit disclosure that policy retrieval withheld citations due to relevance threshold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: str = Field(
        default="Policy Citations", description="Policy query or topic area evaluated"
    )
    reason: str = Field(
        default="No authoritative policy passage met the minimum relevance threshold (0.35)",
        description="Reason for withholding policy citations",
    )
    impact: str = Field(
        default="Policy citations omitted; standard policy baseline rules applied",
        description="Operational impact of abstention",
    )
    is_abstention: bool = Field(default=True, description="Flag indicating abstention status")
    missing_fields: tuple[str, ...] = Field(
        default_factory=tuple, description="Fields affected if applicable"
    )


class BaseReportSection(BaseModel):
    """Abstract baseline for all report sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section_type: SectionType = Field(description="Canonical section type")
    title: str = Field(description="Display title of the section")
    claims: tuple[ClaimReference, ...] = Field(
        default_factory=tuple, description="Grounded claim references within this section"
    )
    evidence_refs: tuple[str, ...] = Field(
        default_factory=tuple, description="All evidence observation IDs referenced in section"
    )
    policy_citation_refs: tuple[str, ...] = Field(
        default_factory=tuple, description="All policy passage IDs referenced in section"
    )
    risk_factor_refs: tuple[str, ...] = Field(
        default_factory=tuple, description="All risk factor names referenced in section"
    )

    @field_validator("risk_factor_refs", mode="before")
    @classmethod
    def normalize_risk_factors(cls, v: Any) -> tuple[str, ...]:
        if isinstance(v, (list, tuple, set)):
            return tuple(item.value if isinstance(item, RiskFactor) else str(item) for item in v)
        return ()


class ExecutiveSummarySection(BaseReportSection):
    """Section 1: High-level narrative summary and overall recommendation."""

    section_type: SectionType = Field(default=SectionType.EXECUTIVE_SUMMARY)
    title: str = Field(default="Executive Summary")
    summary_text: str = Field(description="Comprehensive executive summary narrative")
    outcome: AssessmentOutcome = Field(description="Overall outcome of the assessment")
    key_findings: tuple[str, ...] = Field(
        default_factory=tuple, description="Summary bullet points of primary findings"
    )
    recommendation: str = Field(
        default="Standard review", description="Recommended next action or review posture"
    )


class VehicleIdentitySection(BaseReportSection):
    """Section 2: Verified vehicle identity, specifications, and confidence rating."""

    section_type: SectionType = Field(default=SectionType.VEHICLE_IDENTITY)
    title: str = Field(default="Vehicle Identity")
    vin: str = Field(description="Canonical 17-character VIN")
    make: str | None = Field(default=None, description="Vehicle make/manufacturer")
    model: str | None = Field(default=None, description="Vehicle model name")
    year: int | None = Field(default=None, description="Model year")
    plate: str | None = Field(default=None, description="Registration license plate")
    body_style: str | None = Field(default=None, description="Vehicle body style")
    color: str | None = Field(default=None, description="Exterior color")
    engine_number: str | None = Field(default=None, description="Engine identifier number")
    chassis_number: str | None = Field(default=None, description="Chassis identifier number")
    confidence_score: int | None = Field(
        default=None, ge=0, le=100, description="Overall evidence confidence score"
    )
    confidence_band: ConfidenceBand | None = Field(
        default=None, description="Calibrated confidence band"
    )
    identity_verified: bool = Field(
        default=True, description="Indicates if canonical identity fields were resolved"
    )


class RiskScoreSection(BaseReportSection):
    """Section 3: Deterministic risk score, severity band, and calculation fingerprint."""

    section_type: SectionType = Field(default=SectionType.RISK_SCORE_AND_BAND)
    title: str = Field(default="Risk Score & Band")
    score: int | None = Field(
        default=None, ge=0, le=100, description="Calculated risk score (None if incomplete)"
    )
    band: RiskBand | None = Field(
        default=None, description="Assigned risk band (None if incomplete)"
    )
    raw_score: int | None = Field(
        default=None, description="Uncapped raw score summation (None if incomplete)"
    )
    score_cap: int = Field(default=100, ge=1, le=100)
    policy_id: str = Field(default="risk-policy-v1", description="Policy ID used for calculation")
    policy_version: str = Field(default="v1", description="Policy version evaluated")
    calculation_hash: str = Field(default="", description="Calculation SHA-256 fingerprint")
    is_incomplete: bool = Field(
        default=False, description="True if calculation was withheld due to incomplete evidence"
    )
    scoring_withheld_reason: str | None = Field(
        default=None, description="Reason why scoring was withheld if incomplete"
    )

    @model_validator(mode="after")
    def validate_score_completeness(self) -> RiskScoreSection:
        if self.is_incomplete:
            if self.score is not None or self.band is not None or self.raw_score is not None:
                raise ValueError(
                    "Incomplete RiskScoreSection must not contain numeric score, band, or raw_score"
                )
        else:
            if self.score is None or self.band is None:
                raise ValueError("Complete RiskScoreSection must contain both score and band")
        return self


class MandatoryReviewSection(BaseReportSection):
    """Section 4: Attributable findings for positive risk factors requiring human review."""

    section_type: SectionType = Field(default=SectionType.MANDATORY_REVIEW_FINDINGS)
    title: str = Field(default="Mandatory Review Findings")
    has_mandatory_findings: bool = Field(
        default=False, description="True if one or more mandatory review findings exist"
    )
    findings_count: int = Field(default=0, ge=0)
    findings: tuple[MandatoryFinding, ...] = Field(
        default_factory=tuple, description="Ordered mandatory findings list"
    )

    @model_validator(mode="after")
    def sync_findings_state(self) -> MandatoryReviewSection:
        count = len(self.findings)
        object.__setattr__(self, "findings_count", count)
        object.__setattr__(self, "has_mandatory_findings", count > 0)
        return self


class ContributingFactorsSection(BaseReportSection):
    """Section 5: Detailed factor evaluation breakdown across all policy risk factors."""

    section_type: SectionType = Field(default=SectionType.CONTRIBUTING_FACTORS)
    title: str = Field(default="Contributing Factors")
    factor_breakdown: tuple[RiskFactorResult, ...] = Field(
        default_factory=tuple, description="Evaluation results for all risk factors"
    )
    triggered_factors: tuple[RiskFactor, ...] = Field(
        default_factory=tuple, description="List of positive/triggered risk factors"
    )


class PolicyCitationsSection(BaseReportSection):
    """Section 6: Authoritative policy passages and statutory citations."""

    section_type: SectionType = Field(default=SectionType.POLICY_CITATIONS)
    title: str = Field(default="Policy Citations")
    citations: tuple[PolicyCitation, ...] = Field(
        default_factory=tuple, description="Authoritative cited policy passages"
    )
    has_abstention: bool = Field(
        default=False, description="True if policy retrieval resulted in abstention"
    )
    abstention_notice: AbstentionNotice | None = Field(
        default=None, description="Abstention notice if citations were withheld"
    )


class EvidenceSummarySection(BaseReportSection):
    """Section 7: Upstream evidence snapshot provenance, fields, and conflict summary."""

    section_type: SectionType = Field(default=SectionType.EVIDENCE_SUMMARY)
    title: str = Field(default="Evidence Summary")
    revision_id: str = Field(default="", description="Upstream revision identifier")
    revision_number: int = Field(default=1, ge=1, description="Upstream revision sequence number")
    material_hash: str = Field(default="", description="SHA-256 fingerprint of evidence material")
    as_of: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Upstream observation evaluation timestamp",
    )
    canonical_fields: dict[str, Any] = Field(
        default_factory=dict, description="Resolved canonical vehicle fields"
    )
    conflict_count: int = Field(default=0, ge=0)
    conflicts: tuple[FieldConflict, ...] = Field(
        default_factory=tuple, description="Recorded field conflicts"
    )
    history_depth: int = Field(default=0, ge=0, description="Preceding revision count")


class LimitationsSection(BaseReportSection):
    """Section 8: Explicit disclosures, missing evidence findings, and data boundaries."""

    section_type: SectionType = Field(default=SectionType.LIMITATIONS_AND_MISSING_EVIDENCE)
    title: str = Field(default="Limitations & Missing Evidence")
    standard_limitations: tuple[str, ...] = Field(
        default=(
            (
                "Assessment is based solely on electronic register records "
                "available at the evaluation timestamp."
            ),
            (
                "Physical vehicle condition, internal mechanical defects, "
                "and unregistered modifications are not assessed."
            ),
            (
                "Independent physical inspection and registration verification "
                "are recommended before completing commercial transactions."
            ),
        ),
        description="Standard baseline limitations",
    )
    missing_evidence_notices: tuple[MissingEvidenceNotice, ...] = Field(
        default_factory=tuple, description="Disclosures for missing or unresolved required fields"
    )
    has_missing_evidence: bool = Field(
        default=False, description="True if missing evidence notices are present"
    )
    abstention_notices: tuple[AbstentionNotice, ...] = Field(
        default_factory=tuple, description="Disclosures for policy retrieval abstentions"
    )

    @model_validator(mode="after")
    def sync_missing_state(self) -> LimitationsSection:
        has_missing = len(self.missing_evidence_notices) > 0
        object.__setattr__(self, "has_missing_evidence", has_missing)
        return self


class SyntheticNoticeSection(BaseReportSection):
    """Section 9: Synthetic or demonstration data disclaimers."""

    section_type: SectionType = Field(default=SectionType.SYNTHETIC_DATA_NOTICE)
    title: str = Field(default="Synthetic Data Notice")
    is_synthetic: bool = Field(
        default=False, description="True if any data source contains synthetic data"
    )
    notice: SyntheticNotice | None = Field(
        default=None, description="Synthetic data notice details if applicable"
    )
    disclaimer_text: str | None = Field(default=None, description="Display disclaimer text")


class ReportSections(BaseModel):
    """Strongly-typed container holding all 9 canonical report sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    executive_summary: ExecutiveSummarySection
    vehicle_identity: VehicleIdentitySection
    risk_score_and_band: RiskScoreSection
    mandatory_review_findings: MandatoryReviewSection
    contributing_factors: ContributingFactorsSection
    policy_citations: PolicyCitationsSection
    evidence_summary: EvidenceSummarySection
    limitations_and_missing_evidence: LimitationsSection
    synthetic_data_notice: SyntheticNoticeSection

    def as_list(self) -> list[BaseReportSection]:
        """Return the 9 sections in canonical document order."""
        return [
            self.executive_summary,
            self.vehicle_identity,
            self.risk_score_and_band,
            self.mandatory_review_findings,
            self.contributing_factors,
            self.policy_citations,
            self.evidence_summary,
            self.limitations_and_missing_evidence,
            self.synthetic_data_notice,
        ]

    def get_section(self, section_type: SectionType) -> BaseReportSection:
        """Lookup section by SectionType enum."""
        mapping: dict[SectionType, BaseReportSection] = {
            SectionType.EXECUTIVE_SUMMARY: self.executive_summary,
            SectionType.VEHICLE_IDENTITY: self.vehicle_identity,
            SectionType.RISK_SCORE_AND_BAND: self.risk_score_and_band,
            SectionType.MANDATORY_REVIEW_FINDINGS: self.mandatory_review_findings,
            SectionType.CONTRIBUTING_FACTORS: self.contributing_factors,
            SectionType.POLICY_CITATIONS: self.policy_citations,
            SectionType.EVIDENCE_SUMMARY: self.evidence_summary,
            SectionType.LIMITATIONS_AND_MISSING_EVIDENCE: self.limitations_and_missing_evidence,
            SectionType.SYNTHETIC_DATA_NOTICE: self.synthetic_data_notice,
        }
        return mapping[section_type]


def compute_draft_hash(
    assessment_id: str,
    run_number: int,
    outcome: AssessmentOutcome,
    risk_result_id: str | None,
    sections: ReportSections,
) -> str:
    """Compute deterministic SHA-256 fingerprint for a report draft."""
    payload = {
        "assessment_id": assessment_id,
        "run_number": run_number,
        "outcome": outcome.value,
        "risk_result_id": risk_result_id,
        "sections": {
            s.section_type.value: {
                "title": s.title,
                "evidence_refs": sorted(s.evidence_refs),
                "policy_citation_refs": sorted(s.policy_citation_refs),
                "risk_factor_refs": sorted(s.risk_factor_refs),
                "claims_count": len(s.claims),
            }
            for s in sections.as_list()
        },
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ReportDraft(BaseModel):
    """Authoritative, immutable report draft representing an assessment outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: f"draft-{uuid4()}")
    assessment_id: str = Field(description="Unique assessment identifier")
    run_number: int = Field(ge=1, description="Assessment run sequence number")
    risk_result_id: str | None = Field(
        default=None, description="Associated RiskResult ID if calculated"
    )
    status: ReportDraftStatus = Field(
        default=ReportDraftStatus.DRAFT, description="Lifecycle status of the draft"
    )
    outcome: AssessmentOutcome = Field(
        default=AssessmentOutcome.SCORED, description="SCORED or INCOMPLETE outcome"
    )
    sections: ReportSections = Field(description="The 9 canonical report sections")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    draft_hash: str = Field(default="", description="Deterministic SHA-256 fingerprint")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_and_compute_hash(self) -> ReportDraft:
        # Invariant 1: Completeness alignment with RiskScoreSection
        score_section = self.sections.risk_score_and_band
        if self.outcome == AssessmentOutcome.INCOMPLETE:
            if not score_section.is_incomplete:
                raise ValueError(
                    "INCOMPLETE draft must contain is_incomplete=True RiskScoreSection"
                )
            if score_section.score is not None or score_section.band is not None:
                raise ValueError("INCOMPLETE draft must have score=None and band=None")
        else:
            if score_section.is_incomplete:
                raise ValueError("SCORED draft cannot contain is_incomplete=True RiskScoreSection")
            if score_section.score is None or score_section.band is None:
                raise ValueError("SCORED draft must contain score and band")

        # Invariant 2: Hash calculation
        computed = compute_draft_hash(
            assessment_id=self.assessment_id,
            run_number=self.run_number,
            outcome=self.outcome,
            risk_result_id=self.risk_result_id,
            sections=self.sections,
        )
        if not self.draft_hash:
            object.__setattr__(self, "draft_hash", computed)
        elif self.draft_hash != computed:
            raise ValueError(
                f"Declared draft_hash {self.draft_hash} does not match computed {computed}"
            )
        return self

    @property
    def vin(self) -> str:
        """Vehicle VIN from VehicleIdentitySection."""
        return self.sections.vehicle_identity.vin

    @property
    def policy_id(self) -> str:
        """Policy ID from RiskScoreSection."""
        return self.sections.risk_score_and_band.policy_id

    @property
    def policy_version(self) -> str:
        """Policy version from RiskScoreSection."""
        return self.sections.risk_score_and_band.policy_version

    @property
    def score(self) -> int | None:
        """Calculated score from RiskScoreSection."""
        return self.sections.risk_score_and_band.score

    @property
    def band(self) -> RiskBand | None:
        """Assigned risk band from RiskScoreSection."""
        return self.sections.risk_score_and_band.band

    @property
    def raw_score(self) -> int | None:
        """Raw score summation from RiskScoreSection."""
        return self.sections.risk_score_and_band.raw_score

    @property
    def is_incomplete(self) -> bool:
        """Whether this draft represents an incomplete assessment."""
        return (
            self.outcome == AssessmentOutcome.INCOMPLETE
            or self.sections.risk_score_and_band.is_incomplete
        )

    @property
    def missing_evidence_notices(self) -> tuple[MissingEvidenceNotice, ...]:
        """Missing evidence notices from LimitationsSection."""
        return self.sections.limitations_and_missing_evidence.missing_evidence_notices

    @property
    def synthetic_notice(self) -> SyntheticNotice | None:
        """Synthetic notice from SyntheticNoticeSection."""
        return self.sections.synthetic_data_notice.notice

    @property
    def abstention_notice(self) -> AbstentionNotice | None:
        """Abstention notice from PolicyCitationsSection or LimitationsSection."""
        if self.sections.policy_citations.abstention_notice is not None:
            return self.sections.policy_citations.abstention_notice
        if self.sections.limitations_and_missing_evidence.abstention_notices:
            return self.sections.limitations_and_missing_evidence.abstention_notices[0]
        return None

    @property
    def all_evidence_refs(self) -> tuple[str, ...]:
        """Aggregate sorted, deduplicated evidence refs across all sections and claims."""
        refs: set[str] = set()
        for s in self.sections.as_list():
            refs.update(s.evidence_refs)
            for c in s.claims:
                refs.update(c.evidence_refs)
        return tuple(sorted(refs))

    @property
    def all_policy_citation_refs(self) -> tuple[str, ...]:
        """Aggregate sorted, deduplicated policy citation refs across all sections and claims."""
        refs: set[str] = set()
        for s in self.sections.as_list():
            refs.update(s.policy_citation_refs)
            for c in s.claims:
                refs.update(c.policy_citation_refs)
        return tuple(sorted(refs))

    @property
    def all_risk_factor_refs(self) -> tuple[str, ...]:
        """Aggregate sorted, deduplicated risk factor refs across all sections and claims."""
        refs: set[str] = set()
        for s in self.sections.as_list():
            refs.update(s.risk_factor_refs)
            for c in s.claims:
                refs.update(c.risk_factor_refs)
        return tuple(sorted(refs))

    @property
    def all_claims(self) -> tuple[ClaimReference, ...]:
        """Aggregate all claim references across all sections."""
        claims: list[ClaimReference] = []
        for s in self.sections.as_list():
            claims.extend(s.claims)
        return tuple(claims)
