"""Human review and report decision contracts, models, and idempotency hashing."""

# story: e05s01

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState
from vehicle_risk_agent.reporting.models import (
    AbstentionNotice,
    EvidenceSummarySection,
    MissingEvidenceNotice,
    ReportDraft,
    ReportSections,
    SyntheticNotice,
)
from vehicle_risk_agent.risk.models import AssessmentOutcome, RiskBand


class ReviewActionType(StrEnum):
    """Mutually exclusive Review Action types."""

    APPROVE_REPORT = "APPROVE_REPORT"
    REJECT_REPORT = "REJECT_REPORT"
    REQUEST_REINVESTIGATION = "REQUEST_REINVESTIGATION"


class ReportDisposition(StrEnum):
    """Derived disposition of a Report Draft based on Review Action."""

    PENDING_REVIEW = "PENDING_REVIEW"
    RELEASED = "RELEASED"
    REJECTED = "REJECTED"
    REINVESTIGATION_REQUESTED = "REINVESTIGATION_REQUESTED"


def derive_report_disposition(action_type: ReviewActionType | None) -> ReportDisposition:
    """Derive Report Draft disposition from recorded Review Action type."""
    if action_type is None:
        return ReportDisposition.PENDING_REVIEW
    if action_type == ReviewActionType.APPROVE_REPORT:
        return ReportDisposition.RELEASED
    if action_type == ReviewActionType.REJECT_REPORT:
        return ReportDisposition.REJECTED
    if action_type == ReviewActionType.REQUEST_REINVESTIGATION:
        return ReportDisposition.REINVESTIGATION_REQUESTED
    return ReportDisposition.PENDING_REVIEW


def derive_assessment_state(action_type: ReviewActionType) -> AssessmentLifecycleState:
    """Derive Assessment aggregate lifecycle state transition from Review Action type."""
    if action_type == ReviewActionType.APPROVE_REPORT:
        return AssessmentLifecycleState.RELEASED
    if action_type == ReviewActionType.REJECT_REPORT:
        return AssessmentLifecycleState.REJECTED
    if action_type == ReviewActionType.REQUEST_REINVESTIGATION:
        return AssessmentLifecycleState.IN_PROGRESS
    raise ValueError(f"Unknown ReviewActionType: {action_type}")


class PinnedVersions(BaseModel):
    """Immutable version references pinned at run allocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_id: str | None = Field(default=None, description="Active Policy Corpus ID")
    corpus_manifest_hash: str | None = Field(default=None, description="Active Policy Corpus hash")
    risk_policy_id: str | None = Field(default=None, description="Active Risk Policy ID")
    risk_policy_version: str | None = Field(
        default=None, description="Active Risk Policy version string"
    )
    risk_policy_hash: str | None = Field(default=None, description="Active Risk Policy hash")
    mcp_contract_version: str = Field(default="v1", description="Pinned MCP contract version")
    model_version: str = Field(
        default="claude-3-5-sonnet-20241022", description="Pinned drafting model version"
    )
    prompt_version: str = Field(default="v1", description="Pinned prompt version")


class ApproveReportCommand(BaseModel):
    """Strict command to release one Report Draft as a Reviewed Released Report.

    APPROVE_REPORT for SCORED permits optional notes.
    APPROVE_REPORT for INCOMPLETE requires explicit missing-evidence acknowledgement and rationale.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_id: str = Field(min_length=1, description="Target Assessment identifier")
    run_number: int = Field(ge=1, description="Target Assessment run sequence number")
    reviewer_id: str = Field(min_length=1, description="Authenticated Reviewer principal ID")
    idempotency_key: str = Field(
        min_length=1, max_length=128, description="Reviewer-scoped idempotency key"
    )
    action_type: Literal[ReviewActionType.APPROVE_REPORT] = Field(
        default=ReviewActionType.APPROVE_REPORT,
        description="Fixed action type for approve commands",
    )
    notes: str | None = Field(
        default=None, max_length=1000, description="Optional reviewer notes for SCORED draft"
    )
    acknowledge_missing_evidence: bool = Field(
        default=False, description="Mandatory acknowledgement flag when approving INCOMPLETE draft"
    )
    rationale: str | None = Field(
        default=None,
        max_length=1000,
        description="Mandatory rationale when approving INCOMPLETE draft",
    )
    draft_outcome: AssessmentOutcome = Field(
        default=AssessmentOutcome.SCORED,
        description="Declared draft outcome being approved",
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: ReviewActionType) -> ReviewActionType:
        if v != ReviewActionType.APPROVE_REPORT:
            raise ValueError("ApproveReportCommand action_type must be APPROVE_REPORT")
        return v

    @model_validator(mode="after")
    def validate_conditional_rationale(self) -> ApproveReportCommand:
        if self.draft_outcome == AssessmentOutcome.INCOMPLETE:
            if not self.acknowledge_missing_evidence:
                raise ValueError(
                    "APPROVE_REPORT for INCOMPLETE report draft requires "
                    "explicit missing-evidence acknowledgement"
                )
            if self.rationale is None or not self.rationale.strip():
                raise ValueError(
                    "APPROVE_REPORT for INCOMPLETE report draft requires bounded rationale"
                )
        return self

    def validate_for_draft(self, draft: ReportDraft) -> None:
        """Validate command against the authoritative target ReportDraft."""
        if self.run_number != draft.run_number:
            raise ValueError(
                f"Command run_number {self.run_number} does not match "
                f"draft run_number {draft.run_number}"
            )
        if self.assessment_id != draft.assessment_id:
            raise ValueError(
                f"Command assessment_id {self.assessment_id} does not match "
                f"draft assessment_id {draft.assessment_id}"
            )
        if draft.outcome == AssessmentOutcome.INCOMPLETE or draft.is_incomplete:
            if not self.acknowledge_missing_evidence:
                raise ValueError(
                    "APPROVE_REPORT for INCOMPLETE report draft requires "
                    "explicit missing-evidence acknowledgement"
                )
            if self.rationale is None or not self.rationale.strip():
                raise ValueError(
                    "APPROVE_REPORT for INCOMPLETE report draft requires bounded rationale"
                )


class RejectReportCommand(BaseModel):
    """Strict command to reject one Report Draft.

    REJECT_REPORT always requires bounded non-empty rationale.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_id: str = Field(min_length=1, description="Target Assessment identifier")
    run_number: int = Field(ge=1, description="Target Assessment run sequence number")
    reviewer_id: str = Field(min_length=1, description="Authenticated Reviewer principal ID")
    idempotency_key: str = Field(
        min_length=1, max_length=128, description="Reviewer-scoped idempotency key"
    )
    action_type: Literal[ReviewActionType.REJECT_REPORT] = Field(
        default=ReviewActionType.REJECT_REPORT,
        description="Fixed action type for reject commands",
    )
    rationale: str = Field(
        min_length=1, max_length=1000, description="Mandatory rejection rationale"
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: ReviewActionType) -> ReviewActionType:
        if v != ReviewActionType.REJECT_REPORT:
            raise ValueError("RejectReportCommand action_type must be REJECT_REPORT")
        return v

    @field_validator("rationale")
    @classmethod
    def validate_rationale_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("REJECT_REPORT command requires bounded, non-empty rationale")
        return v.strip()


ALLOWED_EVIDENCE_TARGETS: frozenset[str] = frozenset(
    {
        "ppsr_result",
        "stolen_status",
        "writeoff_status",
        "odometer_reading",
        "registration_status",
        "ownership_history",
        "wof_status",
        "safety_recall",
        "inspection_history",
        "make",
        "model",
        "year",
        "plate",
        "is_commercial",
        "vehicle_usage",
        "damage_history",
        "fuel_type",
    }
)


class RequestReinvestigationCommand(BaseModel):
    """Strict command to request an additive reinvestigation for an Assessment.

    REQUEST_REINVESTIGATION carries one to five 200-character questions
    and/or allowed evidence targets, always with non-empty rationale.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_id: str = Field(min_length=1, description="Target Assessment identifier")
    run_number: int = Field(ge=1, le=3, description="Target Assessment run sequence number")
    reviewer_id: str = Field(min_length=1, description="Authenticated Reviewer principal ID")
    idempotency_key: str = Field(
        min_length=1, max_length=128, description="Reviewer-scoped idempotency key"
    )
    action_type: Literal[ReviewActionType.REQUEST_REINVESTIGATION] = Field(
        default=ReviewActionType.REQUEST_REINVESTIGATION,
        description="Fixed action type for reinvestigation requests",
    )
    rationale: str = Field(
        min_length=1, max_length=1000, description="Mandatory reinvestigation rationale"
    )
    questions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Additive questions (max 200 chars each)",
    )
    evidence_targets: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Additive allowed evidence targets",
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: ReviewActionType) -> ReviewActionType:
        if v != ReviewActionType.REQUEST_REINVESTIGATION:
            raise ValueError(
                "RequestReinvestigationCommand action_type must be REQUEST_REINVESTIGATION"
            )
        return v

    @field_validator("rationale")
    @classmethod
    def validate_rationale_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "REQUEST_REINVESTIGATION command requires bounded, non-empty rationale"
            )
        return v.strip()

    @field_validator("questions", mode="before")
    @classmethod
    def coerce_questions(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return tuple(v)
        return v

    @field_validator("evidence_targets", mode="before")
    @classmethod
    def coerce_evidence_targets(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return tuple(v)
        return v

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for q in v:
            if not isinstance(q, str) or not q.strip():
                raise ValueError("Reinvestigation question cannot be empty or whitespace")
            stripped = q.strip()
            if len(stripped) > 200:
                raise ValueError(
                    f"Reinvestigation question exceeds 200 characters: {len(stripped)}"
                )
            cleaned.append(stripped)
        return tuple(cleaned)

    @field_validator("evidence_targets")
    @classmethod
    def validate_evidence_targets(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for t in v:
            if not isinstance(t, str) or not t.strip():
                raise ValueError("Evidence target cannot be empty or whitespace")
            norm_t = t.strip().lower()
            if norm_t not in ALLOWED_EVIDENCE_TARGETS:
                targets_str = ", ".join(sorted(ALLOWED_EVIDENCE_TARGETS))
                raise ValueError(
                    f"Evidence target '{t}' is not in allowed evidence targets: {targets_str}"
                )
            cleaned.append(norm_t)
        return tuple(cleaned)

    @model_validator(mode="after")
    def validate_cardinality(self) -> RequestReinvestigationCommand:
        total = len(self.questions) + len(self.evidence_targets)
        if total < 1 or total > 5:
            raise ValueError(
                "REQUEST_REINVESTIGATION requires between 1 and 5 questions "
                f"and/or evidence targets, got {total}"
            )
        return self


def compute_review_action_hash(
    assessment_id: str,
    run_number: int,
    reviewer_id: str,
    action_type: ReviewActionType,
    disposition: ReportDisposition,
    idempotency_key: str,
    rationale: str | None,
    notes: str | None,
    acknowledge_missing_evidence: bool,
    questions: tuple[str, ...] = (),
    evidence_targets: tuple[str, ...] = (),
) -> str:
    """Compute deterministic SHA-256 fingerprint for a Review Action."""
    payload: dict[str, Any] = {
        "assessment_id": assessment_id,
        "run_number": run_number,
        "reviewer_id": reviewer_id,
        "action_type": action_type.value,
        "disposition": disposition.value,
        "idempotency_key": idempotency_key,
        "rationale": rationale,
        "notes": notes,
        "acknowledge_missing_evidence": acknowledge_missing_evidence,
    }
    if questions:
        payload["questions"] = list(questions)
    if evidence_targets:
        payload["evidence_targets"] = list(evidence_targets)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ReviewAction(BaseModel):
    """Authoritative, immutable persistent record of a Reviewer decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    assessment_id: str = Field(description="Target Assessment identifier")
    run_number: int = Field(ge=1, description="Target Assessment run sequence number")
    reviewer_id: str = Field(description="Principal ID of the Reviewer")
    action_type: ReviewActionType = Field(description="Action type enum")
    disposition: ReportDisposition = Field(
        default=ReportDisposition.PENDING_REVIEW,
        description="Derived disposition corresponding to action",
    )
    idempotency_key: str = Field(description="Reviewer-scoped idempotency key")
    rationale: str | None = Field(default=None, description="Rationale text if provided")
    notes: str | None = Field(default=None, description="Reviewer notes if provided")
    acknowledge_missing_evidence: bool = Field(
        default=False, description="Flag acknowledging missing evidence"
    )
    questions: tuple[str, ...] = Field(
        default_factory=tuple, description="Additive questions if reinvestigation"
    )
    evidence_targets: tuple[str, ...] = Field(
        default_factory=tuple, description="Additive evidence targets if reinvestigation"
    )
    pinned_versions: PinnedVersions | None = Field(
        default=None, description="Active versions pinned if reinvestigation"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    action_hash: str = Field(default="", description="Deterministic SHA-256 fingerprint")

    @field_validator("questions", mode="before")
    @classmethod
    def coerce_questions(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return tuple(v)
        return v

    @field_validator("evidence_targets", mode="before")
    @classmethod
    def coerce_evidence_targets(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return tuple(v)
        return v

    @model_validator(mode="after")
    def derive_disposition_and_hash(self) -> ReviewAction:
        derived_disp = derive_report_disposition(self.action_type)
        if self.disposition != derived_disp:
            object.__setattr__(self, "disposition", derived_disp)

        computed = compute_review_action_hash(
            assessment_id=self.assessment_id,
            run_number=self.run_number,
            reviewer_id=self.reviewer_id,
            action_type=self.action_type,
            disposition=derived_disp,
            idempotency_key=self.idempotency_key,
            rationale=self.rationale,
            notes=self.notes,
            acknowledge_missing_evidence=self.acknowledge_missing_evidence,
            questions=self.questions,
            evidence_targets=self.evidence_targets,
        )
        if not self.action_hash:
            object.__setattr__(self, "action_hash", computed)
        elif self.action_hash != computed:
            raise ValueError(
                f"Declared action_hash {self.action_hash} does not match computed {computed}"
            )
        return self


class ReleasedReport(BaseModel):
    """Attributable released report projection.

    Formed by applying an APPROVE_REPORT ReviewAction to an immutable ReportDraft.
    Preserves original SCORED or INCOMPLETE outcome without mutating report content.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_draft: ReportDraft = Field(description="Underlying immutable Report Draft")
    review_action: ReviewAction = Field(description="Review Action that released this report")
    released_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when report was approved and released",
    )
    disposition: ReportDisposition = Field(
        default=ReportDisposition.RELEASED,
        description="Must be RELEASED",
    )

    @model_validator(mode="after")
    def validate_released_invariants(self) -> ReleasedReport:
        if self.review_action.action_type != ReviewActionType.APPROVE_REPORT:
            raise ValueError("ReleasedReport requires an APPROVE_REPORT Review Action")
        if self.disposition != ReportDisposition.RELEASED:
            raise ValueError("ReleasedReport disposition must be RELEASED")
        return self

    @property
    def id(self) -> str:
        return self.report_draft.id

    @property
    def assessment_id(self) -> str:
        return self.report_draft.assessment_id

    @property
    def run_number(self) -> int:
        return self.report_draft.run_number

    @property
    def vin(self) -> str:
        return self.report_draft.vin

    @property
    def outcome(self) -> AssessmentOutcome:
        return self.report_draft.outcome

    @property
    def score(self) -> int | None:
        return self.report_draft.score

    @property
    def band(self) -> RiskBand | None:
        return self.report_draft.band

    @property
    def raw_score(self) -> int | None:
        return self.report_draft.raw_score

    @property
    def is_incomplete(self) -> bool:
        return self.report_draft.is_incomplete

    @property
    def missing_evidence_notices(self) -> tuple[MissingEvidenceNotice, ...]:
        return self.report_draft.missing_evidence_notices

    @property
    def synthetic_notice(self) -> SyntheticNotice | None:
        return self.report_draft.synthetic_notice

    @property
    def abstention_notice(self) -> AbstentionNotice | None:
        return self.report_draft.abstention_notice

    @property
    def sections(self) -> ReportSections:
        return self.report_draft.sections

    @property
    def draft_hash(self) -> str:
        return self.report_draft.draft_hash


def compute_review_payload_hash(
    command: ApproveReportCommand | RejectReportCommand | BaseModel,
) -> str:
    """Compute deterministic SHA-256 fingerprint of normalized review command content."""
    data = json.dumps(command.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class RunHistoryItem(BaseModel):
    """Immutable projection of an individual Assessment run in audit history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_number: int = Field(ge=1, description="Run sequence number")
    phase: str = Field(description="Run phase")
    draft: ReportDraft | None = Field(default=None, description="Report draft for this run")
    evidence_summary: EvidenceSummarySection | None = Field(
        default=None, description="Evidence summary for this run"
    )
    pinned_versions: PinnedVersions | None = Field(
        default=None, description="Pinned versions for this run"
    )
    review_action: ReviewAction | None = Field(
        default=None, description="Review action for this run"
    )
    created_at: datetime = Field(description="Run creation timestamp")
    updated_at: datetime = Field(description="Run update timestamp")


class AssessmentHistory(BaseModel):
    """Authoritative audit projection for an Assessment and its complete run/review history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_id: str = Field(description="Unique assessment identifier")
    vin: str = Field(description="Vehicle VIN")
    requester_id: str = Field(description="Owning requester ID")
    lifecycle_state: AssessmentLifecycleState = Field(
        description="Current aggregate assessment lifecycle state"
    )
    current_run_number: int = Field(ge=1, description="Current run sequence number")
    disposition: ReportDisposition = Field(
        default=ReportDisposition.PENDING_REVIEW,
        description="Current report disposition",
    )
    runs: tuple[RunHistoryItem, ...] = Field(
        default=(), description="Ordered immutable run history items"
    )
    released_report: ReleasedReport | None = Field(
        default=None, description="Released report if assessment was approved"
    )
    created_at: datetime = Field(description="Assessment creation timestamp")
    updated_at: datetime = Field(description="Assessment update timestamp")
