"""Human review and report decision contracts, models, and idempotency hashing."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState
from vehicle_risk_agent.reporting.models import (
    AbstentionNotice,
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
    action_type: ReviewActionType = Field(default=ReviewActionType.APPROVE_REPORT)
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
    action_type: ReviewActionType = Field(default=ReviewActionType.REJECT_REPORT)
    rationale: str = Field(
        min_length=1, max_length=1000, description="Mandatory rejection rationale"
    )

    @field_validator("rationale")
    @classmethod
    def validate_rationale_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("REJECT_REPORT command requires bounded, non-empty rationale")
        return v.strip()


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
) -> str:
    """Compute deterministic SHA-256 fingerprint for a Review Action."""
    payload = {
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
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ReviewAction(BaseModel):
    """Authoritative, immutable persistent record of a Reviewer decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: f"action-{uuid4()}")
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    action_hash: str = Field(default="", description="Deterministic SHA-256 fingerprint")

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
