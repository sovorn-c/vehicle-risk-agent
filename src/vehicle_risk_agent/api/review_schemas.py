"""Pydantic request and response schemas for Review API routes."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vehicle_risk_agent.risk.models import AssessmentOutcome


class ApproveReportRequest(BaseModel):
    """Request payload to approve and release a Report Draft."""

    model_config = ConfigDict(extra="forbid")

    run_number: int = Field(default=1, ge=1, description="Target Assessment run sequence number")
    notes: str | None = Field(
        default=None, max_length=1000, description="Optional reviewer notes for SCORED draft"
    )
    acknowledge_missing_evidence: bool = Field(
        default=False, description="Mandatory acknowledgement when approving INCOMPLETE draft"
    )
    rationale: str | None = Field(
        default=None,
        max_length=1000,
        description="Mandatory rationale when approving INCOMPLETE draft",
    )
    draft_outcome: AssessmentOutcome = Field(
        default=AssessmentOutcome.SCORED, description="Declared outcome of draft being approved"
    )


class RejectReportRequest(BaseModel):
    """Request payload to reject a Report Draft."""

    model_config = ConfigDict(extra="forbid")

    run_number: int = Field(default=1, ge=1, description="Target Assessment run sequence number")
    rationale: str = Field(
        min_length=1, max_length=1000, description="Mandatory rejection rationale"
    )

    @field_validator("rationale")
    @classmethod
    def validate_rationale_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Rejection rationale cannot be empty or whitespace")
        return v.strip()


class ReviewDecisionResponse(BaseModel):
    """Response returned upon recording a Review Action."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(description="Unique Review Action identifier")
    assessment_id: str = Field(description="Assessment identifier")
    run_number: int = Field(description="Assessment run sequence number")
    reviewer_id: str = Field(description="Reviewer principal ID")
    action_type: str = Field(
        description="APPROVE_REPORT, REJECT_REPORT, or REQUEST_REINVESTIGATION"
    )
    disposition: str = Field(description="Derived Report Draft disposition")
    assessment_state: str = Field(description="Updated Assessment aggregate lifecycle state")
    notes: str | None = Field(default=None, description="Recorded reviewer notes")
    rationale: str | None = Field(default=None, description="Recorded reviewer rationale")
    created_at: datetime = Field(description="Timestamp of action recording")
    action_hash: str = Field(description="Deterministic SHA-256 fingerprint")
    released_report_id: str | None = Field(default=None, description="Report Draft ID if released")
