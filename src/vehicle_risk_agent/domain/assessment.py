"""Assessment aggregate and Assessment Run domain definitions."""

# story: e01s01

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from vehicle_risk_agent.api.models import AssessmentContext


class AssessmentLifecycleState(StrEnum):
    """Lifecycle states for Assessment aggregate root."""

    IN_PROGRESS = "IN_PROGRESS"
    INCOMPLETE = "INCOMPLETE"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    RELEASED = "RELEASED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class AssessmentRunPhase(StrEnum):
    """Execution phases for Assessment Run."""

    PENDING = "PENDING"
    COLLECTING_EVIDENCE = "COLLECTING_EVIDENCE"
    EVALUATING_SUFFICIENCY = "EVALUATING_SUFFICIENCY"
    RETRIEVING_POLICY = "RETRIEVING_POLICY"
    EVALUATING_RISK = "EVALUATING_RISK"
    DRAFTING_REPORT = "DRAFTING_REPORT"
    COMPLETED = "COMPLETED"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"


class AssessmentRun(BaseModel):
    """One immutable attempt/run within an Assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    assessment_id: str
    run_number: int
    phase: AssessmentRunPhase
    created_at: datetime
    updated_at: datetime


class Assessment(BaseModel):
    """Authoritative Assessment aggregate root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    requester_id: str
    vin: str
    context: AssessmentContext
    lifecycle_state: AssessmentLifecycleState
    current_run_number: int
    runs: list[AssessmentRun]
    created_at: datetime
    updated_at: datetime
