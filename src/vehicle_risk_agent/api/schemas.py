"""Assessment API response models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from vehicle_risk_agent.api.models import AssessmentContext
from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState, AssessmentRunPhase


class AssessmentRunResponse(BaseModel):
    """Response schema for an Assessment Run."""

    model_config = ConfigDict(extra="forbid")

    id: str
    run_number: int
    phase: AssessmentRunPhase
    created_at: datetime
    updated_at: datetime


class AssessmentResponse(BaseModel):
    """Response schema for an Assessment."""

    model_config = ConfigDict(extra="forbid")

    id: str
    requester_id: str
    vin: str
    context: AssessmentContext
    lifecycle_state: AssessmentLifecycleState
    current_run_number: int
    runs: list[AssessmentRunResponse]
    created_at: datetime
    updated_at: datetime


class ErrorDetail(BaseModel):
    """Safe structured error representation."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Error envelope returned by API."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail
