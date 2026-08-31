"""Workflow Progress Event domain models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from vehicle_risk_agent.domain.assessment import AssessmentRunPhase


class WorkflowProgressEvent(BaseModel):
    """Sanitized workflow progress event with monotonic sequence per run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    sequence: int = Field(ge=1)
    assessment_id: str
    run_number: int = Field(ge=1)
    phase: AssessmentRunPhase
    safe_message: str = Field(max_length=500)
    timestamp: datetime
