"""Tests for monotonic Workflow Progress Event contract and sequence."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.domain.events import WorkflowProgressEvent


def test_workflow_progress_event_valid() -> None:
    """Verify well-formed progress event parses and enforces constraints."""
    event = WorkflowProgressEvent(
        event_id="evt-1",
        sequence=1,
        assessment_id="asmt-1",
        run_number=1,
        phase=AssessmentRunPhase.COLLECTING_EVIDENCE,
        safe_message="Gathering vehicle intelligence facts",
        timestamp=datetime.now(timezone.utc),
    )
    assert event.sequence == 1
    assert event.phase == AssessmentRunPhase.COLLECTING_EVIDENCE
    assert event.safe_message == "Gathering vehicle intelligence facts"


def test_workflow_progress_event_rejects_negative_or_zero_sequence() -> None:
    """Verify sequence must be a positive integer (>= 1)."""
    with pytest.raises(ValidationError):
        WorkflowProgressEvent(
            event_id="evt-1",
            sequence=0,
            assessment_id="asmt-1",
            run_number=1,
            phase=AssessmentRunPhase.PENDING,
            safe_message="Initial intake",
            timestamp=datetime.now(timezone.utc),
        )


def test_workflow_progress_event_rejects_extra_fields() -> None:
    """Verify strict validation rejects unsanitized extra fields."""
    with pytest.raises(ValidationError):
        WorkflowProgressEvent.model_validate(
            {
                "event_id": "evt-1",
                "sequence": 1,
                "assessment_id": "asmt-1",
                "run_number": 1,
                "phase": "PENDING",
                "safe_message": "Intake",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_observation_leak": {"internal": "secret"},
            }
        )
