"""Unit tests for Reinvestigation domain models, commands, and contracts (e05s02-t01)."""

# story: e05s02

from uuid import uuid4

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState
from vehicle_risk_agent.review.models import (
    ReportDisposition,
    RequestReinvestigationCommand,
    ReviewAction,
    ReviewActionType,
    compute_review_action_hash,
    derive_assessment_state,
    derive_report_disposition,
)


class TestRequestReinvestigationCommandValidation:
    """Test suite for REQUEST_REINVESTIGATION command validation and bounds."""

    def test_valid_command_with_questions_only(self) -> None:
        cmd = RequestReinvestigationCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="key-001",
            rationale="Need clarification on mileage discrepancy",
            questions=("Can we verify the current odometer reading?",),
        )
        assert cmd.assessment_id == "asmt-001"
        assert cmd.run_number == 1
        assert cmd.action_type == ReviewActionType.REQUEST_REINVESTIGATION
        assert cmd.rationale == "Need clarification on mileage discrepancy"
        assert cmd.questions == ("Can we verify the current odometer reading?",)
        assert cmd.evidence_targets == ()

    def test_valid_command_with_evidence_targets_only(self) -> None:
        cmd = RequestReinvestigationCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="key-002",
            rationale="Missing PPSR and stolen status checks",
            evidence_targets=("ppsr_result", "stolen_status"),
        )
        assert cmd.evidence_targets == ("ppsr_result", "stolen_status")
        assert cmd.questions == ()

    def test_valid_command_with_both_questions_and_targets_within_limit(self) -> None:
        cmd = RequestReinvestigationCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="key-003",
            rationale="Reinvestigate vehicle provenance",
            questions=("Was vehicle imported damaged?", "Confirm current WoF status"),
            evidence_targets=("wof_status", "damage_history"),
        )
        assert len(cmd.questions) + len(cmd.evidence_targets) == 4

    def test_valid_command_at_max_cardinality_five(self) -> None:
        cmd = RequestReinvestigationCommand(
            assessment_id="asmt-001",
            run_number=2,
            reviewer_id="rev-001",
            idempotency_key="key-004",
            rationale="Check five specific targets",
            questions=("Q1?", "Q2?"),
            evidence_targets=("ppsr_result", "stolen_status", "writeoff_status"),
        )
        assert len(cmd.questions) + len(cmd.evidence_targets) == 5

    def test_rejects_empty_questions_and_targets(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            RequestReinvestigationCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="key-err-1",
                rationale="No questions or targets provided",
                questions=(),
                evidence_targets=(),
            )
        err_msg = str(exc_info.value).lower()
        assert "cardinality" in err_msg or "between 1 and 5" in err_msg

    def test_rejects_total_cardinality_exceeding_five(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            RequestReinvestigationCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="key-err-2",
                rationale="Too many questions and targets",
                questions=("Q1?", "Q2?", "Q3?"),
                evidence_targets=("ppsr_result", "stolen_status", "writeoff_status"),
            )
        err_msg = str(exc_info.value).lower()
        assert "between 1 and 5" in err_msg or "cardinality" in err_msg

    def test_rejects_question_exceeding_200_characters(self) -> None:
        long_q = "A" * 201
        with pytest.raises(ValidationError) as exc_info:
            RequestReinvestigationCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="key-err-3",
                rationale="Valid rationale",
                questions=(long_q,),
            )
        assert "200" in str(exc_info.value)

    def test_rejects_empty_or_whitespace_question(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            RequestReinvestigationCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="key-err-4",
                rationale="Valid rationale",
                questions=("   ",),
            )
        assert "empty" in str(exc_info.value).lower() or "whitespace" in str(exc_info.value).lower()

    def test_rejects_disallowed_evidence_target(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            RequestReinvestigationCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="key-err-5",
                rationale="Valid rationale",
                evidence_targets=("invalid_arbitrary_field",),
            )
        err_msg = str(exc_info.value).lower()
        assert "not in allowed evidence targets" in err_msg or "allowed" in err_msg

    def test_rejects_missing_or_blank_rationale(self) -> None:
        with pytest.raises(ValidationError):
            RequestReinvestigationCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="key-err-6",
                rationale="   ",
                questions=("Valid question?",),
            )

    def test_rejects_run_number_beyond_three_run_limit(self) -> None:
        with pytest.raises(ValidationError):
            RequestReinvestigationCommand(
                assessment_id="asmt-001",
                run_number=4,
                reviewer_id="rev-001",
                idempotency_key="key-err-7",
                rationale="Valid rationale",
                questions=("Valid question?",),
            )

    def test_coerces_lists_to_tuples(self) -> None:
        cmd = RequestReinvestigationCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="key-coerce",
            rationale="Coercion check",
            questions=["Question 1?", "Question 2?"],  # type: ignore[arg-type]
            evidence_targets=["ppsr_result"],  # type: ignore[arg-type]
        )
        assert isinstance(cmd.questions, tuple)
        assert isinstance(cmd.evidence_targets, tuple)

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            RequestReinvestigationCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="key-extra",
                rationale="Valid rationale",
                questions=("Valid question?",),
                extra_payload="malicious_injection",  # type: ignore[call-arg]
            )


class TestReinvestigationDispositionAndState:
    """Test suite for derived disposition and lifecycle transitions."""

    def test_derive_report_disposition_for_reinvestigation(self) -> None:
        disp = derive_report_disposition(ReviewActionType.REQUEST_REINVESTIGATION)
        assert disp == ReportDisposition.REINVESTIGATION_REQUESTED

    def test_derive_assessment_state_for_reinvestigation(self) -> None:
        state = derive_assessment_state(ReviewActionType.REQUEST_REINVESTIGATION)
        assert state == AssessmentLifecycleState.IN_PROGRESS

    def test_review_action_hash_reproducibility(self) -> None:
        h1 = compute_review_action_hash(
            assessment_id="asmt-100",
            run_number=1,
            reviewer_id="rev-1",
            action_type=ReviewActionType.REQUEST_REINVESTIGATION,
            disposition=ReportDisposition.REINVESTIGATION_REQUESTED,
            idempotency_key="key-hash-1",
            rationale="Rationale text",
            notes=None,
            acknowledge_missing_evidence=False,
            questions=("Question 1?",),
            evidence_targets=("odometer_reading",),
        )
        h2 = compute_review_action_hash(
            assessment_id="asmt-100",
            run_number=1,
            reviewer_id="rev-1",
            action_type=ReviewActionType.REQUEST_REINVESTIGATION,
            disposition=ReportDisposition.REINVESTIGATION_REQUESTED,
            idempotency_key="key-hash-1",
            rationale="Rationale text",
            notes=None,
            acknowledge_missing_evidence=False,
            questions=("Question 1?",),
            evidence_targets=("odometer_reading",),
        )
        assert h1 == h2
        assert len(h1) == 64

    def test_review_action_model_with_reinvestigation(self) -> None:
        action = ReviewAction(
            id=str(uuid4()),
            assessment_id="asmt-100",
            run_number=1,
            reviewer_id="rev-1",
            action_type=ReviewActionType.REQUEST_REINVESTIGATION,
            idempotency_key="key-action-1",
            rationale="Need more data",
            questions=("Question 1?",),
            evidence_targets=("ppsr_result",),
        )
        assert action.disposition == ReportDisposition.REINVESTIGATION_REQUESTED
        assert action.questions == ("Question 1?",)
        assert action.evidence_targets == ("ppsr_result",)
        assert len(action.action_hash) == 64
