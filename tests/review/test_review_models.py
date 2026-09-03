"""Unit tests for Review domain models, commands, and idempotency contracts (e05s01-t01)."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState
from vehicle_risk_agent.reporting.models import (
    ContributingFactorsSection,
    EvidenceSummarySection,
    ExecutiveSummarySection,
    LimitationsSection,
    MandatoryReviewSection,
    MissingEvidenceNotice,
    PolicyCitationsSection,
    ReportDraft,
    ReportDraftStatus,
    ReportSections,
    RiskScoreSection,
    SyntheticNoticeSection,
    VehicleIdentitySection,
)
from vehicle_risk_agent.review.models import (
    ApproveReportCommand,
    RejectReportCommand,
    ReleasedReport,
    ReportDisposition,
    ReviewAction,
    ReviewActionType,
    compute_review_payload_hash,
    derive_assessment_state,
    derive_report_disposition,
)
from vehicle_risk_agent.risk.models import AssessmentOutcome, RiskBand


def _build_dummy_sections(incomplete: bool = False) -> ReportSections:
    """Helper to build valid ReportSections for testing."""
    return ReportSections(
        executive_summary=ExecutiveSummarySection(
            summary_text="Executive summary",
            outcome=AssessmentOutcome.INCOMPLETE if incomplete else AssessmentOutcome.SCORED,
        ),
        vehicle_identity=VehicleIdentitySection(
            vin="1HGCR2F85HA000000",
            make="HONDA",
            model="ACCORD",
            year=2017,
        ),
        risk_score_and_band=RiskScoreSection(
            score=None if incomplete else 35,
            band=None if incomplete else RiskBand.LOW,
            raw_score=None if incomplete else 35,
            is_incomplete=incomplete,
            scoring_withheld_reason="Missing odometer" if incomplete else None,
        ),
        mandatory_review_findings=MandatoryReviewSection(),
        contributing_factors=ContributingFactorsSection(),
        policy_citations=PolicyCitationsSection(),
        evidence_summary=EvidenceSummarySection(),
        limitations_and_missing_evidence=LimitationsSection(
            missing_evidence_notices=(
                (
                    MissingEvidenceNotice(
                        field_name="odometer_reading",
                        reason="MISSING",
                        details="No odometer reading found",
                    ),
                )
                if incomplete
                else ()
            )
        ),
        synthetic_data_notice=SyntheticNoticeSection(),
    )


def _build_test_draft(incomplete: bool = False) -> ReportDraft:
    """Helper to build valid ReportDraft."""
    return ReportDraft(
        id=f"draft-{uuid4()}",
        assessment_id=f"asmt-{uuid4()}",
        run_number=1,
        status=ReportDraftStatus.AWAITING_REVIEW,
        outcome=AssessmentOutcome.INCOMPLETE if incomplete else AssessmentOutcome.SCORED,
        sections=_build_dummy_sections(incomplete=incomplete),
    )


class TestApproveReportCommand:
    """Test suite for APPROVE_REPORT command validation contracts."""

    def test_approve_scored_report_without_notes_succeeds(self) -> None:
        cmd = ApproveReportCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="idemp-app-01",
        )
        assert cmd.assessment_id == "asmt-001"
        assert cmd.run_number == 1
        assert cmd.reviewer_id == "rev-001"
        assert cmd.idempotency_key == "idemp-app-01"
        assert cmd.notes is None
        assert cmd.acknowledge_missing_evidence is False
        assert cmd.rationale is None

    def test_approve_scored_report_with_notes_succeeds(self) -> None:
        cmd = ApproveReportCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="idemp-app-01",
            notes="Standard approval after visual review",
        )
        assert cmd.notes == "Standard approval after visual review"

    def test_approve_scored_report_notes_exceeding_max_length_fails(self) -> None:
        with pytest.raises(ValidationError):
            ApproveReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="idemp-app-01",
                notes="x" * 1001,
            )

    def test_approve_incomplete_report_missing_acknowledgement_fails(self) -> None:
        with pytest.raises(ValidationError, match="acknowledgement"):
            ApproveReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="idemp-app-01",
                draft_outcome=AssessmentOutcome.INCOMPLETE,
                acknowledge_missing_evidence=False,
                rationale="Proceeding despite missing odometer due to low value vehicle",
            )

    def test_approve_incomplete_report_missing_rationale_fails(self) -> None:
        with pytest.raises(ValidationError, match="rationale"):
            ApproveReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="idemp-app-01",
                draft_outcome=AssessmentOutcome.INCOMPLETE,
                acknowledge_missing_evidence=True,
                rationale=None,
            )

    def test_approve_incomplete_report_whitespace_rationale_fails(self) -> None:
        with pytest.raises(ValidationError, match="rationale"):
            ApproveReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="idemp-app-01",
                draft_outcome=AssessmentOutcome.INCOMPLETE,
                acknowledge_missing_evidence=True,
                rationale="    ",
            )

    def test_approve_incomplete_report_with_acknowledgement_and_rationale_succeeds(self) -> None:
        cmd = ApproveReportCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="idemp-app-01",
            draft_outcome=AssessmentOutcome.INCOMPLETE,
            acknowledge_missing_evidence=True,
            rationale="Approved as-is with explicit acknowledgement of missing odometer",
        )
        assert cmd.acknowledge_missing_evidence is True
        assert cmd.rationale == "Approved as-is with explicit acknowledgement of missing odometer"

    def test_approve_command_validate_against_draft(self) -> None:
        scored_draft = _build_test_draft(incomplete=False)
        incomplete_draft = _build_test_draft(incomplete=True)

        # Scored draft accepts basic command
        basic_cmd = ApproveReportCommand(
            assessment_id=scored_draft.assessment_id,
            run_number=scored_draft.run_number,
            reviewer_id="rev-001",
            idempotency_key="idemp-01",
        )
        basic_cmd.validate_for_draft(scored_draft)

        # Incomplete draft rejects basic command lacking acknowledgement
        basic_incomplete_cmd = ApproveReportCommand(
            assessment_id=incomplete_draft.assessment_id,
            run_number=incomplete_draft.run_number,
            reviewer_id="rev-001",
            idempotency_key="idemp-01",
        )
        with pytest.raises(ValueError, match="acknowledgement"):
            basic_incomplete_cmd.validate_for_draft(incomplete_draft)

        # Incomplete draft accepts compliant command
        compliant_cmd = ApproveReportCommand(
            assessment_id=incomplete_draft.assessment_id,
            run_number=incomplete_draft.run_number,
            reviewer_id="rev-001",
            idempotency_key="idemp-02",
            draft_outcome=AssessmentOutcome.INCOMPLETE,
            acknowledge_missing_evidence=True,
            rationale="Valid rationale for incomplete draft",
        )
        compliant_cmd.validate_for_draft(incomplete_draft)

    def test_approve_command_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ApproveReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="idemp-app-01",
                unknown_field="slop",  # type: ignore[call-arg]
            )

    def test_approve_command_rejects_empty_keys_or_bad_run(self) -> None:
        with pytest.raises(ValidationError):
            ApproveReportCommand(
                assessment_id="",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="key",
            )
        with pytest.raises(ValidationError):
            ApproveReportCommand(
                assessment_id="asmt-001",
                run_number=0,
                reviewer_id="rev-001",
                idempotency_key="key",
            )
        with pytest.raises(ValidationError):
            ApproveReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="",
                idempotency_key="key",
            )
        with pytest.raises(ValidationError):
            ApproveReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="",
            )

    def test_approve_command_rejects_action_type_override(self) -> None:
        """ApproveReportCommand must not allow action_type override."""
        with pytest.raises(ValidationError):
            ApproveReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="idemp-app-01",
                action_type=ReviewActionType.REJECT_REPORT,  # type: ignore[arg-type]
            )


class TestRejectReportCommand:
    """Test suite for REJECT_REPORT command validation contracts."""

    def test_reject_report_with_valid_rationale_succeeds(self) -> None:
        cmd = RejectReportCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="idemp-rej-01",
            rationale="Evidence conflicts unresolvable; vehicle cannot be assessed",
        )
        assert cmd.assessment_id == "asmt-001"
        assert cmd.run_number == 1
        assert cmd.reviewer_id == "rev-001"
        assert cmd.idempotency_key == "idemp-rej-01"
        assert cmd.rationale == "Evidence conflicts unresolvable; vehicle cannot be assessed"

    def test_reject_report_without_rationale_fails(self) -> None:
        with pytest.raises(ValidationError):
            RejectReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="idemp-rej-01",
            )  # type: ignore[call-arg]

    def test_reject_report_with_empty_or_whitespace_rationale_fails(self) -> None:
        with pytest.raises(ValidationError, match="rationale"):
            RejectReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="idemp-rej-01",
                rationale="   ",
            )

    def test_reject_report_rationale_exceeding_bound_fails(self) -> None:
        with pytest.raises(ValidationError):
            RejectReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="idemp-rej-01",
                rationale="r" * 1001,
            )

    def test_reject_command_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            RejectReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="idemp-rej-01",
                rationale="Valid rationale",
                extra_data=123,  # type: ignore[call-arg]
            )

    def test_reject_command_rejects_action_type_override(self) -> None:
        """RejectReportCommand must not allow action_type override."""
        with pytest.raises(ValidationError):
            RejectReportCommand(
                assessment_id="asmt-001",
                run_number=1,
                reviewer_id="rev-001",
                idempotency_key="idemp-rej-01",
                rationale="Valid rationale",
                action_type=ReviewActionType.APPROVE_REPORT,  # type: ignore[arg-type]
            )


class TestReviewActionAndDisposition:
    """Test suite for ReviewAction domain model and disposition derivation."""

    def test_disposition_derivation(self) -> None:
        assert derive_report_disposition(None) == ReportDisposition.PENDING_REVIEW
        assert (
            derive_report_disposition(ReviewActionType.APPROVE_REPORT) == ReportDisposition.RELEASED
        )
        assert (
            derive_report_disposition(ReviewActionType.REJECT_REPORT) == ReportDisposition.REJECTED
        )
        assert (
            derive_report_disposition(ReviewActionType.REQUEST_REINVESTIGATION)
            == ReportDisposition.REINVESTIGATION_REQUESTED
        )

    def test_assessment_state_derivation(self) -> None:
        assert (
            derive_assessment_state(ReviewActionType.APPROVE_REPORT)
            == AssessmentLifecycleState.RELEASED
        )
        assert (
            derive_assessment_state(ReviewActionType.REJECT_REPORT)
            == AssessmentLifecycleState.REJECTED
        )
        assert (
            derive_assessment_state(ReviewActionType.REQUEST_REINVESTIGATION)
            == AssessmentLifecycleState.IN_PROGRESS
        )

    def test_review_action_immutability_and_hash(self) -> None:
        action = ReviewAction(
            id="action-001",
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            action_type=ReviewActionType.APPROVE_REPORT,
            idempotency_key="idemp-01",
            notes="Looks good",
        )
        assert action.disposition == ReportDisposition.RELEASED
        assert action.action_hash != ""
        assert len(action.action_hash) == 64

        # Frozen model rejects mutation
        with pytest.raises(ValidationError):
            action.notes = "Mutated"

    def test_released_report_contract(self) -> None:
        draft = _build_test_draft(incomplete=False)
        action = ReviewAction(
            id="action-001",
            assessment_id=draft.assessment_id,
            run_number=draft.run_number,
            reviewer_id="rev-001",
            action_type=ReviewActionType.APPROVE_REPORT,
            idempotency_key="idemp-01",
        )
        released = ReleasedReport(
            report_draft=draft,
            review_action=action,
            released_at=datetime.now(UTC),
        )
        assert released.disposition == ReportDisposition.RELEASED
        assert released.outcome == AssessmentOutcome.SCORED
        assert released.score == 35
        assert released.band == RiskBand.LOW
        assert released.vin == "1HGCR2F85HA000000"

    def test_released_report_preserves_incomplete_outcome(self) -> None:
        incomplete_draft = _build_test_draft(incomplete=True)
        action = ReviewAction(
            id="action-002",
            assessment_id=incomplete_draft.assessment_id,
            run_number=incomplete_draft.run_number,
            reviewer_id="rev-001",
            action_type=ReviewActionType.APPROVE_REPORT,
            idempotency_key="idemp-02",
            acknowledge_missing_evidence=True,
            rationale="Acknowledge missing odometer",
        )
        released = ReleasedReport(
            report_draft=incomplete_draft,
            review_action=action,
            released_at=datetime.now(UTC),
        )
        assert released.disposition == ReportDisposition.RELEASED
        assert released.outcome == AssessmentOutcome.INCOMPLETE
        assert released.score is None
        assert released.band is None
        assert released.is_incomplete is True
        assert len(released.missing_evidence_notices) == 1

    def test_released_report_rejects_non_approval_action(self) -> None:
        draft = _build_test_draft(incomplete=False)
        reject_action = ReviewAction(
            id="action-003",
            assessment_id=draft.assessment_id,
            run_number=draft.run_number,
            reviewer_id="rev-001",
            action_type=ReviewActionType.REJECT_REPORT,
            idempotency_key="idemp-03",
            rationale="Rejected",
        )
        with pytest.raises(ValueError, match="APPROVE_REPORT"):
            ReleasedReport(
                report_draft=draft,
                review_action=reject_action,
                released_at=datetime.now(UTC),
            )


class TestReviewPayloadHash:
    """Test suite for deterministic payload hashing for review commands."""

    def test_identical_commands_yield_same_hash(self) -> None:
        cmd1 = ApproveReportCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="idemp-01",
            notes="Note A",
        )
        cmd2 = ApproveReportCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="idemp-01",
            notes="Note A",
        )
        assert compute_review_payload_hash(cmd1) == compute_review_payload_hash(cmd2)

    def test_different_commands_yield_different_hashes(self) -> None:
        cmd1 = ApproveReportCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="idemp-01",
            notes="Note A",
        )
        cmd2 = ApproveReportCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="idemp-01",
            notes="Note B",
        )
        assert compute_review_payload_hash(cmd1) != compute_review_payload_hash(cmd2)

        rej_cmd = RejectReportCommand(
            assessment_id="asmt-001",
            run_number=1,
            reviewer_id="rev-001",
            idempotency_key="idemp-01",
            rationale="Reason A",
        )
        assert compute_review_payload_hash(cmd1) != compute_review_payload_hash(rej_cmd)
