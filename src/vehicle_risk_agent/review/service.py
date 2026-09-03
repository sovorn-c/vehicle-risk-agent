"""Transactional service for human review decisions, row locking, and report release."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState
from vehicle_risk_agent.domain.errors import IdempotencyConflictError
from vehicle_risk_agent.persistence.models import (
    AssessmentRecord,
    IdempotencyRecord,
    ReportDraftRecord,
    ReviewActionRecord,
)
from vehicle_risk_agent.reporting.models import ReportDraft
from vehicle_risk_agent.review.errors import (
    AssessmentNotFoundError,
    AssessmentNotReviewableError,
    DraftNotFoundError,
    ReviewActionConflictError,
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


@dataclass(frozen=True)
class ReviewDecisionResult:
    """Outcome of recording a Review Action."""

    action: ReviewAction
    assessment_id: str
    assessment_state: AssessmentLifecycleState
    disposition: ReportDisposition
    released_report: ReleasedReport | None = None


def _record_to_action(record: ReviewActionRecord) -> ReviewAction:
    """Map a SQLAlchemy ReviewActionRecord to domain ReviewAction."""
    return ReviewAction(
        id=record.id,
        assessment_id=record.assessment_id,
        run_number=record.run_number,
        reviewer_id=record.reviewer_id,
        action_type=ReviewActionType(record.action_type),
        disposition=ReportDisposition(record.disposition),
        idempotency_key=record.idempotency_key,
        rationale=record.rationale,
        notes=record.notes,
        acknowledge_missing_evidence=record.acknowledge_missing_evidence,
        created_at=record.created_at,
        action_hash=record.action_hash,
    )


class ReviewDecisionService:
    """Provides transactional review decisions under an Assessment row lock."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_review_action(self, assessment_id: str, run_number: int) -> ReviewAction | None:
        """Retrieve the immutable ReviewAction recorded for an assessment run."""
        stmt = select(ReviewActionRecord).where(
            ReviewActionRecord.assessment_id == assessment_id,
            ReviewActionRecord.run_number == run_number,
        )
        res = await self._session.execute(stmt)
        record = res.scalar_one_or_none()
        if record is None:
            return None
        return _record_to_action(record)

    async def get_review_history(self, assessment_id: str) -> list[ReviewAction]:
        """Retrieve all ReviewActions for an assessment ordered by run number."""
        stmt = (
            select(ReviewActionRecord)
            .where(ReviewActionRecord.assessment_id == assessment_id)
            .order_by(ReviewActionRecord.run_number)
        )
        res = await self._session.execute(stmt)
        records = res.scalars().all()
        return [_record_to_action(r) for r in records]

    async def get_released_report(self, assessment_id: str) -> ReleasedReport | None:
        """Retrieve the ReleasedReport for an assessment if an APPROVE_REPORT action exists."""
        stmt = (
            select(ReviewActionRecord)
            .where(
                ReviewActionRecord.assessment_id == assessment_id,
                ReviewActionRecord.action_type == ReviewActionType.APPROVE_REPORT.value,
            )
            .order_by(ReviewActionRecord.run_number.desc())
        )
        res = await self._session.execute(stmt)
        action_rec = res.scalars().first()
        if action_rec is None:
            return None

        action = _record_to_action(action_rec)

        draft_stmt = select(ReportDraftRecord).where(
            ReportDraftRecord.assessment_id == assessment_id,
            ReportDraftRecord.run_number == action.run_number,
        )
        draft_res = await self._session.execute(draft_stmt)
        draft_rec = draft_res.scalar_one_or_none()
        if draft_rec is None:
            return None

        draft = ReportDraft.model_validate_json(draft_rec.draft_data_json)
        return ReleasedReport(
            report_draft=draft,
            review_action=action,
            released_at=action.created_at,
        )

    async def record_review_action(
        self, command: ApproveReportCommand | RejectReportCommand
    ) -> ReviewDecisionResult:
        """Atomically record a ReviewAction under an Assessment row lock.

        Enforces:
        - Exact idempotency key replay vs conflict detection.
        - Assessment row lock via SELECT FOR UPDATE.
        - Assessment lifecycle state AWAITING_REVIEW.
        - Existence of target ReportDraft.
        - Exactly one ReviewAction per ReportDraft.
        - Command validation against the target draft.
        - Immutable draft preservation (draft data is never updated).
        - Assessment lifecycle transition to RELEASED or REJECTED.
        """
        payload_hash = compute_review_payload_hash(command)

        # 1. Check idempotency record first
        idemp_stmt = select(IdempotencyRecord).where(
            IdempotencyRecord.principal_id == command.reviewer_id,
            IdempotencyRecord.command_type == command.action_type.value,
            IdempotencyRecord.idempotency_key == command.idempotency_key,
        )
        idemp_res = await self._session.execute(idemp_stmt)
        existing_idemp = idemp_res.scalar_one_or_none()

        if existing_idemp is not None:
            if existing_idemp.payload_hash != payload_hash:
                raise IdempotencyConflictError(
                    key=command.idempotency_key,
                    message="Idempotency key reused with different request payload",
                )

            # Exact replay: return existing decision result
            action = await self.get_review_action(command.assessment_id, command.run_number)
            if action is not None:
                released_report: ReleasedReport | None = None
                if action.action_type == ReviewActionType.APPROVE_REPORT:
                    released_report = await self.get_released_report(command.assessment_id)

                asmt_stmt = select(AssessmentRecord).where(
                    AssessmentRecord.id == command.assessment_id
                )
                asmt_rec = (await self._session.execute(asmt_stmt)).scalar_one_or_none()
                current_state = (
                    AssessmentLifecycleState(asmt_rec.lifecycle_state)
                    if asmt_rec
                    else derive_assessment_state(action.action_type)
                )

                return ReviewDecisionResult(
                    action=action,
                    assessment_id=command.assessment_id,
                    assessment_state=current_state,
                    disposition=action.disposition,
                    released_report=released_report,
                )

        # 2. Lock Assessment row with SELECT FOR UPDATE
        asmt_stmt = (
            select(AssessmentRecord)
            .where(AssessmentRecord.id == command.assessment_id)
            .with_for_update()
        )
        asmt_res = await self._session.execute(asmt_stmt)
        asmt_record = asmt_res.scalar_one_or_none()
        if asmt_record is None:
            raise AssessmentNotFoundError(f"Assessment {command.assessment_id} not found")

        # 3. Check if ReviewAction already exists for this draft
        existing_action_stmt = select(ReviewActionRecord).where(
            ReviewActionRecord.assessment_id == command.assessment_id,
            ReviewActionRecord.run_number == command.run_number,
        )
        res = await self._session.execute(existing_action_stmt)
        existing_action_rec = res.scalar_one_or_none()
        if existing_action_rec is not None:
            raise ReviewActionConflictError(
                f"Report draft {command.assessment_id}:{command.run_number} "
                "has already been decided"
            )

        # 4. Check Assessment lifecycle state
        if asmt_record.lifecycle_state != AssessmentLifecycleState.AWAITING_REVIEW.value:
            raise AssessmentNotReviewableError(
                f"Assessment {command.assessment_id} is in state {asmt_record.lifecycle_state}, "
                "expected AWAITING_REVIEW"
            )

        # 5. Check ReportDraft exists
        draft_stmt = select(ReportDraftRecord).where(
            ReportDraftRecord.assessment_id == command.assessment_id,
            ReportDraftRecord.run_number == command.run_number,
        )
        draft_res = await self._session.execute(draft_stmt)
        draft_record = draft_res.scalar_one_or_none()
        if draft_record is None:
            raise DraftNotFoundError(
                f"Report draft not found for assessment {command.assessment_id} "
                f"run {command.run_number}"
            )

        draft = ReportDraft.model_validate_json(draft_record.draft_data_json)

        # 6. Validate command against draft
        if isinstance(command, ApproveReportCommand):
            command.validate_for_draft(draft)

        # 7. Derive disposition and new assessment lifecycle state
        disposition = derive_report_disposition(command.action_type)
        new_asmt_state = derive_assessment_state(command.action_type)

        # 8. Build and insert ReviewAction
        action = ReviewAction(
            id=str(uuid4()),
            assessment_id=command.assessment_id,
            run_number=command.run_number,
            reviewer_id=command.reviewer_id,
            action_type=command.action_type,
            disposition=disposition,
            idempotency_key=command.idempotency_key,
            rationale=getattr(command, "rationale", None),
            notes=getattr(command, "notes", None),
            acknowledge_missing_evidence=getattr(command, "acknowledge_missing_evidence", False),
            created_at=datetime.now(UTC),
        )

        action_rec = ReviewActionRecord(
            id=action.id,
            assessment_id=action.assessment_id,
            run_number=action.run_number,
            reviewer_id=action.reviewer_id,
            action_type=action.action_type.value,
            disposition=action.disposition.value,
            idempotency_key=action.idempotency_key,
            rationale=action.rationale,
            notes=action.notes,
            acknowledge_missing_evidence=action.acknowledge_missing_evidence,
            action_hash=action.action_hash,
            created_at=action.created_at,
        )
        self._session.add(action_rec)

        # 9. Insert IdempotencyRecord
        idemp_record = IdempotencyRecord(
            id=str(uuid4()),
            principal_id=command.reviewer_id,
            command_type=command.action_type.value,
            idempotency_key=command.idempotency_key,
            payload_hash=payload_hash,
            target_resource_id=action.id,
            created_at=datetime.now(UTC),
        )
        self._session.add(idemp_record)

        # 10. Transition Assessment lifecycle state (Draft content is NOT mutated)
        asmt_record.lifecycle_state = new_asmt_state.value
        asmt_record.updated_at = datetime.now(UTC)

        await self._session.commit()

        # 11. Build ReleasedReport projection if approved
        released_report = None
        if command.action_type == ReviewActionType.APPROVE_REPORT:
            released_report = ReleasedReport(
                report_draft=draft,
                review_action=action,
                released_at=action.created_at,
            )

        return ReviewDecisionResult(
            action=action,
            assessment_id=command.assessment_id,
            assessment_state=new_asmt_state,
            disposition=disposition,
            released_report=released_report,
        )
