"""Transactional service for human review decisions, row locking, and report release."""

# story: e05s01
# story: e05s02
# story: e05s03

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.domain.assessment import (
    AssessmentLifecycleState,
    AssessmentRunPhase,
)
from vehicle_risk_agent.domain.errors import IdempotencyConflictError
from vehicle_risk_agent.observability.telemetry import trace_boundary
from vehicle_risk_agent.persistence.models import (
    AssessmentRecord,
    AssessmentRunRecord,
    IdempotencyRecord,
    ReportDraftRecord,
    ReviewActionRecord,
)
from vehicle_risk_agent.policy.corpus_lifecycle import CorpusLifecycleManager
from vehicle_risk_agent.reporting.models import ReportDraft
from vehicle_risk_agent.review.errors import (
    AssessmentNotFoundError,
    AssessmentNotReviewableError,
    DraftNotFoundError,
    ReinvestigationLimitReachedError,
    ReviewActionConflictError,
)
from vehicle_risk_agent.review.models import (
    ApproveReportCommand,
    AssessmentHistory,
    PinnedVersions,
    RejectReportCommand,
    ReleasedReport,
    ReportDisposition,
    RequestReinvestigationCommand,
    ReviewAction,
    ReviewActionType,
    RunHistoryItem,
    compute_review_payload_hash,
    derive_assessment_state,
    derive_report_disposition,
)
from vehicle_risk_agent.risk.repository import RiskPolicyRepository


@dataclass(frozen=True)
class ReviewDecisionResult:
    """Outcome of recording a Review Action."""

    action: ReviewAction
    assessment_id: str
    assessment_state: AssessmentLifecycleState
    disposition: ReportDisposition
    released_report: ReleasedReport | None = None
    next_run_number: int | None = None
    pinned_versions: PinnedVersions | None = None


def _record_to_action(record: ReviewActionRecord) -> ReviewAction:
    """Map a SQLAlchemy ReviewActionRecord to domain ReviewAction."""
    questions: tuple[str, ...] = ()
    evidence_targets: tuple[str, ...] = ()
    pinned_versions: PinnedVersions | None = None
    notes_val: str | None = record.notes

    if record.action_type == ReviewActionType.REQUEST_REINVESTIGATION.value and record.notes:
        try:
            data = json.loads(record.notes)
            if isinstance(data, dict):
                questions = tuple(data.get("questions", ()))
                evidence_targets = tuple(data.get("evidence_targets", ()))
                pv_data = data.get("pinned_versions")
                if pv_data and isinstance(pv_data, dict):
                    pinned_versions = PinnedVersions.model_validate(pv_data)
                notes_val = None
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return ReviewAction(
        id=record.id,
        assessment_id=record.assessment_id,
        run_number=record.run_number,
        reviewer_id=record.reviewer_id,
        action_type=ReviewActionType(record.action_type),
        disposition=ReportDisposition(record.disposition),
        idempotency_key=record.idempotency_key,
        rationale=record.rationale,
        notes=notes_val,
        acknowledge_missing_evidence=record.acknowledge_missing_evidence,
        questions=questions,
        evidence_targets=evidence_targets,
        pinned_versions=pinned_versions,
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

    async def get_assessment_history(self, assessment_id: str) -> AssessmentHistory | None:
        """Project ordered immutable runs, drafts, evidence summaries, and Review Actions."""
        asmt_stmt = select(AssessmentRecord).where(AssessmentRecord.id == assessment_id)
        asmt_res = await self._session.execute(asmt_stmt)
        asmt_rec = asmt_res.scalar_one_or_none()
        if asmt_rec is None:
            return None

        # Fetch runs ordered by run_number
        runs_stmt = (
            select(AssessmentRunRecord)
            .where(AssessmentRunRecord.assessment_id == assessment_id)
            .order_by(AssessmentRunRecord.run_number)
        )
        run_records = (await self._session.execute(runs_stmt)).scalars().all()

        # Fetch drafts ordered by run_number
        drafts_stmt = (
            select(ReportDraftRecord)
            .where(ReportDraftRecord.assessment_id == assessment_id)
            .order_by(ReportDraftRecord.run_number)
        )
        draft_records = (await self._session.execute(drafts_stmt)).scalars().all()
        drafts_by_run: dict[int, ReportDraft] = {
            d.run_number: ReportDraft.model_validate_json(d.draft_data_json) for d in draft_records
        }

        # Fetch review actions ordered by run_number
        actions = await self.get_review_history(assessment_id)
        actions_by_run: dict[int, ReviewAction] = {a.run_number: a for a in actions}

        # Fetch released report if approved
        released_report = await self.get_released_report(assessment_id)

        # Build run history items
        run_items: list[RunHistoryItem] = []
        for run_rec in run_records:
            draft = drafts_by_run.get(run_rec.run_number)
            evidence_summary = draft.sections.evidence_summary if draft else None
            action = actions_by_run.get(run_rec.run_number)

            pinned: PinnedVersions | None = None
            if run_rec.run_number > 1:
                prior_action = actions_by_run.get(run_rec.run_number - 1)
                if prior_action and prior_action.pinned_versions:
                    pinned = prior_action.pinned_versions
            elif draft is not None:
                pinned = PinnedVersions(
                    risk_policy_id=draft.policy_id,
                    risk_policy_version=draft.policy_version,
                )

            run_items.append(
                RunHistoryItem(
                    run_number=run_rec.run_number,
                    phase=run_rec.phase,
                    draft=draft,
                    evidence_summary=evidence_summary,
                    pinned_versions=pinned,
                    review_action=action,
                    created_at=run_rec.created_at,
                    updated_at=run_rec.updated_at,
                )
            )

        # Derive disposition
        if asmt_rec.lifecycle_state == AssessmentLifecycleState.RELEASED.value:
            disposition = ReportDisposition.RELEASED
        elif asmt_rec.lifecycle_state == AssessmentLifecycleState.REJECTED.value:
            disposition = ReportDisposition.REJECTED
        elif asmt_rec.lifecycle_state == AssessmentLifecycleState.IN_PROGRESS.value:
            if any(a.action_type == ReviewActionType.REQUEST_REINVESTIGATION for a in actions):
                disposition = ReportDisposition.REINVESTIGATION_REQUESTED
            else:
                disposition = ReportDisposition.PENDING_REVIEW
        else:
            disposition = ReportDisposition.PENDING_REVIEW

        return AssessmentHistory(
            assessment_id=asmt_rec.id,
            vin=asmt_rec.vin,
            requester_id=asmt_rec.requester_id,
            lifecycle_state=AssessmentLifecycleState(asmt_rec.lifecycle_state),
            current_run_number=asmt_rec.current_run_number,
            disposition=disposition,
            runs=tuple(run_items),
            released_report=released_report,
            created_at=asmt_rec.created_at,
            updated_at=asmt_rec.updated_at,
        )

    async def record_review_action(
        self,
        command: ApproveReportCommand | RejectReportCommand | RequestReinvestigationCommand,
    ) -> ReviewDecisionResult:
        """Record a review decision and emit a safe review boundary trace."""
        with trace_boundary(
            "review.record_action",
            boundary="review",
            assessment_id=command.assessment_id,
            run_number=command.run_number,
            action_type=command.action_type.value,
        ):
            return await self._record_review_action(command)

    async def _record_review_action(
        self,
        command: ApproveReportCommand | RejectReportCommand | RequestReinvestigationCommand,
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
        - Sequential run allocation and active version pinning for reinvestigation.
        - Three-run maximum limit enforcement.
        - Assessment lifecycle transition to RELEASED, REJECTED, or IN_PROGRESS.
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

                next_run: int | None = None
                if action.action_type == ReviewActionType.REQUEST_REINVESTIGATION and asmt_rec:
                    next_run = asmt_rec.current_run_number

                return ReviewDecisionResult(
                    action=action,
                    assessment_id=command.assessment_id,
                    assessment_state=current_state,
                    disposition=action.disposition,
                    released_report=released_report,
                    next_run_number=next_run,
                    pinned_versions=action.pinned_versions,
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

        # Re-check idempotency under row lock in case a concurrent request just committed
        idemp_res = await self._session.execute(idemp_stmt)
        existing_idemp = idemp_res.scalar_one_or_none()
        if existing_idemp is not None:
            if existing_idemp.payload_hash != payload_hash:
                raise IdempotencyConflictError(
                    key=command.idempotency_key,
                    message="Idempotency key reused with different request payload",
                )
            action = await self.get_review_action(command.assessment_id, command.run_number)
            if action is not None:
                released_report = None
                if action.action_type == ReviewActionType.APPROVE_REPORT:
                    released_report = await self.get_released_report(command.assessment_id)
                current_state = AssessmentLifecycleState(asmt_record.lifecycle_state)
                next_run = None
                if action.action_type == ReviewActionType.REQUEST_REINVESTIGATION:
                    next_run = asmt_record.current_run_number
                return ReviewDecisionResult(
                    action=action,
                    assessment_id=command.assessment_id,
                    assessment_state=current_state,
                    disposition=action.disposition,
                    released_report=released_report,
                    next_run_number=next_run,
                    pinned_versions=action.pinned_versions,
                )

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

        # 6. Validate command against draft and enforce action_type consistency
        next_run_number: int | None = None
        pinned_versions: PinnedVersions | None = None

        if isinstance(command, ApproveReportCommand):
            if command.action_type != ReviewActionType.APPROVE_REPORT:
                raise ValueError(
                    "ApproveReportCommand action_type must match ReviewActionType.APPROVE_REPORT"
                )
            command.validate_for_draft(draft)
        elif isinstance(command, RejectReportCommand):
            if command.action_type != ReviewActionType.REJECT_REPORT:
                raise ValueError(
                    "RejectReportCommand action_type must match ReviewActionType.REJECT_REPORT"
                )
        elif isinstance(command, RequestReinvestigationCommand):
            if command.action_type != ReviewActionType.REQUEST_REINVESTIGATION:
                raise ValueError(
                    "RequestReinvestigationCommand action_type must match "
                    "ReviewActionType.REQUEST_REINVESTIGATION"
                )
            if asmt_record.current_run_number >= 3:
                raise ReinvestigationLimitReachedError(
                    f"Assessment {command.assessment_id} has reached the maximum 3-run limit; "
                    f"cannot allocate run {asmt_record.current_run_number + 1}"
                )

            next_run_number = asmt_record.current_run_number + 1
            corpus_mgr = CorpusLifecycleManager(self._session)
            active_corpus = await corpus_mgr.get_active_corpus()
            risk_repo = RiskPolicyRepository(self._session)
            active_policy = await risk_repo.get_active_policy()

            pinned_versions = PinnedVersions(
                corpus_id=active_corpus.id if active_corpus else None,
                corpus_manifest_hash=active_corpus.manifest_hash if active_corpus else None,
                risk_policy_id=active_policy.id if active_policy else None,
                risk_policy_version=active_policy.version if active_policy else None,
                risk_policy_hash=active_policy.policy_hash if active_policy else None,
            )
        else:
            raise TypeError(f"Unsupported review command type: {type(command)}")

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
            questions=getattr(command, "questions", ()),
            evidence_targets=getattr(command, "evidence_targets", ()),
            pinned_versions=pinned_versions,
            created_at=datetime.now(UTC),
        )

        notes_payload = action.notes
        if action.action_type == ReviewActionType.REQUEST_REINVESTIGATION:
            reinvest_dict: dict[str, Any] = {
                "questions": list(action.questions),
                "evidence_targets": list(action.evidence_targets),
            }
            if pinned_versions is not None:
                reinvest_dict["pinned_versions"] = pinned_versions.model_dump(mode="json")
            notes_payload = json.dumps(reinvest_dict, sort_keys=True)

        action_rec = ReviewActionRecord(
            id=action.id,
            assessment_id=action.assessment_id,
            run_number=action.run_number,
            reviewer_id=action.reviewer_id,
            action_type=action.action_type.value,
            disposition=action.disposition.value,
            idempotency_key=action.idempotency_key,
            rationale=action.rationale,
            notes=notes_payload,
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

        # 10. Allocate next run if reinvestigation and transition Assessment state
        if next_run_number is not None:
            new_run_rec = AssessmentRunRecord(
                id=str(uuid4()),
                assessment_id=command.assessment_id,
                run_number=next_run_number,
                phase=AssessmentRunPhase.PENDING.value,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            self._session.add(new_run_rec)
            asmt_record.current_run_number = next_run_number

        asmt_record.lifecycle_state = new_asmt_state.value
        asmt_record.updated_at = datetime.now(UTC)

        await self._session.commit()

        # 11. Build ReleasedReport projection if approved
        released_report = None
        if (
            isinstance(command, ApproveReportCommand)
            and command.action_type == ReviewActionType.APPROVE_REPORT
        ):
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
            next_run_number=next_run_number,
            pinned_versions=pinned_versions,
        )
