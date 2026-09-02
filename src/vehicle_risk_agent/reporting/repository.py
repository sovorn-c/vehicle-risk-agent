"""Transactional repository for Report Draft persistence and Assessment lifecycle transitions."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState
from vehicle_risk_agent.persistence.models import AssessmentRecord, ReportDraftRecord
from vehicle_risk_agent.reporting.models import ReportDraft


class ReportDraftRepository:
    """Provides transactional, idempotent, and immutable persistence for Report Drafts."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_draft(self, draft: ReportDraft) -> ReportDraft:
        """Persist one immutable ReportDraft, enforcing run uniqueness and immutability."""
        draft_json = draft.model_dump_json()
        values = {
            "id": draft.id,
            "assessment_id": draft.assessment_id,
            "run_number": draft.run_number,
            "vehicle_id": draft.vin,
            "policy_id": draft.policy_id,
            "policy_version": draft.policy_version,
            "outcome": draft.outcome.value,
            "score": draft.score,
            "band": draft.band.value if draft.band else None,
            "draft_hash": draft.draft_hash,
            "draft_data_json": draft_json,
            "created_at": draft.created_at,
        }

        stmt = (
            insert(ReportDraftRecord)
            .values(values)
            .on_conflict_do_nothing(index_elements=["assessment_id", "run_number"])
        )
        await self._session.execute(stmt)

        # Verify idempotency and reject conflicting mutation
        query = select(ReportDraftRecord).where(
            ReportDraftRecord.assessment_id == draft.assessment_id,
            ReportDraftRecord.run_number == draft.run_number,
        )
        existing = (await self._session.execute(query)).scalar_one_or_none()
        if existing is None:
            await self._session.rollback()
            raise ValueError("Failed to retrieve persisted report draft")

        stored_domain = ReportDraft.model_validate_json(existing.draft_data_json)
        if stored_domain != draft:
            await self._session.rollback()
            raise ValueError(
                f"ReportDraft {draft.assessment_id}:{draft.run_number} "
                "is immutable and cannot be overwritten"
            )

        await self._session.commit()
        return stored_domain

    async def get_draft(self, assessment_id: str, run_number: int) -> ReportDraft | None:
        """Retrieve a persisted ReportDraft by assessment ID and run number."""
        stmt = select(ReportDraftRecord).where(
            ReportDraftRecord.assessment_id == str(assessment_id),
            ReportDraftRecord.run_number == run_number,
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return ReportDraft.model_validate_json(record.draft_data_json)

    async def save_draft_and_transition_assessment(
        self,
        draft: ReportDraft,
        state: AssessmentLifecycleState = AssessmentLifecycleState.AWAITING_REVIEW,
    ) -> ReportDraft:
        """Atomically persist report draft and transition Assessment lifecycle state."""
        # 1. Insert/Verify draft
        draft_json = draft.model_dump_json()
        values = {
            "id": draft.id,
            "assessment_id": draft.assessment_id,
            "run_number": draft.run_number,
            "vehicle_id": draft.vin,
            "policy_id": draft.policy_id,
            "policy_version": draft.policy_version,
            "outcome": draft.outcome.value,
            "score": draft.score,
            "band": draft.band.value if draft.band else None,
            "draft_hash": draft.draft_hash,
            "draft_data_json": draft_json,
            "created_at": draft.created_at,
        }

        stmt = (
            insert(ReportDraftRecord)
            .values(values)
            .on_conflict_do_nothing(index_elements=["assessment_id", "run_number"])
        )
        await self._session.execute(stmt)

        query = select(ReportDraftRecord).where(
            ReportDraftRecord.assessment_id == draft.assessment_id,
            ReportDraftRecord.run_number == draft.run_number,
        )
        existing = (await self._session.execute(query)).scalar_one_or_none()
        if existing is None:
            await self._session.rollback()
            raise ValueError("Failed to retrieve persisted report draft")

        stored_domain = ReportDraft.model_validate_json(existing.draft_data_json)
        if stored_domain != draft:
            await self._session.rollback()
            raise ValueError(
                f"ReportDraft {draft.assessment_id}:{draft.run_number} "
                "is immutable and cannot be overwritten"
            )

        # 2. Update Assessment record lifecycle_state
        asmt_stmt = select(AssessmentRecord).where(AssessmentRecord.id == draft.assessment_id)
        asmt_result = await self._session.execute(asmt_stmt)
        asmt_record = asmt_result.scalar_one_or_none()
        if asmt_record is not None:
            asmt_record.lifecycle_state = state.value
            asmt_record.updated_at = datetime.now(UTC)

        await self._session.commit()
        return stored_domain
