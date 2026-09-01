"""Event store for appending and retrieving sanitized workflow progress events."""

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.domain.events import WorkflowProgressEvent
from vehicle_risk_agent.events.broadcaster import ProgressEventBroadcaster
from vehicle_risk_agent.persistence.models import WorkflowEventRecord


class EventStore:
    """Provides transactional appending and retrieval for progress events."""

    def __init__(
        self,
        session: AsyncSession,
        broadcaster: ProgressEventBroadcaster | None = None,
    ) -> None:
        self._session = session
        self._broadcaster = broadcaster

    async def append_event(self, event: WorkflowProgressEvent) -> None:
        """Persist a sanitized workflow progress event and broadcast to active subscribers."""
        record = WorkflowEventRecord(
            id=str(uuid4()),
            assessment_id=event.assessment_id,
            run_number=event.run_number,
            sequence=event.sequence,
            phase=event.phase.value,
            safe_message=event.safe_message,
            timestamp=event.timestamp,
        )
        self._session.add(record)
        await self._session.commit()
        if self._broadcaster is not None:
            self._broadcaster.publish(event)

    async def get_events(
        self,
        assessment_id: str,
        after_sequence: int = 0,
    ) -> list[WorkflowProgressEvent]:
        """Retrieve events for an assessment ordered by sequence.

        Only includes events with sequence strictly greater than after_sequence.
        """
        stmt = (
            select(WorkflowEventRecord)
            .where(
                WorkflowEventRecord.assessment_id == assessment_id,
                WorkflowEventRecord.sequence > after_sequence,
            )
            .order_by(WorkflowEventRecord.sequence.asc())
        )
        result = await self._session.execute(stmt)
        records = result.scalars().all()
        return [
            WorkflowProgressEvent(
                event_id=r.id,
                sequence=r.sequence,
                assessment_id=r.assessment_id,
                run_number=r.run_number,
                phase=AssessmentRunPhase(r.phase),
                safe_message=r.safe_message,
                timestamp=r.timestamp,
            )
            for r in records
        ]
