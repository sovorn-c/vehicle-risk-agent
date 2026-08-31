"""Transactional repository for Assessment aggregates."""

import hashlib
import json
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.domain.assessment import (
    Assessment,
    AssessmentLifecycleState,
    AssessmentRun,
    AssessmentRunPhase,
)
from vehicle_risk_agent.domain.errors import IdempotencyConflictError
from vehicle_risk_agent.persistence.models import (
    AssessmentRecord,
    AssessmentRunRecord,
    IdempotencyRecord,
)


def _compute_payload_hash(request: AssessmentCreateRequest) -> str:
    """Compute sha256 hash of normalized request content."""
    data = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _record_to_domain(record: AssessmentRecord) -> Assessment:
    """Map SQLAlchemy record to immutable domain model."""
    context_data = json.loads(record.context_json)
    context = AssessmentContext(
        sale_type=SaleType(context_data["sale_type"]),
        intended_use=context_data.get("intended_use"),
        questions=context_data.get("questions", []),
    )
    runs = [
        AssessmentRun(
            id=r.id,
            assessment_id=r.assessment_id,
            run_number=r.run_number,
            phase=AssessmentRunPhase(r.phase),
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in record.runs
    ]
    return Assessment(
        id=record.id,
        requester_id=record.requester_id,
        vin=record.vin,
        context=context,
        lifecycle_state=AssessmentLifecycleState(record.lifecycle_state),
        current_run_number=record.current_run_number,
        runs=runs,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


class AssessmentRepository:
    """Provides transactional persistence operations for Assessments."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_assessment(self, assessment_id: str) -> Assessment | None:
        """Retrieve an Assessment by ID with its runs."""
        stmt = (
            select(AssessmentRecord)
            .where(AssessmentRecord.id == assessment_id)
            .options(selectinload(AssessmentRecord.runs))
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return _record_to_domain(record)

    async def create_assessment(
        self,
        requester_id: str,
        idempotency_key: str,
        request: AssessmentCreateRequest,
    ) -> Assessment:
        """Atomically create an Assessment and its first Assessment Run.

        Enforces idempotency control.
        """
        payload_hash = _compute_payload_hash(request)
        command_type = "CREATE_ASSESSMENT"

        # Check existing idempotency key for this principal and command type
        stmt = select(IdempotencyRecord).where(
            IdempotencyRecord.principal_id == requester_id,
            IdempotencyRecord.command_type == command_type,
            IdempotencyRecord.idempotency_key == idempotency_key,
        )
        result = await self._session.execute(stmt)
        existing_idemp = result.scalar_one_or_none()

        if existing_idemp is not None:
            if existing_idemp.payload_hash != payload_hash:
                raise IdempotencyConflictError(
                    key=idempotency_key,
                    message="Idempotency key reused with different request payload",
                )
            existing_assessment = await self.get_assessment(existing_idemp.target_resource_id)
            if existing_assessment is not None:
                return existing_assessment

        assessment_id = str(uuid4())
        context_json = json.dumps(request.context.model_dump(mode="json"), sort_keys=True)

        assessment_record = AssessmentRecord(
            id=assessment_id,
            requester_id=requester_id,
            vin=request.vin,
            context_json=context_json,
            lifecycle_state="IN_PROGRESS",
            current_run_number=1,
        )
        self._session.add(assessment_record)

        run_record = AssessmentRunRecord(
            id=str(uuid4()),
            assessment_id=assessment_id,
            run_number=1,
            phase="PENDING",
        )
        self._session.add(run_record)

        idemp_record = IdempotencyRecord(
            principal_id=requester_id,
            command_type=command_type,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            target_resource_id=assessment_id,
        )
        self._session.add(idemp_record)

        await self._session.commit()
        retrieved = await self.get_assessment(assessment_id)
        assert retrieved is not None
        return retrieved
