"""Durable reservation ledger for bounded supplementary investigation."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.investigation.budget import InvestigationLimits
from vehicle_risk_agent.persistence.models import InvestigationLedgerRecord


class InvestigationLedgerStatus(StrEnum):
    READY = "READY"
    COUNTING = "COUNTING"
    PROPOSAL_IN_FLIGHT = "PROPOSAL_IN_FLIGHT"
    ACTION_IN_FLIGHT = "ACTION_IN_FLIGHT"
    COMPLETED = "COMPLETED"
    INDETERMINATE = "INDETERMINATE"
    EXHAUSTED = "EXHAUSTED"


class InvestigationLedger:
    """Immutable application view of one persisted ledger row."""

    def __init__(self, record: InvestigationLedgerRecord) -> None:
        self.id = record.id
        self.assessment_id = record.assessment_id
        self.run_number = record.run_number
        self.vin = record.vin
        self.status = InvestigationLedgerStatus(record.status)
        self.proposal_rounds = record.proposal_rounds
        self.supplementary_attempts = record.supplementary_attempts
        self.supplementary_retries = record.supplementary_retries
        self.input_tokens = record.input_tokens
        self.output_tokens = record.output_tokens
        self.projected_cost = record.projected_cost
        self.actual_cost = record.actual_cost
        self.current_request_hash = record.current_request_hash
        self.result_summary = record.result_summary
        self.references = tuple(json.loads(record.references_json or "[]"))
        self.limits = InvestigationLimits.model_validate(json.loads(record.limits_json))
        self.pins = json.loads(record.pins_json)


class InvestigationLedgerRepository:
    """Own transaction boundaries around reservations and recovery."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def ensure_ledger(
        self,
        assessment_id: str,
        run_number: int,
        vin: str,
        pins: dict[str, str],
        limits: InvestigationLimits,
        now: datetime | None = None,
    ) -> InvestigationLedger:
        record = await self._get(assessment_id, run_number, lock=True)
        if record is None:
            current = now or datetime.now(UTC)
            record = InvestigationLedgerRecord(
                id=str(uuid4()),
                assessment_id=assessment_id,
                run_number=run_number,
                vin=vin,
                status=InvestigationLedgerStatus.READY.value,
                deadline_at=current + timedelta(seconds=limits.max_duration_seconds),
                limits_json=limits.model_dump_json(),
                pins_json=json.dumps(pins, sort_keys=True),
                references_json="[]",
                created_at=current,
                updated_at=current,
            )
            self.session.add(record)
            await self.session.commit()
        return InvestigationLedger(record)

    async def reserve_action(
        self,
        assessment_id: str,
        run_number: int,
        request_hash: str,
        projected_cost: float,
    ) -> InvestigationLedger:
        record = await self._get(assessment_id, run_number, lock=True)
        if record is None:
            raise ValueError("investigation ledger does not exist")
        if record.current_request_hash == request_hash and record.status in {
            InvestigationLedgerStatus.COMPLETED.value,
            InvestigationLedgerStatus.INDETERMINATE.value,
        }:
            return InvestigationLedger(record)
        limits = InvestigationLimits.model_validate(json.loads(record.limits_json))
        if record.status not in {InvestigationLedgerStatus.READY.value}:
            raise ValueError("investigation ledger is not reservable")
        if record.supplementary_attempts >= limits.max_supplementary_attempts:
            record.status = InvestigationLedgerStatus.EXHAUSTED.value
            await self.session.commit()
            return InvestigationLedger(record)
        if record.projected_cost + projected_cost > float(limits.max_cost_usd):
            record.status = InvestigationLedgerStatus.EXHAUSTED.value
            await self.session.commit()
            return InvestigationLedger(record)
        record.status = InvestigationLedgerStatus.ACTION_IN_FLIGHT.value
        record.current_request_hash = request_hash
        record.projected_cost += projected_cost
        record.supplementary_attempts += 1
        record.updated_at = datetime.now(UTC)
        await self.session.commit()
        return InvestigationLedger(record)

    async def complete_action(
        self,
        assessment_id: str,
        run_number: int,
        request_hash: str,
        actual_cost: float,
        result_summary: str,
        references: tuple[str, ...],
    ) -> InvestigationLedger:
        record = await self._get(assessment_id, run_number, lock=True)
        if record is None or record.current_request_hash != request_hash:
            raise ValueError("investigation reservation does not match")
        record.status = InvestigationLedgerStatus.COMPLETED.value
        record.actual_cost += actual_cost
        record.result_summary = result_summary[:500]
        record.references_json = json.dumps(references[:50])
        record.updated_at = datetime.now(UTC)
        await self.session.commit()
        return InvestigationLedger(record)

    async def recover_inflight(self, assessment_id: str, run_number: int) -> InvestigationLedger:
        record = await self._get(assessment_id, run_number, lock=True)
        if record is None:
            raise ValueError("investigation ledger does not exist")
        if record.status in {
            InvestigationLedgerStatus.ACTION_IN_FLIGHT.value,
            InvestigationLedgerStatus.PROPOSAL_IN_FLIGHT.value,
            InvestigationLedgerStatus.COUNTING.value,
        }:
            record.status = InvestigationLedgerStatus.INDETERMINATE.value
            record.updated_at = datetime.now(UTC)
            await self.session.commit()
        return InvestigationLedger(record)

    async def _get(
        self, assessment_id: str, run_number: int, lock: bool = False
    ) -> InvestigationLedgerRecord | None:
        statement = select(InvestigationLedgerRecord).where(
            InvestigationLedgerRecord.assessment_id == assessment_id,
            InvestigationLedgerRecord.run_number == run_number,
        )
        if lock:
            statement = statement.with_for_update()
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()
