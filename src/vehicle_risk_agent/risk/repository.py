"""Transactional repositories for Risk Policy and Risk Result records."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import desc, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.persistence.models import RiskPolicyRecord, RiskResultRecord
from vehicle_risk_agent.risk.models import (
    RiskBand,
    RiskBandDefinition,
    RiskFactor,
    RiskPolicy,
    RiskPolicyLifecycleState,
    RiskResult,
)


def _record_to_policy(record: RiskPolicyRecord) -> RiskPolicy:
    """Convert persistent RiskPolicyRecord to immutable domain RiskPolicy."""
    raw_weights = json.loads(record.factor_weights_json)
    factor_weights = {RiskFactor(k): int(v) for k, v in raw_weights.items()}

    raw_bands = json.loads(record.risk_bands_json)
    risk_bands = tuple(
        RiskBandDefinition(
            band=RiskBand(b["band"]),
            min_score=int(b["min_score"]),
            max_score=int(b["max_score"]),
            description=b.get("description", ""),
        )
        for b in raw_bands
    )

    return RiskPolicy(
        id=record.id,
        version=record.version,
        name=record.name,
        description=record.description,
        lifecycle_state=RiskPolicyLifecycleState(record.lifecycle_state),
        factor_weights=factor_weights,
        score_cap=record.score_cap,
        risk_bands=risk_bands,
        mandatory_review_rules=tuple(json.loads(record.mandatory_review_rules_json)),
        required_evidence_fields=tuple(json.loads(record.required_evidence_fields_json)),
        created_at=record.created_at,
        updated_at=record.updated_at,
        activated_at=record.activated_at,
        activated_by=record.activated_by,
        retired_at=record.retired_at,
        policy_hash=record.policy_hash,
    )


class RiskPolicyRepository:
    """Provides transactional persistence for Risk Policy definitions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_policy(self, policy: RiskPolicy) -> RiskPolicy:
        """Persist a new RiskPolicy record."""
        record = RiskPolicyRecord(
            id=policy.id,
            version=policy.version,
            name=policy.name,
            description=policy.description,
            lifecycle_state=policy.lifecycle_state.value,
            factor_weights_json=json.dumps(
                {k.value: v for k, v in policy.factor_weights.items()}, sort_keys=True
            ),
            score_cap=policy.score_cap,
            risk_bands_json=json.dumps(
                [
                    {
                        "band": b.band.value,
                        "min_score": b.min_score,
                        "max_score": b.max_score,
                        "description": b.description,
                    }
                    for b in policy.risk_bands
                ],
                sort_keys=True,
            ),
            mandatory_review_rules_json=json.dumps(
                list(policy.mandatory_review_rules), sort_keys=True
            ),
            required_evidence_fields_json=json.dumps(
                list(policy.required_evidence_fields), sort_keys=True
            ),
            policy_hash=policy.policy_hash,
            activated_by=policy.activated_by,
            created_at=policy.created_at,
            updated_at=policy.updated_at,
            activated_at=policy.activated_at,
            retired_at=policy.retired_at,
        )
        self._session.add(record)
        await self._session.commit()
        return _record_to_policy(record)

    async def get_policy(self, policy_id: str) -> RiskPolicy | None:
        """Retrieve a RiskPolicy by its identifier."""
        stmt = select(RiskPolicyRecord).where(RiskPolicyRecord.id == policy_id)
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return _record_to_policy(record)

    async def get_active_policy(self) -> RiskPolicy | None:
        """Retrieve the single active RiskPolicy."""
        stmt = select(RiskPolicyRecord).where(
            RiskPolicyRecord.lifecycle_state == RiskPolicyLifecycleState.ACTIVE.value
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return _record_to_policy(record)

    async def list_policies(self) -> list[RiskPolicy]:
        """List all registered Risk Policies ordered by creation timestamp."""
        stmt = select(RiskPolicyRecord).order_by(desc(RiskPolicyRecord.created_at))
        result = await self._session.execute(stmt)
        records = result.scalars().all()
        return [_record_to_policy(r) for r in records]

    async def update_policy(self, policy: RiskPolicy) -> RiskPolicy:
        """Update lifecycle metadata for a policy."""
        stmt = select(RiskPolicyRecord).where(RiskPolicyRecord.id == policy.id)
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            raise ValueError(f"Risk policy {policy.id} not found")

        record.lifecycle_state = policy.lifecycle_state.value
        record.activated_at = policy.activated_at
        record.activated_by = policy.activated_by
        record.retired_at = policy.retired_at
        record.updated_at = datetime.now(UTC)

        await self._session.commit()
        return _record_to_policy(record)


class RiskResultRepository:
    """Provides transactional, idempotent, and immutable persistence for calculated Risk Results."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_result(self, result: RiskResult) -> None:
        """Persist one immutable RiskResult, enforcing run uniqueness."""
        result_json = result.model_dump_json()
        values = {
            "id": result.id,
            "assessment_id": result.assessment_id,
            "run_number": result.run_number,
            "policy_id": result.policy_id,
            "policy_version": result.policy_version,
            "score": result.score,
            "band": result.band.value if result.band else None,
            "raw_score": result.raw_score,
            "is_incomplete": result.is_incomplete,
            "calculation_hash": result.calculation_hash,
            "result_data_json": result_json,
            "calculated_at": result.calculated_at,
        }

        stmt = (
            insert(RiskResultRecord)
            .values(values)
            .on_conflict_do_nothing(index_elements=["assessment_id", "run_number"])
        )
        await self._session.execute(stmt)

        # Verify idempotency and reject conflicting mutation
        query = select(RiskResultRecord).where(
            RiskResultRecord.assessment_id == result.assessment_id,
            RiskResultRecord.run_number == result.run_number,
        )
        existing = (await self._session.execute(query)).scalar_one_or_none()
        if existing is None:
            await self._session.rollback()
            raise ValueError("Failed to retrieve persisted risk result")

        stored_domain = RiskResult(**json.loads(existing.result_data_json))
        if stored_domain != result:
            await self._session.rollback()
            raise ValueError(
                f"RiskResult {result.assessment_id}:{result.run_number} "
                "is immutable and cannot be overwritten"
            )

        await self._session.commit()

    async def get_result(self, assessment_id: str, run_number: int) -> RiskResult | None:
        """Retrieve a persisted RiskResult by assessment ID and run number."""
        stmt = select(RiskResultRecord).where(
            RiskResultRecord.assessment_id == assessment_id,
            RiskResultRecord.run_number == run_number,
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return RiskResult(**json.loads(record.result_data_json))
