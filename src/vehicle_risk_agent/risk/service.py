"""Service and lifecycle manager for Risk Policy registration, validation, and activation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.persistence.models import RiskPolicyRecord
from vehicle_risk_agent.risk.models import (
    DEFAULT_POLICY_V1_BANDS,
    DEFAULT_POLICY_V1_FACTOR_WEIGHTS,
    DEFAULT_REQUIRED_EVIDENCE_FIELDS,
    RiskBandDefinition,
    RiskFactor,
    RiskPolicy,
    RiskPolicyLifecycleState,
    compute_policy_hash,
)
from vehicle_risk_agent.risk.repository import _record_to_policy


class RiskPolicyLifecycleError(ValueError):
    """Raised on invalid lifecycle transitions or operations."""


class RiskPolicyService:
    """Orchestrates Risk Policy creation, validation gates, and atomic activation."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_policy(
        self,
        policy_id: str,
        name: str,
        description: str,
        version: str = "v1",
        factor_weights: dict[RiskFactor, int] | None = None,
        score_cap: int = 100,
        risk_bands: tuple[RiskBandDefinition, ...] | Sequence[RiskBandDefinition] | None = None,
        mandatory_review_rules: tuple[str, ...] | Sequence[str] | None = None,
        required_evidence_fields: tuple[str, ...] | Sequence[str] | None = None,
    ) -> RiskPolicy:
        """Create a new Risk Policy in DRAFT state."""
        weights = (
            dict(DEFAULT_POLICY_V1_FACTOR_WEIGHTS) if factor_weights is None else factor_weights
        )
        bands = tuple(DEFAULT_POLICY_V1_BANDS) if risk_bands is None else tuple(risk_bands)
        req_fields = (
            tuple(DEFAULT_REQUIRED_EVIDENCE_FIELDS)
            if required_evidence_fields is None
            else tuple(required_evidence_fields)
        )
        review_rules = (
            ("EVERY_POSITIVE_FACTOR",)
            if mandatory_review_rules is None
            else tuple(mandatory_review_rules)
        )

        # Validate domain invariants by constructing RiskPolicy model
        policy = RiskPolicy(
            id=policy_id,
            version=version,
            name=name,
            description=description,
            lifecycle_state=RiskPolicyLifecycleState.DRAFT,
            factor_weights=weights,
            score_cap=score_cap,
            risk_bands=bands,
            mandatory_review_rules=review_rules,
            required_evidence_fields=req_fields,
        )

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
            created_at=policy.created_at,
            updated_at=policy.updated_at,
        )

        self._session.add(record)
        await self._session.commit()
        return _record_to_policy(record)

    async def validate_and_mark_ready(self, policy_id: str) -> RiskPolicy:
        """Validate an existing DRAFT policy's rules and transition it to READY state."""
        stmt = select(RiskPolicyRecord).where(RiskPolicyRecord.id == policy_id).with_for_update()
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            raise RiskPolicyLifecycleError(f"Risk policy '{policy_id}' not found")

        current_state = RiskPolicyLifecycleState(record.lifecycle_state)
        if current_state == RiskPolicyLifecycleState.READY:
            return _record_to_policy(record)

        if current_state != RiskPolicyLifecycleState.DRAFT:
            raise RiskPolicyLifecycleError(
                f"Cannot advance risk policy '{policy_id}' in state '{current_state}' to READY"
            )

        # Validate by constructing domain model
        policy = _record_to_policy(record)
        # Verify hash integrity
        computed_hash = compute_policy_hash(
            policy_id=policy.id,
            version=policy.version,
            factor_weights=policy.factor_weights,
            score_cap=policy.score_cap,
            risk_bands=policy.risk_bands,
            required_evidence_fields=policy.required_evidence_fields,
        )
        if policy.policy_hash != computed_hash:
            raise RiskPolicyLifecycleError("Risk policy hash verification failed during validation")

        record.lifecycle_state = RiskPolicyLifecycleState.READY.value
        record.updated_at = datetime.now(UTC)
        await self._session.commit()
        return _record_to_policy(record)

    async def activate_policy(
        self, policy_id: str, operator_id: str
    ) -> tuple[RiskPolicy, RiskPolicy | None]:
        """Atomically activate a READY policy and transition previous ACTIVE policy to RETIRED."""
        # 1. Lock and validate target policy
        stmt_target = (
            select(RiskPolicyRecord).where(RiskPolicyRecord.id == policy_id).with_for_update()
        )
        result_target = await self._session.execute(stmt_target)
        target_record = result_target.scalar_one_or_none()
        if target_record is None:
            raise RiskPolicyLifecycleError(f"Risk policy '{policy_id}' not found")

        target_state = RiskPolicyLifecycleState(target_record.lifecycle_state)
        if target_state == RiskPolicyLifecycleState.ACTIVE:
            raise RiskPolicyLifecycleError(f"Risk policy '{policy_id}' is already ACTIVE")

        if target_state != RiskPolicyLifecycleState.READY:
            raise RiskPolicyLifecycleError(
                f"Cannot activate risk policy '{policy_id}' in state '{target_state}'; "
                "must be READY"
            )

        # 2. Lock and retire existing ACTIVE policy
        stmt_active = (
            select(RiskPolicyRecord)
            .where(RiskPolicyRecord.lifecycle_state == RiskPolicyLifecycleState.ACTIVE.value)
            .with_for_update()
        )
        result_active = await self._session.execute(stmt_active)
        active_record = result_active.scalar_one_or_none()

        retired_policy: RiskPolicy | None = None
        now = datetime.now(UTC)

        if active_record is not None:
            active_record.lifecycle_state = RiskPolicyLifecycleState.RETIRED.value
            active_record.retired_at = now
            active_record.updated_at = now
            retired_policy = _record_to_policy(active_record)
            await self._session.flush()

        # 3. Activate target policy
        target_record.lifecycle_state = RiskPolicyLifecycleState.ACTIVE.value
        target_record.activated_at = now
        target_record.activated_by = operator_id
        target_record.updated_at = now

        await self._session.commit()
        active_policy = _record_to_policy(target_record)

        return active_policy, retired_policy
