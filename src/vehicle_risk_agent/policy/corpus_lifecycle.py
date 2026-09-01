"""Corpus lifecycle manager for validating and activating immutable Policy Corpus Versions."""

import json
from datetime import UTC, datetime

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.persistence.models import PolicyCorpusRecord, PolicySnapshotRecord
from vehicle_risk_agent.policy.corpus_models import (
    CorpusLifecycleState,
    PolicyCorpusManifest,
    RetrievalConfiguration,
    build_corpus_manifest,
)


class CorpusLifecycleError(Exception):
    """Raised when a corpus lifecycle transition or validation fails."""


def _corpus_record_to_manifest(record: PolicyCorpusRecord) -> PolicyCorpusManifest:
    """Map PolicyCorpusRecord to PolicyCorpusManifest domain model."""
    snapshot_ids = json.loads(record.snapshot_ids_json)
    retrieval_data = json.loads(record.retrieval_config_json)
    retrieval_config = RetrievalConfiguration(**retrieval_data)

    return PolicyCorpusManifest(
        id=record.id,
        name=record.name,
        description=record.description,
        lifecycle_state=CorpusLifecycleState(record.lifecycle_state),
        snapshot_ids=snapshot_ids,
        retrieval_config=retrieval_config,
        manifest_hash=record.manifest_hash,
        created_at=record.created_at,
        activated_at=record.activated_at,
        retired_at=record.retired_at,
    )


class CorpusLifecycleManager:
    """Manages transactional state transitions for Policy Corpora."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_corpus(self, corpus_id: str) -> PolicyCorpusManifest | None:
        """Retrieve a corpus manifest by ID."""
        stmt = select(PolicyCorpusRecord).where(PolicyCorpusRecord.id == corpus_id)
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return _corpus_record_to_manifest(record)

    async def get_active_corpus(self) -> PolicyCorpusManifest | None:
        """Retrieve the single currently ACTIVE corpus manifest, if one exists."""
        stmt = select(PolicyCorpusRecord).where(
            PolicyCorpusRecord.lifecycle_state == CorpusLifecycleState.ACTIVE.value
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return _corpus_record_to_manifest(record)

    async def list_corpora(self) -> list[PolicyCorpusManifest]:
        """List all corpus manifests ordered by creation timestamp."""
        stmt = select(PolicyCorpusRecord).order_by(desc(PolicyCorpusRecord.created_at))
        result = await self._session.execute(stmt)
        records = result.scalars().all()
        return [_corpus_record_to_manifest(r) for r in records]

    async def create_corpus(
        self,
        corpus_id: str,
        name: str,
        description: str,
        snapshot_ids: list[str],
        retrieval_config: RetrievalConfiguration | None = None,
    ) -> PolicyCorpusManifest:
        """Create a new DRAFT corpus manifest."""
        manifest = build_corpus_manifest(
            corpus_id=corpus_id,
            name=name,
            description=description,
            snapshot_ids=snapshot_ids,
            retrieval_config=retrieval_config,
            lifecycle_state=CorpusLifecycleState.DRAFT,
        )

        record = PolicyCorpusRecord(
            id=manifest.id,
            name=manifest.name,
            description=manifest.description,
            lifecycle_state=manifest.lifecycle_state.value,
            snapshot_ids_json=json.dumps(manifest.snapshot_ids),
            retrieval_config_json=manifest.retrieval_config.model_dump_json(),
            manifest_hash=manifest.manifest_hash,
            created_at=manifest.created_at,
        )
        self._session.add(record)
        await self._session.commit()

        retrieved = await self.get_corpus(corpus_id)
        assert retrieved is not None
        return retrieved

    async def validate_and_mark_ready(self, corpus_id: str) -> PolicyCorpusManifest:
        """Validate referenced snapshots and advance DRAFT corpus to READY."""
        stmt = select(PolicyCorpusRecord).where(PolicyCorpusRecord.id == corpus_id)
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()

        if record is None:
            raise CorpusLifecycleError(f"Corpus {corpus_id} not found")

        if record.lifecycle_state != CorpusLifecycleState.DRAFT.value:
            raise CorpusLifecycleError(
                f"Corpus {corpus_id} is in {record.lifecycle_state} state; "
                "only DRAFT can become READY"
            )

        snapshot_ids: list[str] = json.loads(record.snapshot_ids_json)
        if not snapshot_ids:
            raise CorpusLifecycleError("Corpus must include at least one snapshot")

        # Verify all snapshots exist in database and are VALID
        for snap_id in snapshot_ids:
            snap_stmt = select(PolicySnapshotRecord).where(PolicySnapshotRecord.id == snap_id)
            snap_res = await self._session.execute(snap_stmt)
            snap_rec = snap_res.scalar_one_or_none()
            if snap_rec is None:
                raise CorpusLifecycleError(f"Referenced snapshot {snap_id} does not exist")
            if snap_rec.validation_outcome != "VALID":
                raise CorpusLifecycleError(
                    f"Referenced snapshot {snap_id} has invalid validation outcome: "
                    f"{snap_rec.validation_outcome}"
                )

        record.lifecycle_state = CorpusLifecycleState.READY.value
        await self._session.commit()

        retrieved = await self.get_corpus(corpus_id)
        assert retrieved is not None
        return retrieved

    async def activate_corpus(
        self,
        corpus_id: str,
        principal_id: str,
    ) -> tuple[PolicyCorpusManifest, PolicyCorpusManifest | None]:
        """Atomically activate a READY corpus and retire the previous ACTIVE corpus."""
        stmt = select(PolicyCorpusRecord).where(PolicyCorpusRecord.id == corpus_id)
        result = await self._session.execute(stmt)
        target_record = result.scalar_one_or_none()

        if target_record is None:
            raise CorpusLifecycleError(f"Corpus {corpus_id} not found")

        if target_record.lifecycle_state != CorpusLifecycleState.READY.value:
            raise CorpusLifecycleError(
                f"Corpus {corpus_id} is in {target_record.lifecycle_state} state; "
                "must be in READY state to activate"
            )

        now = datetime.now(UTC)

        # Find currently active corpus
        active_stmt = select(PolicyCorpusRecord).where(
            PolicyCorpusRecord.lifecycle_state == CorpusLifecycleState.ACTIVE.value
        )
        active_res = await self._session.execute(active_stmt)
        current_active = active_res.scalar_one_or_none()

        prior_manifest: PolicyCorpusManifest | None = None
        if current_active is not None:
            current_active.lifecycle_state = CorpusLifecycleState.RETIRED.value
            current_active.retired_at = now
            prior_manifest = _corpus_record_to_manifest(current_active)

        target_record.lifecycle_state = CorpusLifecycleState.ACTIVE.value
        target_record.activated_at = now
        target_record.activated_by = principal_id

        await self._session.commit()

        active_manifest = await self.get_corpus(corpus_id)
        assert active_manifest is not None

        return active_manifest, prior_manifest
