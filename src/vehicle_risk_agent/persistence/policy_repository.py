"""Transactional persistence repository for Policy Sources, Snapshots, and Passages."""

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from vehicle_risk_agent.persistence.models import (
    PolicyPassageRecord,
    PolicySnapshotRecord,
    PolicySourceRecord,
)
from vehicle_risk_agent.policy.models import (
    AuthorityClassification,
    PolicyPassage,
    PolicySnapshot,
    PolicySource,
    SourceStatus,
    ValidationOutcome,
)


def _source_record_to_domain(record: PolicySourceRecord) -> PolicySource:
    """Map PolicySourceRecord to PolicySource domain model."""
    return PolicySource(
        id=record.id,
        title=record.title,
        issuing_authority=record.issuing_authority,
        jurisdiction=record.jurisdiction,
        canonical_origin=record.canonical_origin,
        authority_classification=AuthorityClassification(record.authority_classification),
        reuse_terms=record.reuse_terms,
        expected_update_cadence=record.expected_update_cadence,
        status=SourceStatus(record.status),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _snapshot_record_to_domain(record: PolicySnapshotRecord) -> PolicySnapshot:
    """Map PolicySnapshotRecord to PolicySnapshot domain model."""
    passages = [
        PolicyPassage(
            id=p.id,
            snapshot_id=p.snapshot_id,
            source_id=p.source_id,
            section_identifier=p.section_identifier,
            heading=p.heading,
            text=p.text,
            sequence=p.sequence,
            char_offset_start=p.char_offset_start,
            char_offset_end=p.char_offset_end,
            content_hash=p.content_hash,
        )
        for p in record.passages
    ]
    metadata = json.loads(record.metadata_json) if record.metadata_json else {}
    return PolicySnapshot(
        id=record.id,
        source_id=record.source_id,
        retrieved_at=record.retrieved_at,
        effective_date=record.effective_date,
        publication_date=record.publication_date,
        content_hash=record.content_hash,
        raw_content=record.raw_content,
        parser_version=record.parser_version,
        validation_outcome=ValidationOutcome(record.validation_outcome),
        passages=tuple(passages),
        metadata=metadata,
    )


class PolicyRepository:
    """Provides transactional persistence for policy knowledge entities."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_source(self, source_id: str) -> PolicySource | None:
        """Retrieve a PolicySource by its ID."""
        stmt = select(PolicySourceRecord).where(PolicySourceRecord.id == source_id)
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return _source_record_to_domain(record)

    async def list_sources(self) -> list[PolicySource]:
        """List all active PolicySources."""
        stmt = select(PolicySourceRecord).order_by(PolicySourceRecord.id)
        result = await self._session.execute(stmt)
        records = result.scalars().all()
        return [_source_record_to_domain(r) for r in records]

    async def create_or_update_source(self, source: PolicySource) -> PolicySource:
        """Register or update a PolicySource."""
        stmt = select(PolicySourceRecord).where(PolicySourceRecord.id == source.id)
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()

        if record is None:
            record = PolicySourceRecord(
                id=source.id,
                title=source.title,
                issuing_authority=source.issuing_authority,
                jurisdiction=source.jurisdiction,
                canonical_origin=source.canonical_origin,
                authority_classification=str(source.authority_classification),
                reuse_terms=source.reuse_terms,
                expected_update_cadence=source.expected_update_cadence,
                status=str(source.status),
                created_at=source.created_at,
                updated_at=source.updated_at,
            )
            self._session.add(record)
        else:
            record.title = source.title
            record.issuing_authority = source.issuing_authority
            record.canonical_origin = source.canonical_origin
            record.authority_classification = str(source.authority_classification)
            record.reuse_terms = source.reuse_terms
            record.expected_update_cadence = source.expected_update_cadence
            record.status = str(source.status)
            record.updated_at = source.updated_at

        await self._session.commit()
        retrieved = await self.get_source(source.id)
        assert retrieved is not None
        return retrieved

    async def get_snapshot(self, snapshot_id: str) -> PolicySnapshot | None:
        """Retrieve a PolicySnapshot with its passages by snapshot ID."""
        stmt = (
            select(PolicySnapshotRecord)
            .where(PolicySnapshotRecord.id == snapshot_id)
            .options(selectinload(PolicySnapshotRecord.passages))
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return _snapshot_record_to_domain(record)

    async def get_snapshot_by_hash(
        self, source_id: str, content_hash: str
    ) -> PolicySnapshot | None:
        """Retrieve an existing snapshot for a source by its content hash."""
        stmt = (
            select(PolicySnapshotRecord)
            .where(
                PolicySnapshotRecord.source_id == source_id,
                PolicySnapshotRecord.content_hash == content_hash,
            )
            .options(selectinload(PolicySnapshotRecord.passages))
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return _snapshot_record_to_domain(record)

    async def create_snapshot(self, snapshot: PolicySnapshot) -> PolicySnapshot:
        """Persist a new immutable PolicySnapshot and its passages idempotently."""
        existing = await self.get_snapshot_by_hash(snapshot.source_id, snapshot.content_hash)
        if existing is not None:
            return existing

        snapshot_record = PolicySnapshotRecord(
            id=snapshot.id,
            source_id=snapshot.source_id,
            retrieved_at=snapshot.retrieved_at,
            effective_date=snapshot.effective_date,
            publication_date=snapshot.publication_date,
            content_hash=snapshot.content_hash,
            raw_content=snapshot.raw_content,
            parser_version=snapshot.parser_version,
            validation_outcome=str(snapshot.validation_outcome),
            metadata_json=json.dumps(snapshot.metadata),
            created_at=snapshot.retrieved_at,
        )
        self._session.add(snapshot_record)

        for p in snapshot.passages:
            passage_record = PolicyPassageRecord(
                id=p.id,
                snapshot_id=p.snapshot_id,
                source_id=p.source_id,
                section_identifier=p.section_identifier,
                heading=p.heading,
                text=p.text,
                sequence=p.sequence,
                char_offset_start=p.char_offset_start,
                char_offset_end=p.char_offset_end,
                content_hash=p.content_hash,
            )
            self._session.add(passage_record)

        await self._session.commit()
        retrieved = await self.get_snapshot(snapshot.id)
        assert retrieved is not None
        return retrieved
