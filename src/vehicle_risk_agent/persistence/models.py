"""SQLAlchemy ORM models for domain records and idempotency tracking."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for domain models."""


class AssessmentRecord(Base):
    """Authoritative persistent record for Assessment aggregate."""

    __tablename__ = "assessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    requester_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    vin: Mapped[str] = mapped_column(String(17), nullable=False, index=True)
    context_json: Mapped[str] = mapped_column(Text, nullable=False)
    lifecycle_state: Mapped[str] = mapped_column(String(32), nullable=False, default="IN_PROGRESS")
    current_run_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    runs: Mapped[list["AssessmentRunRecord"]] = relationship(
        "AssessmentRunRecord",
        back_populates="assessment",
        cascade="all, delete-orphan",
        order_by="AssessmentRunRecord.run_number",
    )


class AssessmentRunRecord(Base):
    """Persistent record for an individual Assessment Run."""

    __tablename__ = "assessment_runs"
    __table_args__ = (
        UniqueConstraint("assessment_id", "run_number", name="uq_assessment_run_number"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    assessment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_number: Mapped[int] = mapped_column(Integer, nullable=False)
    phase: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    assessment: Mapped["AssessmentRecord"] = relationship("AssessmentRecord", back_populates="runs")


class IdempotencyRecord(Base):
    """Persistent tracking of idempotency keys per principal and command type."""

    __tablename__ = "idempotency_keys"
    __table_args__ = (
        UniqueConstraint(
            "principal_id", "command_type", "idempotency_key", name="uq_idempotency_key"
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    principal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    command_type: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    target_resource_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class WorkflowEventRecord(Base):
    """Persistent storage for sanitized workflow progress events."""

    __tablename__ = "workflow_events"
    __table_args__ = (
        UniqueConstraint("assessment_id", "run_number", "sequence", name="uq_event_sequence"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    assessment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_number: Mapped[int] = mapped_column(Integer, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    phase: Mapped[str] = mapped_column(String(32), nullable=False)
    safe_message: Mapped[str] = mapped_column(String(500), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class PolicySourceRecord(Base):
    """Authoritative persistent record for official Policy Source definitions."""

    __tablename__ = "policy_sources"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    issuing_authority: Mapped[str] = mapped_column(String(256), nullable=False)
    jurisdiction: Mapped[str] = mapped_column(String(2), nullable=False, default="NZ")
    canonical_origin: Mapped[str] = mapped_column(String(1024), nullable=False)
    authority_classification: Mapped[str] = mapped_column(String(64), nullable=False)
    reuse_terms: Mapped[str] = mapped_column(String(512), nullable=False)
    expected_update_cadence: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    snapshots: Mapped[list["PolicySnapshotRecord"]] = relationship(
        "PolicySnapshotRecord",
        back_populates="source",
        cascade="all, delete-orphan",
        order_by="desc(PolicySnapshotRecord.retrieved_at)",
    )


class PolicySnapshotRecord(Base):
    """Immutable captured point-in-time snapshot of a Policy Source."""

    __tablename__ = "policy_snapshots"
    __table_args__ = (
        UniqueConstraint("source_id", "content_hash", name="uq_snapshot_source_hash"),
    )

    id: Mapped[str] = mapped_column(String(256), primary_key=True)
    source_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("policy_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    effective_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    publication_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    parser_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="policy-parser-v1"
    )
    validation_outcome: Mapped[str] = mapped_column(String(32), nullable=False, default="VALID")
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    source: Mapped["PolicySourceRecord"] = relationship(
        "PolicySourceRecord", back_populates="snapshots"
    )
    passages: Mapped[list["PolicyPassageRecord"]] = relationship(
        "PolicyPassageRecord",
        back_populates="snapshot",
        cascade="all, delete-orphan",
        order_by="PolicyPassageRecord.sequence",
    )


class PolicyPassageRecord(Base):
    """Immutable section passage extracted from a Policy Snapshot."""

    __tablename__ = "policy_passages"
    __table_args__ = (UniqueConstraint("snapshot_id", "sequence", name="uq_passage_snapshot_seq"),)

    id: Mapped[str] = mapped_column(String(256), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        String(256),
        ForeignKey("policy_snapshots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    section_identifier: Mapped[str] = mapped_column(String(128), nullable=False)
    heading: Mapped[str] = mapped_column(String(256), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    char_offset_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_offset_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    snapshot: Mapped["PolicySnapshotRecord"] = relationship(
        "PolicySnapshotRecord", back_populates="passages"
    )


class PolicyCorpusRecord(Base):
    """Authoritative persistent record for versioned Policy Corpus manifests."""

    __tablename__ = "policy_corpora"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(String(1024), nullable=False)
    lifecycle_state: Mapped[str] = mapped_column(
        String(32), nullable=False, default="DRAFT", index=True
    )
    snapshot_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_config_json: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    activated_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
