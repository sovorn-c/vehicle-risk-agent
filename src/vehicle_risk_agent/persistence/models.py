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
