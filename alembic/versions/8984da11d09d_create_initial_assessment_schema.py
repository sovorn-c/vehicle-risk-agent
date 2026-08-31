"""Create initial assessment, assessment_runs, and idempotency_keys tables.

Revision ID: 8984da11d09d
Revises:
Create Date: 2026-08-31 22:06:54.248766

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8984da11d09d"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create domain tables."""
    op.create_table(
        "assessments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("requester_id", sa.String(length=64), nullable=False),
        sa.Column("vin", sa.String(length=17), nullable=False),
        sa.Column("context_json", sa.Text(), nullable=False),
        sa.Column(
            "lifecycle_state", sa.String(length=32), nullable=False, server_default="IN_PROGRESS"
        ),
        sa.Column("current_run_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_assessments_requester_id", "assessments", ["requester_id"], unique=False)
    op.create_index("ix_assessments_vin", "assessments", ["vin"], unique=False)

    op.create_table(
        "assessment_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("assessment_id", sa.String(length=36), nullable=False),
        sa.Column("run_number", sa.Integer(), nullable=False),
        sa.Column("phase", sa.String(length=32), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["assessment_id"], ["assessments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("assessment_id", "run_number", name="uq_assessment_run_number"),
    )
    op.create_index(
        "ix_assessment_runs_assessment_id", "assessment_runs", ["assessment_id"], unique=False
    )

    op.create_table(
        "idempotency_keys",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("principal_id", sa.String(length=64), nullable=False),
        sa.Column("command_type", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("target_resource_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "principal_id", "command_type", "idempotency_key", name="uq_idempotency_key"
        ),
    )


def downgrade() -> None:
    """Drop domain tables."""
    op.drop_table("idempotency_keys")
    op.drop_index("ix_assessment_runs_assessment_id", table_name="assessment_runs")
    op.drop_table("assessment_runs")
    op.drop_index("ix_assessments_vin", table_name="assessments")
    op.drop_index("ix_assessments_requester_id", table_name="assessments")
    op.drop_table("assessments")
