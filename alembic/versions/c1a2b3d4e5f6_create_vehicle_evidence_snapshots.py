"""Create vehicle_evidence_snapshots table.

Revision ID: c1a2b3d4e5f6
Revises: 9a01b1c2d3e4
Create Date: 2026-09-01 15:06:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c1a2b3d4e5f6"
down_revision: str | None = "9a01b1c2d3e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create vehicle_evidence_snapshots table and indexes."""
    op.create_table(
        "vehicle_evidence_snapshots",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("assessment_id", sa.String(length=36), nullable=False),
        sa.Column("run_number", sa.Integer(), nullable=False),
        sa.Column("vin", sa.String(length=17), nullable=False),
        sa.Column("revision_id", sa.String(length=64), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("material_hash", sa.String(length=64), nullable=False),
        sa.Column("snapshot_data_json", sa.Text(), nullable=False),
        sa.Column("sufficiency_json", sa.Text(), nullable=True),
        sa.Column(
            "collected_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["assessment_id"], ["assessments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("assessment_id", "run_number", name="uq_evidence_snapshot_run"),
    )
    op.create_index(
        "ix_vehicle_evidence_snapshots_assessment_id",
        "vehicle_evidence_snapshots",
        ["assessment_id"],
        unique=False,
    )
    op.create_index(
        "ix_vehicle_evidence_snapshots_vin",
        "vehicle_evidence_snapshots",
        ["vin"],
        unique=False,
    )


def downgrade() -> None:
    """Drop vehicle_evidence_snapshots table and indexes."""
    op.drop_index("ix_vehicle_evidence_snapshots_vin", table_name="vehicle_evidence_snapshots")
    op.drop_index(
        "ix_vehicle_evidence_snapshots_assessment_id",
        table_name="vehicle_evidence_snapshots",
    )
    op.drop_table("vehicle_evidence_snapshots")
