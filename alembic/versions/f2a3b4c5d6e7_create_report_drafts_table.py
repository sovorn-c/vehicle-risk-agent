"""Create report_drafts table with run constraint and indices.

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-09-01 19:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f2a3b4c5d6e7"
down_revision: str | Sequence[str] | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "report_drafts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("assessment_id", sa.String(length=36), nullable=False),
        sa.Column("run_number", sa.Integer(), nullable=False),
        sa.Column("vehicle_id", sa.String(length=64), nullable=False),
        sa.Column("policy_id", sa.String(length=128), nullable=False),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("band", sa.String(length=32), nullable=True),
        sa.Column("draft_hash", sa.String(length=64), nullable=False),
        sa.Column("draft_data_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["assessment_id"], ["assessments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["policy_id"], ["risk_policies.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("assessment_id", "run_number", name="uq_report_draft_run"),
    )
    op.create_index(
        "ix_report_drafts_assessment_id",
        "report_drafts",
        ["assessment_id"],
        unique=False,
    )
    op.create_index(
        "ix_report_drafts_vehicle_id",
        "report_drafts",
        ["vehicle_id"],
        unique=False,
    )
    op.create_index(
        "ix_report_drafts_policy_id",
        "report_drafts",
        ["policy_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_report_drafts_policy_id", table_name="report_drafts")
    op.drop_index("ix_report_drafts_vehicle_id", table_name="report_drafts")
    op.drop_index("ix_report_drafts_assessment_id", table_name="report_drafts")
    op.drop_table("report_drafts")
