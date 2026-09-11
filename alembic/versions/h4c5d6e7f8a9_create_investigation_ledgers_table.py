"""Create durable bounded investigation ledgers.

Revision ID: h4c5d6e7f8a9
Revises: g3b4c5d6e7f8
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "h4c5d6e7f8a9"
down_revision: str | Sequence[str] | None = "g3b4c5d6e7f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "investigation_ledgers",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("assessment_id", sa.String(length=36), nullable=False),
        sa.Column("run_number", sa.Integer(), nullable=False),
        sa.Column("vin", sa.String(length=17), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("proposal_rounds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("supplementary_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("supplementary_retries", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("projected_cost", sa.Float(), nullable=False, server_default="0"),
        sa.Column("actual_cost", sa.Float(), nullable=False, server_default="0"),
        sa.Column("current_request_hash", sa.String(length=64), nullable=True),
        sa.Column("result_summary", sa.String(length=500), nullable=True),
        sa.Column("references_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("limits_json", sa.Text(), nullable=False),
        sa.Column("pins_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["assessment_id"], ["assessments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("assessment_id", "run_number", name="uq_investigation_ledger_run"),
    )
    op.create_index(
        "ix_investigation_ledgers_assessment_id",
        "investigation_ledgers",
        ["assessment_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_investigation_ledgers_assessment_id", table_name="investigation_ledgers")
    op.drop_table("investigation_ledgers")
