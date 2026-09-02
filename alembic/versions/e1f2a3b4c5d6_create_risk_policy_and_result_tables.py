"""Create risk_policies and risk_results tables with single active index and run constraint.

Revision ID: e1f2a3b4c5d6
Revises: d3e4f5a6b7c8
Create Date: 2026-09-01 18:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: str | Sequence[str] | None = "d3e4f5a6b7c8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. risk_policies table
    op.create_table(
        "risk_policies",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False, server_default="v1"),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False, server_default="DRAFT"),
        sa.Column("factor_weights_json", sa.Text(), nullable=False),
        sa.Column("score_cap", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("risk_bands_json", sa.Text(), nullable=False),
        sa.Column("mandatory_review_rules_json", sa.Text(), nullable=False),
        sa.Column("required_evidence_fields_json", sa.Text(), nullable=False),
        sa.Column("policy_hash", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("activated_by", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_risk_policies_single_active",
        "risk_policies",
        ["lifecycle_state"],
        unique=True,
        postgresql_where=sa.text("lifecycle_state = 'ACTIVE'"),
    )
    op.create_index(
        "ix_risk_policies_lifecycle_state",
        "risk_policies",
        ["lifecycle_state"],
        unique=False,
    )

    # 2. risk_results table
    op.create_table(
        "risk_results",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("assessment_id", sa.String(length=36), nullable=False),
        sa.Column("run_number", sa.Integer(), nullable=False),
        sa.Column("policy_id", sa.String(length=128), nullable=False),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("band", sa.String(length=32), nullable=True),
        sa.Column("raw_score", sa.Integer(), nullable=True),
        sa.Column("is_incomplete", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("calculation_hash", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("result_data_json", sa.Text(), nullable=False),
        sa.Column("calculated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["assessment_id"], ["assessments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["policy_id"], ["risk_policies.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("assessment_id", "run_number", name="uq_risk_results_run_id"),
    )
    op.create_index(
        "ix_risk_results_assessment_id", "risk_results", ["assessment_id"], unique=False
    )
    op.create_index("ix_risk_results_policy_id", "risk_results", ["policy_id"], unique=False)


def downgrade() -> None:
    op.drop_table("risk_results")
    op.drop_index("ix_risk_policies_lifecycle_state", table_name="risk_policies")
    op.drop_index("uq_risk_policies_single_active", table_name="risk_policies")
    op.drop_table("risk_policies")
