"""Create policy_sources, policy_snapshots, policy_passages, policy_corpora, and policy_corpus_snapshots tables.

Revision ID: 9a01b1c2d3e4
Revises: 8984da11d09d
Create Date: 2026-09-01 14:15:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9a01b1c2d3e4"
down_revision: str | Sequence[str] | None = "8984da11d09d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create policy tables, pgvector extension, and partial unique active corpus index."""
    # Ensure pgvector extension exists
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 1. Policy Sources
    op.create_table(
        "policy_sources",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("issuing_authority", sa.String(length=255), nullable=False),
        sa.Column("jurisdiction", sa.String(length=10), nullable=False, server_default="NZ"),
        sa.Column("canonical_origin", sa.String(length=1024), nullable=False),
        sa.Column("authority_classification", sa.String(length=32), nullable=False),
        sa.Column("reuse_terms", sa.String(length=255), nullable=False),
        sa.Column("expected_update_cadence", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="ACTIVE"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # 2. Policy Snapshots
    op.create_table(
        "policy_snapshots",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("raw_content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("parser_version", sa.String(length=32), nullable=False),
        sa.Column("validation_outcome", sa.String(length=32), nullable=False),
        sa.Column("effective_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("publication_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["policy_sources.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_policy_snapshots_source_id", "policy_snapshots", ["source_id"], unique=False)
    op.create_index("ix_policy_snapshots_content_hash", "policy_snapshots", ["content_hash"], unique=False)

    # 3. Policy Passages
    op.create_table(
        "policy_passages",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("section_identifier", sa.String(length=64), nullable=False),
        sa.Column("heading", sa.String(length=255), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("char_offset_start", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_offset_end", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.ForeignKeyConstraint(["snapshot_id"], ["policy_snapshots.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_id"], ["policy_sources.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_policy_passages_snapshot_id", "policy_passages", ["snapshot_id"], unique=False)
    op.create_index("ix_policy_passages_source_id", "policy_passages", ["source_id"], unique=False)

    # 4. Policy Corpora
    op.create_table(
        "policy_corpora",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False, server_default="DRAFT"),
        sa.Column("retrieval_config_json", sa.Text(), nullable=False),
        sa.Column("manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("activated_by", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    # Partial unique index: at most one active corpus at any time
    op.create_index(
        "uq_policy_corpora_single_active",
        "policy_corpora",
        ["lifecycle_state"],
        unique=True,
        postgresql_where=sa.text("lifecycle_state = 'ACTIVE'"),
    )

    # 5. Policy Corpus Snapshots Association Table
    op.create_table(
        "policy_corpus_snapshots",
        sa.Column("corpus_id", sa.String(length=64), nullable=False),
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["corpus_id"], ["policy_corpora.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["policy_snapshots.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("corpus_id", "snapshot_id"),
    )
    op.create_index("ix_policy_corpus_snapshots_corpus_id", "policy_corpus_snapshots", ["corpus_id"], unique=False)
    op.create_index("ix_policy_corpus_snapshots_snapshot_id", "policy_corpus_snapshots", ["snapshot_id"], unique=False)


def downgrade() -> None:
    """Drop policy tables and indexes."""
    op.drop_table("policy_corpus_snapshots")
    op.drop_index("uq_policy_corpora_single_active", table_name="policy_corpora")
    op.drop_table("policy_corpora")
    op.drop_index("ix_policy_passages_source_id", table_name="policy_passages")
    op.drop_index("ix_policy_passages_snapshot_id", table_name="policy_passages")
    op.drop_table("policy_passages")
    op.drop_index("ix_policy_snapshots_content_hash", table_name="policy_snapshots")
    op.drop_index("ix_policy_snapshots_source_id", table_name="policy_snapshots")
    op.drop_table("policy_snapshots")
    op.drop_table("policy_sources")
