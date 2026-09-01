"""Add local integrity hash for vehicle evidence snapshots.

Revision ID: d3e4f5a6b7c8
Revises: c1a2b3d4e5f6
Create Date: 2026-09-01 16:30:00.000000
"""

import hashlib
import hmac
import os
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d3e4f5a6b7c8"
down_revision: str | None = "c1a2b3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add and backfill the serialized snapshot integrity hash."""
    op.add_column(
        "vehicle_evidence_snapshots",
        sa.Column("snapshot_integrity_hash", sa.String(length=64), nullable=True),
    )
    connection = op.get_bind()
    secret = os.environ.get(
        "SNAPSHOT_INTEGRITY_SECRET",
        "dev-snapshot-integrity-secret-v1-32chars",
    ).encode("utf-8")
    rows = connection.execute(
        sa.select(
            sa.column("id", sa.String(length=36)),
            sa.column("snapshot_data_json", sa.Text),
        ).select_from(sa.table("vehicle_evidence_snapshots"))
    )
    for row in rows:
        integrity_hash = hmac.new(
            secret,
            row.snapshot_data_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        connection.execute(
            sa.text(
                "UPDATE vehicle_evidence_snapshots "
                "SET snapshot_integrity_hash = :integrity_hash WHERE id = :id"
            ),
            {"integrity_hash": integrity_hash, "id": row.id},
        )
    op.alter_column(
        "vehicle_evidence_snapshots",
        "snapshot_integrity_hash",
        existing_type=sa.String(length=64),
        nullable=False,
    )


def downgrade() -> None:
    """Remove the serialized snapshot integrity hash."""
    op.drop_column("vehicle_evidence_snapshots", "snapshot_integrity_hash")
