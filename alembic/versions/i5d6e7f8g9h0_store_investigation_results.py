"""Store complete typed investigation results for replay.

Revision ID: i5d6e7f8g9h0
Revises: h4c5d6e7f8g9
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "i5d6e7f8g9h0"
down_revision: str | Sequence[str] | None = "h4c5d6e7f8g9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "investigation_ledgers",
        sa.Column("result_data_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("investigation_ledgers", "result_data_json")
