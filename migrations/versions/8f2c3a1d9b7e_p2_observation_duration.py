"""preserve rendered duration label in observations

Revision ID: 8f2c3a1d9b7e
Revises: 372f4cd54c3f
Create Date: 2026-09-21 01:40:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8f2c3a1d9b7e"
down_revision: str | None = "372f4cd54c3f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("observations", sa.Column("duration_label", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("observations", "duration_label")
