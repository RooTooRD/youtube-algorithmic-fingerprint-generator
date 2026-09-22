"""p3 scale, resume checkpoints, and enrichment state

Revision ID: c31d9e7a4f20
Revises: 8f2c3a1d9b7e
Create Date: 2026-09-21 03:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c31d9e7a4f20"
down_revision: str | None = "8f2c3a1d9b7e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("accounts", sa.Column("lease_id", sa.String(length=36), nullable=True))
    op.add_column("accounts", sa.Column("lease_acquired_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_accounts_lease_id", "accounts", ["lease_id"], unique=False)
    op.create_index("uq_accounts_profile_dir", "accounts", ["profile_dir"], unique=True)

    op.add_column("runs", sa.Column("checkpoint", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))
    op.add_column("runs", sa.Column("resume_count", sa.Integer(), nullable=False, server_default="0"))
    # Keep one run per experiment arm and repetition on both supported databases.
    op.create_index(
        "uq_run_experiment_arm_rep",
        "runs",
        ["experiment_id", "arm", "repetition"],
        unique=True,
    )

    op.add_column("videos", sa.Column("captions_available", sa.Boolean(), nullable=True))
    op.add_column(
        "videos", sa.Column("metadata_status", sa.String(length=16), nullable=False, server_default="pending")
    )
    op.add_column("videos", sa.Column("metadata_error", sa.Text(), nullable=True))
    op.add_column(
        "videos", sa.Column("transcript_status", sa.String(length=16), nullable=False, server_default="pending")
    )
    op.add_column("videos", sa.Column("transcript_language", sa.String(length=16), nullable=True))
    op.add_column("videos", sa.Column("transcript_error", sa.Text(), nullable=True))
    op.add_column("videos", sa.Column("transcript_enriched_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_videos_metadata_status", "videos", ["metadata_status"], unique=False)
    op.create_index("ix_videos_transcript_status", "videos", ["transcript_status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_videos_transcript_status", table_name="videos")
    op.drop_index("ix_videos_metadata_status", table_name="videos")
    op.drop_column("videos", "transcript_enriched_at")
    op.drop_column("videos", "transcript_error")
    op.drop_column("videos", "transcript_language")
    op.drop_column("videos", "transcript_status")
    op.drop_column("videos", "metadata_error")
    op.drop_column("videos", "metadata_status")
    op.drop_column("videos", "captions_available")

    op.drop_index("uq_run_experiment_arm_rep", table_name="runs")
    op.drop_column("runs", "resume_count")
    op.drop_column("runs", "checkpoint")

    op.drop_index("uq_accounts_profile_dir", table_name="accounts")
    op.drop_index("ix_accounts_lease_id", table_name="accounts")
    op.drop_column("accounts", "lease_acquired_at")
    op.drop_column("accounts", "lease_id")
