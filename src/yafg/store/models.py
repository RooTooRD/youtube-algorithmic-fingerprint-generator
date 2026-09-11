"""Persistence model.

Design rule: **every row is append-only evidence**. The tables below record what the
agent saw and what it chose, never a mutated "current state". Reproducibility comes
from snapshotting the resolved persona and experiment into the run row, so config
edits cannot retroactively rewrite what a finished run meant.

Runs on SQLite (default, single-researcher) or Postgres (parallel agents). JSON
columns use SQLAlchemy's dialect-neutral `JSON` type for that reason.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, ClassVar

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON, list[str]: JSON}


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(64), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    mode: Mapped[str] = mapped_column(String(16))
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON)
    manifest_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    runs: Mapped[list[Run]] = relationship(back_populates="experiment")


class Run(Base):
    """One agent executing one arm of an experiment: a persona (or persona policy),
    on one account, for N steps."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    arm: Mapped[str] = mapped_column(String(64), doc="Human-readable arm label, e.g. 'amina-dz/rep2'.")
    repetition: Mapped[int] = mapped_column(Integer, default=0)
    account_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    persona_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)
    context_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    failure: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    steps: Mapped[list[Step]] = relationship(back_populates="run")


class Step(Base):
    """One observe -> decide -> act cycle."""

    __tablename__ = "steps"
    __table_args__ = (UniqueConstraint("run_id", "index", name="uq_step_run_index"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    phase: Mapped[str] = mapped_column(String(16), doc="'context' (warm-up) or 'exploration'.")
    active_persona: Mapped[str] = mapped_column(String(64), doc="Matters in mixed/sequential modes.")
    surface: Mapped[str] = mapped_column(String(24))
    source_video_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    chosen_video_id: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    chosen_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    watched_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    watch_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_justification: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_usage: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[Run] = relationship(back_populates="steps")
    observations: Mapped[list[Observation]] = relationship(back_populates="step")


class Observation(Base):
    """One recommendation slot as rendered, with its rank. This is the primary
    measurement; everything else is provenance for it."""

    __tablename__ = "observations"
    __table_args__ = (Index("ix_obs_step_rank", "step_id", "rank"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    step_id: Mapped[str] = mapped_column(ForeignKey("steps.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    video_id: Mapped[str] = mapped_column(String(16), index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    channel_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    channel_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    badge: Mapped[str | None] = mapped_column(String(64), nullable=True, doc="e.g. 'Live', 'Mix', 'Shorts'.")
    was_chosen: Mapped[bool] = mapped_column(Boolean, default=False)

    step: Mapped[Step] = relationship(back_populates="observations")


class Video(Base):
    """Enrichment cache, filled asynchronously by the YouTube Data API worker so the
    browsing loop never blocks on metadata."""

    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    channel_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    channel_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    view_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    category_id: Mapped[str | None] = mapped_column(String(8), nullable=True)
    default_language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    enriched_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)


class Event(Base):
    """Structured audit log: navigations, rate-limit waits, consent walls, CAPTCHA
    detections, login-state losses. Separate from steps so operational noise never
    pollutes the measurement tables."""

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), index=True, nullable=True)
    kind: Mapped[str] = mapped_column(String(48), index=True)
    level: Mapped[str] = mapped_column(String(8), default="info")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
