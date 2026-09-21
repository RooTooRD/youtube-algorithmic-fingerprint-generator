"""Persistence model for experiment evidence and provisioned account metadata."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, ClassVar

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON, list[str]: JSON}


class AccountRecord(Base):
    """Persistent metadata for one manually provisioned browser identity.

    Secrets are never stored here. Proxy passwords are represented only by the name
    of an environment variable in ``proxy_config``.
    """

    __tablename__ = "accounts"

    label: Mapped[str] = mapped_column(String(64), primary_key=True)
    persona_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(16), default="unprovisioned", index=True)
    profile_dir: Mapped[str] = mapped_column(Text)
    locale: Mapped[str] = mapped_column(String(32), default="en-US")
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    country: Mapped[str | None] = mapped_column(String(2), nullable=True)
    geolocation: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    viewport: Mapped[list[int]] = mapped_column(JSON, default=lambda: [1440, 900])
    proxy_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_checked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


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
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    arm: Mapped[str] = mapped_column(String(64))
    repetition: Mapped[int] = mapped_column(Integer, default=0)
    account_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    persona_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)
    context_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)
    behavior_seed: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    failure: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    steps: Mapped[list[Step]] = relationship(back_populates="run")


class Step(Base):
    __tablename__ = "steps"
    __table_args__ = (UniqueConstraint("run_id", "index", name="uq_step_run_index"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    phase: Mapped[str] = mapped_column(String(16))
    active_persona: Mapped[str | None] = mapped_column(String(64), nullable=True)
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
    """One rendered recommendation slot. Metadata may be absent; rank may not."""

    __tablename__ = "observations"
    __table_args__ = (UniqueConstraint("step_id", "rank", name="uq_observation_step_rank"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    step_id: Mapped[str] = mapped_column(ForeignKey("steps.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    video_id: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    channel_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    channel_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    duration_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    badge: Mapped[str | None] = mapped_column(String(64), nullable=True)
    was_chosen: Mapped[bool] = mapped_column(Boolean, default=False)

    step: Mapped[Step] = relationship(back_populates="observations")


class Video(Base):
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
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), index=True, nullable=True)
    kind: Mapped[str] = mapped_column(String(48), index=True)
    level: Mapped[str] = mapped_column(String(8), default="info")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
