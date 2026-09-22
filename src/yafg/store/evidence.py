"""Evidence-store port and SQLAlchemy implementation used by the agent protocol."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from yafg.llm.base import Candidate, Choice, Usage
from yafg.store.models import Event, Observation, Run, Step


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class EvidenceStore(Protocol):
    async def start_run(self, run_id: str, *, resume: bool = False) -> None: ...

    async def record_event(
        self,
        run_id: str,
        kind: str,
        *,
        payload: Mapping[str, Any] | None = None,
        level: str = "info",
    ) -> None: ...

    async def record_step(
        self,
        *,
        run_id: str,
        index: int,
        phase: str,
        active_persona: str | None,
        surface: str,
        source_video_id: str | None,
        candidates: list[Candidate],
        choice: Choice | None,
        usage: Usage | None,
        chosen_video_id: str | None,
    ) -> str: ...

    async def record_watch_result(
        self,
        step_id: str,
        *,
        watched_seconds: float,
        watch_fraction: float,
    ) -> None: ...

    async def save_checkpoint(self, run_id: str, checkpoint: Mapping[str, Any]) -> None: ...

    async def load_checkpoint(self, run_id: str) -> dict[str, Any]: ...

    async def finish_run(self, run_id: str) -> None: ...

    async def fail_run(self, run_id: str, failure: str) -> None: ...


class SQLAlchemyEvidenceStore:
    """SQLAlchemy adapter.

    Observation rows are inserted once and never updated. Run lifecycle fields,
    deterministic resume checkpoints, and watch-result fields are mutable metadata.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def start_run(self, run_id: str, *, resume: bool = False) -> None:
        async with AsyncSession(self.engine) as session:
            run = await session.get(Run, run_id)
            if run is None:
                raise KeyError(f"unknown run {run_id}")
            run.status = "running"
            if run.started_at is None:
                run.started_at = _now()
            run.ended_at = None
            run.failure = None
            if resume:
                run.resume_count += 1
            await session.commit()

    async def record_event(
        self,
        run_id: str,
        kind: str,
        *,
        payload: Mapping[str, Any] | None = None,
        level: str = "info",
    ) -> None:
        async with AsyncSession(self.engine) as session:
            session.add(Event(run_id=run_id, kind=kind, level=level, payload=dict(payload or {})))
            await session.commit()

    async def record_step(
        self,
        *,
        run_id: str,
        index: int,
        phase: str,
        active_persona: str | None,
        surface: str,
        source_video_id: str | None,
        candidates: list[Candidate],
        choice: Choice | None,
        usage: Usage | None,
        chosen_video_id: str | None,
    ) -> str:
        async with AsyncSession(self.engine) as session:
            step = Step(
                run_id=run_id,
                index=index,
                phase=phase,
                active_persona=active_persona,
                surface=surface,
                source_video_id=source_video_id,
                chosen_video_id=chosen_video_id,
                chosen_rank=choice.rank if choice else None,
                llm_justification=choice.justification if choice else None,
                llm_usage=usage.model_dump(mode="json") if usage else {},
            )
            session.add(step)
            await session.flush()
            step_id = step.id
            chosen_rank = choice.rank if choice else None
            session.add_all(
                Observation(
                    step_id=step_id,
                    rank=candidate.rank,
                    video_id=candidate.video_id,
                    title=candidate.title,
                    channel_name=candidate.channel_name,
                    channel_id=candidate.channel_id,
                    duration_label=candidate.duration_label,
                    badge=candidate.badge,
                    was_chosen=candidate.rank == chosen_rank,
                )
                for candidate in candidates
            )
            await session.commit()
            return step_id

    async def record_watch_result(
        self,
        step_id: str,
        *,
        watched_seconds: float,
        watch_fraction: float,
    ) -> None:
        async with AsyncSession(self.engine) as session:
            step = await session.get(Step, step_id)
            if step is None:
                raise KeyError(f"unknown step {step_id}")
            step.watched_seconds = watched_seconds
            step.watch_fraction = watch_fraction
            await session.commit()

    async def save_checkpoint(self, run_id: str, checkpoint: Mapping[str, Any]) -> None:
        async with AsyncSession(self.engine) as session:
            run = await session.get(Run, run_id)
            if run is None:
                raise KeyError(f"unknown run {run_id}")
            run.checkpoint = dict(checkpoint)
            await session.commit()

    async def load_checkpoint(self, run_id: str) -> dict[str, Any]:
        async with AsyncSession(self.engine) as session:
            run = await session.get(Run, run_id)
            if run is None:
                raise KeyError(f"unknown run {run_id}")
            return dict(run.checkpoint or {})

    async def finish_run(self, run_id: str) -> None:
        async with AsyncSession(self.engine) as session:
            run = await session.get(Run, run_id)
            if run is None:
                raise KeyError(f"unknown run {run_id}")
            run.status = "complete"
            run.failure = None
            run.ended_at = _now()
            await session.commit()

    async def fail_run(self, run_id: str, failure: str) -> None:
        async with AsyncSession(self.engine) as session:
            run = await session.get(Run, run_id)
            if run is None:
                raise KeyError(f"unknown run {run_id}")
            run.status = "failed"
            run.failure = failure[:4000]
            run.ended_at = _now()
            await session.commit()

    async def steps_for_run(self, run_id: str) -> list[Step]:
        """Convenience for tests/recovery inspection; the agent loop does not depend on it."""
        async with AsyncSession(self.engine) as session:
            query = select(Step).where(Step.run_id == run_id).order_by(Step.index)
            return list((await session.scalars(query)).all())
