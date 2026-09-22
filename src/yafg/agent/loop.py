"""The two-phase experimental agent protocol.

Phase 1 watches an identical context list with no active persona. Phase 2 repeatedly
observes a ranked surface, chooses one rendered candidate, watches it according to
reproducible behavioral habits, and persists the complete evidence set.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yafg.agent.choice import NoSelectableCandidate, ResolvedChoice, resolve_choice
from yafg.agent.prompt import render_persona_prompt
from yafg.browser.driver import YouTubeDriver
from yafg.experiment.schema import Context, Experiment, Surface
from yafg.llm.base import Candidate, Choice, LLMProvider, Usage
from yafg.personas.schema import Persona, ViewingHabits
from yafg.store.evidence import EvidenceStore

Sleep = Callable[[float], Awaitable[None]]


class RunCheckpoint(BaseModel):
    """Deterministic continuation state committed after a protocol step is durable."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    phase: Literal["context", "exploration", "complete"] = "context"
    next_context_index: int = Field(default=0, ge=0)
    next_exploration_index: int = Field(default=0, ge=0)
    global_step_index: int = Field(default=0, ge=0)
    last_video_id: str | None = None
    history: list[str] = Field(default_factory=list)
    rng_state: list[Any] | None = None
    session_remaining: int = Field(default=0, ge=0)
    session_index: int = -1
    not_before: float = Field(default=0, ge=0)
    in_flight: str | None = None


def _state_to_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_state_to_json(item) for item in value]
    if isinstance(value, list):
        return [_state_to_json(item) for item in value]
    return value


def _state_to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_state_to_tuple(item) for item in value)
    return value


@dataclass(slots=True)
class PersonaPolicy:
    """Resolve the active persona at exploration step ``N``.

    Mixed-mode draws are derived from ``behavior_seed`` and the step number rather
    than mutable global RNG state. Asking for the same step twice therefore returns
    the same persona, which makes retries and debugging reproducible.
    """

    experiment: Experiment
    personas: dict[str, Persona]

    def __post_init__(self) -> None:
        missing = [ref.persona for ref in self.experiment.personas if ref.persona not in self.personas]
        if missing:
            raise ValueError(f"missing personas for policy: {', '.join(missing)}")

    def active_at(self, step_index: int) -> Persona | None:
        if step_index < 0:
            raise ValueError("step_index must be non-negative")
        refs = self.experiment.personas
        mode = self.experiment.mode
        if mode == "random":
            return None
        if mode == "single":
            return self.personas[refs[0].persona]
        if mode == "sequential":
            switch_every = self.experiment.switch_every
            if switch_every is None:
                raise AssertionError("sequential mode has no switch_every")
            ref = refs[(step_index // switch_every) % len(refs)]
            return self.personas[ref.persona]
        if mode == "mixed":
            rng = random.Random(f"{self.experiment.behavior_seed}:persona:{step_index}")
            chosen = rng.choices(refs, weights=[ref.weight for ref in refs], k=1)[0]
            return self.personas[chosen.persona]
        raise AssertionError(f"unhandled behavior mode {mode!r}")

    @property
    def baseline_habits(self) -> ViewingHabits:
        """Habits for random mode without making the baseline persona model-visible."""
        return self.personas[self.experiment.personas[0].persona].habits


@dataclass(slots=True)
class _RateGate:
    max_requests_per_hour: int
    enabled: bool = True
    sleep: Sleep = asyncio.sleep
    not_before: float = field(default=0, init=False)

    async def before_request(self) -> None:
        if not self.enabled:
            return
        await self.sleep(max(0.0, self.not_before - time.time()))
        self.not_before = time.time() + 3600.0 / self.max_requests_per_hour


class AgentRun:
    def __init__(
        self,
        *,
        experiment: Experiment,
        context: Context,
        policy: PersonaPolicy,
        driver: YouTubeDriver,
        llm: LLMProvider,
        store: EvidenceStore,
        run_id: str,
        checkpoint: Mapping[str, Any] | None = None,
        sleep: Sleep = asyncio.sleep,
        enforce_pacing: bool = True,
    ) -> None:
        self.experiment = experiment
        self.context = context
        self.policy = policy
        self.driver = driver
        self.llm = llm
        self.store = store
        self.run_id = run_id
        self.sleep = sleep
        self.enforce_pacing = enforce_pacing
        self.rng = random.Random(f"{experiment.behavior_seed}:behavior")
        self.rate_gate = _RateGate(
            experiment.pacing.max_requests_per_hour,
            enabled=enforce_pacing,
            sleep=sleep,
        )
        self.checkpoint = RunCheckpoint.model_validate(checkpoint or {})
        if self.checkpoint.rng_state is not None:
            self.rng.setstate(_state_to_tuple(self.checkpoint.rng_state))
        self.rate_gate.not_before = self.checkpoint.not_before
        self._history = list(self.checkpoint.history)
        self._last_video_id = self.checkpoint.last_video_id
        self._global_step_index = self.checkpoint.global_step_index
        self._session_remaining = self.checkpoint.session_remaining
        self._session_index = self.checkpoint.session_index

    @property
    def is_resume(self) -> bool:
        return bool(
            self.checkpoint.global_step_index
            or self.checkpoint.next_context_index
            or self.checkpoint.next_exploration_index
        )

    async def execute(self) -> None:
        """Run context + exploration, preserving all evidence written before failure."""
        await self.store.start_run(self.run_id, resume=self.is_resume)
        await self.store.record_event(
            self.run_id,
            "run_resumed" if self.is_resume else "run_started",
            payload={
                "mode": self.experiment.mode,
                "next_context_index": self.checkpoint.next_context_index,
                "next_exploration_index": self.checkpoint.next_exploration_index,
            },
        )
        try:
            await self._save_checkpoint(
                phase=self.checkpoint.phase,
                next_context_index=self.checkpoint.next_context_index,
                next_exploration_index=self.checkpoint.next_exploration_index,
                in_flight="sign_in",
            )
            await self.rate_gate.before_request()
            await self.driver.assert_signed_in()
            await self._save_checkpoint(
                phase=self.checkpoint.phase,
                next_context_index=self.checkpoint.next_context_index,
                next_exploration_index=self.checkpoint.next_exploration_index,
            )
            await self._run_context_phase()
            await self._run_exploration_phase()
        except Exception as exc:
            await self.store.record_event(
                self.run_id,
                "run_failed",
                payload={"error_type": type(exc).__name__, "message": str(exc)},
                level="error",
            )
            await self.store.fail_run(self.run_id, f"{type(exc).__name__}: {exc}")
            raise
        await self._save_checkpoint(
            phase="complete",
            next_context_index=len(self.context.videos),
            next_exploration_index=self.experiment.steps,
        )
        await self.store.record_event(self.run_id, "run_completed")
        await self.store.finish_run(self.run_id)

    async def _run_context_phase(self) -> None:
        start = self.checkpoint.next_context_index
        if start >= len(self.context.videos):
            return
        await self.store.record_event(
            self.run_id,
            "context_started",
            payload={"context": self.context.id, "videos": len(self.context.videos), "resume_from": start},
        )
        for phase_index in range(start, len(self.context.videos)):
            video_id = self.context.videos[phase_index]
            step_id = await self.store.record_step(
                run_id=self.run_id,
                index=self._next_step_index(),
                phase="context",
                active_persona=None,
                surface="context",
                source_video_id=self._last_video_id,
                candidates=[],
                choice=None,
                usage=None,
                chosen_video_id=video_id,
            )
            await self.rate_gate.before_request()
            result = await self.driver.watch(
                video_id,
                seconds=self.context.max_watch_seconds,
                fraction=self.context.watch_fraction,
            )
            await self.store.record_watch_result(
                step_id,
                watched_seconds=result.watched_seconds,
                watch_fraction=result.watch_fraction,
            )
            self._last_video_id = video_id
            await self._save_checkpoint(
                phase="exploration" if phase_index + 1 == len(self.context.videos) else "context",
                next_context_index=phase_index + 1,
                next_exploration_index=0,
            )
            await self.store.record_event(
                self.run_id,
                "context_video_watched",
                payload={
                    "phase_index": phase_index,
                    "video_id": video_id,
                    "watched_seconds": result.watched_seconds,
                    "watch_fraction": result.watch_fraction,
                },
            )
        await self.store.record_event(self.run_id, "context_completed")

    async def _run_exploration_phase(self) -> None:
        start = self.checkpoint.next_exploration_index
        if start >= self.experiment.steps:
            return
        await self.store.record_event(
            self.run_id,
            "exploration_started",
            payload={"steps": self.experiment.steps, "resume_from": start},
        )
        for phase_index in range(start, self.experiment.steps):
            persona = self.policy.active_at(phase_index)
            habits = persona.habits if persona is not None else self.policy.baseline_habits
            if self._session_remaining <= 0:
                if phase_index > 0:
                    gap = self.rng.uniform(
                        self.experiment.pacing.session_gap_seconds.min,
                        self.experiment.pacing.session_gap_seconds.max,
                    )
                    await self.store.record_event(
                        self.run_id,
                        "session_gap",
                        payload={"seconds": gap, "after_session": self._session_index},
                    )
                    if self.enforce_pacing:
                        await self.sleep(gap)
                self._session_index += 1
                self._session_remaining = self.rng.randint(
                    int(habits.videos_per_session.min),
                    int(habits.videos_per_session.max),
                )
                await self.store.record_event(
                    self.run_id,
                    "session_started",
                    payload={
                        "session_index": self._session_index,
                        "planned_videos": min(self._session_remaining, self.experiment.steps - phase_index),
                        "persona_at_start": persona.id if persona else None,
                    },
                )
            surface = self._surface_for_step(phase_index, habits)
            await self._save_checkpoint(
                phase="exploration",
                next_context_index=len(self.context.videos),
                next_exploration_index=phase_index,
                in_flight=f"collect:{surface}",
            )

            await self.rate_gate.before_request()
            candidates = await self._collect(surface, persona)
            try:
                resolved = await self._choose(persona, candidates)
            except Exception as exc:
                await self.store.record_step(
                    run_id=self.run_id,
                    index=self._next_step_index(),
                    phase="exploration",
                    active_persona=persona.id if persona else None,
                    surface=surface,
                    source_video_id=self._last_video_id,
                    candidates=candidates,
                    choice=None,
                    usage=None,
                    chosen_video_id=None,
                )
                await self._save_checkpoint(
                    phase="exploration",
                    next_context_index=len(self.context.videos),
                    next_exploration_index=phase_index + 1,
                )
                await self.store.record_event(
                    self.run_id,
                    "decision_failed",
                    payload={"phase_index": phase_index, "error_type": type(exc).__name__, "message": str(exc)},
                    level="error",
                )
                raise

            video_id = resolved.candidate.video_id
            if not video_id:
                raise NoSelectableCandidate("resolved candidate unexpectedly has no video ID")
            step_id = await self.store.record_step(
                run_id=self.run_id,
                index=self._next_step_index(),
                phase="exploration",
                active_persona=persona.id if persona else None,
                surface=surface,
                source_video_id=self._last_video_id,
                candidates=candidates,
                choice=resolved.choice,
                usage=resolved.usage,
                chosen_video_id=video_id,
            )

            skipped = self.rng.random() < habits.skip_probability
            watch_fraction = 0.0 if skipped else self.rng.uniform(habits.watch_fraction.min, habits.watch_fraction.max)
            dwell_seconds = 0.0 if skipped else self.rng.uniform(habits.dwell_seconds.min, habits.dwell_seconds.max)
            await self.store.record_event(
                self.run_id,
                "decision",
                payload={
                    "phase_index": phase_index,
                    "surface": surface,
                    "persona": persona.id if persona else None,
                    "rank": resolved.choice.rank,
                    "video_id": video_id,
                    "skipped": skipped,
                    "fallback": resolved.usage.fallback_reason,
                },
            )

            await self.rate_gate.before_request()
            result = await self.driver.watch(video_id, seconds=dwell_seconds, fraction=watch_fraction)
            await self.store.record_watch_result(
                step_id,
                watched_seconds=result.watched_seconds,
                watch_fraction=result.watch_fraction,
            )
            self._last_video_id = video_id
            self._history.append(resolved.candidate.title or video_id)
            self._session_remaining -= 1

            delay = None
            if phase_index + 1 < self.experiment.steps:
                delay = self.rng.uniform(
                    self.experiment.pacing.step_delay_seconds.min,
                    self.experiment.pacing.step_delay_seconds.max,
                )
            if delay is not None and self.enforce_pacing:
                self.rate_gate.not_before = max(self.rate_gate.not_before, time.time() + delay)
            await self._save_checkpoint(
                phase="complete" if phase_index + 1 == self.experiment.steps else "exploration",
                next_context_index=len(self.context.videos),
                next_exploration_index=phase_index + 1,
            )
            if delay is not None and self.enforce_pacing:
                await self.sleep(delay)
        await self.store.record_event(self.run_id, "exploration_completed")

    async def _save_checkpoint(
        self,
        *,
        phase: Literal["context", "exploration", "complete"],
        next_context_index: int,
        next_exploration_index: int,
        in_flight: str | None = None,
    ) -> None:
        checkpoint = RunCheckpoint(
            phase=phase,
            next_context_index=next_context_index,
            next_exploration_index=next_exploration_index,
            global_step_index=self._global_step_index,
            last_video_id=self._last_video_id,
            history=list(self._history),
            rng_state=_state_to_json(self.rng.getstate()),
            session_remaining=self._session_remaining,
            session_index=self._session_index,
            not_before=self.rate_gate.not_before,
            in_flight=in_flight,
        )
        self.checkpoint = checkpoint
        await self.store.save_checkpoint(self.run_id, checkpoint.model_dump(mode="json"))

    def _next_step_index(self) -> int:
        value = self._global_step_index
        self._global_step_index += 1
        return value

    def _surface_for_step(self, phase_index: int, habits: ViewingHabits) -> Surface:
        surfaces = self.experiment.surfaces
        if "search" in surfaces and self.rng.random() < habits.search_probability:
            return "search"
        if "shorts" in surfaces and self.rng.random() < habits.shorts_probability:
            return "shorts"
        regular = [surface for surface in surfaces if surface not in {"search", "shorts"}]
        if not regular:
            return surfaces[phase_index % len(surfaces)]
        return regular[phase_index % len(regular)]

    async def _collect(self, surface: Surface, persona: Persona | None) -> list[Candidate]:
        if surface == "search":
            if persona is None:
                raise ValueError("random mode cannot use search without an active persona/query policy")
            queries = [query for interest in persona.interests for query in interest.seed_queries]
            if not queries:
                raise ValueError(f"persona {persona.id!r} has no seed_queries for search surface")
            query = self.rng.choice(queries)
            await self.store.record_event(self.run_id, "search", payload={"query": query, "persona": persona.id})
            return await self.driver.search(query)
        source = self._last_video_id if surface == "watch_next" else None
        return await self.driver.collect(surface, source_video_id=source)

    async def _choose(self, persona: Persona | None, candidates: list[Candidate]) -> ResolvedChoice:
        selectable = [candidate for candidate in candidates if candidate.video_id]
        if not selectable:
            raise NoSelectableCandidate("recommendation set contains no selectable video IDs")

        if persona is None:
            selected = self.rng.choice(selectable)
            choice = Choice(
                rank=selected.rank,
                justification="Uniform random baseline over selectable rendered candidates.",
            )
            return resolve_choice(choice, Usage(model="random-baseline"), candidates)

        choice, usage = await self.llm.choose(
            system_prompt=render_persona_prompt(persona),
            history=list(self._history),
            candidates=candidates,
        )
        return resolve_choice(choice, usage, candidates)
