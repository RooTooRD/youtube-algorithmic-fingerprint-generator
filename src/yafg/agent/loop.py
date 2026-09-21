"""The two-phase experimental agent protocol.

Phase 1 watches an identical context list with no active persona. Phase 2 repeatedly
observes a ranked surface, chooses one rendered candidate, watches it according to
reproducible behavioral habits, and persists the complete evidence set.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from yafg.agent.choice import NoSelectableCandidate, ResolvedChoice, resolve_choice
from yafg.agent.prompt import render_persona_prompt
from yafg.browser.driver import YouTubeDriver
from yafg.experiment.schema import Context, Experiment, Surface
from yafg.llm.base import Candidate, Choice, LLMProvider, Usage
from yafg.personas.schema import Persona, ViewingHabits
from yafg.store.evidence import EvidenceStore

Sleep = Callable[[float], Awaitable[None]]


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
            if switch_every is None:  # schema validation should make this unreachable
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
    _last_request_at: float | None = field(default=None, init=False)

    async def before_request(self) -> None:
        if not self.enabled:
            return
        min_interval = 3600.0 / self.max_requests_per_hour
        now = time.monotonic()
        if self._last_request_at is not None:
            await self.sleep(max(0.0, min_interval - (now - self._last_request_at)))
        self._last_request_at = time.monotonic()


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
        self._history: list[str] = []
        self._last_video_id: str | None = None
        self._global_step_index = 0

    async def execute(self) -> None:
        """Run context + exploration, preserving all evidence written before failure."""
        await self.store.start_run(self.run_id)
        await self.store.record_event(self.run_id, "run_started", payload={"mode": self.experiment.mode})
        try:
            await self.driver.assert_signed_in()
            await self._run_context_phase()
            await self._run_exploration_phase()
        except Exception as exc:
            # Keeping the broad catch is deliberate: partial runs are evidence even
            # when a selector, provider, or persistence boundary fails mid-study.
            await self.store.record_event(
                self.run_id,
                "run_failed",
                payload={"error_type": type(exc).__name__, "message": str(exc)},
                level="error",
            )
            await self.store.fail_run(self.run_id, f"{type(exc).__name__}: {exc}")
            raise
        await self.store.record_event(self.run_id, "run_completed")
        await self.store.finish_run(self.run_id)

    async def _run_context_phase(self) -> None:
        await self.store.record_event(
            self.run_id,
            "context_started",
            payload={"context": self.context.id, "videos": len(self.context.videos)},
        )
        for phase_index, video_id in enumerate(self.context.videos):
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
        await self.store.record_event(
            self.run_id,
            "exploration_started",
            payload={"steps": self.experiment.steps},
        )
        session_remaining = 0
        session_index = -1
        for phase_index in range(self.experiment.steps):
            persona = self.policy.active_at(phase_index)
            habits = persona.habits if persona is not None else self.policy.baseline_habits
            if session_remaining <= 0:
                if phase_index > 0:
                    gap = self.rng.uniform(
                        self.experiment.pacing.session_gap_seconds.min,
                        self.experiment.pacing.session_gap_seconds.max,
                    )
                    await self.store.record_event(
                        self.run_id,
                        "session_gap",
                        payload={"seconds": gap, "after_session": session_index},
                    )
                    if self.enforce_pacing:
                        await self.sleep(gap)
                session_index += 1
                session_remaining = self.rng.randint(
                    int(habits.videos_per_session.min),
                    int(habits.videos_per_session.max),
                )
                await self.store.record_event(
                    self.run_id,
                    "session_started",
                    payload={
                        "session_index": session_index,
                        "planned_videos": min(session_remaining, self.experiment.steps - phase_index),
                        "persona_at_start": persona.id if persona else None,
                    },
                )
            surface = self._surface_for_step(phase_index, habits)

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
                await self.store.record_event(
                    self.run_id,
                    "decision_failed",
                    payload={"phase_index": phase_index, "error_type": type(exc).__name__, "message": str(exc)},
                    level="error",
                )
                raise

            # ``resolve_choice`` guarantees a video ID; the guard makes that invariant
            # explicit for type checkers and future Candidate changes.
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
            session_remaining -= 1

            if self.enforce_pacing and phase_index + 1 < self.experiment.steps:
                delay = self.rng.uniform(
                    self.experiment.pacing.step_delay_seconds.min,
                    self.experiment.pacing.step_delay_seconds.max,
                )
                await self.sleep(delay)
        await self.store.record_event(self.run_id, "exploration_completed")

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
            # A config can intentionally be search-only/shorts-only. Respect that
            # rather than silently changing the experimental surface.
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
