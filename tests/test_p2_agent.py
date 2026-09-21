from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from yafg.agent.choice import NoSelectableCandidate
from yafg.agent.loop import AgentRun, PersonaPolicy
from yafg.agent.prompt import PROMPT_TEMPLATE, PROMPT_VERSION, render_persona_prompt
from yafg.browser.driver import WatchResult, YouTubeDriver
from yafg.experiment.schema import Context, Experiment
from yafg.llm.base import Candidate, Choice, LLMProvider, Usage
from yafg.personas.schema import Persona


def _persona(persona_id: str, topic: str = "Programming") -> Persona:
    return Persona.model_validate(
        {
            "id": persona_id,
            "display_name": persona_id,
            "demographics": {"age": 25, "nationality": "US", "languages": ["en-US"]},
            "interests": [{"topic": topic, "seed_queries": [f"{topic} tutorial"]}],
            "habits": {
                "videos_per_session": {"min": 2, "max": 2},
                "watch_fraction": {"min": 0.5, "max": 0.5},
                "dwell_seconds": {"min": 0, "max": 0},
                "skip_probability": 0,
                "search_probability": 0,
            },
            "bio": f"I am a synthetic viewer who consistently prefers {topic} videos for this research test.",
            "priors": {"hidden-analysis-label": 0.7},
            "notes": "researcher-only note",
        }
    )


def _experiment(mode: str = "single", **updates: Any) -> Experiment:
    data: dict[str, Any] = {
        "id": "p2-test",
        "description": "P2 protocol test",
        "mode": mode,
        "personas": [{"persona": "alpha"}],
        "context": "warm",
        "steps": 2,
        "surfaces": ["home", "watch_next"],
        "behavior_seed": 123,
    }
    data.update(updates)
    return Experiment.model_validate(data)


@dataclass
class MemoryStore:
    started: bool = False
    finished: bool = False
    failure: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    events: list[tuple[str, dict[str, Any], str]] = field(default_factory=list)
    watch_results: list[tuple[str, float, float]] = field(default_factory=list)

    async def start_run(self, run_id: str) -> None:
        self.started = True

    async def record_event(
        self, run_id: str, kind: str, *, payload: Mapping[str, Any] | None = None, level: str = "info"
    ) -> None:
        self.events.append((kind, dict(payload or {}), level))

    async def record_step(self, **kwargs: Any) -> str:
        self.steps.append(kwargs)
        return f"step-{len(self.steps)}"

    async def record_watch_result(self, step_id: str, *, watched_seconds: float, watch_fraction: float) -> None:
        self.watch_results.append((step_id, watched_seconds, watch_fraction))

    async def finish_run(self, run_id: str) -> None:
        self.finished = True

    async def fail_run(self, run_id: str, failure: str) -> None:
        self.failure = failure


class FakeDriver:
    def __init__(self, candidates: list[Candidate] | None = None) -> None:
        self.candidates = candidates or [
            Candidate(rank=0),
            Candidate(rank=1, video_id="aaaaaaaaaaa", title="One"),
            Candidate(rank=2, video_id="bbbbbbbbbbb", title="Two", duration_label="4:20"),
        ]
        self.collections: list[tuple[str, str | None]] = []
        self.watches: list[tuple[str, float, float]] = []

    async def assert_signed_in(self) -> str:
        return "test"

    async def collect(self, surface: str, *, source_video_id: str | None = None) -> list[Candidate]:
        self.collections.append((surface, source_video_id))
        return list(self.candidates)

    async def search(self, query: str) -> list[Candidate]:
        self.collections.append((f"search:{query}", None))
        return list(self.candidates)

    async def watch(self, video_id: str, *, seconds: float, fraction: float) -> WatchResult:
        self.watches.append((video_id, seconds, fraction))
        return WatchResult(video_id=video_id, watched_seconds=seconds, watch_fraction=fraction, completed=False)


class FakeLLM:
    def __init__(self, rank: int = 2) -> None:
        self.rank = rank
        self.calls = 0
        self.prompts: list[str] = []

    async def choose(self, *, system_prompt: str, history: list[str], candidates: list[Candidate]):
        self.calls += 1
        self.prompts.append(system_prompt)
        return Choice(rank=self.rank, justification="Matches the persona interests."), Usage(model="fake")


@pytest.mark.asyncio
async def test_agent_executes_context_and_exploration_with_rank_fidelity() -> None:
    persona = _persona("alpha")
    experiment = _experiment()
    context = Context(id="warm", description="warmup", videos=["dQw4w9WgXcQ"], max_watch_seconds=0)
    store = MemoryStore()
    driver = FakeDriver()
    llm = FakeLLM(rank=2)
    agent = AgentRun(
        experiment=experiment,
        context=context,
        policy=PersonaPolicy(experiment, {"alpha": persona}),
        driver=cast(YouTubeDriver, driver),
        llm=llm,
        store=store,
        run_id="run-1",
        enforce_pacing=False,
    )

    await agent.execute()

    assert store.started and store.finished and store.failure is None
    assert [step["phase"] for step in store.steps] == ["context", "exploration", "exploration"]
    assert [step["index"] for step in store.steps] == [0, 1, 2]
    assert store.steps[1]["candidates"][0].video_id is None
    assert store.steps[1]["choice"].rank == 2
    assert store.steps[1]["chosen_video_id"] == "bbbbbbbbbbb"
    assert driver.collections == [("home", None), ("watch_next", "bbbbbbbbbbb")]
    assert llm.calls == 2
    assert all(PROMPT_VERSION.startswith("persona-choice") for _ in llm.prompts)


@pytest.mark.asyncio
async def test_unselectable_candidate_set_is_persisted_before_failure() -> None:
    persona = _persona("alpha")
    experiment = _experiment(steps=1)
    context = Context(id="warm", description="warmup", videos=["dQw4w9WgXcQ"], max_watch_seconds=0)
    store = MemoryStore()
    driver = FakeDriver([Candidate(rank=0), Candidate(rank=1, title="unparseable")])
    agent = AgentRun(
        experiment=experiment,
        context=context,
        policy=PersonaPolicy(experiment, {"alpha": persona}),
        driver=cast(YouTubeDriver, driver),
        llm=FakeLLM(),
        store=store,
        run_id="run-2",
        enforce_pacing=False,
    )

    with pytest.raises(NoSelectableCandidate):
        await agent.execute()

    assert len(store.steps) == 2
    failed_step = store.steps[1]
    assert [candidate.rank for candidate in failed_step["candidates"]] == [0, 1]
    assert failed_step["chosen_video_id"] is None
    assert failed_step["choice"] is None
    assert store.failure and "NoSelectableCandidate" in store.failure


@pytest.mark.asyncio
async def test_random_baseline_never_calls_llm() -> None:
    persona = _persona("alpha")
    experiment = _experiment("random", steps=1)
    context = Context(id="warm", description="warmup", videos=["dQw4w9WgXcQ"], max_watch_seconds=0)
    store = MemoryStore()
    driver = FakeDriver()

    class NeverLLM:
        async def choose(self, **kwargs: Any):
            raise AssertionError("random baseline called the LLM")

    agent = AgentRun(
        experiment=experiment,
        context=context,
        policy=PersonaPolicy(experiment, {"alpha": persona}),
        driver=cast(YouTubeDriver, driver),
        llm=cast(LLMProvider, NeverLLM()),
        store=store,
        run_id="run-random",
        enforce_pacing=False,
    )
    await agent.execute()

    step = store.steps[1]
    assert step["active_persona"] is None
    assert step["usage"].model == "random-baseline"
    assert step["choice"].justification.startswith("Uniform random baseline")


def test_persona_policy_modes_are_deterministic() -> None:
    alpha = _persona("alpha", "Programming")
    beta = _persona("beta", "Cooking")
    personas = {"alpha": alpha, "beta": beta}

    sequential = _experiment(
        "sequential",
        personas=[{"persona": "alpha"}, {"persona": "beta"}],
        switch_every=2,
    )
    policy = PersonaPolicy(sequential, personas)
    assert [cast(Persona, policy.active_at(i)).id for i in range(6)] == [
        "alpha",
        "alpha",
        "beta",
        "beta",
        "alpha",
        "alpha",
    ]

    mixed = _experiment("mixed", personas=[{"persona": "alpha", "weight": 1}, {"persona": "beta", "weight": 2}])
    a = PersonaPolicy(mixed, personas)
    b = PersonaPolicy(mixed, personas)
    assert [cast(Persona, a.active_at(i)).id for i in range(20)] == [
        cast(Persona, b.active_at(i)).id for i in range(20)
    ]

    random_exp = _experiment("random")
    assert PersonaPolicy(random_exp, {"alpha": alpha}).active_at(0) is None


def test_prompt_is_versioned_and_excludes_analysis_only_fields() -> None:
    prompt = render_persona_prompt(_persona("alpha"))
    assert PROMPT_VERSION == "persona-choice-v1"
    assert "Programming" in prompt
    assert "hidden-analysis-label" not in prompt
    assert "researcher-only note" not in prompt
    assert "{persona_profile}" in PROMPT_TEMPLATE
