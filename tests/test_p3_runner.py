from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from yafg.browser.driver import ChallengeDetected, WatchResult, YouTubeDriver
from yafg.experiment.runner import (
    ExperimentExecutionError,
    ResolvedExperiment,
    _require_parallel_backend,
    bounded_map,
    estimate_llm_cost,
    execute_experiment,
    plan_runs,
    resolve_experiment,
)
from yafg.identity.registry import AccountRegistry
from yafg.identity.schema import Account
from yafg.llm.base import Candidate
from yafg.personas.schema import Persona
from yafg.store.database import ensure_schema, make_engine
from yafg.store.models import Run, Step

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_single_mode_expands_persona_x_repetition_matrix() -> None:
    resolved = resolve_experiment(CONFIGS / "experiments" / "demo-single-persona.yaml")
    alpha = resolved.personas["amina-22-dz-student"]
    beta = Persona.model_validate(
        alpha.model_dump(mode="python")
        | {
            "id": "beta",
            "display_name": "Beta",
            "bio": "I am a synthetic Beta viewer used only to verify the P3 arm-matrix planner behavior.",
        }
    )
    payload = resolved.experiment.model_dump(mode="python")
    payload.update(
        {
            "personas": [
                {"persona": "amina-22-dz-student", "weight": 1.0},
                {"persona": "beta", "weight": 1.0},
            ],
            "repetitions": 2,
            "accounts": ["a-01", "a-02", "b-01", "b-02"],
            "concurrency": 3,
            "behavior_seed": 70,
        }
    )
    experiment = type(resolved.experiment).model_validate(payload)
    changed = ResolvedExperiment(
        experiment=experiment,
        context=resolved.context,
        personas={alpha.id: alpha, beta.id: beta},
        manifest=resolved.manifest,
        manifest_hash=resolved.manifest_hash,
    )

    plan = plan_runs(changed)
    assert [(item.arm, item.repetition) for item in plan] == [
        ("amina-22-dz-student", 0),
        ("amina-22-dz-student", 1),
        ("beta", 0),
        ("beta", 1),
    ]
    assert [item.behavior_seed for item in plan] == [70, 71, 70, 71]
    assert [item.account_label for item in plan] == ["a-01", "a-02", "b-01", "b-02"]


@pytest.mark.asyncio
async def test_bounded_map_never_exceeds_limit_and_preserves_order() -> None:
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def worker(value: int) -> int:
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return value * 2

    result = await bounded_map(list(range(8)), 3, worker)
    assert result == [value * 2 for value in range(8)]
    assert peak == 3


def test_parallel_execution_requires_postgres() -> None:
    engine = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
    with pytest.raises(ValueError, match="PostgreSQL"):
        _require_parallel_backend(cast(AsyncEngine, engine), 2)
    _require_parallel_backend(cast(AsyncEngine, engine), 1)


def test_random_cost_estimate_has_no_model_calls() -> None:
    resolved = resolve_experiment(CONFIGS / "experiments" / "demo-single-persona.yaml")
    experiment = resolved.experiment.model_copy(update={"mode": "random"})
    changed = ResolvedExperiment(
        experiment=experiment,
        context=resolved.context,
        personas=resolved.personas,
        manifest=resolved.manifest,
        manifest_hash=resolved.manifest_hash,
    )
    estimate = estimate_llm_cost(changed, plan_runs(changed))
    assert estimate.llm_calls == 0
    assert estimate.estimated_usd == 0.0


@pytest.mark.asyncio
async def test_execute_experiment_completes_and_releases_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeDriver:
        async def __aenter__(self) -> FakeDriver:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def assert_signed_in(self) -> str:
            return "test"

        async def collect(self, surface: str, *, source_video_id: str | None = None) -> list[Candidate]:
            return [Candidate(rank=0, video_id="aaaaaaaaaaa", title="A")]

        async def watch(self, video_id: str, *, seconds: float, fraction: float) -> WatchResult:
            return WatchResult(
                video_id=video_id,
                watched_seconds=0,
                watch_fraction=fraction,
                completed=True,
            )

    now = 0.0

    def monotonic() -> float:
        nonlocal now
        now += 100
        return now

    monkeypatch.setattr("yafg.agent.loop.time.time", monotonic)
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'runner.db'}")
    await ensure_schema(engine)
    registry = AccountRegistry(engine)
    await registry.save(Account(label="test-01", status="active", profile_dir=tmp_path / "profile"))
    resolved = resolve_experiment(CONFIGS / "experiments" / "demo-single-persona.yaml")
    experiment = resolved.experiment.model_copy(update={"mode": "random", "steps": 1, "accounts": ["test-01"]})
    changed = ResolvedExperiment(
        experiment=experiment,
        context=resolved.context,
        personas=resolved.personas,
        manifest=resolved.manifest,
        manifest_hash=resolved.manifest_hash,
    )

    run_ids = await execute_experiment(
        engine,
        changed,
        driver_factory=lambda *_: cast(YouTubeDriver, FakeDriver()),
    )

    async with AsyncSession(engine) as session:
        run = await session.get(Run, run_ids[0])
        steps = list((await session.scalars(select(Step).where(Step.run_id == run_ids[0]))).all())
    assert run is not None
    assert run.status == "complete"
    assert run.checkpoint["phase"] == "complete"
    assert len(steps) == len(changed.context.videos) + changed.experiment.steps
    assert await registry.lease_owner("test-01") is None
    await engine.dispose()


@pytest.mark.asyncio
async def test_challenge_marks_account_and_blocks_automatic_resume(tmp_path: Path) -> None:
    class ChallengeDriver:
        async def __aenter__(self) -> ChallengeDriver:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def assert_signed_in(self) -> str:
            raise ChallengeDetected("challenge")

    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'challenge.db'}")
    await ensure_schema(engine)
    registry = AccountRegistry(engine)
    await registry.save(Account(label="test-01", status="active", profile_dir=tmp_path / "profile"))
    resolved = resolve_experiment(CONFIGS / "experiments" / "demo-single-persona.yaml")
    experiment = resolved.experiment.model_copy(update={"mode": "random", "steps": 1, "accounts": ["test-01"]})
    changed = ResolvedExperiment(
        experiment=experiment,
        context=resolved.context,
        personas=resolved.personas,
        manifest=resolved.manifest,
        manifest_hash=resolved.manifest_hash,
    )

    with pytest.raises(ExperimentExecutionError) as failure:
        await execute_experiment(
            engine,
            changed,
            driver_factory=lambda *_: cast(YouTubeDriver, ChallengeDriver()),
        )

    assert any(isinstance(exc, ChallengeDetected) for exc in failure.value.failures.values())
    account = await registry.get("test-01")
    assert account is not None and account.status == "challenged"
    assert await registry.lease_owner("test-01") is None
    with pytest.raises(ValueError, match="challenged"):
        await execute_experiment(
            engine,
            changed,
            resume=True,
            driver_factory=lambda *_: cast(YouTubeDriver, ChallengeDriver()),
        )
    await engine.dispose()
