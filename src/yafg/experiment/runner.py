"""Resolve, plan, schedule, resume, and execute experiments."""

from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from yafg import __version__
from yafg.agent.loop import AgentRun, PersonaPolicy, RunCheckpoint
from yafg.agent.prompt import PROMPT_TEMPLATE, PROMPT_VERSION, render_persona_prompt
from yafg.browser.driver import ChallengeDetected, LoginRequired, YouTubeDriver
from yafg.experiment.manifest import build_manifest, manifest_hash
from yafg.experiment.schema import Context, Experiment, PersonaRef
from yafg.identity.registry import AccountInUse, AccountRegistry
from yafg.identity.schema import Account
from yafg.llm.base import Candidate, Choice, LLMProvider, Usage
from yafg.llm.providers import make_provider
from yafg.personas.schema import Persona
from yafg.settings import settings
from yafg.store.evidence import SQLAlchemyEvidenceStore
from yafg.store.models import Experiment as ExperimentRecord
from yafg.store.models import Run, Step


@dataclass(frozen=True, slots=True)
class ResolvedExperiment:
    experiment: Experiment
    context: Context
    personas: dict[str, Persona]
    manifest: dict[str, Any]
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class PlannedRun:
    index: int
    repetition: int
    account_label: str
    arm: str
    behavior_seed: int


@dataclass(frozen=True, slots=True)
class CostEstimate:
    llm_calls: int
    estimated_input_tokens: int
    max_output_tokens: int
    estimated_usd: float | None
    assumption: str


class RecoveryUnsafe(RuntimeError):
    """Raised when persisted evidence is newer than the deterministic checkpoint."""


class ExperimentExecutionError(RuntimeError):
    def __init__(self, failures: Mapping[str, BaseException]) -> None:
        self.failures = dict(failures)
        summary = "; ".join(f"{run_id}: {type(exc).__name__}: {exc}" for run_id, exc in self.failures.items())
        super().__init__(f"{len(self.failures)} run(s) failed: {summary}")


class _UnusedProvider:
    async def choose(
        self, *, system_prompt: str, history: list[str], candidates: list[Candidate]
    ) -> tuple[Choice, Usage]:
        raise AssertionError("random baseline must never call an LLM provider")


DriverFactory = Callable[[Account, Experiment, bool], YouTubeDriver]


def _default_driver_factory(account: Account, experiment: Experiment, headless: bool) -> YouTubeDriver:
    return YouTubeDriver(account, headless=headless, interactions=experiment.interactions)


def _yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return payload


def _config_root(experiment_path: Path) -> Path:
    """Find the config root while supporting paths outside the default tree."""
    resolved = experiment_path.resolve()
    if resolved.parent.name == "experiments":
        return resolved.parent.parent
    return settings.config_dir.resolve()


def resolve_experiment(experiment_path: Path) -> ResolvedExperiment:
    experiment = Experiment.model_validate(_yaml_mapping(experiment_path))
    if experiment_path.stem != experiment.id:
        raise ValueError(f"experiment id {experiment.id!r} must match filename {experiment_path.stem!r}")

    root = _config_root(experiment_path)
    context_path = root / "contexts" / f"{experiment.context}.yaml"
    if not context_path.exists():
        raise ValueError(f"context config not found: {context_path}")
    context = Context.model_validate(_yaml_mapping(context_path))
    if context.id != context_path.stem:
        raise ValueError(f"context id {context.id!r} must match filename {context_path.stem!r}")

    personas: dict[str, Persona] = {}
    for ref in experiment.personas:
        path = root / "personas" / f"{ref.persona}.yaml"
        if not path.exists():
            raise ValueError(f"persona config not found: {path}")
        persona = Persona.model_validate(_yaml_mapping(path))
        if persona.id != path.stem:
            raise ValueError(f"persona id {persona.id!r} must match filename {path.stem!r}")
        personas[persona.id] = persona

    manifest = build_manifest(
        experiment=experiment,
        context=context,
        personas=personas,
        prompt_version=PROMPT_VERSION,
        prompt_text=PROMPT_TEMPLATE,
        implementation_version=__version__,
    )
    return ResolvedExperiment(
        experiment=experiment,
        context=context,
        personas=personas,
        manifest=manifest,
        manifest_hash=manifest_hash(manifest),
    )


def _single_refs(experiment: Experiment) -> list[PersonaRef]:
    return list(experiment.personas) if experiment.mode == "single" else []


def plan_runs(resolved: ResolvedExperiment) -> list[PlannedRun]:
    """Expand the experiment into its arm x repetition matrix.

    ``single`` means one persona per run, so every configured persona becomes its own
    arm. Mixed/sequential/random are policies inside a run and therefore contribute
    one arm each.
    """
    experiment = resolved.experiment
    supported_surfaces = {"home", "watch_next", "search"}
    unsupported = sorted(set(experiment.surfaces) - supported_surfaces)
    if unsupported:
        raise ValueError(
            f"P3 supports only home, watch_next, and search surfaces; unsupported: {', '.join(unsupported)}"
        )

    arm_names = [ref.persona for ref in _single_refs(experiment)] if experiment.mode == "single" else [experiment.mode]
    required = len(arm_names) * experiment.repetitions
    if len(experiment.accounts) < required:
        raise ValueError(
            f"{required} planned runs require {required} distinct accounts; only {len(experiment.accounts)} configured"
        )

    plan: list[PlannedRun] = []
    account_index = 0
    for arm in arm_names:
        for repetition in range(experiment.repetitions):
            plan.append(
                PlannedRun(
                    index=len(plan),
                    repetition=repetition,
                    account_label=experiment.accounts[account_index],
                    arm=arm,
                    behavior_seed=experiment.behavior_seed + repetition,
                )
            )
            account_index += 1
    return plan


async def validate_accounts(engine: AsyncEngine, resolved: ResolvedExperiment, plan: list[PlannedRun]) -> list[Account]:
    registry = AccountRegistry(engine)
    accounts: list[Account] = []
    profiles: dict[Path, str] = {}
    for item in plan:
        account = await registry.get(item.account_label)
        if account is None:
            raise ValueError(f"account {item.account_label!r} is not provisioned")
        if account.status != "active":
            raise ValueError(f"account {item.account_label!r} is {account.status!r}, expected 'active'")
        if resolved.experiment.mode == "single" and account.persona_id and account.persona_id != item.arm:
            raise ValueError(
                f"account {account.label!r} is bound to persona {account.persona_id!r}, not arm {item.arm!r}"
            )
        if resolved.experiment.mode != "single" and account.persona_id:
            raise ValueError(
                f"account {account.label!r} carries single-persona provenance {account.persona_id!r}; "
                f"use an unbound account for {resolved.experiment.mode!r} mode"
            )
        profile = account.profile_dir.resolve()
        if owner := profiles.get(profile):
            raise ValueError(f"accounts {owner!r} and {account.label!r} share profile {profile}")
        profiles[profile] = account.label
        accounts.append(account)
    return accounts


def experiment_for_plan(resolved: ResolvedExperiment, planned: PlannedRun) -> Experiment:
    experiment = resolved.experiment
    updates: dict[str, Any] = {"behavior_seed": planned.behavior_seed}
    if experiment.mode == "single":
        ref = next(ref for ref in experiment.personas if ref.persona == planned.arm)
        updates["personas"] = [ref]
    return experiment.model_copy(update=updates)


def _persona_snapshot(resolved: ResolvedExperiment, run_experiment: Experiment) -> dict[str, Any]:
    return {
        "mode": run_experiment.mode,
        "personas": [
            {
                "weight": ref.weight,
                "persona": resolved.personas[ref.persona].model_dump(mode="json"),
            }
            for ref in run_experiment.personas
        ],
    }


def estimate_llm_cost(resolved: ResolvedExperiment, plan: list[PlannedRun]) -> CostEstimate:
    """Return a transparent dry-run token/cost estimate.

    Candidate count is unknowable before the live page is rendered, so input tokens
    use a deliberately simple 20-candidate heuristic. Monetary pricing is read from
    settings rather than hard-coded because vendor prices are time-sensitive.
    """
    if resolved.experiment.mode == "random":
        return CostEstimate(0, 0, 0, 0.0, "random baseline makes no LLM calls")

    calls = len(plan) * resolved.experiment.steps
    prompt_chars = 0
    for item in plan:
        run_experiment = experiment_for_plan(resolved, item)
        personas = [resolved.personas[ref.persona] for ref in run_experiment.personas]
        mean_chars = sum(len(render_persona_prompt(persona)) for persona in personas) / len(personas)
        prompt_chars += int(mean_chars) * resolved.experiment.steps

    # Roughly 4 chars/token plus candidate/history/JSON framing. This is an estimate,
    # not a billing promise; actual usage is persisted from every provider response.
    prompt_tokens = math.ceil(prompt_chars / 4)
    candidate_and_history_tokens = calls * (20 * 45 + 350)
    estimated_input = prompt_tokens + candidate_and_history_tokens
    max_output = calls * resolved.experiment.llm.max_tokens
    usd = None
    if settings.llm_input_usd_per_million is not None and settings.llm_output_usd_per_million is not None:
        usd = (
            estimated_input * settings.llm_input_usd_per_million + max_output * settings.llm_output_usd_per_million
        ) / 1_000_000
    return CostEstimate(
        llm_calls=calls,
        estimated_input_tokens=estimated_input,
        max_output_tokens=max_output,
        estimated_usd=usd,
        assumption="~20 rendered candidates/call; output uses configured max_tokens as an upper bound",
    )


async def _create_experiment_record(engine: AsyncEngine, resolved: ResolvedExperiment) -> str:
    record = ExperimentRecord(
        slug=resolved.experiment.id,
        description=resolved.experiment.description,
        mode=resolved.experiment.mode,
        manifest=resolved.manifest,
        manifest_hash=resolved.manifest_hash,
    )
    async with AsyncSession(engine, expire_on_commit=False) as session:
        session.add(record)
        await session.commit()
        return record.id


async def _find_resume_experiment(engine: AsyncEngine, resolved: ResolvedExperiment) -> str:
    query = (
        select(ExperimentRecord)
        .where(
            ExperimentRecord.slug == resolved.experiment.id,
            ExperimentRecord.manifest_hash == resolved.manifest_hash,
        )
        .order_by(ExperimentRecord.created_at.desc())
    )
    async with AsyncSession(engine) as session:
        records = list((await session.scalars(query)).all())
        for record in records:
            statuses = list((await session.scalars(select(Run.status).where(Run.experiment_id == record.id))).all())
            if not statuses or any(status != "complete" for status in statuses):
                return record.id
    raise ValueError("no incomplete experiment with this manifest is available to resume")


async def _create_run_record(
    session: AsyncSession,
    *,
    experiment_id: str,
    resolved: ResolvedExperiment,
    planned: PlannedRun,
) -> Run:
    run_experiment = experiment_for_plan(resolved, planned)
    run = Run(
        id=str(uuid.uuid4()),
        experiment_id=experiment_id,
        arm=planned.arm,
        repetition=planned.repetition,
        account_label=planned.account_label,
        persona_snapshot=_persona_snapshot(resolved, run_experiment),
        context_snapshot=resolved.context.model_dump(mode="json"),
        behavior_seed=planned.behavior_seed,
        checkpoint={},
        status="pending",
    )
    session.add(run)
    return run


async def _prepare_run_records(
    engine: AsyncEngine,
    resolved: ResolvedExperiment,
    plan: list[PlannedRun],
    *,
    resume: bool,
) -> tuple[str, dict[int, Run]]:
    experiment_id = (
        await _find_resume_experiment(engine, resolved) if resume else await _create_experiment_record(engine, resolved)
    )
    async with AsyncSession(engine, expire_on_commit=False) as session:
        existing = list((await session.scalars(select(Run).where(Run.experiment_id == experiment_id))).all())
        by_key = {(run.arm, run.repetition): run for run in existing}
        records: dict[int, Run] = {}
        for planned in plan:
            key = (planned.arm, planned.repetition)
            run = by_key.get(key)
            if run is None:
                if resume and existing:
                    raise RecoveryUnsafe(
                        f"resume experiment is missing run arm={planned.arm!r} repetition={planned.repetition}"
                    )
                run = await _create_run_record(
                    session,
                    experiment_id=experiment_id,
                    resolved=resolved,
                    planned=planned,
                )
            if run.account_label != planned.account_label or run.behavior_seed != planned.behavior_seed:
                raise RecoveryUnsafe(
                    f"run {run.id} no longer matches its plan (account/seed changed); refuse to resume"
                )
            records[planned.index] = run
        await session.commit()
        return experiment_id, records


async def _assert_resume_safe(engine: AsyncEngine, run: Run) -> RunCheckpoint:
    raw = dict(run.checkpoint or {})
    async with AsyncSession(engine) as session:
        steps = list((await session.scalars(select(Step).where(Step.run_id == run.id).order_by(Step.index))).all())
    if not raw:
        if steps:
            raise RecoveryUnsafe(f"run {run.id} has evidence but no P3 checkpoint; it cannot be resumed automatically")
        return RunCheckpoint()

    checkpoint = RunCheckpoint.model_validate(raw)
    if checkpoint.in_flight:
        raise RecoveryUnsafe(f"run {run.id} was interrupted during {checkpoint.in_flight!r}; manual review required")
    if steps and steps[-1].index >= checkpoint.global_step_index:
        raise RecoveryUnsafe(
            f"run {run.id} contains uncheckpointed evidence at step {steps[-1].index}; manual review required"
        )
    for step in steps:
        if step.index >= checkpoint.global_step_index:
            continue
        if step.chosen_video_id is not None and step.watched_seconds is None:
            raise RecoveryUnsafe(f"run {run.id} step {step.index} chose a video but has no durable watch result")
    return checkpoint


async def bounded_map[T, R](items: list[T], limit: int, worker: Callable[[T], Awaitable[R]]) -> list[R | BaseException]:
    """Run ``worker`` with bounded concurrency while preserving input order."""
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _one(item: T) -> R | BaseException:
        async with semaphore:
            try:
                return await worker(item)
            except Exception as exc:  # ordinary arm failures do not cancel sibling runs
                return exc

    return await asyncio.gather(*(_one(item) for item in items))


def _require_parallel_backend(engine: AsyncEngine, concurrency: int) -> None:
    if concurrency > 1 and engine.dialect.name != "postgresql":
        raise ValueError(
            "parallel execution requires PostgreSQL (postgresql+asyncpg://...); "
            "SQLite remains supported for concurrency=1 and dry-run planning"
        )


async def execute_experiment(
    engine: AsyncEngine,
    resolved: ResolvedExperiment,
    *,
    headless: bool | None = None,
    resume: bool = False,
    driver_factory: DriverFactory = _default_driver_factory,
) -> list[str]:
    """Execute the full arm matrix with bounded concurrency.

    Failures are isolated per arm. Other scheduled runs continue, then an
    ``ExperimentExecutionError`` reports every failed run so a subsequent ``--resume``
    can continue safe checkpoints without replaying completed account history.
    """
    plan = plan_runs(resolved)
    _require_parallel_backend(engine, resolved.experiment.concurrency)
    if resume:
        _, records = await _prepare_run_records(engine, resolved, plan, resume=True)
        runnable = [item for item in plan if records[item.index].status != "complete"]
        await validate_accounts(engine, resolved, runnable)
    else:
        await validate_accounts(engine, resolved, plan)
        _, records = await _prepare_run_records(engine, resolved, plan, resume=False)
        runnable = plan
    provider: LLMProvider = (
        _UnusedProvider() if resolved.experiment.mode == "random" else make_provider(resolved.experiment.llm)
    )
    evidence = SQLAlchemyEvidenceStore(engine)
    registry = AccountRegistry(engine)
    effective_headless = settings.headless if headless is None else headless

    async def _run_one(planned: PlannedRun) -> str:
        record = records[planned.index]
        checkpoint = await _assert_resume_safe(engine, record) if resume else RunCheckpoint()
        run_experiment = experiment_for_plan(resolved, planned)
        policy = PersonaPolicy(run_experiment, resolved.personas)
        lease_id = str(uuid.uuid4())
        leased = False
        try:
            await registry.acquire_lease(planned.account_label, lease_id, required_status="active")
            leased = True
            account = await registry.get(planned.account_label)
            if account is None or account.status != "active":
                raise ValueError(f"account {planned.account_label!r} is no longer active")
            driver = driver_factory(account, run_experiment, effective_headless)
            await evidence.save_checkpoint(
                record.id,
                checkpoint.model_copy(update={"in_flight": "browser_open"}).model_dump(mode="json"),
            )
            async with driver:
                agent = AgentRun(
                    experiment=run_experiment,
                    context=resolved.context,
                    policy=policy,
                    driver=driver,
                    llm=provider,
                    store=evidence,
                    run_id=record.id,
                    checkpoint=checkpoint.model_dump(mode="json"),
                )
                await agent.execute()
            return record.id
        except AccountInUse:
            raise
        except Exception as exc:
            if isinstance(exc, ChallengeDetected):
                await registry.set_status(planned.account_label, "challenged")
            elif isinstance(exc, LoginRequired):
                await registry.set_status(planned.account_label, "logged_out")
            await evidence.record_event(
                record.id,
                "runner_failed",
                payload={"error_type": type(exc).__name__, "message": str(exc)},
                level="error",
            )
            await evidence.fail_run(record.id, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if leased:
                await registry.release_lease(planned.account_label, lease_id)

    try:
        results = await bounded_map(runnable, resolved.experiment.concurrency, _run_one)
        failures: dict[str, BaseException] = {}
        for planned, result in zip(runnable, results, strict=True):
            if isinstance(result, BaseException):
                failures[records[planned.index].id] = result
        if failures:
            raise ExperimentExecutionError(failures)
        return [records[item.index].id for item in plan]
    finally:
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()
