"""Resolve, plan, and execute P2 experiments serially.

P3 adds bounded multi-account concurrency. P2 deliberately runs repetitions one at a
time so the core protocol can be exercised end-to-end without introducing scheduling
as another source of failure.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from yafg import __version__
from yafg.agent.loop import AgentRun, PersonaPolicy
from yafg.agent.prompt import PROMPT_TEMPLATE, PROMPT_VERSION
from yafg.browser.driver import YouTubeDriver
from yafg.experiment.manifest import build_manifest, manifest_hash
from yafg.experiment.schema import Context, Experiment
from yafg.identity.registry import AccountRegistry
from yafg.identity.schema import Account
from yafg.llm.base import Candidate, Choice, LLMProvider, Usage
from yafg.llm.providers import make_provider
from yafg.personas.schema import Persona
from yafg.settings import settings
from yafg.store.evidence import SQLAlchemyEvidenceStore
from yafg.store.models import Experiment as ExperimentRecord
from yafg.store.models import Run


@dataclass(frozen=True, slots=True)
class ResolvedExperiment:
    experiment: Experiment
    context: Context
    personas: dict[str, Persona]
    manifest: dict[str, Any]
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class PlannedRun:
    repetition: int
    account_label: str
    arm: str
    behavior_seed: int


class _UnusedProvider:
    async def choose(
        self, *, system_prompt: str, history: list[str], candidates: list[Candidate]
    ) -> tuple[Choice, Usage]:
        raise AssertionError("random baseline must never call an LLM provider")


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


def plan_runs(resolved: ResolvedExperiment) -> list[PlannedRun]:
    experiment = resolved.experiment
    supported_surfaces = {"home", "watch_next", "search"}
    unsupported = sorted(set(experiment.surfaces) - supported_surfaces)
    if unsupported:
        raise ValueError(
            f"P2 supports only home, watch_next, and search surfaces; unsupported: {', '.join(unsupported)}"
        )
    if experiment.concurrency != 1:
        raise ValueError("P2 executes serially and requires concurrency=1; bounded concurrency arrives in P3")
    if len(experiment.accounts) < experiment.repetitions:
        raise ValueError(
            f"{experiment.repetitions} repetitions require {experiment.repetitions} distinct accounts; "
            f"only {len(experiment.accounts)} configured"
        )
    arm = experiment.personas[0].persona if experiment.mode == "single" else experiment.mode
    return [
        PlannedRun(
            repetition=repetition,
            account_label=experiment.accounts[repetition],
            arm=arm,
            behavior_seed=experiment.behavior_seed + repetition,
        )
        for repetition in range(experiment.repetitions)
    ]


async def validate_accounts(engine: AsyncEngine, resolved: ResolvedExperiment, plan: list[PlannedRun]) -> list[Account]:
    registry = AccountRegistry(engine)
    accounts: list[Account] = []
    single_persona = resolved.experiment.personas[0].persona if resolved.experiment.mode == "single" else None
    for item in plan:
        account = await registry.get(item.account_label)
        if account is None:
            raise ValueError(f"account {item.account_label!r} is not provisioned")
        if account.status != "active":
            raise ValueError(f"account {item.account_label!r} is {account.status!r}, expected 'active'")
        if single_persona and account.persona_id and account.persona_id != single_persona:
            raise ValueError(
                f"account {account.label!r} is bound to persona {account.persona_id!r}, not {single_persona!r}"
            )
        accounts.append(account)
    return accounts


def _persona_snapshot(resolved: ResolvedExperiment) -> dict[str, Any]:
    return {
        "mode": resolved.experiment.mode,
        "personas": [
            {
                "weight": ref.weight,
                "persona": resolved.personas[ref.persona].model_dump(mode="json"),
            }
            for ref in resolved.experiment.personas
        ],
    }


async def _create_experiment_record(engine: AsyncEngine, resolved: ResolvedExperiment) -> str:
    record = ExperimentRecord(
        slug=resolved.experiment.id,
        description=resolved.experiment.description,
        mode=resolved.experiment.mode,
        manifest=resolved.manifest,
        manifest_hash=resolved.manifest_hash,
    )
    async with AsyncSession(engine) as session:
        session.add(record)
        await session.commit()
    return record.id


async def _create_run_record(
    engine: AsyncEngine,
    *,
    experiment_id: str,
    resolved: ResolvedExperiment,
    planned: PlannedRun,
) -> str:
    run = Run(
        id=str(uuid.uuid4()),
        experiment_id=experiment_id,
        arm=planned.arm,
        repetition=planned.repetition,
        account_label=planned.account_label,
        persona_snapshot=_persona_snapshot(resolved),
        context_snapshot=resolved.context.model_dump(mode="json"),
        behavior_seed=planned.behavior_seed,
        status="pending",
    )
    async with AsyncSession(engine) as session:
        session.add(run)
        await session.commit()
    return run.id


async def execute_experiment(
    engine: AsyncEngine,
    resolved: ResolvedExperiment,
    *,
    headless: bool | None = None,
) -> list[str]:
    """Execute all P2 repetitions serially and return their run IDs."""
    plan = plan_runs(resolved)
    accounts = await validate_accounts(engine, resolved, plan)
    provider: LLMProvider = (
        _UnusedProvider() if resolved.experiment.mode == "random" else make_provider(resolved.experiment.llm)
    )
    experiment_record_id = await _create_experiment_record(engine, resolved)
    evidence = SQLAlchemyEvidenceStore(engine)
    run_ids: list[str] = []

    try:
        for planned, account in zip(plan, accounts, strict=True):
            run_experiment = resolved.experiment.model_copy(update={"behavior_seed": planned.behavior_seed})
            run_id = await _create_run_record(
                engine,
                experiment_id=experiment_record_id,
                resolved=resolved,
                planned=planned,
            )
            run_ids.append(run_id)
            policy = PersonaPolicy(run_experiment, resolved.personas)
            driver = YouTubeDriver(
                account,
                headless=settings.headless if headless is None else headless,
                interactions=run_experiment.interactions,
            )
            try:
                async with driver:
                    agent = AgentRun(
                        experiment=run_experiment,
                        context=resolved.context,
                        policy=policy,
                        driver=driver,
                        llm=provider,
                        store=evidence,
                        run_id=run_id,
                    )
                    await agent.execute()
            except Exception as exc:
                # Browser startup can fail before AgentRun gets a chance to mark the run.
                # Re-marking an already-failed AgentRun is harmless and keeps lifecycle
                # state correct for both failure locations.
                await evidence.record_event(
                    run_id,
                    "runner_failed",
                    payload={"error_type": type(exc).__name__, "message": str(exc)},
                    level="error",
                )
                await evidence.fail_run(run_id, f"{type(exc).__name__}: {exc}")
                raise
        return run_ids
    finally:
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()
