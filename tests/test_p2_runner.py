from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yafg.agent.prompt import PROMPT_TEMPLATE, PROMPT_VERSION
from yafg.experiment.runner import plan_runs, resolve_experiment

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_demo_resolves_prompt_into_manifest() -> None:
    resolved = resolve_experiment(CONFIGS / "experiments" / "demo-single-persona.yaml")
    assert resolved.manifest["prompt"] == {"version": PROMPT_VERSION, "text": PROMPT_TEMPLATE}
    assert len(resolved.manifest_hash) == 64
    plan = plan_runs(resolved)
    assert len(plan) == 1
    assert plan[0].account_label == "amina-01"


def test_repetition_plan_uses_distinct_accounts_and_stable_seeds(tmp_path: Path) -> None:
    root = tmp_path / "configs"
    for dirname in ("experiments", "contexts", "personas"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    neutral = (CONFIGS / "personas" / "neutral-baseline.yaml").read_text()
    (root / "personas" / "alpha.yaml").write_text(neutral.replace("neutral-baseline", "alpha"))
    (root / "contexts" / "warmup-neutral.yaml").write_text((CONFIGS / "contexts" / "warmup-neutral.yaml").read_text())
    payload = yaml.safe_load((CONFIGS / "experiments" / "demo-single-persona.yaml").read_text())
    payload.update(
        {
            "id": "repeat-test",
            "personas": [{"persona": "alpha"}],
            "context": "warmup-neutral",
            "repetitions": 2,
            "accounts": ["account-01", "account-02"],
            "behavior_seed": 50,
        }
    )
    path = root / "experiments" / "repeat-test.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    plan = plan_runs(resolve_experiment(path))
    assert [item.account_label for item in plan] == ["account-01", "account-02"]
    assert [item.behavior_seed for item in plan] == [50, 51]


def test_p2_rejects_unimplemented_surface() -> None:
    resolved = resolve_experiment(CONFIGS / "experiments" / "demo-single-persona.yaml")
    changed = resolved.__class__(
        experiment=resolved.experiment.model_copy(update={"surfaces": ["home", "shorts"]}),
        context=resolved.context,
        personas=resolved.personas,
        manifest=resolved.manifest,
        manifest_hash=resolved.manifest_hash,
    )
    with pytest.raises(ValueError, match="unsupported: shorts"):
        plan_runs(changed)


def test_p3_planner_allows_bounded_concurrency() -> None:
    resolved = resolve_experiment(CONFIGS / "experiments" / "demo-single-persona.yaml")
    changed = resolved.__class__(
        experiment=resolved.experiment.model_copy(update={"concurrency": 2}),
        context=resolved.context,
        personas=resolved.personas,
        manifest=resolved.manifest,
        manifest_hash=resolved.manifest_hash,
    )
    plan = plan_runs(changed)
    assert len(plan) == 1
