"""Every shipped YAML must satisfy its schema.

This is the only test that matters at P0: the configs are the public contract, and a
schema change that silently invalidates them should fail CI, not a run at step 0.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yafg.experiment.schema import Context, Experiment, Interactions
from yafg.personas.schema import Persona

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("personas/*.yaml")), ids=lambda p: p.stem)
def test_persona_yaml_validates(path: Path) -> None:
    persona = Persona.model_validate(_load(path))
    assert persona.id == path.stem
    assert abs(sum(w for _, w in persona.normalised_interests) - 1.0) < 1e-9


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("contexts/*.yaml")), ids=lambda p: p.stem)
def test_context_yaml_validates(path: Path) -> None:
    Context.model_validate(_load(path))


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("experiments/*.yaml")), ids=lambda p: p.stem)
def test_experiment_yaml_validates(path: Path) -> None:
    experiment = Experiment.model_validate(_load(path))
    persona_ids = {Persona.model_validate(_load(p)).id for p in CONFIGS.glob("personas/*.yaml")}
    for ref in experiment.personas:
        assert ref.persona in persona_ids, f"unknown persona {ref.persona!r}"
    context_ids = {Context.model_validate(_load(p)).id for p in CONFIGS.glob("contexts/*.yaml")}
    assert experiment.context in context_ids


def test_commenting_cannot_be_enabled() -> None:
    """The ethics boundary is a type, not a convention. See docs/ETHICS.md."""
    with pytest.raises(ValueError):
        Interactions.model_validate({"comment": True})


def test_accounts_are_not_shared_between_concurrent_arms() -> None:
    with pytest.raises(ValueError, match="planned runs"):
        Experiment.model_validate(
            {
                "id": "bad",
                "description": "two arms, one account",
                "mode": "single",
                "personas": [{"persona": "neutral-baseline"}],
                "context": "warmup-neutral",
                "steps": 5,
                "repetitions": 2,
                "accounts": ["only-one"],
                "concurrency": 2,
            }
        )


def test_single_mode_rejects_duplicate_persona_arms() -> None:
    payload = _load(CONFIGS / "experiments" / "demo-single-persona.yaml")
    payload["personas"] = [payload["personas"][0], payload["personas"][0]]
    payload["accounts"] = ["amina-01", "amina-02"]
    with pytest.raises(ValueError, match="persona arms must be unique"):
        Experiment.model_validate(payload)
