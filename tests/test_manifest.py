from __future__ import annotations

import yaml

from yafg.experiment.manifest import build_manifest, manifest_hash
from yafg.experiment.schema import Context, Experiment
from yafg.personas.schema import Persona


def test_manifest_hash_is_canonical_and_includes_behavior_seed() -> None:
    persona = Persona.model_validate(
        {
            "id": "neutral",
            "display_name": "Neutral",
            "demographics": {"age": 30, "nationality": "US", "languages": ["en-US"]},
            "interests": [{"topic": "General"}],
            "bio": "A neutral synthetic viewer used solely to test canonical manifest hashing.",
        }
    )
    context = Context(id="warm", description="warmup", videos=["dQw4w9WgXcQ"])
    base = {
        "id": "exp",
        "description": "hash test",
        "mode": "single",
        "personas": [{"persona": "neutral"}],
        "context": "warm",
        "steps": 1,
        "behavior_seed": 7,
    }
    experiment = Experiment.model_validate(base)
    manifest = build_manifest(experiment=experiment, context=context, personas={"neutral": persona})
    assert manifest_hash(manifest) == manifest_hash(yaml.safe_load(yaml.safe_dump(manifest)))

    changed = Experiment.model_validate({**base, "behavior_seed": 8})
    changed_manifest = build_manifest(experiment=changed, context=context, personas={"neutral": persona})
    assert manifest_hash(manifest) != manifest_hash(changed_manifest)
