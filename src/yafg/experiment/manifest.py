"""Canonical experiment manifests and hashes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from yafg.experiment.schema import Context, Experiment
from yafg.personas.schema import Persona


def build_manifest(
    *,
    experiment: Experiment,
    context: Context,
    personas: Mapping[str, Persona],
    prompt_version: str | None = None,
    prompt_text: str | None = None,
    implementation_version: str | None = None,
) -> dict[str, Any]:
    """Return the resolved object whose canonical bytes define experimental identity.

    Referenced personas are included in declared order. Prompt fields are optional in
    P1 and become mandatory at the P2 runner boundary where prompting exists.
    """
    referenced: list[dict[str, Any]] = []
    for ref in experiment.personas:
        try:
            persona = personas[ref.persona]
        except KeyError as exc:
            raise ValueError(f"unknown persona {ref.persona!r}") from exc
        referenced.append({"weight": ref.weight, "persona": persona.model_dump(mode="json")})
    return {
        "schema": 1,
        "experiment": experiment.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "personas": referenced,
        "prompt": {"version": prompt_version, "text": prompt_text},
        "implementation_version": implementation_version,
    }


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
