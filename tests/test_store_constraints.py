from __future__ import annotations

from sqlalchemy import inspect

from yafg.store.models import Observation, Step


def test_rank_is_unique_per_step() -> None:
    constraints = {
        tuple(c.columns.keys())
        for c in Observation.__table__.constraints  # ty: ignore[unresolved-attribute]
        if hasattr(c, "columns")
    }
    assert ("step_id", "rank") in constraints


def test_nullable_evidence_fields_match_failure_model() -> None:
    mapper = inspect(Observation)
    assert mapper.columns.video_id.nullable
    assert mapper.columns.title.nullable
    step = inspect(Step)
    assert step.columns.active_persona.nullable
