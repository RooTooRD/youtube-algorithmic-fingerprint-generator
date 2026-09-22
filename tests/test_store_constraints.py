from __future__ import annotations

from sqlalchemy import inspect

from yafg.store.models import Observation, Run, Step


def test_rank_is_unique_per_step() -> None:
    constraint = Observation.__table_args__[0]
    assert tuple(constraint.columns.keys()) == ("step_id", "rank")


def test_nullable_evidence_fields_match_failure_model() -> None:
    mapper = inspect(Observation)
    assert mapper.columns.video_id.nullable
    assert mapper.columns.title.nullable
    step = inspect(Step)
    assert step.columns.active_persona.nullable


def test_run_arm_repetition_is_unique_per_experiment() -> None:
    index = Run.__table_args__[0]
    assert index.unique
    assert tuple(index.columns.keys()) == ("experiment_id", "arm", "repetition")
