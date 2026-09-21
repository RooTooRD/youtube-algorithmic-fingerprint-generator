from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from yafg.browser.driver import YouTubeDriver, _video_id_from_href
from yafg.experiment.schema import Experiment, Interactions
from yafg.identity.schema import Account
from yafg.llm.base import Candidate
from yafg.personas.schema import ViewingHabits


def _experiment(**updates: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "contract-test",
        "description": "contract test",
        "mode": "single",
        "personas": [{"persona": "neutral-baseline"}],
        "context": "warmup-neutral",
        "steps": 5,
        "accounts": ["account-01"],
        "concurrency": 1,
    }
    data.update(updates)
    return data


def test_duplicate_accounts_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Experiment.model_validate(_experiment(accounts=["same", "same"], concurrency=2))


def test_empty_surfaces_rejected() -> None:
    with pytest.raises(ValidationError):
        Experiment.model_validate(_experiment(surfaces=[]))


def test_request_ceiling_is_human_scale() -> None:
    with pytest.raises(ValidationError):
        Experiment.model_validate(_experiment(pacing={"max_requests_per_hour": 241}))


def test_behavior_seed_is_distinct_from_llm_seed() -> None:
    experiment = Experiment.model_validate(_experiment(behavior_seed=123, llm={"seed": 456}))
    assert experiment.behavior_seed == 123
    assert experiment.llm.seed == 456


def test_habit_ranges_have_semantic_bounds() -> None:
    with pytest.raises(ValidationError, match="watch_fraction"):
        ViewingHabits.model_validate({"watch_fraction": {"min": -2, "max": 3}})
    with pytest.raises(ValidationError, match="whole numbers"):
        ViewingHabits.model_validate({"videos_per_session": {"min": 0.5, "max": 0.75}})


def test_account_rejects_unknown_timezone(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unknown IANA timezone"):
        Account(label="test-01", profile_dir=tmp_path, timezone="Not/A_Zone")


def test_candidate_preserves_unparseable_slot() -> None:
    candidate = Candidate(rank=3)
    assert candidate.rank == 3
    assert candidate.video_id is None
    assert candidate.title is None


def test_video_id_extraction() -> None:
    assert _video_id_from_href("/watch?v=dQw4w9WgXcQ&list=x") == "dQw4w9WgXcQ"
    assert _video_id_from_href("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _video_id_from_href("/playlist?list=abc") is None


def test_interactions_require_both_gates(tmp_path: Path) -> None:
    account = Account(label="test-01", profile_dir=tmp_path)
    driver = YouTubeDriver(account, interactions=Interactions(like=True), allow_interactions=False)
    with pytest.raises(PermissionError, match="both"):
        driver._require_interaction("like")

    driver = YouTubeDriver(account, interactions=Interactions(like=False), allow_interactions=True)
    with pytest.raises(PermissionError, match="both"):
        driver._require_interaction("like")

    driver = YouTubeDriver(account, interactions=Interactions(like=True), allow_interactions=True)
    driver._require_interaction("like")


def test_choice_validation_falls_back_to_first_selectable() -> None:
    from yafg.agent.choice import resolve_choice
    from yafg.llm.base import Choice, Usage

    resolved = resolve_choice(
        Choice(rank=99, justification="bad rank"),
        Usage(model="test"),
        [Candidate(rank=0), Candidate(rank=1, video_id="dQw4w9WgXcQ", title="ok")],
    )
    assert resolved.choice.rank == 1
    assert resolved.candidate.video_id == "dQw4w9WgXcQ"
    assert resolved.usage.fallback_reason is not None
