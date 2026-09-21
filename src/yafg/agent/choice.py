"""Central choice validation shared by every LLM provider."""

from __future__ import annotations

from dataclasses import dataclass

from yafg.llm.base import Candidate, Choice, Usage


class NoSelectableCandidate(RuntimeError):
    """The rendered set contains no video ID that can be watched."""


@dataclass(frozen=True, slots=True)
class ResolvedChoice:
    choice: Choice
    usage: Usage
    candidate: Candidate


def resolve_choice(choice: Choice, usage: Usage, candidates: list[Candidate]) -> ResolvedChoice:
    """Validate provider output and apply a deterministic first-selectable fallback."""
    by_rank = {candidate.rank: candidate for candidate in candidates if candidate.video_id is not None}
    selected = by_rank.get(choice.rank)
    if selected is not None:
        return ResolvedChoice(choice=choice, usage=usage, candidate=selected)
    if not by_rank:
        raise NoSelectableCandidate("recommendation set contains no selectable video IDs")
    fallback_rank = min(by_rank)
    fallback = by_rank[fallback_rank]
    reason = f"provider returned unselectable/out-of-range rank {choice.rank}; fell back to rank {fallback_rank}"
    return ResolvedChoice(
        choice=Choice(rank=fallback_rank, justification=reason),
        usage=usage.model_copy(update={"fallback_reason": reason}),
        candidate=fallback,
    )
