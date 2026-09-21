"""LLM port used by the agent protocol."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """One rendered recommendation slot.

    Metadata is nullable by design. Rank fidelity is evidence: a slot that fails to
    parse remains present rather than shifting every recommendation below it.
    """

    rank: int = Field(ge=0)
    video_id: str | None = None
    title: str | None = None
    channel_name: str | None = None
    channel_id: str | None = None
    duration_label: str | None = None
    badge: str | None = None


class Choice(BaseModel):
    """Structured model output; the agent validates the rank against candidates."""

    rank: int = Field(ge=0)
    justification: str = Field(min_length=1, max_length=600)
    interest_match: str | None = Field(default=None)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    fallback_reason: str | None = None


class LLMProvider(Protocol):
    async def choose(
        self,
        *,
        system_prompt: str,
        history: list[str],
        candidates: list[Candidate],
    ) -> tuple[Choice, Usage]:
        """Return a proposed choice; the agent owns final rank validation/fallback."""
        ...
