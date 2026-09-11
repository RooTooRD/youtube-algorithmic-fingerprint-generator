"""LLM port.

The agent loop depends on this interface only. Swapping the model is an experimental
variable (the TRACE paper used Mistral Small 3.2 at <$0.01/agent), so no provider
specifics leak past this boundary.

The model's job is narrow: given a persona and a ranked candidate list, pick one index
and justify it. It does not drive the browser, choose pacing, or decide watch depth —
those are sampled from `ViewingHabits` so they stay reproducible and cheap.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """One recommendation offered to the model, in rendered rank order."""

    rank: int
    video_id: str
    title: str
    channel_name: str | None = None
    duration_label: str | None = None
    badge: str | None = None


class Choice(BaseModel):
    """Structured model output. `rank` indexes into the candidate list it was given."""

    rank: int = Field(ge=0)
    justification: str = Field(min_length=1, max_length=600)
    interest_match: str | None = Field(
        default=None, description="Which of the persona's stated interests this choice serves, if any."
    )


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class LLMProvider(Protocol):
    """Implemented by `yafg.llm.claude.ClaudeProvider` and `yafg.llm.ollama.OllamaProvider`."""

    async def choose(
        self,
        *,
        system_prompt: str,
        history: list[str],
        candidates: list[Candidate],
    ) -> tuple[Choice, Usage]:
        """Pick one candidate.

        Implementations MUST return a rank present in `candidates`; on a malformed or
        out-of-range response they retry once, then fall back to rank 0 and record the
        failure in the usage payload. A hard error here must never abort a run —
        an agent that cannot decide behaves like a user who takes the top result.
        """
        ...
