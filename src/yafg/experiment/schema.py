"""Experiment and context schemas.

An experiment is config-as-code: a YAML file that fully determines a run, hashed into
a `manifest_hash` so two runs claiming to be the same experiment can be proven equal.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yafg.personas.schema import Range, Slug

BehaviorMode = Literal["single", "mixed", "sequential", "random"]
Surface = Literal["home", "watch_next", "search", "shorts", "subscriptions"]
VideoId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{11}$")]


def _default_surfaces() -> list[Surface]:
    return ["home", "watch_next"]


class Context(BaseModel):
    """Phase 1: the shared warm-up exposure.

    Every agent in an experiment watches this identical list before its persona takes
    over. This is what makes cross-persona comparison meaningful — divergence after
    the context can be attributed to the persona rather than to cold-start noise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Slug
    description: str
    videos: list[VideoId] = Field(min_length=1)
    watch_fraction: float = Field(default=0.9, gt=0, le=1)


class Interactions(BaseModel):
    """Which side effects the agent is permitted to produce on the platform.

    Defaults are read-mostly. `comment` is pinned to False by a validator: this
    framework does not author public content under a synthetic identity, because that
    is manipulation of a public commons rather than measurement of a private feed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    like: bool = False
    dislike: bool = False
    subscribe: bool = False
    comment: Literal[False] = False

    @model_validator(mode="after")
    def _no_public_authoring(self) -> Self:
        if self.comment:
            raise ValueError("commenting is not supported: see docs/ETHICS.md")
        return self


class PersonaRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    persona: Slug
    weight: float = Field(default=1.0, gt=0)


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["claude", "ollama"] = "claude"
    model: str = "claude-sonnet-5"
    temperature: float = Field(default=1.0, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=64)
    seed: int | None = None


class Pacing(BaseModel):
    """Wall-clock politeness. Caps exist so a misconfigured run cannot turn into load."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_delay_seconds: Range = Range(min=4, max=20)
    session_gap_seconds: Range = Range(min=300, max=3600)
    max_requests_per_hour: int = Field(default=240, ge=1, le=2000)


class Experiment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Slug
    description: str
    mode: BehaviorMode
    personas: list[PersonaRef] = Field(min_length=1)
    context: Slug
    steps: int = Field(ge=1, le=500)
    repetitions: int = Field(default=1, ge=1, le=100)
    surfaces: list[Surface] = Field(default_factory=_default_surfaces)
    accounts: list[str] = Field(
        default_factory=list,
        description="Account labels provisioned via `yafg account login`. Must be >= number of "
        "concurrent arms; one account is never shared across two arms.",
    )
    llm: LLMConfig = LLMConfig()
    interactions: Interactions = Interactions()
    pacing: Pacing = Pacing()
    concurrency: int = Field(default=1, ge=1, le=32)
    switch_every: int | None = Field(default=None, description="Sequential mode only: steps between persona switches.")

    @model_validator(mode="after")
    def _mode_consistency(self) -> Self:
        if self.mode == "single" and len(self.personas) != 1:
            raise ValueError("single mode takes exactly one persona; use 'mixed' for a weighted pool")
        if self.mode == "sequential":
            if len(self.personas) < 2:
                raise ValueError("sequential mode needs at least two personas to switch between")
            if self.switch_every is None:
                raise ValueError("sequential mode requires switch_every")
        # `random` mode is the no-LLM baseline: `llm` is ignored rather than rejected,
        # so the same file can be re-run under a persona mode by changing one line.
        if self.accounts and len(self.accounts) < self.concurrency:
            raise ValueError(
                f"{len(self.accounts)} accounts for concurrency {self.concurrency}: "
                "accounts are never shared between concurrent arms"
            )
        return self
