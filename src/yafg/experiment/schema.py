"""Experiment and context schemas.

An experiment is config-as-code: a YAML file that fully determines a run. The
resolved manifest (experiment + referenced configs + prompt version + behavior seed)
is what should be hashed at run start; hashing raw YAML alone is intentionally not
part of this schema contract.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yafg.personas.schema import Range, Slug

BehaviorMode = Literal["single", "mixed", "sequential", "random"]
Surface = Literal["home", "watch_next", "search", "shorts", "subscriptions"]
VideoId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{11}$")]
AccountLabel = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,63}$")]


def _default_surfaces() -> list[Surface]:
    return ["home", "watch_next"]


class Context(BaseModel):
    """Phase 1: the shared warm-up exposure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Slug
    description: str
    videos: list[VideoId] = Field(min_length=1)
    watch_fraction: float = Field(default=0.9, gt=0, le=1)
    max_watch_seconds: float = Field(default=600, ge=0, le=3600, description="Hard wall-clock cap per warm-up video.")


class Interactions(BaseModel):
    """Which side effects the agent is permitted to produce on the platform."""

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
    """Wall-clock politeness. Caps exist so a bad config cannot become load."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_delay_seconds: Range = Range(min=4, max=20)
    session_gap_seconds: Range = Range(min=300, max=3600)
    max_requests_per_hour: int = Field(default=240, ge=1, le=240)

    @model_validator(mode="after")
    def _non_negative_ranges(self) -> Self:
        if self.step_delay_seconds.min < 0 or self.session_gap_seconds.min < 0:
            raise ValueError("pacing ranges cannot contain negative values")
        return self


class Experiment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Slug
    description: str
    mode: BehaviorMode
    personas: list[PersonaRef] = Field(min_length=1)
    context: Slug
    steps: int = Field(ge=1, le=500)
    repetitions: int = Field(default=1, ge=1, le=100)
    surfaces: list[Surface] = Field(default_factory=_default_surfaces, min_length=1)
    accounts: list[AccountLabel] = Field(
        default_factory=list,
        description="Provisioned account labels. Labels must be unique; an account is never shared concurrently.",
    )
    llm: LLMConfig = LLMConfig()
    interactions: Interactions = Interactions()
    pacing: Pacing = Pacing()
    concurrency: int = Field(default=1, ge=1, le=32)
    switch_every: int | None = Field(default=None, ge=1, description="Sequential mode only: steps between switches.")
    behavior_seed: int = Field(
        default=0,
        description="Seed for persona draws, random-baseline choices, pacing, skips and watch-depth sampling.",
    )

    @model_validator(mode="after")
    def _mode_consistency(self) -> Self:
        if self.mode == "single" and not self.personas:
            raise ValueError("single mode requires at least one persona arm")
        if self.mode == "single" and len({ref.persona for ref in self.personas}) != len(self.personas):
            raise ValueError("single mode persona arms must be unique")
        if self.mode == "sequential":
            if len(self.personas) < 2:
                raise ValueError("sequential mode needs at least two personas to switch between")
            if self.switch_every is None:
                raise ValueError("sequential mode requires switch_every")
        elif self.switch_every is not None:
            raise ValueError("switch_every is only valid in sequential mode")
        if len(set(self.accounts)) != len(self.accounts):
            raise ValueError("account labels must be unique: accounts are never shared between arms")
        arm_count = len(self.personas) if self.mode == "single" else 1
        required_accounts = arm_count * self.repetitions
        if self.accounts and len(self.accounts) < required_accounts:
            raise ValueError(
                f"{required_accounts} planned runs require {required_accounts} distinct accounts; "
                f"only {len(self.accounts)} configured"
            )
        return self
