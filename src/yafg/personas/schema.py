"""Persona schema.

A persona is the *experimental variable*: everything the researcher varies between
agents. It is deliberately separated from the `Account` (the identity material) and
from the `Context` (the shared warm-up exposure), so that a difference observed in
recommendations can be attributed to the persona and not to the account or the
starting state.

Personas are authored as YAML in `configs/personas/` and resolved into these models.
The resolved persona is snapshotted verbatim into the database at run start, so a
later edit to the YAML can never silently change what an old run meant.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Slug = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=64)]
CountryCode = Annotated[str, Field(pattern=r"^[A-Z]{2}$")]
LanguageTag = Annotated[str, Field(pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")]

Sex = Literal["male", "female", "unspecified"]
Religiosity = Literal["none", "nominal", "observant", "devout"]
Education = Literal["primary", "secondary", "vocational", "undergraduate", "postgraduate"]
Socioeconomic = Literal["low", "lower_middle", "middle", "upper_middle", "high"]
Urbanicity = Literal["rural", "suburban", "urban"]
Device = Literal["desktop", "mobile"]


class Range(BaseModel):
    """Inclusive numeric range sampled per session or per step."""

    model_config = ConfigDict(frozen=True)

    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.min > self.max:
            raise ValueError(f"range min ({self.min}) must be <= max ({self.max})")
        return self


class Demographics(BaseModel):
    """The axes the audit varies. Every field is optional except age and nationality,
    so a study can hold most of the profile fixed and move one axis at a time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    age: int = Field(ge=13, le=99)
    sex: Sex = "unspecified"
    nationality: CountryCode
    residence_country: CountryCode | None = None
    languages: list[LanguageTag] = Field(min_length=1)
    religion: str | None = None
    religiosity: Religiosity | None = None
    education: Education | None = None
    occupation: str | None = None
    socioeconomic: Socioeconomic | None = None
    urbanicity: Urbanicity | None = None
    political_lean: str | None = Field(
        default=None,
        description="Free text, study-defined (e.g. 'pro-Palestine', 'green-left'). Kept unstructured "
        "because the meaningful axis differs per audit; the lean *scoring* lives in analysis, not here.",
    )

    @model_validator(mode="after")
    def _default_residence(self) -> Self:
        if self.residence_country is None:
            object.__setattr__(self, "residence_country", self.nationality)
        return self


class Interest(BaseModel):
    """A weighted topical pull. Weights are relative, normalised at prompt time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(min_length=2, max_length=120)
    weight: float = Field(default=1.0, gt=0, le=10)
    seed_queries: list[str] = Field(default_factory=list)


class ViewingHabits(BaseModel):
    """Behavioural realism knobs.

    These are *not* decided by the LLM — they are sampled by the agent loop so that
    pacing and watch depth stay reproducible and cheap. The LLM only chooses *which*
    video to watch next and why.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    videos_per_session: Range = Range(min=5, max=15)
    watch_fraction: Range = Field(
        default=Range(min=0.25, max=0.9),
        description="Fraction of each video actually watched before moving on.",
    )
    dwell_seconds: Range = Field(
        default=Range(min=45, max=420),
        description="Hard cap on real wall-clock watch time per video, independent of watch_fraction.",
    )
    skip_probability: float = Field(default=0.15, ge=0, le=1)
    search_probability: float = Field(
        default=0.1, ge=0, le=1, description="Chance a step starts from a search instead of a recommendation."
    )
    shorts_probability: float = Field(default=0.0, ge=0, le=1)
    active_hours: list[int] = Field(
        default_factory=lambda: list(range(8, 24)),
        description="Local hours (0-23) during which this persona plausibly browses.",
    )
    device: Device = "desktop"

    @model_validator(mode="after")
    def _hours_in_range(self) -> Self:
        if any(h < 0 or h > 23 for h in self.active_hours):
            raise ValueError("active_hours entries must be 0-23")
        return self


class Persona(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Slug
    display_name: str = Field(min_length=1, max_length=120)
    demographics: Demographics
    interests: list[Interest] = Field(min_length=1)
    habits: ViewingHabits = ViewingHabits()
    bio: str = Field(
        min_length=40,
        description="Free-text first-person description handed to the LLM. This is the highest-signal "
        "field: the structured demographics mostly exist so studies can be indexed and grouped.",
    )
    priors: dict[str, float] = Field(
        default_factory=dict,
        description="Optional stance scores in [-1, 1] keyed by study-defined dimension, used by analysis "
        "to compute rank-weighted lean. Never shown to the LLM.",
    )
    notes: str | None = None

    @model_validator(mode="after")
    def _priors_bounded(self) -> Self:
        for key, value in self.priors.items():
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"prior {key!r} must be within [-1, 1], got {value}")
        return self

    @property
    def normalised_interests(self) -> list[tuple[str, float]]:
        total = sum(i.weight for i in self.interests)
        return [(i.topic, i.weight / total) for i in self.interests]
