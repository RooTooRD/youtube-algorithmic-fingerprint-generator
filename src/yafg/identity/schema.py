"""Account / identity material.

Accounts are provisioned by a human, never by code. The framework owns only the
persistent browser profile and the stable browser/network envelope around it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yafg.personas.schema import CountryCode, LanguageTag

AccountStatus = Literal["unprovisioned", "active", "logged_out", "challenged", "suspended", "retired"]


class ProxyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    server: str = Field(min_length=1, description="e.g. 'http://gate.example.com:8000'")
    username: str | None = None
    password_env: str | None = Field(
        default=None,
        description="Env var containing the proxy password; the secret itself is never persisted.",
    )


class Account(BaseModel):
    """One synthetic identity envelope.

    `persona_id` is optional provenance for single-persona studies. Mixed and
    sequential studies bind the account to the arm/policy instead; they must not
    pretend that one browser history is 1:1 with several personas.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,63}$")
    persona_id: str | None = None
    status: AccountStatus = "unprovisioned"

    profile_dir: Path = Field(description="Playwright persistent context directory. Gitignored.")
    locale: LanguageTag = "en-US"
    timezone: str = Field(default="UTC", description="IANA tz, e.g. 'Africa/Algiers'.")
    country: CountryCode | None = None
    geolocation: tuple[float, float] | None = None
    viewport: tuple[int, int] = (1440, 900)
    proxy: ProxyConfig | None = None

    notes: str | None = None

    @model_validator(mode="after")
    def _consistent_envelope(self) -> Self:
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone {self.timezone!r}") from exc
        if self.geolocation is not None:
            lat, lon = self.geolocation
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("geolocation out of bounds")
        width, height = self.viewport
        if width <= 0 or height <= 0:
            raise ValueError("viewport dimensions must be positive")
        return self
