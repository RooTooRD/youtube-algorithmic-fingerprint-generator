"""Account / identity material.

Hard boundary of this project: **accounts are provisioned by a human, never by code.**
`yafg account login <label>` opens a headed browser at accounts.google.com and waits
for the operator to sign in themselves. This package never types credentials, never
registers an account, and never attempts a CAPTCHA. See docs/ETHICS.md.

What the code owns after that: the persistent browser profile, the locale/timezone/
proxy envelope that must stay pinned to the account for its whole life, and the
health check that detects a logged-out or challenged session and halts the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yafg.personas.schema import CountryCode, LanguageTag

AccountStatus = Literal["unprovisioned", "active", "logged_out", "challenged", "suspended", "retired"]


class ProxyConfig(BaseModel):
    """Residential proxy pinned to the account. The paper used residential proxies at
    ~$0.10/agent specifically to keep a synthetic profile's network origin consistent
    with its claimed geography — an account that logs in from Belgium and then browses
    from a datacentre IP is both a detection risk and a confound."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    server: str = Field(description="e.g. 'http://gate.example.com:8000'")
    username: str | None = None
    password_env: str | None = Field(
        default=None,
        description="Name of the env var holding the proxy password. The password itself is never "
        "stored in config or in the database.",
    )


class Account(BaseModel):
    """One synthetic identity. Bound 1:1 to a persona for the life of a study —
    reusing an account across personas contaminates its history."""

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
        if self.geolocation is not None:
            lat, lon = self.geolocation
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("geolocation out of bounds")
        return self
