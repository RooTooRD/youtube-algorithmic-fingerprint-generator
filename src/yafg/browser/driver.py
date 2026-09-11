"""Playwright driver.

One persistent context per account (`launch_persistent_context`), so cookies, local
storage and the login session survive between sessions the way a real browser profile
does. Contexts are never shared or copied between accounts.

Everything the agent can do on the platform passes through this module, which is what
makes the interaction budget in `Interactions` enforceable in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from yafg.experiment.schema import Surface
from yafg.identity.schema import Account
from yafg.llm.base import Candidate

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Page


class LoginRequired(RuntimeError):
    """Raised when a page renders signed-out. The run halts; a human re-provisions."""


class ChallengeDetected(RuntimeError):
    """Raised on a CAPTCHA, consent interstitial or 'unusual activity' wall.

    The framework never attempts to solve these. It records an Event, stops the run,
    and surfaces the account label for manual attention.
    """


@dataclass(slots=True)
class WatchResult:
    video_id: str
    watched_seconds: float
    watch_fraction: float
    completed: bool


class YouTubeDriver:
    """Thin, typed surface over Playwright. Deliberately not a page-object zoo:
    YouTube's DOM churns, so selectors live in `yafg.browser.selectors` and every
    scrape degrades to "return what parsed" rather than raising."""

    def __init__(self, account: Account, *, headless: bool = True) -> None:
        self.account = account
        self.headless = headless
        self._context: BrowserContext | None = None

    async def __aenter__(self) -> YouTubeDriver:
        raise NotImplementedError("P1: launch_persistent_context with account envelope")

    async def __aexit__(self, *exc: object) -> None:
        raise NotImplementedError("P1: graceful close, flush profile to disk")

    async def assert_signed_in(self) -> str:
        """Return the account's channel handle, or raise LoginRequired/ChallengeDetected."""
        raise NotImplementedError("P1")

    async def collect(self, surface: Surface, *, source_video_id: str | None = None) -> list[Candidate]:
        """Scrape every visible recommendation on `surface`, **in rendered order**.

        Rank fidelity is the whole measurement, so this scrolls the container to a
        fixed depth rather than a fixed count, and records the rank as rendered even
        when metadata fails to parse.
        """
        raise NotImplementedError("P1")

    async def watch(self, video_id: str, *, seconds: float, fraction: float) -> WatchResult:
        """Navigate to a video and let it play. Real wall-clock time — a watch that is
        not actually watched does not register with the recommender."""
        raise NotImplementedError("P1")

    async def search(self, query: str) -> list[Candidate]:
        raise NotImplementedError("P2")

    async def like(self, video_id: str) -> None:
        """Gated behind `Interactions.like`. Off by default."""
        raise NotImplementedError("P2")

    async def subscribe(self, channel_id: str) -> None:
        """Gated behind `Interactions.subscribe`. Off by default."""
        raise NotImplementedError("P2")

    @staticmethod
    async def interactive_login(account: Account) -> None:
        """Open a HEADED browser and wait for a human to sign in.

        This function types nothing and reads no credential. It waits for the session
        cookie to appear, verifies the signed-in state, then closes. It is the only
        way an account becomes `active`.
        """
        raise NotImplementedError("P1")

    @property
    def page(self) -> Page:
        raise NotImplementedError("P1")
