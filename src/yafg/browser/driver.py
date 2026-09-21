"""Conservative Playwright driver for observation and real-time watching."""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, quote_plus, urlparse

from yafg.browser import selectors
from yafg.experiment.schema import Interactions, Surface
from yafg.identity.schema import Account
from yafg.llm.base import Candidate
from yafg.settings import settings

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Locator, Page, Playwright


class LoginRequired(RuntimeError):
    """Raised when a page renders signed-out. The run halts; a human re-provisions."""


class ChallengeDetected(RuntimeError):
    """Raised on a CAPTCHA, consent interstitial or unusual-activity wall."""


@dataclass(slots=True)
class WatchResult:
    video_id: str
    watched_seconds: float
    watch_fraction: float
    completed: bool


def _video_id_from_href(href: str | None) -> str | None:
    if not href:
        return None
    parsed = urlparse(href)
    if parsed.path == "/watch":
        value = parse_qs(parsed.query).get("v", [None])[0]
        return value if value and re.fullmatch(r"[A-Za-z0-9_-]{11}", value) else None
    if parsed.path.startswith("/shorts/"):
        value = parsed.path.split("/", 2)[-1].split("?", 1)[0]
        return value if re.fullmatch(r"[A-Za-z0-9_-]{11}", value) else None
    return None


async def _text(locator: Locator) -> str | None:
    try:
        value = (await locator.first.inner_text(timeout=750)).strip()
        return value or None
    except Exception:  # DOM churn is an expected scrape condition.
        return None


async def _attr(locator: Locator, name: str) -> str | None:
    try:
        return await locator.first.get_attribute(name, timeout=750)
    except Exception:
        return None


class YouTubeDriver:
    """Typed boundary around all platform interaction."""

    def __init__(
        self,
        account: Account,
        *,
        headless: bool = True,
        interactions: Interactions | None = None,
        allow_interactions: bool | None = None,
    ) -> None:
        self.account = account
        self.headless = headless
        self.interactions = interactions or Interactions()
        self.allow_interactions = settings.allow_interactions if allow_interactions is None else allow_interactions
        self._context: BrowserContext | None = None
        self._playwright: Playwright | None = None
        self._page: Page | None = None

    def _proxy_options(self) -> dict[str, str] | None:
        if self.account.proxy is None:
            return None
        result = {"server": self.account.proxy.server}
        if self.account.proxy.username:
            result["username"] = self.account.proxy.username
        if self.account.proxy.password_env:
            password = os.environ.get(self.account.proxy.password_env)
            if not password:
                raise RuntimeError(f"proxy password env var {self.account.proxy.password_env!r} is not set")
            result["password"] = password
        return result

    async def __aenter__(self) -> YouTubeDriver:
        from playwright.async_api import async_playwright

        self.account.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()
        options: dict[str, Any] = {
            "user_data_dir": str(self.account.profile_dir),
            "headless": self.headless,
            "locale": self.account.locale,
            "timezone_id": self.account.timezone,
            "viewport": {"width": self.account.viewport[0], "height": self.account.viewport[1]},
        }
        if self.account.geolocation is not None:
            lat, lon = self.account.geolocation
            options["geolocation"] = {"latitude": lat, "longitude": lon}
            options["permissions"] = ["geolocation"]
        proxy = self._proxy_options()
        if proxy:
            options["proxy"] = proxy
        try:
            self._context = await self._playwright.chromium.launch_persistent_context(**options)
            pages = self._context.pages
            self._page = pages[0] if pages else await self._context.new_page()
        except Exception:
            await self._playwright.stop()
            self._playwright = None
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        try:
            if self._context is not None:
                await self._context.close()
        finally:
            self._context = None
            self._page = None
            if self._playwright is not None:
                await self._playwright.stop()
                self._playwright = None

    async def _goto(self, url: str) -> None:
        await self.page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        await self._raise_if_challenged()

    async def _raise_if_challenged(self) -> None:
        page = self.page
        try:
            if await page.locator(selectors.CHALLENGE_SELECTORS).count():
                raise ChallengeDetected(f"challenge detected for account {self.account.label}")
            text = (await page.locator("body").inner_text(timeout=2_000)).lower()
        except ChallengeDetected:
            raise
        except Exception:
            return
        if any(marker in text for marker in selectors.CHALLENGE_TEXT):
            raise ChallengeDetected(f"challenge/consent wall detected for account {self.account.label}")

    async def assert_signed_in(self) -> str:
        if "youtube.com" not in self.page.url:
            await self._goto("https://www.youtube.com/")
        else:
            await self._raise_if_challenged()
        if await self.page.locator(selectors.AVATAR_BUTTON).count():
            label = await _attr(self.page.locator(selectors.AVATAR_BUTTON), "aria-label")
            return label or self.account.label
        if await self.page.locator(selectors.SIGN_IN_LINK).count():
            raise LoginRequired(f"account {self.account.label} is signed out")
        # UI can be partially loaded; signed-in Google session cookies are a stronger fallback.
        cookies = await self.page.context.cookies("https://www.youtube.com")
        names = {cookie["name"] for cookie in cookies}
        if names.intersection({"SAPISID", "__Secure-3PAPISID", "__Secure-1PAPISID"}):
            return self.account.label
        raise LoginRequired(f"could not verify signed-in state for account {self.account.label}")

    async def _scroll_for_observation(self, *, rounds: int = 3) -> None:
        for _ in range(rounds):
            await self.page.evaluate("window.scrollBy(0, Math.max(window.innerHeight, 800))")
            await self.page.wait_for_timeout(600)

    async def _candidate_from_slot(self, slot: Locator, rank: int) -> Candidate:
        link = slot.locator(selectors.TITLE_LINK)
        href = await _attr(link, "href")
        title = await _attr(link, "title") or await _text(link)
        channel = slot.locator(selectors.CHANNEL_LINK)
        channel_href = await _attr(channel, "href")
        channel_id = None
        if channel_href and "/channel/" in channel_href:
            channel_id = channel_href.split("/channel/", 1)[1].split("/", 1)[0].split("?", 1)[0]
        return Candidate(
            rank=rank,
            video_id=_video_id_from_href(href),
            title=title,
            channel_name=await _text(channel),
            channel_id=channel_id,
            duration_label=await _text(slot.locator(selectors.DURATION)),
            badge=await _text(slot.locator(selectors.BADGE)),
        )

    async def collect(self, surface: Surface, *, source_video_id: str | None = None) -> list[Candidate]:
        if surface == "home":
            await self._goto("https://www.youtube.com/")
            await self.assert_signed_in()
            await self._scroll_for_observation()
            locator = self.page.locator(selectors.HOME_SLOTS)
        elif surface == "watch_next":
            if source_video_id is not None and _video_id_from_href(self.page.url) != source_video_id:
                await self._goto(f"https://www.youtube.com/watch?v={source_video_id}")
            elif source_video_id is None and "/watch" not in self.page.url:
                raise ValueError("watch_next requires source_video_id unless already on a watch page")
            await self.assert_signed_in()
            await self._scroll_for_observation()
            locator = self.page.locator(selectors.WATCH_NEXT_SLOTS)
        else:
            raise ValueError(f"surface {surface!r} is not implemented in P1")

        count = await locator.count()
        candidates: list[Candidate] = []
        for rank in range(count):
            slot = locator.nth(rank)
            try:
                candidates.append(await self._candidate_from_slot(slot, rank))
            except Exception:
                # Preserve the rendered rank even when a slot mutates during parsing.
                candidates.append(Candidate(rank=rank))
        return candidates

    async def watch(self, video_id: str, *, seconds: float, fraction: float) -> WatchResult:
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        if not 0 <= fraction <= 1:
            raise ValueError("fraction must be within [0, 1]")
        await self._goto(f"https://www.youtube.com/watch?v={video_id}")
        await self.assert_signed_in()
        video = self.page.locator("video.html5-main-video, video").first
        await video.wait_for(state="attached", timeout=20_000)
        duration = await video.evaluate("el => Number.isFinite(el.duration) ? el.duration : null")
        target = seconds
        if isinstance(duration, (int, float)) and duration > 0:
            target = min(seconds, float(duration) * fraction)
        await video.evaluate("el => el.play().catch(() => undefined)")
        started = time.monotonic()
        await asyncio.sleep(target)
        watched = time.monotonic() - started
        current = await video.evaluate("el => Number.isFinite(el.currentTime) ? el.currentTime : 0")
        actual_fraction = min(1.0, float(current) / float(duration)) if duration else fraction
        completed = bool(duration and current >= max(float(duration) - 1.0, 0.0))
        return WatchResult(
            video_id=video_id, watched_seconds=watched, watch_fraction=actual_fraction, completed=completed
        )

    async def search(self, query: str) -> list[Candidate]:
        query = query.strip()
        if not query:
            raise ValueError("search query cannot be empty")
        await self._goto(f"https://www.youtube.com/results?search_query={quote_plus(query)}")
        await self.assert_signed_in()
        await self._scroll_for_observation()
        locator = self.page.locator(selectors.SEARCH_SLOTS)
        count = await locator.count()
        candidates: list[Candidate] = []
        for rank in range(count):
            slot = locator.nth(rank)
            try:
                candidates.append(await self._candidate_from_slot(slot, rank))
            except Exception:
                candidates.append(Candidate(rank=rank))
        return candidates

    def _require_interaction(self, name: str) -> None:
        configured = bool(getattr(self.interactions, name, False))
        if not self.allow_interactions or not configured:
            raise PermissionError(
                f"{name} requires both experiment interactions.{name}=true and YAFG_ALLOW_INTERACTIONS=1"
            )

    async def _ensure_video_page(self, video_id: str) -> None:
        if _video_id_from_href(self.page.url) != video_id:
            await self._goto(f"https://www.youtube.com/watch?v={video_id}")
        await self.assert_signed_in()

    async def like(self, video_id: str) -> None:
        self._require_interaction("like")
        await self._ensure_video_page(video_id)
        button = self.page.locator(selectors.LIKE_BUTTON).first
        await button.wait_for(state="visible", timeout=10_000)
        if (await button.get_attribute("aria-pressed")) != "true":
            await button.click()

    async def dislike(self, video_id: str) -> None:
        self._require_interaction("dislike")
        await self._ensure_video_page(video_id)
        button = self.page.locator(selectors.DISLIKE_BUTTON).first
        await button.wait_for(state="visible", timeout=10_000)
        if (await button.get_attribute("aria-pressed")) != "true":
            await button.click()

    async def subscribe(self, channel_id: str) -> None:
        self._require_interaction("subscribe")
        if "/watch" not in self.page.url:
            raise ValueError("subscribe requires an open watch page for the selected channel")
        channel_href = await _attr(self.page.locator(selectors.CHANNEL_LINK), "href")
        rendered_channel_id = None
        if channel_href and "/channel/" in channel_href:
            rendered_channel_id = channel_href.split("/channel/", 1)[1].split("/", 1)[0].split("?", 1)[0]
        if rendered_channel_id != channel_id:
            raise ValueError("refusing subscription: current watch page channel could not be verified")
        button = self.page.locator(selectors.SUBSCRIBE_BUTTON).first
        await button.wait_for(state="visible", timeout=10_000)
        text = ((await button.inner_text()) or "").strip().lower()
        if "subscribed" not in text:
            await button.click()

    @staticmethod
    async def interactive_login(account: Account, *, timeout_seconds: float = 900) -> None:
        """Open a headed persistent browser and only observe until a human signs in."""
        driver = YouTubeDriver(account, headless=False)
        async with driver:
            await driver._goto("https://www.youtube.com/")
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                # Do not navigate while the operator is on Google's sign-in pages. We
                # merely observe the persistent context until YouTube session cookies
                # appear, then return to YouTube once for verification.
                await driver._raise_if_challenged()
                cookies = await driver.page.context.cookies("https://www.youtube.com")
                names = {cookie["name"] for cookie in cookies}
                if names.intersection({"SAPISID", "__Secure-3PAPISID", "__Secure-1PAPISID"}):
                    await driver._goto("https://www.youtube.com/")
                    await driver.assert_signed_in()
                    return
                await driver.page.wait_for_timeout(1_000)
            raise LoginRequired(f"sign-in was not verified for account {account.label} before timeout")

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("driver is not open; use 'async with YouTubeDriver(...)'")
        return self._page
