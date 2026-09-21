"""YouTube DOM selectors.

Keep every selector here. YouTube changes markup frequently; callers should treat
these as best-effort probes and preserve slot rank even when metadata extraction
fails.
"""

from __future__ import annotations

HOME_SLOTS = "ytd-rich-grid-renderer ytd-rich-item-renderer"
WATCH_NEXT_SLOTS = ", ".join(
    [
        "ytd-watch-next-secondary-results-renderer ytd-compact-video-renderer",
        "ytd-watch-next-secondary-results-renderer ytd-compact-playlist-renderer",
        "ytd-watch-next-secondary-results-renderer ytd-compact-radio-renderer",
    ]
)

TITLE_LINK = "a#video-title-link, a#video-title"
CHANNEL_LINK = "ytd-channel-name a, #channel-name a"
DURATION = "ytd-thumbnail-overlay-time-status-renderer #text, #overlays #text"
BADGE = "ytd-badge-supported-renderer, ytd-thumbnail-overlay-now-playing-renderer"

AVATAR_BUTTON = "button#avatar-btn, ytd-topbar-menu-button-renderer #avatar-btn"
SIGN_IN_LINK = "a[href*='accounts.google.com/ServiceLogin'], tp-yt-paper-button[aria-label*='Sign in']"

CHALLENGE_SELECTORS = ", ".join(
    [
        "iframe[src*='recaptcha']",
        "form[action*='challenge']",
        "#captcha",
    ]
)
CHALLENGE_TEXT = (
    "unusual traffic",
    "unusual activity",
    "verify it's you",
    "verify that it's you",
    "confirm you're not a robot",
    "before you continue to youtube",
)
