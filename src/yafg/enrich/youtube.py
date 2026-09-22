"""Out-of-band YouTube metadata and public-caption enrichment.

Metadata uses the supported YouTube Data API. Public transcript retrieval is a
best-effort separate adapter because the official captions API requires OAuth and
permission to edit the video; transcript collection is therefore explicit opt-in.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import and_, or_, select, union
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from yafg.store.models import Observation, Step, Video

_VIDEO_API = "https://www.googleapis.com/youtube/v3/videos"
_WATCH_URL = "https://www.youtube.com/watch"
_DURATION_RE = re.compile(r"^P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$")
_BLOCK_MARKERS = ("captcha", "unusual traffic", "before you continue to youtube")
Sleep = Callable[[float], Awaitable[None]]


class TranscriptPageError(RuntimeError):
    pass


class TranscriptRateLimited(RuntimeError):
    pass


class TranscriptBlocked(RuntimeError):
    pass


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _error_message(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{type(exc).__name__}: HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.RequestError):
        return f"{type(exc).__name__}: request failed"
    return f"{type(exc).__name__}: {exc}"[:2000]


def parse_duration_seconds(value: str) -> int:
    match = _DURATION_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported ISO-8601 duration {value!r}")
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _parse_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


class VideoMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    published_at: dt.datetime | None = None
    duration_seconds: int | None = Field(default=None, ge=0)
    view_count: int | None = Field(default=None, ge=0)
    category_id: str | None = None
    default_language: str | None = None
    tags: list[str] = Field(default_factory=list)
    description: str | None = None
    captions_available: bool | None = None


class YouTubeMetadataClient:
    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY is required for metadata enrichment")
        self.api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=30.0)

    async def fetch(self, video_ids: list[str]) -> dict[str, VideoMetadata]:
        if len(video_ids) > 50:
            raise ValueError("YouTube videos.list accepts at most 50 IDs per request")
        if not video_ids:
            return {}
        response = await self._client.get(
            _VIDEO_API,
            params={
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(video_ids),
                "key": self.api_key,
                "maxResults": 50,
            },
        )
        response.raise_for_status()
        body = response.json()
        items = body.get("items")
        if not isinstance(items, list):
            raise ValueError("YouTube API response has no items list")

        result: dict[str, VideoMetadata] = {}
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            snippet = item.get("snippet") if isinstance(item.get("snippet"), dict) else {}
            details = item.get("contentDetails") if isinstance(item.get("contentDetails"), dict) else {}
            statistics = item.get("statistics") if isinstance(item.get("statistics"), dict) else {}
            duration_raw = details.get("duration")
            duration = parse_duration_seconds(duration_raw) if isinstance(duration_raw, str) else None
            view_raw = statistics.get("viewCount")
            view_count = int(view_raw) if isinstance(view_raw, str) and view_raw.isdigit() else None
            tags_raw = snippet.get("tags")
            tags = [str(tag) for tag in tags_raw] if isinstance(tags_raw, list) else []
            result[item["id"]] = VideoMetadata(
                id=item["id"],
                title=snippet.get("title") if isinstance(snippet.get("title"), str) else None,
                channel_id=snippet.get("channelId") if isinstance(snippet.get("channelId"), str) else None,
                channel_name=snippet.get("channelTitle") if isinstance(snippet.get("channelTitle"), str) else None,
                published_at=_parse_datetime(
                    snippet.get("publishedAt") if isinstance(snippet.get("publishedAt"), str) else None
                ),
                duration_seconds=duration,
                view_count=view_count,
                category_id=snippet.get("categoryId") if isinstance(snippet.get("categoryId"), str) else None,
                default_language=(
                    snippet.get("defaultLanguage")
                    if isinstance(snippet.get("defaultLanguage"), str)
                    else snippet.get("defaultAudioLanguage")
                    if isinstance(snippet.get("defaultAudioLanguage"), str)
                    else None
                ),
                tags=tags,
                description=snippet.get("description") if isinstance(snippet.get("description"), str) else None,
                captions_available=(details.get("caption") == "true") if "caption" in details else None,
            )
        return result

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    text: str
    language: str | None


class PublicTranscriptClient:
    """Best-effort reader for caption tracks exposed on a public watch page."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        max_requests_per_hour: int = 120,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if not 1 <= max_requests_per_hour <= 240:
            raise ValueError("max_requests_per_hour must be between 1 and 240")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self._min_interval = 3600 / max_requests_per_hour
        self._sleep = sleep
        self._last_request_at: float | None = None
        self._request_lock = asyncio.Lock()

    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        async with self._request_lock:
            now = time.monotonic()
            if self._last_request_at is not None:
                await self._sleep(max(0, self._min_interval - (now - self._last_request_at)))
            self._last_request_at = time.monotonic()
        response = await self._client.get(url, **kwargs)
        if response.status_code == 429:
            raise TranscriptRateLimited("YouTube rate-limited transcript retrieval")
        if response.status_code == 403:
            raise TranscriptBlocked("YouTube blocked public transcript retrieval")
        response.raise_for_status()
        return response

    @staticmethod
    def _caption_tracks(html: str) -> list[dict[str, Any]]:
        marker = '"captionTracks":'
        start = html.find(marker)
        if start < 0:
            raise TranscriptPageError("watch page has no caption-track payload")
        payload = html[start + len(marker) :]
        try:
            value, _ = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError as exc:
            raise TranscriptPageError("watch page caption-track payload is malformed") from exc
        if not isinstance(value, list):
            raise TranscriptPageError("watch page caption-track payload is not a list")
        return [track for track in value if isinstance(track, dict)]

    @staticmethod
    def _select_track(tracks: list[dict[str, Any]], preferred_language: str | None) -> dict[str, Any] | None:
        if not tracks:
            return None
        preferred_prefix = preferred_language.split("-", 1)[0] if preferred_language else None

        def score(track: dict[str, Any]) -> tuple[int, int]:
            language = track.get("languageCode")
            language_match = int(
                bool(preferred_prefix and isinstance(language, str) and language.split("-", 1)[0] == preferred_prefix)
            )
            manual = int(track.get("kind") != "asr")
            return language_match, manual

        return max(tracks, key=score)

    @staticmethod
    def _text_from_json3(body: dict[str, Any]) -> str:
        lines: list[str] = []
        events = body.get("events")
        if not isinstance(events, list):
            return ""
        for event in events:
            if not isinstance(event, dict):
                continue
            segments = event.get("segs")
            if not isinstance(segments, list):
                continue
            text = "".join(
                segment.get("utf8", "")
                for segment in segments
                if isinstance(segment, dict) and isinstance(segment.get("utf8"), str)
            )
            text = " ".join(text.replace("\n", " ").split())
            if text:
                lines.append(text)
        return "\n".join(lines)

    async def fetch(self, video_id: str, *, preferred_language: str | None = None) -> TranscriptResult | None:
        page = await self._get(_WATCH_URL, params={"v": video_id, "hl": "en"})
        if page.url.host not in {"youtube.com", "www.youtube.com", "m.youtube.com"} or any(
            marker in page.text.lower() for marker in _BLOCK_MARKERS
        ):
            raise TranscriptBlocked("YouTube blocked public transcript retrieval")
        track = self._select_track(self._caption_tracks(page.text), preferred_language)
        if track is None:
            return None
        base_url = track.get("baseUrl")
        if not isinstance(base_url, str):
            raise TranscriptPageError("caption track has no download URL")
        separator = "&" if "?" in base_url else "?"
        captions = await self._get(f"{base_url}{separator}fmt=json3")
        body = captions.json()
        if not isinstance(body, dict):
            raise TranscriptPageError("caption response is not an object")
        text = self._text_from_json3(body)
        if not text:
            raise TranscriptPageError("caption response has no text")
        language = track.get("languageCode") if isinstance(track.get("languageCode"), str) else None
        return TranscriptResult(text=text, language=language)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


@dataclass(frozen=True, slots=True)
class EnrichmentStats:
    discovered: int = 0
    metadata_complete: int = 0
    metadata_not_found: int = 0
    metadata_errors: int = 0
    transcripts_complete: int = 0
    transcripts_unavailable: int = 0
    transcript_errors: int = 0


class EnrichmentWorker:
    def __init__(
        self,
        engine: AsyncEngine,
        metadata: YouTubeMetadataClient,
        *,
        transcripts: PublicTranscriptClient | None = None,
    ) -> None:
        self.engine = engine
        self.metadata = metadata
        self.transcripts = transcripts

    async def _discover(self, limit: int, *, include_transcripts: bool) -> tuple[list[str], list[str]]:
        ids_query = union(
            select(Observation.video_id.label("video_id")).where(Observation.video_id.is_not(None)),
            select(Step.chosen_video_id.label("video_id")).where(Step.chosen_video_id.is_not(None)),
        ).subquery()
        needs_enrichment = or_(Video.id.is_(None), Video.metadata_status.in_({"pending", "error"}))
        if include_transcripts:
            needs_enrichment = or_(
                needs_enrichment,
                and_(
                    Video.metadata_status == "complete",
                    Video.transcript_status.in_({"pending", "error"}),
                ),
            )
        async with AsyncSession(self.engine) as session:
            video_ids = list(
                (
                    await session.scalars(
                        select(ids_query.c.video_id)
                        .outerjoin(Video, Video.id == ids_query.c.video_id)
                        .where(needs_enrichment)
                        .order_by(ids_query.c.video_id)
                        .limit(limit)
                    )
                ).all()
            )
            if not video_ids:
                return [], []
            existing = {
                video.id: video for video in (await session.scalars(select(Video).where(Video.id.in_(video_ids)))).all()
            }
            metadata_ids: list[str] = []
            transcript_ids: list[str] = []
            for video_id in video_ids:
                video = existing.get(video_id)
                if video is None:
                    session.add(Video(id=video_id))
                    metadata_ids.append(video_id)
                elif video.metadata_status in {"pending", "error"}:
                    metadata_ids.append(video_id)
                elif include_transcripts and video.transcript_status in {"pending", "error"}:
                    transcript_ids.append(video_id)
            await session.commit()
            # Videos enriched for metadata in this pass are also eligible for transcript
            # work after metadata is written. The transcript method rechecks status.
            transcript_candidates = list(dict.fromkeys([*transcript_ids, *metadata_ids])) if include_transcripts else []
            return metadata_ids, transcript_candidates

    async def _write_metadata(self, requested: list[str], result: dict[str, VideoMetadata]) -> tuple[int, int]:
        completed = 0
        not_found = 0
        now = _now()
        async with AsyncSession(self.engine) as session:
            for video_id in requested:
                video = await session.get(Video, video_id)
                if video is None:
                    continue
                metadata = result.get(video_id)
                if metadata is None:
                    video.metadata_status = "not_found"
                    video.metadata_error = "videos.list returned no public resource for this ID"
                    video.enriched_at = now
                    video.transcript_status = "unavailable"
                    video.transcript_enriched_at = now
                    not_found += 1
                    continue
                video.title = metadata.title
                video.channel_id = metadata.channel_id
                video.channel_name = metadata.channel_name
                video.published_at = metadata.published_at
                video.duration_seconds = metadata.duration_seconds
                video.view_count = metadata.view_count
                video.category_id = metadata.category_id
                video.default_language = metadata.default_language
                video.tags = metadata.tags
                video.description = metadata.description
                video.captions_available = metadata.captions_available
                video.metadata_status = "complete"
                video.metadata_error = None
                video.enriched_at = now
                if metadata.captions_available is False:
                    video.transcript_status = "unavailable"
                    video.transcript_enriched_at = now
                completed += 1
            await session.commit()
        return completed, not_found

    async def _mark_metadata_error(self, video_ids: list[str], exc: BaseException) -> None:
        message = _error_message(exc)
        async with AsyncSession(self.engine) as session:
            for video_id in video_ids:
                video = await session.get(Video, video_id)
                if video is not None:
                    video.metadata_status = "error"
                    video.metadata_error = message
            await session.commit()

    async def _enrich_transcripts(self, video_ids: list[str]) -> tuple[int, int, int]:
        if self.transcripts is None:
            return 0, 0, 0
        complete = unavailable = errors = 0
        for video_id in video_ids:
            async with AsyncSession(self.engine) as session:
                video = await session.get(Video, video_id)
                if video is None or video.metadata_status != "complete":
                    continue
                if video.transcript_status not in {"pending", "error"}:
                    continue
                preferred_language = video.default_language
            try:
                transcript = await self.transcripts.fetch(video_id, preferred_language=preferred_language)
            except Exception as exc:
                async with AsyncSession(self.engine) as session:
                    video = await session.get(Video, video_id)
                    if video is not None:
                        video.transcript_status = "blocked" if isinstance(exc, TranscriptBlocked) else "error"
                        video.transcript_error = _error_message(exc)
                        await session.commit()
                errors += 1
                if isinstance(exc, TranscriptBlocked):
                    raise
                if isinstance(exc, TranscriptRateLimited):
                    break
                continue

            async with AsyncSession(self.engine) as session:
                video = await session.get(Video, video_id)
                if video is None:
                    continue
                video.transcript_enriched_at = _now()
                if transcript is None:
                    video.transcript_status = "unavailable"
                    video.transcript_error = None
                    unavailable += 1
                else:
                    video.transcript = transcript.text
                    video.transcript_language = transcript.language
                    video.transcript_status = "complete"
                    video.transcript_error = None
                    complete += 1
                await session.commit()
        return complete, unavailable, errors

    async def run(self, *, limit: int = 500, include_transcripts: bool = False) -> EnrichmentStats:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        metadata_ids, transcript_ids = await self._discover(limit, include_transcripts=include_transcripts)
        discovered = len(set(metadata_ids) | set(transcript_ids))
        completed = not_found = metadata_errors = 0
        for offset in range(0, len(metadata_ids), 50):
            batch = metadata_ids[offset : offset + 50]
            try:
                result = await self.metadata.fetch(batch)
                batch_completed, batch_not_found = await self._write_metadata(batch, result)
                completed += batch_completed
                not_found += batch_not_found
            except Exception as exc:
                await self._mark_metadata_error(batch, exc)
                metadata_errors += len(batch)

        transcript_complete = transcript_unavailable = transcript_errors = 0
        if include_transcripts:
            transcript_complete, transcript_unavailable, transcript_errors = await self._enrich_transcripts(
                transcript_ids
            )
        return EnrichmentStats(
            discovered=discovered,
            metadata_complete=completed,
            metadata_not_found=not_found,
            metadata_errors=metadata_errors,
            transcripts_complete=transcript_complete,
            transcripts_unavailable=transcript_unavailable,
            transcript_errors=transcript_errors,
        )

    async def aclose(self) -> None:
        await self.metadata.aclose()
        if self.transcripts is not None:
            await self.transcripts.aclose()
