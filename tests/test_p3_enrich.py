from __future__ import annotations

import json

import httpx
import pytest

from yafg.enrich.youtube import (
    PublicTranscriptClient,
    TranscriptBlocked,
    TranscriptPageError,
    TranscriptRateLimited,
    YouTubeMetadataClient,
    _error_message,
    parse_duration_seconds,
)


def test_iso_duration_parser() -> None:
    assert parse_duration_seconds("PT15M33S") == 933
    assert parse_duration_seconds("PT1H2M3S") == 3723
    assert parse_duration_seconds("P1DT2H") == 93600
    with pytest.raises(ValueError):
        parse_duration_seconds("15:33")


@pytest.mark.asyncio
async def test_metadata_client_parses_videos_list() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["part"] == "snippet,contentDetails,statistics"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "dQw4w9WgXcQ",
                        "snippet": {
                            "title": "Example",
                            "channelId": "UC123",
                            "channelTitle": "Channel",
                            "publishedAt": "2020-01-02T03:04:05Z",
                            "categoryId": "22",
                            "defaultLanguage": "en",
                            "tags": ["a", "b"],
                            "description": "desc",
                        },
                        "contentDetails": {"duration": "PT1M2S", "caption": "true"},
                        "statistics": {"viewCount": "1234"},
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = YouTubeMetadataClient("test-key", client=client)
    try:
        result = await provider.fetch(["dQw4w9WgXcQ"])
    finally:
        await client.aclose()
    item = result["dQw4w9WgXcQ"]
    assert item.duration_seconds == 62
    assert item.view_count == 1234
    assert item.captions_available is True
    assert item.default_language == "en"


def test_http_errors_do_not_persist_api_keys() -> None:
    request = httpx.Request("GET", "https://example.test/videos?key=secret-key")
    response = httpx.Response(403, request=request)
    error = httpx.HTTPStatusError("forbidden", request=request, response=response)

    message = _error_message(error)

    assert message == "HTTPStatusError: HTTP 403"
    assert "secret-key" not in message


def test_public_transcript_helpers_preserve_caption_text() -> None:
    tracks = [
        {
            "baseUrl": "https://example.test/captions?id=x",
            "languageCode": "en",
            "kind": "asr",
        },
        {
            "baseUrl": "https://example.test/captions?id=y",
            "languageCode": "fr",
        },
    ]
    html = '<script>var x={"captionTracks":' + json.dumps(tracks) + "};</script>"
    parsed = PublicTranscriptClient._caption_tracks(html)
    selected = PublicTranscriptClient._select_track(parsed, "fr-FR")
    assert selected is not None and selected["languageCode"] == "fr"
    text = PublicTranscriptClient._text_from_json3(
        {"events": [{"segs": [{"utf8": "hello "}, {"utf8": "world"}]}, {"segs": [{"utf8": "next"}]}]}
    )
    assert text == "hello world\nnext"
    with pytest.raises(TranscriptPageError):
        PublicTranscriptClient._caption_tracks("<html>no caption payload</html>")


@pytest.mark.asyncio
async def test_public_transcript_client_stops_on_rate_limit() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = PublicTranscriptClient(client=client)
    try:
        with pytest.raises(TranscriptRateLimited):
            await provider.fetch("dQw4w9WgXcQ")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_public_transcript_client_stops_on_block() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = PublicTranscriptClient(client=client)
    try:
        with pytest.raises(TranscriptBlocked):
            await provider.fetch("dQw4w9WgXcQ")
    finally:
        await client.aclose()


def test_public_transcript_rate_has_human_plausible_ceiling() -> None:
    with pytest.raises(ValueError, match="between 1 and 240"):
        PublicTranscriptClient(max_requests_per_hour=241)
