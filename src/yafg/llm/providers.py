"""Concrete LLM providers.

Provider failures are retried once. A malformed or failed response then returns rank 0
with a recorded fallback reason, matching ADR 0004. The agent still validates whether
rank 0 is actually selectable and can move to the first selectable rendered slot.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import httpx
from anthropic.types import MessageParam, ToolChoiceToolParam, ToolParam

from yafg.experiment.schema import LLMConfig
from yafg.llm.base import Candidate, Choice, LLMProvider, Usage
from yafg.settings import settings

_CHOICE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rank", "justification"],
    "properties": {
        "rank": {"type": "integer", "minimum": 0},
        "justification": {"type": "string", "minLength": 1, "maxLength": 600},
        "interest_match": {"type": "string"},
    },
}


def _candidate_payload(candidates: list[Candidate]) -> list[dict[str, Any]]:
    return [candidate.model_dump(mode="json", exclude_none=True) for candidate in candidates]


def _user_message(history: list[str], candidates: list[Candidate]) -> str:
    payload = {
        "recent_selected_videos": history[-12:],
        "candidates": _candidate_payload(candidates),
    }
    return "Choose the next candidate. Return only the structured choice requested by the provider.\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    )


def _fallback(model: str, reason: str) -> tuple[Choice, Usage]:
    clean_reason = reason.strip()[:500] or "provider response could not be parsed"
    return (
        Choice(rank=0, justification=f"LLM fallback to rendered rank 0: {clean_reason}"),
        Usage(model=model, fallback_reason=clean_reason),
    )


class ClaudeProvider:
    """Anthropic Claude via a forced tool call for structured output."""

    def __init__(self, config: LLMConfig, *, api_key: str | None = None) -> None:
        self.config = config
        self.api_key = api_key if api_key is not None else settings.anthropic_api_key
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for llm.provider=claude")
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - installed by the project environment
            raise RuntimeError("anthropic package is not installed") from exc
        self._client = AsyncAnthropic(api_key=self.api_key)

    async def _once(
        self, *, system_prompt: str, history: list[str], candidates: list[Candidate]
    ) -> tuple[Choice, Usage]:
        response = await self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_prompt,
            messages=[MessageParam(role="user", content=_user_message(history, candidates))],
            tools=[
                ToolParam(
                    name="choose_video",
                    description="Choose exactly one rendered candidate rank.",
                    input_schema=_CHOICE_SCHEMA,
                )
            ],
            tool_choice=ToolChoiceToolParam(type="tool", name="choose_video"),
        )
        raw: Mapping[str, Any] | None = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "choose_video":
                value = getattr(block, "input", None)
                if isinstance(value, Mapping):
                    raw = value
                    break
        if raw is None:
            raise ValueError("Claude returned no choose_video tool payload")
        choice = Choice.model_validate(dict(raw))
        response_usage = getattr(response, "usage", None)
        return choice, Usage(
            input_tokens=int(getattr(response_usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(response_usage, "output_tokens", 0) or 0),
            model=self.config.model,
        )

    async def choose(
        self, *, system_prompt: str, history: list[str], candidates: list[Candidate]
    ) -> tuple[Choice, Usage]:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return await self._once(system_prompt=system_prompt, history=history, candidates=candidates)
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(0.5)
        assert last_error is not None
        return _fallback(self.config.model, f"{type(last_error).__name__}: {last_error}")

    async def aclose(self) -> None:
        await self._client.close()


class OllamaProvider:
    """Local Ollama chat API using its JSON-schema structured-output mode."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=60.0)

    async def _once(
        self, *, system_prompt: str, history: list[str], candidates: list[Candidate]
    ) -> tuple[Choice, Usage]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "stream": False,
            "format": _CHOICE_SCHEMA,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _user_message(history, candidates)},
            ],
            "options": {"temperature": self.config.temperature},
        }
        if self.config.seed is not None:
            payload["options"]["seed"] = self.config.seed

        response = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        body = response.json()

        content = body.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama returned no message.content")
        choice = Choice.model_validate_json(content)
        return choice, Usage(
            input_tokens=int(body.get("prompt_eval_count") or 0),
            output_tokens=int(body.get("eval_count") or 0),
            model=str(body.get("model") or self.config.model),
        )

    async def choose(
        self, *, system_prompt: str, history: list[str], candidates: list[Candidate]
    ) -> tuple[Choice, Usage]:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return await self._once(system_prompt=system_prompt, history=history, candidates=candidates)
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(0.5)
        assert last_error is not None
        return _fallback(self.config.model, f"{type(last_error).__name__}: {last_error}")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def make_provider(config: LLMConfig) -> LLMProvider:
    if config.provider == "claude":
        return ClaudeProvider(config)
    if config.provider == "ollama":
        return OllamaProvider(config)
    raise ValueError(f"unsupported LLM provider: {config.provider}")
