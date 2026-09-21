from __future__ import annotations

import json

import httpx
import pytest

from yafg.experiment.schema import LLMConfig
from yafg.llm.base import Candidate
from yafg.llm.providers import ClaudeProvider, OllamaProvider


def test_claude_provider_requires_key() -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        ClaudeProvider(LLMConfig(provider="claude", model="test"), api_key=None)


@pytest.mark.asyncio
async def test_ollama_structured_choice_and_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["format"]["properties"]["rank"]["type"] == "integer"
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "model": "local-test",
                "message": {"content": json.dumps({"rank": 1, "justification": "Relevant"})},
                "prompt_eval_count": 21,
                "eval_count": 7,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = OllamaProvider(LLMConfig(provider="ollama", model="local-test"), client=client)
        choice, usage = await provider.choose(
            system_prompt="persona",
            history=["older video"],
            candidates=[Candidate(rank=1, video_id="aaaaaaaaaaa", title="Candidate")],
        )
    finally:
        await client.aclose()

    assert choice.rank == 1
    assert choice.justification == "Relevant"
    assert usage.input_tokens == 21
    assert usage.output_tokens == 7
    assert usage.model == "local-test"
