"""Upstream LLM providers for the guardrail gateway.

Two implementations behind one interface:

* :class:`MockProvider`   - deterministic, offline, and deliberately unkind about
  where it splits chunks. The default, so the gateway and its tests run with no
  API key and no network.
* :class:`UpstreamProvider` - any OpenAI-compatible ``/chat/completions`` endpoint.

Both yield the *payload* of each SSE ``data:`` line, ending with the literal
``[DONE]``. Parsing the deltas is the gateway's job; owning the transport format
is theirs.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Final, Protocol

import httpx

DONE_SENTINEL: Final = "[DONE]"

# Canned reply used when the prompt does not ask for something specific. It carries
# one of each pattern so a manual curl shows the guardrail working.
DEFAULT_MOCK_REPLY: Final = (
    "Sure - I found the account. The contact address is ada.lovelace@example.com, "
    "the SSN on file is 123-45-6789, and the card ending in 1111 is 4111 1111 1111 1111. "
    "Let me know if you need anything else."
)


class LLMProvider(Protocol):
    """What the gateway needs from an upstream model provider."""

    name: str

    def stream(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Yield each SSE data payload, finishing with ``[DONE]``."""
        ...

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return one non-streaming chat completion."""
        ...


def _chunk_frame(completion_id: str, model: str, created: int, delta: dict[str, Any], finish_reason: str | None) -> str:
    """Serialise one ``chat.completion.chunk`` the way an OpenAI-compatible API would."""
    return json.dumps(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


def _reply_for(payload: dict[str, Any]) -> str:
    """Pick the mock's reply.

    A user message of ``echo: <text>`` returns ``<text>`` verbatim, which is what
    lets tests drive exact content through the gateway.
    """
    messages = payload.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.startswith("echo:"):
                return content[len("echo:") :].lstrip()
            break
    return DEFAULT_MOCK_REPLY


class MockProvider:
    """Offline provider that splits its reply at fixed character counts.

    Splitting on a character count rather than on word boundaries is deliberate:
    it guarantees emails, SSNs and card numbers land across chunk boundaries,
    which is the case the guardrail exists for.
    """

    name = "mock"

    def __init__(self, chunk_size: int = 4, delay_seconds: float = 0.0) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1 character")
        self.chunk_size = chunk_size
        self.delay_seconds = max(0.0, delay_seconds)

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Emit the reply as chat.completion.chunk frames."""
        reply = _reply_for(payload)
        completion_id = f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"
        model = str(payload.get("model", "mock-model"))
        created = int(time.time())

        yield _chunk_frame(completion_id, model, created, {"role": "assistant", "content": ""}, None)
        for start in range(0, len(reply), self.chunk_size):
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            piece = reply[start : start + self.chunk_size]
            yield _chunk_frame(completion_id, model, created, {"content": piece}, None)

        yield _chunk_frame(completion_id, model, created, {}, "stop")
        yield DONE_SENTINEL

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return the same reply as a single non-streaming completion."""
        reply = _reply_for(payload)
        return {
            "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model", "mock-model")),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(reply.split()), "total_tokens": len(reply.split())},
        }


class UpstreamProvider:
    """Any OpenAI-compatible chat-completions endpoint."""

    name = "upstream"

    def __init__(self, base_url: str, api_key: str, client: httpx.AsyncClient, timeout_seconds: float = 60.0) -> None:
        if not api_key:
            raise ValueError("An API key is required for the upstream provider")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client
        self._timeout = timeout_seconds

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Relay the upstream SSE body one data payload at a time."""
        request = self._client.build_request(
            "POST",
            f"{self.base_url}/chat/completions",
            json={**payload, "stream": True},
            headers=self._headers,
            timeout=self._timeout,
        )
        response = await self._client.send(request, stream=True)
        try:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    yield line[len("data:") :].strip()
        finally:
            await response.aclose()

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Forward a single non-streaming completion."""
        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            json={**payload, "stream": False},
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()


def build_provider(client: httpx.AsyncClient) -> LLMProvider:
    """Choose a provider from the environment.

    ``LLM_PROVIDER=upstream`` needs ``LLM_API_KEY``; anything else gives the mock,
    so the gateway is runnable straight after clone.
    """
    if os.getenv("LLM_PROVIDER", "mock").strip().lower() != "upstream":
        return MockProvider(
            chunk_size=int(os.getenv("MOCK_CHUNK_SIZE", "4")),
            delay_seconds=float(os.getenv("MOCK_CHUNK_DELAY_MS", "0")) / 1000.0,
        )

    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        raise ValueError("LLM_PROVIDER=upstream requires LLM_API_KEY to be set")
    return UpstreamProvider(
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key,
        client=client,
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
    )
