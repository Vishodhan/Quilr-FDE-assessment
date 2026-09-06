"""LLM gateway with a streaming PII guardrail.

Exposes an OpenAI-compatible ``POST /v1/chat/completions``. When ``stream`` is
true the upstream SSE body is parsed frame by frame, each ``delta.content`` is
pushed through a :class:`~redactor.StreamingRedactor`, and the redacted text is
re-emitted immediately - nothing waits for the end of the response.

The guardrail holds back only the tail that could still turn out to be PII
(usually a word or two, never more than ``max_holdback`` characters), so Time To
First Token stays close to the upstream's and memory does not grow with the
length of the response.

Run it with ``python llm_gateway.py`` (or ``uvicorn llm_gateway:app``).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Final

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from providers import DONE_SENTINEL, LLMProvider, build_provider
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from redactor import DEFAULT_MAX_HOLDBACK, REPLACEMENT, StreamingRedactor

try:  # .env is a convenience for local runs, not a hard dependency
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger("llm-guardrail-gateway")

SSE_MEDIA_TYPE: Final = "text/event-stream"
# Proxies love to buffer event streams; these tell the common ones not to.
SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class Message(BaseModel):
    """One chat message. Content may be a string or a multipart list."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None


class ChatCompletionRequest(BaseModel):
    """The subset of the chat-completions body this gateway cares about.

    ``extra="allow"`` so temperature, tools, and anything else the caller sends
    reaches the provider untouched.
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    stream: bool = False


def _sse(payload: str) -> bytes:
    """Frame one SSE ``data:`` line."""
    return f"data: {payload}\n\n".encode()


def _error_payload(message: str, error_type: str, status_code: int) -> JSONResponse:
    """OpenAI-shaped error body, so existing clients can parse it."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
    )


def transform_chunk(chunk: dict[str, Any], redactor: StreamingRedactor) -> dict[str, Any] | None:
    """Rewrite one upstream chunk in place, returning None if it has nothing left to say.

    A chunk whose entire content is still held back would otherwise become an
    empty delta on the wire, which is pure noise for the client.
    """
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return chunk  # usage-only or keep-alive frames pass straight through

    choice = choices[0]
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return chunk

    finished = choice.get("finish_reason") is not None
    text = delta.get("content")

    emitted = redactor.feed(text) if isinstance(text, str) else ""
    if finished:
        # Last chance: nothing may stay in the buffer once the turn is over.
        emitted += redactor.flush()

    if emitted or text is not None:
        delta["content"] = emitted

    if not emitted and not finished and set(delta) <= {"content"}:
        return None
    return chunk


def create_app(provider: LLMProvider | None = None, max_holdback: int = DEFAULT_MAX_HOLDBACK) -> FastAPI:
    """Build the gateway. ``provider`` is injectable so tests can pin the upstream."""
    totals: Counter[str] = Counter()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Hold one pooled HTTP client, and pick the provider once."""
        client = httpx.AsyncClient()
        app.state.http_client = client
        app.state.provider = provider or build_provider(client)
        logger.info("LLM guardrail gateway ready; provider=%s", app.state.provider.name)
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(title="LLM Gateway - streaming PII guardrail", version="1.0.0", lifespan=lifespan)

    async def _guarded_stream(payload: dict[str, Any], request_id: str) -> AsyncIterator[bytes]:
        """Relay the upstream stream, redacting each delta as it goes."""
        redactor = StreamingRedactor(max_holdback=max_holdback)
        started = time.perf_counter()
        first_token_at: float | None = None
        template: dict[str, Any] | None = None
        saw_finish = False
        failure: dict[str, Any] | None = None

        # The outer finally only logs. Yielding from it would break on client
        # disconnect, when GeneratorExit is thrown at the pending yield.
        try:
            try:
                async for data in app.state.provider.stream(payload):
                    if data == DONE_SENTINEL:
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("[%s] dropping unparseable upstream frame", request_id)
                        continue
                    if not isinstance(chunk, dict):
                        continue

                    if template is None:
                        template = {key: chunk.get(key) for key in ("id", "model", "created")}
                    choices = chunk.get("choices")
                    if isinstance(choices, list) and choices and choices[0].get("finish_reason") is not None:
                        saw_finish = True

                    outgoing = transform_chunk(chunk, redactor)
                    if outgoing is None:
                        continue  # everything in this delta is still held back
                    if first_token_at is None and _carries_content(outgoing):
                        first_token_at = time.perf_counter()
                    yield _sse(json.dumps(outgoing))

                # An upstream that stopped without a finish_reason still must not
                # leave buffered text unsent.
                if not saw_finish:
                    tail = redactor.flush()
                    if tail:
                        yield _sse(json.dumps(_tail_chunk(template, tail)))

            except httpx.HTTPStatusError as exc:
                logger.warning("[%s] upstream returned HTTP %s", request_id, exc.response.status_code)
                failure = {"error": {"message": "Upstream provider error", "type": "upstream_error"}}
            except httpx.RequestError as exc:
                logger.warning("[%s] upstream unreachable (%s)", request_id, type(exc).__name__)
                failure = {"error": {"message": "Upstream provider unavailable", "type": "upstream_error"}}
            except Exception:
                logger.exception("[%s] guardrail stream failed", request_id)
                failure = {"error": {"message": "Gateway error", "type": "gateway_error"}}

            if failure is not None:
                yield _sse(json.dumps(failure))
            yield b"data: [DONE]\n\n"
        finally:
            totals.update(redactor.counts)
            totals["streams"] += 1
            ttft_ms = (first_token_at - started) * 1000 if first_token_at is not None else -1.0
            logger.info(
                "[%s] stream done: redactions=%s ttft=%.1fms held_back=%d",
                request_id,
                dict(redactor.counts) or "none",
                ttft_ms,
                redactor.pending,
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        """Chat completions, streaming or not, with the guardrail applied either way."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]

        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return _error_payload("Request body is not valid JSON", "invalid_request_error", 400)

        try:
            parsed = ChatCompletionRequest.model_validate(body)
        except ValidationError as exc:
            first = exc.errors(include_url=False)[0]
            field = ".".join(str(part) for part in first["loc"]) or "<root>"
            return _error_payload(f"Invalid request: {field}: {first['msg']}", "invalid_request_error", 400)

        payload = parsed.model_dump(exclude_none=False)

        if parsed.stream:
            return StreamingResponse(
                _guarded_stream(payload, request_id),
                media_type=SSE_MEDIA_TYPE,
                headers={**SSE_HEADERS, "X-Request-Id": request_id},
            )

        # Non-streaming: same redactor, applied to the whole message at once.
        redactor = StreamingRedactor(max_holdback=max_holdback)
        try:
            completion = await app.state.provider.complete(payload)
        except httpx.HTTPStatusError as exc:
            logger.warning("[%s] upstream returned %s", request_id, exc.response.status_code)
            return _error_payload("Upstream provider error", "upstream_error", 502)
        except httpx.RequestError as exc:
            logger.warning("[%s] upstream unreachable (%s)", request_id, type(exc).__name__)
            return _error_payload("Upstream provider unavailable", "upstream_error", 502)

        for choice in completion.get("choices") or []:
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                message["content"] = redactor.scrub(message["content"])

        totals.update(redactor.counts)
        totals["completions"] += 1
        return JSONResponse(
            content=completion,
            headers={"X-Request-Id": request_id, "X-Guardrail-Redactions": str(redactor.total_redactions)},
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness, the active provider, and lifetime redaction counts."""
        return {
            "status": "ok",
            "provider": getattr(app.state, "provider", None).name if hasattr(app.state, "provider") else None,
            "replacement": REPLACEMENT,
            "max_holdback": max_holdback,
            "totals": dict(totals),
        }

    return app


def _carries_content(chunk: dict[str, Any]) -> bool:
    """True when a chunk actually delivers text, as opposed to a role or finish marker."""
    choices = chunk.get("choices") or []
    return bool(choices and (choices[0].get("delta") or {}).get("content"))


def _tail_chunk(template: dict[str, Any] | None, text: str) -> dict[str, Any]:
    """Wrap flushed tail text in a chunk shaped like the ones around it."""
    base = template or {}
    return {
        "id": base.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": base.get("created") or int(time.time()),
        "model": base.get("model") or "unknown",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }


app = create_app()


def main() -> None:
    """Serve the gateway with uvicorn."""
    import uvicorn

    logging.basicConfig(
        level=os.getenv("GATEWAY_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    uvicorn.run(app, host=os.getenv("GATEWAY_HOST", "127.0.0.1"), port=int(os.getenv("GATEWAY_PORT", "8000")))


if __name__ == "__main__":
    main()
