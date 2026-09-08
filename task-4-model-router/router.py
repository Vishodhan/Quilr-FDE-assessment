"""Model routing with automatic failover, plus the gateway's error vocabulary.

The primary provider gets one attempt bounded by a hard deadline. If it answers
``429``, blows the deadline, or cannot be reached, the request goes to the
secondary. Whatever happens, the client sees a standardised payload - never an
upstream body, hostname, key or traceback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

import httpx

logger = logging.getLogger("model-router")

DEFAULT_TIMEOUT_SECONDS: Final = 3.0
DEFAULT_TOKEN_LIMIT: Final = 50_000
DEFAULT_WINDOW_SECONDS: Final = 60

# Characters per token for the offline estimate. Roughly right for English, and
# it never needs the network - the real figure is reconciled after the call.
_CHARS_PER_TOKEN: Final = 4


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class GatewayError(Exception):
    """A failure that is safe to hand to the client verbatim."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        error_type: str = "gateway_error",
        retry_after_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.retry_after_ms = retry_after_ms

    def to_payload(self, request_id: str) -> dict[str, Any]:
        """The one error shape this gateway ever emits."""
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "type": self.error_type,
            "request_id": request_id,
        }
        if self.retry_after_ms is not None:
            error["retry_after_ms"] = self.retry_after_ms
        return {"error": error}


class UpstreamRateLimited(Exception):
    """The provider answered 429."""


class UpstreamUnavailable(Exception):
    """The provider could not be reached, or answered with an error status."""


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProviderResponse:
    """What a provider gives back, normalised."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelProvider(Protocol):
    """What the router needs from a model backend."""

    name: str

    async def complete(self, payload: dict[str, Any]) -> ProviderResponse:
        """Run one completion, or raise UpstreamRateLimited / UpstreamUnavailable."""
        ...


class MockProvider:
    """Scriptable provider: the default backend, and what the tests drive.

    ``behaviour`` is one of ``ok``, ``rate_limited``, ``unavailable`` or ``slow``.
    ``slow`` sleeps for ``delay_seconds`` before answering, which is how the
    timeout path gets exercised without a real network.
    """

    def __init__(
        self,
        name: str,
        behaviour: str = "ok",
        delay_seconds: float = 0.0,
        model: str = "mock-model",
        completion_tokens: int = 64,
    ) -> None:
        self.name = name
        self.behaviour = behaviour
        self.delay_seconds = delay_seconds
        self.model = model
        self.completion_tokens = completion_tokens
        self.calls = 0

    async def complete(self, payload: dict[str, Any]) -> ProviderResponse:
        """Answer according to the configured behaviour."""
        self.calls += 1
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)

        if self.behaviour == "rate_limited":
            raise UpstreamRateLimited(f"{self.name} returned 429")
        if self.behaviour == "unavailable":
            raise UpstreamUnavailable(f"{self.name} is unreachable")

        prompt = estimate_tokens(payload).prompt
        return ProviderResponse(
            content=f"[{self.name}] completed request for model {payload.get('model', self.model)!r}",
            model=self.model,
            prompt_tokens=prompt,
            completion_tokens=self.completion_tokens,
        )


class HttpProvider:
    """Any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str, client: httpx.AsyncClient) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._client = client

    async def complete(self, payload: dict[str, Any]) -> ProviderResponse:
        """POST the request, mapping upstream failures onto the router's exceptions."""
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json={**payload, "model": payload.get("model") or self.model, "stream": False},
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            )
        except httpx.RequestError as exc:
            raise UpstreamUnavailable(f"{self.name}: {type(exc).__name__}") from exc

        if response.status_code == 429:
            raise UpstreamRateLimited(f"{self.name} returned 429")
        if response.status_code >= 400:
            raise UpstreamUnavailable(f"{self.name} returned {response.status_code}")

        body = response.json()
        usage = body.get("usage") or {}
        choices = body.get("choices") or [{}]
        message = choices[0].get("message") or {}
        return ProviderResponse(
            content=message.get("content") or "",
            model=body.get("model") or self.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            raw=body,
        )


# --------------------------------------------------------------------------- #
# token estimation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TokenEstimate:
    """Pre-flight guess at what a request will cost."""

    prompt: int
    completion: int

    @property
    def total(self) -> int:
        return self.prompt + self.completion


def estimate_tokens(payload: dict[str, Any], default_completion: int = 256) -> TokenEstimate:
    """Estimate prompt and completion tokens for a request.

    Deliberately offline: roughly four characters per token. tiktoken would be
    more precise but downloads its encoding on first use, and the number is
    corrected against the provider's reported usage right after the call anyway.
    """
    characters = 0
    for message in payload.get("messages") or []:
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                characters += len(content)
            elif isinstance(content, list):  # multipart content
                characters += sum(len(part.get("text", "")) for part in content if isinstance(part, dict))
            characters += len(str(message.get("role", "")))

    requested = payload.get("max_tokens")
    completion = int(requested) if isinstance(requested, int) and requested > 0 else default_completion
    # +1 so an empty prompt still costs something and cannot be spammed for free.
    return TokenEstimate(prompt=characters // _CHARS_PER_TOKEN + 1, completion=completion)


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RoutingResult:
    """A completion, plus how it was obtained."""

    response: ProviderResponse
    provider: str
    failed_over: bool
    failover_reason: str | None
    latency_ms: float


class ModelRouter:
    """Tries the primary, falls back to the secondary on 429, timeout or outage."""

    def __init__(
        self,
        primary: ModelProvider,
        secondary: ModelProvider | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.primary = primary
        self.secondary = secondary
        self.timeout_seconds = timeout_seconds

    async def complete(self, payload: dict[str, Any], request_id: str) -> RoutingResult:
        """Route one request, failing over if the primary cannot serve it."""
        started = time.perf_counter()

        try:
            response = await self._attempt(self.primary, payload)
        except (UpstreamRateLimited, UpstreamUnavailable, TimeoutError) as exc:
            failure_reason = _reason_for(exc, self.timeout_seconds)
            logger.warning("[%s] primary %s failed (%s)", request_id, self.primary.name, failure_reason)
        else:
            # No exception: the primary answered in time, nothing to fail over.
            return RoutingResult(
                response=response,
                provider=self.primary.name,
                failed_over=False,
                failover_reason=None,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if self.secondary is None:
            raise GatewayError(
                code="upstream_unavailable",
                message="The model provider is unavailable and no fallback is configured.",
                status_code=503,
                error_type="upstream_error",
            )

        try:
            response = await self._attempt(self.secondary, payload)
        except (UpstreamRateLimited, UpstreamUnavailable, TimeoutError) as exc:
            logger.warning(
                "[%s] fallback %s also failed (%s)", request_id, self.secondary.name, _reason_for(exc, self.timeout_seconds)
            )
            raise GatewayError(
                code="all_providers_failed",
                message="Every configured model provider failed to serve this request.",
                status_code=503,
                error_type="upstream_error",
            ) from exc

        logger.info("[%s] served by fallback %s after %s", request_id, self.secondary.name, failure_reason)
        return RoutingResult(
            response=response,
            provider=self.secondary.name,
            failed_over=True,
            failover_reason=failure_reason,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def _attempt(self, provider: ModelProvider, payload: dict[str, Any]) -> ProviderResponse:
        """Call a provider under a hard deadline.

        ``asyncio.timeout`` cancels the in-flight call, so a slow provider that
        answers later cannot race ahead of the fallback's response.
        """
        async with asyncio.timeout(self.timeout_seconds):
            return await provider.complete(payload)


def _reason_for(exc: BaseException, timeout_seconds: float) -> str:
    """Short, non-sensitive label for why a provider was abandoned."""
    if isinstance(exc, UpstreamRateLimited):
        return "429 too many requests"
    if isinstance(exc, TimeoutError):
        return f"timeout after {int(timeout_seconds * 1000)}ms"
    return "provider unavailable"


def build_router(client: httpx.AsyncClient) -> ModelRouter:
    """Build the router from the environment.

    Providers default to mocks so the gateway runs straight after clone; set
    ``PRIMARY_API_KEY`` (and optionally ``FALLBACK_API_KEY``) for real ones.
    """
    timeout = float(os.getenv("ROUTER_TIMEOUT_MS", "3000")) / 1000.0

    primary_key = os.getenv("PRIMARY_API_KEY", "")
    if primary_key:
        primary: ModelProvider = HttpProvider(
            name=os.getenv("PRIMARY_NAME", "primary"),
            base_url=os.getenv("PRIMARY_BASE_URL", "https://api.openai.com/v1"),
            api_key=primary_key,
            model=os.getenv("PRIMARY_MODEL", "gpt-4o-mini"),
            client=client,
        )
    else:
        primary = MockProvider(
            name=os.getenv("PRIMARY_NAME", "primary-mock"),
            behaviour=os.getenv("PRIMARY_BEHAVIOUR", "ok"),
            delay_seconds=float(os.getenv("PRIMARY_DELAY_MS", "0")) / 1000.0,
        )

    fallback_key = os.getenv("FALLBACK_API_KEY", "")
    if fallback_key:
        secondary: ModelProvider | None = HttpProvider(
            name=os.getenv("FALLBACK_NAME", "fallback"),
            base_url=os.getenv("FALLBACK_BASE_URL", "https://api.anthropic.com/v1"),
            api_key=fallback_key,
            model=os.getenv("FALLBACK_MODEL", "claude-sonnet-5"),
            client=client,
        )
    else:
        secondary = MockProvider(name=os.getenv("FALLBACK_NAME", "fallback-mock"), behaviour="ok")

    return ModelRouter(primary=primary, secondary=secondary, timeout_seconds=timeout)
