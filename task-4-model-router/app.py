"""HTTP surface for the rate-limited, failover-capable LLM gateway.

Per request: authenticate the tenant, estimate the token cost, reserve it against
the sliding window, route to a provider (failing over if needed), then reconcile
the reservation against what the provider actually charged.

Every failure - rate limit, bad request, dead providers, internal bug - leaves by
the same door: :meth:`GatewayError.to_payload`.

Run it with ``python app.py`` (or ``uvicorn app:app``).
"""

from __future__ import annotations

import logging
import math
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Final

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from router import (
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOKEN_LIMIT,
    DEFAULT_WINDOW_SECONDS,
    GatewayError,
    ModelRouter,
    build_router,
    estimate_tokens,
)
from store import Reservation, TokenLedger

try:  # .env is a convenience for local runs, not a hard dependency
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger("model-gateway")

DEFAULT_DB_PATH: Final = "gateway.db"


class Message(BaseModel):
    """One chat message."""

    model_config = ConfigDict(extra="allow")

    role: str = Field(min_length=1)
    content: Any = None


class CompletionRequest(BaseModel):
    """The request body. ``extra="allow"`` so provider-specific params pass through."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0, le=32_000)


def _authenticate(request: Request) -> str:
    """Resolve the tenant API key from the Authorization header.

    With ``GATEWAY_API_KEYS`` set, only listed keys are accepted. Unset, any
    non-empty bearer is treated as its own tenant - handy for a demo, and the
    single place to swap in real authentication.
    """
    header = request.headers.get("authorization", "")
    scheme, separator, token = header.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        raise GatewayError(
            code="missing_credentials",
            message="An 'Authorization: Bearer <api-key>' header is required.",
            status_code=401,
            error_type="authentication_error",
        )

    api_key = token.strip()
    allowlist = [key.strip() for key in os.getenv("GATEWAY_API_KEYS", "").split(",") if key.strip()]
    if allowlist and api_key not in allowlist:
        raise GatewayError(
            code="invalid_api_key",
            message="The supplied API key is not recognised.",
            status_code=401,
            error_type="authentication_error",
        )
    return api_key


def _error_response(error: GatewayError, request_id: str, extra_headers: dict[str, str] | None = None) -> JSONResponse:
    """Render a GatewayError, adding Retry-After when there is one."""
    headers = {"X-Request-Id": request_id, **(extra_headers or {})}
    if error.retry_after_ms is not None:
        headers["Retry-After"] = str(max(1, math.ceil(error.retry_after_ms / 1000)))
    return JSONResponse(status_code=error.status_code, content=error.to_payload(request_id), headers=headers)


def _rate_limit_headers(limit_tokens: int, remaining_tokens: int, window_seconds: int) -> dict[str, str]:
    """Standard rate-limit headers, so clients can pace themselves."""
    return {
        "X-RateLimit-Limit": str(limit_tokens),
        "X-RateLimit-Remaining": str(remaining_tokens),
        "X-RateLimit-Reset": str(window_seconds),
    }


def create_app(ledger: TokenLedger | None = None, router: ModelRouter | None = None) -> FastAPI:
    """Build the gateway. Both dependencies are injectable for testing."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Open the on-disk ledger and the shared HTTP client."""
        client = httpx.AsyncClient(timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS))
        app.state.http_client = client
        app.state.ledger = ledger or TokenLedger(
            database_path=os.getenv("RATE_LIMIT_DB", DEFAULT_DB_PATH),
            limit_tokens=int(os.getenv("RATE_LIMIT_TOKENS", str(DEFAULT_TOKEN_LIMIT))),
            window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", str(DEFAULT_WINDOW_SECONDS))),
        )
        await app.state.ledger.open()
        app.state.router = router or build_router(client)
        logger.info(
            "model gateway ready; primary=%s fallback=%s limit=%d/%ds",
            app.state.router.primary.name,
            app.state.router.secondary.name if app.state.router.secondary else "none",
            app.state.ledger.limit_tokens,
            app.state.ledger.window_seconds,
        )
        try:
            yield
        finally:
            await client.aclose()
            if ledger is None:  # only close what we opened
                await app.state.ledger.close()

    app = FastAPI(title="LLM Gateway - rate limiting & model failover", version="1.0.0", lifespan=lifespan)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        """Rate-limit, route, reconcile."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        ledger_: TokenLedger = app.state.ledger
        reservation: Reservation | None = None
        prompt_estimate = 0

        try:
            api_key = _authenticate(request)

            try:
                body = await request.json()
            except (ValueError, UnicodeDecodeError) as exc:
                raise GatewayError(
                    code="invalid_json",
                    message="Request body is not valid JSON.",
                    status_code=400,
                    error_type="invalid_request_error",
                ) from exc

            try:
                parsed = CompletionRequest.model_validate(body)
            except ValidationError as exc:
                first = exc.errors(include_url=False)[0]
                field = ".".join(str(part) for part in first["loc"]) or "<root>"
                raise GatewayError(
                    code="invalid_request",
                    message=f"Invalid request: {field}: {first['msg']}",
                    status_code=400,
                    error_type="invalid_request_error",
                ) from exc

            payload = parsed.model_dump(exclude_none=True)
            estimate = estimate_tokens(payload)
            prompt_estimate = estimate.prompt

            reservation = await ledger_.reserve(api_key, estimate.total, request_id=request_id)
            if not reservation.allowed:
                raise GatewayError(
                    code="rate_limit_exceeded",
                    message=(
                        f"Token budget exhausted: {reservation.limit_tokens} tokens per "
                        f"{ledger_.window_seconds}s for this API key."
                    ),
                    status_code=429,
                    error_type="rate_limit_error",
                    retry_after_ms=reservation.retry_after_ms,
                )

            result = await app.state.router.complete(payload, request_id)

        except GatewayError as error:
            extra_headers: dict[str, str] = {}
            if reservation is not None:
                if reservation.allowed:
                    # Refund down to the prompt estimate *before* reading back the
                    # headers below: the provider did some work, but the client got
                    # no completion, and a full refund would let a client hammer a
                    # failing provider for free. Reconciling first (rather than
                    # reporting the pre-refund reservation, as an earlier version of
                    # this handler did) keeps X-RateLimit-Remaining in sync with
                    # what /v1/usage would report a moment later.
                    await ledger_.reconcile(reservation, prompt_estimate)
                    fresh_usage = await ledger_.usage(api_key)
                    extra_headers = _rate_limit_headers(
                        fresh_usage.limit_tokens, fresh_usage.remaining_tokens, fresh_usage.window_seconds
                    )
                else:
                    # Denied outright: nothing was reserved, so the reservation's own
                    # numbers are already the current state - no re-read needed.
                    extra_headers = _rate_limit_headers(
                        reservation.limit_tokens, reservation.remaining_tokens, ledger_.window_seconds
                    )
            logger.info("[%s] returning %s (%d)", request_id, error.code, error.status_code)
            return _error_response(error, request_id, extra_headers)
        except Exception:
            if reservation is not None and reservation.allowed:
                await ledger_.release(reservation)
            logger.exception("[%s] unhandled gateway error", request_id)
            return _error_response(
                GatewayError("internal_error", "The gateway encountered an internal error.", 500), request_id
            )

        # Correct the estimate against what the provider actually charged.
        await ledger_.reconcile(reservation, result.response.total_tokens)
        usage = await ledger_.usage(api_key)

        return JSONResponse(
            content={
                "id": f"cmpl-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": result.response.model,
                "provider": result.provider,
                "failed_over": result.failed_over,
                "failover_reason": result.failover_reason,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.response.content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": result.response.prompt_tokens,
                    "completion_tokens": result.response.completion_tokens,
                    "total_tokens": result.response.total_tokens,
                },
            },
            headers={
                "X-Request-Id": request_id,
                "X-Gateway-Provider": result.provider,
                "X-Gateway-Failover": str(result.failed_over).lower(),
                "X-Gateway-Latency-Ms": f"{result.latency_ms:.1f}",
                **_rate_limit_headers(usage.limit_tokens, usage.remaining_tokens, usage.window_seconds),
            },
        )

    @app.get("/v1/usage")
    async def usage_endpoint(request: Request) -> Response:
        """Current window usage for the calling key."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        try:
            api_key = _authenticate(request)
        except GatewayError as error:
            return _error_response(error, request_id)

        usage = await app.state.ledger.usage(api_key)
        return JSONResponse(
            {
                "limit_tokens": usage.limit_tokens,
                "used_tokens": usage.used_tokens,
                "remaining_tokens": usage.remaining_tokens,
                "requests_in_window": usage.requests,
                "window_seconds": usage.window_seconds,
            }
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness plus the routing and limiter configuration."""
        router_: ModelRouter = app.state.router
        ledger_: TokenLedger = app.state.ledger
        return {
            "status": "ok",
            "primary": router_.primary.name,
            "fallback": router_.secondary.name if router_.secondary else None,
            "timeout_ms": int(router_.timeout_seconds * 1000),
            "limit_tokens": ledger_.limit_tokens,
            "window_seconds": ledger_.window_seconds,
            "database": str(ledger_.database_path),
        }

    return app


app = create_app()


def main() -> None:
    """Serve the gateway with uvicorn."""
    import uvicorn

    logging.basicConfig(
        level=os.getenv("GATEWAY_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    uvicorn.run(app, host=os.getenv("GATEWAY_HOST", "127.0.0.1"), port=int(os.getenv("GATEWAY_PORT", "8100")))


if __name__ == "__main__":
    main()
