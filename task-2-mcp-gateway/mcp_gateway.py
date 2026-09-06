"""MCP security gateway: an HTTP/JSON-RPC reverse proxy with method-level authorization.

Sits between an AI agent client and a downstream MCP server:

  agent --Bearer token--> gateway --(policy)--> downstream MCP server

``tools/list`` and every non-tool method are forwarded untouched. ``tools/call`` is
inspected: a tool named ``admin_*`` requires the admin role, and an unauthorized
attempt is short-circuited with JSON-RPC ``-32001 Unauthorized Tool Call`` without
the downstream server ever being contacted.

Run it with ``python mcp_gateway.py`` (or ``uvicorn mcp_gateway:app``).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Final

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from policy import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    PARSE_ERROR,
    UNAUTHORIZED_TOOL_CALL,
    JsonRpcProtocolError,
    JsonRpcRequest,
    Principal,
    Role,
    TokenRegistry,
    authorize,
    error_body,
    extract_bearer_token,
    parse_jsonrpc,
)

try:  # .env is a convenience for local runs, not a hard dependency
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger("mcp-gateway")

# Headers that describe a single hop and must not be relayed onward.
_HOP_BY_HOP: Final = frozenset(
    {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}
)
# The client's own credential stops here: the gateway presents its own to the upstream.
_STRIP_FROM_REQUEST: Final = _HOP_BY_HOP | {"host", "content-length", "authorization"}
# httpx has already decoded the body, so the upstream's content-encoding no longer applies.
_STRIP_FROM_RESPONSE: Final = _HOP_BY_HOP | {"content-length", "content-encoding"}

# A well-formed request refused on policy grounds is still a 200 with a JSON-RPC error
# body: MCP clients raise a transport error on non-2xx and would never surface -32001.
_HTTP_STATUS_FOR_CODE: Final[dict[int, int]] = {
    PARSE_ERROR: 400,
    INVALID_REQUEST: 400,
    INVALID_PARAMS: 200,
    UNAUTHORIZED_TOOL_CALL: 200,
    INTERNAL_ERROR: 502,
}


@dataclass(frozen=True)
class Settings:
    """Gateway configuration, all overridable by environment variable."""

    upstream_url: str = "http://127.0.0.1:9001/mcp"
    upstream_token: str | None = None
    listen_path: str = "/mcp"
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> Settings:
        """Read settings from the environment, falling back to the defaults above."""
        try:
            timeout = float(os.getenv("GATEWAY_UPSTREAM_TIMEOUT", "30"))
        except ValueError as exc:
            raise ValueError("GATEWAY_UPSTREAM_TIMEOUT must be a number of seconds") from exc
        if timeout <= 0:
            raise ValueError("GATEWAY_UPSTREAM_TIMEOUT must be greater than zero")

        return cls(
            upstream_url=os.getenv("GATEWAY_UPSTREAM_URL", cls.upstream_url),
            upstream_token=os.getenv("GATEWAY_UPSTREAM_TOKEN") or None,
            listen_path=os.getenv("GATEWAY_LISTEN_PATH", cls.listen_path),
            timeout_seconds=timeout,
        )


def _error_response(
    error: JsonRpcProtocolError,
    status_code: int | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Render a JsonRpcProtocolError as a JSON-RPC error response."""
    return JSONResponse(
        status_code=status_code if status_code is not None else _HTTP_STATUS_FOR_CODE.get(error.code, 400),
        content=error_body(error.code, error.message, error.request_id, error.data),
        headers=headers,
    )


def create_app(
    settings: Settings | None = None,
    registry: TokenRegistry | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Build the gateway application.

    ``http_client`` is injectable so tests can point the proxy at an in-process
    upstream (or a fault-injecting transport) without opening a socket. A client
    passed in here is owned by the caller and is not closed on shutdown.
    """
    settings = settings or Settings.from_env()
    registry = registry or TokenRegistry.from_env()
    counters: Counter[str] = Counter()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Hold one pooled HTTP client for the process lifetime."""
        owns_client = http_client is None
        client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(settings.timeout_seconds))
        app.state.http_client = client
        logger.info("MCP gateway ready; upstream=%s", settings.upstream_url)
        try:
            yield
        finally:
            if owns_client:
                await client.aclose()

    app = FastAPI(title="MCP Security Gateway", version="1.0.0", lifespan=lifespan)

    def _request_headers(request: Request, principal: Principal, correlation_id: str) -> dict[str, str]:
        """Copy the client's headers onward, minus the per-hop and credential ones."""
        headers = {name: value for name, value in request.headers.items() if name.lower() not in _STRIP_FROM_REQUEST}
        if settings.upstream_token:
            headers["authorization"] = f"Bearer {settings.upstream_token}"

        # Let the downstream server see who the gateway decided this was.
        headers["x-mcp-gateway-role"] = principal.role.value
        headers["x-mcp-gateway-principal"] = principal.token_id
        headers["x-request-id"] = correlation_id

        client_host = request.client.host if request.client else None
        if client_host:
            existing = request.headers.get("x-forwarded-for")
            headers["x-forwarded-for"] = f"{existing}, {client_host}" if existing else client_host
        return headers

    def _response_headers(headers: httpx.Headers) -> dict[str, str]:
        """Copy the upstream response headers back, minus the per-hop ones."""
        return {name: value for name, value in headers.items() if name.lower() not in _STRIP_FROM_RESPONSE}

    async def _relay(upstream: httpx.Response, correlation_id: str) -> AsyncIterator[bytes]:
        """Pump an SSE body through chunk by chunk so events are not held back."""
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        except httpx.RequestError:
            logger.warning("[%s] upstream stream aborted mid-body", correlation_id)
        finally:
            await upstream.aclose()

    async def _forward(
        request: Request, body: bytes, principal: Principal, rpc: JsonRpcRequest, correlation_id: str
    ) -> Response:
        """Relay the request downstream and hand the response back unchanged."""
        client: httpx.AsyncClient = app.state.http_client
        upstream_request = client.build_request(
            "POST", settings.upstream_url, content=body, headers=_request_headers(request, principal, correlation_id)
        )

        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.TimeoutException:
            counters["upstream_timeout"] += 1
            logger.warning("[%s] upstream timed out after %.1fs", correlation_id, settings.timeout_seconds)
            return _error_response(
                JsonRpcProtocolError(INTERNAL_ERROR, "Upstream MCP server timed out", rpc.request_id), status_code=504
            )
        except httpx.RequestError as exc:
            counters["upstream_unreachable"] += 1
            # Log the class, return a generic message: connection detail is not the client's business.
            logger.warning("[%s] upstream unreachable (%s)", correlation_id, type(exc).__name__)
            return _error_response(
                JsonRpcProtocolError(INTERNAL_ERROR, "Upstream MCP server is unavailable", rpc.request_id), status_code=502
            )

        counters["forwarded"] += 1
        response_headers = _response_headers(upstream.headers)
        content_type = upstream.headers.get("content-type", "")

        if content_type.startswith("text/event-stream"):
            return StreamingResponse(
                _relay(upstream, correlation_id),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=content_type,
            )

        try:
            payload = await upstream.aread()
        except httpx.RequestError:
            logger.warning("[%s] upstream body read failed", correlation_id)
            return _error_response(
                JsonRpcProtocolError(INTERNAL_ERROR, "Upstream MCP server is unavailable", rpc.request_id), status_code=502
            )
        finally:
            await upstream.aclose()

        return Response(content=payload, status_code=upstream.status_code, headers=response_headers)

    async def _handle(request: Request, correlation_id: str) -> Response:
        """Authenticate, parse, authorize, then forward or short-circuit."""
        # 1. Who is calling?
        try:
            token = extract_bearer_token(request.headers.get("authorization"))
            principal = registry.resolve(token)
            if principal is None:
                raise JsonRpcProtocolError(UNAUTHORIZED_TOOL_CALL, "Unknown or revoked bearer token")
        except JsonRpcProtocolError as exc:
            counters["rejected_auth"] += 1
            logger.warning("[%s] authentication rejected: %s", correlation_id, exc.message)
            return _error_response(exc, status_code=401, headers={"WWW-Authenticate": "Bearer"})

        # 2. Is the body a JSON-RPC frame we understand?
        body = await request.body()
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            counters["malformed"] += 1
            return _error_response(JsonRpcProtocolError(PARSE_ERROR, "Request body is not valid JSON"))

        try:
            rpc = parse_jsonrpc(payload)
        except JsonRpcProtocolError as exc:
            counters["malformed"] += 1
            logger.info("[%s] rejected malformed frame: %s", correlation_id, exc.message)
            return _error_response(exc)

        # 3. Is this caller allowed to make this call?
        try:
            decision = authorize(rpc, principal)
        except JsonRpcProtocolError as exc:
            counters["malformed"] += 1
            return _error_response(exc)

        if not decision.allowed:
            counters["denied"] += 1
            logger.warning(
                "[%s] DENY tool=%s principal=%s role=%s: %s",
                correlation_id,
                decision.tool_name,
                principal.token_id,
                principal.role.value,
                decision.reason,
            )
            # Short-circuit: the downstream server is never contacted.
            return _error_response(
                JsonRpcProtocolError(
                    UNAUTHORIZED_TOOL_CALL,
                    "Unauthorized Tool Call",
                    rpc.request_id,
                    data={
                        "tool": decision.tool_name,
                        "required_role": Role.ADMIN.value,
                        "caller_role": principal.role.value,
                    },
                ),
                headers={"x-mcp-gateway-decision": "denied"},
            )

        logger.info(
            "[%s] ALLOW method=%s tool=%s role=%s", correlation_id, rpc.method, decision.tool_name, principal.role.value
        )
        return await _forward(request, body, principal, rpc, correlation_id)

    @app.post(settings.listen_path)
    async def proxy(request: Request) -> Response:
        """Single JSON-RPC entrypoint for agent clients."""
        correlation_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        try:
            response = await _handle(request, correlation_id)
        except Exception:  # a gateway bug must not leak a traceback to the caller
            counters["gateway_error"] += 1
            logger.exception("[%s] unhandled gateway error", correlation_id)
            return _error_response(JsonRpcProtocolError(INTERNAL_ERROR, "Gateway error"), status_code=500)
        response.headers["x-request-id"] = correlation_id
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness plus the running decision counters."""
        return {"status": "ok", "upstream": settings.upstream_url, "counters": dict(counters)}

    return app


app = create_app()


def main() -> None:
    """Serve the gateway with uvicorn."""
    import uvicorn

    logging.basicConfig(
        level=os.getenv("GATEWAY_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    uvicorn.run(
        app,
        host=os.getenv("GATEWAY_HOST", "127.0.0.1"),
        port=int(os.getenv("GATEWAY_PORT", "9000")),
        log_level=os.getenv("GATEWAY_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
