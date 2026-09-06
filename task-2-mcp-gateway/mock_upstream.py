"""Downstream mock MCP server that the gateway proxies to.

Stands in for a real MCP server over HTTP/JSON-RPC. It intentionally applies **no**
authorization of its own - that is the gateway's job, and leaving it out is what
makes "the downstream server was never contacted" a meaningful assertion.

Every request is recorded, so tests (and a human poking at it with curl) can
confirm exactly what did and did not reach it:

    GET  /__stats        -> what arrived
    POST /__stats/reset  -> clear the log

Run it with ``python mock_upstream.py`` (defaults to port 9001).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("mock-upstream")

PROTOCOL_VERSION: Final = "2025-06-18"

TOOLS: Final[list[dict[str, Any]]] = [
    {
        "name": "get_customer_record",
        "description": "Read a customer record.",
        "inputSchema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_docs",
        "description": "Full-text search over the knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "stream_echo",
        "description": "Echo a message back as a server-sent event stream.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
    },
    {
        "name": "admin_reset_key",
        "description": "Rotate a tenant's API key. Privileged.",
        "inputSchema": {
            "type": "object",
            "properties": {"tenant_id": {"type": "string"}},
            "required": ["tenant_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "admin_delete_tenant",
        "description": "Delete a tenant and all of its data. Privileged.",
        "inputSchema": {
            "type": "object",
            "properties": {"tenant_id": {"type": "string"}},
            "required": ["tenant_id"],
            "additionalProperties": False,
        },
    },
]

_TOOL_NAMES: Final = {tool["name"] for tool in TOOLS}


def fingerprint(value: str) -> str:
    """Short, non-reversible label for a header value, so logs never hold a credential."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


class CallLog:
    """In-memory record of everything that reached this server."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def record(self, method: str, tool: str | None, headers: dict[str, str]) -> None:
        """Note one inbound request and the gateway headers that came with it.

        The Authorization value is fingerprinted rather than stored, so the log
        can be asserted on without holding a credential in memory.
        """
        authorization = headers.get("authorization")
        self._entries.append(
            {
                "method": method,
                "tool": tool,
                "gateway_role": headers.get("x-mcp-gateway-role"),
                "gateway_principal": headers.get("x-mcp-gateway-principal"),
                "authorization_fingerprint": fingerprint(authorization) if authorization else None,
                "request_id": headers.get("x-request-id"),
            }
        )

    def snapshot(self) -> dict[str, Any]:
        """Everything seen so far, plus the counts tests assert on."""
        return {
            "total_requests": len(self._entries),
            "tools_called": [entry["tool"] for entry in self._entries if entry["tool"]],
            "entries": list(self._entries),
        }

    def reset(self) -> None:
        self._entries.clear()


def _result(request_id: Any, result: Any) -> JSONResponse:
    """JSON-RPC success envelope."""
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id: Any, code: int, message: str) -> JSONResponse:
    """JSON-RPC error envelope."""
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


async def _sse_body(message: str) -> AsyncIterator[bytes]:
    """Emit the echoed message one word per event, to exercise stream passthrough."""
    for index, word in enumerate(message.split()):
        frame = json.dumps({"index": index, "text": word})
        yield f"event: chunk\ndata: {frame}\n\n".encode()
    yield b"event: done\ndata: [DONE]\n\n"


def create_upstream_app(call_log: CallLog | None = None) -> FastAPI:
    """Build the mock upstream. Tests inject a CallLog to inspect it directly."""
    log = call_log if call_log is not None else CallLog()
    app = FastAPI(title="Mock downstream MCP server", version="1.0.0")
    app.state.call_log = log

    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        """Handle one JSON-RPC frame."""
        try:
            payload = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(None, -32700, "Parse error")
        if not isinstance(payload, dict):
            return _error(None, -32600, "Invalid Request")

        method = payload.get("method", "")
        request_id = payload.get("id")
        params = payload.get("params") or {}
        tool_name = params.get("name") if method == "tools/call" else None
        log.record(method, tool_name, dict(request.headers))
        logger.info("upstream received %s %s", method, tool_name or "")

        if request_id is None:
            # A notification gets no JSON-RPC body, only an acknowledgement.
            return Response(status_code=202)

        if method == "initialize":
            return _result(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "mock-downstream-mcp", "version": "1.0.0"},
                },
            )

        if method == "ping":
            return _result(request_id, {})

        if method == "tools/list":
            return _result(request_id, {"tools": TOOLS})

        if method == "tools/call":
            if tool_name not in _TOOL_NAMES:
                return _error(request_id, -32602, f"Unknown tool: {tool_name!r}")

            if tool_name == "stream_echo":
                message = params.get("arguments", {}).get("message", "")
                return StreamingResponse(_sse_body(str(message)), media_type="text/event-stream")

            arguments = params.get("arguments", {})
            return _result(
                request_id,
                {
                    "content": [{"type": "text", "text": f"{tool_name} executed with {json.dumps(arguments)}"}],
                    "structuredContent": {"tool": tool_name, "arguments": arguments, "ok": True},
                    "isError": False,
                },
            )

        return _error(request_id, -32601, f"Method not found: {method!r}")

    @app.get("/__stats")
    async def stats() -> dict[str, Any]:
        """What actually reached this server."""
        return log.snapshot()

    @app.post("/__stats/reset")
    async def reset_stats() -> dict[str, str]:
        """Clear the call log between test cases."""
        log.reset()
        return {"status": "reset"}

    return app


app = create_upstream_app()


def main() -> None:
    """Serve the mock upstream with uvicorn."""
    import uvicorn

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    uvicorn.run(app, host=os.getenv("UPSTREAM_HOST", "127.0.0.1"), port=int(os.getenv("UPSTREAM_PORT", "9001")))


if __name__ == "__main__":
    main()
