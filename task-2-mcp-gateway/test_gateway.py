"""Tests for the MCP security gateway.

Layered to match the scoring criteria:

1. ``TestWireFormatParsing`` / ``TestTokenRegistry`` / ``TestAuthorizationPolicy``
   - the pure logic, no server in the loop.
2. ``TestProxyForwarding`` / ``TestShortCircuit`` - the gateway end to end against an
   in-process mock upstream, including the assertion that a denied call never
   reaches the downstream server.
3. ``TestUpstreamFailures`` - fault injection, checking errors stay sanitised.

Run from this directory: ``pytest -v``
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

GATEWAY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GATEWAY_DIR))

from mcp_gateway import Settings, create_app  # noqa: E402
from mock_upstream import TOOLS as UPSTREAM_TOOLS  # noqa: E402
from mock_upstream import CallLog, create_upstream_app  # noqa: E402
from mock_upstream import fingerprint as upstream_fingerprint  # noqa: E402
from policy import (  # noqa: E402
    INVALID_PARAMS,
    INVALID_REQUEST,
    PARSE_ERROR,
    UNAUTHORIZED_TOOL_CALL,
    Decision,
    JsonRpcProtocolError,
    JsonRpcRequest,
    Principal,
    Role,
    TokenRegistry,
    authorize,
    extract_bearer_token,
    parse_jsonrpc,
)

ADMIN_TOKEN = "admin-token-abc123"
VIEWER_TOKEN = "viewer-token-xyz789"
UPSTREAM_URL = "http://upstream.test/mcp"

REGISTRY = TokenRegistry({ADMIN_TOKEN: Role.ADMIN, VIEWER_TOKEN: Role.VIEWER})


def rpc(method: str, params: dict[str, Any] | None = None, request_id: Any = 1) -> dict[str, Any]:
    """Build a JSON-RPC request body."""
    body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        body["id"] = request_id
    if params is not None:
        body["params"] = params
    return body


def auth(token: str) -> dict[str, str]:
    """Bearer header for a token."""
    return {"Authorization": f"Bearer {token}"}


@dataclass
class Harness:
    """A gateway wired to an in-process mock upstream."""

    client: httpx.AsyncClient
    log: CallLog

    def upstream_requests(self) -> int:
        """How many requests actually reached the downstream server."""
        return self.log.snapshot()["total_requests"]


@pytest_asyncio.fixture
async def harness() -> AsyncIterator[Harness]:
    """Gateway + upstream, both in-process: no sockets, no port juggling."""
    log = CallLog()
    upstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_upstream_app(log)), base_url="http://upstream.test"
    )
    gateway_app = create_app(
        settings=Settings(upstream_url=UPSTREAM_URL, upstream_token="upstream-service-token"),
        registry=REGISTRY,
        http_client=upstream_client,
    )
    async with gateway_app.router.lifespan_context(gateway_app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_app), base_url="http://gateway.test"
    ) as client:
        yield Harness(client=client, log=log)
    await upstream_client.aclose()


@pytest_asyncio.fixture
async def failing_gateway() -> AsyncIterator[Any]:
    """Factory yielding a gateway whose upstream always raises the given exception."""
    opened: list[httpx.AsyncClient] = []

    async def build(exc: Exception) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            raise exc

        upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://upstream.test")
        opened.append(upstream_client)
        gateway_app = create_app(
            settings=Settings(upstream_url=UPSTREAM_URL), registry=REGISTRY, http_client=upstream_client
        )
        context = gateway_app.router.lifespan_context(gateway_app)
        await context.__aenter__()
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gateway.test")
        opened.append(client)
        return client

    yield build
    for client in opened:
        await client.aclose()


# --------------------------------------------------------------------------- #
# 1. wire format
# --------------------------------------------------------------------------- #
class TestWireFormatParsing:
    """JSON-RPC structures are parsed strictly, and errors keep the request id."""

    def test_parses_a_request(self) -> None:
        parsed = parse_jsonrpc(rpc("tools/call", {"name": "search_docs"}, 7))
        assert parsed == JsonRpcRequest("tools/call", 7, {"name": "search_docs"}, is_notification=False)

    def test_parses_a_notification(self) -> None:
        parsed = parse_jsonrpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert parsed.is_notification is True
        assert parsed.request_id is None

    def test_accepts_a_string_id(self) -> None:
        assert parse_jsonrpc(rpc("ping", None, "req-abc")).request_id == "req-abc"

    def test_rejects_batches(self) -> None:
        """MCP dropped JSON-RPC batching in 2025-06-18."""
        with pytest.raises(JsonRpcProtocolError) as exc:
            parse_jsonrpc([rpc("tools/list"), rpc("tools/list", request_id=2)])
        assert exc.value.code == INVALID_REQUEST

    @pytest.mark.parametrize("payload", ["a string", 42, None, True])
    def test_rejects_non_object_bodies(self, payload: Any) -> None:
        with pytest.raises(JsonRpcProtocolError) as exc:
            parse_jsonrpc(payload)
        assert exc.value.code == INVALID_REQUEST

    @pytest.mark.parametrize("version", ["1.0", "2", 2.0, None, ""])
    def test_rejects_wrong_jsonrpc_version(self, version: Any) -> None:
        with pytest.raises(JsonRpcProtocolError) as exc:
            parse_jsonrpc({"jsonrpc": version, "method": "tools/list", "id": 1})
        assert exc.value.code == INVALID_REQUEST

    @pytest.mark.parametrize("method", [None, "", 123, {"a": 1}])
    def test_rejects_bad_methods(self, method: Any) -> None:
        with pytest.raises(JsonRpcProtocolError) as exc:
            parse_jsonrpc({"jsonrpc": "2.0", "method": method, "id": 1})
        assert exc.value.code == INVALID_REQUEST

    @pytest.mark.parametrize("bad_id", [None, True, False, 1.5, [1], {"id": 1}])
    def test_rejects_bad_ids(self, bad_id: Any) -> None:
        """bool is an int subclass, and MCP forbids a null id on a request."""
        with pytest.raises(JsonRpcProtocolError) as exc:
            parse_jsonrpc({"jsonrpc": "2.0", "method": "tools/list", "id": bad_id})
        assert exc.value.code == INVALID_REQUEST

    @pytest.mark.parametrize("params", ["not-an-object", 5, [1, 2]])
    def test_rejects_non_object_params(self, params: Any) -> None:
        with pytest.raises(JsonRpcProtocolError) as exc:
            parse_jsonrpc({"jsonrpc": "2.0", "method": "tools/call", "id": 1, "params": params})
        assert exc.value.code == INVALID_PARAMS

    def test_errors_carry_the_request_id_for_correlation(self) -> None:
        """The id is recovered before the rest is validated, so clients can match up the error."""
        with pytest.raises(JsonRpcProtocolError) as exc:
            parse_jsonrpc({"jsonrpc": "1.0", "method": "tools/list", "id": 99})
        assert exc.value.request_id == 99


class TestTokenRegistry:
    """Bearer token parsing and role resolution."""

    def test_resolves_known_tokens(self) -> None:
        assert REGISTRY.resolve(ADMIN_TOKEN).role is Role.ADMIN  # type: ignore[union-attr]
        assert REGISTRY.resolve(VIEWER_TOKEN).role is Role.VIEWER  # type: ignore[union-attr]

    @pytest.mark.parametrize("token", ["", "nope", ADMIN_TOKEN + "x", ADMIN_TOKEN.upper(), ADMIN_TOKEN[:-1]])
    def test_rejects_unknown_tokens(self, token: str) -> None:
        assert REGISTRY.resolve(token) is None

    def test_fingerprint_does_not_reveal_the_secret(self) -> None:
        principal = REGISTRY.resolve(ADMIN_TOKEN)
        assert principal is not None
        assert ADMIN_TOKEN not in principal.token_id
        assert len(principal.token_id) == 12

    def test_from_env_parses_pairs(self) -> None:
        registry = TokenRegistry.from_env("tok-a:admin, tok-b:viewer")
        assert registry.resolve("tok-a").role is Role.ADMIN  # type: ignore[union-attr]
        assert registry.resolve("tok-b").role is Role.VIEWER  # type: ignore[union-attr]

    def test_from_env_falls_back_to_demo_tokens(self) -> None:
        registry = TokenRegistry.from_env("   ")
        assert registry.resolve(ADMIN_TOKEN) is not None

    @pytest.mark.parametrize("raw", ["tok-a:superuser", "no-role-here", ":admin"])
    def test_from_env_rejects_bad_config(self, raw: str) -> None:
        with pytest.raises(ValueError):
            TokenRegistry.from_env(raw)

    def test_from_env_tolerates_colons_in_the_token(self) -> None:
        """Split on the last colon, so JWT-ish secrets survive."""
        registry = TokenRegistry.from_env("head:body:sig:admin")
        assert registry.resolve("head:body:sig").role is Role.ADMIN  # type: ignore[union-attr]

    @pytest.mark.parametrize("header", [None, "", "Basic abc", "Bearer", "Bearer    ", "token abc"])
    def test_rejects_bad_authorization_headers(self, header: str | None) -> None:
        with pytest.raises(JsonRpcProtocolError) as exc:
            extract_bearer_token(header)
        assert exc.value.code == UNAUTHORIZED_TOOL_CALL

    @pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
    def test_bearer_scheme_is_case_insensitive(self, scheme: str) -> None:
        assert extract_bearer_token(f"{scheme} abc123") == "abc123"


class TestAuthorizationPolicy:
    """Method-level policy, checked without any HTTP involved."""

    ADMIN = Principal("admin-fp", Role.ADMIN)
    VIEWER = Principal("viewer-fp", Role.VIEWER)

    @pytest.mark.parametrize("method", ["tools/list", "initialize", "ping", "resources/list"])
    def test_non_tool_call_methods_pass_through(self, method: str) -> None:
        assert authorize(parse_jsonrpc(rpc(method)), self.VIEWER) == Decision(
            allowed=True, reason=f"{method} is not subject to tool policy"
        )

    @pytest.mark.parametrize("tool", ["search_docs", "get_customer_record", "administer_thing", "Admin_reset_key"])
    def test_unprivileged_tools_pass_for_any_role(self, tool: str) -> None:
        """The prefix check is case-sensitive and anchored: only a literal admin_ prefix is privileged."""
        decision = authorize(parse_jsonrpc(rpc("tools/call", {"name": tool})), self.VIEWER)
        assert decision.allowed is True

    @pytest.mark.parametrize("tool", ["admin_reset_key", "admin_delete_tenant", "admin_"])
    def test_admin_tools_are_denied_to_viewers(self, tool: str) -> None:
        decision = authorize(parse_jsonrpc(rpc("tools/call", {"name": tool})), self.VIEWER)
        assert decision.allowed is False
        assert decision.tool_name == tool

    @pytest.mark.parametrize("tool", ["admin_reset_key", "admin_delete_tenant"])
    def test_admin_tools_are_allowed_for_admins(self, tool: str) -> None:
        assert authorize(parse_jsonrpc(rpc("tools/call", {"name": tool})), self.ADMIN).allowed is True

    @pytest.mark.parametrize("params", [{}, {"name": ""}, {"name": 42}, {"name": None}, {"arguments": {}}])
    def test_malformed_tool_calls_are_invalid_params(self, params: dict[str, Any]) -> None:
        with pytest.raises(JsonRpcProtocolError) as exc:
            authorize(parse_jsonrpc(rpc("tools/call", params)), self.VIEWER)
        assert exc.value.code == INVALID_PARAMS


# --------------------------------------------------------------------------- #
# 2. proxying
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestProxyForwarding:
    """Allowed traffic reaches the downstream server unchanged."""

    async def test_tools_list_is_forwarded_transparently(self, harness: Harness) -> None:
        response = await harness.client.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER_TOKEN))
        assert response.status_code == 200
        assert response.json()["result"]["tools"] == UPSTREAM_TOOLS
        assert harness.upstream_requests() == 1

    async def test_ordinary_tool_call_is_forwarded_for_a_viewer(self, harness: Harness) -> None:
        body = rpc("tools/call", {"name": "search_docs", "arguments": {"query": "refunds"}}, 5)
        response = await harness.client.post("/mcp", json=body, headers=auth(VIEWER_TOKEN))
        assert response.status_code == 200
        payload = response.json()
        assert payload["id"] == 5
        assert payload["result"]["structuredContent"] == {
            "tool": "search_docs",
            "arguments": {"query": "refunds"},
            "ok": True,
        }

    async def test_admin_tool_call_is_forwarded_for_an_admin(self, harness: Harness) -> None:
        body = rpc("tools/call", {"name": "admin_reset_key", "arguments": {"tenant_id": "t-1"}}, 6)
        response = await harness.client.post("/mcp", json=body, headers=auth(ADMIN_TOKEN))
        assert response.status_code == 200
        assert response.json()["result"]["isError"] is False
        assert harness.log.snapshot()["tools_called"] == ["admin_reset_key"]

    async def test_notifications_are_forwarded_without_a_body(self, harness: Harness) -> None:
        body = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        response = await harness.client.post("/mcp", json=body, headers=auth(VIEWER_TOKEN))
        assert response.status_code == 202
        assert harness.upstream_requests() == 1

    async def test_resolved_role_is_passed_downstream(self, harness: Harness) -> None:
        """The downstream server learns who the gateway decided this was."""
        await harness.client.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER_TOKEN))
        entry = harness.log.snapshot()["entries"][0]
        assert entry["gateway_role"] == "viewer"
        assert entry["gateway_principal"] == TokenRegistry.fingerprint(VIEWER_TOKEN)

    async def test_client_credentials_are_not_relayed_downstream(self, harness: Harness) -> None:
        """The gateway terminates the client's token and presents its own service token."""
        await harness.client.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER_TOKEN))
        seen = harness.log.snapshot()["entries"][0]["authorization_fingerprint"]
        assert seen == upstream_fingerprint("Bearer upstream-service-token")
        assert seen != upstream_fingerprint(f"Bearer {VIEWER_TOKEN}")

    async def test_request_id_header_is_echoed(self, harness: Harness) -> None:
        response = await harness.client.post(
            "/mcp", json=rpc("tools/list"), headers={**auth(VIEWER_TOKEN), "X-Request-Id": "trace-42"}
        )
        assert response.headers["x-request-id"] == "trace-42"
        assert harness.log.snapshot()["entries"][0]["request_id"] == "trace-42"

    async def test_event_stream_responses_are_relayed(self, harness: Harness) -> None:
        """A text/event-stream body is passed through rather than buffered into JSON."""
        body = rpc("tools/call", {"name": "stream_echo", "arguments": {"message": "one two three"}}, 8)
        async with harness.client.stream("POST", "/mcp", json=body, headers=auth(VIEWER_TOKEN)) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            text = "".join([chunk async for chunk in response.aiter_text()])
        assert text.count("event: chunk") == 3
        assert text.rstrip().endswith("data: [DONE]")

    async def test_upstream_errors_are_passed_through_untouched(self, harness: Harness) -> None:
        """A downstream -32601 is the downstream's answer, not something the gateway rewrites."""
        response = await harness.client.post("/mcp", json=rpc("resources/list"), headers=auth(VIEWER_TOKEN))
        assert response.json()["error"]["code"] == -32601


@pytest.mark.asyncio
class TestShortCircuit:
    """Denied traffic is answered by the gateway and never leaves it."""

    async def test_viewer_calling_an_admin_tool_gets_32001(self, harness: Harness) -> None:
        body = rpc("tools/call", {"name": "admin_reset_key", "arguments": {"tenant_id": "t-1"}}, 11)
        response = await harness.client.post("/mcp", json=body, headers=auth(VIEWER_TOKEN))

        assert response.status_code == 200  # so MCP clients surface the JSON-RPC error
        payload = response.json()
        assert payload["id"] == 11
        assert payload["error"]["code"] == UNAUTHORIZED_TOOL_CALL
        assert payload["error"]["message"] == "Unauthorized Tool Call"
        assert payload["error"]["data"] == {
            "tool": "admin_reset_key",
            "required_role": "admin",
            "caller_role": "viewer",
        }
        assert response.headers["x-mcp-gateway-decision"] == "denied"

    async def test_denied_call_never_reaches_the_downstream_server(self, harness: Harness) -> None:
        """The core requirement: intercept *without* calling downstream."""
        assert harness.upstream_requests() == 0
        body = rpc("tools/call", {"name": "admin_delete_tenant", "arguments": {"tenant_id": "t-9"}}, 12)
        await harness.client.post("/mcp", json=body, headers=auth(VIEWER_TOKEN))
        assert harness.upstream_requests() == 0
        assert harness.log.snapshot()["tools_called"] == []

    async def test_denial_survives_a_mixed_sequence(self, harness: Harness) -> None:
        """Allowed calls still go through either side of a denial."""
        await harness.client.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER_TOKEN))
        await harness.client.post(
            "/mcp", json=rpc("tools/call", {"name": "admin_reset_key"}, 2), headers=auth(VIEWER_TOKEN)
        )
        await harness.client.post(
            "/mcp", json=rpc("tools/call", {"name": "search_docs", "arguments": {"query": "x"}}, 3),
            headers=auth(VIEWER_TOKEN),
        )
        assert harness.upstream_requests() == 2
        assert harness.log.snapshot()["tools_called"] == ["search_docs"]


@pytest.mark.asyncio
class TestAuthenticationErrors:
    """Transport-level credential problems answer 401 and stop there."""

    @pytest.mark.parametrize(
        "headers",
        [{}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer "}, {"Authorization": "Bearer wrong-token"}],
    )
    async def test_bad_credentials_are_rejected(self, harness: Harness, headers: dict[str, str]) -> None:
        response = await harness.client.post("/mcp", json=rpc("tools/list"), headers=headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["error"]["code"] == UNAUTHORIZED_TOOL_CALL
        assert harness.upstream_requests() == 0

    async def test_rejection_body_is_still_json_rpc(self, harness: Harness) -> None:
        response = await harness.client.post("/mcp", json=rpc("tools/list"), headers={})
        assert response.json()["jsonrpc"] == "2.0"


@pytest.mark.asyncio
class TestMalformedRequests:
    """Bad wire format is answered by the gateway, not forwarded."""

    async def test_invalid_json_is_parse_error(self, harness: Harness) -> None:
        response = await harness.client.post(
            "/mcp", content=b"{not json", headers={**auth(VIEWER_TOKEN), "Content-Type": "application/json"}
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == PARSE_ERROR
        assert harness.upstream_requests() == 0

    async def test_batch_is_rejected(self, harness: Harness) -> None:
        response = await harness.client.post("/mcp", json=[rpc("tools/list")], headers=auth(VIEWER_TOKEN))
        assert response.status_code == 400
        assert response.json()["error"]["code"] == INVALID_REQUEST
        assert harness.upstream_requests() == 0

    async def test_wrong_version_is_rejected(self, harness: Harness) -> None:
        response = await harness.client.post(
            "/mcp", json={"jsonrpc": "1.0", "method": "tools/list", "id": 1}, headers=auth(VIEWER_TOKEN)
        )
        assert response.json()["error"]["code"] == INVALID_REQUEST
        assert harness.upstream_requests() == 0

    async def test_tool_call_without_a_name_is_invalid_params(self, harness: Harness) -> None:
        response = await harness.client.post(
            "/mcp", json=rpc("tools/call", {"arguments": {}}, 4), headers=auth(VIEWER_TOKEN)
        )
        assert response.status_code == 200
        assert response.json()["error"]["code"] == INVALID_PARAMS
        assert response.json()["id"] == 4
        assert harness.upstream_requests() == 0


@pytest.mark.asyncio
class TestUpstreamFailures:
    """Upstream faults become sanitised gateway errors."""

    SECRET = "postgres://user:hunter2@10.0.0.5/prod"

    async def test_connection_failure_returns_502(self, failing_gateway: Any) -> None:
        client = await failing_gateway(httpx.ConnectError(f"failed to connect: {self.SECRET}"))
        response = await client.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER_TOKEN))
        assert response.status_code == 502
        assert response.json()["error"]["message"] == "Upstream MCP server is unavailable"

    async def test_timeout_returns_504(self, failing_gateway: Any) -> None:
        client = await failing_gateway(httpx.ReadTimeout(f"timed out talking to {self.SECRET}"))
        response = await client.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER_TOKEN))
        assert response.status_code == 504
        assert response.json()["error"]["message"] == "Upstream MCP server timed out"

    async def test_upstream_detail_is_never_leaked(self, failing_gateway: Any) -> None:
        """No connection strings, hostnames or exception text in the client's payload."""
        client = await failing_gateway(httpx.ConnectError(f"failed to connect: {self.SECRET}"))
        response = await client.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER_TOKEN))
        raw = json.dumps(response.json())
        for fragment in (self.SECRET, "hunter2", "10.0.0.5", "ConnectError", "Traceback"):
            assert fragment not in raw

    async def test_failures_preserve_the_request_id(self, failing_gateway: Any) -> None:
        client = await failing_gateway(httpx.ConnectError("nope"))
        response = await client.post("/mcp", json=rpc("tools/list", None, 77), headers=auth(VIEWER_TOKEN))
        assert response.json()["id"] == 77


@pytest.mark.asyncio
async def test_healthz_reports_decision_counters(harness: Harness) -> None:
    """The gateway exposes what it allowed and what it blocked."""
    await harness.client.post("/mcp", json=rpc("tools/list"), headers=auth(VIEWER_TOKEN))
    await harness.client.post("/mcp", json=rpc("tools/call", {"name": "admin_reset_key"}, 2), headers=auth(VIEWER_TOKEN))
    await harness.client.post("/mcp", json=rpc("tools/list"), headers={})

    counters = (await harness.client.get("/healthz")).json()["counters"]
    assert counters["forwarded"] == 1
    assert counters["denied"] == 1
    assert counters["rejected_auth"] == 1
