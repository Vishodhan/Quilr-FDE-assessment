# Task 2 — MCP security gateway proxy (tool filtering & auth)

An HTTP/JSON-RPC reverse proxy that sits between an AI agent client and a downstream MCP
server and decides, per method and per tool, what gets through.

```
agent ──Bearer token──▶ gateway ──policy──▶ downstream MCP server
                           │
                           └─▶ -32001 Unauthorized Tool Call  (downstream never contacted)
```

| File | Purpose |
|------|---------|
| `policy.py` | JSON-RPC wire parsing, token→role registry, the authorization rules. No HTTP. |
| `mcp_gateway.py` | FastAPI reverse proxy: auth, dispatch, short-circuit, faithful forwarding. |
| `mock_upstream.py` | Downstream mock MCP server, with a call log so tests can prove what did *not* reach it. |
| `test_gateway.py` | 93 tests across wire format, policy, proxying, short-circuit and fault injection. |

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # optional; sensible defaults are built in

python mock_upstream.py       # terminal 1 — downstream MCP server on :9001
python mcp_gateway.py             # terminal 2 — gateway on :9000

pytest -v                     # or just run the suite; it needs no servers
```

Demo tokens (used when `GATEWAY_TOKENS` is unset):

| Token | Role |
|-------|------|
| `admin-token-abc123` | `admin` |
| `viewer-token-xyz789` | `viewer` |

### Try it

```bash
# viewer lists tools -> forwarded
curl -s localhost:9000/mcp -H "Authorization: Bearer viewer-token-xyz789" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# viewer calls an admin tool -> blocked at the gateway
curl -s localhost:9000/mcp -H "Authorization: Bearer viewer-token-xyz789" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"tenant_id":"t-1"}}}'
# {"jsonrpc":"2.0","id":2,"error":{"code":-32001,"message":"Unauthorized Tool Call", ...}}

# admin calls the same tool -> forwarded
curl -s localhost:9000/mcp -H "Authorization: Bearer admin-token-abc123" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"tenant_id":"t-1"}}}'

# confirm the blocked call never arrived downstream
curl -s localhost:9001/__stats
```

---

## Policy

| Method | Behaviour |
|--------|-----------|
| `tools/list` | Forwarded transparently. |
| `tools/call` where `params.name` starts with `admin_` | Requires role `admin`; otherwise `-32001`, short-circuited. |
| `tools/call` for any other tool | Forwarded. |
| Anything else (`initialize`, `ping`, `resources/*`, notifications) | Forwarded. |

The prefix test is case-sensitive and anchored, so `Admin_reset_key` and
`administer_thing` are *not* privileged — matching the literal requirement rather than
guessing at intent. `admin_` on its own is privileged.

---

## Response codes

Two different failures deserve two different answers, and conflating them breaks clients.

| Situation | HTTP | JSON-RPC |
|-----------|------|----------|
| Missing / malformed / unknown bearer token | `401` + `WWW-Authenticate: Bearer` | `-32001` |
| Body is not valid JSON | `400` | `-32700` |
| Not a JSON-RPC 2.0 frame (bad version, bad id, batch) | `400` | `-32600` |
| `tools/call` with no usable `params.name` | `200` | `-32602` |
| **Unauthorized admin tool call** | **`200`** | **`-32001 Unauthorized Tool Call`** |
| Upstream unreachable | `502` | `-32603` |
| Upstream timed out | `504` | `-32603` |

**Why `200` for the denial:** MCP clients raise a *transport* error on any non-2xx
response and never look at the body — the model would see "HTTP 403" instead of the
`-32001` it is supposed to read and react to. Credential failures are genuinely
transport-level, so those do get a `401`. A `X-MCP-Gateway-Decision: denied` header is set
alongside the `200` so proxies and dashboards can still count blocks.

---

## Design decisions

### Policy is separated from transport

`policy.py` has no FastAPI, no httpx, no I/O. `parse_jsonrpc`, `authorize` and
`TokenRegistry` are plain functions and classes over plain data, so 67 of the tests need
no server at all and run in milliseconds.

### The request id is recovered *before* the frame is validated

`parse_jsonrpc` reads `id` first, then checks `jsonrpc`, `method` and `params`. A client
that sends a malformed frame still gets an error it can correlate with its own request.

### The client's credential stops at the gateway

`Authorization` is on the strip list for forwarded requests. If `GATEWAY_UPSTREAM_TOKEN`
is set, the gateway presents its own. It also injects `X-MCP-Gateway-Role`,
`X-MCP-Gateway-Principal` (a SHA-256 prefix, never the token) and `X-Request-Id`, so the
downstream server can log and re-check the decision without ever seeing a user secret.

### Forwarding uses a denylist, not an allowlist

Everything except hop-by-hop headers, `Host`, `Content-Length` and `Authorization` is
relayed. An allowlist would silently drop `Mcp-Session-Id`, `MCP-Protocol-Version`,
`Last-Event-Id` and any header a future MCP revision adds. On the way back,
`Content-Encoding` is dropped as well, since httpx has already decoded the body.

### SSE responses are streamed, not buffered

MCP's Streamable HTTP transport can answer with `text/event-stream`. Those are relayed
chunk by chunk through a `StreamingResponse`, so a long tool call reaches the client as it
happens. `test_event_stream_responses_are_relayed` covers it.

### Token comparison is constant-time

`TokenRegistry.resolve` compares every entry with `secrets.compare_digest` and does not
break out early, so response time carries no information about how much of a token was
correct.

### Errors are sanitised

Upstream exception text never reaches the client — the gateway logs
`type(exc).__name__` with a correlation id and returns a fixed message.
`test_upstream_detail_is_never_leaked` asserts a planted connection string
(`postgres://user:hunter2@10.0.0.5/prod`) appears nowhere in the response.

---

## Definition of done

| Criterion | Where it is proven |
|-----------|--------------------|
| **Correct JSON-RPC wire-format parsing** | `TestWireFormatParsing` (27 tests): batches, non-object bodies, wrong versions, bad methods, `bool`/`float`/`null` ids, non-object params, and id preservation through errors. |
| **Clean proxy middleware and faithful forwarding** | `TestProxyForwarding` (9 tests): `tools/list` body matches the upstream's byte for byte, ordinary and admin tool calls, `202` for notifications, header hygiene, `X-Request-Id` echo, SSE relayed as a stream, and a downstream `-32601` passed through unrewritten. |
| **Method-level authorization with tidy error handling** | `TestAuthorizationPolicy` (18) + `TestTokenRegistry` (22) + `TestShortCircuit` (3) + `TestAuthenticationErrors` (5) + `TestMalformedRequests` (4) + `TestUpstreamFailures` (4). The central one is `test_denied_call_never_reaches_the_downstream_server`, which asserts the upstream call log stays empty. |

Tests run the gateway and the mock upstream **in-process** over `httpx.ASGITransport`,
so there are no ports to allocate and no sleeps. Upstream faults are injected with
`httpx.MockTransport`.
