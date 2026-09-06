# Task 2 — Implementation plan

The steps actually followed to build the MCP security gateway, in order.

---

## Step 0 — Scope

**Asked for:** an HTTP/JSON-RPC reverse proxy between an agent client and a downstream MCP
server. Read `Bearer <token>` → role. Forward `tools/list`. Inspect `tools/call`: if
`params.name` starts with `admin_`, require role `admin`, otherwise return
`-32001 Unauthorized Tool Call` **without calling downstream**.

**In scope:** bearer→role resolution, strict JSON-RPC parsing, method-level policy,
faithful forwarding, short-circuit denial, sanitised upstream-failure handling, a mock
downstream server, tests.

**Out of scope:** JWT/OIDC verification, rate limiting, audit persistence, per-tool ACLs
beyond the `admin_` prefix. `TokenRegistry` is the seam where real auth would land.

**Extra pieces the requirements imply but do not spell out**, and why they are in:

- A **mock downstream server** — "without calling downstream" is untestable without
  something that can report it was not called.
- **Notification handling** (`id`-less frames) — MCP clients send `notifications/initialized`
  during handshake; a proxy that only understands requests breaks on connect.
- **SSE passthrough** — MCP Streamable HTTP can answer `text/event-stream`; buffering that
  into JSON would not be "faithful forwarding".

---

## Step 1 — Split policy from transport

Put every rule in `policy.py` with no FastAPI, no httpx, no I/O: `parse_jsonrpc`,
`authorize`, `TokenRegistry`, `extract_bearer_token`, `error_body`. 67 of the 93 tests
then need no server at all.

## Step 2 — `policy.py`: wire format first

Rules settled while writing it:

- **Recover `id` before validating anything else**, so a client that sends a malformed
  frame still gets a correlatable error.
- **`bool` must be excluded from the `id` check** — `isinstance(True, int)` is `True` in
  Python, so a naive `isinstance(raw_id, (str, int))` would accept `"id": true`.
- **`null` id is rejected.** JSON-RPC tolerates it; MCP forbids it.
- **Arrays are rejected** with `-32600` — MCP dropped batching in 2025-06-18.
- **`params` must be an object when present** → `-32602`.

## Step 3 — `policy.py`: tokens and roles

- `Role` as a `str` enum, so a bad value in `GATEWAY_TOKENS` fails at startup with a
  message naming the valid roles, not at the first request.
- `resolve()` compares **every** entry with `secrets.compare_digest` and does not break
  early — no timing oracle on token prefixes.
- `Principal.token_id` is `sha256(token)[:12]`, so logs can identify a caller without ever
  holding the secret.
- `from_env` splits on the **last** colon (`rpartition`), so a JWT-shaped token containing
  colons still parses.

## Step 4 — `policy.py`: the authorization rule

`authorize()` returns a `Decision`, and raises `JsonRpcProtocolError(-32602)` when a
`tools/call` has no usable `params.name` — a call with no tool name cannot be authorized
either way, so it is not a policy answer.

The prefix check is deliberately literal: `startswith("admin_")`, case-sensitive. Guessing
at intent (case-insensitive, fuzzy) would be inventing a requirement.

## Step 5 — `mock_upstream.py`

FastAPI JSON-RPC server handling `initialize`, `ping`, `tools/list`, `tools/call` and
notifications, exposing five tools including two `admin_*` ones and a `stream_echo` that
answers with SSE.

It applies **no authorization of its own** — on purpose. If it enforced roles too, "the
downstream server was never contacted" would stop proving anything.

Every request lands in a `CallLog`, exposed at `GET /__stats`. The `Authorization` value is
**fingerprinted, not stored**, so tests can assert which credential arrived without keeping
one in memory.

## Step 6 — `mcp_gateway.py`

Order of operations in `_handle`: authenticate → decode → parse → authorize → forward.
Authentication first, so an anonymous caller cannot even get its JSON parsed.

Decisions made here:

- **`create_app(settings, registry, http_client)`** — the injectable client is what lets
  tests run the gateway against an in-process upstream, and against fault-injecting
  transports, with no sockets.
- **Header handling is a denylist**, not an allowlist: strip hop-by-hop plus `Host`,
  `Content-Length` and `Authorization`; relay everything else. An allowlist would silently
  drop `Mcp-Session-Id`, `MCP-Protocol-Version` and anything a future revision adds.
- **`Content-Encoding` is stripped from responses** — httpx has already decoded the body,
  so relaying the header would hand the client bytes it would try to decompress again.
- **SSE is streamed**, buffered bodies are read whole; the branch is on the upstream's
  `content-type`.
- **HTTP status mapping**, the one genuinely debatable call:

  | Failure | HTTP |
  |---------|------|
  | credential problem | `401` |
  | malformed JSON / frame | `400` |
  | **policy denial** | **`200`** with `-32001` |

  MCP clients raise a transport error on non-2xx and never read the body, so a `403` would
  hide `-32001` from the model that is supposed to react to it. A
  `X-MCP-Gateway-Decision: denied` header keeps the denial visible to infrastructure.
- **Error sanitisation:** `type(exc).__name__` and a correlation id to the log, a fixed
  message to the client.
- **An outer `try/except`** around the whole handler, so a gateway bug returns `-32603`
  rather than a FastAPI traceback.

## Step 7 — Tests

Both apps run in-process over `httpx.ASGITransport`; lifespan is entered directly via
`app.router.lifespan_context(app)`, which avoids an `asgi-lifespan` dependency. Upstream
faults come from `httpx.MockTransport` handlers that raise `ConnectError` / `ReadTimeout`.

Load-bearing tests:

- `test_denied_call_never_reaches_the_downstream_server` — the actual requirement.
- `test_denial_survives_a_mixed_sequence` — allowed calls still work either side of a denial.
- `test_client_credentials_are_not_relayed_downstream` — compares fingerprints to show the
  upstream saw the service token, not the caller's.
- `test_upstream_detail_is_never_leaked` — plants
  `postgres://user:hunter2@10.0.0.5/prod` in the upstream exception and asserts no
  fragment of it appears in the response.

**First run: 93 passed.** Two cleanups afterwards: a leftover stub test with a stray
attribute access was replaced by the fingerprint comparison above, and the `CallLog` was
changed from recording `authorization_present: bool` to a fingerprint so the credential
claim could be asserted properly rather than approximately.

## Step 8 — Documentation

`README.md` with the curl walkthrough, the response-code table and its rationale, the
seven design decisions, and a definition-of-done table mapping each criterion to the tests
that prove it. `.env.example` covers every variable both processes read.

## Step 9 — Rename for cross-task collection

Running `pytest` from the repository root failed at collection: Task 3 also had a
`gateway.py`, and because both test files insert their own directory onto `sys.path`,
whichever imported first won `sys.modules["gateway"]` for both.

Renamed this one to `mcp_gateway.py` (Task 3's became `llm_gateway.py`) and updated the
imports, docs and run commands. Each folder still stands alone, and the whole repository
now collects in one run: **359 passed**.
