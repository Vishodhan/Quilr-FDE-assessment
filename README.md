# Quilr-FDE-assessment

# QuilrAI - FDE Assessment - Vishodhan Krishnan

A four-part build covering **Model Context Protocol (MCP) servers, MCP gateways, LLM gateways, and
security guardrails** — the plumbing that sits between an AI agent and the tools and models it calls.

---

## The shape of the project


| # | Layer | What gets built | Core concern |
|---|-------|-----------------|--------------|
| 1 | MCP server | A tool server with strict validation over stdio | Protocol correctness |
| 2 | MCP gateway | A reverse proxy that authorizes tool calls | Policy enforcement |
| 3 | LLM gateway | A streaming proxy that redacts PII mid-stream | Real-time data handling |
| 4 | LLM gateway | A rate limiter + model failover router | Resilience under load |

---

## Part 1 — Custom MCP server with strict validation & transport handling

**Stack:**  Python (`mcp`)

A runnable MCP server exposing exactly two tools:

- **`get_customer_record`** — input: `customer_id`, a string that must match the `CUST-XXXXX` format.
- **`trigger_refund`** — inputs: `customer_id`, `amount` (positive float), and `reason` (string, minimum 10 characters).

**Requirements**

- Validate inputs strictly with Pydantic (Py), and reject anything malformed with
  **standard MCP JSON-RPC error codes** — not ad-hoc strings or silent coercion.
- Speak over **stdio transport**, with a hard rule: `stdout` carries JSON-RPC frames and nothing else.
  Every log, warning, and debug line goes to `stderr`.

**Definition of done**

- **STDIO isolation** — no stray `console.log` / `print` poisoning the protocol channel.
- **Protocol compliance** — correct JSON-RPC error mapping and execution flow.
- **Validation depth** — schemas that hold up against edge cases in every field.


---

## Part 2 — MCP security gateway proxy (tool filtering & auth)

**Stack:**  Python

A lightweight HTTP/JSON-RPC **reverse proxy** that sits between an AI agent client and a downstream
mock MCP server, and decides what gets through.

**Requirements**

- Read the `Bearer <token>` HTTP header and resolve the caller's **role** (e.g. `admin`, `viewer`).
- Inspect the incoming MCP JSON-RPC payload and branch on `method`:
  - **`tools/list`** → forward transparently to the downstream server.
  - **`tools/call`** → inspect `params.name`. If the tool name starts with `admin_`
    (e.g. `admin_reset_key`), require the token role to be `admin`.
- On an unauthorized admin tool call, **intercept and short-circuit**: return JSON-RPC error
  **`-32001: Unauthorized Tool Call`** without ever touching the downstream server.

**Definition of done**

- Correct parsing of JSON-RPC wire-format structures.
- Clean proxy middleware and faithful HTTP request/response forwarding.
- Fine-grained, **method-level** authorization with tidy error handling.


---

## Part 3 — LLM gateway streaming guardrail (PII redaction)

An **LLM gateway** endpoint that forwards text-generation requests to a provider and streams the
response back to the client — scrubbing sensitive data on the way out.

**Requirements**

- Intercept the returning chunk stream **in real time**.
- Parse stream deltas to detect and redact sensitive patterns (emails, SSNs, credit card numbers),
  replacing each match with `[REDACTED]`.
- Keep the stream responsive: **never buffer the full response in memory**, and keep Time To First Token
  (TTFT) and overall stream latency low.

**Definition of done**

- Efficient async chunking and buffer-state management.
- Performant regex/string matching over **partial** text — patterns that straddle chunk boundaries still
  have to be caught.
- Memory efficiency and a genuinely low-latency proxy design.

> The hard part is the boundary case: a credit card number split across two chunks must not leak.
> That means holding a bounded tail buffer, not the whole response.

---

## Part 4 — Rate-limiting & model fallback router

A resilient **model-routing module** for an LLM gateway that accepts incoming completion requests and
stays up when the primary provider does not.

**Requirements**

- Implement a **token-aware sliding-window rate limiter** — e.g. max 50,000 tokens/minute, per tenant API key.
- **Failover automatically** to a secondary backup provider when the primary returns `429 Too Many Requests`
  or exceeds a **3000 ms** timeout.
- Return a **standardized gateway error payload** — no raw upstream stack traces, no leaked internals.
- Persist state in **on-disk SQLite**.

**Definition of done**

- Async concurrency handling and timeout race conditions.
- Accurate rate-limiter state eviction and token accounting.
- Graceful fallback mechanics and consistent error sanitization.


---


## Repository layout

Each task is self-contained: its own `README.md`, `IMPLEMENTATION_PLAN.md`,
`requirements.txt`, source, tests, and a `.env.example` where there is anything to
configure.

```
task-1-mcp-server/          models.py  server.py                        test_server.py
task-2-mcp-gateway/         policy.py  mcp_gateway.py  mock_upstream.py test_gateway.py
task-3-streaming-guardrail/ redactor.py providers.py   llm_gateway.py   test_guardrail.py
task-4-model-router/        store.py   router.py       app.py           test_router.py
```

Every task's `README.md` carries a **definition-of-done table** mapping each scoring
criterion to the specific tests that prove it, plus a "notes and known edges" section.
Every `IMPLEMENTATION_PLAN.md` records the steps actually followed, including the
decisions that were reconsidered and what the first test run turned up.

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate       # Windows
# python -m venv .venv && source .venv/bin/activate  # macOS / Linux

pip install -r task-1-mcp-server/requirements.txt \
            -r task-2-mcp-gateway/requirements.txt \
            -r task-3-streaming-guardrail/requirements.txt \
            -r task-4-model-router/requirements.txt

pytest                    # all four suites
pytest task-3-streaming-guardrail -v    # or one at a time
```

**No API keys and no network are required.** Tasks 3 and 4 default to in-repo mock
providers; set `LLM_PROVIDER=upstream` / `PRIMARY_API_KEY` to point them at a real one.

Servers, if you want to poke at them by hand:

| Task | Command | Port |
|------|---------|------|
| 1 | `python task-1-mcp-server/server.py` | stdio |
| 2 | `python task-2-mcp-gateway/mock_upstream.py` then `python task-2-mcp-gateway/mcp_gateway.py` | 9001, 9000 |
| 3 | `python task-3-streaming-guardrail/llm_gateway.py` | 8000 |
| 4 | `python task-4-model-router/app.py` | 8100 |

---

## Test results

```
task-1-mcp-server             85 passed
task-2-mcp-gateway            93 passed
task-3-streaming-guardrail   131 passed
task-4-model-router           67 passed
                            ─────────────
                             376 passed in 33.37s
```

Verified on Python 3.12.3 (Windows). `mcp` is pinned to `>=1.17,<2.0`: the 2.x line
renames `McpError` to `MCPError` and restructures `mcp.server`.
