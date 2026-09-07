# Task 1 — Custom MCP server with strict validation & transport handling

A runnable MCP server over **stdio** exposing two tools, with Pydantic-enforced input
schemas and standard JSON-RPC error codes.

| File | Purpose |
|------|---------|
| `models.py` | Strict Pydantic input/output schemas; the source of truth for the advertised JSON Schemas. |
| `server.py` | Tool implementations, JSON-RPC handlers, stdio isolation, entrypoint. |
| `test_server.py` | 85 tests across validation depth, refund arithmetic, stdio isolation and protocol compliance. |

---

## Quickstart

```bash
pip install -r requirements.txt
python server.py          # speaks JSON-RPC on stdin/stdout, logs on stderr
pytest -v                 # run the suite
```

To attach it to an MCP client (Claude Desktop, MCP Inspector, etc.):

```json
{
  "mcpServers": {
    "customer-support": {
      "command": "python",
      "args": ["/absolute/path/to/task-1-mcp-server/server.py"]
    }
  }
}
```

---

## Tools

### `get_customer_record`

| Field | Rule |
|-------|------|
| `customer_id` | string matching `^CUST-[A-Z0-9]{5}$` |

### `trigger_refund`

| Field | Rule |
|-------|------|
| `customer_id` | string matching `^CUST-[A-Z0-9]{5}$` |
| `amount` | number, `> 0`, finite (NaN/Infinity rejected), at most 2 decimal places |
| `reason` | string, 10–500 characters, not whitespace-only |

Both schemas set `additionalProperties: false`, so an unknown argument is an error
rather than something silently dropped.

Seeded customers: `CUST-10042` (balance 250.00), `CUST-20099` (75.25), `CUST-AB123` (0.00).

---

## Design decisions

### 1. stdout is reserved for the protocol, at the file-descriptor level

A `print()` guideline is not an enforcement mechanism — a dependency can still poison
the channel. `reserve_stdout_for_protocol()` in [server.py](server.py) does this once at
startup:

1. `os.dup(1)` — take a private duplicate of the real stdout and hand it to the transport.
2. `os.dup2(2, 1)` — fd 1 now aliases **stderr**, so C-level and subprocess writes are redirected too.
3. `sys.stdout = sys.stderr` — catches writes made through the Python object.

After that the transport holds the only handle that reaches the true stdout.
`test_stray_print_cannot_reach_stdout` proves it: it runs the fd swap, then fires a
`print()` and a `sys.stdout.write()`, and asserts neither reaches the real stdout.

### 2. Validation errors are real JSON-RPC errors, not `isError` results

The SDK's `@server.call_tool()` decorator wraps handlers in a `try/except` that turns
**every** exception into a `CallToolResult(isError=True)` — which would bury the `-32602`
codes this task requires. So `handle_call_tool` is registered directly on
`server.request_handlers[types.CallToolRequest]`, bypassing that wrapper. Raising
`McpError` from there produces a genuine JSON-RPC error frame.

### 3. Protocol errors vs. business errors

This is the split the MCP spec asks for, and the server honours it:

| Situation | Response |
|-----------|----------|
| Malformed arguments (bad ID, negative or sub-cent amount, short reason, extra field) | JSON-RPC `-32602 Invalid params` |
| Unknown tool name | JSON-RPC `-32602 Invalid params` (matches the official TS SDK) |
| Customer does not exist | `CallToolResult(isError=true)` |
| Refund exceeds refundable balance, or rounds to zero cents | `CallToolResult(isError=true)` |
| Unexpected server fault | JSON-RPC `-32603 Internal error`, message sanitised |

A model calling the tool can *react* to "no such customer"; it cannot react to "your
JSON was malformed". Protocol faults belong on the JSON-RPC error channel, business
outcomes belong in the result.

### 4. Error payloads never echo the input

`format_validation_errors()` emits `{field, message, type}` and deliberately drops
Pydantic's `input` value — error payloads end up in logs, and the rejected value may be
customer PII. `test_errors_never_echo_the_offending_input` locks this in.

### 5. One source of truth for the schema

`tools/list` advertises `Model.model_json_schema()`, generated from the same models that
enforce validation, so the published contract cannot drift from the enforcement.

---

## Definition of done

| Criterion | Where it is proven |
|-----------|--------------------|
| **STDIO isolation** — no stray writes on the protocol channel | `TestStdioIsolation` (4 tests): every stdout line parses as JSON-RPC 2.0; `print()`/`sys.stdout.write()` are shown landing on stderr; logs confirmed on stderr. |
| **Protocol compliance** — correct JSON-RPC error mapping and flow | `TestProtocolCompliance` (13 tests) drives a real subprocess through initialize → initialized → tools/list → 9 tool calls and asserts the code on each reply. Plus `test_official_client_can_drive_the_server`, which runs the real SDK `ClientSession` end to end. |
| **Business-rule correctness** — the books balance | `TestRefundArithmetic` (6 tests): a zero balance cannot be overdrawn, an exact-balance refund leaves `0.0` rather than `-0.0`, receipt/ledger/balance always agree, repeated refunds conserve the total, and a sub-cent request is refused even when schema validation is bypassed. |
| **Validation depth** — schemas that hold up on every field | `TestValidationDepth` (61 tests): casing, length, separators, whitespace-padding, newline injection, non-string types, `0`/negative/NaN/±Infinity amounts, string-vs-number coercion, the 9-vs-10 character reason boundary, whitespace-only reasons, unknown and missing fields, and frozen-model immutability. |

```
85 passed in 9.24s
```

---

## Notes and known edges

- **Money is handled in whole cents.** `amount` is rejected above 2 decimal places, and the
  balance check and the debit are both computed in integer cents. Comparing rounded floats
  while debiting unrounded ones previously let a sub-cent request overdraw a balance; see
  `TestRefundArithmetic`.
- **`amount` accepts JSON integers.** `strict=True` blocks `"50"` → `50.0` and `True` → `1.0`,
  but still admits `50` for a float field. JSON has no separate integer type, so rejecting
  `{"amount": 50}` would be wrong.
- **`CUST-XXXXX` is read as five uppercase alphanumerics**, so both `CUST-10042` and
  `CUST-AB123` are valid. Lowercase is rejected rather than up-cased — the task calls for
  rejection, not silent coercion.
- **A line that is not valid JSON gets no `-32700` reply.** The SDK's stdio transport logs
  the parse failure and emits a `notifications/message` frame instead. There is no request
  `id` to answer against, so this is reasonable, but it is SDK behaviour rather than ours.
- **An unrecognised JSON-RPC *method* returns `-32602`, not `-32601`.** The SDK validates
  the frame against its `ClientRequest` union before dispatch, and a failed union match is
  reported as invalid params. Method-level dispatch inside `tools/call` is fully ours and
  is covered above.
- **Closing stdin mid-flight discards pending replies.** `Server.run()` cancels in-flight
  handlers as soon as the transport closes. That is deliberate SDK behaviour; the test
  harness holds stdin open until its replies arrive, exactly as a real client does.
