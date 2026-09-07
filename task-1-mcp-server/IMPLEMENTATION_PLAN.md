# Task 1 — Implementation plan

The steps actually followed to build the MCP server, in order, with the findings that
changed the design along the way.

---

## Step 0 — Scope

**Asked for:** a runnable MCP server exposing `get_customer_record` and `trigger_refund`,
strict schema validation, standard JSON-RPC error codes, stdio transport with stdout
reserved for protocol frames.

**In scope:** the two tools, Pydantic schemas, JSON-RPC error mapping, fd-level stdout
isolation, an in-memory backing store, a test suite covering all three scoring criteria.

**Out of scope:** persistence, auth, HTTP/SSE transport, more tools. A dict-backed store
is enough to exercise both a happy path and a business-rule failure.

---

## Step 1 — Pin the SDK before writing anything

`pip install "mcp>=1.17"` resolved to **2.1.1**, which renames `McpError` → `MCPError` and
restructures `mcp.server`. Pinned `mcp>=1.17,<2.0` (resolves to 1.29.1) so the code, the
docs and `requirements.txt` all describe one API.

## Step 2 — Read the SDK to find where errors actually go

Traced the request path in `mcp/server/lowlevel/server.py`:

- `_handle_request` catches `McpError` and turns `err.error` into a JSON-RPC error frame —
  **this is the hook needed for `-32602`.**
- Anything else becomes `ErrorData(code=0, ...)`, which is not a standard code, so every
  exception must be caught and mapped deliberately.
- **The blocker:** the `@server.call_tool()` decorator wraps handlers in
  `try/except Exception` → `_make_error_result(...)`, converting every raised exception
  into a `CallToolResult(isError=True)`. Raising `McpError` inside it would never reach
  the wire.

**Decision:** register `handle_call_tool` directly on
`server.request_handlers[types.CallToolRequest]`, bypassing the decorator.

## Step 3 — Confirm Pydantic strict-mode semantics empirically

Rather than assume, ran the matrix for `float` under `strict=True`:

| Input | Result |
|-------|--------|
| `100` (int) | accepted → `100.0` |
| `"100"` | rejected (`float_type`) |
| `True` | rejected (`float_type`) |
| `NaN`, `±inf` | rejected via `allow_inf_nan=False` |
| `0`, `-1` | rejected via `gt=0` |

Exactly the wanted behaviour: no string/bool coercion, but JSON integers still work for a
float field.

## Step 4 — Write `models.py`

- `CUSTOMER_ID_PATTERN = ^CUST-[A-Z0-9]{5}$` — "XXXXX" read as five uppercase alphanumerics.
- Shared config: `extra="forbid"` (unknown args are errors), `strict=True` (no coercion),
  `frozen=True` (validated args are immutable).
- `amount`: `gt=0`, `allow_inf_nan=False`.
- `reason`: `min_length=10`, `max_length=500`, plus a `field_validator` rejecting
  whitespace-only strings — ten spaces passes `min_length` but is not a reason.
- Output models `CustomerRecord` / `RefundReceipt` so `outputSchema` and
  `structuredContent` can be advertised.
- `format_validation_errors()` flattens `ValidationError` to `{field, message, type}` and
  **drops the `input` value**, which may be PII.

## Step 5 — Write `server.py`

1. `reserve_stdout_for_protocol()` — `os.dup(1)` for the transport, `os.dup2(2, 1)` to
   repoint fd 1 at stderr, `sys.stdout = sys.stderr` for the Python-level path. Verified
   `stdio_server()` accepts an explicit `stdout=`, so the private handle can be injected.
2. `configure_logging()` — `stream=sys.stderr`, level from `MCP_LOG_LEVEL`, `force=True`.
3. In-memory store plus an `asyncio.Lock`, because `tools/call` requests are dispatched
   concurrently via `tg.start_soon`.
4. `ToolExecutionError` for business failures, carrying a machine-readable `code`.
5. `_TOOL_REGISTRY: name -> (model, coroutine)` so dispatch is a lookup, not a chain of ifs.
6. `handle_call_tool`: unknown tool → `-32602`; non-object `arguments` → `-32602`;
   `ValidationError` → `-32602` with structured `data`; `ToolExecutionError` → errored tool
   result; anything else → `-32603` with the traceback logged, not transmitted.
7. `TOOLS` built from `model_json_schema()`, so the advertised contract cannot drift.

## Step 6 — Write the test suite

Three classes mirroring the three scoring criteria, plus an interop test:

- `TestValidationDepth` — field-by-field edge cases (51 tests).
- `TestStdioIsolation` — stdout purity across a session, plus a subprocess probe that runs
  the fd swap then fires `print()` and `sys.stdout.write()` and asserts neither escapes.
- `TestProtocolCompliance` — one subprocess session covering 11 request ids, asserting the
  exact code on each reply.
- `test_official_client_can_drive_the_server` — the real SDK `ClientSession` over stdio.

## Step 7 — Fix what the tests found

First run: **7 failures.** Replies for the later request ids never arrived.

Root cause, from `Server.run()`:

```python
finally:
    # Transport closed: cancel in-flight handlers.
    tg.cancel_scope.cancel()
```

The harness used `subprocess.run(input=...)`, which closes stdin immediately; the SDK then
cancelled handlers that had already computed their answers. **Not a server bug** — a real
client keeps stdin open until its replies arrive.

Rewrote `_drive_server` to behave like a real client: `Popen`, write the frames, hold stdin
open and read replies until every expected id has been seen, *then* close stdin. stderr is
routed to a temp file rather than a pipe so a chatty server cannot deadlock on a full pipe
buffer, and a `threading.Timer` watchdog kills a hung process.

Also moved a class-scoped fixture to module scope to clear a `PytestRemovedIn10Warning`.

**Second run: 68 passed, 0 warnings.**

## Step 8 — Document

`README.md` covering the tool contracts, the five design decisions, a definition-of-done
table mapping each criterion to the tests that prove it, and the known edges
(int-for-float, no `-32700` for unparseable lines, `-32602` for unknown methods).

## Step 9 — Fix a money bug found on review

Reading back through `trigger_refund`, the balance check and the debit were not the same
number:

```python
if round(args.amount, 2) > round(balance, 2):   # compared rounded
    ...
remaining = round(balance - args.amount, 2)     # debited unrounded
```

Reproduced it before changing anything. Against `CUST-AB123`, whose refundable balance is
`0.00`:

```
refund 0.004 -> ACCEPTED amount=0.0 remaining=-0.0
refund 0.004 -> ACCEPTED amount=0.0 remaining=-0.0
```

Three distinct faults from one cause:

1. **A zero balance could be overdrawn.** `round(0.004, 2)` is `0.0`, so the guard passed.
2. **Any balance could be overdrawn by just under half a cent** — `250.004` against
   `250.00` was accepted.
3. **The receipt understated the request.** It recorded `amount: 250.0` while `250.004`
   had been debited, so the receipt, the ledger and the balance disagreed. Balances also
   landed on `-0.0`.

Wrote six failing tests first (`TestRefundArithmetic`) plus schema cases, confirmed **6
red**, then fixed in two places:

- **`models.py`** — reject an `amount` with more than 2 decimal places, so it fails at the
  protocol boundary as `-32602`. The check reads the exponent of `Decimal(str(value))`,
  whose shortest round-trip repr reflects what the caller sent rather than binary float
  noise; `1.1` passes, `10.005` does not.
- **`server.py`** — compare and debit in **whole cents**. `requested_cents > balance_cents`
  and `remaining = (balance_cents - requested_cents) / 100` are exact, symmetric by
  construction, and cannot produce `-0.0`. A request rounding to zero cents is refused as
  `amount_below_minimum`, so the rule still holds for a caller that reaches the tool
  without passing through the schema.

`reset_store()` had also been dead code with a docstring claiming the test suite used it.
`TestRefundArithmetic` now genuinely uses it as an autouse fixture, so the claim is true.

**85 passed** (was 68). ruff and mypy clean.
