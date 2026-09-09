# Task 4 — Rate-limiting & model fallback router

A model-routing gateway that keeps serving when the primary provider does not: a
token-aware sliding-window limiter backed by on-disk SQLite, automatic failover on `429`
or a 3000 ms deadline, and one standardised error payload for every failure.

| File | Purpose |
|------|---------|
| `store.py` | `TokenLedger`: sliding-window token accounting in on-disk SQLite. |
| `router.py` | Providers, failover logic, token estimation, and the `GatewayError` vocabulary. |
| `app.py` | FastAPI surface: authenticate → reserve → route → reconcile. |
| `test_router.py` | 67 tests across the window, concurrency, failover, timeouts and sanitisation. |

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # optional; mock providers are the default

python app.py                 # :8100
pytest -v                     # no key, no network needed
```

```bash
curl -s localhost:8100/v1/completions \
  -H 'Authorization: Bearer tenant-alpha' -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}],"max_tokens":128}'

curl -s localhost:8100/v1/usage -H 'Authorization: Bearer tenant-alpha'
```

Watch failover happen by making the primary misbehave:

```bash
PRIMARY_BEHAVIOUR=rate_limited python app.py   # 429 -> fallback
PRIMARY_DELAY_MS=5000       python app.py      # blows the 3000ms deadline -> fallback
```

The response says which provider served it:

```json
{ "provider": "fallback-mock", "failed_over": true, "failover_reason": "429 too many requests", ... }
```

---

## Request lifecycle

```
Bearer key ──▶ authenticate
                   │
                   ▼
            estimate tokens (prompt + max_tokens)
                   │
                   ▼
            ledger.reserve()  ──refused──▶ 429 + Retry-After
                   │ allowed
                   ▼
            router.complete()
              primary, deadline 3000ms
                   │
        429 / timeout / unreachable
                   ▼
              fallback provider
                   │
                   ▼
            ledger.reconcile(actual tokens)
```

---

## The rate limiter

**One row per request**, holding the tokens it accounted for and when. A window's usage is
the sum of the rows inside it — a genuine sliding window, not fixed buckets. That matters:
a calendar-minute bucket lets a client spend the full budget at `t=59s` and again at
`t=61s`. `test_the_window_is_continuous_not_bucketed` asserts this gateway does not.

**Reserving is one `BEGIN IMMEDIATE` transaction**: evict expired rows, sum the window,
then insert — atomic against other tasks in the process (an `asyncio.Lock`, since the
connection is shared) and against other processes on the same file (SQLite's write lock,
with `busy_timeout`). `test_parallel_reservations_do_not_exceed_the_limit` fires 20
concurrent 1,000-token requests at a 10,000 budget and asserts **exactly ten** win.

**Eviction deletes, it does not just filter.** Expired rows are removed on the read path,
so the table does not grow without bound. A refusal `COMMIT`s rather than `ROLLBACK`s, so
its eviction survives — `test_eviction_also_runs_on_a_refused_request` covers that
specifically, since rolling back would have quietly leaked dead rows.

**Reservations are estimates; the ledger is corrected afterwards.** `estimate_tokens`
reserves `prompt + max_tokens` up front, then `reconcile()` rewrites the row with what the
provider actually charged. Without it, a client asking for `max_tokens: 4096` and getting
50 tokens back would burn 4096 of its budget every time.

**`Retry-After` is computed, not guessed.** The ledger walks the window oldest-first until
enough tokens would have aged out, and reports when that row leaves the window.
`test_retry_after_points_at_the_oldest_relevant_row` waits exactly that long and asserts
the retry succeeds.

---

## Failover

| Primary outcome | Action |
|-----------------|--------|
| Success within 3000 ms | Return it. The fallback is never called. |
| `429 Too Many Requests` | Fail over. |
| Exceeds 3000 ms | Cancel and fail over. |
| Unreachable / connection error | Fail over. |
| `5xx` | **Not** a failover trigger by default. |

`429` and timeout are what the task specifies. Connection errors are included because an
unreachable primary is operationally identical to one that timed out — the difference is
only how quickly you find out. `5xx` is deliberately *not* included: a provider returning
500 for a malformed request would otherwise have every such request replayed against the
fallback.

The deadline is enforced with `asyncio.timeout`, which **cancels** the in-flight call. A
slow primary that finishes later cannot race ahead of the fallback's answer —
`test_the_abandoned_call_cannot_win_the_race` waits past the primary's completion time and
re-asserts the result.

---

## The error contract

Every failure — bad JSON, missing credentials, rate limit, dead providers, an internal bug
— leaves through `GatewayError.to_payload()`:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Token budget exhausted: 50000 tokens per 60s for this API key.",
    "type": "rate_limit_error",
    "request_id": "9f2c1a7b4e0d3c88",
    "retry_after_ms": 41200
  }
}
```

| Code | HTTP | Type |
|------|------|------|
| `missing_credentials`, `invalid_api_key` | 401 | `authentication_error` |
| `invalid_json`, `invalid_request` | 400 | `invalid_request_error` |
| `rate_limit_exceeded` | 429 | `rate_limit_error` |
| `upstream_unavailable`, `all_providers_failed` | 503 | `upstream_error` |
| `internal_error` | 500 | `gateway_error` |

Upstream status codes, bodies, hostnames, keys and tracebacks never appear. The full
detail goes to the log with the same `request_id` the client is given, so a support
question can still be traced end to end.

**Failed requests are refunded down to the prompt estimate**, not to zero. The client is
not billed for a completion it never received, but a client hammering a failing provider
still pays for the attempts.

---

## Definition of done

| Criterion | Where it is proven |
|-----------|--------------------|
| **Async concurrency and timeout race conditions** | `TestConcurrency` (3): 20 parallel reservations against a 10,000 budget admit exactly 10; row ids are unique with none lost or duplicated. `TestTimeoutRace` (4): a 5-second primary is abandoned in under a second, a fast primary is not failed over, and a late response cannot overwrite the fallback's. |
| **Accurate state eviction and token accounting** | `TestSlidingWindow` (15): the limit boundary, a genuinely sliding window, eviction that actually deletes rows, eviction on the refusal path, per-key isolation, and a `Retry-After` that is verified by waiting exactly that long. `TestTokenAccounting` (4) + `TestTokenEstimation` (5) cover reconciliation up and down and refunds. `TestPersistence` (3) reopens the file and finds the usage still there. |
| **Graceful fallback and error sanitisation** | `TestFailover` (6) + `TestGatewayFailover` (3): 429, outage, both-down, and no-fallback-configured. `TestErrorSanitisation` (4) + `TestErrorContract` (3): a planted `sk-live-9f2c-INTERNAL` and `10.0.0.5` appear nowhere in any response, an internal exception becomes a clean 500 with the reservation refunded, and every error carries the same four keys with a `request_id` matching the response header. |

Time-dependent behaviour is tested with an **explicit clock** (`reserve(..., at_ms=...)`)
rather than `sleep`, so window-expiry tests are exact and instant.
