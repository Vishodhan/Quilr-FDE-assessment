# Task 4 — Implementation plan

The steps actually followed to build the rate-limiting failover router, in order.

---

## Step 0 — Scope

**Asked for:** a resilient model-routing module for an LLM gateway with a token-aware
sliding-window rate limiter (e.g. 50,000 tokens/minute per tenant API key), automatic
failover to a secondary provider on `429` or a 3000 ms timeout, a standardised error
payload with no leaked internals, and on-disk SQLite for state.

**In scope:** the ledger, the router, an HTTP surface to exercise both, and tests covering
concurrency, eviction, timeout races and sanitisation.

**Out of scope:** streaming responses (Task 3), distributed coordination, per-model
pricing, cost accounting in currency.

**Split into three files along the seams that matter:**

- `store.py` — knows about SQLite and time, nothing about HTTP or models.
- `router.py` — knows about providers and failure, nothing about SQLite or HTTP.
- `app.py` — wires them together and speaks HTTP.

That is what allows 30 of the tests to run against the ledger alone and 10 against the
router alone.

---

## Step 1 — Choose a window representation

| Option | Verdict |
|--------|---------|
| Fixed buckets (`minute` key, counter) | One row per tenant, but allows 2× the budget across a boundary: full spend at `t=59s`, full spend again at `t=61s`. Rejected. |
| Sliding log — **one row per request** | Exact, evictable, and the row is the natural place to reconcile actual usage into. Chosen. |
| Approximate sliding window (weighted previous bucket) | Cheaper, but the task scores "accurate state eviction". Rejected. |

Cost of the sliding log is rows-per-window, bounded by request rate and evicted on every
read.

## Step 2 — `store.py`

- Schema: `token_usage(id, api_key, ts_ms, tokens, request_id)` with an index on
  `(api_key, ts_ms)` — the only access pattern.
- `isolation_level=None` so `BEGIN IMMEDIATE` is ours to issue rather than sqlite3's
  implicit transaction fighting us.
- `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`.
- `reserve()` is one transaction: evict → sum → decide → insert. An `asyncio.Lock` wraps
  it because one connection is shared and statements from two tasks would otherwise
  interleave inside the transaction.
- **`at_ms` parameter on every time-dependent method.** This is what lets the tests
  exercise window expiry exactly and instantly, instead of sleeping for 61 seconds.
- `_retry_after_ms()` walks the window oldest-first accumulating tokens until enough would
  have aged out, then returns when that row leaves. A request larger than the whole budget
  gets a full window rather than `0`, since waiting cannot help and `0` would invite a
  hot retry loop.
- `reconcile()` / `release()` for correcting and refunding.

**Two bugs caught by re-reading before running anything:**

1. The refusal path originally did `ROLLBACK`, which would have undone its own eviction —
   dead rows would accumulate for exactly the clients hitting the limit hardest. Changed
   to `COMMIT` (nothing was inserted, so there is nothing to undo) and pinned with
   `test_eviction_also_runs_on_a_refused_request`.
2. `prune()` called `commit()` under `isolation_level=None`, where it is a no-op. Removed.

## Step 3 — `router.py`

- `GatewayError` carries `code`, `message`, `status_code`, `error_type` and optional
  `retry_after_ms`, and knows how to render itself. Making it the *only* way a failure
  leaves is what keeps the payload consistent.
- `UpstreamRateLimited` and `UpstreamUnavailable` as sibling exceptions — the router
  branches on type, so `test_provider_exceptions_are_distinct_types` asserts neither
  subclasses the other.
- `MockProvider` with `ok` / `rate_limited` / `unavailable` / a delay. It is both the
  default backend (so the gateway runs after clone) and what the tests drive.
- `_attempt()` wraps each call in `asyncio.timeout`, which **cancels** the in-flight
  request. That is the answer to the timeout-race criterion: a slow primary that finishes
  later has already been cancelled and cannot overwrite the fallback's result.
- **Failover triggers**, decided explicitly: `429` and timeout are the specified ones;
  connection errors are included because an unreachable primary is operationally identical
  to a timed-out one; `5xx` is deliberately excluded, or a provider returning 500 for a
  malformed request would have every such request replayed against the fallback.
- `estimate_tokens()` is offline (≈4 chars/token). tiktoken was considered and rejected:
  it downloads its encoding on first use, which breaks air-gapped hosts and networkless
  test runs, and `reconcile()` corrects the figure moments later anyway.

## Step 4 — `app.py`

Order: authenticate → parse → estimate → reserve → route → reconcile.

- Reserve **before** routing, so a rate-limited request never reaches a provider.
- On success, reconcile to `response.total_tokens`.
- On a `GatewayError`, reconcile down to the **prompt estimate** rather than refunding in
  full. A full refund would let a client hammer a failing provider for free; charging the
  full reservation would bill for a completion never delivered.
- On an unexpected exception, `release()` the reservation entirely and return a clean 500
  — a gateway bug is not the client's fault.
- `X-RateLimit-Limit` / `-Remaining` / `-Reset` on success *and* on the 429, plus
  `Retry-After` in whole seconds.
- `X-Gateway-Provider` and `X-Gateway-Failover` so a caller can see failover happening
  without parsing the body.

## Step 5 — Tests

Window → accounting → persistence → concurrency → failover → timeout race → HTTP →
sanitisation.

Load-bearing ones:

- `test_parallel_reservations_do_not_exceed_the_limit` — 20 concurrent 1,000-token
  requests against 10,000 must admit exactly 10.
- `test_the_window_is_continuous_not_bucketed` — the case a bucketed limiter gets wrong.
- `test_retry_after_points_at_the_oldest_relevant_row` — waits exactly `retry_after_ms`
  and asserts the retry then succeeds.
- `test_the_abandoned_call_cannot_win_the_race` — sleeps past the cancelled primary's
  completion time and re-asserts the result.
- `test_internal_exceptions_become_a_clean_500` — plants `sk-live-9f2c-INTERNAL` and
  `10.0.0.5` in an exception and asserts neither appears in the response.

## Step 6 — Fix what the tests found

First run: **1 failed, 63 passed, 6 warnings.**

The failure was `test_requests_are_refused_once_the_budget_is_gone`: 40 requests never
tripped a 10,000-token limit. Not a bug — **reconciliation working correctly.** Each
request reserved ~2,003 tokens and was then corrected down to the ~70 it actually cost,
so 40 requests spent about 2,800 tokens in total.

That is the behaviour the design wants, so the test was wrong, not the code. Rewrote it
against a provider that genuinely returns 3,000 completion tokens: three requests succeed,
the fourth is refused with ~988 tokens left — less than the ~3,004 it asked to reserve.
Added `test_an_oversized_request_is_refused_outright` for the single-request case, and
strengthened `test_tenants_have_separate_budgets` to actually exhaust one tenant first
rather than merely assuming it had.

The six warnings were synchronous tests sitting inside `@pytest.mark.asyncio` classes.
Moved them into two plain classes, `TestTokenEstimation` and `TestErrorContract`.

**Second run: 67 passed, 0 warnings.**

## Step 7 — Documentation

`README.md` with the request-lifecycle diagram, the failover table and why `5xx` is not on
it, the error-code table, and a definition-of-done table. Known edges written down:
optimistic in-flight reservations, the offline estimator, the optional tenant allowlist,
and SQLite being a single-node choice.
