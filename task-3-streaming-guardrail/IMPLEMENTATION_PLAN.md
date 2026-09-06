# Task 3 — Implementation plan

The steps actually followed to build the streaming guardrail, in order.

---

## Step 0 — Scope

**Asked for:** an LLM gateway endpoint that forwards text-generation requests to a
provider and streams the response back, redacting emails, SSNs and card numbers to
`[REDACTED]` in real time, without accumulating the full response in memory, keeping TTFT
and stream latency low.

**In scope:** the redaction engine, an OpenAI-compatible SSE endpoint, a pluggable
provider layer, and tests that prove boundary-straddling PII cannot leak.

**Out of scope:** authentication, rate limiting (that is Task 4), redacting tool-call
arguments, and patterns beyond the three named.

**Decision taken up front:** speak **OpenAI-compatible SSE** (`data: {...}` frames ending
in `data: [DONE]`) rather than plain chunked text. The requirement says "parse incoming
stream deltas", and a plain-text stream would sidestep exactly that.

**Second decision:** ship a **mock provider as the default** with a real upstream adapter
behind `LLM_PROVIDER=upstream`. The whole suite then runs offline with no API key, and the
mock can be made deliberately hostile in a way a real provider cannot.

---

## Step 1 — Settle the algorithm before writing code

Three options considered for catching a match split across chunks:

| Approach | Verdict |
|----------|---------|
| Buffer the whole response, redact at the end | Correct, but destroys streaming. Rejected — it is the thing the task forbids. |
| Always hold back the last N characters | Simple, but pays the latency on *every* chunk even when the tail obviously cannot be PII. |
| **Hold back only a suffix that could still grow into a match, capped at N** | Chosen. Zero cost on ordinary prose, bounded memory, no leaks. |

The third needs a way to ask "could this suffix become a match?". Python's `re` has no
partial matching (the third-party `regex` module does, via `partial=True`), so the answer
is a pair of **anchored tail patterns** — one per character set:

- `[A-Za-z0-9._%+@-]+$` for email-shaped runs (no spaces).
- `(?:\d[ -]?)+$` for digit runs that may carry separators (cards, SSNs).

The boundary is the earlier of the two match starts, floored at `len(buffer) - max_holdback`.

## Step 2 — `redactor.py`

- One compiled alternation with named groups, so the buffer is scanned **once** per feed
  and `match.lastgroup` gives the category for counting.
- **Card before SSN before email.** A 16-digit run must be scored as a card, not an SSN
  plus stray digits.
- **`feed()` computes the boundary on raw text, then redacts only the released slice.**
  This keeps `[REDACTED]` out of the buffer, so a replacement can never be re-scanned or
  half-held-back.
- `_TRIGGER_RE = [@\d]` as a pre-filter: no `@` and no digit means no pattern can match,
  so the expensive pass is skipped for most prose.
- `max_holdback` default **256** — above the longest realistic email, so a real match
  never gets force-released, while the buffer stays bounded.
- SSN pattern rejects never-issued ranges (`000`/`666`/`9xx` areas, `00` group, `0000`
  serial) to cut false positives on ordinary 9-digit numbers.
- **No Luhn check on cards.** Considered and rejected: a guardrail that lets a card
  through because its checksum failed is failing open. Fail closed, accept that long
  numeric identifiers get redacted.

Verified interactively before building on it — including feeding a sentence one character
at a time, which reassembled correctly with all three values redacted.

## Step 3 — `providers.py`

One `LLMProvider` protocol, two implementations. Both yield the *payload* of each SSE
`data:` line; parsing deltas is the gateway's job, owning the transport format is theirs.

`MockProvider` splits its reply at a **fixed character count**, not on word boundaries —
that is what guarantees emails and card numbers land across chunk boundaries. `echo: ...`
in the user message returns that text verbatim, so tests drive exact content.

## Step 4 — `llm_gateway.py`

- `transform_chunk()` is a pure function over one chunk plus a redactor, so it is unit
  testable without a stream.
- Rules it applies: feed `delta.content` through the redactor; on `finish_reason`, also
  `flush()`; a chunk whose content is entirely held back returns `None` and is skipped
  rather than emitted as an empty delta.
- If the upstream ends without a finish frame, the gateway **synthesises** a final chunk
  from the surrounding `id`/`model`/`created` so buffered text is never lost.
- `X-Accel-Buffering: no` and `Cache-Control: no-transform`, because intermediate proxies
  buffer event streams by default and would undo the whole design.
- Non-streaming requests use the same redactor via `scrub()`, and report the count in
  `X-Guardrail-Redactions`.
- Errors become one `{"error": ...}` frame followed by `[DONE]`, so client parsers
  terminate normally rather than hanging.

## Step 5 — Tests

Structure: patterns → false positives → **chunk boundaries** → buffer discipline →
gateway → latency.

The load-bearing one is `test_every_possible_two_way_split_is_caught`: for each of six PII
samples, the containing sentence is split at **every index in turn** and the output must be
identical every time. That is 63 boundary tests rather than a couple of hand-picked splits.

## Step 6 — Fix what the tests found

First run: **4 failed, 126 passed.**

Three were wrong expectations of mine, not bugs. I had assumed `flush()` returns the whole
string; in fact `feed()` had already released the safe prefix, and only the held-back tail
comes back from `flush()`. The tests now assert both halves, which documents the behaviour
better than the original ones would have.

The fourth was real, and instructive: **TTFT equalled total stream time.** The cause was
the harness, not the gateway — `httpx.ASGITransport` collects every `http.response.body`
message into `body_parts` and only returns a response once the app has finished, so it
cannot show when bytes were produced.

Rewrote the latency tests to drive the ASGI app directly and timestamp each
`http.response.body`. That surfaced a second harness detail: at `spec_version` `2.3`
Starlette runs `listen_for_disconnect` alongside the stream, so a naive `receive()` that
returns `http.disconnect` cancels the response after one frame. Advertising `2.4` — what
uvicorn sends — takes the direct streaming path.

**Second run: 131 passed.** First body message now arrives well inside half the total
stream time, with no gap over 200 ms.

## Step 7 — Documentation

`README.md` with the buffer diagram, the pattern table, the seven design decisions, and a
definition-of-done table. Both harness gotchas are written down under "Notes and known
edges", along with the `max_holdback` trade-off, since the next person to touch this will
hit them.

## Step 8 — Rename for cross-task collection

Running `pytest` from the repository root failed at collection: Task 2 also had a
`gateway.py`, and since both test files put their own directory on `sys.path`, whichever
imported first claimed `sys.modules["gateway"]` for both.

Renamed this one to `llm_gateway.py` (Task 2's became `mcp_gateway.py`) and updated the
imports, docs and run commands. Each folder still stands alone, and the whole repository
now collects in one run: **359 passed**.
