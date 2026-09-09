# Task 3 — LLM gateway streaming guardrail (PII redaction)

An OpenAI-compatible gateway that streams model output back to the client while scrubbing
emails, SSNs and card numbers **in flight** — including values split across chunk
boundaries.

| File | Purpose |
|------|---------|
| `redactor.py` | `StreamingRedactor`: bounded tail buffer, single-pass matching. The whole idea lives here. |
| `providers.py` | `MockProvider` (offline, splits mid-PII on purpose) and `UpstreamProvider` (any OpenAI-compatible API). |
| `llm_gateway.py` | FastAPI SSE proxy: parses deltas, applies the guardrail, re-emits immediately. |
| `test_guardrail.py` | 131 tests, 63 of them dedicated to chunk-boundary splitting. |

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # optional; the mock provider is the default

python llm_gateway.py             # :8000
pytest -v                     # no key, no network needed
```

```bash
curl -N localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"demo","stream":true,"messages":[{"role":"user","content":"account details"}]}'
```

```
data: {"choices":[{"delta":{"role":"assistant","content":""},...}]}
data: {"choices":[{"delta":{"content":"Sure - I found the account. The contact address is "},...}]}
data: {"choices":[{"delta":{"content":"[REDACTED]"},...}]}
...
data: [DONE]
```

`echo: <text>` as the user message makes the mock return `<text>` verbatim, which is handy
for trying your own payloads:

```bash
curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"demo","stream":true,"messages":[{"role":"user","content":"echo: card 4111 1111 1111 1111 ok"}]}'
```

To point at a real provider: `LLM_PROVIDER=upstream`, `LLM_API_KEY=...`, and
`LLM_BASE_URL` if it is not OpenAI.

---

## How the guardrail works

The problem: `4111 1111 1111` arrives in one chunk and `1111` in the next. Neither half
matches. Buffering the whole response would fix it and destroy the point of streaming.

The answer is a **bounded tail buffer**. On every `feed()`:

1. Append the new text to the buffer.
2. Find the **longest suffix that could still grow into a match**, using two cheap
   anchored patterns — one for email-shaped runs (`[A-Za-z0-9._%+@-]+$`), one for
   digit runs that may carry separators (`(?:\d[ -]?)+$`).
3. Emit everything *before* that suffix, redacted. Keep the suffix.

```
buffer:  "...the card is 4111 1111 11"
                         └──────────┘ digit-tail candidate → held back
         └────────────────┘ emitted now
```

Why this is both safe and fast:

- **Safe** — a partial match can never be emitted, because the boundary is placed at or
  before the start of any candidate run.
- **Fast** — in ordinary prose the candidate suffix is the last word, so the delay is a
  few characters. `"Hello there, how can I help you today?"` is released with a holdback
  of **zero**.
- **Bounded** — the suffix is capped at `max_holdback` (256 chars). Memory does not grow
  with the length of the response, however long the model rambles.

A one-line pre-filter (`re.search(r"[@\d]")`) skips the matching pass entirely for text
that cannot possibly contain a match, which is most text.

### Patterns

| Kind | Matches | Deliberately does not match |
|------|---------|------------------------------|
| `credit_card` | 13–19 digits, optional single `space`/`-` separators (Visa, MC, Amex, Discover) | 12-digit refs, 20+ digit runs |
| `ssn` | `123-45-6789`, `123 45 6789`, `123456789` | areas `000`/`666`/`9xx`, group `00`, serial `0000` |
| `email` | `first.last%tag@sub.domain.example.co.uk` | bare `@mention`, URLs |

All three live in **one compiled alternation**, so the buffer is scanned once per feed,
not three times. Card is tried first so a 16-digit run is never mis-scored as an SSN plus
stray digits.

---

## Design decisions

### The redactor knows nothing about SSE

`StreamingRedactor` is `feed()` / `flush()` / `scrub()` over plain strings. That is why 90
of the tests need no HTTP, no async and no provider — and why the same object serves the
streaming and the non-streaming path.

### Fail closed on card numbers

Any 13–19 digit run is redacted, with no Luhn check. Luhn would cut false positives, but a
guardrail that lets an invalid-looking card through because the checksum failed is the
wrong trade. Long numeric identifiers are the acceptable cost.

### A held-back chunk is dropped, not emitted empty

If a delta's entire content is still in the buffer, `transform_chunk` returns `None` and
the frame is skipped, rather than putting `{"content": ""}` on the wire.

### The finish frame flushes

`finish_reason != null` triggers `flush()` and carries the remaining text, so PII at the
very end of a response is redacted rather than silently dropped. If the upstream ends
*without* a finish frame, the gateway synthesises one from the surrounding chunks.

### One redactor per response

State is per-stream, so instances are never shared across concurrent requests.

### Errors never leak

Upstream failures become one `{"error": {...}}` SSE frame followed by `[DONE]`, so the
client's parser terminates normally instead of hanging. Only the exception *class* is
logged; the message is not.

### `yield` never happens inside `finally`

The stream generator's `finally` only logs. Yielding from it would raise
`RuntimeError: async generator ignored GeneratorExit` the moment a client disconnects
mid-stream.

---

## Definition of done

| Criterion | Where it is proven |
|-----------|--------------------|
| **Efficient async chunking and buffer-state management** | `TestBufferDiscipline` (9) + `TestChunkTransform` (4): held-back text is released as soon as it is safe, `flush()` is idempotent, and the buffer stays `<= max_holdback` under 50,000 characters of unbroken digits and of email-shaped runs. |
| **Performant matching over partial text** | `TestChunkBoundaries` (63): for every sample, the sentence is split at **every index in turn** and none may leak; plus fixed chunk sizes 1–16 and a one-character-at-a-time torture test. `TestFalsePositives` (15) checks years, prices, versions, phone numbers, order ids, timestamps, URLs and never-issued SSN ranges survive untouched. |
| **Memory efficiency and low-latency proxying** | `TestLatency` (3): the ASGI app is driven directly and each `http.response.body` message timestamped — the first byte arrives in under half the total stream time, gaps between messages stay under 200 ms, and clean prose is released with zero holdback. |

The latency tests bypass `httpx.ASGITransport` on purpose: it collects every body message
into a list before returning a response, so it cannot show *when* bytes were produced. The
raw-ASGI harness timestamps each `http.response.body` as the gateway emits it.
