"""Tests for the LLM gateway streaming guardrail.

Layered to match the scoring criteria:

1. ``TestPatternDetection`` / ``TestFalsePositives`` - what the patterns catch and
   what they correctly leave alone.
2. ``TestChunkBoundaries`` - the requirement that matters: PII split across chunk
   boundaries, checked at *every* split point rather than a couple of samples.
3. ``TestBufferDiscipline`` - the tail buffer stays bounded and gets released.
4. ``TestGatewayStreaming`` / ``TestLatency`` - end to end over SSE, including
   evidence that the response is not accumulated before being sent.

Run from this directory: ``pytest -v``
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from llm_gateway import create_app, transform_chunk  # noqa: E402
from providers import MockProvider  # noqa: E402
from redactor import DEFAULT_MAX_HOLDBACK, REPLACEMENT, StreamingRedactor  # noqa: E402

EMAIL = "ada.lovelace@example.com"
SSN = "123-45-6789"
VISA = "4111 1111 1111 1111"
VISA_DASHED = "4111-1111-1111-1111"
VISA_BARE = "4111111111111111"
AMEX = "3782 822463 10005"

PII_SAMPLES = [EMAIL, SSN, VISA, VISA_DASHED, VISA_BARE, AMEX]


def stream_text(text: str, chunk_size: int) -> str:
    """Push text through a fresh redactor in fixed-size pieces and reassemble."""
    redactor = StreamingRedactor()
    pieces = [redactor.feed(text[i : i + chunk_size]) for i in range(0, len(text), chunk_size)]
    pieces.append(redactor.flush())
    return "".join(pieces)


def split_at(text: str, index: int) -> str:
    """Push text through a redactor split into exactly two pieces at ``index``."""
    redactor = StreamingRedactor()
    return redactor.feed(text[:index]) + redactor.feed(text[index:]) + redactor.flush()


# --------------------------------------------------------------------------- #
# 1. detection
# --------------------------------------------------------------------------- #
class TestPatternDetection:
    """Whole-string redaction, before any streaming is involved."""

    @pytest.mark.parametrize(
        "email",
        [
            "ada@example.com",
            "ada.lovelace@example.com",
            "ada+billing@example.co.uk",
            "ada_l@mail.example.org",
            "a@b.io",
            "first.last%tag@sub.domain.example.com",
        ],
    )
    def test_redacts_emails(self, email: str) -> None:
        assert StreamingRedactor().scrub(f"write to {email} today") == f"write to {REPLACEMENT} today"

    @pytest.mark.parametrize("ssn", ["123-45-6789", "123 45 6789", "123456789", "001-01-0001"])
    def test_redacts_ssns(self, ssn: str) -> None:
        assert StreamingRedactor().scrub(f"ssn {ssn} on file") == f"ssn {REPLACEMENT} on file"

    @pytest.mark.parametrize("card", [VISA, VISA_DASHED, VISA_BARE, AMEX, "5555 5555 5555 4444", "6011111111111117"])
    def test_redacts_cards(self, card: str) -> None:
        assert StreamingRedactor().scrub(f"card {card} expires") == f"card {REPLACEMENT} expires"

    def test_redacts_several_matches_in_one_string(self) -> None:
        text = f"{EMAIL} / {SSN} / {VISA}"
        redactor = StreamingRedactor()
        assert redactor.scrub(text) == f"{REPLACEMENT} / {REPLACEMENT} / {REPLACEMENT}"
        assert redactor.counts == {"email": 1, "ssn": 1, "credit_card": 1}

    def test_a_sixteen_digit_run_is_a_card_not_an_ssn(self) -> None:
        """Ordering matters: the card alternative is tried first."""
        redactor = StreamingRedactor()
        redactor.scrub(VISA_BARE)
        assert redactor.counts == {"credit_card": 1}


class TestFalsePositives:
    """Text that merely looks numeric must survive untouched."""

    @pytest.mark.parametrize(
        "text",
        [
            "The year 2024 was busy.",
            "Call 555-1234 for support.",
            "Order 12345 shipped on 2024-01-15.",
            "Version 1.2.3 released.",
            "It cost $1,234.56 in total.",
            "Meet at 10:30 or 11:45.",
            "Use the @mention syntax.",
            "Read docs at https://example.com/guide.",
            "99 bottles of beer on the wall.",
        ],
    )
    def test_ordinary_text_is_untouched(self, text: str) -> None:
        assert StreamingRedactor().scrub(text) == text

    @pytest.mark.parametrize("invalid", ["000-12-3456", "666-12-3456", "900-12-3456", "123-00-6789", "123-45-0000"])
    def test_never_issued_ssn_ranges_are_not_matched(self, invalid: str) -> None:
        """Areas 000/666/9xx, group 00 and serial 0000 are never allocated."""
        assert StreamingRedactor().scrub(f"id {invalid} here") == f"id {invalid} here"

    def test_short_and_long_digit_runs_are_not_cards(self) -> None:
        assert StreamingRedactor().scrub("ref 123456789012 ok") == "ref 123456789012 ok"  # 12 digits
        long_run = "9" * 25
        assert StreamingRedactor().scrub(f"ref {long_run} ok") == f"ref {long_run} ok"


# --------------------------------------------------------------------------- #
# 2. chunk boundaries - the requirement this task exists for
# --------------------------------------------------------------------------- #
class TestChunkBoundaries:
    """A pattern split across chunks must still be caught."""

    @pytest.mark.parametrize("sample", PII_SAMPLES)
    def test_every_possible_two_way_split_is_caught(self, sample: str) -> None:
        """Exhaustive: split the sentence at each index in turn, none may leak."""
        text = f"the value is {sample} and that is all"
        expected = f"the value is {REPLACEMENT} and that is all"
        for index in range(len(text) + 1):
            assert split_at(text, index) == expected, f"leak splitting {sample!r} at index {index}"

    @pytest.mark.parametrize("sample", PII_SAMPLES)
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5, 7, 11, 16])
    def test_fixed_size_chunking_is_caught(self, sample: str, chunk_size: int) -> None:
        text = f"contact {sample} now"
        assert stream_text(text, chunk_size) == f"contact {REPLACEMENT} now"

    def test_one_character_at_a_time(self) -> None:
        """The worst case a token stream can produce."""
        text = f"Mail {EMAIL}, SSN {SSN}, card {VISA_DASHED}. Done."
        expected = f"Mail {REPLACEMENT}, SSN {REPLACEMENT}, card {REPLACEMENT}. Done."
        assert stream_text(text, 1) == expected

    def test_pii_at_the_very_end_is_flushed(self) -> None:
        """Nothing may be left in the buffer when the stream ends."""
        assert stream_text(f"card {VISA}", 3) == f"card {REPLACEMENT}"
        assert stream_text(EMAIL, 1) == REPLACEMENT

    def test_pii_at_the_very_start(self) -> None:
        assert stream_text(f"{EMAIL} is the address", 2) == f"{REPLACEMENT} is the address"

    @pytest.mark.parametrize("sample", PII_SAMPLES)
    def test_the_raw_value_never_appears_in_the_output(self, sample: str) -> None:
        """Stronger than equality: no fragment of the emitted text contains it."""
        for chunk_size in (1, 3, 8):
            assert sample not in stream_text(f"value {sample} end", chunk_size)


# --------------------------------------------------------------------------- #
# 3. buffer discipline
# --------------------------------------------------------------------------- #
class TestBufferDiscipline:
    """Memory stays bounded and text is not held longer than it must be."""

    def test_clean_prose_is_released_immediately(self) -> None:
        """No PII candidate in the tail means no latency cost at all."""
        redactor = StreamingRedactor()
        assert redactor.feed("Hello there, how can I help you today?") == "Hello there, how can I help you today?"
        assert redactor.pending == 0

    def test_partial_match_is_held_back_then_released(self) -> None:
        """Text before the candidate goes out at once; only the candidate waits."""
        redactor = StreamingRedactor()
        assert redactor.feed("mail me at ada.love") == "mail me at "
        assert redactor.pending == len("ada.love")

        # "done" is itself a candidate tail, so it stays behind until the stream ends.
        assert redactor.feed("lace@example.com done") == f"{REPLACEMENT} "
        assert redactor.pending == len("done")
        assert redactor.flush() == "done"

    def test_buffer_never_exceeds_max_holdback(self) -> None:
        """A pathological unbroken run must not grow the buffer without limit."""
        redactor = StreamingRedactor()
        for _ in range(500):
            redactor.feed("1234567890" * 10)  # 100 digits at a time, no separators
            assert redactor.pending <= DEFAULT_MAX_HOLDBACK

    def test_buffer_is_bounded_for_email_like_runs(self) -> None:
        redactor = StreamingRedactor()
        for _ in range(200):
            redactor.feed("a.b_c%d+e-" * 10)
            assert redactor.pending <= DEFAULT_MAX_HOLDBACK

    def test_holdback_is_configurable(self) -> None:
        redactor = StreamingRedactor(max_holdback=8)
        redactor.feed("x" + "9" * 100)
        assert redactor.pending <= 8

    def test_rejects_a_nonsensical_holdback(self) -> None:
        with pytest.raises(ValueError):
            StreamingRedactor(max_holdback=0)

    def test_empty_feed_is_a_no_op(self) -> None:
        redactor = StreamingRedactor()
        assert redactor.feed("") == ""
        assert redactor.flush() == ""

    def test_flush_is_idempotent(self) -> None:
        redactor = StreamingRedactor()
        assert redactor.feed("card 4111 1111 1111 1111") == "card "
        assert redactor.flush() == REPLACEMENT
        assert redactor.flush() == ""

    def test_counts_accumulate_across_feeds(self) -> None:
        redactor = StreamingRedactor()
        redactor.feed(f"{EMAIL} and ")
        redactor.feed(f"{SSN} and ")
        redactor.feed(f"{VISA} .")
        redactor.flush()
        assert redactor.counts == {"email": 1, "ssn": 1, "credit_card": 1}
        assert redactor.total_redactions == 3


class TestChunkTransform:
    """The delta-rewriting rules, checked directly."""

    def test_held_back_chunks_are_dropped_rather_than_emitted_empty(self) -> None:
        redactor = StreamingRedactor()
        chunk = {"choices": [{"index": 0, "delta": {"content": "ada@ex"}, "finish_reason": None}]}
        assert transform_chunk(chunk, redactor) is None

    def test_role_only_chunks_pass_through(self) -> None:
        redactor = StreamingRedactor()
        chunk = {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        assert transform_chunk(chunk, redactor) is chunk

    def test_finish_chunk_flushes_the_buffer(self) -> None:
        """The finish frame picks up whatever was still held back."""
        redactor = StreamingRedactor()
        assert redactor.feed("my card is 4111 1111 1111 1111") == "my card is "
        chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        result = transform_chunk(chunk, redactor)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == REPLACEMENT
        assert redactor.pending == 0

    def test_frames_without_choices_pass_through(self) -> None:
        chunk = {"usage": {"total_tokens": 10}}
        assert transform_chunk(chunk, StreamingRedactor()) is chunk


# --------------------------------------------------------------------------- #
# 4. the gateway end to end
# --------------------------------------------------------------------------- #
class ExplodingProvider:
    """Provider that always fails, for the error-handling tests."""

    name = "exploding"

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        raise self._error
        yield ""  # unreachable, but makes this an async generator

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise self._error


def build_client(provider: Any) -> tuple[Any, httpx.AsyncClient]:
    """Wire a gateway with the given provider to an in-process client."""
    app = create_app(provider=provider)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test")
    return app, client


async def drive_asgi(app: Any, body: dict[str, Any]) -> list[tuple[float, bytes]]:
    """Call the ASGI app directly, timestamping each response body message.

    Needed because httpx's ASGITransport buffers the whole body before returning,
    which hides *when* the gateway produced each piece.
    """
    payload = json.dumps(body).encode("utf-8")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},  # 2.4 = server streams without a disconnect listener
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": payload, "more_body": False}

    timings: list[tuple[float, bytes]] = []

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            timings.append((time.perf_counter(), message["body"]))

    await app(scope, receive, send)
    return timings


@pytest_asyncio.fixture
async def gateway() -> AsyncIterator[httpx.AsyncClient]:
    """Gateway backed by a mock that splits its reply every 3 characters."""
    app, client = build_client(MockProvider(chunk_size=3))
    async with app.router.lifespan_context(app), client:
        yield client


async def collect_stream(client: httpx.AsyncClient, prompt: str) -> tuple[str, list[str]]:
    """Drive one streaming request; return (assembled text, raw data payloads)."""
    body = {"model": "test-model", "messages": [{"role": "user", "content": prompt}], "stream": True}
    payloads: list[str] = []
    async with client.stream("POST", "/v1/chat/completions", json=body) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                payloads.append(line[len("data: ") :])

    assembled = ""
    for payload in payloads:
        if payload == "[DONE]":
            continue
        chunk = json.loads(payload)
        for choice in chunk.get("choices") or []:
            assembled += (choice.get("delta") or {}).get("content") or ""
    return assembled, payloads


@pytest.mark.asyncio
class TestGatewayStreaming:
    """SSE proxying with the guardrail applied in flight."""

    async def test_response_is_a_well_formed_event_stream(self, gateway: httpx.AsyncClient) -> None:
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        async with gateway.stream("POST", "/v1/chat/completions", json=body) as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers["cache-control"] == "no-cache, no-transform"
            assert response.headers["x-accel-buffering"] == "no"
            payloads = [line[6:] async for line in response.aiter_lines() if line.startswith("data: ")]
        assert payloads[-1] == "[DONE]"
        assert len(payloads) > 5, "response was not actually streamed in pieces"

    async def test_default_reply_is_fully_redacted(self, gateway: httpx.AsyncClient) -> None:
        assembled, payloads = await collect_stream(gateway, "tell me about the account")
        assert assembled.count(REPLACEMENT) == 3
        raw = "".join(payloads)
        for secret in ("ada.lovelace@example.com", "123-45-6789", "4111 1111 1111 1111"):
            assert secret not in raw

    async def test_pii_split_across_frames_does_not_leak(self, gateway: httpx.AsyncClient) -> None:
        """3-character frames guarantee every value straddles a boundary."""
        prompt = f"echo: reach me at {EMAIL} or on {VISA_DASHED}, ssn {SSN}."
        assembled, payloads = await collect_stream(gateway, prompt)
        assert assembled == f"reach me at {REPLACEMENT} or on {REPLACEMENT}, ssn {REPLACEMENT}."
        raw = "".join(payloads)
        for secret in (EMAIL, VISA_DASHED, SSN):
            assert secret not in raw

    async def test_one_character_frames_do_not_leak(self) -> None:
        app, client = build_client(MockProvider(chunk_size=1))
        async with app.router.lifespan_context(app), client:
            assembled, payloads = await collect_stream(client, f"echo: card {VISA} end")
        assert assembled == f"card {REPLACEMENT} end"
        assert VISA not in "".join(payloads)

    async def test_clean_text_passes_through_unchanged(self, gateway: httpx.AsyncClient) -> None:
        message = "The quick brown fox jumps over the lazy dog on 15 January 2024."
        assembled, _ = await collect_stream(gateway, f"echo: {message}")
        assert assembled == message

    async def test_stream_carries_role_and_finish_markers(self, gateway: httpx.AsyncClient) -> None:
        _, payloads = await collect_stream(gateway, "echo: hello world")
        chunks = [json.loads(p) for p in payloads if p != "[DONE]"]
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    async def test_pii_at_the_end_is_flushed_before_done(self, gateway: httpx.AsyncClient) -> None:
        """The last thing in the response is still redacted, not dropped or leaked."""
        assembled, _ = await collect_stream(gateway, f"echo: the card is {VISA}")
        assert assembled == f"the card is {REPLACEMENT}"

    async def test_healthz_reports_lifetime_counts(self, gateway: httpx.AsyncClient) -> None:
        await collect_stream(gateway, "tell me about the account")
        totals = (await gateway.get("/healthz")).json()["totals"]
        assert totals["email"] == 1 and totals["ssn"] == 1 and totals["credit_card"] == 1


@pytest.mark.asyncio
class TestNonStreaming:
    """The same guardrail applies when the client does not ask for a stream."""

    async def test_content_is_redacted(self, gateway: httpx.AsyncClient) -> None:
        body = {"model": "m", "messages": [{"role": "user", "content": f"echo: mail {EMAIL}"}], "stream": False}
        response = await gateway.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == f"mail {REPLACEMENT}"
        assert response.headers["x-guardrail-redactions"] == "1"

    async def test_stream_defaults_to_false(self, gateway: httpx.AsyncClient) -> None:
        body = {"model": "m", "messages": [{"role": "user", "content": "echo: plain text"}]}
        response = await gateway.post("/v1/chat/completions", json=body)
        assert response.json()["object"] == "chat.completion"


@pytest.mark.asyncio
class TestRequestValidation:
    """Malformed requests are refused with an OpenAI-shaped error."""

    async def test_invalid_json(self, gateway: httpx.AsyncClient) -> None:
        response = await gateway.post(
            "/v1/chat/completions", content=b"{oops", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"

    @pytest.mark.parametrize(
        "body",
        [
            {"messages": [{"role": "user", "content": "hi"}]},  # no model
            {"model": "m"},  # no messages
            {"model": "", "messages": [{"role": "user", "content": "hi"}]},  # empty model
            {"model": "m", "messages": []},  # empty messages
            {"model": "m", "messages": "not-a-list"},
        ],
    )
    async def test_schema_violations_are_rejected(self, gateway: httpx.AsyncClient, body: dict[str, Any]) -> None:
        response = await gateway.post("/v1/chat/completions", json=body)
        assert response.status_code == 400
        assert "error" in response.json()


@pytest.mark.asyncio
class TestUpstreamFailures:
    """Provider faults become clean errors, never tracebacks."""

    SECRET = "sk-live-abcdef123456"

    async def test_streaming_failure_ends_the_stream_cleanly(self) -> None:
        app, client = build_client(ExplodingProvider(httpx.ConnectError(f"boom {self.SECRET}")))
        async with app.router.lifespan_context(app), client:
            _, payloads = await collect_stream(client, "hi")
        assert payloads[-1] == "[DONE]"
        assert json.loads(payloads[0])["error"]["type"] == "upstream_error"
        assert self.SECRET not in "".join(payloads)

    async def test_unexpected_errors_are_sanitised(self) -> None:
        app, client = build_client(ExplodingProvider(RuntimeError(f"internal {self.SECRET}")))
        async with app.router.lifespan_context(app), client:
            _, payloads = await collect_stream(client, "hi")
        raw = "".join(payloads)
        assert json.loads(payloads[0])["error"]["type"] == "gateway_error"
        for fragment in (self.SECRET, "RuntimeError", "Traceback"):
            assert fragment not in raw

    async def test_non_streaming_failure_returns_502(self) -> None:
        app, client = build_client(ExplodingProvider(httpx.ConnectError(f"boom {self.SECRET}")))
        async with app.router.lifespan_context(app), client:
            response = await client.post(
                "/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            )
        assert response.status_code == 502
        assert self.SECRET not in response.text


@pytest.mark.asyncio
class TestLatency:
    """Evidence that the response is relayed, not accumulated."""

    async def test_body_is_emitted_progressively_not_at_the_end(self) -> None:
        """With a 20ms-per-chunk upstream, a buffering proxy would send nothing until the end.

        Driven at the raw ASGI level on purpose: ``httpx.ASGITransport`` collects
        every ``http.response.body`` message into a list before handing back a
        response, so it cannot show when bytes were actually produced.
        """
        message = "echo: " + ("the quick brown fox jumps over the lazy dog. " * 3)
        app = create_app(provider=MockProvider(chunk_size=8, delay_seconds=0.02))
        async with app.router.lifespan_context(app):
            started = time.perf_counter()
            timings = await drive_asgi(app, {"model": "m", "messages": [{"role": "user", "content": message}], "stream": True})
            total = time.perf_counter() - started

        assert len(timings) > 10, "response was delivered as a single body message"
        first_byte = timings[0][0] - started
        last_byte = timings[-1][0] - started
        assert first_byte < total / 2, f"first byte at {first_byte:.3f}s of a {total:.3f}s stream"
        assert last_byte > first_byte, "all body messages arrived at the same instant"

    async def test_holding_back_pii_costs_at_most_one_chunk(self) -> None:
        """The guardrail must not delay the stream beyond the tail it is inspecting."""
        clean = create_app(provider=MockProvider(chunk_size=8, delay_seconds=0.01))
        async with clean.router.lifespan_context(clean):
            timings = await drive_asgi(
                clean,
                {"model": "m", "messages": [{"role": "user", "content": "echo: " + "word " * 40}], "stream": True},
            )
        gaps = [later[0] - earlier[0] for earlier, later in zip(timings, timings[1:], strict=False)]
        assert max(gaps) < 0.2, f"largest gap between body messages was {max(gaps):.3f}s"

    async def test_guardrail_does_not_hold_clean_text(self) -> None:
        """Prose with no PII candidate in the tail is emitted with zero delay."""
        redactor = StreamingRedactor()
        emitted = redactor.feed("Here is a perfectly ordinary sentence. ")
        assert emitted == "Here is a perfectly ordinary sentence. "
        assert redactor.pending == 0
