"""Tests for the rate-limiting and model-failover gateway.

Layered to match the scoring criteria:

1. ``TestSlidingWindow`` / ``TestTokenAccounting`` / ``TestPersistence`` - the
   limiter's window, eviction and reconciliation, against a real on-disk file.
2. ``TestConcurrency`` - many simultaneous reservations must not oversubscribe.
3. ``TestFailover`` / ``TestTimeoutRace`` - 429, timeout and outage handling.
4. ``TestErrorSanitisation`` - nothing internal ever reaches the client.

Run from this directory: ``pytest -v``
"""

from __future__ import annotations

import asyncio
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

from app import create_app  # noqa: E402
from router import (  # noqa: E402
    GatewayError,
    MockProvider,
    ModelRouter,
    UpstreamRateLimited,
    UpstreamUnavailable,
    estimate_tokens,
)
from store import TokenLedger, now_ms  # noqa: E402

API_KEY = "tenant-alpha-key"
OTHER_KEY = "tenant-beta-key"


def body(content: str = "hello there", max_tokens: int = 100) -> dict[str, Any]:
    """A minimal valid completion request."""
    return {"model": "test-model", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}


@pytest_asyncio.fixture
async def ledger(tmp_path: Path) -> AsyncIterator[TokenLedger]:
    """A real on-disk ledger, 10,000 tokens per 60s."""
    store = TokenLedger(tmp_path / "usage.db", limit_tokens=10_000, window_seconds=60)
    await store.open()
    try:
        yield store
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# 1. the sliding window
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestSlidingWindow:
    """Window arithmetic and eviction, driven by an explicit clock."""

    async def test_reservations_under_the_limit_are_allowed(self, ledger: TokenLedger) -> None:
        reservation = await ledger.reserve(API_KEY, 4_000)
        assert reservation.allowed is True
        assert reservation.used_tokens == 4_000
        assert reservation.remaining_tokens == 6_000

    async def test_usage_accumulates(self, ledger: TokenLedger) -> None:
        await ledger.reserve(API_KEY, 3_000)
        await ledger.reserve(API_KEY, 3_000)
        usage = await ledger.usage(API_KEY)
        assert usage.used_tokens == 6_000
        assert usage.requests == 2

    async def test_exceeding_the_limit_is_refused(self, ledger: TokenLedger) -> None:
        await ledger.reserve(API_KEY, 9_000)
        refused = await ledger.reserve(API_KEY, 2_000)
        assert refused.allowed is False
        assert refused.used_tokens == 9_000  # the refused request is not counted
        assert refused.retry_after_ms > 0

    async def test_the_limit_itself_is_allowed(self, ledger: TokenLedger) -> None:
        """Exactly at the limit passes; one token more does not."""
        assert (await ledger.reserve(API_KEY, 10_000)).allowed is True
        assert (await ledger.reserve(API_KEY, 1)).allowed is False

    async def test_the_window_slides(self, ledger: TokenLedger) -> None:
        """Budget spent 61 seconds ago no longer counts against the window."""
        start = now_ms()
        await ledger.reserve(API_KEY, 9_000, at_ms=start)
        assert (await ledger.reserve(API_KEY, 5_000, at_ms=start + 30_000)).allowed is False
        assert (await ledger.reserve(API_KEY, 5_000, at_ms=start + 61_000)).allowed is True

    async def test_the_window_is_continuous_not_bucketed(self, ledger: TokenLedger) -> None:
        """A fixed-bucket limiter would let this through; a sliding window does not."""
        start = now_ms()
        await ledger.reserve(API_KEY, 6_000, at_ms=start + 59_000)
        # 1s later a calendar-minute bucket would have reset. It must not.
        assert (await ledger.reserve(API_KEY, 6_000, at_ms=start + 60_000)).allowed is False

    async def test_expired_rows_are_deleted_not_just_ignored(self, ledger: TokenLedger) -> None:
        """Eviction must reclaim disk, not merely filter on read."""
        start = now_ms()
        for _ in range(5):
            await ledger.reserve(API_KEY, 100, at_ms=start)
        assert await ledger.row_count() == 5

        await ledger.reserve(API_KEY, 100, at_ms=start + 61_000)
        assert await ledger.row_count() == 1  # the five old rows are gone

    async def test_eviction_also_runs_on_a_refused_request(self, ledger: TokenLedger) -> None:
        """A denial still commits its eviction rather than rolling it back."""
        start = now_ms()
        await ledger.reserve(API_KEY, 9_000, at_ms=start)
        await ledger.reserve(API_KEY, 9_000, at_ms=start + 61_000)
        refused = await ledger.reserve(API_KEY, 9_000, at_ms=start + 61_100)
        assert refused.allowed is False
        assert await ledger.row_count() == 1  # the first row was evicted, not retained

    async def test_prune_clears_every_key(self, ledger: TokenLedger) -> None:
        start = now_ms()
        await ledger.reserve(API_KEY, 100, at_ms=start)
        await ledger.reserve(OTHER_KEY, 100, at_ms=start)
        assert await ledger.prune(at_ms=start + 61_000) == 2
        assert await ledger.row_count() == 0

    async def test_keys_are_isolated(self, ledger: TokenLedger) -> None:
        await ledger.reserve(API_KEY, 10_000)
        assert (await ledger.reserve(OTHER_KEY, 10_000)).allowed is True

    async def test_retry_after_points_at_the_oldest_relevant_row(self, ledger: TokenLedger) -> None:
        """Waiting exactly retry_after_ms must actually be enough."""
        start = now_ms()
        await ledger.reserve(API_KEY, 6_000, at_ms=start)
        await ledger.reserve(API_KEY, 4_000, at_ms=start + 10_000)

        refused = await ledger.reserve(API_KEY, 3_000, at_ms=start + 20_000)
        assert refused.allowed is False
        retry_at = start + 20_000 + refused.retry_after_ms
        assert (await ledger.reserve(API_KEY, 3_000, at_ms=retry_at)).allowed is True

    async def test_a_request_larger_than_the_budget_gets_a_full_window_backoff(self, ledger: TokenLedger) -> None:
        """Waiting cannot help, so report the window rather than zero."""
        refused = await ledger.reserve(API_KEY, 20_000)
        assert refused.allowed is False
        assert refused.retry_after_ms == 60_000

    @pytest.mark.parametrize("bad", [(0, 60), (10, 0), (-1, 60)])
    async def test_rejects_nonsensical_configuration(self, tmp_path: Path, bad: tuple[int, int]) -> None:
        with pytest.raises(ValueError):
            TokenLedger(tmp_path / "x.db", limit_tokens=bad[0], window_seconds=bad[1])


@pytest.mark.asyncio
class TestTokenAccounting:
    """Reservations are estimates; the ledger is corrected against reality."""

    async def test_reconcile_lowers_usage(self, ledger: TokenLedger) -> None:
        reservation = await ledger.reserve(API_KEY, 5_000)
        await ledger.reconcile(reservation, 1_200)
        assert (await ledger.usage(API_KEY)).used_tokens == 1_200

    async def test_reconcile_raises_usage_when_the_estimate_was_low(self, ledger: TokenLedger) -> None:
        reservation = await ledger.reserve(API_KEY, 500)
        await ledger.reconcile(reservation, 4_000)
        assert (await ledger.usage(API_KEY)).used_tokens == 4_000

    async def test_release_refunds_completely(self, ledger: TokenLedger) -> None:
        reservation = await ledger.reserve(API_KEY, 5_000)
        await ledger.release(reservation)
        assert (await ledger.usage(API_KEY)).used_tokens == 0
        assert await ledger.row_count() == 0

    async def test_reconciling_a_refused_reservation_is_a_no_op(self, ledger: TokenLedger) -> None:
        await ledger.reserve(API_KEY, 10_000)
        refused = await ledger.reserve(API_KEY, 5_000)
        await ledger.reconcile(refused, 100)
        await ledger.release(refused)
        assert (await ledger.usage(API_KEY)).used_tokens == 10_000


class TestTokenEstimation:
    """The pre-flight estimate that gets reserved before a request is routed."""

    def test_estimate_scales_with_prompt_length(self) -> None:
        short = estimate_tokens(body("hi", max_tokens=50))
        long = estimate_tokens(body("word " * 400, max_tokens=50))
        assert long.prompt > short.prompt
        assert short.completion == long.completion == 50

    def test_estimate_uses_the_requested_completion_budget(self) -> None:
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 900}
        assert estimate_tokens(payload).completion == 900

    def test_estimate_falls_back_to_a_default_completion(self) -> None:
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        assert estimate_tokens(payload).completion == 256

    def test_an_empty_prompt_still_costs_something(self) -> None:
        assert estimate_tokens({"model": "m", "messages": []}).prompt >= 1

    def test_total_is_prompt_plus_completion(self) -> None:
        estimate = estimate_tokens(body("some words here", max_tokens=120))
        assert estimate.total == estimate.prompt + estimate.completion


@pytest.mark.asyncio
class TestPersistence:
    """The requirement is on-disk SQLite, so state must outlive the process."""

    async def test_usage_survives_reopening(self, tmp_path: Path) -> None:
        path = tmp_path / "persist.db"
        first = TokenLedger(path, limit_tokens=10_000, window_seconds=60)
        await first.open()
        await first.reserve(API_KEY, 7_500)
        await first.close()

        second = TokenLedger(path, limit_tokens=10_000, window_seconds=60)
        await second.open()
        try:
            assert (await second.usage(API_KEY)).used_tokens == 7_500
            assert (await second.reserve(API_KEY, 5_000)).allowed is False
        finally:
            await second.close()

    async def test_the_database_file_is_actually_created(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "usage.db"
        async with TokenLedger(path, limit_tokens=100, window_seconds=60) as store:
            await store.reserve(API_KEY, 10)
        assert path.exists() and path.stat().st_size > 0

    async def test_using_a_closed_ledger_is_a_clear_error(self, tmp_path: Path) -> None:
        store = TokenLedger(tmp_path / "closed.db")
        with pytest.raises(RuntimeError, match="open"):
            await store.reserve(API_KEY, 1)


@pytest.mark.asyncio
class TestConcurrency:
    """Simultaneous reservations must not oversubscribe the window."""

    async def test_parallel_reservations_do_not_exceed_the_limit(self, ledger: TokenLedger) -> None:
        """20 tasks want 1,000 each against a 10,000 budget: exactly 10 may win."""
        results = await asyncio.gather(*(ledger.reserve(API_KEY, 1_000) for _ in range(20)))
        allowed = [r for r in results if r.allowed]
        assert len(allowed) == 10
        assert (await ledger.usage(API_KEY)).used_tokens == 10_000

    async def test_parallel_reservations_across_keys_are_independent(self, ledger: TokenLedger) -> None:
        await asyncio.gather(
            *(ledger.reserve(API_KEY, 1_000) for _ in range(15)),
            *(ledger.reserve(OTHER_KEY, 1_000) for _ in range(15)),
        )
        assert (await ledger.usage(API_KEY)).used_tokens == 10_000
        assert (await ledger.usage(OTHER_KEY)).used_tokens == 10_000

    async def test_no_reservation_is_lost_or_duplicated(self, ledger: TokenLedger) -> None:
        results = await asyncio.gather(*(ledger.reserve(API_KEY, 100) for _ in range(40)))
        row_ids = [r.row_id for r in results if r.allowed]
        assert len(row_ids) == len(set(row_ids)) == 40
        assert await ledger.row_count() == 40


# --------------------------------------------------------------------------- #
# 3. failover
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestFailover:
    """The primary gets one bounded attempt; the secondary picks up the pieces."""

    async def test_healthy_primary_is_used_and_secondary_is_not_touched(self) -> None:
        primary = MockProvider("primary", "ok")
        secondary = MockProvider("fallback", "ok")
        result = await ModelRouter(primary, secondary).complete(body(), "req-1")

        assert result.provider == "primary"
        assert result.failed_over is False
        assert secondary.calls == 0

    async def test_429_triggers_failover(self) -> None:
        primary = MockProvider("primary", "rate_limited")
        secondary = MockProvider("fallback", "ok")
        result = await ModelRouter(primary, secondary).complete(body(), "req-2")

        assert result.provider == "fallback"
        assert result.failed_over is True
        assert result.failover_reason == "429 too many requests"
        assert secondary.calls == 1

    async def test_unreachable_primary_triggers_failover(self) -> None:
        primary = MockProvider("primary", "unavailable")
        result = await ModelRouter(primary, MockProvider("fallback", "ok")).complete(body(), "req-3")
        assert result.failed_over is True
        assert result.failover_reason == "provider unavailable"

    async def test_both_providers_failing_yields_a_gateway_error(self) -> None:
        router = ModelRouter(MockProvider("primary", "rate_limited"), MockProvider("fallback", "unavailable"))
        with pytest.raises(GatewayError) as exc:
            await router.complete(body(), "req-4")
        assert exc.value.code == "all_providers_failed"
        assert exc.value.status_code == 503

    async def test_no_fallback_configured_yields_a_gateway_error(self) -> None:
        with pytest.raises(GatewayError) as exc:
            await ModelRouter(MockProvider("primary", "rate_limited"), None).complete(body(), "req-5")
        assert exc.value.code == "upstream_unavailable"

    async def test_rejects_a_nonsensical_timeout(self) -> None:
        with pytest.raises(ValueError):
            ModelRouter(MockProvider("primary"), timeout_seconds=0)


@pytest.mark.asyncio
class TestTimeoutRace:
    """The 3000ms deadline, and what happens to the abandoned call."""

    async def test_slow_primary_is_abandoned_and_the_fallback_answers(self) -> None:
        primary = MockProvider("primary", "ok", delay_seconds=5.0)
        secondary = MockProvider("fallback", "ok")
        router = ModelRouter(primary, secondary, timeout_seconds=0.2)

        started = time.perf_counter()
        result = await router.complete(body(), "req-6")
        elapsed = time.perf_counter() - started

        assert result.provider == "fallback"
        assert result.failover_reason == "timeout after 200ms"
        # The 5s primary must not have been waited out.
        assert elapsed < 1.0, f"waited {elapsed:.2f}s, so the deadline did not fire"

    async def test_a_primary_inside_the_deadline_is_not_failed_over(self) -> None:
        primary = MockProvider("primary", "ok", delay_seconds=0.05)
        secondary = MockProvider("fallback", "ok")
        result = await ModelRouter(primary, secondary, timeout_seconds=0.5).complete(body(), "req-7")
        assert result.failed_over is False
        assert secondary.calls == 0

    async def test_the_abandoned_call_cannot_win_the_race(self) -> None:
        """A late primary response must not overwrite the fallback's answer."""
        primary = MockProvider("primary", "ok", delay_seconds=0.3)
        secondary = MockProvider("fallback", "ok")
        result = await ModelRouter(primary, secondary, timeout_seconds=0.1).complete(body(), "req-8")

        assert result.provider == "fallback"
        await asyncio.sleep(0.4)  # well past when the primary would have finished
        assert result.provider == "fallback"

    async def test_default_deadline_is_three_seconds(self) -> None:
        assert ModelRouter(MockProvider("p")).timeout_seconds == 3.0


# --------------------------------------------------------------------------- #
# 4. the HTTP surface
# --------------------------------------------------------------------------- #
def build_gateway(ledger: TokenLedger, router: ModelRouter) -> tuple[Any, httpx.AsyncClient]:
    """Wire the app to an in-process client with the given dependencies."""
    application = create_app(ledger=ledger, router=router)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://gateway.test")
    return application, client


@pytest_asyncio.fixture
async def gateway(ledger: TokenLedger) -> AsyncIterator[httpx.AsyncClient]:
    """Gateway with a healthy primary and a healthy fallback."""
    application, client = build_gateway(ledger, ModelRouter(MockProvider("primary", "ok"), MockProvider("fallback", "ok")))
    async with application.router.lifespan_context(application), client:
        yield client


def auth(key: str = API_KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.mark.asyncio
class TestGatewayEndpoint:
    """End-to-end behaviour of POST /v1/completions."""

    async def test_successful_completion(self, gateway: httpx.AsyncClient) -> None:
        response = await gateway.post("/v1/completions", json=body(), headers=auth())
        assert response.status_code == 200
        payload = response.json()
        assert payload["provider"] == "primary"
        assert payload["failed_over"] is False
        assert payload["choices"][0]["message"]["role"] == "assistant"
        assert payload["usage"]["total_tokens"] > 0

    async def test_rate_limit_headers_are_present(self, gateway: httpx.AsyncClient) -> None:
        response = await gateway.post("/v1/completions", json=body(), headers=auth())
        assert response.headers["x-ratelimit-limit"] == "10000"
        assert int(response.headers["x-ratelimit-remaining"]) < 10_000
        assert response.headers["x-gateway-failover"] == "false"

    async def test_usage_is_reconciled_to_the_providers_report(self, gateway: httpx.AsyncClient) -> None:
        """The window records what was charged, not the pre-flight estimate."""
        response = await gateway.post("/v1/completions", json=body(max_tokens=5_000), headers=auth())
        charged = response.json()["usage"]["total_tokens"]

        usage = (await gateway.get("/v1/usage", headers=auth())).json()
        assert usage["used_tokens"] == charged
        assert charged < 5_000, "the 5,000-token reservation was not corrected downwards"

    async def test_requests_are_refused_once_the_budget_is_gone(self, ledger: TokenLedger) -> None:
        """Accumulated *actual* usage eventually trips the limiter.

        Uses an expensive provider on purpose: with the cheap default, reconciliation
        corrects each 3,000-token reservation down to about 70 and the budget never runs out.
        """
        application, client = build_gateway(
            ledger, ModelRouter(MockProvider("primary", "ok", completion_tokens=3_000), None)
        )
        async with application.router.lifespan_context(application), client:
            statuses, response = [], None
            for _ in range(6):
                response = await client.post("/v1/completions", json=body(max_tokens=3_000), headers=auth())
                statuses.append(response.status_code)
                if response.status_code == 429:
                    break

        assert statuses[:3] == [200, 200, 200], statuses
        assert statuses[-1] == 429, f"the limiter never refused: {statuses}"
        assert response is not None
        payload = response.json()
        assert payload["error"]["code"] == "rate_limit_exceeded"
        assert payload["error"]["type"] == "rate_limit_error"
        assert payload["error"]["retry_after_ms"] > 0
        assert int(response.headers["retry-after"]) >= 1
        # Budget is nearly gone but not zero - which is exactly why the 4th request
        # was refused: what is left is smaller than the ~3,004 it asked to reserve.
        assert 0 < int(response.headers["x-ratelimit-remaining"]) < 3_000

    async def test_an_oversized_request_is_refused_outright(self, gateway: httpx.AsyncClient) -> None:
        """Asking for more than the whole window can never succeed."""
        response = await gateway.post("/v1/completions", json=body(max_tokens=20_000), headers=auth())
        assert response.status_code == 429
        assert response.json()["error"]["retry_after_ms"] == 60_000

    @pytest.mark.parametrize("headers", [{}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer "}])
    async def test_missing_credentials_are_refused(self, gateway: httpx.AsyncClient, headers: dict[str, str]) -> None:
        response = await gateway.post("/v1/completions", json=body(), headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "authentication_error"

    async def test_invalid_json_is_refused(self, gateway: httpx.AsyncClient) -> None:
        response = await gateway.post(
            "/v1/completions", content=b"{nope", headers={**auth(), "Content-Type": "application/json"}
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_json"

    @pytest.mark.parametrize(
        "payload",
        [
            {"messages": [{"role": "user", "content": "hi"}]},
            {"model": "m"},
            {"model": "", "messages": [{"role": "user", "content": "hi"}]},
            {"model": "m", "messages": []},
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 0},
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 99_999},
        ],
    )
    async def test_schema_violations_are_refused(self, gateway: httpx.AsyncClient, payload: dict[str, Any]) -> None:
        response = await gateway.post("/v1/completions", json=payload, headers=auth())
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"

    async def test_tenants_have_separate_budgets(self, ledger: TokenLedger) -> None:
        """One tenant exhausting its window must not affect another."""
        application, client = build_gateway(
            ledger, ModelRouter(MockProvider("primary", "ok", completion_tokens=3_000), None)
        )
        async with application.router.lifespan_context(application), client:
            for _ in range(6):
                spent = await client.post("/v1/completions", json=body(max_tokens=3_000), headers=auth())
                if spent.status_code == 429:
                    break
            assert spent.status_code == 429, "alpha's budget was never exhausted"
            beta = await client.post("/v1/completions", json=body(max_tokens=3_000), headers=auth(OTHER_KEY))
        assert beta.status_code == 200

    async def test_healthz_reports_configuration(self, gateway: httpx.AsyncClient) -> None:
        health = (await gateway.get("/healthz")).json()
        assert health == {
            "status": "ok",
            "primary": "primary",
            "fallback": "fallback",
            "timeout_ms": 3_000,
            "limit_tokens": 10_000,
            "window_seconds": 60,
            "database": health["database"],
        }


@pytest.mark.asyncio
class TestGatewayFailover:
    """Failover as seen from the HTTP surface."""

    async def test_failover_is_reported_to_the_client(self, ledger: TokenLedger) -> None:
        router = ModelRouter(MockProvider("primary", "rate_limited"), MockProvider("fallback", "ok"))
        application, client = build_gateway(ledger, router)
        async with application.router.lifespan_context(application), client:
            response = await client.post("/v1/completions", json=body(), headers=auth())

        assert response.status_code == 200
        assert response.json()["provider"] == "fallback"
        assert response.json()["failed_over"] is True
        assert response.headers["x-gateway-failover"] == "true"

    async def test_total_outage_returns_503_not_a_crash(self, ledger: TokenLedger) -> None:
        router = ModelRouter(MockProvider("primary", "unavailable"), MockProvider("fallback", "unavailable"))
        application, client = build_gateway(ledger, router)
        async with application.router.lifespan_context(application), client:
            response = await client.post("/v1/completions", json=body(), headers=auth())

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "all_providers_failed"

    async def test_a_failed_request_is_refunded_to_the_prompt_estimate(self, ledger: TokenLedger) -> None:
        """A client is not billed for a completion it never received."""
        router = ModelRouter(MockProvider("primary", "unavailable"), MockProvider("fallback", "unavailable"))
        application, client = build_gateway(ledger, router)
        async with application.router.lifespan_context(application), client:
            await client.post("/v1/completions", json=body(max_tokens=5_000), headers=auth())
            usage = (await client.get("/v1/usage", headers=auth())).json()

        assert 0 < usage["used_tokens"] < 100, "the 5,000-token reservation was not refunded"

    async def test_rate_limit_headers_on_a_failed_request_reflect_the_refund(self, ledger: TokenLedger) -> None:
        """X-RateLimit-Remaining must match what /v1/usage reports right after.

        The refund happens before the response is built, so its header must not
        be computed from the pre-refund reservation.
        """
        router = ModelRouter(MockProvider("primary", "unavailable"), MockProvider("fallback", "unavailable"))
        application, client = build_gateway(ledger, router)
        async with application.router.lifespan_context(application), client:
            response = await client.post("/v1/completions", json=body(max_tokens=5_000), headers=auth())
            usage = (await client.get("/v1/usage", headers=auth())).json()

        assert response.status_code == 503
        header_remaining = int(response.headers["x-ratelimit-remaining"])
        assert header_remaining == usage["remaining_tokens"], (
            f"header said {header_remaining} remaining but the ledger already showed "
            f"{usage['remaining_tokens']} by the time the response was sent"
        )


@pytest.mark.asyncio
class TestErrorSanitisation:
    """No upstream detail, key or traceback may reach the client."""

    SECRET = "sk-live-9f2c-INTERNAL"

    async def test_upstream_messages_are_not_echoed(self, ledger: TokenLedger) -> None:
        primary = MockProvider("primary", "unavailable")
        primary.name = f"primary-{self.SECRET}"  # planted where an error message would pick it up
        application, client = build_gateway(ledger, ModelRouter(primary, None))
        async with application.router.lifespan_context(application), client:
            response = await client.post("/v1/completions", json=body(), headers=auth())

        assert response.status_code == 503
        assert self.SECRET not in response.text

    async def test_internal_exceptions_become_a_clean_500(self, ledger: TokenLedger) -> None:
        class ExplodingRouter(ModelRouter):
            async def complete(self, payload: dict[str, Any], request_id: str) -> Any:
                raise RuntimeError(f"database at 10.0.0.5 said no: {TestErrorSanitisation.SECRET}")

        application, client = build_gateway(ledger, ExplodingRouter(MockProvider("p"), MockProvider("s")))
        async with application.router.lifespan_context(application), client:
            response = await client.post("/v1/completions", json=body(), headers=auth())

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"
        for fragment in (self.SECRET, "10.0.0.5", "RuntimeError", "Traceback"):
            assert fragment not in response.text

    async def test_an_internal_failure_refunds_the_reservation(self, ledger: TokenLedger) -> None:
        class ExplodingRouter(ModelRouter):
            async def complete(self, payload: dict[str, Any], request_id: str) -> Any:
                raise RuntimeError("boom")

        application, client = build_gateway(ledger, ExplodingRouter(MockProvider("p"), MockProvider("s")))
        async with application.router.lifespan_context(application), client:
            await client.post("/v1/completions", json=body(max_tokens=5_000), headers=auth())
            usage = (await client.get("/v1/usage", headers=auth())).json()
        assert usage["used_tokens"] == 0

    async def test_every_error_shares_one_payload_shape(self, gateway: httpx.AsyncClient) -> None:
        """Clients should be able to parse any failure the same way."""
        responses = [
            await gateway.post("/v1/completions", json=body(), headers={}),
            await gateway.post("/v1/completions", json={"model": "m"}, headers=auth()),
            await gateway.post("/v1/completions", content=b"{", headers={**auth(), "Content-Type": "application/json"}),
        ]
        for response in responses:
            error = response.json()["error"]
            assert {"code", "message", "type", "request_id"} <= set(error)
            assert response.headers["x-request-id"] == error["request_id"]


class TestErrorContract:
    """The shape of the error vocabulary itself."""

    def test_gateway_error_payload_is_json_serialisable(self) -> None:
        error = GatewayError("x", "y", 503, "upstream_error", retry_after_ms=1_500)
        assert json.loads(json.dumps(error.to_payload("req-9")))["error"]["retry_after_ms"] == 1_500

    def test_retry_after_is_omitted_when_there_is_none(self) -> None:
        assert "retry_after_ms" not in GatewayError("x", "y", 500).to_payload("req-9")["error"]

    def test_provider_exceptions_are_distinct_types(self) -> None:
        """The router branches on type, so these must not collapse into one."""
        assert not issubclass(UpstreamRateLimited, UpstreamUnavailable)
        assert not issubclass(UpstreamUnavailable, UpstreamRateLimited)
