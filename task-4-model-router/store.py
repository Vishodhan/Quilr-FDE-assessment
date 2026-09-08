"""On-disk SQLite ledger backing a token-aware sliding-window rate limiter.

One row per request, holding the tokens it accounted for and when. A window's
usage is the sum of the rows inside it, so the window really does slide - there
are no fixed buckets to let a client spend twice the budget across a boundary.

Reserving is a single ``BEGIN IMMEDIATE`` transaction: check, then insert. That
makes the decision atomic against other tasks in this process (via the lock) and
against other processes sharing the file (via SQLite's write lock).

    ledger = TokenLedger("gateway.db", limit_tokens=50_000, window_seconds=60)
    await ledger.open()
    reservation = await ledger.reserve("tenant-key", tokens=1200)
    if reservation.allowed:
        ...
        await ledger.reconcile(reservation, actual_tokens=980)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import aiosqlite

logger = logging.getLogger("token-ledger")

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS token_usage (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key    TEXT    NOT NULL,
    ts_ms      INTEGER NOT NULL,
    tokens     INTEGER NOT NULL,
    request_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_token_usage_key_ts ON token_usage (api_key, ts_ms);
"""

# WAL keeps readers off the writer's back; busy_timeout absorbs cross-process contention.
_PRAGMAS: Final = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=5000",
)


def now_ms() -> int:
    """Current wall-clock time in milliseconds."""
    return int(time.time() * 1000)


@dataclass(frozen=True)
class Reservation:
    """The outcome of asking for budget."""

    allowed: bool
    tokens: int
    used_tokens: int
    limit_tokens: int
    row_id: int | None = None
    retry_after_ms: int = 0

    @property
    def remaining_tokens(self) -> int:
        """Budget left in the window after this decision."""
        return max(0, self.limit_tokens - self.used_tokens)


@dataclass(frozen=True)
class WindowUsage:
    """A snapshot of one key's current window."""

    api_key: str
    used_tokens: int
    limit_tokens: int
    requests: int
    window_seconds: int

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.limit_tokens - self.used_tokens)


class TokenLedger:
    """Token accounting over a sliding window, persisted in SQLite."""

    def __init__(self, database_path: str | Path, limit_tokens: int = 50_000, window_seconds: int = 60) -> None:
        if limit_tokens < 1:
            raise ValueError("limit_tokens must be at least 1")
        if window_seconds < 1:
            raise ValueError("window_seconds must be at least 1")

        self.database_path = Path(database_path)
        self.limit_tokens = limit_tokens
        self.window_seconds = window_seconds
        self._window_ms = window_seconds * 1000
        self._connection: aiosqlite.Connection | None = None
        # One connection is shared, so transactions must not interleave in-process.
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        """Connect, apply pragmas and create the schema. Safe to call once."""
        if self._connection is not None:
            return

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None turns off implicit transactions so BEGIN IMMEDIATE is ours to issue.
        connection = await aiosqlite.connect(str(self.database_path), isolation_level=None)
        try:
            for pragma in _PRAGMAS:
                await connection.execute(pragma)
            await connection.executescript(_SCHEMA)
        except Exception:
            await connection.close()
            raise

        self._connection = connection
        logger.info("token ledger open at %s (limit=%d/%ds)", self.database_path, self.limit_tokens, self.window_seconds)

    async def close(self) -> None:
        """Close the connection if it is open."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def __aenter__(self) -> TokenLedger:
        await self.open()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("TokenLedger.open() must be awaited before use")
        return self._connection

    async def reserve(self, api_key: str, tokens: int, at_ms: int | None = None, request_id: str | None = None) -> Reservation:
        """Claim ``tokens`` of budget for ``api_key``, or refuse with a retry hint.

        ``at_ms`` overrides the clock, which is what lets the tests exercise
        window expiry without sleeping.
        """
        if tokens < 0:
            raise ValueError("tokens must not be negative")

        stamp = at_ms if at_ms is not None else now_ms()
        cutoff = stamp - self._window_ms

        async with self._lock:
            await self._db.execute("BEGIN IMMEDIATE")
            try:
                # Eviction happens on the read path: expired rows are gone before
                # anything is summed, so the window can never over-count.
                await self._db.execute("DELETE FROM token_usage WHERE ts_ms <= ?", (cutoff,))

                async with self._db.execute(
                    "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE api_key = ? AND ts_ms > ?",
                    (api_key, cutoff),
                ) as cursor:
                    row = await cursor.fetchone()
                used = int(row[0]) if row else 0

                if used + tokens > self.limit_tokens:
                    retry_after = await self._retry_after_ms(api_key, cutoff, stamp, used + tokens - self.limit_tokens)
                    # COMMIT, not ROLLBACK: nothing was inserted, but the eviction
                    # above is real work and rolling it back would leak dead rows.
                    await self._db.execute("COMMIT")
                    # Show just enough of the key to correlate logs, never the whole thing.
                    masked_key = f"{api_key[:4]}...{api_key[-2:]}" if len(api_key) > 8 else "***"
                    logger.info("rate limit hit for %s: %d + %d > %d", masked_key, used, tokens, self.limit_tokens)
                    return Reservation(
                        allowed=False,
                        tokens=tokens,
                        used_tokens=used,
                        limit_tokens=self.limit_tokens,
                        retry_after_ms=retry_after,
                    )

                cursor = await self._db.execute(
                    "INSERT INTO token_usage (api_key, ts_ms, tokens, request_id) VALUES (?, ?, ?, ?)",
                    (api_key, stamp, tokens, request_id),
                )
                await self._db.execute("COMMIT")
                return Reservation(
                    allowed=True,
                    tokens=tokens,
                    used_tokens=used + tokens,
                    limit_tokens=self.limit_tokens,
                    row_id=cursor.lastrowid,
                )
            except Exception:
                await self._db.execute("ROLLBACK")
                raise

    async def reconcile(self, reservation: Reservation, actual_tokens: int) -> None:
        """Correct a reservation to what the request really cost.

        Reserving works off an estimate; the provider reports the truth. Without
        this the window drifts from reality on every request.
        """
        if reservation.row_id is None or actual_tokens < 0:
            return
        async with self._lock:
            await self._db.execute("UPDATE token_usage SET tokens = ? WHERE id = ?", (actual_tokens, reservation.row_id))

    async def release(self, reservation: Reservation) -> None:
        """Drop a reservation entirely, for a request that consumed nothing."""
        if reservation.row_id is None:
            return
        async with self._lock:
            await self._db.execute("DELETE FROM token_usage WHERE id = ?", (reservation.row_id,))

    async def usage(self, api_key: str, at_ms: int | None = None) -> WindowUsage:
        """Current window usage for one key."""
        stamp = at_ms if at_ms is not None else now_ms()
        cutoff = stamp - self._window_ms
        async with self._lock, self._db.execute(
            "SELECT COALESCE(SUM(tokens), 0), COUNT(*) FROM token_usage WHERE api_key = ? AND ts_ms > ?",
            (api_key, cutoff),
        ) as cursor:
            row = await cursor.fetchone()
        used, requests = (int(row[0]), int(row[1])) if row else (0, 0)
        return WindowUsage(
            api_key=api_key,
            used_tokens=used,
            limit_tokens=self.limit_tokens,
            requests=requests,
            window_seconds=self.window_seconds,
        )

    async def prune(self, at_ms: int | None = None) -> int:
        """Delete every expired row across all keys. Returns how many went."""
        stamp = at_ms if at_ms is not None else now_ms()
        async with self._lock:
            cursor = await self._db.execute("DELETE FROM token_usage WHERE ts_ms <= ?", (stamp - self._window_ms,))
        return cursor.rowcount or 0

    async def row_count(self) -> int:
        """Total rows on disk. Used by the tests to prove eviction really deletes."""
        async with self._lock, self._db.execute("SELECT COUNT(*) FROM token_usage") as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def _retry_after_ms(self, api_key: str, cutoff: int, stamp: int, tokens_to_free: int) -> int:
        """How long until enough of the oldest rows expire to fit the request.

        Walks the window oldest-first, accumulating until ``tokens_to_free`` would
        have aged out, and returns when that row leaves the window.
        """
        async with self._db.execute(
            "SELECT ts_ms, tokens FROM token_usage WHERE api_key = ? AND ts_ms > ? ORDER BY ts_ms ASC",
            (api_key, cutoff),
        ) as cursor:
            freed = 0
            async for row_ts, row_tokens in cursor:
                freed += int(row_tokens)
                if freed >= tokens_to_free:
                    return max(0, int(row_ts) + self._window_ms - stamp)

        # The request is larger than the whole budget: waiting will not help,
        # but report a full window rather than zero so clients back off.
        return self._window_ms
