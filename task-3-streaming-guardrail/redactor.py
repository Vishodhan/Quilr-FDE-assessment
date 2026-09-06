"""Streaming PII redaction over text that arrives in arbitrary pieces.

The hard part of redacting a token stream is that a match can straddle a chunk
boundary: ``4111 1111 1111`` in one chunk and ``1111`` in the next is a card
number, but neither half matches on its own.

The approach here is a **bounded tail buffer**. On every ``feed()`` the redactor
finds the longest suffix that could still grow into a match, emits everything
before it, and keeps only that suffix. In ordinary prose the suffix is the last
word or two, so latency is negligible; it can never exceed ``max_holdback``
characters, so memory is bounded no matter how long the response runs.

    >>> r = StreamingRedactor()
    >>> r.feed("Card 4111 1111 ")
    'Card '
    >>> r.feed("1111 1111 expires soon")
    '[REDACTED] expires '
    >>> r.flush()
    'soon'
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Final

REPLACEMENT: Final = "[REDACTED]"

# Longest suffix ever held back. Comfortably above the longest realistic match
# (an email at RFC's practical limit), which keeps the buffer from ever growing
# with the length of the response.
DEFAULT_MAX_HOLDBACK: Final = 256

# One alternation, tried in order, so the whole buffer is scanned in a single pass.
# Card first: a 16-digit run must not be picked up as an SSN plus stray digits.
_PATTERN_SOURCES: Final[tuple[tuple[str, str], ...]] = (
    # 13-19 digits, optionally grouped by single spaces or hyphens.
    ("credit_card", r"(?P<credit_card>\b(?:\d[ -]?){12,18}\d\b)"),
    # US SSN: rejects the never-issued 000/666/9xx areas, 00 group and 0000 serial.
    ("ssn", r"(?P<ssn>\b(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b)"),
    # Email: lookbehind rather than \b, because a local part may start with '.' or '_'.
    (
        "email",
        r"(?P<email>(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b)",
    ),
)

_MATCH_RE: Final = re.compile("|".join(source for _, source in _PATTERN_SOURCES))

# Cheap pre-filter: no '@' and no digit means no pattern can possibly match.
_TRIGGER_RE: Final = re.compile(r"[@\d]")

# Suffixes that could still grow into a match. Two of them, because the character
# sets differ: emails never contain a space, grouped card digits do.
_EMAIL_TAIL_RE: Final = re.compile(r"[A-Za-z0-9._%+@-]+$")
_DIGIT_TAIL_RE: Final = re.compile(r"(?:\d[ -]?)+$")


class StreamingRedactor:
    """Redacts emails, SSNs and card numbers from text arriving in pieces.

    Not safe to share across concurrent streams - give each response its own
    instance, which is what the gateway does.
    """

    __slots__ = ("_buffer", "_counts", "_max_holdback", "_replacement")

    def __init__(self, replacement: str = REPLACEMENT, max_holdback: int = DEFAULT_MAX_HOLDBACK) -> None:
        if max_holdback < 1:
            raise ValueError("max_holdback must be at least 1 character")
        self._buffer = ""
        self._counts: Counter[str] = Counter()
        self._max_holdback = max_holdback
        self._replacement = replacement

    @property
    def pending(self) -> int:
        """Characters currently held back. Always <= max_holdback."""
        return len(self._buffer)

    @property
    def counts(self) -> Mapping[str, int]:
        """How many matches of each kind have been redacted so far."""
        return dict(self._counts)

    @property
    def total_redactions(self) -> int:
        """Total matches redacted so far."""
        return sum(self._counts.values())

    def feed(self, text: str) -> str:
        """Take the next piece of the stream, return what is safe to emit now.

        The return value may be empty - that means every character so far could
        still turn out to be part of a match.
        """
        if not text:
            return ""

        self._buffer += text
        boundary = self._safe_boundary(self._buffer)
        if boundary == 0:
            return ""

        ready = self._buffer[:boundary]
        self._buffer = self._buffer[boundary:]
        return self._redact(ready)

    def flush(self) -> str:
        """Redact and release whatever is still held back. Call once, at end of stream."""
        if not self._buffer:
            return ""
        ready, self._buffer = self._buffer, ""
        return self._redact(ready)

    def scrub(self, text: str) -> str:
        """Redact a complete string in one go, for non-streaming responses."""
        return self._redact(text)

    def _safe_boundary(self, buffer: str) -> int:
        """Index up to which the buffer can be released.

        That is the start of the longest suffix which might still be the
        beginning of a match, floored so the held-back tail never exceeds
        ``max_holdback``.
        """
        boundary = len(buffer)
        for pattern in (_EMAIL_TAIL_RE, _DIGIT_TAIL_RE):
            candidate = pattern.search(buffer)
            if candidate is not None:
                boundary = min(boundary, candidate.start())
        return max(boundary, len(buffer) - self._max_holdback)

    def _redact(self, text: str) -> str:
        """Replace every complete match in ``text``, counting them by kind."""
        if not text or _TRIGGER_RE.search(text) is None:
            return text

        def replace(match: re.Match[str]) -> str:
            self._counts[match.lastgroup or "unknown"] += 1
            return self._replacement

        return _MATCH_RE.sub(replace, text)
