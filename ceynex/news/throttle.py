"""Supports docs/ARCHITECTURE_DELTA.md D11 — the outbound call gate for GDELT.

`api/rate_limit.py` throttles callers coming *in*. This throttles us going
*out*, and it is a deliberate mirror of that module — the same Protocol shape,
the same Redis-or-in-process choice, the same never-raise `build_*` — so there
is one pattern in this codebase rather than two.

Why we need one at all
----------------------
GDELT DOC 2.0 is free and publishes no rate limit. That is not permission to
hammer it: one request every five seconds is the community convention, and this
project is a guest on someone else's free service. The gate is set conservatively
rather than tuned to what we can get away with.

There is a self-interested reason too. `/api/news/search` fires on every
submitted query, so demo traffic maps one-to-one onto outbound calls. Without a
gate, five people asking questions in the same minute is a burst that looks
exactly like abuse from the far end.

Why a gate and not a token bucket
---------------------------------
A bucket would let a burst through and smooth afterwards, which is the wrong
shape when the thing being protected is a courtesy limit rather than a capacity
one. `SET <key> <token> NX PX 5000` is three lines, is correct across workers,
and the key's own expiry means a crashed holder cannot wedge it — no lease
tracking, no cleanup path.

Refusing rather than raising
----------------------------
`acquire()` returns False when it cannot get a slot in time. The interactive
path then skips GDELT entirely and answers from the vector store, which is a
fast, useful, already-implemented degraded mode. Blocking a user's request for
fifteen seconds to be polite to an API would be the wrong trade.

Failing open
------------
Any Redis exception allows the call, with a warning. Same direction as
`RedisWindow.check`: the exposure is exceeding a courtesy rate for as long as
Redis is down, which is smaller than making an optional feature depend on a
store it does not otherwise need.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Protocol

from ceynex import settings

log = logging.getLogger(__name__)

#: The community convention for GDELT. Overridable from config/news.yaml.
MIN_INTERVAL_S = 5.0

GATE_KEY = "ceynex:news:gdelt:gate"

#: How often a waiter re-asks Redis for the gate. Short enough that the wait is
#: not visibly longer than the interval, long enough not to spin.
_POLL_INTERVAL_S = 0.25


class Throttle(Protocol):
    async def acquire(self, *, max_wait_s: float) -> bool: ...


class NoThrottle:
    """Always allows immediately. For tests, and for `--dry-run`."""

    async def acquire(self, *, max_wait_s: float) -> bool:  # noqa: ARG002 - protocol shape
        return True


class InProcessThrottle:
    """One slot every `interval_s`, in this process.

    Correct for `make up`, a single-worker run and the test suite. Understated
    for the deployed two-worker image — two of these give two calls per interval
    between them — which is why `build_throttle` prefers Redis and says which it
    chose.

    The lock is held across the sleep on purpose: that is what makes waiters
    queue in order rather than all waking at the same instant and firing
    together, which would defeat the whole point.
    """

    def __init__(self, interval_s: float = MIN_INTERVAL_S) -> None:
        self._interval_s = max(0.0, interval_s)
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self, *, max_wait_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, max_wait_s)
        try:
            await asyncio.wait_for(
                self._lock.acquire(), timeout=max(0.0, deadline - time.monotonic())
            )
        except TimeoutError:
            return False

        try:
            wait = self._next_allowed - time.monotonic()
            if wait > 0:
                if time.monotonic() + wait > deadline:
                    return False
                await asyncio.sleep(wait)
            self._next_allowed = time.monotonic() + self._interval_s
            return True
        finally:
            self._lock.release()


class RedisThrottle:
    """One slot every `interval_s`, shared across workers.

    `SET GATE_KEY <token> NX PX interval` — whoever sets it owns the next
    interval, and the key expiring *is* the release. Nothing has to remember to
    unlock, which is what makes a crash during a GDELT call harmless.

    The token is written but never read back. It is there so `docker exec redis
    get` during an incident says which process is holding the gate.
    """

    def __init__(self, client: Any, interval_s: float = MIN_INTERVAL_S) -> None:
        self._client = client
        self._interval_s = max(0.0, interval_s)
        self._token = uuid.uuid4().hex

    async def acquire(self, *, max_wait_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, max_wait_s)
        px = max(1, int(self._interval_s * 1000))
        while True:
            try:
                acquired = await self._client.set(GATE_KEY, self._token, nx=True, px=px)
            except Exception:  # noqa: BLE001 - see "Failing open" in the module docstring
                log.warning("news throttle store unavailable; allowing the call", exc_info=True)
                return True

            if acquired:
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(_POLL_INTERVAL_S, remaining))


def build_throttle(interval_s: float = MIN_INTERVAL_S) -> Throttle:
    """Redis when `REDIS_URL` is set and the client imports, else in-process.

    Never raises, for the same reason `build_window()` does not: a misconfigured
    throttle must not stop the API from starting.
    """
    url = settings.redis_url()
    if not url:
        log.info(
            "REDIS_URL unset — the GDELT gate is per-process "
            "(one call per %.1fs per worker, not per deployment)",
            interval_s,
        )
        return InProcessThrottle(interval_s)
    try:
        from redis.asyncio import from_url  # noqa: PLC0415 - optional at import time

        log.info("GDELT gate backed by Redis (one call per %.1fs across workers)", interval_s)
        return RedisThrottle(from_url(url, encoding="utf-8", decode_responses=True), interval_s)
    except Exception:  # noqa: BLE001 - a throttle we cannot build must not fail startup
        log.warning("could not build the Redis GDELT gate; falling back to per-process", exc_info=True)
        return InProcessThrottle(interval_s)


__all__ = [
    "GATE_KEY",
    "MIN_INTERVAL_S",
    "InProcessThrottle",
    "NoThrottle",
    "RedisThrottle",
    "Throttle",
    "build_throttle",
]
