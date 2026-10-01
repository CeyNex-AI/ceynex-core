"""Implements SRS 3.4.6 — per-user rate limiting on the query submission endpoint.

The requirement exists to protect SRS 3.3.1 (availability) and 3.4.1 (response
time) *for everyone else*: one account submitting queries in a loop should
degrade its own service, not the system's. A CeyNex query is unusually expensive
to serve — a graph fan-out, several Neo4j round trips and up to two LLM calls —
so the abusive case is cheap to send and costly to answer, which is exactly the
asymmetry a limiter is for.

Why Redis rather than a dict
----------------------------
`ceynex-infra/backend/Dockerfile` runs `uvicorn --workers 2`. A counter in
process memory would be per-worker, so the effective limit would be double the
configured one, and which limit a caller hit would depend on which worker
accepted the connection. Redis is already provisioned on the database VM and
`REDIS_URL` is already handed to the api container — it was, until now, the one
piece of the deployed stack nothing in the code used.

`InProcessWindow` remains for local development and tests, where there is one
process and no Redis. It is chosen automatically when `REDIS_URL` is unset, and
it says so at startup rather than silently pretending to be the real thing.

Fixed window, not sliding
-------------------------
A fixed window lets a caller send `limit` requests at the end of one window and
`limit` again at the start of the next — a 2x burst across the boundary. That is
accepted deliberately: the sliding-window alternative costs a sorted set and a
read-modify-write per request, and the requirement is about sustained abuse
rather than instantaneous burst shaping. The burst is bounded, and
`query_runner.REQUEST_TIMEOUT_S` bounds what any one request can consume — as of
the conversational layer it is genuinely applied, having previously been declared
and never passed to a timeout, so this sentence was aspirational until then.

Failing over, not open
----------------------
If Redis is unreachable the request is counted by this worker alone, with a
warning logged, and never refused for the outage itself: an unavailable store
taking down query submission would cause the outage the limiter exists to
prevent (SRS 3.4.3's degrade-don't-fail). Until 2026-10 it was *allowed*
outright, which left abuse unthrottled for the length of a Redis outage. Now the
limit still holds per worker, so at most `workers x limit` per window: the same
"fail open, but never weaker than before" line D16 draws for the spend cap
(docs/ARCHITECTURE_DELTA.md). The Redis client gets short timeouts, so a hung
Redis costs a request half a second rather than its whole budget.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from ceynex import settings

log = logging.getLogger(__name__)

KEY_PREFIX = "ceynex:rl:query"


@dataclass(frozen=True)
class Decision:
    """The outcome of one rate-limit check."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_s: int


class Window(Protocol):
    async def check(self, identity: str, limit: int, window_s: int) -> Decision: ...


def _window_index(now: float, window_s: int) -> int:
    return int(now // window_s)


def _retry_after(now: float, window_s: int) -> int:
    """Whole seconds until the current window rolls over, never 0."""
    return max(1, int(window_s - (now % window_s)) or window_s)


class InProcessWindow:
    """Fixed-window counter in this process's memory. One process only.

    Correct for `make up`, a single-worker run and the test suite. Wrong for the
    deployed two-worker image, which is why `build_window` prefers Redis and
    logs when it cannot.
    """

    def __init__(self) -> None:
        self._hits: dict[tuple[str, int], int] = {}

    async def check(self, identity: str, limit: int, window_s: int) -> Decision:
        now = time.time()
        index = _window_index(now, window_s)
        key = (identity, index)

        # Drop counters from windows that have rolled over. Without this the
        # dict grows once per distinct caller per window, forever — a slow leak
        # that only shows up in a long-running process, which is the only kind
        # this runs in.
        if len(self._hits) > 1024:
            self._hits = {k: v for k, v in self._hits.items() if k[1] >= index}

        count = self._hits.get(key, 0) + 1
        self._hits[key] = count
        return _decide(count, limit, window_s, now)


class RedisWindow:
    """Fixed-window counter shared across workers.

    `INCR` on a key that embeds the window index, with `EXPIRE` set only on the
    first hit of that window. The window index in the key is what makes this
    safe without a transaction: a counter is never reused across windows, so an
    `EXPIRE` that fails leaves a key that is stale rather than one that
    throttles the wrong window. Redis reclaims it on the next `EXPIRE` anyway.
    """

    def __init__(self, client: Any, fallback: Window | None = None) -> None:
        self._client = client
        # Counts while Redis is unreachable; see "Failing over" above.
        self._fallback = fallback or InProcessWindow()

    async def check(self, identity: str, limit: int, window_s: int) -> Decision:
        now = time.time()
        key = f"{KEY_PREFIX}:{identity}:{_window_index(now, window_s)}"
        try:
            count = int(await self._client.incr(key))
            if count == 1:
                # Two windows' grace so a clock skew between workers cannot
                # expire a counter that is still in use.
                await self._client.expire(key, window_s * 2)
        except Exception:  # noqa: BLE001 - see "Failing over" in the module docstring
            log.warning("rate-limit store unavailable; counting in this worker", exc_info=True)
            return await self._fallback.check(identity, limit, window_s)
        return _decide(count, limit, window_s, now)


def _decide(count: int, limit: int, window_s: int, now: float) -> Decision:
    if count > limit:
        return Decision(
            allowed=False, limit=limit, remaining=0, retry_after_s=_retry_after(now, window_s)
        )
    return Decision(allowed=True, limit=limit, remaining=limit - count, retry_after_s=0)


def build_window() -> Window:
    """Redis when `REDIS_URL` is set and the client imports, else in-process.

    Never raises: a misconfigured rate-limit store must not stop the API from
    starting, for the same reason `RedisWindow.check` fails open.
    """
    url = settings.redis_url()
    if not url:
        log.info("REDIS_URL unset — rate limiting is per-process (fine for one worker, not two)")
        return InProcessWindow()
    try:
        from redis.asyncio import from_url  # noqa: PLC0415 - optional at import time

        log.info("rate limiting backed by Redis")
        return RedisWindow(
            from_url(
                url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
            )
        )
    except Exception:  # noqa: BLE001
        log.warning(
            "could not build the Redis rate-limit store; falling back to per-process", exc_info=True
        )
        return InProcessWindow()


def client_ip(request: Any) -> str | None:
    """The real client address behind the nginx proxy.

    `request.client.host` is nginx's own IP for *every* proxied request —
    Starlette doesn't parse forwarded headers without `ProxyHeadersMiddleware`
    — so using it directly collapses every caller into one rate-limit bucket.
    `ceynex-infra/frontend/nginx.conf.template` sets both headers below.

    `X-Real-IP` is preferred: nginx sets it to the actual socket peer
    (`$remote_addr`) and a client can't spoof past it. `X-Forwarded-For` is
    `$proxy_add_x_forwarded_for`, i.e. it *appends* to any client-supplied
    value, so only its **last** hop (added by our nginx) is trustworthy. The
    backend is VPC-internal (port 8000 is not internet-reachable), so there is
    exactly one proxy in front and this is safe; it would not be if the app
    were directly exposed.
    """
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip() or None
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[-1].strip() or None
    return request.client.host if request.client else None


def identity_of(user_email: str | None, client_host: str | None) -> str:
    """Who this request counts against.

    Per *user* when there is one, per client address when there is not. The
    query, chat and news routes require sign-in, so for them it is always the
    user; `/api/graph/expand` still answers a shared link's anonymous reader,
    which is what the address is for. The two namespaces are kept distinct so
    one shared NAT address cannot exhaust a signed-in user's own allowance.
    """
    if user_email:
        return f"user:{user_email}"
    return f"ip:{client_host or 'unknown'}"


__all__ = [
    "Decision",
    "InProcessWindow",
    "KEY_PREFIX",
    "RedisWindow",
    "Window",
    "build_window",
    "client_ip",
    "identity_of",
]
