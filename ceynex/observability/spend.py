"""Daily model-spend limits that hold across workers — SRS 3.4.3, 3.4.6, D16.

R5 (LLM quota and cost exhaustion) has had a guard since the client shipped:
`daily_spend_cap_usd`, checked before every paid call. D15 named its two
weaknesses, and this module is the follow-up D15 promised.

1. **It was per worker.** The cap read a counter in process memory, and
   `ceynex-infra/backend/Dockerfile` runs `uvicorn --workers 2`, so true spend
   could reach twice the configured figure. `GET /api/usage/limits` said so in
   `cap_is_per_worker` rather than hide it, but saying so is not fixing it.
2. **It was per process lifetime, not per day.** The counter reset on restart,
   never at midnight — a long-running worker capped itself for good, and a
   restarted one forgot everything it had spent.

Now the cap is counted in Redis, keyed by UTC day, incremented after every paid
call — the same infrastructure and the same shape as `api/rate_limit.py`'s
`RedisWindow`. Both workers read and write one number, and it rolls over at
00:00 UTC. And because a single account is the likeliest way to run up a bill,
there is now a **per-user** daily budget beside the global one.

**What happens at a limit is the degraded path, not a refusal.** A reader over
their budget still gets figures, evidence and confidence — the deterministic
answer SRS 3.4.3 already requires — and the free OpenRouter failsafe is still
tried first, because it costs nothing against either limit. The limits are
disclosed on the Usage page (SRS 3.4.6), including when they reset.

**Failing open, but never weaker than before.** A Redis error is logged and the
check falls back to this worker's own count for the day, which every call also
updates. So an outage degrades the guard to exactly the per-worker cap it used
to be, never to no cap at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from ceynex import settings

log = logging.getLogger(__name__)

KEY_PREFIX = "ceynex:spend"

#: Long enough to outlive the day it counts by a comfortable margin; the key
#: names the day, so nothing depends on the expiry being exact.
KEY_TTL_S = 2 * 24 * 3600


def today(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y-%m-%d")


def resets_at(now: datetime | None = None) -> datetime:
    """The next 00:00 UTC — when both daily limits start again."""
    current = now or datetime.now(UTC)
    return (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass(frozen=True)
class Spent:
    """What has been spent today, deployment-wide and by one reader."""

    total_usd: float
    user_usd: float


class SpendCounter(Protocol):
    shared: bool

    async def spent(self, user_email: str | None) -> Spent: ...

    async def add(self, amount_usd: float, user_email: str | None) -> None: ...


@dataclass
class InProcessSpendCounter:
    """This process's own spend, by day. The fallback, and Redis's shadow.

    `shared = False` is what `/api/usage/limits` reads to tell the reader the
    cap is per worker — true here, false once Redis carries the count.
    """

    shared: bool = False
    _by_key: dict[tuple[str, str], float] = field(default_factory=dict)

    async def spent(self, user_email: str | None) -> Spent:
        day = today()
        return Spent(
            total_usd=self._by_key.get((day, ""), 0.0),
            user_usd=self._by_key.get((day, user_email), 0.0) if user_email else 0.0,
        )

    async def add(self, amount_usd: float, user_email: str | None) -> None:
        if amount_usd <= 0:
            return
        day = today()
        # Earlier days are dropped as each new one starts: this is a day's count,
        # not a ledger (`observability/ledger.py` is the ledger).
        for key in [k for k in self._by_key if k[0] != day]:
            del self._by_key[key]
        self._by_key[(day, "")] = self._by_key.get((day, ""), 0.0) + amount_usd
        if user_email:
            self._by_key[(day, user_email)] = self._by_key.get((day, user_email), 0.0) + amount_usd


class RedisSpendCounter:
    """One count per UTC day, deployment-wide and per reader, shared by workers."""

    shared = True

    def __init__(self, client: Any, shadow: InProcessSpendCounter | None = None) -> None:
        self._redis = client
        self._shadow = shadow or InProcessSpendCounter()

    @staticmethod
    def _keys(user_email: str | None) -> tuple[str, str | None]:
        day = today()
        total = f"{KEY_PREFIX}:{day}:total"
        user = f"{KEY_PREFIX}:{day}:user:{user_email}" if user_email else None
        return total, user

    async def spent(self, user_email: str | None) -> Spent:
        total_key, user_key = self._keys(user_email)
        shadow = await self._shadow.spent(user_email)
        try:
            values = await self._redis.mget([total_key, user_key] if user_key else [total_key])
        except Exception as exc:  # noqa: BLE001 - fail open, to the per-worker count
            log.warning("spend counter unavailable; using this worker's own count: %s", exc)
            return shadow
        total = _as_float(values[0])
        user = _as_float(values[1]) if user_key else 0.0
        # Never weaker than this worker's own count: a key that expired early or
        # a replica that lost writes must not make spent money reappear.
        return Spent(total_usd=max(total, shadow.total_usd), user_usd=max(user, shadow.user_usd))

    async def add(self, amount_usd: float, user_email: str | None) -> None:
        if amount_usd <= 0:
            return
        await self._shadow.add(amount_usd, user_email)
        total_key, user_key = self._keys(user_email)
        try:
            await self._redis.incrbyfloat(total_key, amount_usd)
            await self._redis.expire(total_key, KEY_TTL_S)
            if user_key:
                await self._redis.incrbyfloat(user_key, amount_usd)
                await self._redis.expire(user_key, KEY_TTL_S)
        except Exception as exc:  # noqa: BLE001 - fail open; the shadow still counted it
            log.warning("spend counter unavailable; this call is counted locally only: %s", exc)


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bytes):
        value = value.decode()
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_spend_counter() -> SpendCounter:
    """Redis when `REDIS_URL` is set, this process's own count otherwise."""
    url = settings.redis_url()
    if not url:
        log.info("spend limits: no REDIS_URL, so the daily cap is counted per worker")
        return InProcessSpendCounter()
    try:
        from redis.asyncio import from_url  # noqa: PLC0415 - optional at import time
    except ImportError:
        log.warning("spend limits: redis client not installed; counting per worker")
        return InProcessSpendCounter()
    return RedisSpendCounter(from_url(url))


_shared: SpendCounter | None = None


def shared_counter() -> SpendCounter:
    """The process's counter, built on first use — `REDIS_URL` is read then."""
    global _shared  # noqa: PLW0603 - one process-lifetime object
    if _shared is None:
        _shared = build_spend_counter()
    return _shared


def set_shared_counter(counter: SpendCounter | None) -> None:
    """Test seam. Production never calls this."""
    global _shared  # noqa: PLW0603
    _shared = counter


__all__ = [
    "InProcessSpendCounter",
    "RedisSpendCounter",
    "SpendCounter",
    "Spent",
    "build_spend_counter",
    "resets_at",
    "set_shared_counter",
    "shared_counter",
    "today",
]
