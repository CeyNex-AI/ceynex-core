"""The LLM spend ledger — SRS 3.4.6's usage restrictions, made visible.

SRS 3.4.6 and the usage-restriction clause require rate limits to be *"disclosed
to the user within the application rather than enforced silently."* Today nothing
is disclosed and nothing is even recorded: `LLMUsage` counts calls and dollars on
a process-lifetime dataclass, and `client.py::_cost()` reads `prompt_tokens` /
`completion_tokens` off every provider response only to multiply them into a
dollar figure and throw the counts away.

Additive to the frozen `ceynex-contracts` schema, same reasoning `api/history.py`
sets out for `query_history`: nobody else's code reads or writes this, so it has
no business behind that repo's three-way-approval gate.

**One row per call, not per request.** A request that times out halfway has still
spent real money, and recording partial spend is the entire point of a ledger. It
also makes every rollup below a plain `GROUP BY` rather than a nested-JSON blob.

**A cache hit is 0 tokens and $0, and that is correct.** No API call happened, so
no spend happened. It is recorded with `cache_hit = true` so a breakdown can show
how much the prompt cache is actually saving — not as a gap in the data.

**What this deliberately does not do.** `LLMReasoningClient._cap_reached()` keeps
reading its own process-global `usage.cost_usd`. With `uvicorn --workers 2` each
worker builds its own client, so true daily spend can reach **2x**
`daily_spend_cap_usd`. `headroom()` below is cross-worker accurate and is
reporting only — wiring it into enforcement means a Redis-backed shared counter
shaped like `api/rate_limit.py::RedisWindow`, which is a separate decision with
its own delta entry, not a side effect of adding a ledger.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import psycopg

from ceynex.observability.context import LLMCall
from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS llm_usage (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    user_email TEXT,
    conversation_id BIGINT,
    role TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    cache_hit BOOLEAN NOT NULL DEFAULT false,
    fallback BOOLEAN NOT NULL DEFAULT false,
    failed BOOLEAN NOT NULL DEFAULT false,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    elapsed_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    called_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS llm_usage_user_day_idx ON llm_usage (user_email, called_at DESC);
CREATE INDEX IF NOT EXISTS llm_usage_request_idx ON llm_usage (request_id);
CREATE INDEX IF NOT EXISTS llm_usage_role_model_idx ON llm_usage (role, model, called_at DESC);
CREATE INDEX IF NOT EXISTS llm_usage_called_at_idx ON llm_usage (called_at DESC);
"""


def ensure_table() -> None:
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()
    except psycopg.Error as exc:
        log.warning("llm_usage table not ensured (postgres unreachable?): %s", exc)


def _record_sync(
    *,
    request_id: str,
    user_email: str | None,
    conversation_id: int | None,
    calls: Sequence[LLMCall],
) -> None:
    if not calls:
        return
    rows = [
        (
            request_id,
            user_email,
            conversation_id,
            call.role,
            call.model,
            call.provider,
            call.cache_hit,
            call.fallback,
            call.failed,
            call.tokens_in,
            call.tokens_out,
            call.cost_usd,
            call.elapsed_ms,
        )
        for call in calls
    ]
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO llm_usage (
                    request_id, user_email, conversation_id, role, model, provider,
                    cache_hit, fallback, failed, tokens_in, tokens_out, cost_usd, elapsed_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
            conn.commit()
    except psycopg.Error as exc:
        # Opportunistic, exactly like history.record(): losing a usage row is
        # never worth failing, or even degrading, the answer it belongs to.
        log.warning("failed to record %d llm_usage rows for %s: %s", len(rows), request_id, exc)


async def record(
    *,
    request_id: str,
    user_email: str | None,
    conversation_id: int | None,
    calls: Sequence[LLMCall],
) -> None:
    """Write one turn's calls, off the event loop.

    `asyncio.to_thread` rather than `history.py`'s inline `psycopg.connect()`: a
    blocking call here would stall the loop while an SSE connection is expected
    to be emitting heartbeats, and with two uvicorn workers it stalls every other
    request on the same process too. A named departure from that template.
    """
    await asyncio.to_thread(
        _record_sync,
        request_id=request_id,
        user_email=user_email,
        conversation_id=conversation_id,
        calls=list(calls),
    )


@dataclass(frozen=True)
class UsageRollup:
    key: str
    calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float


def _rows_to_rollups(rows: Sequence[tuple]) -> list[UsageRollup]:
    return [
        UsageRollup(
            key=str(row[0]),
            calls=int(row[1]),
            tokens_in=int(row[2] or 0),
            tokens_out=int(row[3] or 0),
            cost_usd=float(row[4] or 0.0),
        )
        for row in rows
    ]


def _query(sql: str, params: Sequence[object]) -> list[tuple]:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


async def for_request(request_id: str) -> list[LLMCall]:
    """Every call made by one turn — the per-message footer, re-read from history."""
    rows = await asyncio.to_thread(
        _query,
        """
        SELECT role, model, provider, tokens_in, tokens_out, cost_usd,
               elapsed_ms, cache_hit, fallback, failed
        FROM llm_usage WHERE request_id = %s ORDER BY id
        """,
        (request_id,),
    )
    return [
        LLMCall(
            role=row[0],
            model=row[1],
            provider=row[2],
            tokens_in=int(row[3]),
            tokens_out=int(row[4]),
            cost_usd=float(row[5]),
            elapsed_ms=float(row[6]),
            cache_hit=bool(row[7]),
            fallback=bool(row[8]),
            failed=bool(row[9]),
        )
        for row in rows
    ]


async def by_day(user_email: str | None, *, days: int = 30) -> list[UsageRollup]:
    """Per-day rollup. `user_email=None` means every user — admin only."""
    if user_email is None:
        sql = """
            SELECT to_char(date_trunc('day', called_at), 'YYYY-MM-DD'), count(*),
                   sum(tokens_in), sum(tokens_out), sum(cost_usd)
            FROM llm_usage
            WHERE called_at >= now() - make_interval(days => %s)
            GROUP BY 1 ORDER BY 1 DESC
        """
        params: tuple[object, ...] = (days,)
    else:
        sql = """
            SELECT to_char(date_trunc('day', called_at), 'YYYY-MM-DD'), count(*),
                   sum(tokens_in), sum(tokens_out), sum(cost_usd)
            FROM llm_usage
            WHERE user_email = %s AND called_at >= now() - make_interval(days => %s)
            GROUP BY 1 ORDER BY 1 DESC
        """
        params = (user_email, days)
    return _rows_to_rollups(await asyncio.to_thread(_query, sql, params))


async def by_role_and_model(user_email: str | None, *, days: int = 30) -> list[UsageRollup]:
    """Where the money actually goes. Cache hits kept separate, not folded in."""
    if user_email is None:
        sql = """
            SELECT role || ' · ' || model || (CASE WHEN cache_hit THEN ' (cached)' ELSE '' END),
                   count(*), sum(tokens_in), sum(tokens_out), sum(cost_usd)
            FROM llm_usage
            WHERE called_at >= now() - make_interval(days => %s)
            GROUP BY 1 ORDER BY 5 DESC
        """
        params: tuple[object, ...] = (days,)
    else:
        sql = """
            SELECT role || ' · ' || model || (CASE WHEN cache_hit THEN ' (cached)' ELSE '' END),
                   count(*), sum(tokens_in), sum(tokens_out), sum(cost_usd)
            FROM llm_usage
            WHERE user_email = %s AND called_at >= now() - make_interval(days => %s)
            GROUP BY 1 ORDER BY 5 DESC
        """
        params = (user_email, days)
    return _rows_to_rollups(await asyncio.to_thread(_query, sql, params))


async def spent_today() -> float:
    """Today's spend across every worker — what the process-local cap cannot see."""
    rows = await asyncio.to_thread(
        _query,
        """
        SELECT coalesce(sum(cost_usd), 0) FROM llm_usage
        WHERE called_at >= date_trunc('day', now())
        """,
        (),
    )
    return float(rows[0][0]) if rows else 0.0


__all__ = [
    "CREATE_TABLE_SQL",
    "UsageRollup",
    "by_day",
    "by_role_and_model",
    "ensure_table",
    "for_request",
    "record",
    "spent_today",
]
