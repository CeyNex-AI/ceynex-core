"""GET /health — what the container healthcheck and the deploy runbook read.

Deliberately reports each dependency separately rather than a single boolean. The
system is *designed* to run degraded when the LLM is unavailable (SRS 3.4.3), so
an LLM outage must not mark the container unhealthy and trigger a restart loop.
Neo4j and Postgres being down is a different matter, and shows as `degraded`.
"""

from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, Depends

from ceynex.api.deps import Runtime, get_runtime
from ceynex.api.schemas import HealthResponse
from ceynex.settings import postgres_dsn, redacted_dsn

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    runtime: Runtime = Depends(get_runtime),  # noqa: B008 - FastAPI's dependency idiom
) -> HealthResponse:
    neo4j_ok = await runtime.kg.verify_connectivity()
    postgres_ok, row_count = _check_postgres()
    llm_ok = runtime.llm.available

    # "ok" as long as the process can answer at all. A missing LLM key degrades
    # the answers; it does not make the service unhealthy.
    status = "ok" if (neo4j_ok and postgres_ok) else "degraded"

    return HealthResponse(
        status=status,
        neo4j=neo4j_ok,
        postgres=postgres_ok,
        llm=llm_ok,
        detail={
            "fact_trade_rows": row_count,
            "postgres": redacted_dsn(),
            "reasoning": "available" if llm_ok else "degraded: no API key, figures only",
        },
    )


def _check_postgres() -> tuple[bool, int | None]:
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM fact_trade")
            row = cur.fetchone()
        return True, int(row[0]) if row else 0
    except psycopg.Error as exc:
        log.warning("postgres health check failed: %s", exc)
        return False, None
