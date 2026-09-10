"""How current the data is — the first question a reader asks about a figure.

An answer that says "USD 1.37 billion in 2024" invites exactly one follow-up:
*is that the latest you have?* Until now the only way to find out was the admin
page, which most readers cannot open. This is one query against a table the
pipeline already maintains.

Deliberately `require_user` rather than public: it describes the state of the
deployment, and SRS 3.1.11 already requires an account before anything else here.
"""

from __future__ import annotations

import asyncio
import logging

import psycopg
from fastapi import APIRouter, Depends

from ceynex.api.routes.auth import TokenPayload, require_user
from ceynex.api.schemas import DataFreshnessResponse
from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

router = APIRouter(tags=["data"])


def _freshness_sync() -> dict:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute("SELECT max(period_start), count(*) FROM fact_trade")
        latest_period, rows = cur.fetchone()
        # The most recent run that actually finished. A failed or running row
        # says nothing about how current the data is, and showing one would be
        # the opposite of the reassurance this endpoint exists to give.
        cur.execute(
            """
            SELECT max(finished_at) FROM ingest_run
            WHERE status = 'ok' AND finished_at IS NOT NULL
            """
        )
        (last_ingest,) = cur.fetchone()
    return {
        "latest_observation": latest_period.isoformat() if latest_period else None,
        "observations": int(rows or 0),
        "last_ingest_at": last_ingest.isoformat() if last_ingest else None,
    }


@router.get("/api/data/freshness", response_model=DataFreshnessResponse)
async def data_freshness(
    user: TokenPayload = Depends(require_user),  # noqa: B008 - FastAPI's dependency idiom
) -> DataFreshnessResponse:
    try:
        payload = await asyncio.to_thread(_freshness_sync)
    except psycopg.Error as exc:
        # Always 200 with `available: false`. A ribbon that 500s would make a
        # working page look broken over a decoration.
        log.warning("could not read data freshness: %s", exc)
        return DataFreshnessResponse(available=False)
    return DataFreshnessResponse(available=True, **payload)


__all__ = ["router"]
