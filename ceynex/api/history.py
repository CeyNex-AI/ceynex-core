"""Query history — SRS 3.5.2.

Additive to the frozen `ceynex-contracts` schema (`dim_country`, `dim_hs`,
`fact_trade`, `dq_flag`, `ingest_run`) rather than touching it — `query_history`
is a `ceynex-core`-only concern that no other member's code reads or writes, so
it has no business living behind that repo's 3-way-approval PR gate.
`ensure_table()` runs once at API startup (see `main.py`'s lifespan), same
idempotent-DDL spirit as `ceynex.data.bootstrap.apply_schema`.

Recording is opportunistic: a query answered while signed out, or while
Postgres happens to be unreachable, still returns its answer normally — losing
a history row is not worth failing, or even degrading, the query itself.
Listing is not: a caller asking for their history and silently getting an
empty list back on a database outage would be misleading, so `list_for_user`
raises and the route layer turns that into a 503.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg

from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_history (
    id BIGSERIAL PRIMARY KEY,
    user_email TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    degraded BOOLEAN NOT NULL,
    asked_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS query_history_user_asked_idx
    ON query_history (user_email, asked_at DESC);
"""


def ensure_table() -> None:
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()
    except psycopg.Error as exc:
        log.warning("query_history table not ensured (postgres unreachable?): %s", exc)


def record(*, user_email: str, query: str, answer: str, confidence: float, degraded: bool) -> None:
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_history (user_email, query, answer, confidence, degraded)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_email, query, answer, confidence, degraded),
            )
            conn.commit()
    except psycopg.Error as exc:
        log.warning("failed to record query history for %s: %s", user_email, exc)


@dataclass(frozen=True)
class HistoryEntry:
    id: int
    query: str
    answer: str
    confidence: float
    degraded: bool
    asked_at: str


def list_for_user(user_email: str, limit: int = 20) -> list[HistoryEntry]:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, query, answer, confidence, degraded, asked_at
            FROM query_history
            WHERE user_email = %s
            ORDER BY asked_at DESC
            LIMIT %s
            """,
            (user_email, limit),
        )
        rows = cur.fetchall()
    return [
        HistoryEntry(
            id=row[0],
            query=row[1],
            answer=row[2],
            confidence=row[3],
            degraded=row[4],
            asked_at=row[5].isoformat(),
        )
        for row in rows
    ]
