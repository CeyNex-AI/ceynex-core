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

`saved` (SRS 3.5.2's second half — explicitly bookmarking a query, distinct
from it simply appearing in history) is a column on the same table rather than
a second one: a saved query *is* a history entry, just flagged, and every
query that could ever be saved already has a row here the moment it's asked.
Added via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` in `ensure_table()`
rather than only in `CREATE_TABLE_SQL`, since `CREATE TABLE IF NOT EXISTS` is a
no-op against a table that already exists from before this column did.
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
    asked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    saved BOOLEAN NOT NULL DEFAULT false
);
ALTER TABLE query_history ADD COLUMN IF NOT EXISTS saved BOOLEAN NOT NULL DEFAULT false;
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
    saved: bool


def list_for_user(user_email: str, limit: int = 20, *, saved: bool | None = None) -> list[HistoryEntry]:
    conditions = ["user_email = %s"]
    params: list[object] = [user_email]
    if saved is not None:
        conditions.append("saved = %s")
        params.append(saved)
    params.append(limit)

    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, query, answer, confidence, degraded, asked_at, saved
            FROM query_history
            WHERE {" AND ".join(conditions)}
            ORDER BY asked_at DESC
            LIMIT %s
            """,  # noqa: S608 - `conditions` is built from a fixed, hardcoded set above, no raw input
            params,
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
            saved=row[6],
        )
        for row in rows
    ]


def set_saved(entry_id: int, user_email: str, *, saved: bool) -> bool:
    """True if a history entry with this id, owned by this user, was updated.

    Scoped to `user_email` in the `UPDATE` itself, not checked separately —
    the only way to tell "doesn't exist" apart from "exists but isn't yours"
    is to not distinguish them, same reasoning as `auth.authenticate`'s
    single failure outcome. Either way the route returns 404, never leaking
    which case it was.
    """
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE query_history SET saved = %s WHERE id = %s AND user_email = %s",
            (saved, entry_id, user_email),
        )
        updated = cur.rowcount > 0
        conn.commit()
    return updated
