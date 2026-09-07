"""Append-only audit trail for admin actions — SRS 3.4.7.

`ceynex/api/history.py` records what users ask; this records what admins do.
`POST /api/admin/retrain`, `/pipeline/ingest` and `/dq-flags/{id}/resolve` all
mutate real state and, before this module, left no record of who did it or
when — the gap `docs/DEFERRED.md`'s "Audit logging" section calls out.

Deliberately **not** opportunistic like `history.record()`. A query still
answers normally if its history row fails to write, because losing one row of
"what a user asked" costs nothing. An admin mutation happening with no audit
row is exactly the failure DEFERRED.md calls worse than having no audit log at
all — "it invites the reader to trust a record that is not complete." So
`record()` lets `psycopg.Error` propagate, and the route layer (`routes/admin.py`)
calls it *before* performing the mutation it covers and turns a failure into a
503 — no admin action ever executes unlogged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg

from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    actor_email TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT,
    logged_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS audit_log_logged_at_idx ON audit_log (logged_at DESC);
"""


def ensure_table() -> None:
    # Swallowed like history.ensure_table(): this only runs once at API
    # startup and a Postgres that isn't up yet shouldn't crash the process.
    # record()'s own fail-closed behaviour is what actually protects SRS
    # 3.4.7 once the table exists.
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()
    except psycopg.Error as exc:
        log.warning("audit_log table not ensured (postgres unreachable?): %s", exc)


def record(*, actor_email: str, action: str, target: str | None) -> None:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO audit_log (actor_email, action, target) VALUES (%s, %s, %s)",
            (actor_email, action, target),
        )
        conn.commit()


@dataclass(frozen=True)
class AuditEntry:
    id: int
    actor_email: str
    action: str
    target: str | None
    logged_at: str


def list_entries(limit: int = 50) -> list[AuditEntry]:
    """Most-recent-first — a review queue reads newest first, same as
    `admin.list_dq_flags`."""
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, actor_email, action, target, logged_at
            FROM audit_log
            ORDER BY logged_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [
        AuditEntry(
            id=row[0], actor_email=row[1], action=row[2], target=row[3],
            logged_at=row[4].isoformat(),
        )
        for row in rows
    ]
