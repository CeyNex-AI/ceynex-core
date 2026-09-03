"""Supports docs/ARCHITECTURE_DELTA.md D11 — where a computed trending panel rests.

The refresher computes; the route reads. Something has to sit between them, and
the choice is less obvious than it looks:

- **Process memory** is disqualified by the deployment. `uvicorn --workers 2` and
  a lock that lets exactly one worker refresh means the *other* worker's
  `/api/news/trending` would be empty forever — and which one a browser hits is
  down to whichever accepted the connection.
- **Redis** has the right shape but the wrong constraint: `redis_url()` is None
  on a developer's machine, and the panel has to work there.
- **Postgres** is the one datastore that is not optional. `/health` already
  reports `degraded` without it, both workers read it, and it survives a restart
  — so the panel is warm immediately after a deploy rather than blank for up to
  an hour.

Additive to the frozen contracts schema rather than part of it, for exactly the
reason `api/history.py` gives for `query_history`: this is a `ceynex-core`-only
concern no other member's code touches, so it has no business behind that repo's
three-reviewer gate. `ensure_table()` runs at startup, same idempotent-DDL
spirit as `data.bootstrap.apply_schema`.

JSONB rather than a normalised topic table
------------------------------------------
The snapshot is read whole and never queried by field. A blob also means adding a
field to `TrendingTopic` needs no migration, which matters when the alternative
lives in a repo behind a PR gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg

from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

#: Rows kept per scope. Two days of hourly refreshes — enough to see whether the
#: refresher has been running, far short of a time series nobody reads.
HISTORY_ROWS = 48

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS news_trending_snapshot (
    id BIGSERIAL PRIMARY KEY,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    scope TEXT NOT NULL,
    payload JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS news_trending_snapshot_scope_idx
    ON news_trending_snapshot (scope, computed_at DESC);
"""


@dataclass(frozen=True)
class Snapshot:
    scope: str
    computed_at: datetime
    topics: list[dict]
    partial: bool = False


def ensure_table() -> None:
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()
    except psycopg.Error as exc:
        log.warning("news_trending_snapshot table not ensured (postgres unreachable?): %s", exc)


def write(scope: str, topics: list[dict], *, partial: bool = False) -> bool:
    """Store one scope's panel. Reports rather than raises.

    Appends and then trims, rather than updating in place: a reader mid-request
    never sees a half-written panel, and the trailing rows are a cheap record of
    whether the refresher has actually been running.
    """
    payload = json.dumps({"topics": topics, "partial": partial})
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO news_trending_snapshot (scope, payload) VALUES (%s, %s::jsonb)",
                (scope, payload),
            )
            cur.execute(
                """
                DELETE FROM news_trending_snapshot
                WHERE scope = %s AND id NOT IN (
                    SELECT id FROM news_trending_snapshot
                    WHERE scope = %s ORDER BY computed_at DESC LIMIT %s
                )
                """,
                (scope, scope, HISTORY_ROWS),
            )
            conn.commit()
    except psycopg.Error as exc:
        log.warning("could not store the %s trending snapshot: %s", scope, exc)
        return False
    return True


def latest(scope: str) -> Snapshot | None:
    """The most recent panel for a scope, or None if there has never been one.

    None is an ordinary state — before the first refresh completes, and after a
    fresh deploy against an empty database. The route turns it into `warming`
    rather than into an error.
    """
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT computed_at, payload FROM news_trending_snapshot
                WHERE scope = %s ORDER BY computed_at DESC LIMIT 1
                """,
                (scope,),
            )
            row = cur.fetchone()
    except psycopg.Error as exc:
        log.warning("could not read the %s trending snapshot: %s", scope, exc)
        return None

    if row is None:
        return None

    computed_at, payload = row
    if isinstance(payload, str):  # psycopg returns jsonb as dict, but be tolerant
        payload = json.loads(payload)
    return Snapshot(
        scope=scope,
        computed_at=computed_at if computed_at.tzinfo else computed_at.replace(tzinfo=UTC),
        topics=list(payload.get("topics") or []),
        partial=bool(payload.get("partial")),
    )


def age_seconds(snapshot: Snapshot | None) -> float | None:
    if snapshot is None:
        return None
    return (datetime.now(tz=UTC) - snapshot.computed_at).total_seconds()


__all__ = [
    "CREATE_TABLE_SQL",
    "HISTORY_ROWS",
    "Snapshot",
    "age_seconds",
    "ensure_table",
    "latest",
    "write",
]
