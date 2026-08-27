"""Notification preferences — Account.tsx's "Notification preferences" stub,
now built. Stored settings only: CeyNex has no notification-delivery system
(no outbound email, no push) to wire these into yet, and building one is out
of scope here — these are what a future delivery system would read, not a
delivery system itself.

Same persistence shape as `ceynex/api/history.py`. One row per user rather
than an event log, upserted via `ON CONFLICT` — unlike history, there is only
ever "the current preferences", never a list of past ones. A user who has
never saved a preference gets the defaults back from `get_for_user` without a
row being written just for reading; only `upsert` writes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg

from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS notification_preferences (
    user_email TEXT PRIMARY KEY,
    dq_flag_alerts BOOLEAN NOT NULL DEFAULT true,
    forecast_updates BOOLEAN NOT NULL DEFAULT true,
    weekly_digest BOOLEAN NOT NULL DEFAULT false,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_DEFAULTS = {"dq_flag_alerts": True, "forecast_updates": True, "weekly_digest": False}


def ensure_table() -> None:
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()
    except psycopg.Error as exc:
        log.warning("notification_preferences table not ensured (postgres unreachable?): %s", exc)


@dataclass(frozen=True)
class Preferences:
    dq_flag_alerts: bool
    forecast_updates: bool
    weekly_digest: bool


def get_for_user(user_email: str) -> Preferences:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT dq_flag_alerts, forecast_updates, weekly_digest "
            "FROM notification_preferences WHERE user_email = %s",
            (user_email,),
        )
        row = cur.fetchone()
    if row is None:
        return Preferences(**_DEFAULTS)
    return Preferences(dq_flag_alerts=row[0], forecast_updates=row[1], weekly_digest=row[2])


def upsert(
    user_email: str, *, dq_flag_alerts: bool, forecast_updates: bool, weekly_digest: bool
) -> Preferences:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO notification_preferences
                (user_email, dq_flag_alerts, forecast_updates, weekly_digest, updated_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (user_email) DO UPDATE SET
                dq_flag_alerts = EXCLUDED.dq_flag_alerts,
                forecast_updates = EXCLUDED.forecast_updates,
                weekly_digest = EXCLUDED.weekly_digest,
                updated_at = now()
            """,
            (user_email, dq_flag_alerts, forecast_updates, weekly_digest),
        )
        conn.commit()
    return Preferences(
        dq_flag_alerts=dq_flag_alerts, forecast_updates=forecast_updates, weekly_digest=weekly_digest
    )
