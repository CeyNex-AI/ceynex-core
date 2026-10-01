"""Implements FR-DAT-03's other half and SAD C11: knowing a source has gone stale.

A scheduled refresh (ceynex-infra's `ops/refresh.sh`) is only half of
keeping data current; the other half is noticing when it has stopped working.
This reads the newest successful and the newest failed `ingest_run` for every
source and judges each against its cadence in `config/sources.yaml`. A source
with no cadence is refreshed by hand from saved files and is never counted
stale.

Three readers:
- `GET /api/admin/pipeline/freshness`: the Admin page's freshness card;
- `/health`'s `detail.stale_sources`: a count only, because /health is public,
  which an uptime check can alert on with no email infrastructure;
- the refresh script's log line.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg

from ceynex import settings
from ceynex.settings import postgres_dsn

SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class SourceFreshness:
    source_id: str
    #: None: refreshed by hand, never counted stale.
    cadence_days: int | None
    #: Re-ingested by the monthly cron.
    refresh: bool
    last_success_at: datetime | None
    last_success_rows: int | None
    last_failure_at: datetime | None
    last_error: str | None
    #: Days since the last success; None if there has never been one.
    age_days: float | None
    stale: bool


def configured() -> dict[str, dict]:
    return settings.load_config("sources").get("sources", {})


def assess(
    config: dict[str, dict],
    successes: dict[str, tuple[datetime, int]],
    failures: dict[str, tuple[datetime, str | None]],
    now: datetime,
) -> list[SourceFreshness]:
    """Pure: every configured source, plus any source with runs but no config."""
    names = list(config) + sorted((set(successes) | set(failures)) - set(config))
    rows = []
    for name in names:
        entry = config.get(name, {})
        cadence = entry.get("cadence_days")
        success_at, success_rows = successes.get(name, (None, None))
        failure_at, error = failures.get(name, (None, None))
        age = (now - success_at).total_seconds() / SECONDS_PER_DAY if success_at else None
        stale = cadence is not None and (age is None or age > cadence)
        rows.append(SourceFreshness(
            source_id=name,
            cadence_days=cadence,
            refresh=bool(entry.get("refresh", False)),
            last_success_at=success_at,
            last_success_rows=success_rows,
            last_failure_at=failure_at,
            last_error=error,
            age_days=round(age, 1) if age is not None else None,
            stale=stale,
        ))
    return rows


def per_source(dsn: str | None = None, now: datetime | None = None) -> list[SourceFreshness]:
    """Two indexed reads of `ingest_run`; raises psycopg.Error if Postgres is down."""
    with psycopg.connect(dsn or postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (source_id) source_id, finished_at, rows_written
              FROM ingest_run
             WHERE status = 'success' AND finished_at IS NOT NULL
             ORDER BY source_id, finished_at DESC
            """
        )
        successes = {row[0]: (row[1], int(row[2] or 0)) for row in cur.fetchall()}
        cur.execute(
            """
            SELECT DISTINCT ON (source_id) source_id, coalesce(finished_at, started_at), error
              FROM ingest_run
             WHERE status = 'failed'
             ORDER BY source_id, coalesce(finished_at, started_at) DESC
            """
        )
        failures = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    return assess(configured(), successes, failures, now or datetime.now(UTC))


def stale_count(rows: list[SourceFreshness]) -> int:
    return sum(1 for row in rows if row.stale)
