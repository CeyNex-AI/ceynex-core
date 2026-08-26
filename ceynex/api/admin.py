"""Reads (and one write) behind the admin routes (SRS 3.5.4) that don't
already live somewhere else. Retraining and ingestion reuse the real hooks —
`ceynex.models.registry.retrain` and `ceynex.data.pipeline.run_source` — this
module only covers the two things nothing else exposes yet: `ingest_run`
status (the table's own schema.sql comment names it as backing this) and
`dq_flag` review.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import psycopg

from ceynex.settings import postgres_dsn


@dataclass(frozen=True)
class PipelineRun:
    run_id: int
    source_id: str
    started_at: str
    finished_at: str | None
    status: str
    rows_written: int
    error: str | None


def pipeline_status(limit: int = 20) -> list[PipelineRun]:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT run_id, source_id, started_at, finished_at, status, rows_written, error
            FROM ingest_run
            ORDER BY started_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [
        PipelineRun(
            run_id=row[0],
            source_id=row[1],
            started_at=row[2].isoformat(),
            finished_at=row[3].isoformat() if row[3] else None,
            status=row[4],
            rows_written=row[5],
            error=row[6],
        )
        for row in rows
    ]


@dataclass(frozen=True)
class DQFlag:
    flag_id: int
    item: str | None
    hs_code: str | None
    partner_iso3: str | None
    period_start: str | None
    metric: str | None
    source_a: str | None
    value_a: float | None
    source_b: str | None
    value_b: float | None
    pct_diff: float | None
    severity: str | None
    detected_at: str
    resolved: bool


def _stamp(value: date | datetime | None) -> str | None:
    return value.isoformat() if value else None


def _num(value: object) -> float | None:
    return float(value) if value is not None else None  # type: ignore[arg-type]


def list_dq_flags(
    *, resolved: bool | None = None, severity: str | None = None, limit: int = 50
) -> list[DQFlag]:
    """Unresolved-first, most-recent-first — that ordering is the point of a
    review queue: nothing already looked at should push down something new."""
    conditions = []
    params: list[object] = []
    if resolved is not None:
        conditions.append("resolved = %s")
        params.append(resolved)
    if severity is not None:
        conditions.append("severity = %s")
        params.append(severity)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT flag_id, item, hs_code, partner_iso3, period_start, metric,
                   source_a, value_a, source_b, value_b, pct_diff, severity,
                   detected_at, resolved
            FROM dq_flag
            {where}
            ORDER BY resolved ASC, detected_at DESC
            LIMIT %s
            """,  # noqa: S608 - `where` is built from a fixed condition set above, no raw input
            params,
        )
        rows = cur.fetchall()
    return [
        DQFlag(
            flag_id=row[0],
            item=row[1],
            hs_code=row[2],
            partner_iso3=row[3],
            period_start=_stamp(row[4]),
            metric=row[5],
            source_a=row[6],
            value_a=_num(row[7]),
            source_b=row[8],
            value_b=_num(row[9]),
            pct_diff=_num(row[10]),
            severity=row[11],
            detected_at=_stamp(row[12]) or "",
            resolved=bool(row[13]),
        )
        for row in rows
    ]


def resolve_dq_flag(flag_id: int) -> bool:
    """True if a flag with this id existed and was marked resolved."""
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute("UPDATE dq_flag SET resolved = true WHERE flag_id = %s", (flag_id,))
        updated = cur.rowcount > 0
        conn.commit()
    return updated
