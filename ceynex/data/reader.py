"""Implements SRS 3.10 — reads from the unified dataset (SAD §5.1).

The layer rule in `CLAUDE.md` says Postgres and Parquet are reached only through
`ceynex/data/writer.py` for writes and the unified dataset client for reads.
This is that client. It stayed unwritten while everything that needed history
read it from the knowledge graph instead; the backtest harness is the first
caller that needs the *fact table*, because a model has to train on the same
records the writer persisted, not on their projection into the graph.

Deliberately small. It answers "give me this item's series" and nothing else;
aggregation, joins and analysis belong to the caller.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

ANNUAL = "A"


class DatasetUnavailableError(RuntimeError):
    """Raised when the unified dataset cannot be read."""


# How each measured column collapses many rows in a year into one number.
#
# Values and volumes are extensive: partner rows add up to a national total.
# A price is intensive and does not -- summing 71 per-partner unit values gives a
# number in no unit at all, which is exactly the bug this table exists to stop.
# `data/align.py` states the same rule for the resampling path, and
# `DataCleaner.resample` implements it; this is the read side of one rule.
_EXTENSIVE_TARGETS = frozenset({"export_value_usd", "export_volume"})
MEASURED_TARGETS = _EXTENSIVE_TARGETS | {"price"}


def annual_series(
    item: str,
    *,
    sector: str | None = None,
    target: str = "export_value_usd",
    partner_iso3: str | None = None,
    source_id: str | None = None,
    dsn: str | None = None,
) -> pd.DataFrame:
    """One row per year for an item: `period`, `value`.

    Partner rows are summed to a national total unless `partner_iso3` narrows it.
    `partner_iso3 IS NULL` marks a row that is *already* a world total, so mixing
    those with per-partner rows would double-count; the filter below keeps the
    per-partner rows when any exist and falls back to the world rows when they do
    not, rather than adding the two together.

    **`price` is averaged, not summed** -- volume-weighted where the same rows
    carry a volume, unweighted where they do not. Found live 2026-09-03: cinnamon
    prices in `fact_trade` are UN Comtrade per-partner unit values already in
    USD/kg, roughly 6-16 of them per partner, and summing the 71 partner rows for
    2015 produced 854.01, served to the user as "854.01 USD/kg". The mean is
    12.03, which is what cinnamon costs.

    `source_id` narrows the series to one source. Without it this blends every
    source holding a price for the item -- a Comtrade unit value and a FAOSTAT
    producer price are different measurements of different things, and averaging
    across them produces a number neither source would recognise, attributed in
    the evidence panel to whichever one the caller named.
    """
    if target not in MEASURED_TARGETS:
        raise ValueError(f"{target} is not a measured column on fact_trade")

    conditions = ["frequency = %(frequency)s", "lower(item) = lower(%(item)s)", f"{target} IS NOT NULL"]
    params: dict[str, object] = {"item": item, "frequency": ANNUAL}

    if sector:
        conditions.append("lower(sector) = lower(%(sector)s)")
        params["sector"] = sector
    if partner_iso3:
        conditions.append("partner_iso3 = %(partner)s")
        params["partner"] = partner_iso3
    if source_id:
        conditions.append("source_id = %(source_id)s")
        params["source_id"] = source_id

    if target in _EXTENSIVE_TARGETS:
        aggregate = f"sum({target})::float8"
    else:
        # Weighted by export_volume, so a partner taking 90% of the volume moves
        # the national price 90% as much as it should.
        #
        # Only when *every* row in the year carries a volume (`count(w) =
        # count(*)`): weighting a subset would drop the unweighted rows from the
        # average entirely, so one large partner with a missing volume would
        # vanish from the year's price rather than merely be weighted oddly. A
        # plain mean over all the rows is the honest fallback, and it is what
        # single-row-per-year sources like FAOSTAT get anyway.
        aggregate = (
            f"(CASE WHEN count(export_volume) = count(*) AND sum(export_volume) > 0 "
            f"      THEN sum({target} * export_volume) / sum(export_volume) "
            f"      ELSE avg({target}) "
            f" END)::float8"
        )

    where = " AND ".join(conditions)
    sql = f"""
        WITH matched AS (SELECT * FROM fact_trade WHERE {where}),
             per_partner AS (SELECT * FROM matched WHERE partner_iso3 IS NOT NULL)
        SELECT extract(year FROM period_start)::int AS period,
               {aggregate}                          AS value
        FROM (
            SELECT * FROM per_partner
            UNION ALL
            SELECT * FROM matched
            WHERE partner_iso3 IS NULL
              AND NOT EXISTS (SELECT 1 FROM per_partner)
        ) AS series
        GROUP BY 1
        ORDER BY 1
    """

    try:
        with psycopg.connect(dsn or postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    except psycopg.Error as exc:
        raise DatasetUnavailableError(f"could not read fact_trade: {exc}") from exc

    frame = pd.DataFrame(rows, columns=["period", "value"])
    log.info("%s: %d annual observations of %s", item, len(frame), target)
    return frame


def items(dsn: str | None = None) -> list[tuple[str, str]]:
    """Every `(sector, item)` pair present in the fact table."""
    try:
        with psycopg.connect(dsn or postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT sector, item FROM fact_trade ORDER BY 1, 2")
            return [(str(s), str(i)) for s, i in cur.fetchall()]
    except psycopg.Error as exc:
        raise DatasetUnavailableError(f"could not read fact_trade: {exc}") from exc


def relevant_dq_flags(
    item: str,
    metric: str,
    *,
    period_start: int | None = None,
    period_end: int | None = None,
    dsn: str | None = None,
) -> list[dict[str, Any]]:
    """Material/severe cross-source flags for an answer's source window.

    This is deliberately a read-only companion to :func:`annual_series`.
    Callers keep and report their measured value; a DQ flag is evidence of a
    discrepancy, never authority to silently reconcile or delete it.
    """
    if not item or not metric:
        raise ValueError("item and metric are required to look up data-quality flags")
    start = date(period_start, 1, 1) if period_start is not None else None
    end = date(period_end, 12, 31) if period_end is not None else None
    statement = """
        SELECT period_start::date AS period_start,
               metric,
               source_a,
               value_a::float8 AS value_a,
               source_b,
               value_b::float8 AS value_b,
               pct_diff::float8 AS pct_diff,
               severity
        FROM dq_flag
        WHERE lower(item) = lower(%(item)s)
          AND metric = %(metric)s
          AND severity IN ('material', 'severe')
          AND (%(period_start)s::date IS NULL OR period_start >= %(period_start)s::date)
          AND (%(period_end)s::date IS NULL OR period_start <= %(period_end)s::date)
        ORDER BY period_start, severity, source_a, source_b
    """
    try:
        with psycopg.connect(dsn or postgres_dsn(), connect_timeout=3) as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                statement,
                {"item": item, "metric": metric, "period_start": start, "period_end": end},
            )
            rows = cur.fetchall()
    except psycopg.Error as exc:
        raise DatasetUnavailableError(f"could not read dq_flag: {exc}") from exc

    return [dict(row) for row in rows]


__all__ = ["ANNUAL", "DatasetUnavailableError", "annual_series", "items", "relevant_dq_flags"]
