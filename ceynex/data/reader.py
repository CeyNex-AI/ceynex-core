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

import pandas as pd
import psycopg

from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

ANNUAL = "A"


class DatasetUnavailableError(RuntimeError):
    """Raised when the unified dataset cannot be read."""


def annual_series(
    item: str,
    *,
    sector: str | None = None,
    target: str = "export_value_usd",
    partner_iso3: str | None = None,
    dsn: str | None = None,
) -> pd.DataFrame:
    """One row per year for an item: `period`, `value`.

    Partner rows are summed to a national total unless `partner_iso3` narrows it.
    `partner_iso3 IS NULL` marks a row that is *already* a world total, so mixing
    those with per-partner rows would double-count; the filter below keeps the
    per-partner rows when any exist and falls back to the world rows when they do
    not, rather than adding the two together.
    """
    if target not in {"export_value_usd", "export_volume", "price"}:
        raise ValueError(f"{target} is not a measured column on fact_trade")

    conditions = ["frequency = %(frequency)s", "lower(item) = lower(%(item)s)", f"{target} IS NOT NULL"]
    params: dict[str, object] = {"item": item, "frequency": ANNUAL}

    if sector:
        conditions.append("lower(sector) = lower(%(sector)s)")
        params["sector"] = sector
    if partner_iso3:
        conditions.append("partner_iso3 = %(partner)s")
        params["partner"] = partner_iso3

    where = " AND ".join(conditions)
    sql = f"""
        WITH matched AS (SELECT * FROM fact_trade WHERE {where}),
             per_partner AS (SELECT * FROM matched WHERE partner_iso3 IS NOT NULL)
        SELECT extract(year FROM period_start)::int AS period,
               sum({target})::float8               AS value
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
        with psycopg.connect(dsn or postgres_dsn()) as conn, conn.cursor() as cur:
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
        with psycopg.connect(dsn or postgres_dsn()) as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT sector, item FROM fact_trade ORDER BY 1, 2")
            return [(str(s), str(i)) for s, i in cur.fetchall()]
    except psycopg.Error as exc:
        raise DatasetUnavailableError(f"could not read fact_trade: {exc}") from exc


__all__ = ["ANNUAL", "DatasetUnavailableError", "annual_series", "items"]
