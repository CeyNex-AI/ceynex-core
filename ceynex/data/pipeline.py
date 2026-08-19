"""Implements SRS 3.1.7 — the ingestion entrypoint.

    make ingest
    python -m ceynex.data.pipeline --sources all
    python -m ceynex.data.pipeline --sources comtrade --years 2020 2024 --offline

Connectors are registered here by source id. M1's and M3's connectors register
themselves the same way, so `--sources all` picks them up without this file
learning anything about their internals.

Idempotent: running it twice yields the same `fact_trade` row count. That is a
property of the writer's upsert, and `--verify` prints the count so it can be
checked rather than assumed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

import pandas as pd
import psycopg

from ceynex.contracts import DataSourceConnector
from ceynex.data.connectors.apparel_sources import EDB_SOURCE, JAAF_SOURCE
from ceynex.data.connectors.comtrade import ComtradeConnector
from ceynex.data.writer import UnifiedDatasetWriter, WriteResult
from ceynex.settings import postgres_dsn, redacted_dsn

log = logging.getLogger(__name__)

# source id -> factory. M1 and M3 add theirs here; nothing else changes.
#
# EDB and JAAF ignore `years`/`offline`: each is a fixed set of manually-saved
# report editions (`apparel_sources.py`), not an API pull with a date range or
# a cache to bypass, so there is nothing for those flags to select between.
CONNECTORS: dict[str, Callable[..., DataSourceConnector]] = {
    "comtrade": ComtradeConnector,
    "edb": lambda **_kwargs: EDB_SOURCE,
    "jaaf": lambda **_kwargs: JAAF_SOURCE,
}


def run_source(
    name: str,
    writer: UnifiedDatasetWriter,
    **kwargs: object,
) -> WriteResult:
    factory = CONNECTORS[name]
    connector = factory(**kwargs)  # type: ignore[arg-type]

    log.info("--- %s ---", name)
    raw = connector.fetch()
    records = connector.to_fact_trade(raw)
    log.info("%s: %d raw rows -> %d fact_trade rows", name, len(raw), len(records))

    result = writer.write(records, source_id=connector.source_id)

    # A year the source did not report is a hole in every time series built on
    # it. Reported here rather than left in the logs, because CAGR endpoints and
    # forecast windows both break quietly on a gap.
    try:
        missing = connector.manifest().notes.get("missing_years") or []
    except RuntimeError:
        missing = []
    if missing:
        result.warnings.append(
            f"{connector.source_id} reported no data at all for {missing} — "
            "any growth rate or forecast spanning those years has a gap in it"
        )
    return result


def verify(dsn: str | None = None) -> list[tuple[str, int]]:
    """Row counts per source. The Day 3 acceptance check from the plan."""
    with psycopg.connect(dsn or postgres_dsn()) as conn, conn.cursor() as cur:
        cur.execute("SELECT source_id, count(*) FROM fact_trade GROUP BY 1 ORDER BY 1")
        return [(str(row[0]), int(row[1])) for row in cur.fetchall()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest external sources into fact_trade.")
    parser.add_argument("--sources", nargs="+", default=["all"], help="'all' or source names")
    parser.add_argument("--years", nargs=2, type=int, metavar=("FROM", "TO"), default=None)
    parser.add_argument("--offline", action="store_true", help="use only cached raw responses")
    parser.add_argument("--verify", action="store_true", help="print row counts per source and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.verify:
        return _print_counts()

    names = list(CONNECTORS) if "all" in args.sources else args.sources
    unknown = [n for n in names if n not in CONNECTORS]
    if unknown:
        log.error("unknown sources: %s (known: %s)", unknown, sorted(CONNECTORS))
        return 2

    kwargs: dict[str, object] = {}
    if args.years:
        kwargs["years"] = tuple(range(args.years[0], args.years[1] + 1))
    if args.offline:
        kwargs["offline"] = True

    writer = UnifiedDatasetWriter()
    results: list[WriteResult] = []
    for name in names:
        try:
            results.append(run_source(name, writer, **kwargs))
        except Exception as exc:  # noqa: BLE001 - one bad source must not stop the rest
            log.exception("%s failed", name)
            results.append(
                WriteResult(name, None, 0, 0, 0, None, status="failed", error=str(exc))
            )

    print()
    for result in results:
        marker = "ok " if result.status == "success" else "FAIL"
        print(f"  {marker} {result.source_id:<14} {result.rows_written:>7,} rows"
              f"  {result.dq_flags:>4} dq flags")
        if result.error:
            print(f"       {result.error}")
        for warning in result.warnings:
            print(f"       warning: {warning}")

    print()
    _print_counts()
    return 0 if all(r.status == "success" for r in results) else 1


def _print_counts() -> int:
    try:
        counts = verify()
    except psycopg.Error as exc:
        log.error("could not read fact_trade from %s: %s", redacted_dsn(), exc)
        return 1
    if not counts:
        print("  fact_trade is empty")
        return 0
    width = max(len(name) for name, _ in counts)
    print("  fact_trade by source:")
    for name, count in counts:
        print(f"    {name:<{width}}  {count:>8,}")
    print(f"    {'TOTAL':<{width}}  {sum(c for _, c in counts):>8,}")
    return 0


def concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine connector outputs, tolerating empty ones."""
    non_empty = [f for f in frames if not f.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


if __name__ == "__main__":
    sys.exit(main())
