"""Implements SRS 3.10.2 — applies the unified dataset schema and seeds its dimensions.

Replaces the `docker-entrypoint-initdb.d` mount the dev stack used to carry. An
initdb script only ever runs against a first-boot empty volume, so it cannot
apply a migration to a database that already exists — which is exactly the
situation on the deployed VM, whose compose has no initdb hook at all. Applying
the schema from code means one command serves both.

    make db-init
    python -m ceynex.data.bootstrap --dry-run

Idempotent, and safe to run against a database two teammates are already
loading into: every statement in `schema.sql` is `CREATE ... IF NOT EXISTS`, and
the dimension seeds are `ON CONFLICT DO UPDATE`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from importlib.resources import files

import psycopg

from ceynex.data.crosswalk import dim_country_rows, dim_hs_rows
from ceynex.settings import postgres_dsn, redacted_dsn

log = logging.getLogger(__name__)

CONTRACT_TABLES = ("dim_country", "dim_hs", "fact_trade", "dq_flag", "ingest_run")


def schema_sql() -> str:
    """The frozen DDL, read from the installed ceynex-contracts package."""
    return (files("ceynex.contracts") / "schema" / "schema.sql").read_text(encoding="utf-8")


def apply_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(schema_sql())


UPSERT_CONSTRAINT = "fact_trade_upsert_key"


def apply_upsert_constraint(conn: psycopg.Connection) -> None:
    """Deviation D5 — a NULLS NOT DISTINCT unique constraint to upsert against.

    `fact_trade`'s declared UNIQUE spans `hs_code` and `partner_iso3`, both
    nullable. Postgres treats NULLs as distinct inside a unique constraint, so
    rows differing only in a NULL never conflict: `ON CONFLICT` would miss them
    and every re-ingest would insert duplicates instead of updating, silently.
    `NULLS NOT DISTINCT` (Postgres 15+) is the fix.

    A *constraint* rather than a bare index, because `ON CONFLICT ON CONSTRAINT`
    names it explicitly. Inferring an arbiter from a column list would be
    ambiguous here — the contract's own constraint covers the same columns, and
    letting Postgres pick between them is how this silently reverts.

    The declared constraint is left in place. It is strictly weaker than this
    one, so it never blocks anything, and keeping it means the frozen DDL remains
    literally true of the database.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_constraint WHERE conname = %s", (UPSERT_CONSTRAINT,))
        if cur.fetchone():
            return

        # An earlier version of this function created a bare unique index under
        # the same name. `ON CONFLICT ON CONSTRAINT` cannot name an index, so the
        # index has to give way to a real constraint — and checking pg_constraint
        # alone would miss it and fail with "relation already exists".
        cur.execute(
            "SELECT 1 FROM pg_class WHERE relname = %s AND relkind = 'i'",
            (UPSERT_CONSTRAINT,),
        )
        if cur.fetchone():
            log.info("replacing the legacy %s index with a constraint", UPSERT_CONSTRAINT)
            cur.execute(f"DROP INDEX IF EXISTS {UPSERT_CONSTRAINT}")  # noqa: S608

        cur.execute(
            f"""
            ALTER TABLE fact_trade
              ADD CONSTRAINT {UPSERT_CONSTRAINT}
              UNIQUE NULLS NOT DISTINCT (
                  source_id, item, hs_code, reporter_iso3,
                  partner_iso3, period_start, frequency
              )
            """  # noqa: S608 - identifier is a module constant
        )


def seed_dimensions(conn: psycopg.Connection) -> tuple[int, int]:
    """Load dim_country and dim_hs from the committed reference tables."""
    countries = dim_country_rows()
    hs_codes = dim_hs_rows()

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO dim_country (iso3, m49, name) VALUES (%s, %s, %s)
            ON CONFLICT (iso3) DO UPDATE SET m49 = EXCLUDED.m49, name = EXCLUDED.name
            """,
            countries,
        )
        cur.executemany(
            """
            INSERT INTO dim_hs (hs_code, description, sector) VALUES (%s, %s, %s)
            ON CONFLICT (hs_code) DO UPDATE
                SET description = EXCLUDED.description, sector = EXCLUDED.sector
            """,
            hs_codes,
        )
    return len(countries), len(hs_codes)


def verify(conn: psycopg.Connection) -> dict[str, int]:
    """Row counts per contracted table, so the caller can see it worked."""
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for table in CONTRACT_TABLES:
            cur.execute(
                "SELECT to_regclass(%s) IS NOT NULL",
                (f"public.{table}",),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                raise RuntimeError(f"{table} was not created — schema.sql did not apply")
            cur.execute(f"SELECT count(*) FROM {table}")  # noqa: S608 - fixed table list
            fetched = cur.fetchone()
            counts[table] = fetched[0] if fetched else 0
    return counts


def bootstrap(dsn: str | None = None, *, dry_run: bool = False) -> dict[str, int]:
    dsn = dsn or postgres_dsn()
    log.info("connecting to %s", redacted_dsn(dsn))

    if dry_run:
        countries, hs_codes = len(dim_country_rows()), len(dim_hs_rows())
        log.info(
            "dry run: would apply %d chars of DDL and seed %d countries, %d hs codes",
            len(schema_sql()),
            countries,
            hs_codes,
        )
        return {"dim_country": countries, "dim_hs": hs_codes}

    with psycopg.connect(dsn) as conn:
        apply_schema(conn)
        apply_upsert_constraint(conn)
        countries, hs_codes = seed_dimensions(conn)
        conn.commit()
        counts = verify(conn)

    log.info("seeded %d countries and %d hs codes", countries, hs_codes)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the CeyNex Postgres schema and seed dimensions.")
    parser.add_argument("--dsn", default=None, help="override POSTGRES_URL / POSTGRES_* from the environment")
    parser.add_argument("--dry-run", action="store_true", help="report what would happen, connect to nothing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        counts = bootstrap(args.dsn, dry_run=args.dry_run)
    except psycopg.OperationalError as exc:
        log.error("could not connect to %s\n%s", redacted_dsn(args.dsn or postgres_dsn()), exc)
        log.error("is the stack up? `make up` locally, or check DATABASE_INTERNAL_IP on the VM")
        return 1

    width = max(len(name) for name in counts)
    for table, count in counts.items():
        print(f"  {table:<{width}}  {count:>7,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
