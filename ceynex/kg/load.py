"""Apparel knowledge graph loader (SRS 3.3.4) — provisional, apparel-only.

STUB — like `ceynex/data/pipeline.py`, scoped to what M3 owns (apparel) and
meant to be swapped for a real multi-sector loader, not extended in place.
The schema it writes to (`ceynex/kg/schema.py`) is deliberately sector-agnostic
so that swap doesn't require a graph migration, only a different Python
entrypoint pointed at the same node/relationship shape.

Reads `fact_trade`-shaped rows (currently: the staging parquet produced by
`ceynex/data/pipeline.py` — swap for a Postgres read once that's the source
of truth) and MERGEs them into Neo4j via one batched `UNWIND` per run, so
re-running this loader is a no-op for records already present.

Usage: `python -m ceynex.kg.load --schema` (apply constraints first, then
load) or `python -m ceynex.kg.load` (load only, assumes schema already
applied). `--agreements` is accepted for compatibility with the Makefile's
`kg-load` target but not implemented here — GSP+/FTA agreement edges are
Trade Economics Agent scope (SRS 3.1.5), not Apparel & Manufacturing's.

First-loaded-wins on a given `source_hash`: `MERGE ... ON CREATE SET` only
writes properties the first time a node is created, so if two EDB editions'
overlapping trailing years ever genuinely disagree (a revision, not just a
duplicate), whichever edition is listed first in `apparel_sources.py` wins,
not the most recent one. Confirmed on the real data that the one pair of
overlapping years checked reported identical figures across editions, so
this hasn't been a live issue — but it's a policy choice, not a guarantee,
and worth a real "latest edition wins" rule if revisions turn out to matter.
"""

import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd
from neo4j import AsyncGraphDatabase

from ceynex.kg.schema import apply_schema

STAGING_PARQUET = Path("data/staging/fact_trade_apparel.parquet")

_LOAD_QUERY = """
UNWIND $rows AS row
MERGE (reporter:Country {iso3: row.reporter_iso3})
  ON CREATE SET reporter.m49 = row.reporter_m49
MERGE (partner:Country {iso3: row.partner_iso3})
  ON CREATE SET partner.m49 = row.partner_m49
MERGE (product:Product {key: row.product_key})
  ON CREATE SET product.name = row.item, product.sector = row.sector
MERGE (record:ExportRecord {source_hash: row.source_hash})
  ON CREATE SET
    record.source_id = row.source_id,
    record.period_start = date(row.period_start),
    record.period_end = date(row.period_end),
    record.frequency = row.frequency,
    record.export_value_usd = row.export_value_usd,
    record.export_volume = row.export_volume,
    record.volume_unit = row.volume_unit,
    record.price = row.price,
    record.price_unit = row.price_unit,
    record.fx_usd_lkr = row.fx_usd_lkr
MERGE (reporter)-[:REPORTED]->(record)
MERGE (record)-[:TO]->(partner)
MERGE (record)-[:OF]->(product)
"""


def _neo4j_uri() -> str:
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687")


def _neo4j_auth() -> tuple[str, str]:
    return (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "ceynex_dev_pw"),
    )


def fact_trade_to_rows(df: pd.DataFrame) -> list[dict]:
    """Convert a `fact_trade`-shaped DataFrame to Bolt-serializable row dicts.

    - `partner_iso3 = NULL` ("World", per `schema.sql`) becomes the graph's
      `"WLD"` sentinel (`ceynex/kg/schema.py`), not a null property.
    - NaN (pandas' float null) becomes `None` — the Neo4j driver rejects NaN.
    """
    df = df.copy()
    df["product_key"] = df["sector"].str.lower() + ":" + df["item"].str.strip().str.lower()
    df["partner_iso3"] = df["partner_iso3"].fillna("WLD")
    df["partner_m49"] = df["partner_m49"].fillna(0).astype(int)
    df["period_start"] = df["period_start"].astype(str)
    df["period_end"] = df["period_end"].astype(str)
    records = df.to_dict(orient="records")
    for row in records:
        for key, value in row.items():
            if isinstance(value, float) and pd.isna(value):
                row[key] = None
    return records


async def load_fact_trade(driver, df: pd.DataFrame, batch_size: int = 500) -> int:
    """MERGE every row of `df` into the graph. Returns the row count loaded."""
    rows = fact_trade_to_rows(df)
    async with driver.session() as session:
        for i in range(0, len(rows), batch_size):
            await session.run(_LOAD_QUERY, rows=rows[i : i + batch_size])
    return len(rows)


async def _main_async(apply_schema_first: bool, parquet_path: Path) -> None:
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found — run `python -m ceynex.data.pipeline "
            "--sources apparel` first to produce it."
        )
    driver = AsyncGraphDatabase.driver(_neo4j_uri(), auth=_neo4j_auth())
    try:
        if apply_schema_first:
            await apply_schema(driver)
            print("Applied schema constraints.")
        df = pd.read_parquet(parquet_path)
        count = await load_fact_trade(driver, df)
        print(f"Loaded {count} ExportRecord nodes from {parquet_path}")
    finally:
        await driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load apparel fact_trade data into the KG.")
    parser.add_argument("--schema", action="store_true", help="apply constraints before loading")
    parser.add_argument(
        "--agreements",
        action="store_true",
        help="accepted for Makefile compatibility; not implemented (Trade Economics scope)",
    )
    parser.add_argument("--parquet", default=str(STAGING_PARQUET), type=Path)
    args = parser.parse_args()

    if args.agreements:
        print("--agreements: not implemented in this loader (Trade Economics Agent scope).")

    asyncio.run(_main_async(args.schema, args.parquet))


if __name__ == "__main__":
    main()
