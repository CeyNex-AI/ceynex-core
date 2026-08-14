"""Knowledge graph schema — constraints for the shared Neo4j instance (SAD Figure 3).

STUB scope note: written to load apparel `fact_trade` data (SRS 3.3.4's named
apparel KG loader deliverable), but the *schema itself* is deliberately
sector-agnostic — `Country`/`Product`/`ExportRecord`, not apparel-specific
node labels — since Neo4j Community only serves one database per instance
(`docs/ARCHITECTURE_DELTA.md` D2), so agriculture data will land in the same
graph eventually. Flag it if a different shape is wanted before other
sectors' loaders are built against this one.

Graph shape
-----------
    (:Country {iso3, m49})-[:REPORTED]->(:ExportRecord)-[:TO]->(:Country)
                                              |
                                            [:OF]
                                              v
                                          (:Product {key, name, sector})

- `Country.iso3 = "WLD"` is an explicit World sentinel node, not a null
  partner — lets every query use one `-[:TO]->` pattern instead of an
  optional-match branch for world-aggregate rows (mirrors how `schema.sql`'s
  `fact_trade.partner_iso3 IS NULL` means "World").
- `ExportRecord.source_hash` is the connectors' own idempotency key
  (`ceynex/data/connectors/edb.py`/`jaaf.py`) reused as the graph's MERGE
  key — re-running the loader is a no-op for records already present.
- `Product.key` is `f"{sector}:{item.strip().lower()}"` — a light
  normalization, not a real controlled vocabulary. Known limitation carried
  over from the connectors: two source editions with punctuation-only
  differences in a product name (see `data/raw/edb/PROFILE.md`) still
  produce two distinct `Product` nodes here. Not solved in this loader.
"""

CONSTRAINTS = [
    "CREATE CONSTRAINT country_iso3 IF NOT EXISTS FOR (c:Country) REQUIRE c.iso3 IS UNIQUE",
    "CREATE CONSTRAINT product_key IF NOT EXISTS FOR (p:Product) REQUIRE p.key IS UNIQUE",
    "CREATE CONSTRAINT export_record_hash IF NOT EXISTS "
    "FOR (r:ExportRecord) REQUIRE r.source_hash IS UNIQUE",
]

WORLD_SENTINEL = {"iso3": "WLD", "m49": 0}


async def apply_schema(driver) -> None:
    """Create the graph's uniqueness constraints and the World sentinel node. Idempotent."""
    async with driver.session() as session:
        for statement in CONSTRAINTS:
            await session.run(statement)
        await session.run(
            "MERGE (w:Country {iso3: $iso3}) ON CREATE SET w.m49 = $m49",
            **WORLD_SENTINEL,
        )
