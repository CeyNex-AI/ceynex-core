"""Apparel KG spot-check suite (SRS 3.3.4, named deliverable — never cut).

Loads the real apparel `fact_trade` staging data into a live Neo4j instance
and asserts specific figures already independently verified against the
source files directly (see `data/raw/edb/PROFILE.md`, `data/raw/jaaf/PROFILE.md`)
still come back correctly via Cypher. This is what makes SRS 3.1.6's "query
the graph, not a model" claim checkable end to end, not just at the
connector layer — a bug in `ceynex/kg/load.py` (a wrong MERGE key, a mangled
date, a units mistake) would pass every connector test and still corrupt
what the agent reports.

Requires a running Neo4j (`make up`) and the staging parquet already produced
(`python -m ceynex.data.pipeline --sources apparel`) — skips cleanly if
either is missing rather than failing the whole suite when no docker stack
is up. Marked `integration`; deselect with `pytest -m "not integration"`.
"""

import os

import pandas as pd
import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase

from ceynex.kg.load import STAGING_PARQUET, load_fact_trade
from ceynex.kg.schema import apply_schema

# loop_scope="module" matches the fixture's scope="module" below — without
# it, pytest-asyncio gives each test its own event loop by default, and the
# driver's connections (opened in the first test's loop) break on every
# later test ("'NoneType' object has no attribute 'send'").
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]


def _neo4j_uri() -> str:
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687")


def _neo4j_auth() -> tuple[str, str]:
    return (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "ceynex_dev_pw"),
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def loaded_driver():
    if not STAGING_PARQUET.exists():
        pytest.skip(f"{STAGING_PARQUET} not found — run the apparel pipeline first")
    driver = AsyncGraphDatabase.driver(_neo4j_uri(), auth=_neo4j_auth())
    try:
        await driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001 — any connection failure means "skip", not "fail"
        await driver.close()
        pytest.skip(f"Neo4j not reachable at {_neo4j_uri()}: {exc}")
    await apply_schema(driver)
    df = pd.read_parquet(STAGING_PARQUET)
    await load_fact_trade(driver, df)
    yield driver
    await driver.close()


async def _run(driver, cypher: str, **params) -> list[dict]:
    async with driver.session() as session:
        result = await session.run(cypher, **params)
        return [record.data() async for record in result]


async def test_world_sentinel_exists(loaded_driver):
    rows = await _run(loaded_driver, "MATCH (w:Country {iso3: 'WLD'}) RETURN w.m49 AS m49")
    assert rows == [{"m49": 0}]


async def test_edb_2023_us_apparel_matches_the_source_pdf(loaded_driver):
    # Verified directly against export-performance-indicators-of-sri-lanka-2024.pdf,
    # table 25.79 "APPREL", United States, 2023 column: 1,782.72 (USD Mn).
    rows = await _run(
        loaded_driver,
        """
        MATCH (:Country {iso3: 'LKA'})-[:REPORTED]->(r:ExportRecord {source_id: 'EDB'})
              -[:TO]->(:Country {iso3: 'USA'}),
              (r)-[:OF]->(:Product {key: 'apparel:apprel'})
        WHERE r.period_start = date('2023-01-01')
        RETURN r.export_value_usd AS value
        """,
    )
    assert rows == [{"value": pytest.approx(1782.72 * 1_000_000.0)}]


async def test_jaaf_2025_january_total_matches_the_source_page(loaded_driver):
    # Verified directly against the real saved "Annual Exports" page, Total
    # table, 2025 row, January column: 437.07 (USD Mn).
    rows = await _run(
        loaded_driver,
        """
        MATCH (:Country {iso3: 'LKA'})-[:REPORTED]->(r:ExportRecord {source_id: 'JAAF'})
              -[:TO]->(:Country {iso3: 'WLD'})
        WHERE r.period_start = date('2025-01-01')
        RETURN r.export_value_usd AS value
        """,
    )
    assert rows == [{"value": pytest.approx(437.07 * 1_000_000.0)}]


async def test_archive_2018_us_apparel_matches_the_source_pdf(loaded_driver):
    # Verified directly against export-performance-indicators-2009-2018.pdf,
    # table 17.82 "APPAREL", United States, 2018 column: 2,269.47 (USD Mn).
    rows = await _run(
        loaded_driver,
        """
        MATCH (:Country {iso3: 'LKA'})-[:REPORTED]->(r:ExportRecord {source_id: 'EDB'})
              -[:TO]->(:Country {iso3: 'USA'}),
              (r)-[:OF]->(:Product {key: 'apparel:apparel'})
        WHERE r.period_start = date('2018-01-01')
        RETURN r.export_value_usd AS value
        """,
    )
    assert rows == [{"value": pytest.approx(2269.47 * 1_000_000.0)}]


async def test_no_export_record_has_a_null_partner(loaded_driver):
    # partner_iso3 = NULL means "World" in schema.sql — the loader must map
    # that to the WLD sentinel, never leave :TO pointing at nothing.
    rows = await _run(
        loaded_driver,
        "MATCH (r:ExportRecord) WHERE NOT (r)-[:TO]->(:Country) RETURN count(r) AS n",
    )
    assert rows == [{"n": 0}]
