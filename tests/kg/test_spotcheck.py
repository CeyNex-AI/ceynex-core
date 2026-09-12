"""M3's spot-check suite (SRS 3.3.4) — real EDB/JAAF figures survive the full path.

`ceynex/kg/queries.py` and `load.py` both call this suite out by name as the
thing that exercises the shared Cypher library against a real graph, not a
fake. The original (built 2026-08-14) was deleted resolving PR #1's merge
conflicts because it queried the retired `(Country)-[:REPORTED]->(ExportRecord)`
schema and was never rebuilt against the frozen
`(:Commodity|:ApparelCategory)-[:EXPORTS_TO]->(:Country)` shape — this is that
rebuild.

**What "spot check" means here**: every expected number below was read by hand
from the checked-in raw source (an EDB PDF table or a JAAF HTML page), not
copied from `fact_trade` or the graph. A bug anywhere in the path — the PDF/
HTML parser, the crosswalk, `kg/loaders/apparel.py`'s frequency handling, or
the Cypher in `kg/queries.py` — has an equal chance of producing a wrong
number, so checking the *query result* against the *source document* is what
actually verifies the path end to end, unlike a unit test that checks the
loader's SQL/Cypher shape against a synthetic fixture (`tests/kg/test_client.py`
and `tests/data/test_edb_connector.py` already do that half).

**These figures are pinned to specific snapshots checked into `data/raw/`**,
not to "whatever EDB/JAAF says today":

- EDB: `data/raw/edb/manual/export-performance-indicators-of-sri-lanka-2024.pdf`,
  Table 25.79 "APPREL", p.236 — United States and United Kingdom rows.
  `layout="annual"`'s highest-`edition_year` dedup rule (`edb.py`'s
  `to_fact_trade` docstring) means the 2024 edition's figures win for every
  year it covers (2020-2024), including 2023, over the 2023 edition's own
  2023 figure.
- JAAF: `data/raw/jaaf/2026-08-21/annual_exports.html`'s US and UK tables,
  2025 rows — each independently re-summed from the 12 monthly cells below,
  and cross-checked against the sibling `market_wise.html`'s 2025 pie chart
  (same two figures, to the cent for UK and to within 1 cent for US, which is
  the source site's own rounding, not a parsing bug — see `jaaf.py`'s module
  docstring on how these two pages relate).

If EDB or JAAF publish a new edition and someone re-saves these files, these
numbers stop matching and the tests correctly fail — re-derive them from the
new snapshot the same way, by hand, rather than by re-reading whatever
`fact_trade` says.

Requires `make up && make kg-load` (schema, apparel, flows) first — see
`tests/kg/test_client.py`'s module docstring for the same precondition.
"""

import pytest

from ceynex.kg.client import KnowledgeGraphClient
from ceynex.kg.queries import cagr, district_concentration, market_share, top_partners

pytestmark = pytest.mark.integration


# EDB, Table 25.79 "APPREL", 2024 edition (p.236). USD Millions -> USD.
_EDB_USA_2023 = 1_782_720_000.0  # column "2023": 1,782.72
_EDB_USA_2020 = 1_649_270_000.0  # column "2020": 1,649.27
_EDB_USA_2024 = 1_875_850_000.0  # column "2024": 1,875.85
_EDB_GBR_2023 = 614_620_000.0  # column "2023": 614.62

# JAAF, annual_exports.html, US table, 2025 row: Jan..Dec re-summed by hand:
#   169.75 + 153.11 + 171.79 + 121.89 + 129.99 + 164.39 + 178.26 + 211.32
#   + 160.52 + 155.74 + 152.32 + 178.29 = 1947.37 (matches the page's own
#   "Total" cell and the market_wise.html pie chart's 1947.38, 1 cent off on
#   JAAF's side).
_JAAF_USA_2025 = 1_947_370_000.0
# UK table, 2025 row: 61.57+54.85+77.13+53.94+51.71+67.33+56.12+60.77+50.31
#   +47.18+43.63+55.12 = 679.66 (matches the page's own total and the
#   market_wise.html pie chart exactly).
_JAAF_GBR_2025 = 679_660_000.0


async def test_edb_apparel_reaches_the_graph_for_the_united_states():
    """kg/loaders/apparel.py's EDB half, end to end: PDF figure -> fact_trade -> EXPORTS_TO."""
    async with KnowledgeGraphClient() as kg:
        rows, _ = await kg.run(*top_partners("apparel_edb", 2023))

    usa = next((r for r in rows if r["partner_iso3"] == "USA"), None)
    assert usa is not None, "no apparel_edb->USA edge for 2023 — run `make kg-load`"
    assert usa["export_value_usd"] == pytest.approx(_EDB_USA_2023, rel=1e-6)


async def test_edb_apparel_reaches_the_graph_for_the_united_kingdom():
    async with KnowledgeGraphClient() as kg:
        rows, _ = await kg.run(*top_partners("apparel_edb", 2023))

    gbr = next((r for r in rows if r["partner_iso3"] == "GBR"), None)
    assert gbr is not None, "no apparel_edb->GBR edge for 2023 — run `make kg-load`"
    assert gbr["export_value_usd"] == pytest.approx(_EDB_GBR_2023, rel=1e-6)


async def test_jaaf_apparel_reaches_the_graph_for_the_united_states():
    """kg/loaders/apparel.py's JAAF half: monthly rows summed to annual, end to end."""
    async with KnowledgeGraphClient() as kg:
        rows, _ = await kg.run(*top_partners("apparel_textiles", 2025))

    usa = next((r for r in rows if r["partner_iso3"] == "USA"), None)
    assert usa is not None, "no apparel_textiles->USA edge for 2025 — run `make kg-load`"
    assert usa["export_value_usd"] == pytest.approx(_JAAF_USA_2025, rel=1e-6)


async def test_jaaf_apparel_reaches_the_graph_for_the_united_kingdom():
    async with KnowledgeGraphClient() as kg:
        rows, _ = await kg.run(*top_partners("apparel_textiles", 2025))

    gbr = next((r for r in rows if r["partner_iso3"] == "GBR"), None)
    assert gbr is not None, "no apparel_textiles->GBR edge for 2025 — run `make kg-load`"
    assert gbr["export_value_usd"] == pytest.approx(_JAAF_GBR_2025, rel=1e-6)


async def test_market_share_of_verified_apparel_partners_sums_to_one():
    """market_share's denominator is the filtered set itself (kg/queries.py's own
    docstring warning about a stale denominator) — only US and UK are JAAF-projected
    for apparel_textiles, so their two shares alone must sum to 1, and each share
    must match the verified figures' own ratio, not just add up by construction.
    """
    async with KnowledgeGraphClient() as kg:
        rows, _ = await kg.run(*market_share("apparel_textiles", 2025))

    shares = {r["partner_iso3"]: r["share"] for r in rows}
    assert shares.keys() == {"USA", "GBR"}
    assert sum(shares.values()) == pytest.approx(1.0, rel=1e-9)

    expected_total = _JAAF_USA_2025 + _JAAF_GBR_2025
    assert shares["USA"] == pytest.approx(_JAAF_USA_2025 / expected_total, rel=1e-6)
    assert shares["GBR"] == pytest.approx(_JAAF_GBR_2025 / expected_total, rel=1e-6)


async def test_cagr_endpoints_match_the_verified_edb_figures():
    """cagr() returns the raw endpoint values for the caller to compute the rate
    from (kg/queries.py's own docstring: CAGR itself is computed by the caller,
    not in Cypher) — checked against the same PDF table as the point-in-time
    checks above, five columns apart (2020 and 2024).
    """
    async with KnowledgeGraphClient() as kg:
        rows, _ = await kg.run(*cagr("apparel_edb", "USA", 2020, 2024))

    by_year = {r["year"]: r["export_value_usd"] for r in rows}
    assert by_year[2020] == pytest.approx(_EDB_USA_2020, rel=1e-6)
    assert by_year[2024] == pytest.approx(_EDB_USA_2024, rel=1e-6)


async def test_apparel_has_no_district_concentration():
    """Apparel is not modelled by district (kg/queries.py's own docstring on
    district_concentration, and kg/loaders/apparel.py never writes PRODUCED_IN) —
    an apparel item must return no rows here rather than erroring or aliasing
    onto an agriculture commodity that happens to share graph space.
    """
    async with KnowledgeGraphClient() as kg:
        rows, _ = await kg.run(*district_concentration("apparel_edb"))

    assert rows == []
