"""Implements SRS 3.1.9 — agriculture nodes and relationships in Neo4j.

The loader owns agriculture's ``Commodity``, ``District``, ``CLASSIFIED_AS``,
and supported ``COVERED_BY`` relationships.  It is deliberately separate from
the generic trade-flow loader: Tea Board and DEA/EAC totals have no partner, so
they cannot honestly create a destination-country ``EXPORTS_TO`` relationship.
M2's partner-level Comtrade loader remains the source of those edges.

All statements use ``MERGE``.  District membership is seeded from the approved
project plan, but no production or export share is available in a local raw
source, so this loader explicitly stores a data-gap note rather than inventing
a number.
"""

from __future__ import annotations

import logging
from typing import Any

from ceynex.data.crosswalk import hs_description
from ceynex.kg.client import KnowledgeGraphClient
from ceynex.kg.loaders.trade_agreements import coverage_rows

log = logging.getLogger(__name__)

COMMODITIES: tuple[dict[str, object], ...] = (
    {"name": "tea", "hs_codes": ("0902",)},
    {"name": "cinnamon", "hs_codes": ("0906",)},
    {"name": "rubber", "hs_codes": ("4001",)},
    {"name": "coconut", "hs_codes": ("0801", "1513")},
)

# The plan identifies these producing districts, but neither the Tea Board nor
# the cinnamon fallback workbook contains district-level production.  ``share``
# is intentionally absent: null/guessed shares would make the concentration
# query look quantitative when the underlying source is not.
DISTRICTS: tuple[dict[str, str], ...] = (
    {"commodity": "tea", "district": "Nuwara Eliya"},
    {"commodity": "tea", "district": "Badulla"},
    {"commodity": "tea", "district": "Kandy"},
    {"commodity": "cinnamon", "district": "Matara"},
    {"commodity": "cinnamon", "district": "Galle"},
    {"commodity": "cinnamon", "district": "Ratnapura"},
)

DISTRICT_SOURCE = "CeyNex M1 project plan; district membership only"
DISTRICT_SHARE_NOTE = "No sourced district share is available; no share is claimed."

MERGE_COMMODITIES = """
UNWIND $rows AS row
MERGE (c:Commodity {name: row.name})
  SET c.hs_codes = row.hs_codes
WITH c, row
UNWIND row.hs_codes AS hs_code
MERGE (h:HSCode {code: hs_code})
  SET h.description = row.descriptions[hs_code]
MERGE (c)-[:CLASSIFIED_AS]->(h)
"""

MERGE_DISTRICTS = """
UNWIND $rows AS row
MERGE (c:Commodity {name: row.commodity})
MERGE (d:District {name: row.district})
MERGE (c)-[p:PRODUCED_IN]->(d)
  SET p.source = $source,
      p.share_note = $share_note
"""

MERGE_SUPPORTED_COVERAGE = """
UNWIND $rows AS row
MATCH (h:HSCode {code: row.hs_code})
MATCH (t:TradeAgreement {name: row.agreement})
MERGE (h)-[cov:COVERED_BY]->(t)
  SET cov.from_year = row.from_year,
      cov.to_year = row.to_year,
      cov.note = row.note,
      cov.verified = row.verified,
      cov.inherited_from_hs = row.inherited_from_hs
"""


def commodity_rows() -> list[dict[str, object]]:
    """Rows for commodity/HS node creation, using the committed HS crosswalk."""
    rows: list[dict[str, object]] = []
    for commodity in COMMODITIES:
        codes = tuple(str(code) for code in commodity["hs_codes"])
        rows.append(
            {
                "name": commodity["name"],
                "hs_codes": list(codes),
                "descriptions": {code: hs_description(code) for code in codes},
            }
        )
    return rows


def supported_coverage_rows() -> list[dict[str, Any]]:
    """Expand the maintained agreement coverage table to agriculture HS codes.

    Coverage is often declared at chapter level (for example ``09``).  The
    exact commodity codes inherit it so sector queries need not recreate that
    hierarchy.  ``verified`` is copied unchanged; current source rows are
    marked ``unverified`` and must not be presented as official confirmation.
    """
    rows: list[dict[str, Any]] = []
    codes = [code for commodity in COMMODITIES for code in commodity["hs_codes"]]
    for coverage in coverage_rows():
        prefix = str(coverage["hs_code"])
        for code in codes:
            if str(code).startswith(prefix):
                rows.append(
                    {
                        "hs_code": str(code),
                        "agreement": coverage["agreement"],
                        "from_year": coverage["from_year"],
                        "to_year": coverage["to_year"],
                        "note": coverage["note"],
                        "verified": coverage["verified"],
                        "inherited_from_hs": prefix,
                    }
                )
    return rows


async def load(kg: KnowledgeGraphClient) -> dict[str, int]:
    """Merge agriculture graph nodes and supported agreement coverage safely."""
    commodities = commodity_rows()
    coverage = supported_coverage_rows()

    await kg.write(MERGE_COMMODITIES, {"rows": commodities})
    await kg.write(
        MERGE_DISTRICTS,
        {"rows": list(DISTRICTS), "source": DISTRICT_SOURCE, "share_note": DISTRICT_SHARE_NOTE},
    )
    # Agreement nodes are owned by M2's loader.  MATCH makes this a no-op if
    # it has not been run yet, rather than creating incomplete agreement nodes.
    await kg.write(MERGE_SUPPORTED_COVERAGE, {"rows": coverage})

    log.info(
        "merged %d commodities, %d district relationships, and %d supported coverage relationships",
        len(commodities),
        len(DISTRICTS),
        len(coverage),
    )
    return {
        "commodities": len(commodities),
        "classifications": sum(len(row["hs_codes"]) for row in COMMODITIES),
        "districts": len(DISTRICTS),
        "coverage_edges": len(coverage),
    }
