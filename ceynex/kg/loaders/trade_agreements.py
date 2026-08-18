"""Implements SRS 3.1.9 — TradeAgreement nodes and their COVERED_BY edges.

M2 owns this loader because both sector loaders attach to the nodes it creates:
M1's agriculture loader and M3's apparel loader each add `COVERED_BY` edges for
their own HS codes, and neither should be the one deciding what GSP+ is.

Everything is `MERGE`. Re-running this against a graph two teammates are loading
into adds nothing and removes nothing.

The data comes from `ceynex/data/reference/trade_agreements.csv` and
`trade_agreement_coverage.csv`, both committed and both currently marked
`unverified` — the graph carries that status on the node so that any answer built
on it can say so rather than implying a certainty nobody has checked.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from ceynex.data.crosswalk import hs_description, normalize_hs
from ceynex.kg.client import KnowledgeGraphClient

log = logging.getLogger(__name__)

REFERENCE_DIR = Path(__file__).parent.parent.parent / "data" / "reference"


def _rows(filename: str) -> list[dict[str, str]]:
    """Read a reference CSV, skipping the `#` provenance header block."""
    path = REFERENCE_DIR / filename
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def agreement_rows() -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "type": row["type"],
            "in_force_from": row["in_force_from"],
            "partners": [p for p in row["partners"].split(";") if p],
            "source": row["source"],
            "verified": row["verified"],
        }
        for row in _rows("trade_agreements.csv")
    ]


def coverage_rows() -> list[dict[str, Any]]:
    rows = []
    for row in _rows("trade_agreement_coverage.csv"):
        code = normalize_hs(row["hs_code"], digits=len(row["hs_code"].strip()))
        rows.append(
            {
                "hs_code": code,
                "agreement": row["agreement"],
                "from_year": int(row["from_year"]) if row["from_year"] else None,
                "to_year": int(row["to_year"]) if row["to_year"] else None,
                "note": row["note"],
                "verified": row["verified"],
                "description": _describe(code),
            }
        )
    return rows


def _describe(code: str) -> str:
    try:
        return hs_description(code)
    except Exception:  # noqa: BLE001 - a coverage row may name a chapter we do not stock
        return f"HS {code}"


MERGE_AGREEMENTS = """
UNWIND $rows AS row
MERGE (t:TradeAgreement {name: row.name})
  SET t.type          = row.type,
      t.in_force_from = row.in_force_from,
      t.partners      = row.partners,
      t.source        = row.source,
      t.verified      = row.verified
RETURN count(t) AS merged
"""

MERGE_COVERAGE = """
UNWIND $rows AS row
MERGE (h:HSCode {code: row.hs_code})
  ON CREATE SET h.description = row.description
MERGE (t:TradeAgreement {name: row.agreement})
MERGE (h)-[cov:COVERED_BY]->(t)
  SET cov.from_year = row.from_year,
      cov.to_year   = row.to_year,
      cov.note      = row.note,
      cov.verified  = row.verified
RETURN count(cov) AS merged
"""


async def load(kg: KnowledgeGraphClient) -> dict[str, int]:
    """Merge agreements and their coverage edges. Idempotent."""
    agreements = agreement_rows()
    coverage = coverage_rows()

    await kg.write(MERGE_AGREEMENTS, {"rows": agreements})
    await kg.write(MERGE_COVERAGE, {"rows": coverage})

    log.info(
        "merged %d trade agreements and %d coverage edges",
        len(agreements),
        len(coverage),
    )
    return {"agreements": len(agreements), "coverage_edges": len(coverage)}
