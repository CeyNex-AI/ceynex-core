"""Implements SRS 3.1.9's apparel half — projects EDB/JAAF into EXPORTS_TO edges.

Ownership boundary: see kg/loaders/trade_flows.py's docstring — M2 projects
UN_COMTRADE, M1 projects agriculture, this is M3's loader for JAAF/EDB, the
long-tracked SRS 3.3.4 gap (no apparel data ever reached the graph after the
old `(Country)-[:REPORTED]->(ExportRecord)` schema was retired).

Reuses trade_flows.py's country-merge and EXPORTS_TO-merge Cypher directly
(same shared graph, same edge shape) rather than duplicating it. Two things
this loader does NOT reuse trade_flows.read_flows()/MERGE_ITEMS for:

1. **Frequency.** trade_flows.SELECT_FLOWS hardcodes `frequency = 'A'`
   (correct for EDB, wrong for JAAF — JAAF's fact_trade rows are monthly,
   `frequency = 'M'`; a bare reuse would silently drop JAAF entirely, zero
   rows, no error). JAAF is aggregated to annual here before merging.
2. **hs_code.** EDB/JAAF's PDF/HTML sources give no HS code (unlike
   Comtrade's HS-driven classification) — fact_trade.hs_code is NULL for
   both, confirmed against the live data. trade_flows.MERGE_ITEMS requires
   one (apoc.merge.node's onCreate properties include it via
   CLASSIFIED_AS/HSCode); merging a NULL-keyed HSCode node on every load is
   not something to paper over with a fabricated code. This loader creates
   the ApparelCategory node directly, with no HSCode/CLASSIFIED_AS edge — an
   honest gap, not an invented one, and it only costs `competing_exporters`
   (kg/queries.py), which the apparel agent never calls.

EDB's raw product text also drifts across PDF editions/layouts (see
data/raw/edb/PROFILE.md's own flagged follow-up) — the narrow "Apparel"
sub-category specifically appears as 'APPAREL' (2009-2018 archive layout) and
'APPREL' (a typo in the annual-layout PDFs, both 2023/2024 editions, already
deduped by EDBConnector.to_fact_trade). Confirmed against live data that the
two spellings cover disjoint year ranges (2014-2018 vs 2019-2024) — not a
double-count risk — so both normalize to one item key, `apparel_edb`,
distinct from JAAF's own `apparel_textiles` and Comtrade's
`apparel_knit`/`apparel_woven` (kg/loaders/trade_flows.ITEM_NODES), so
trade_flows.MERGE_FLOWS's (item-node, country, year) merge key never
collides across sources — the concern the 2026-08-19 PR #1 review thread
raised and left unresolved. EDB's broader "Apparel & Textiles ... Total"
aggregate rows (both spelling variants seen live) are deliberately excluded:
they already sum every EDB sub-category table, so loading them under
`apparel_edb` too would double-count against itself.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg

from ceynex.kg.client import KnowledgeGraphClient
from ceynex.kg.loaders.trade_flows import MERGE_COUNTRIES, MERGE_FLOWS, _country_rows
from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

# fact_trade.item (EDB) -> graph item name.
#
# `EDBConnector.to_fact_trade` now writes the canonical key via
# `crosswalk.canonical_item`, so new rows already arrive as `apparel_edb`. Rows
# ingested before that still carry the raw spellings, and both must resolve
# while the backfill migration is outstanding — hence the identity entry as
# well as the two historical ones. The raw keys can be dropped once no
# `source_id='EDB'` row in fact_trade has an uppercase `item`.
_EDB_GRAPH_ITEM = "apparel_edb"
_EDB_ITEM_MAP = {
    "APPAREL": _EDB_GRAPH_ITEM,
    "APPREL": _EDB_GRAPH_ITEM,
    _EDB_GRAPH_ITEM: _EDB_GRAPH_ITEM,
}
_JAAF_ITEM = "apparel_textiles"

_SELECT_EDB = """
    SELECT item, partner_iso3, partner_m49,
           EXTRACT(YEAR FROM period_start)::int AS year,
           sum(export_value_usd) AS value,
           sum(export_volume)    AS volume,
           max(volume_unit)      AS unit
      FROM fact_trade
     WHERE source_id = 'EDB'
       AND item = ANY(%s)
       AND partner_iso3 IS NOT NULL
       AND frequency = 'A'
     GROUP BY item, partner_iso3, partner_m49, year
"""

# JAAF's fact_trade rows are monthly (frequency='M'); summed to annual here to
# match trade_flows.py's (item, partner, year) grain — the old, pre-schema-
# retirement agent code did this same year-truncation in Cypher instead.
_SELECT_JAAF = """
    SELECT partner_iso3, partner_m49,
           EXTRACT(YEAR FROM period_start)::int AS year,
           sum(export_value_usd) AS value,
           sum(export_volume)    AS volume,
           max(volume_unit)      AS unit
      FROM fact_trade
     WHERE source_id = 'JAAF'
       AND partner_iso3 IS NOT NULL
       AND frequency = 'M'
     GROUP BY partner_iso3, partner_m49, year
"""

# No HSCode/CLASSIFIED_AS step, deliberately — see module docstring point 2.
MERGE_ITEMS = """
UNWIND $rows AS row
MERGE (i:ApparelCategory {name: row.name})
"""


def read_flows(dsn: str | None = None) -> list[dict[str, Any]]:
    """EDB (annual) + JAAF (monthly, summed to annual) apparel flows, item-normalized."""
    with psycopg.connect(dsn or postgres_dsn()) as conn, conn.cursor() as cur:
        cur.execute(_SELECT_EDB, (list(_EDB_ITEM_MAP),))
        columns = [d[0] for d in cur.description or []]
        edb = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
        for row in edb:
            row["item"] = _EDB_ITEM_MAP[row["item"]]

        cur.execute(_SELECT_JAAF)
        columns = [d[0] for d in cur.description or []]
        jaaf = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
        for row in jaaf:
            row["item"] = _JAAF_ITEM

    return edb + jaaf


def _flow_rows(flows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "item": flow["item"],
            "partner_iso3": flow["partner_iso3"],
            "year": int(flow["year"]),
            "value": float(flow["value"]) if flow["value"] is not None else None,
            "volume": float(flow["volume"]) if flow["volume"] is not None else None,
            "unit": flow["unit"],
            "source": "EDB" if flow["item"] == "apparel_edb" else "JAAF",
        }
        for flow in flows
    ]


async def load(kg: KnowledgeGraphClient, dsn: str | None = None) -> dict[str, int]:
    """Project EDB/JAAF apparel data into the graph. Idempotent."""
    flows = read_flows(dsn)
    if not flows:
        log.warning("no EDB/JAAF apparel rows in fact_trade — run `make ingest` first")
        return {"countries": 0, "items": 0, "flows": 0}

    countries = _country_rows(flows)
    items = [{"name": name} for name in sorted({flow["item"] for flow in flows})]
    edges = _flow_rows(flows)

    await kg.write(MERGE_COUNTRIES, {"rows": countries})
    await kg.write(MERGE_ITEMS, {"rows": items})
    # Batched, matching trade_flows.load()'s reasoning: one huge UNWIND
    # transaction against Neo4j's default 1G dev-stack heap is asking for it.
    for start in range(0, len(edges), 1000):
        await kg.write(MERGE_FLOWS, {"rows": edges[start : start + 1000]})

    log.info(
        "merged %d countries, %d items, %d export flows (EDB+JAAF)",
        len(countries),
        len(items),
        len(edges),
    )
    return {"countries": len(countries), "items": len(items), "flows": len(edges)}
