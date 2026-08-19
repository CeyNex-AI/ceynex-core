"""Implements SRS 3.1.9 — projects fact_trade rows into EXPORTS_TO edges.

**Scope boundary.** This loader projects only the sources M2 ingested
(`UN_COMTRADE` by default). M1's `kg/loaders/agriculture.py` projects his
FAOSTAT/Tea Board/cinnamon sources and M3's `kg/loaders/apparel.py` projects his
JAAF/EDB sources. Whoever ingests a source loads it — otherwise two people write
the same edge with two different value conventions and the graph quietly
disagrees with itself.

Everything is `MERGE`, so this runs alongside both teammates' loaders against one
shared Neo4j (deviation D2: Community edition serves a single database, so there
is no per-member namespace).

Graph shape is the frozen one from team overview §4.3 / SAD §9:

    (:Commodity|:ApparelCategory)-[:EXPORTS_TO {volume, value, year, unit}]->(:Country)
    (:Commodity|:ApparelCategory)-[:CLASSIFIED_AS]->(:HSCode)
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg

from ceynex.data.crosswalk import hs_description
from ceynex.kg.client import KnowledgeGraphClient
from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

# item -> (node label, display name). Apparel categories are chapter-level, which
# is the granularity JAAF and the SRS both talk about (HS 61 knit, HS 62 woven).
ITEM_NODES: dict[str, tuple[str, str]] = {
    "tea": ("Commodity", "tea"),
    "cinnamon": ("Commodity", "cinnamon"),
    "rubber": ("Commodity", "rubber"),
    "coconut": ("Commodity", "coconut"),
    "apparel_knit": ("ApparelCategory", "apparel_knit"),
    "apparel_woven": ("ApparelCategory", "apparel_woven"),
}

SELECT_FLOWS = """
    SELECT item,
           hs_code,
           partner_iso3,
           partner_m49,
           EXTRACT(YEAR FROM period_start)::int AS year,
           sum(export_value_usd) AS value,
           sum(export_volume)    AS volume,
           max(volume_unit)      AS unit
      FROM fact_trade
     WHERE source_id = ANY(%s)
       AND partner_iso3 IS NOT NULL
       AND frequency = 'A'
     GROUP BY item, hs_code, partner_iso3, partner_m49, year
"""

MERGE_COUNTRIES = """
UNWIND $rows AS row
MERGE (c:Country {iso3: row.iso3})
  ON CREATE SET c.m49 = row.m49, c.name = row.name
  ON MATCH  SET c.m49 = coalesce(c.m49, row.m49),
                c.name = coalesce(c.name, row.name)
"""

MERGE_ITEMS = """
UNWIND $rows AS row
CALL apoc.merge.node([row.label], {name: row.name}, {hs_code: row.hs_code}, {}) YIELD node
MERGE (h:HSCode {code: row.hs_code})
  ON CREATE SET h.description = row.description
MERGE (node)-[:CLASSIFIED_AS]->(h)
"""

# apoc.merge.node lets the label be a parameter; without APOC the label would have
# to be interpolated into the query string, which the layer rules forbid. This is
# why the deployed Neo4j now enables APOC to match the dev stack.
MERGE_FLOWS = """
UNWIND $rows AS row
MATCH (i {name: row.item})
WHERE i:Commodity OR i:ApparelCategory
MATCH (c:Country {iso3: row.partner_iso3})
MERGE (i)-[e:EXPORTS_TO {year: row.year}]->(c)
  SET e.value  = row.value,
      e.volume = row.volume,
      e.unit   = row.unit,
      e.source = row.source
"""


def read_flows(dsn: str | None = None, sources: tuple[str, ...] = ("UN_COMTRADE",)) -> list[dict[str, Any]]:
    """Annual export flows from fact_trade, aggregated to (item, partner, year)."""
    with psycopg.connect(dsn or postgres_dsn()) as conn, conn.cursor() as cur:
        cur.execute(SELECT_FLOWS, (list(sources),))
        columns = [description[0] for description in cur.description or []]
        return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def _country_rows(flows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from ceynex.data.crosswalk import country_name

    seen: dict[str, dict[str, Any]] = {}
    for flow in flows:
        iso3 = flow["partner_iso3"]
        if iso3 in seen:
            continue
        try:
            name = country_name(iso3)
        except Exception:  # noqa: BLE001 - an unnamed country is still a valid node
            name = iso3
        seen[iso3] = {"iso3": iso3, "m49": flow["partner_m49"], "name": name}
    return list(seen.values())


def _item_rows(flows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for flow in flows:
        item, hs_code = flow["item"], flow["hs_code"]
        if (item, hs_code) in seen or item not in ITEM_NODES:
            continue
        label, name = ITEM_NODES[item]
        try:
            description = hs_description(hs_code)
        except Exception:  # noqa: BLE001
            description = f"HS {hs_code}"
        seen[(item, hs_code)] = {
            "label": label,
            "name": name,
            "hs_code": hs_code,
            "description": description,
        }
    return list(seen.values())


def _flow_rows(flows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [
        {
            "item": ITEM_NODES[flow["item"]][1],
            "partner_iso3": flow["partner_iso3"],
            "year": int(flow["year"]),
            "value": float(flow["value"]) if flow["value"] is not None else None,
            "volume": float(flow["volume"]) if flow["volume"] is not None else None,
            "unit": flow["unit"],
            "source": source,
        }
        for flow in flows
        if flow["item"] in ITEM_NODES
    ]


async def load(
    kg: KnowledgeGraphClient,
    dsn: str | None = None,
    sources: tuple[str, ...] = ("UN_COMTRADE",),
) -> dict[str, int]:
    """Project fact_trade into the graph. Idempotent."""
    flows = read_flows(dsn, sources)
    if not flows:
        log.warning("no annual flows in fact_trade for %s — run `make ingest` first", list(sources))
        return {"countries": 0, "items": 0, "flows": 0}

    countries = _country_rows(flows)
    items = _item_rows(flows)
    edges = _flow_rows(flows, source=",".join(sources))

    await kg.write(MERGE_COUNTRIES, {"rows": countries})
    await kg.write(MERGE_ITEMS, {"rows": items})

    # Batched: a single UNWIND of tens of thousands of rows builds one enormous
    # transaction and Neo4j's default heap is 1G on the dev stack.
    for start in range(0, len(edges), 1000):
        await kg.write(MERGE_FLOWS, {"rows": edges[start : start + 1000]})

    log.info(
        "merged %d countries, %d items, %d export flows",
        len(countries),
        len(items),
        len(edges),
    )
    return {"countries": len(countries), "items": len(items), "flows": len(edges)}
