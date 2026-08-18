"""Implements SRS 3.1.9 — applies schema.cypher and loads the shared graph nodes.

    make kg-load
    python -m ceynex.kg.load --schema --agreements
    python -m ceynex.kg.load --verify        # report what is in the graph

Idempotent by construction: every schema statement is `IF NOT EXISTS` and every
loader statement is `MERGE`. Running this against a graph M1 and M3 are already
loading into adds their missing pieces and touches nothing else, which is what
makes a shared Neo4j workable at all (see deviation D2 — Community edition
serves one database, so there is no per-member namespace to hide behind).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from importlib.resources import files

from ceynex.kg.client import KnowledgeGraphClient, KnowledgeGraphUnavailableError
from ceynex.kg.loaders import trade_agreements
from ceynex.kg.queries import graph_summary
from ceynex.settings import neo4j_config

log = logging.getLogger(__name__)


def schema_cypher() -> str:
    """The frozen constraints and indexes, from the installed ceynex-contracts."""
    return (files("ceynex.contracts") / "schema" / "schema.cypher").read_text(encoding="utf-8")


async def apply_schema(kg: KnowledgeGraphClient) -> int:
    applied = await kg.execute_script(schema_cypher())
    log.info("applied %d constraints and indexes", applied)
    return applied


async def summarize(kg: KnowledgeGraphClient) -> list[dict[str, object]]:
    cypher, params = graph_summary()
    rows, _ = await kg.run(cypher, params)
    return rows


async def run(*, schema: bool, agreements: bool, verify: bool) -> int:
    async with KnowledgeGraphClient() as kg:
        if not await kg.verify_connectivity():
            uri = neo4j_config()[0]
            log.error("neo4j unreachable at %s — is the stack up? `make up`", uri)
            return 1

        if schema:
            await apply_schema(kg)
        if agreements:
            counts = await trade_agreements.load(kg)
            print(
                f"  trade agreements  {counts['agreements']:>4}"
                f"\n  coverage edges    {counts['coverage_edges']:>4}"
            )
        if verify or schema or agreements:
            rows = await summarize(kg)
            if not rows:
                print("  graph is empty")
            for row in rows:
                print(f"  {str(row['label'] or '(no label)'):<18} {row['nodes']:>6} nodes")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the CeyNex Neo4j schema and shared nodes.")
    parser.add_argument("--schema", action="store_true", help="apply constraints and indexes")
    parser.add_argument("--agreements", action="store_true", help="merge TradeAgreement nodes and coverage")
    parser.add_argument("--verify", action="store_true", help="report node counts and exit")
    args = parser.parse_args(argv)

    # Bare `python -m ceynex.kg.load` should do the useful thing, not nothing.
    if not (args.schema or args.agreements or args.verify):
        args.schema = args.agreements = True

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        return asyncio.run(run(schema=args.schema, agreements=args.agreements, verify=args.verify))
    except KnowledgeGraphUnavailableError as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
