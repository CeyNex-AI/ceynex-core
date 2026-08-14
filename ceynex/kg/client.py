"""Neo4j-backed `KnowledgeGraphClientProtocol` implementation (SAD Figure 3).

Query-side counterpart to `ceynex/kg/load.py` (write-side). Nothing else in
`ceynex/kg/` implements the protocol yet, so agent nodes need this to query
the graph at all — see `ceynex/contracts/protocols.py` for the frozen
interface this structurally satisfies (a `Protocol`, so no explicit subclass
is required, only the matching `async def run(...)` signature).
"""

import os

from neo4j import AsyncDriver, AsyncGraphDatabase


def _default_uri() -> str:
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687")


def _default_auth() -> tuple[str, str]:
    return (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "ceynex_dev_pw"),
    )


class Neo4jClient:
    """Thin async Cypher-execution wrapper. Owns its driver unless one is injected."""

    def __init__(self, driver: AsyncDriver | None = None):
        self._owns_driver = driver is None
        self._driver = driver or AsyncGraphDatabase.driver(_default_uri(), auth=_default_auth())

    async def run(self, cypher: str, params: dict | None = None) -> tuple[list[dict], str]:
        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            rows = [record.data() async for record in result]
        return rows, cypher

    async def close(self) -> None:
        if self._owns_driver:
            await self._driver.close()
