"""Implements SRS 3.1.9 and SAD §8 Knowledge Layer — the only way into Neo4j.

Two rules, both from the layer rules in the root CLAUDE.md:

1. **Parameterized Cypher only.** Never an f-string with a value in it. Beyond
   injection, string-built queries defeat Neo4j's plan cache, so every query
   re-plans and the 10-second single-sector budget (SRS 3.4.1) goes on
   compilation.

2. **`run()` returns the rows *and the Cypher text*.** Agents must put the query
   that produced a figure into `Evidence.detail` (SRS 3.1.4), and the only way to
   guarantee that is to hand it back with the result rather than trusting each
   agent to remember what it asked.

Degraded behaviour is a requirement, not a nicety: if Neo4j is unreachable the
client raises `KnowledgeGraphUnavailableError`, agents catch it, call `failed_output`
and the orchestrator answers with whatever else succeeded (SAD §4.1).
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any, Self

from neo4j import AsyncDriver, AsyncGraphDatabase, NotificationDisabledClassification
from neo4j import exceptions as neo4j_exceptions

from ceynex.settings import neo4j_config

log = logging.getLogger(__name__)

# One retry, then degrade. Retrying harder eats the response-time budget that
# SRS 3.4.1 spends on the LLM call.
MAX_ATTEMPTS = 2
DEFAULT_TIMEOUT_S = 5.0


class KnowledgeGraphUnavailableError(RuntimeError):
    """Neo4j could not be reached or the query failed.

    Agents catch this and degrade. They never let it propagate — an agent that
    raises breaks the orchestrator's partial-result guarantee.
    """


class KnowledgeGraphClient:
    """Async, pooled Neo4j client. One instance per process, shared by all agents.

    The driver holds its own connection pool, so constructing a client per query
    would open a new pool per query. Build it once at application start and pass
    it into the agent nodes.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        *,
        database: str | None = None,
        max_pool_size: int = 20,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        default_uri, default_user, default_password = neo4j_config()
        self._uri = uri or default_uri
        self._user = user or default_user
        self._password = password or default_password
        self._database = database
        self._timeout_s = timeout_s
        self._driver: AsyncDriver | None = None
        self._max_pool_size = max_pool_size

    # --- lifecycle -------------------------------------------------------

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                max_connection_pool_size=self._max_pool_size,
                connection_acquisition_timeout=self._timeout_s,
                # UNRECOGNIZED fires for every optional property that is null on
                # some rows — `cov.to_year` on an agreement still in force, for
                # one — and logs the whole query text each time. Real problems
                # (deprecation, performance, security) still come through.
                notifications_disabled_classifications=[
                    NotificationDisabledClassification.UNRECOGNIZED,
                ],
            )
        return self._driver

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def verify_connectivity(self) -> bool:
        """Cheap liveness probe for /health and the degraded-mode check."""
        try:
            await self.driver.verify_connectivity()
        except Exception as exc:  # noqa: BLE001 - liveness probe reports, never raises
            log.warning("neo4j unreachable at %s: %s", self._uri, exc)
            return False
        return True

    # --- querying --------------------------------------------------------

    async def run(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Execute parameterized Cypher. Returns `(rows, cypher_text)`.

        The Cypher comes back so the caller can put it straight into
        `Evidence.detail` — that traceability is the architectural claim behind
        the Export Analytics agent (SRS 3.1.6), and it only holds if the string
        in the evidence is the string that ran.
        """
        params = params or {}
        _warn_if_unparameterized(cypher)

        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                async with self.driver.session(database=self._database) as session:
                    result = await session.run(cypher, params)  # type: ignore[arg-type]
                    rows = [record.data() async for record in result]
                return rows, cypher
            except (
                neo4j_exceptions.ServiceUnavailable,
                neo4j_exceptions.SessionExpired,
                neo4j_exceptions.TransientError,
            ) as exc:
                last_error = exc
                log.warning("neo4j attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc)
            except neo4j_exceptions.Neo4jError as exc:
                # A malformed query is our bug; retrying cannot help.
                raise KnowledgeGraphUnavailableError(f"cypher rejected by neo4j: {exc}") from exc

        raise KnowledgeGraphUnavailableError(
            f"neo4j unreachable at {self._uri} after {MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    async def run_one(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """`run()` for queries expected to return at most one row."""
        rows, text = await self.run(cypher, params)
        return (rows[0] if rows else None), text

    async def write(self, cypher: str, params: dict[str, Any] | None = None) -> int:
        """Run a write statement, returning the number of rows it touched.

        Loaders use this. Every loader statement must be `MERGE`, never `CREATE`:
        three members load into one graph and re-run their loaders freely.
        """
        params = params or {}
        _warn_if_unparameterized(cypher)
        try:
            async with self.driver.session(database=self._database) as session:
                result = await session.run(cypher, params)  # type: ignore[arg-type]
                summary = await result.consume()
        except neo4j_exceptions.Neo4jError as exc:
            raise KnowledgeGraphUnavailableError(f"write failed: {exc}") from exc

        counters = summary.counters
        return counters.nodes_created + counters.relationships_created + counters.properties_set

    async def execute_script(self, script: str) -> int:
        """Apply a multi-statement `.cypher` file, one statement at a time.

        Neo4j accepts a single statement per `run()`, so schema.cypher has to be
        split. Comments are stripped first; blank statements are skipped.
        """
        applied = 0
        for statement in split_statements(script):
            await self.write(statement)
            applied += 1
        return applied


def split_statements(script: str) -> list[str]:
    """Split a .cypher file into executable statements, dropping `//` comments."""
    without_comments = "\n".join(
        line for line in script.splitlines() if not line.strip().startswith("//")
    )
    return [chunk.strip() for chunk in without_comments.split(";") if chunk.strip()]


def _warn_if_unparameterized(cypher: str) -> None:
    """Log Cypher that carries a filter but no parameters — the f-string smell.

    A literal `{` in Cypher is map syntax — `MERGE (c:Country {iso3: $iso3})` — so
    we cannot ban braces. What we can catch is an f-string that was already
    evaluated: those leave no `$param` markers behind while carrying quoted
    literals that should have been parameters.
    """
    if "$" in cypher:
        return
    lowered = cypher.lower()
    if any(keyword in lowered for keyword in (" where ", " set ", "merge (", "create (")):
        log.debug("cypher with no parameters: %s", cypher.strip()[:120])
