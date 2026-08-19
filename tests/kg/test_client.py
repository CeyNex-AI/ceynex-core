"""Assertions for the Knowledge Layer client (SAD §8, SRS 3.1.9).

The unit tests need no database. The integration ones need `make up` and are
marked so `make test-unit` skips them.
"""

import pytest

from ceynex.kg.client import (
    KnowledgeGraphClient,
    KnowledgeGraphUnavailableError,
    split_statements,
)
from ceynex.kg.queries import agreement_coverage, graph_summary


def test_split_statements_drops_comments_and_blanks():
    script = """
    // a comment
    CREATE CONSTRAINT a IF NOT EXISTS FOR (c:Country) REQUIRE c.iso3 IS UNIQUE;

    // another
    CREATE INDEX b IF NOT EXISTS FOR (c:Country) ON (c.name);
    """
    statements = split_statements(script)
    assert len(statements) == 2
    assert all(not s.startswith("//") for s in statements)
    assert "CONSTRAINT" in statements[0]


def test_split_statements_handles_a_trailing_semicolon():
    assert split_statements("RETURN 1;") == ["RETURN 1"]
    assert split_statements("RETURN 1") == ["RETURN 1"]
    assert split_statements("   \n // only a comment\n") == []


def test_unavailable_is_a_runtime_error_agents_can_catch():
    """Agents catch this and degrade; they must never let it propagate."""
    assert issubclass(KnowledgeGraphUnavailableError, RuntimeError)


async def test_unreachable_neo4j_raises_rather_than_hanging():
    """A dead graph must fail fast — the 10s budget in SRS 3.4.1 is the whole reason."""
    client = KnowledgeGraphClient(
        uri="bolt://127.0.0.1:9",  # discard port: refuses immediately
        user="neo4j",
        password="nope",
        timeout_s=1.0,
    )
    try:
        assert await client.verify_connectivity() is False
        with pytest.raises(KnowledgeGraphUnavailableError):
            await client.run("RETURN 1 AS x")
    finally:
        await client.close()


@pytest.mark.integration
async def test_run_returns_rows_and_the_cypher_that_produced_them():
    """SRS 3.1.4 — the query text is evidence, so it comes back with the rows."""
    async with KnowledgeGraphClient() as kg:
        rows, cypher = await kg.run("RETURN $n AS answer", {"n": 42})
    assert rows == [{"answer": 42}]
    assert cypher == "RETURN $n AS answer"


@pytest.mark.integration
async def test_gsp_plus_covers_hs_6109():
    """The SRS 3.1.9 worked example, and the plan's Day 4 acceptance criterion.

    Coverage is declared at chapter 61, so this also proves the hierarchy
    expansion works end to end against a real graph.
    """
    async with KnowledgeGraphClient() as kg:
        rows, _ = await kg.run(*agreement_coverage("6109"))

    agreements = {row["agreement"] for row in rows}
    assert "GSP+" in agreements
    gsp = next(row for row in rows if row["agreement"] == "GSP+")
    assert gsp["matched_on"] == "61"
    assert gsp["covered_from"] == 2017


@pytest.mark.integration
async def test_an_out_of_scope_code_has_no_coverage():
    """Negative case. Over-attachment is the likelier bug than under-attachment."""
    async with KnowledgeGraphClient() as kg:
        rows, _ = await kg.run(*agreement_coverage("8703"))  # motor cars
    assert rows == []


@pytest.mark.integration
async def test_the_shared_agreement_nodes_are_loaded():
    async with KnowledgeGraphClient() as kg:
        rows, _ = await kg.run(*graph_summary())
    labels = {row["label"]: row["nodes"] for row in rows}
    assert labels.get("TradeAgreement", 0) >= 1, "run `make kg-load`"
