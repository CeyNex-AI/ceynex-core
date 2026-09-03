"""Assertions for the named Cypher library (SRS 3.1.6, 3.1.9).

These run without a database. The queries are data — `(cypher, params)` — which
is the point: M3's spot-check suite and the agents both consume the same
definitions, and a query that stops being parameterized should fail here rather
than in a security review.
"""

import re

import pytest

from ceynex.data.crosswalk import CrosswalkError
from ceynex.kg import queries as q

ALL_QUERIES = [
    ("top_partners", lambda: q.top_partners("tea", 2024)),
    ("market_share", lambda: q.market_share("tea", 2024)),
    ("cagr_all_partners", lambda: q.cagr("cinnamon", None, 2015, 2024)),
    ("cagr_one_partner", lambda: q.cagr("cinnamon", "DEU", 2015, 2024)),
    ("district_concentration", lambda: q.district_concentration("cinnamon")),
    ("agreement_coverage", lambda: q.agreement_coverage("6109")),
    ("competing_exporters", lambda: q.competing_exporters("tea", 2024)),
    ("items_in_sector", lambda: q.items_in_sector("agriculture")),
    ("graph_summary", lambda: q.graph_summary()),
    ("latest_observation_year", lambda: q.latest_observation_year()),
    ("latest_observation_year_scoped", lambda: q.latest_observation_year("tea")),
    ("export_subgraph", lambda: q.export_subgraph("tea", 2024)),
    ("classification_subgraph", lambda: q.classification_subgraph("tea")),
    ("production_subgraph", lambda: q.production_subgraph("cinnamon")),
    ("neighbours", lambda: q.neighbours("Country", "USA")),
]

#: The subset that must project a drawable triple. Kept separate from
#: ALL_QUERIES because the scalar queries above deliberately do not.
SUBGRAPH_QUERIES = [
    ("export_subgraph", lambda: q.export_subgraph("tea", 2024)),
    ("classification_subgraph", lambda: q.classification_subgraph("tea")),
    ("production_subgraph", lambda: q.production_subgraph("cinnamon")),
    ("neighbours", lambda: q.neighbours("Country", "USA")),
]


def test_latest_observation_year_scopes_to_one_item_when_given():
    """Regression: unscoped latest_observation_year() found live 2026-08-26 --
    a single item's later (and erroneous) latest year silently made every
    other item's "what year should I query" lookup return a year with no
    data for that other item. Callers that then query one specific item
    (export_analytics, trade_economics) must scope this to that item.
    """
    cypher, params = q.latest_observation_year("tea")
    assert "toLower(i.name) = toLower($item)" in cypher
    assert params == {"item": "tea"}

    cypher, params = q.latest_observation_year()
    assert "$item" not in cypher
    assert params == {}


@pytest.mark.parametrize(("name", "build"), ALL_QUERIES, ids=[n for n, _ in ALL_QUERIES])
def test_every_query_returns_cypher_and_params(name, build):
    cypher, params = build()
    assert isinstance(cypher, str) and cypher.strip()
    assert isinstance(params, dict)


@pytest.mark.parametrize(("name", "build"), ALL_QUERIES, ids=[n for n, _ in ALL_QUERIES])
def test_no_query_interpolates_a_value(name, build):
    """Every filtered value is a $param. This is the layer rule from CLAUDE.md."""
    cypher, params = build()
    for key in params:
        assert f"${key}" in cypher, f"{name}: param {key!r} is passed but never referenced"

    # Any $name in the cypher must be supplied, or the query fails at runtime
    # with a message that does not say which one.
    referenced = set(re.findall(r"\$(\w+)", cypher))
    assert referenced <= set(params), f"{name}: unbound params {referenced - set(params)}"


@pytest.mark.parametrize(("name", "build"), ALL_QUERIES, ids=[n for n, _ in ALL_QUERIES])
def test_no_query_writes(name, build):
    """The query library reads. Loaders write, and they live somewhere else."""
    cypher = build()[0].upper()
    for keyword in ("CREATE ", "MERGE ", "DELETE ", "SET ", "REMOVE ", "DROP "):
        assert keyword not in cypher, f"{name} contains {keyword.strip()}"


# --- agreement_coverage: the SRS 3.1.9 worked example --------------------


def test_coverage_expands_the_hs_hierarchy():
    """GSP+ declared at chapter 61 must answer "does it cover 6109?" with yes."""
    _, params = q.agreement_coverage("610910")
    assert params["hs_prefixes"] == ["610910", "6109", "61"]


def test_coverage_handles_leading_zero_codes():
    """Agriculture codes all have one; 902 must not be read as chapter 90."""
    _, params = q.agreement_coverage("0902")
    assert params["hs_prefixes"] == ["0902", "09"]
    _, params = q.agreement_coverage(902)
    assert params["hs_prefixes"] == ["0902", "09"]


def test_coverage_prefers_the_most_specific_match():
    cypher, _ = q.agreement_coverage("610910")
    assert "ORDER BY size(h.code) DESC" in cypher


def test_coverage_rejects_a_non_hs_code():
    with pytest.raises(CrosswalkError):
        q.agreement_coverage("gsp-plus")


# --- the rest ------------------------------------------------------------


def test_cagr_resolves_a_partner_name_to_iso3():
    """Callers pass whatever the user typed; the crosswalk normalizes it."""
    _, params = q.cagr("tea", "Germany", 2015, 2024)
    assert params["partner_iso3"] == "DEU"


def test_cagr_without_a_partner_does_not_filter_on_one():
    cypher, params = q.cagr("tea", None, 2015, 2024)
    assert "partner_iso3" not in params
    assert "c.iso3" not in cypher


def test_market_share_computes_its_own_denominator():
    """A separately-fetched total goes stale and produces shares summing past 1."""
    cypher, _ = q.market_share("tea", 2024)
    assert "sum(e.value) AS total" in cypher
    assert "e.value / total" in cypher


def test_market_share_guards_against_dividing_by_zero():
    cypher, _ = q.market_share("tea", 2024)
    assert "CASE WHEN total > 0" in cypher


def test_top_partners_ranks_by_value_not_volume():
    """A tonne of tea and a tonne of T-shirts are not the same question."""
    cypher, _ = q.top_partners("tea", 2024)
    assert "ORDER BY e.value DESC" in cypher


def test_item_matching_is_case_insensitive():
    """Users type "Cinnamon"; the graph stores "cinnamon"."""
    for _, build in ALL_QUERIES:
        cypher, params = build()
        if "item" in params or "commodity" in params:
            assert "toLower" in cypher


# --- the subgraph projections --------------------------------------------


@pytest.mark.parametrize(("name", "build"), SUBGRAPH_QUERIES, ids=[n for n, _ in SUBGRAPH_QUERIES])
def test_subgraph_queries_project_the_full_triple(name, build):
    """All nine columns, or `kg/subgraph.py` skips the row as malformed.

    This is the assertion that matters most in this file. `record.data()`
    flattens a Node to its properties alone, so a projection that drops an `AS`
    alias does not fail — it returns rows that quietly decode to a node labelled
    `None`, joined to the real graph.
    """
    cypher, _ = build()
    for column in (
        "source_label",
        "source_key",
        "source_name",
        "rel_type",
        "rel_props",
        "target_label",
        "target_key",
        "target_name",
        "weight",
    ):
        assert f"AS {column}" in cypher, f"{name}: no `AS {column}` in the projection"


@pytest.mark.parametrize(("name", "build"), SUBGRAPH_QUERIES, ids=[n for n, _ in SUBGRAPH_QUERIES])
def test_subgraph_queries_never_return_a_bare_node(name, build):
    """`RETURN n` would arrive as a property bag that no longer knows its label."""
    cypher, _ = build()
    assert not re.search(r"RETURN\s+[a-z]\s*(,|$)", cypher, re.MULTILINE), (
        f"{name} returns a whole node; labels do not survive record.data()"
    )


def test_export_subgraph_carries_value_as_the_weight():
    """Stroke width has to mean something; the something is export value."""
    cypher, _ = q.export_subgraph("tea", 2024)
    assert "e.value       AS weight" in cypher
    assert "ORDER BY e.value DESC" in cypher


def test_export_subgraph_is_bounded():
    cypher, params = q.export_subgraph("tea", 2024, limit=5)
    assert "LIMIT $limit" in cypher
    assert params["limit"] == 5


def test_classification_unions_rather_than_optional_matching():
    """An OPTIONAL MATCH on the agreement leg returns a row with null agreement
    columns for every uncovered HS code, which decodes to a `TradeAgreement:None`
    node hanging off the real graph. A union returns only edges that exist."""
    cypher, _ = q.classification_subgraph("tea")
    assert "UNION" in cypher
    assert "OPTIONAL MATCH" not in cypher
    assert "'CLASSIFIED_AS'" in cypher
    assert "'COVERED_BY'" in cypher


def test_neighbours_recovers_edge_direction():
    """Matched undirected so expanding a Country finds what exports *to* it.
    Without startNode/endNode every expanded edge would be drawn pointing away
    from whatever the user happened to click."""
    cypher, _ = q.neighbours("Country", "USA")
    assert "-[r]-()" in cypher
    assert "startNode(r)" in cypher and "endNode(r)" in cypher


def test_neighbours_substitutes_the_label_and_its_key_property():
    """Cypher has no parameter form for either — `MATCH (n:$label)` is a syntax
    error. Each label's key comes from NODE_KEYS, not from the caller."""
    cypher, _ = q.neighbours("Country", "USA")
    assert "(n:Country {iso3: $key})" in cypher

    cypher, _ = q.neighbours("HSCode", "6109")
    assert "(n:HSCode {code: $key})" in cypher

    cypher, _ = q.neighbours("PolicyDocument", "edb-2024")
    assert "(n:PolicyDocument {doc_id: $key})" in cypher


def test_neighbours_keeps_the_value_a_parameter():
    """The label is substituted; the value never is. A key that reached the
    query text would be the injection this module's docstring forbids."""
    cypher, params = q.neighbours("Country", "'; MATCH (n) DETACH DELETE n //")
    assert "DETACH DELETE" not in cypher
    assert params["key"] == "'; MATCH (n) DETACH DELETE n //"
    assert "$key" in cypher


def test_neighbours_rejects_a_label_that_is_not_in_the_graph():
    """The allowlist is what stands between a query string and query structure."""
    with pytest.raises(CrosswalkError):
        q.neighbours("Country) MATCH (x", "USA")
    with pytest.raises(CrosswalkError):
        q.neighbours("User", "admin")


def test_node_keys_covers_every_constrained_label():
    """schema.cypher constrains six labels; PolicyDocument is the seventh that
    the loader merges on `doc_id` (deviation D10). A label missing here cannot
    be expanded, which shows up as a click that silently does nothing."""
    assert set(q.NODE_KEYS) == {
        "Country",
        "HSCode",
        "Commodity",
        "ApparelCategory",
        "District",
        "TradeAgreement",
        "PolicyDocument",
    }
