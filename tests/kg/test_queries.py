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
]


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
