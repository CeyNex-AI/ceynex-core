"""Assertions for SRS 3.1.6 — trends answered from the graph, not from a model.

That is an architectural claim, not a preference, so the test that matters most
is the one asserting every figure this agent reports is traceable to a Cypher
query it actually ran. If this agent ever starts inferring numbers, these fail.
"""

import asyncio

import pytest

from ceynex.agents.common import AgentDeps
from ceynex.agents.export_analytics import AGENT, export_analytics_node
from ceynex.contracts import new_state
from ceynex.llm import FakeLLMClient

MARKET_SHARE = [
    {"partner": "United States", "partner_iso3": "USA", "export_value_usd": 600.0, "total_export_value_usd": 1000.0, "share": 0.6},
    {"partner": "United Kingdom", "partner_iso3": "GBR", "export_value_usd": 300.0, "total_export_value_usd": 1000.0, "share": 0.3},
    {"partner": "Germany", "partner_iso3": "DEU", "export_value_usd": 100.0, "total_export_value_usd": 1000.0, "share": 0.1},
]

CAGR_ROWS = [
    {"year": 2020, "export_value_usd": 800.0, "export_volume": 80.0},
    {"year": 2024, "export_value_usd": 1000.0, "export_volume": 95.0},
]

GROWTH_BY_PARTNER = [
    {"partner": "United States", "start_value": 5_000_000.0, "end_value": 6_000_000.0},
    {"partner": "Germany", "start_value": 2_000_000.0, "end_value": 4_000_000.0},
]


class KG:
    """Routes on the query text, and records every Cypher it was asked to run."""

    def __init__(self, *, share=None, growth=None, by_partner=None, districts=None, raises=None):
        self.share = MARKET_SHARE if share is None else share
        self.growth = CAGR_ROWS if growth is None else growth
        self.by_partner = GROWTH_BY_PARTNER if by_partner is None else by_partner
        self.districts = districts or []
        self.raises = raises
        self.seen: list[str] = []

    async def run(self, cypher, params=None):
        self.seen.append(cypher)
        if self.raises:
            raise self.raises
        if "latest_year" in cypher:
            return [{"latest_year": 2024}], cypher
        if "District" in cypher:
            return list(self.districts), cypher
        if "start_value" in cypher:
            return list(self.by_partner), cypher
        if "e.year AS year" in cypher and "sum(e.value)" in cypher:
            return list(self.growth), cypher
        return list(self.share), cypher


def run(query="which market takes the largest share of tea exports?", kg=None):
    kg = kg or KG()
    patch = asyncio.run(
        export_analytics_node(new_state(query, "test"), AgentDeps(kg=kg, llm=FakeLLMClient(available=False)))
    )
    return patch["agent_outputs"][AGENT], kg


# --- the architectural claim ---------------------------------------------


def test_every_figure_is_backed_by_cypher_the_agent_actually_ran():
    """SRS 3.1.6. A figure with no query behind it is a model answer in disguise."""
    out, kg = run()

    assert out["figures"], "the agent reported nothing at all"
    details = " ".join(e["detail"] for e in out["evidence"])
    assert "MATCH" in details, "no evidence carries a Cypher query"

    # `evidence_from_query` collapses whitespace, so compare on that basis
    # rather than on the raw text the driver was handed.
    ran = {" ".join(cypher.split()) for cypher in kg.seen}
    for evidence in out["evidence"]:
        if evidence["source_id"] == "KG":
            assert " ".join(evidence["detail"].split()) in ran, (
                "evidence cites a query that was never run"
            )


def test_the_agent_queries_the_graph_rather_than_answering_from_nothing():
    _, kg = run()
    assert kg.seen, "the agent produced an answer without touching the graph"


# --- the numbers ---------------------------------------------------------


def test_market_share_is_reported_from_the_leading_partner_row():
    out, _ = run()
    assert out["figures"]["top_partner_share"] == pytest.approx(0.6)
    assert out["figures"]["partner_count"] == 3.0


def test_the_concentration_index_is_the_sum_of_squared_shares():
    """HHI for 0.6/0.3/0.1 is 0.36 + 0.09 + 0.01 = 0.46. Verified by hand."""
    out, _ = run()
    assert out["figures"]["hhi"] == pytest.approx(0.46, abs=0.005)


def test_a_single_destination_market_is_maximally_concentrated():
    single = [{"partner": "United States", "partner_iso3": "USA", "export_value_usd": 1000.0, "total_export_value_usd": 1000.0, "share": 1.0}]
    out, _ = run(kg=KG(share=single))
    assert out["figures"]["hhi"] == pytest.approx(1.0)


def test_compound_growth_is_annualised_not_totalled():
    """800 -> 1000 over 4 years is 5.7% a year, not 25%."""
    out, _ = run()
    assert out["figures"]["cagr"] == pytest.approx(0.0574, abs=0.001)


def test_a_zero_starting_year_is_reported_rather_than_dividing_by_zero():
    zero_start = [
        {"year": 2020, "export_value_usd": 0.0, "export_volume": 0.0},
        {"year": 2024, "export_value_usd": 1000.0, "export_volume": 95.0},
    ]
    out, _ = run(kg=KG(growth=zero_start))
    assert "cagr" not in out["figures"]
    assert any("CAGR could not be computed" in a for a in out["assumptions"])


# --- the agent node contract ---------------------------------------------


def test_the_node_writes_exactly_one_output_key():
    patch = asyncio.run(
        export_analytics_node(new_state("tea exports", "test"), AgentDeps(kg=KG(), llm=FakeLLMClient(available=False)))
    )
    assert set(patch["agent_outputs"]) == {AGENT}


def test_at_least_two_evidence_entries_are_attached():
    out, _ = run()
    assert len(out["evidence"]) >= 2


def test_an_empty_graph_is_reported_rather_than_guessed_around():
    out, _ = run(kg=KG(share=[], growth=[], by_partner=[]))
    assert out["assumptions"], "the agent found nothing and said nothing about it"
    assert "top_partner_share" not in out["figures"]


def test_the_node_never_raises_when_the_graph_is_down():
    """SAD §4.1 partial-result guarantee."""
    out, _ = run(kg=KG(raises=RuntimeError("neo4j unreachable")))
    assert out["agent"] == AGENT


def test_confidence_falls_when_the_graph_returns_less():
    full, _ = run()
    sparse, _ = run(kg=KG(share=[], growth=[], by_partner=[]))
    assert sparse["confidence"] < full["confidence"]


def test_the_agent_degrades_without_prose_when_the_llm_is_down():
    """SRS 3.4.3: figures and evidence, no generated explanation."""
    out, _ = run()
    assert out["degraded"] is True
    assert out["figures"]
    assert out["evidence"]
