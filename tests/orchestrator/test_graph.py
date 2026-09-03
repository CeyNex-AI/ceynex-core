"""Assertions for SRS 3.6.4 — one LangGraph graph containing all five agents.

The constraint is architectural, so it is tested structurally: five nodes, all
present, all reachable, none of them a separate service. The rest of the file is
resilience — the graph must answer when a node fails, hangs, or does not exist.
"""

import asyncio
import re

import pytest

from ceynex.agents.common import AgentDeps
from ceynex.contracts import ALL_AGENTS, new_state
from ceynex.llm import FakeLLMClient
from ceynex.orchestrator.graph import (
    NODE_TIMEOUT_S,
    agent_registry,
    build_graph,
)


class FakeKG:
    """A knowledge graph that returns whatever it is told to, and records queries."""

    def __init__(self, rows=None, raises=None):
        self._rows = rows if rows is not None else []
        self._raises = raises
        self.queries: list[str] = []

    async def run(self, cypher, params=None):
        self.queries.append(cypher)
        if self._raises:
            raise self._raises
        return list(self._rows), cypher


def deps(kg=None, llm=None):
    return AgentDeps(kg=kg or FakeKG(), llm=llm or FakeLLMClient(available=False))


# --- the single-graph constraint ----------------------------------------


def test_all_five_agents_are_nodes_in_the_registry():
    """SRS 3.6.4. Four would mean one is being invoked some other way."""
    registry = agent_registry()
    assert set(registry) == set(ALL_AGENTS)
    assert len(registry) == 5


def test_every_agent_resolves_to_something_callable():
    for agent, node in agent_registry().items():
        assert callable(node), agent


def test_the_compiled_graph_contains_every_agent_node():
    graph = build_graph(deps(), use_llm_router=False)
    nodes = set(graph.get_graph().nodes)
    for agent in ALL_AGENTS:
        assert agent in nodes, f"{agent} is not a node in the graph"
    assert {"route", "merge"} <= nodes


def test_teammates_agents_resolve_or_degrade_rather_than_being_missing():
    """Each external node is either real or a contract-conformant fallback."""
    registry = agent_registry()
    assert registry["agriculture_commodity"] is not None
    assert registry["apparel_manufacturing"] is not None


async def test_the_real_agriculture_agent_is_resolved_not_the_placeholder():
    registry = agent_registry()
    from ceynex.agents.agriculture_commodity import agriculture_commodity_node

    assert registry["agriculture_commodity"] is agriculture_commodity_node


# --- end to end ----------------------------------------------------------


async def test_a_query_runs_through_the_graph_and_produces_an_answer():
    graph = build_graph(deps(), use_llm_router=False)
    final = await graph.ainvoke(new_state("cinnamon export trend", "u"))

    assert final["final_answer"]
    assert 0.0 < final["final_confidence"] < 1.0
    assert "merged_evidence" in final


async def test_the_route_is_recorded_in_state():
    graph = build_graph(deps(), use_llm_router=False)
    final = await graph.ainvoke(new_state("cinnamon export trend", "u"))
    assert final["route"]
    assert all(agent in ALL_AGENTS for agent in final["route"])


@pytest.mark.parametrize(
    "query",
    [
        "What is the outlook for Sri Lankan gem exports?",
        "How are shipping costs affecting Sri Lankan exporters?",
    ],
)
async def test_an_out_of_scope_question_is_not_answered_about_tea(query):
    """The route is never empty, so `export_analytics` and `forecast` run and
    default to tea (`item = intent.item or "tea"`). Their output is a real answer
    to a question nobody asked, and the merger drops it -- but only when the
    `out_of_scope_no_topic` marker reaches it.

    Keyed on `no_topic_recognized`, the marker was absent whenever an excluded
    topic *was* recognised, so "what is the outlook for Sri Lankan gem exports?"
    came back with a tea export-value forecast and an 80% interval, with the
    scope note appended after it. A confident answer to a question about a sector
    CeyNex does not cover is worse than no answer.
    """
    graph = build_graph(deps(), use_llm_router=False)
    final = await graph.ainvoke(new_state(query, "u"))

    answer = final["final_answer"]
    assert "could not be answered" in answer.lower()
    # Neither "tea" nor a bare digit is the tell: the scope sentence names tea as
    # something CeyNex *does* cover, and cites "HS 61/62" and "SRS 2.4". What a
    # served answer always carries and a refusal never does is a measured
    # quantity -- a USD value or a percentage.
    assert not re.search(r"USD|\d+(\.\d+)?%", answer), (
        f"a figure reached an out-of-scope answer: {answer}"
    )
    assert final["merged_evidence"] == []
    assert final["final_confidence"] == pytest.approx(0.15)


async def test_a_mixed_out_of_scope_question_still_answers_its_in_scope_half():
    """The guard above must not swallow the half CeyNex can answer."""
    graph = build_graph(deps(), use_llm_router=False)
    final = await graph.ainvoke(new_state("Should Sri Lanka prioritise gems or tea next year?", "u"))

    assert "tea" in final["final_answer"].lower()
    assert "gem" in final["final_answer"].lower(), "the excluded half must still be named"


async def test_a_cross_sector_query_fans_out_to_several_agents():
    """The parallel branch. Without the state reducers this raises InvalidUpdateError."""
    graph = build_graph(deps(), use_llm_router=False)
    final = await graph.ainvoke(
        new_state(
            "How would a 5% rupee depreciation affect apparel exports compared to agriculture?",
            "u",
        )
    )
    assert len(final["route"]) >= 3
    assert len(final["agent_outputs"]) >= 3, "parallel writes were lost"


async def test_parallel_agent_outputs_all_survive_the_merge():
    """The reducer on agent_outputs is what makes this true (contracts/CLAUDE.md)."""
    graph = build_graph(deps(), use_llm_router=False)
    final = await graph.ainvoke(
        new_state("tariffs on tea and garments", "u")
    )
    assert set(final["agent_outputs"]) == set(final["route"])


# --- resilience ----------------------------------------------------------


async def test_a_dead_knowledge_graph_degrades_rather_than_500s():
    from ceynex.kg.client import KnowledgeGraphUnavailableError

    graph = build_graph(
        deps(kg=FakeKG(raises=KnowledgeGraphUnavailableError("neo4j is down"))),
        use_llm_router=False,
    )
    final = await graph.ainvoke(new_state("cinnamon export trend", "u"))

    assert final["final_answer"], "the system must respond, not fail"
    assert final["degraded"]
    assert final["errors"]


async def test_an_agent_that_raises_does_not_take_down_the_invocation():
    """Belt to the agent contract's braces."""

    class Exploding:
        async def run(self, cypher, params=None):
            raise RuntimeError("unexpected boom")

    graph = build_graph(deps(kg=Exploding()), use_llm_router=False)
    final = await graph.ainvoke(new_state("cinnamon export trend", "u"))
    assert final["final_answer"]
    assert final["errors"]


async def test_an_unavailable_llm_still_produces_an_answer():
    """SRS 3.4.3 — figures and evidence without prose is a conforming answer."""
    graph = build_graph(deps(llm=FakeLLMClient(available=False)), use_llm_router=False)
    final = await graph.ainvoke(new_state("cinnamon export trend", "u"))
    assert final["final_answer"]
    assert final["degraded"]


async def test_a_hanging_node_is_timed_out_rather_than_blocking_the_budget():
    """SRS 3.4.1 — one slow agent must not spend the whole response-time budget."""

    class Hanging:
        async def run(self, cypher, params=None):
            await asyncio.sleep(NODE_TIMEOUT_S + 5)
            return [], cypher

    # Shortened so the test does not sit out the real budget. The graph reads
    # the module global at call time, so it has to be patched before building.
    import ceynex.orchestrator.graph as graph_module

    original = graph_module.NODE_TIMEOUT_S
    graph_module.NODE_TIMEOUT_S = 0.1
    try:
        graph = build_graph(deps(kg=Hanging()), use_llm_router=False)
        final = await asyncio.wait_for(graph.ainvoke(new_state("cinnamon trend", "u")), timeout=10)
    finally:
        graph_module.NODE_TIMEOUT_S = original

    assert final["final_answer"]
    assert any("budget" in error for error in final["errors"])


async def test_an_empty_query_still_routes_somewhere():
    graph = build_graph(deps(), use_llm_router=False)
    final = await graph.ainvoke(new_state("", "u"))
    assert final["route"]
    assert final["final_answer"]


# --- state shape ---------------------------------------------------------


async def test_the_final_state_matches_the_contract():
    graph = build_graph(deps(), use_llm_router=False)
    final = await graph.ainvoke(new_state("cinnamon export trend", "u"))

    for key in ("query", "user_id", "route", "sectors", "relevance", "agent_outputs",
                "degraded", "errors", "final_answer", "final_confidence", "merged_evidence"):
        assert key in final, f"{key} missing from the final state"

    assert isinstance(final["agent_outputs"], dict)
    assert isinstance(final["errors"], list)
    assert isinstance(final["degraded"], bool)


@pytest.mark.parametrize("agent", ALL_AGENTS)
async def test_every_agent_output_conforms_to_the_contract(agent):
    """Whatever an agent does, what it writes has a fixed shape."""
    registry = agent_registry()
    patch = await registry[agent](new_state("tea exports to Germany", "u"), deps())
    output = patch["agent_outputs"][agent]

    assert output["agent"] == agent
    assert isinstance(output["summary"], str)
    assert isinstance(output["figures"], dict)
    assert isinstance(output["assumptions"], list)
    assert isinstance(output["evidence"], list)
    assert 0.0 <= output["confidence"] <= 1.0
    assert isinstance(output["degraded"], bool)
