"""Assertions for SRS 3.1.5 — the simulation agent, and its refusal path.

The refusal path is the one worth testing hardest. SAD §4.1 says that when the
graph cannot support a simulation the agent reports that rather than producing a
number, and a number produced from a missing premise is the single worst output
this system could give: it looks exactly like a real answer.
"""

import asyncio

import pytest

from ceynex.agents.common import AgentDeps
from ceynex.agents.trade_economics import AGENT, trade_economics_node
from ceynex.contracts import new_state
from ceynex.llm import FakeLLMClient

BASELINE_USD = 1_000_000.0


class KG:
    """A graph with baseline trade data and configurable agreement coverage."""

    def __init__(self, *, coverage: list[dict] | None = None, raises=None):
        self._coverage = coverage if coverage is not None else []
        self._raises = raises

    async def run(self, cypher, params=None):
        if self._raises:
            raise self._raises
        if "latest_year" in cypher:
            return [{"latest_year": 2024}], cypher
        if "COVERED_BY" in cypher:
            return list(self._coverage), cypher
        return [{"total_export_value_usd": BASELINE_USD}], cypher


GSP_PLUS = [
    {
        "agreement": "GSP+",
        "agreement_type": "unilateral_preference",
        "matched_on": "61",
        "agreement_verified": "verified",
    }
]


async def run(query: str, kg: KG):
    patch = await trade_economics_node(new_state(query, "test"), AgentDeps(kg=kg, llm=FakeLLMClient(available=False)))
    return patch["agent_outputs"][AGENT]


# --- the refusal path (SAD §4.1) -----------------------------------------


def test_missing_coverage_refuses_rather_than_inventing_a_number():
    out = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=[])))

    assert out["figures"] == {}, "a refusal that still reports an impact figure is not a refusal"
    assert any("cannot be simulated" in a for a in out["assumptions"])


def test_the_refusal_cites_the_query_that_found_no_coverage():
    """Evidence has to support its own claim, or grounding is theatre."""
    out = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=[])))

    coverage_claims = [e for e in out["evidence"] if "No trade-agreement coverage" in e["claim"]]
    assert coverage_claims, "the refusal did not say why it refused"
    assert "COVERED_BY" in coverage_claims[0]["detail"], (
        "the refusal cited some other query as evidence that coverage is missing"
    )


def test_a_refusal_still_meets_the_two_evidence_floor():
    out = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=[])))
    assert len(out["evidence"]) >= 2


def test_coverage_that_is_not_a_preference_is_also_a_refusal():
    """An FTA is not GSP+. Losing a preference you never had is not a shock."""
    fta = [dict(GSP_PLUS[0], agreement="ISFTA", agreement_type="bilateral_fta")]
    out = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=fta)))
    assert out["figures"] == {}


def test_present_coverage_produces_a_simulation():
    """The mirror of the refusal tests: with coverage, a number is expected."""
    out = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=GSP_PLUS)))

    assert out["figures"], "coverage was present and the simulation still produced nothing"
    assert out["figures"]["apparel_impact_usd"] < 0, "losing a preference cannot raise revenue"


# --- the agent node contract ---------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "How would a 5% rupee depreciation affect apparel exports?",
        "What if the EU raises tariffs on tea by 10%?",
        "What happens to agriculture if Sri Lanka loses GSP+?",
    ],
)
def test_every_simulation_states_its_assumptions(query):
    """SRS 3.1.5 requires the assumptions, not just the number."""
    out = asyncio.run(run(query, KG(coverage=GSP_PLUS)))
    assert out["assumptions"], "a simulation with no stated assumptions is an unfalsifiable claim"


def test_the_node_writes_exactly_one_output_key():
    state = new_state("How would a 5% rupee depreciation affect apparel exports?", "test")
    deps = AgentDeps(kg=KG(coverage=GSP_PLUS), llm=FakeLLMClient(available=False))
    patch = asyncio.run(trade_economics_node(state, deps))
    assert set(patch["agent_outputs"]) == {AGENT}


def test_the_node_never_raises_when_the_graph_is_down():
    """SAD §4.1 partial-result guarantee."""
    out = asyncio.run(run("How would a 5% depreciation affect apparel?", KG(raises=RuntimeError("neo4j down"))))
    assert out["agent"] == AGENT
    assert out["confidence"] == 0.0 or out["figures"] == {}


def test_confidence_is_derived_rather_than_hardcoded():
    """Two runs with materially different evidence must not score identically."""
    refused = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=[])))
    answered = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=GSP_PLUS)))
    assert refused["confidence"] != answered["confidence"]
