"""Every agent's real output satisfies the frozen `AgentOutput` contract.

This is the consumer-side half of contract testing: the merger, the API's
response models and ceynex-web all trust that an agent node returns an
`AgentOutput` with exactly the keys and types `ceynex.contracts` declares.
Nothing enforced that until now, because a TypedDict is not checked at runtime,
so a change to `ceynex/contracts/state.py` could silently break every consumer.

These tests run each of the five agents through a stub knowledge graph and a
`FakeLLMClient` (no docker, no network, per the repo's testing rule) and assert
the output validates against the contract as it currently stands. If the frozen
shape changes, they fail here in core CI rather than as a KeyError in the
browser.
"""

from __future__ import annotations

import asyncio

import pytest

from ceynex.agents.agriculture_commodity import agriculture_commodity_node
from ceynex.agents.apparel_manufacturing import apparel_manufacturing_node
from ceynex.agents.common import AgentDeps
from ceynex.agents.export_analytics import export_analytics_node
from ceynex.agents.forecast import forecast_node
from ceynex.agents.trade_economics import trade_economics_node
from ceynex.contracts import AgentOutput, new_state
from ceynex.contracts.state import AgentState
from ceynex.llm import FakeLLMClient

from ._validate import assert_matches


class _StubKG:
    """Answers every Cypher with no rows. An agent that finds nothing must still
    return a well-formed AgentOutput (a refusal or a degraded note), which is
    exactly the shape under test."""

    async def run(self, cypher: str, params: dict | None = None):
        return [], cypher


_AGENTS = {
    "export_analytics": (export_analytics_node, "tea export market share by partner"),
    "agriculture_commodity": (agriculture_commodity_node, "cinnamon price trend"),
    "apparel_manufacturing": (apparel_manufacturing_node, "knitted apparel exports to the US"),
    "trade_economics": (trade_economics_node, "impact of losing GSP+ on apparel"),
    "forecast": (forecast_node, "forecast tea exports next year"),
}


def _run(node, query: str) -> dict:
    state = new_state(query, "tester@ceynex.dev")
    deps = AgentDeps(kg=_StubKG(), llm=FakeLLMClient())
    return asyncio.run(node(state, deps))


@pytest.mark.parametrize("name", sorted(_AGENTS))
def test_agent_output_matches_the_frozen_contract(name: str) -> None:
    node, query = _AGENTS[name]
    result = _run(node, query)

    assert "agent_outputs" in result, f"{name} returned no agent_outputs"
    outputs = result["agent_outputs"]
    assert outputs, f"{name} wrote an empty agent_outputs"

    for key, output in outputs.items():
        errors = assert_matches(output, AgentOutput)
        assert not errors, f"{name} output under {key!r} violates AgentOutput:\n" + "\n".join(
            errors
        )
        # The agent must key its output under its own name (SAD 4.1).
        assert output["agent"] == key


def test_trade_economics_states_its_assumptions() -> None:
    """SRS 3.1.5: trade_economics may never return an empty `assumptions`, even
    on the no-data path — a simulation figure with no stated basis is exactly
    the defect the contract's docstring calls out."""
    result = _run(trade_economics_node, "impact of a 10% EU tariff on Sri Lankan tea")
    output = result["agent_outputs"]["trade_economics"]
    assert output["assumptions"], "trade_economics returned no assumptions"


def test_new_state_is_a_valid_agent_state() -> None:
    errors = assert_matches(new_state("q", "u@x"), AgentState)
    assert not errors, "new_state() drifted from AgentState:\n" + "\n".join(errors)
