"""The contracts are frozen, so these tests are the thing that notices if they thaw."""

import operator

import pytest

from ceynex.contracts import (
    ALL_AGENTS,
    AgentOutput,
    Evidence,
    ForecastPoint,
    failed_output,
    merge_agent_outputs,
    new_state,
)
from ceynex.contracts.state import AgentState


def _output(agent, confidence=0.8, degraded=False) -> AgentOutput:
    return AgentOutput(
        agent=agent,
        summary="s",
        figures={"x": 1.0},
        assumptions=[],
        evidence=[],
        confidence=confidence,
        degraded=degraded,
    )


def test_new_state_has_every_required_key():
    state = new_state("query", "user-1")
    for key in ("query", "user_id", "route", "sectors", "relevance", "agent_outputs", "degraded", "errors"):
        assert key in state
    assert state["degraded"] is False
    assert state["errors"] == []


def test_five_agents_exactly():
    """SRS 3.6.4 — five agents, all as nodes in the one graph."""
    assert len(ALL_AGENTS) == 5
    assert set(ALL_AGENTS) == {
        "export_analytics",
        "agriculture_commodity",
        "apparel_manufacturing",
        "trade_economics",
        "forecast",
    }


def test_parallel_state_keys_carry_reducers():
    """Regression guard for the Day 7 fan-out.

    LangGraph raises InvalidUpdateError when concurrent branches write a state
    key with no reducer. agent_outputs, errors and degraded are all written by
    agent nodes running in parallel, so all three must stay Annotated.
    """
    hints = AgentState.__annotations__
    assert getattr(hints["agent_outputs"], "__metadata__", None) == (merge_agent_outputs,)
    assert getattr(hints["errors"], "__metadata__", None) == (operator.add,)
    assert getattr(hints["degraded"], "__metadata__", None) == (operator.or_,)


def test_merge_agent_outputs_combines_parallel_branches():
    left = {"export_analytics": _output("export_analytics")}
    right = {"trade_economics": _output("trade_economics")}
    merged = merge_agent_outputs(left, right)
    assert set(merged) == {"export_analytics", "trade_economics"}
    assert left == {"export_analytics": _output("export_analytics")}  # inputs untouched


def test_failed_output_never_raises_and_scores_zero():
    """SAD 4.1 — an agent reports failure into state instead of raising."""
    out = failed_output("forecast", "neo4j unreachable")
    assert out["confidence"] == 0.0
    assert out["degraded"] is True
    assert out["error"] == "neo4j unreachable"
    assert out["agent"] == "forecast"


def test_forecast_point_requires_an_interval():
    """SRS 3.1.3 — a forecast is never an unqualified number."""
    point = ForecastPoint(period="2026-Q4", point=10.0, lower=8.0, upper=12.0, unit="USD_mn")
    assert point["lower"] < point["point"] < point["upper"]
    with pytest.raises(KeyError):
        _ = ForecastPoint(period="2026-Q4", point=10.0)["lower"]  # type: ignore[typeddict-item]


def test_evidence_carries_human_and_machine_halves():
    """SRS 3.1.4 — claim is readable; detail is traceable."""
    ev = Evidence(
        source_id="KG",
        claim="Germany was the largest EU buyer of Sri Lankan knit apparel in 2023.",
        detail="MATCH (a:ApparelCategory {name:$name})-[e:EXPORTS_TO]->(c:Country) ...",
        period="2023",
    )
    assert ev["claim"].endswith(".")
    assert ev["detail"]
