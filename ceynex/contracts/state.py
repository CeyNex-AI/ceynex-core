"""Implements SRS 3.1.2, 3.1.4, 3.6.4 — the LangGraph state shared by all five agents.

FROZEN CONTRACT. Changes require 3-way approval (M1, M2, M3).

Every agent is a pure function `AgentState -> AgentState` that writes exactly one
key into `agent_outputs` and appends to `errors` on failure. An agent that raises
instead of writing an error breaks the orchestrator's partial-result guarantee
(SAD 4.1, cross-sector fail condition).

Reducers
--------
`agent_outputs`, `errors` and `degraded` carry `Annotated[..., reducer]`
annotations. This is not decoration: during a cross-sector query the orchestrator
fans out to 2-3 agent nodes in parallel, and LangGraph raises
`InvalidUpdateError` when concurrent branches write the same state key without a
reducer telling it how to combine them. The keys written once by the router
before fan-out (`route`, `sectors`, `query`, `user_id`) need no reducer.
"""

import operator
from typing import Annotated, Literal, TypedDict

from typing_extensions import NotRequired

from ceynex.contracts.evidence import Evidence
from ceynex.contracts.forecast import ForecastPoint

Sector = Literal["agriculture", "apparel", "cross_sector", "macro"]

AgentName = Literal[
    "export_analytics",
    "agriculture_commodity",
    "apparel_manufacturing",
    "trade_economics",
    "forecast",
]

ALL_AGENTS: tuple[AgentName, ...] = (
    "export_analytics",
    "agriculture_commodity",
    "apparel_manufacturing",
    "trade_economics",
    "forecast",
)

DEFAULT_AGENT: AgentName = "export_analytics"
"""The router must never return an empty route; it falls back to this."""


class AgentOutput(TypedDict):
    """What one agent node contributes to the shared state.

    `assumptions` is required non-empty for `trade_economics` (SRS 3.1.5); other
    agents may pass an empty list. `degraded` is True when the LLM was
    unavailable and the agent returned figures and evidence without a
    natural-language explanation (SRS 3.4.3).
    """

    agent: AgentName
    summary: str  # 2-4 sentences, plain English
    figures: dict[str, float]
    forecast: NotRequired[list[ForecastPoint]]
    assumptions: list[str]
    evidence: list[Evidence]
    confidence: float  # 0.0-1.0
    degraded: bool
    error: NotRequired[str]


def merge_agent_outputs(
    left: dict[AgentName, AgentOutput],
    right: dict[AgentName, AgentOutput],
) -> dict[AgentName, AgentOutput]:
    """Reducer for `AgentState.agent_outputs` under parallel fan-out.

    Each agent writes under its own `AgentName` key, so a plain dict merge is
    collision-free by construction. Kept as a named function rather than
    `operator.or_` so the intent survives a reader who has not read this module.
    """
    return {**left, **right}


class AgentState(TypedDict):
    """The single shared state passed through the one LangGraph graph (SRS 3.6.4)."""

    query: str
    user_id: str
    route: list[AgentName]
    sectors: list[Sector]
    relevance: dict[AgentName, float]  # router's per-agent weight, feeds confidence.py
    agent_outputs: Annotated[dict[AgentName, AgentOutput], merge_agent_outputs]
    final_answer: NotRequired[str]
    final_confidence: NotRequired[float]
    merged_evidence: NotRequired[list[Evidence]]
    degraded: Annotated[bool, operator.or_]
    errors: Annotated[list[str], operator.add]


def new_state(query: str, user_id: str) -> AgentState:
    """Build an empty, valid `AgentState`. Use this rather than a dict literal."""
    return AgentState(
        query=query,
        user_id=user_id,
        route=[],
        sectors=[],
        relevance={},
        agent_outputs={},
        degraded=False,
        errors=[],
    )


def failed_output(agent: AgentName, error: str) -> AgentOutput:
    """The output an agent node returns instead of raising (SAD 4.1).

    Confidence is zero and `degraded` is True so the orchestrator's merge step
    can report which part of the question could not be answered rather than
    silently omitting it (SRS 3.4.3).
    """
    return AgentOutput(
        agent=agent,
        summary=f"The {agent.replace('_', ' ')} agent could not answer this part of the question.",
        figures={},
        assumptions=[],
        evidence=[],
        confidence=0.0,
        degraded=True,
        error=error,
    )
