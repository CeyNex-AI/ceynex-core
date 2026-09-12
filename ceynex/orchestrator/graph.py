"""Implements SRS 3.6.4 and 3.1.2 — the single LangGraph orchestration graph.

**All five agents are nodes in one graph.** Not five services, not five ad-hoc
calls. That is a design constraint the SRS states outright, and it is what lets
the orchestrator fan out to several agents against shared state and merge their
outputs without reconciling protocols between them.

    route ──┬──▶ export_analytics ─────┬──▶ merge ──▶ END
            ├──▶ agriculture_commodity ┤
            ├──▶ apparel_manufacturing ┤
            ├──▶ trade_economics ──────┤
            └──▶ forecast ─────────────┘

The router writes `route`, `sectors` and `relevance`; LangGraph fans out to the
routed nodes in parallel; `merge` gathers everything. The state reducers on
`agent_outputs`, `errors` and `degraded` are what make the parallel writes safe
— without them LangGraph raises `InvalidUpdateError` (see contracts/CLAUDE.md).

**M1's and M3's nodes are resolved dynamically.** Their agents live in their own
slices and land on their own schedule. `_resolve_agent` imports the real module
if it is importable and otherwise registers a node that returns `failed_output`
naming what is missing. The graph is therefore complete and the partial-result
path is exercised from today, without either teammate's file existing yet and
without this module being edited when they do.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from ceynex.agents.common import AgentDeps
from ceynex.agents.export_analytics import export_analytics_node
from ceynex.agents.forecast import forecast_node
from ceynex.agents.trade_economics import trade_economics_node
from ceynex.chat import instructions
from ceynex.contracts import ALL_AGENTS, AgentName, AgentState, failed_output
from ceynex.observability import context as obs
from ceynex.observability import trace
from ceynex.orchestrator import planner
from ceynex.orchestrator.merger import MERGE_RULES_PRESENTATION, merge
from ceynex.orchestrator.router import RouteDecision, keyword_route, llm_route

log = logging.getLogger(__name__)

AgentNode = Callable[[AgentState, AgentDeps], Awaitable[dict[str, Any]]]

# SRS 3.4.1 gives 10s single-sector and 20s cross-sector end to end. One slow
# node must not be able to spend the whole budget, so each gets a slice of it and
# a node that overruns degrades to a partial result rather than blocking merge.
NODE_TIMEOUT_S = 12.0

# Agents owned by other members. Imported by name so this file needs no edit when
# they land, and so their absence is a partial result rather than an ImportError.
EXTERNAL_AGENTS: dict[AgentName, tuple[str, str]] = {
    "agriculture_commodity": ("ceynex.agents.agriculture_commodity", "agriculture_commodity_node"),
    "apparel_manufacturing": ("ceynex.agents.apparel_manufacturing", "apparel_manufacturing_node"),
}

OWNER_OF: dict[AgentName, str] = {
    "agriculture_commodity": "M1 (agriculture)",
    "apparel_manufacturing": "M3 (apparel)",
}


def _missing_agent_node(agent: AgentName) -> AgentNode:
    """Stand-in for an agent whose module is not present yet.

    Returns a contract-conformant failure rather than raising, so the orchestrator
    answers with whatever else succeeded and names the gap (SAD §4.1).
    """

    async def node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
        owner = OWNER_OF.get(agent, "another member")
        reason = f"the {agent.replace('_', ' ')} agent is not implemented yet ({owner})"
        return {
            "agent_outputs": {agent: failed_output(agent, reason)},
            "errors": [f"{agent}: {reason}"],
        }

    return node


def _resolve_agent(agent: AgentName) -> AgentNode:
    """The real node if importable, otherwise the missing-agent stand-in."""
    if agent in EXTERNAL_AGENTS:
        module_name, attribute = EXTERNAL_AGENTS[agent]
        try:
            module = importlib.import_module(module_name)
            node = getattr(module, attribute)
        except (ImportError, AttributeError):
            log.info("%s not available yet — registering the partial-result stand-in", agent)
            return _missing_agent_node(agent)
        log.info("%s resolved from %s", agent, module_name)
        return node

    return {
        "export_analytics": export_analytics_node,
        "trade_economics": trade_economics_node,
        "forecast": forecast_node,
    }[agent]


def agent_registry() -> dict[AgentName, AgentNode]:
    """Every one of the five agents, resolved. Never a partial mapping."""
    return {agent: _resolve_agent(agent) for agent in ALL_AGENTS}


def build_graph(deps: AgentDeps, *, use_llm_router: bool = True) -> Any:
    """Compile the one graph. Called once per process, not per query."""
    registry = agent_registry()
    builder = StateGraph(AgentState)

    async def route_node(state: AgentState) -> dict[str, Any]:
        with trace.node("route"):
            # Concurrently, not in sequence: both take only the question, and
            # routing is the single largest fixed cost in a query (4.6s measured
            # live). Overlapping them makes the plan effectively free on a path
            # that is already over its SRS 3.4.1 budget.
            decision, (steps, method) = await asyncio.gather(
                _decide_route(state, deps, use_llm_router),
                _plan_steps(state["query"], deps),
            )
            for position, step in enumerate(steps, start=1):
                trace.emit("thought", step=step, position=position, of=len(steps), method=method)
            trace.emit(
                "route",
                route=list(decision.route),
                sectors=list(decision.sectors),
                method=decision.method,
                out_of_scope=decision.out_of_scope,
                relevance={agent: round(score, 3) for agent, score in decision.relevance.items()},
            )
        patch = decision.as_state_patch()
        log.info("routed to %s (%s)", decision.route, decision.method)
        if decision.out_of_scope:
            errors = [f"out_of_scope: {decision.notes[0] if decision.notes else ''}"]
            if decision.nothing_in_scope:
                # A second, machine-only marker (never surfaced as prose) --
                # merger.py reads it to tell "named an excluded sector, still
                # answer the in-scope part" from "named nothing CeyNex covers,
                # answer nothing".
                #
                # Keyed on `nothing_in_scope`, not `no_topic_recognized`: naming
                # only an excluded topic leaves just as little to answer as
                # naming nothing at all. Keyed on the narrower flag, "what is the
                # outlook for Sri Lankan gem exports?" was served a confident tea
                # forecast with a scope note stapled to the end.
                errors.append("out_of_scope_no_topic: true")
            patch["errors"] = errors
        return patch

    builder.add_node("route", route_node)

    for agent, node in registry.items():
        builder.add_node(agent, _wrap(agent, node, deps))

    async def merge_node(state: AgentState) -> dict[str, Any]:
        with trace.node("merge"):
            # Read from ambient request context rather than passed down the
            # graph: `AgentState` is frozen and this node's shape is not the
            # place to carry a presentation preference (D15). Empty outside a
            # request and for every reader who has not set one, which is the
            # common case and reproduces the original prompt exactly.
            instruction = obs.current_instruction()
            presentation = (
                instructions.presentation_block(instruction, MERGE_RULES_PRESENTATION)
                if instruction
                else None
            )
            if instruction:
                trace.emit("instruction", applied=True, chars=len(instruction))
            result = await merge(state, deps.llm, presentation=presentation)
            patch = result.as_state_patch()
            # `AgentState` is frozen at three keys out of merge, so the working
            # behind the score travels on the request rather than the state.
            observation = obs.current()
            if observation is not None:
                observation.confidence_breakdown = result.confidence_breakdown
            trace.emit(
                "merge",
                confidence_breakdown=result.confidence_breakdown,
                confidence=round(float(patch.get("final_confidence", 0.0)), 3),
                evidence_count=len(patch.get("merged_evidence", []) or []),
                degraded=bool(patch.get("degraded", False)),
            )
        return patch

    builder.add_node("merge", merge_node)

    builder.add_edge(START, "route")
    # Conditional fan-out: LangGraph runs the returned nodes in parallel, and the
    # reducers on AgentState merge their writes.
    builder.add_conditional_edges("route", _fan_out, list(ALL_AGENTS))
    for agent in ALL_AGENTS:
        builder.add_edge(agent, "merge")
    builder.add_edge("merge", END)

    return builder.compile()


def _fan_out(state: AgentState) -> list[AgentName]:
    """Which nodes run. Never empty — an empty fan-out ends the graph silently."""
    route = [agent for agent in state.get("route", []) if agent in ALL_AGENTS]
    return route or ["export_analytics"]


def _wrap(agent: AgentName, node: AgentNode, deps: AgentDeps) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Give every node the deps, a timeout, and a guarantee that it cannot raise.

    The agent node contract already says agents never raise. This is the belt to
    that braces: a teammate's node that does raise degrades this one query into a
    partial result instead of taking down the whole graph invocation.
    """

    async def run(state: AgentState) -> dict[str, Any]:
        # `trace.node` both times the agent and attributes everything emitted
        # beneath it — a Cypher query from three layers down inside kg/client.py
        # carries this agent's name rather than arriving unattributed in the
        # middle of a five-way fan-out.
        with trace.node(agent):
            try:
                patch = await asyncio.wait_for(node(state, deps), timeout=NODE_TIMEOUT_S)
            except TimeoutError:
                reason = f"exceeded its {NODE_TIMEOUT_S:.0f}s slice of the response-time budget"
                log.warning("%s timed out", agent)
                trace.emit("agent_result", status="timeout", error=reason)
                return {
                    "agent_outputs": {agent: failed_output(agent, reason)},
                    "errors": [f"{agent}: {reason}"],
                    "degraded": True,
                }
            except Exception as exc:  # noqa: BLE001 - one node must not fail the invocation
                log.exception("%s raised", agent)
                trace.emit("agent_result", status="failed", error=str(exc))
                return {
                    "agent_outputs": {agent: failed_output(agent, str(exc))},
                    "errors": [f"{agent}: {exc}"],
                    "degraded": True,
                }

            output = (patch.get("agent_outputs") or {}).get(agent, {})
            trace.emit(
                "agent_result",
                status="declined" if output.get("error") else "ok",
                confidence=round(float(output.get("confidence", 0.0)), 3),
                figures=len(output.get("figures", {}) or {}),
                evidence_count=len(output.get("evidence", []) or []),
                degraded=bool(output.get("degraded", False)),
                error=output.get("error"),
            )
            return patch

    return run


async def _plan_steps(query: str, deps: AgentDeps) -> tuple[list[str], str]:
    """The plan to show, never a reason to fail.

    Wrapped rather than called directly because `planner.plan` is the one thing
    in this node whose only job is presentation — a query must still be answered
    if it raises.

    **Skipped entirely when nothing is listening.** The plan exists to be
    streamed; `trace.emit("thought", ...)` is a no-op without a sink, so on
    `POST /api/query` and in `eval/harness.py` this call was paid for and its
    result discarded — an extra LLM call per query, and extra concurrent load on
    the client during the one node whose output decides which agents run.
    `trace.active()` is the module's own answer to "is anyone listening", and it
    is what keeps the non-streaming path exactly as expensive as it was before
    the plan existed.
    """
    if not trace.active():
        return [], "none"
    try:
        return await planner.plan(query, deps.llm)
    except Exception:  # noqa: BLE001 - a plan must never fail a query
        log.warning("planner raised; answering without a plan", exc_info=True)
        return [], "none"


async def _decide_route(state: AgentState, deps: AgentDeps, use_llm: bool) -> RouteDecision:
    if not use_llm:
        return keyword_route(state["query"])
    try:
        return await llm_route(state["query"], deps.llm)
    except Exception as exc:  # noqa: BLE001 - routing must never be the thing that fails
        log.warning("llm routing raised, falling back to keyword: %s", exc)
        decision = keyword_route(state["query"])
        decision.method = "llm->keyword"
        return decision


__all__ = ["NODE_TIMEOUT_S", "AgentNode", "agent_registry", "build_graph"]
