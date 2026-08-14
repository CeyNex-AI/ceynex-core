"""Minimal LangGraph wiring for the apparel_manufacturing node (SRS 3.6.4).

STUB — the real orchestrator (routing to 2-3 agents in parallel, merging
their outputs, cross-agent confidence aggregation via
`ceynex/orchestrator/confidence.py`) is core-systems scope and isn't built
here. This file exists only so `ceynex/agents/apparel_manufacturing.py` can
run as an actual LangGraph node end to end — per
`.claude/commands/new-agent.md`'s step 3 ("register the node ... if not
already wired") — not to be the final graph. Wires exactly one node: no
router, no fan-out, no merge step. Replace, don't extend, once the real
multi-agent graph lands (SRS 3.6.4 fixes the agent set at five; this graph
only knows about one of them).
"""

from langgraph.graph import END, StateGraph

from ceynex.agents.apparel_manufacturing import apparel_manufacturing_node
from ceynex.contracts.state import AgentState


def build_apparel_only_graph():
    """A trivial one-node graph: entry -> apparel_manufacturing -> END."""
    graph = StateGraph(AgentState)
    graph.add_node("apparel_manufacturing", apparel_manufacturing_node)
    graph.set_entry_point("apparel_manufacturing")
    graph.add_edge("apparel_manufacturing", END)
    return graph.compile()
