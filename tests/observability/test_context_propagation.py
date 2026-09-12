"""The canary for `ceynex/observability/context.py`'s load-bearing assumption.

The whole trace bus rests on one thing: a `ContextVar` bound before
`graph.ainvoke()` is still visible inside agent nodes that LangGraph runs
*concurrently*. That is true because `langgraph/pregel/_executor.py` calls
`copy_context()` per dispatched node and threads it into `loop.create_task(...,
context=...)` — but those are private, underscore-prefixed modules, not a
documented public guarantee.

So this file exists to fail loudly on a LangGraph upgrade rather than have the
trace quietly go empty in production. `pyproject.toml` pins `langgraph>=1.2,<1.3`
and points here. If this breaks, the fallback is LangGraph's public
`RunnableConfig` mechanism — see the module docstring in `context.py`.

It also pins the two rules that follow from *copy* semantics, because getting
either wrong produces a bus that works in a unit test and loses half its events
under a real fan-out:

1. a binding installed before `ainvoke()` reaches every parallel node;
2. a `set()` *inside* a node does not escape it — which is why the sink must be
   a mutable object appended to, never a value reassigned.
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from ceynex.observability import context, trace


class _State(TypedDict):
    query: str
    seen: Annotated[list, operator.add]


def _graph(node_names: tuple[str, ...]):
    """A miniature of the real graph: one router, a parallel fan-out, one merge."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(_State)

    async def route(state: _State) -> dict[str, Any]:
        return {}

    def make(name: str):
        async def run(state: _State) -> dict[str, Any]:
            # A real await, so the tasks genuinely interleave rather than each
            # running to completion before the next is scheduled.
            await asyncio.sleep(0.01)
            with trace.node(name):
                trace.emit("kg_query", cypher=f"MATCH ({name})", row_count=1)
            return {"seen": [name]}

        return run

    async def merge(state: _State) -> dict[str, Any]:
        with trace.node("merge"):
            trace.emit("merge_done")
        return {"seen": ["merge"]}

    builder.add_node("route", route)
    for name in node_names:
        builder.add_node(name, make(name))
    builder.add_node("merge", merge)

    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", lambda s: list(node_names), list(node_names))
    for name in node_names:
        builder.add_edge(name, "merge")
    builder.add_edge("merge", END)
    return builder.compile()


async def test_sink_reaches_every_parallel_node() -> None:
    """The canary. If this fails, the trace bus is silently broken."""
    names = ("alpha", "beta", "gamma")
    sink = trace.TraceSink(request_id="canary", loop=asyncio.get_running_loop())
    token = context.install(context.RequestObservability(request_id="canary", trace=sink))
    try:
        await _graph(names).ainvoke({"query": "x", "seen": []})
    finally:
        context.reset(token)

    emitted = {event.node for event in sink.history if event.kind == "kg_query"}
    assert emitted == set(names), f"nodes that lost the sink: {set(names) - emitted}"
    assert any(event.kind == "merge_done" for event in sink.history)


async def test_events_are_attributed_to_the_node_that_caused_them() -> None:
    """A Cypher query emitted inside an agent carries that agent's name.

    This is what makes the timeline readable: without it every `kg_query` from a
    five-way fan-out is an unattributed row.
    """
    sink = trace.TraceSink(request_id="attrib", loop=asyncio.get_running_loop())
    token = context.install(context.RequestObservability(request_id="attrib", trace=sink))
    try:
        await _graph(("alpha", "beta")).ainvoke({"query": "x", "seen": []})
    finally:
        context.reset(token)

    for event in sink.history:
        if event.kind == "kg_query":
            assert event.payload["cypher"] == f"MATCH ({event.node})"


async def test_seq_is_strictly_increasing_across_concurrent_nodes() -> None:
    """Ordering must survive several nodes emitting at once, or the replayed
    trace is a different story from the live one."""
    sink = trace.TraceSink(request_id="seq", loop=asyncio.get_running_loop())
    token = context.install(context.RequestObservability(request_id="seq", trace=sink))
    try:
        await _graph(("alpha", "beta", "gamma")).ainvoke({"query": "x", "seen": []})
    finally:
        context.reset(token)

    seqs = [event.seq for event in sink.history]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs)), "duplicate seq — the counter is not atomic"


async def test_a_set_inside_a_node_does_not_escape_it() -> None:
    """Pins *why* the sink is a mutable object rather than a reassigned value.

    Context is copied into each task, so this is a language guarantee we depend
    on rather than a bug — but it is the exact mistake that would make the bus
    work in a single-node test and lose events under fan-out.
    """
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(_State)

    async def mutate(state: _State) -> dict[str, Any]:
        context.install(context.RequestObservability(request_id="inner"))
        return {"seen": ["mutate"]}

    builder.add_node("mutate", mutate)
    builder.add_edge(START, "mutate")
    builder.add_edge("mutate", END)

    outer = context.RequestObservability(request_id="outer")
    token = context.install(outer)
    try:
        await builder.compile().ainvoke({"query": "x", "seen": []})
        assert context.current() is outer
    finally:
        context.reset(token)


async def test_emit_is_a_no_op_with_no_sink_installed() -> None:
    """The guarantee that keeps demo.py, eval/harness.py and ~1,220 tests intact."""
    assert context.current() is None
    assert trace.active() is False
    trace.emit("kg_query", cypher="MATCH (n)")  # must not raise

    with trace.node("orphan"):
        trace.emit("anything")


async def test_usage_collection_works_without_a_trace_sink() -> None:
    """`POST /api/query` has nowhere to stream events but its spend still counts.

    Usage and tracing are independent: one can be on with the other off.
    """
    obs = context.RequestObservability(request_id="usage-only", trace=None)
    token = context.install(obs)
    try:
        context.record_llm_call(
            context.LLMCall(
                role="merge", model="gpt-4o", provider="openai",
                tokens_in=100, tokens_out=50, cost_usd=0.001,
            )
        )
        trace.emit("kg_query", cypher="MATCH (n)")  # no sink — silently ignored
    finally:
        context.reset(token)

    assert obs.usage.tokens_in == 100
    assert obs.usage.as_summary()["calls"] == 1


async def test_queue_drops_rather_than_blocking_when_full() -> None:
    """An answer must never wait on its own progress reporting."""
    sink = trace.TraceSink(
        request_id="flood", loop=asyncio.get_running_loop(), max_queued=4
    )
    token = context.install(context.RequestObservability(request_id="flood", trace=sink))
    try:
        for index in range(20):
            trace.emit("kg_query", index=index)
    finally:
        context.reset(token)

    assert sink.dropped == 16
    assert len(sink.history) == 20, "history must keep every event even when the queue drops"


async def test_long_payload_text_is_clipped() -> None:
    sink = trace.TraceSink(request_id="clip", loop=asyncio.get_running_loop())
    token = context.install(context.RequestObservability(request_id="clip", trace=sink))
    try:
        trace.emit("kg_query", cypher="X" * 5000)
    finally:
        context.reset(token)

    assert len(sink.history[0].payload["cypher"]) <= trace.MAX_TEXT + 2
