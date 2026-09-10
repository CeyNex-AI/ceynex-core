"""The trace must describe what actually ran. This is the file that proves it.

The whole project rests on one claim: every figure traces to its source. A
progress timeline is a claim of the same kind — it asserts *this query ran, and
took this long* — so a step that did not happen is the same category of defect as
a figure that was never sourced, and a marker diffing the trace against the logs
would find it.

The brief that produced this feature allowed for faking the activity display.
These assertions are what makes that unnecessary and, more usefully, what stops
it from creeping back in later: every `kg_query` event has to correspond to a
Cypher string that a real agent really put into `Evidence.detail`.
"""

from __future__ import annotations

import asyncio

from ceynex.agents.common import AgentDeps
from ceynex.contracts import new_state
from ceynex.llm import FakeLLMClient
from ceynex.observability import context, trace
from ceynex.orchestrator.graph import build_graph

ROWS = [
    {"partner": "USA", "value": 4_200_000.0, "year": 2025},
    {"partner": "Germany", "value": 1_100_000.0, "year": 2025},
]


class _Record:
    def __init__(self, data: dict) -> None:
        self._data = data

    def data(self) -> dict:
        return self._data


class _Result:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __aiter__(self):
        async def gen():
            for row in self._rows:
                yield _Record(row)

        return gen()


class _StubSession:
    """Stands in for a neo4j session, recording what the driver was actually sent."""

    def __init__(self, recorder: list[str]) -> None:
        self._recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, cypher, params):
        self._recorder.append(cypher)
        # Shaped to satisfy `latest_observation_year`, which every agent calls
        # first; anything else gets the partner rows. Agents that want a column
        # neither provides decline, which is a valid path to trace.
        rows = [{"latest_year": 2025}] if "latest_year" in cypher else list(ROWS)
        return _Result(rows)


class _StubDriver:
    def __init__(self, recorder: list[str]) -> None:
        self._recorder = recorder

    def session(self, database=None):
        return _StubSession(self._recorder)


def _real_client_with_stub_driver() -> tuple[object, list[str]]:
    """A genuine `KnowledgeGraphClient` over a fake driver.

    Deliberately not a hand-written fake KG: the emit being tested lives inside
    `kg/client.py::run`, so a stand-in that never calls it would let this whole
    file pass while the real path emitted nothing. That is exactly what happened
    on the first draft of these tests.
    """
    from ceynex.kg.client import KnowledgeGraphClient

    recorder: list[str] = []
    client = KnowledgeGraphClient.__new__(KnowledgeGraphClient)
    client._driver = _StubDriver(recorder)  # noqa: SLF001 - standing in for the pool
    client._database = None  # noqa: SLF001
    client._uri = "bolt://test"  # noqa: SLF001
    return client, recorder


def _deps(kg) -> AgentDeps:
    return AgentDeps(kg=kg, llm=FakeLLMClient(available=False))


async def _run_traced(query: str) -> tuple[dict, trace.TraceSink, list[str]]:
    kg, executed = _real_client_with_stub_driver()
    graph = build_graph(_deps(kg), use_llm_router=False)
    sink = trace.TraceSink(request_id="truthful", loop=asyncio.get_running_loop())
    token = context.install(context.RequestObservability(request_id="truthful", trace=sink))
    try:
        final = await graph.ainvoke(new_state(query, "u"))
    finally:
        context.reset(token)
    return final, sink, executed


async def test_every_traced_cypher_is_a_query_that_actually_ran() -> None:
    """No invented steps: every traced query was really sent to the driver.

    Guarded against passing vacuously — an empty trace makes the subset check
    trivially true, which is precisely how a broken emit path would slip through.
    """
    _, sink, executed = await _run_traced("cinnamon exports to Germany in 2025")

    traced = [event.payload["cypher"] for event in sink.history if event.kind == "kg_query"]
    assert traced, "no KG activity traced — the emit path is not being exercised"
    for cypher in traced:
        assert cypher in executed, f"traced a query that never ran: {cypher[:80]}"


async def test_every_query_that_ran_was_traced() -> None:
    """And the converse: no silent steps either.

    A timeline that omits work is as misleading as one that invents it — it makes
    the system look like it did less than it did, and hides a slow query.
    """
    _, sink, executed = await _run_traced("cinnamon exports to Germany in 2025")

    traced = {event.payload["cypher"] for event in sink.history if event.kind == "kg_query"}
    for cypher in executed:
        assert cypher in traced, f"ran a query that was never traced: {cypher[:80]}"


async def test_the_client_emits_the_cypher_it_executed_verbatim() -> None:
    """The string in the event is the string that was sent, not a paraphrase.

    `kg/client.py::run` returns `(rows, cypher)` precisely so the caller can put
    the executed text into `Evidence.detail`. The trace makes the same promise
    and must keep it the same way.
    """
    from ceynex.kg.client import KnowledgeGraphClient

    cypher = "MATCH (c:Commodity {name: $item})-[r:EXPORTS_TO]->(p:Country) RETURN p.name, r.value"

    class _Result:
        def __aiter__(self):
            async def gen():
                for row in ROWS:
                    yield _Record(row)

            return gen()

    class _Record:
        def __init__(self, data):
            self._data = data

        def data(self):
            return self._data

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run(self, sent, params):
            assert sent == cypher
            return _Result()

    class _Driver:
        def session(self, database=None):
            return _Session()

    client = KnowledgeGraphClient.__new__(KnowledgeGraphClient)
    client._driver = _Driver()  # noqa: SLF001 - standing in for the neo4j pool
    client._database = None  # noqa: SLF001
    client._uri = "bolt://test"  # noqa: SLF001

    sink = trace.TraceSink(request_id="verbatim", loop=asyncio.get_running_loop())
    token = context.install(context.RequestObservability(request_id="verbatim", trace=sink))
    try:
        rows, returned = await client.run(cypher, {"item": "cinnamon"})
    finally:
        context.reset(token)

    events = [event for event in sink.history if event.kind == "kg_query"]
    assert len(events) == 1
    assert events[0].payload["cypher"] == cypher == returned
    assert events[0].payload["row_count"] == len(rows) == 2
    assert events[0].payload["status"] == "ok"
    # Bind parameters are shown, because "WHERE c.name = $item" without knowing
    # that $item was "cinnamon" does not explain the answer it produced.
    assert "item='cinnamon'" in events[0].payload["params"]


async def test_a_traced_duration_is_measured_not_asserted() -> None:
    """Durations come from a real clock around a real call."""
    _, sink, _ = await _run_traced("cinnamon export trend")

    timed = [e for e in sink.history if "elapsed_ms" in e.payload]
    assert timed, "nothing reported a duration"
    for event in timed:
        assert isinstance(event.payload["elapsed_ms"], float)
        assert event.payload["elapsed_ms"] >= 0.0


async def test_events_are_attributed_to_the_agent_that_caused_them() -> None:
    """A Cypher query emitted inside a fan-out carries the agent's name.

    Without this the timeline for a five-way fan-out is a flat list of queries
    with no way to tell which analysis asked for which.
    """
    _, sink, _ = await _run_traced("how would a 10% EU tariff affect cinnamon exports")

    kg_events = [e for e in sink.history if e.kind == "kg_query"]
    assert kg_events, "no KG activity traced"
    for event in kg_events:
        assert event.node, f"unattributed query: {event.payload['cypher'][:60]}"
        assert event.node not in ("route", "merge")


async def test_the_route_event_reports_the_route_actually_taken() -> None:
    final, sink, _ = await _run_traced("cinnamon export trend")

    routed = [e for e in sink.history if e.kind == "route"]
    assert len(routed) == 1
    assert routed[0].payload["route"] == list(final["route"])
    assert routed[0].payload["sectors"] == list(final["sectors"])


async def test_the_merge_event_reports_the_confidence_that_was_served() -> None:
    """The number in the trace and the number in the answer are the same number."""
    final, sink, _ = await _run_traced("cinnamon export trend")

    merged = [e for e in sink.history if e.kind == "merge"]
    assert len(merged) == 1
    assert merged[0].payload["confidence"] == round(float(final["final_confidence"]), 3)
    assert merged[0].payload["evidence_count"] == len(final["merged_evidence"])


async def test_every_routed_agent_reports_a_result_exactly_once() -> None:
    """No agent silently missing from the timeline, and none reported twice."""
    final, sink, _ = await _run_traced("cinnamon export trend")

    reported = [e.node for e in sink.history if e.kind == "agent_result"]
    assert sorted(reported) == sorted(final["route"])


async def test_a_declining_agent_is_traced_as_declining_not_as_success() -> None:
    """An honest refusal must look like one in the timeline too.

    The system's declines are a feature (SAD §4.1) — reporting one as a green
    tick would be the timeline lying about the thing the project is proudest of.
    """

    class DeadKG:
        async def run(self, cypher, params=None):
            from ceynex.kg.client import KnowledgeGraphUnavailableError

            raise KnowledgeGraphUnavailableError("neo4j is down")

        async def run_one(self, cypher, params=None):
            from ceynex.kg.client import KnowledgeGraphUnavailableError

            raise KnowledgeGraphUnavailableError("neo4j is down")

    graph = build_graph(_deps(DeadKG()), use_llm_router=False)
    sink = trace.TraceSink(request_id="declining", loop=asyncio.get_running_loop())
    token = context.install(context.RequestObservability(request_id="declining", trace=sink))
    try:
        await graph.ainvoke(new_state("cinnamon export trend", "u"))
    finally:
        context.reset(token)

    results = [e for e in sink.history if e.kind == "agent_result"]
    assert results
    assert all(e.payload["status"] != "ok" for e in results)


async def test_an_untraced_run_produces_the_same_answer_as_a_traced_one() -> None:
    """Observing the system must not change it.

    This is what protects `demo.py`, `eval/harness.py` and the 30-question
    evaluation: if the streaming transport altered what the graph produced, every
    published number would describe a system that no longer exists.
    """
    traced_final, _, _ = await _run_traced("cinnamon export trend")

    plain_kg, _ = _real_client_with_stub_driver()
    graph = build_graph(_deps(plain_kg), use_llm_router=False)
    plain_final = await graph.ainvoke(new_state("cinnamon export trend", "u"))

    assert plain_final["final_answer"] == traced_final["final_answer"]
    assert plain_final["final_confidence"] == traced_final["final_confidence"]
    assert plain_final["merged_evidence"] == traced_final["merged_evidence"]
    assert plain_final["route"] == traced_final["route"]
