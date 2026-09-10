"""One orchestration path, two transports — SRS 3.1.1-3.1.4.

`POST /api/query` returns a whole answer in one JSON body. `POST /api/chat/stream`
returns the same answer, but reports each step as it happens. They must not be
two implementations of "answer a question": the 30-question evaluation
(`make eval`) measures the first, and the day it stops describing the second,
every published number is about a system nobody uses.

So the orchestration lives here and the routes are thin. `routes/query.py` calls
`run_query()` and serialises the result; the streaming route calls the same
function with a `TraceSink` attached and serialises the same result at the end.

**Deliberately in the API layer, not `orchestrator/`.** This builds
`QueryResponse`, an API-layer shape. The layer rule in CLAUDE.md has the API
depending on the orchestrator and never the reverse, and moving this into
`ceynex/orchestrator/` would invert that for the sake of a filename.

**The timeout here is new behaviour.** `routes/query.py` declared
`REQUEST_TIMEOUT_S = 25.0` with a comment calling it "the outer wall", but never
passed it to anything — the real ceiling was nginx's 60s default, which is why
`docs/EVALUATION.md` records a 29s tail rather than a 25s failure. It is applied
below. A request that would previously have run long now fails cleanly at the
declared budget, which is what the constant always claimed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from dataclasses import dataclass

from ceynex.agents.common import evidence_from_web, parse_intent
from ceynex.api import history
from ceynex.api.deps import Runtime
from ceynex.api.schemas import AnswerGraph, QueryResponse
from ceynex.chat import instructions
from ceynex.contracts import new_state
from ceynex.kg import queries as kg_queries
from ceynex.kg import subgraph as kg_subgraph
from ceynex.observability import context as obs
from ceynex.observability import ledger, trace
from ceynex.orchestrator.confidence import confidence_band
from ceynex.orchestrator.merger import (
    agents_used_from_outputs,
    no_topic_recognized,
    unanswered_from_outputs,
)
from ceynex.websearch import safe_search, wants_current_context

log = logging.getLogger(__name__)

# SRS 3.4.1 allows 20s for a cross-sector answer. This is the outer wall: past
# it something is wrong that per-node timeouts did not catch, and a caller
# waiting forever is worse than a clear failure.
REQUEST_TIMEOUT_S = 25.0

# What the drawable graph may add to a request that has already been answered.
# Small on purpose: the answer is the product and the picture is an illustration
# of it, so the illustration does not get to spend the response-time budget. Past
# this the request returns without a graph rather than late with one.
GRAPH_BUDGET_S = 3.0


class OrchestrationError(RuntimeError):
    """The graph itself failed. The route turns this into a 500 or an error frame."""


@dataclass
class QueryOutcome:
    """The answer, plus what it cost to produce and what happened along the way."""

    response: QueryResponse
    request_id: str
    usage: obs.RequestLLMUsage
    trace_events: list[trace.TraceEvent]
    #: The `query_history` row this answer was recorded under, or None when it
    #: was not recorded (anonymous, or Postgres unreachable). A chat turn stores
    #: it so its save star links to the row the History panel shows.
    history_id: int | None = None


async def run_query(
    runtime: Runtime,
    query: str,
    *,
    user_id: str = "anonymous",
    user_email: str | None = None,
    overall_timeout_s: float = REQUEST_TIMEOUT_S,
    trace_sink: trace.TraceSink | None = None,
    conversation_id: int | None = None,
    record_history: bool = True,
    request_id: str | None = None,
) -> QueryOutcome:
    """Route, fan out, merge, and assemble the response.

    `trace_sink=None` is the `POST /api/query` case: no events go anywhere, but
    LLM spend is still accounted, because usage and tracing are independent.

    `request_id` is supplied by a conversational turn, which mints it before the
    first frame so the reader can resume by it (`api/turn_runner.py`); the
    ledger, the trace and the transcript then all name the turn the same way.
    Absent, a fresh one is minted here, as it always was.
    """
    started = time.perf_counter()
    # Read once per request, before anything installs it, so a single database
    # round trip serves the whole turn rather than one per merge.
    instruction, instruction_on = await instructions.get(user_email)
    observation = obs.RequestObservability(
        trace=trace_sink,
        user_email=user_email,
        conversation_id=conversation_id,
        instruction=instruction if instruction_on else "",
    )
    if request_id:
        observation.request_id = request_id
    token = obs.install(observation)
    try:
        # Started before the graph and gathered after it, so it costs no wall
        # clock at all on a path that already breaches SRS 3.4.1. Never awaited
        # *before* the graph: an answer must not wait on the web.
        web_task = (
            asyncio.create_task(safe_search(runtime.websearch, query))
            if getattr(runtime, "websearch", None) is not None and wants_current_context(query)
            else None
        )

        try:
            final = await asyncio.wait_for(
                runtime.graph.ainvoke(new_state(query, user_id=user_id)),
                timeout=overall_timeout_s,
            )
        except TimeoutError as exc:
            log.warning("query exceeded its %.0fs budget: %s", overall_timeout_s, query[:80])
            if web_task is not None:
                web_task.cancel()
            raise OrchestrationError(
                f"the request exceeded its {overall_timeout_s:.0f}s budget"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - the graph should never raise; if it does, say so
            log.exception("graph invocation failed")
            if web_task is not None:
                web_task.cancel()
            raise OrchestrationError(f"orchestration failed: {exc}") from exc

        # Gathered here, after the graph has finished and merge has already
        # written its prose. Everything about the safety of D14 rests on this
        # ordering rather than on any check — see `evidence_from_web`.
        web_results = []
        if web_task is not None:
            try:
                web_results = await web_task
            except Exception:  # noqa: BLE001 - a web result never fails an answer
                log.warning("web search task failed", exc_info=True)

        response = await _assemble(runtime, query, final, started, web_results, observation)
    finally:
        obs.reset(token)

    history_id: int | None = None
    if record_history and user_email is not None:
        # Off the loop, unlike the inline call this replaces: on the streaming
        # transport a blocking connect stalls the heartbeat, and with two uvicorn
        # workers it stalls every other request on the process too.
        history_id = await asyncio.to_thread(
            history.record,
            user_email=user_email,
            query=query,
            answer=response.answer,
            confidence=response.confidence,
            degraded=response.degraded,
        )

    await ledger.record(
        request_id=observation.request_id,
        user_email=user_email,
        conversation_id=conversation_id,
        calls=observation.usage.calls,
    )

    return QueryOutcome(
        response=response,
        request_id=observation.request_id,
        usage=observation.usage,
        trace_events=list(trace_sink.history) if trace_sink else [],
        history_id=history_id if isinstance(history_id, int) else None,
    )


async def _assemble(
    runtime: Runtime,
    query: str,
    final: dict,
    started: float,
    web_results=(),
    observation: obs.RequestObservability | None = None,
) -> QueryResponse:
    outputs = final.get("agent_outputs", {})
    confidence = float(final.get("final_confidence", 0.0))

    # Before the response is built, so `elapsed_ms` below counts it. The graph
    # is time the caller actually waited; excluding it would make the number
    # that SRS 3.4.1 is measured against quietly optimistic.
    graph = await _answer_graph(runtime, query, final)

    return QueryResponse(
        answer=final.get("final_answer", ""),
        confidence=confidence,
        confidence_band=confidence_band(confidence),
        # SRS 3.1.4's working, when there is any. Absent for an out-of-scope
        # question, whose score is a fixed floor rather than a computation.
        confidence_breakdown=(observation.confidence_breakdown if observation else None),
        agents_used=agents_used_from_outputs(final),
        # Merged evidence first, web last and clearly separate. By the time this
        # runs, merge has written its prose, grounding has checked it and
        # confidence has been computed — so a web result cannot have influenced
        # any of the three. That is D14's entire safety argument, and it is a
        # property of *when* this line runs.
        evidence=[*final.get("merged_evidence", []), *_web_evidence(web_results)],
        # A routed agent's forecast is noise, not an answer, for a question
        # that named nothing CeyNex covers -- same suppression as agents_used
        # and unanswered below (see merger.no_topic_recognized's docstring).
        forecast=(_forecast_of(outputs) if not no_topic_recognized(final) else []) or None,
        degraded=bool(final.get("degraded", False)),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        route=list(final.get("route", [])),
        sectors=list(final.get("sectors", [])),
        # Same "could not be answered" list merge() itself uses -- hard
        # failures, honest low-confidence declines, and out-of-scope gaps.
        unanswered=unanswered_from_outputs(final),
        graph=graph,
    )


def _web_evidence(results) -> list:
    """Web hits as `Evidence`, or nothing at all."""
    return [
        evidence_from_web(result.title, result.snippet, result.url, period=result.published)
        for result in results or ()
    ]


def _forecast_of(outputs: dict) -> list:
    if outputs.get("forecast", {}).get("forecast"):
        return list(outputs["forecast"]["forecast"])
    for output in outputs.values():
        if output.get("forecast"):
            return list(output["forecast"])
    return []


# --- the drawable graph (SRS 3.1.4) -----------------------------------------


async def _answer_graph(runtime: Runtime, query: str, final: dict) -> AnswerGraph | None:
    """The subgraph to draw beside this answer, or None.

    Three things have to hold, and the first is the important one.

    **It must be an answer the graph actually produced.** A drawing is a claim
    about where a number came from, so it is built only when the merged evidence
    carries a `KG` entry, and never for a question that named nothing CeyNex
    covers. A node-link diagram beside a model-derived or out-of-scope answer
    would assert a provenance that isn't there — the same failure
    `orchestrator/grounding.py` exists to catch in prose.

    **It must not cost the answer.** Everything below is inside one budget and
    one `except`. A graph that is slow, broken, or asked of a dead Neo4j returns
    None, and the user gets the answer without a picture.

    **It must be the same graph the agents used.** See `_graph_subject`.
    """
    evidence = final.get("merged_evidence") or []
    if not any(item.get("source_id") == "KG" for item in evidence):
        return None
    if no_topic_recognized(final):
        return None

    try:
        return await asyncio.wait_for(
            _build_answer_graph(runtime, query, final), timeout=GRAPH_BUDGET_S
        )
    except TimeoutError:
        log.info("graph build exceeded %.1fs; answering without it", GRAPH_BUDGET_S)
        return None
    except Exception:  # noqa: BLE001 - an illustration must never fail an answer
        log.warning("graph build failed; answering without it", exc_info=True)
        return None


async def _build_answer_graph(runtime: Runtime, query: str, final: dict) -> AnswerGraph | None:
    item, year = await _graph_subject(runtime, query)
    if item is None:
        return None

    built = await kg_subgraph.build_answer_subgraph(
        runtime.kg, item=item, year=year, sectors=tuple(final.get("sectors", []))
    )
    if built.is_empty:
        # An empty panel is worse than none: it reads as "the graph knows
        # nothing about this", when the truth is that this answer's figures came
        # from somewhere the drawing does not cover.
        return None

    return AnswerGraph(
        nodes=[dataclasses.asdict(node) for node in built.nodes],
        edges=[dataclasses.asdict(edge) for edge in built.edges],
        focus_id=built.focus_id,
        queries=built.queries,
        truncated=built.truncated,
    )


async def _graph_subject(runtime: Runtime, query: str) -> tuple[str | None, int | None]:
    """What to centre the drawing on, derived the way the agents derive it.

    `AgentState` is a frozen contract with nowhere to put the item and year an
    agent settled on, so this re-derives them rather than reading them back. That
    is only sound if it derives them *identically*: `parse_intent` is the same
    keyword parser `ceynex/agents/common.py` gives every agent, and the year
    falls back to `latest_observation_year(item)` — scoped to the item, which is
    the precedence `export_analytics` and `trade_economics` both use. Drawing a
    different year than the one analysed would put a picture that disagrees with
    the prose right next to it.
    """
    intent = parse_intent(query)
    if intent.item is None:
        return None, None
    if intent.year is not None:
        return intent.item, intent.year

    row, _ = await runtime.kg.run_one(*kg_queries.latest_observation_year(intent.item))
    latest = (row or {}).get("latest_year")
    return intent.item, int(latest) if latest is not None else None


__all__ = [
    "GRAPH_BUDGET_S",
    "REQUEST_TIMEOUT_S",
    "OrchestrationError",
    "QueryOutcome",
    "run_query",
]
