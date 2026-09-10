"""Implements SRS 3.1.1-3.1.4 and 3.9.3 — POST /api/query.

One endpoint, one graph invocation, one merged answer with its confidence and
evidence. Async throughout, because SRS 3.4.2's 50 concurrent users rests
entirely on the server not blocking while an agent waits on Neo4j or the LLM.

**Scope note.** This was M2's seed of `ceynex/api/`; the admin routes (SRS 3.5.4)
and the help content are still M3's and still not pre-empted here. Auth (SRS
3.1.11) and query history (SRS 3.5.2) are now wired in, but deliberately as an
*optional* dependency — a request with no (or an invalid) token still answers
normally, it just isn't attributed to anyone. The endpoint stays open rather
than gated, recorded in docs/DEFERRED.md.

Rate limiting (SRS 3.4.6) applies here and only here; `ceynex/api/rate_limit.py`
explains the store, the window and why it fails open.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from ceynex import settings
from ceynex.agents.common import parse_intent
from ceynex.api import history, rate_limit
from ceynex.api.deps import Runtime, get_runtime
from ceynex.api.routes.auth import TokenPayload, get_optional_user
from ceynex.api.schemas import AnswerGraph, QueryRequest, QueryResponse
from ceynex.contracts import new_state
from ceynex.kg import queries as kg_queries
from ceynex.kg import subgraph as kg_subgraph
from ceynex.orchestrator.confidence import confidence_band
from ceynex.orchestrator.merger import (
    agents_used_from_outputs,
    no_topic_recognized,
    unanswered_from_outputs,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

# SRS 3.4.1 allows 20s for a cross-sector answer. This is the outer wall: past
# it something is wrong that per-node timeouts did not catch, and a caller
# waiting forever is worse than a clear failure.
REQUEST_TIMEOUT_S = 25.0

# What the drawable graph may add to a request that has already been answered.
# Small on purpose: the answer is the product and the picture is an illustration
# of it, so the illustration does not get to spend the response-time budget. Past
# this the request returns without a graph rather than late with one.
GRAPH_BUDGET_S = 3.0


async def enforce_rate_limit(
    http_request: Request,
    user: TokenPayload | None = Depends(get_optional_user),  # noqa: B008
) -> None:
    """SRS 3.4.6. A dependency rather than middleware, so it applies to this
    endpoint alone — logging in, reading your own history and the admin routes
    are not what the requirement is about, and throttling them would be a
    different decision needing its own justification.

    Runs before the graph is invoked: the whole point is not to pay for the
    fan-out. `get_optional_user` is shared with the handler below, and FastAPI
    resolves a dependency once per request, so the token is not verified twice.
    """
    config = settings.load_config("api").get("rate_limit", {})
    if not config.get("enabled", True):
        return

    limit = int(config.get("query_per_minute", 30))
    window_s = int(config.get("window_seconds", 60))
    identity = rate_limit.identity_of(
        user.email if user else None,
        rate_limit.client_ip(http_request),
    )

    decision = await _window().check(identity, limit, window_s)
    if decision.allowed:
        return

    log.info("rate limit hit by %s (%d/%ds)", identity, limit, window_s)
    raise HTTPException(
        status_code=429,
        detail=(
            f"rate limit exceeded: at most {limit} queries per {window_s} seconds. "
            f"Try again in {decision.retry_after_s}s."
        ),
        headers={"Retry-After": str(decision.retry_after_s)},
    )


_window_singleton: rate_limit.Window | None = None


def _window() -> rate_limit.Window:
    """Built on first use, not at import: `build_window` reads REDIS_URL, and
    at import time the app may not have loaded its environment yet."""
    global _window_singleton  # noqa: PLW0603 - one process-lifetime object
    if _window_singleton is None:
        _window_singleton = rate_limit.build_window()
    return _window_singleton


def set_window(window: rate_limit.Window | None) -> None:
    """Test seam. Production never calls this."""
    global _window_singleton  # noqa: PLW0603
    _window_singleton = window


@router.post("/api/query", response_model=QueryResponse, dependencies=[Depends(enforce_rate_limit)])
async def submit_query(
    request: QueryRequest,
    runtime: Runtime = Depends(get_runtime),  # noqa: B008 - FastAPI's dependency idiom
    user: TokenPayload | None = Depends(get_optional_user),  # noqa: B008
) -> QueryResponse:
    started = time.perf_counter()
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    try:
        final = await runtime.graph.ainvoke(new_state(query, user_id="anonymous"))
    except Exception as exc:  # noqa: BLE001 - the graph should never raise; if it does, say so
        log.exception("graph invocation failed")
        raise HTTPException(status_code=500, detail=f"orchestration failed: {exc}") from exc

    outputs = final.get("agent_outputs", {})
    confidence = float(final.get("final_confidence", 0.0))

    # Before the response is built, so `elapsed_ms` below counts it. The graph
    # is time the caller actually waited; excluding it would make the number
    # that SRS 3.4.1 is measured against quietly optimistic.
    graph = await _answer_graph(runtime, query, final)

    response = QueryResponse(
        answer=final.get("final_answer", ""),
        confidence=confidence,
        confidence_band=confidence_band(confidence),
        agents_used=agents_used_from_outputs(final),
        evidence=final.get("merged_evidence", []),
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
        # Previously just hard failures (`out.get("error")`), which silently
        # dropped declines and out-of-scope notes from the public response
        # even though the prose answer already mentioned them correctly.
        unanswered=unanswered_from_outputs(final),
        graph=graph,
    )

    if user is not None:
        history.record(
            user_email=user.email,
            query=query,
            answer=response.answer,
            confidence=response.confidence,
            degraded=response.degraded,
        )

    return response


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
    `orchestrator/grounding.py` exists to catch in prose, and the same
    suppression `forecast=` already applies two lines up in the caller.

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
    the precedence `export_analytics` and `trade_economics` both use and the
    thing the 2026-08-26 bug in that query's docstring was about. Drawing a
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
