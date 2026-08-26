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
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from ceynex.api import history
from ceynex.api.deps import Runtime, get_runtime
from ceynex.api.routes.auth import TokenPayload, get_optional_user
from ceynex.api.schemas import QueryRequest, QueryResponse
from ceynex.contracts import new_state
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


@router.post("/api/query", response_model=QueryResponse)
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
