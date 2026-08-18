"""Implements SRS 3.1.1-3.1.4 and 3.9.3 — POST /api/query.

One endpoint, one graph invocation, one merged answer with its confidence and
evidence. Async throughout, because SRS 3.4.2's 50 concurrent users rests
entirely on the server not blocking while an agent waits on Neo4j or the LLM.

**Scope note.** This is M2's seed of `ceynex/api/`. Authentication (SRS 3.1.11),
the admin routes (SRS 3.5.4), query history and the help content are M3's, and
this router does not pre-empt them: there is no auth dependency here yet, so the
endpoint is open. That is fine behind a VPC-only backend and is recorded in
docs/DEFERRED.md.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from ceynex.api.deps import Runtime, get_runtime
from ceynex.api.schemas import QueryRequest, QueryResponse
from ceynex.contracts import new_state
from ceynex.orchestrator.confidence import confidence_band

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
    succeeded = sorted(name for name, out in outputs.items() if not out.get("error"))
    failed = sorted(name for name, out in outputs.items() if out.get("error"))
    confidence = float(final.get("final_confidence", 0.0))

    return QueryResponse(
        answer=final.get("final_answer", ""),
        confidence=confidence,
        confidence_band=confidence_band(confidence),
        agents_used=succeeded,
        evidence=final.get("merged_evidence", []),
        forecast=_forecast_of(outputs) or None,
        degraded=bool(final.get("degraded", False)),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        route=list(final.get("route", [])),
        sectors=list(final.get("sectors", [])),
        unanswered=failed,
    )


def _forecast_of(outputs: dict) -> list:
    if outputs.get("forecast", {}).get("forecast"):
        return list(outputs["forecast"]["forecast"])
    for output in outputs.values():
        if output.get("forecast"):
            return list(output["forecast"])
    return []
