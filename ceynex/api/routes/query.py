"""Implements SRS 3.1.1-3.1.4 and 3.9.3 — POST /api/query.

One endpoint, one graph invocation, one merged answer with its confidence and
evidence. Async throughout, because SRS 3.4.2's 50 concurrent users rests
entirely on the server not blocking while an agent waits on Neo4j or the LLM.

**The orchestration itself lives in `ceynex/api/query_runner.py`**, because
`POST /api/chat/stream` runs exactly the same pipeline and only differs in how it
reports progress. Two implementations of "answer a question" would mean the
30-question evaluation measures one of them and users get the other.

**Sign-in required** (SRS 3.1.11, "account based authentication for all users
prior to query submission"; FR-ACC-01). Until 2026-10 the token was optional and
an anonymous query answered without being recorded; that left the paid LLM path
open to anyone, bounded only per address. A missing or invalid token is now a
401, before the rate limit or the graph run. An API key (`Bearer ck_...`) is a
token like any other. Query history (SRS 3.5.2) therefore records every query.

Rate limiting (SRS 3.4.6) applies here and to `/api/chat/*`;
`ceynex/api/rate_limit.py` explains the store, the window and why it fails open.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from ceynex import settings
from ceynex.api import rate_limit
from ceynex.api.deps import Runtime, get_runtime
from ceynex.api.query_runner import OrchestrationError, run_query
from ceynex.api.routes.auth import TokenPayload, require_user
from ceynex.api.schemas import QueryRequest, QueryResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


async def enforce_rate_limit(
    http_request: Request,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> None:
    """SRS 3.4.6. A dependency rather than middleware, so it applies to the
    endpoints the requirement is about — query submission and the chat surface
    that invokes the identical fan-out — and not to logging in, reading your own
    history or the admin routes, which would be a different decision needing its
    own justification.

    Runs before the graph is invoked: the whole point is not to pay for the
    fan-out. `require_user` is shared with the handler below, and FastAPI
    resolves a dependency once per request, so the token is not verified twice
    and an anonymous call is refused before it is counted.
    """
    config = settings.load_config("api").get("rate_limit", {})
    if not config.get("enabled", True):
        return

    limit = int(config.get("query_per_minute", 30))
    window_s = int(config.get("window_seconds", 60))
    identity = rate_limit.identity_of(user.email, rate_limit.client_ip(http_request))

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
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> QueryResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    try:
        outcome = await run_query(runtime, query, user_email=user.email)
    except OrchestrationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = outcome.response
    # Deferred from Phase 1 deliberately: the ledger recorded this from the
    # start, but adding a field to a shape `ceynex-web` binds to before there
    # was anywhere to show it would have been a contract change for nothing.
    response.usage = outcome.usage.as_summary()
    return response
