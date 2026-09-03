"""Implements SRS 3.1.4 and 3.1.6 — GET /api/graph/expand.

`POST /api/query` returns the subgraph behind an answer. This is what happens
when the user clicks a node in it: one hop out, so the drawing can be walked
rather than only looked at.

**One hop, always.** There is no depth parameter to raise. The bound is the
shape of the endpoint rather than a default someone can pass past, because a
graph traversal with a caller-supplied depth is an unbounded query wearing a
number — and this endpoint is reachable without a token, exactly like
`POST /api/query` is.

**The node id is a semantic key, not a database id.** `Country:USA` is the label
plus its schema.cypher uniqueness property, validated against
`kg/queries.py::NODE_KEYS` before anything reaches Neo4j. Cypher cannot
parameterize a label, so that allowlist is what stands between a query string
and the query's own structure; `kg/queries.py::neighbours` explains the
substitution it permits.

**Always 200 when the graph is merely unreachable.** A dead Neo4j returns an
empty fragment and the canvas keeps what it is already showing. A 5xx here would
make a working answer page look broken over a click that was optional — the same
posture `routes/news.py` takes, and for the same reason. A *malformed* node id is
different: that is a caller bug, and it gets a 422.
"""

from __future__ import annotations

import dataclasses
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ceynex import settings
from ceynex.api import rate_limit
from ceynex.api.deps import Runtime, get_runtime
from ceynex.api.routes.auth import TokenPayload, get_optional_user
from ceynex.api.schemas import GraphFragment
from ceynex.kg import subgraph as kg_subgraph

log = logging.getLogger(__name__)

router = APIRouter(tags=["graph"])


# --- rate limiting --------------------------------------------------------

_window_singleton: rate_limit.Window | None = None


def _window() -> rate_limit.Window:
    """Built on first use, not at import — `build_window` reads REDIS_URL."""
    global _window_singleton  # noqa: PLW0603 - one process-lifetime object
    if _window_singleton is None:
        _window_singleton = rate_limit.build_window()
    return _window_singleton


def set_window(window: rate_limit.Window | None) -> None:
    """Test seam. Production never calls this."""
    global _window_singleton  # noqa: PLW0603
    _window_singleton = window


async def enforce_graph_rate_limit(
    http_request: Request,
    user: TokenPayload | None = Depends(get_optional_user),  # noqa: B008
) -> None:
    """SRS 3.4.6's shape, on this endpoint's own allowance.

    Prefixed `graph:` for the reason `routes/news.py` prefixes `news:`:
    `rate_limit.KEY_PREFIX` is shared by every `Window` built from that module,
    so an unprefixed identity would spend the same Redis budget as
    `POST /api/query` — and exploring a graph a few clicks deep would then eat
    the allowance for asking questions.

    More generous than the query limit, not less. A click is one small read with
    no LLM and no fan-out behind it, and exploring is a burst activity: a user
    following a chain of partners will fire several in the seconds a single
    query takes to answer.
    """
    config = settings.load_config("api").get("graph_rate_limit", {})
    if not config.get("enabled", True):
        return

    limit = int(config.get("expand_per_minute", 120))
    window_s = int(config.get("window_seconds", 60))
    identity = "graph:" + rate_limit.identity_of(
        user.email if user else None,
        http_request.client.host if http_request.client else None,
    )

    decision = await _window().check(identity, limit, window_s)
    if decision.allowed:
        return

    raise HTTPException(
        status_code=429,
        detail=f"rate limit exceeded: at most {limit} graph expansions per {window_s} seconds.",
        headers={"Retry-After": str(decision.retry_after_s)},
    )


# --- expand ---------------------------------------------------------------


@router.get(
    "/api/graph/expand",
    response_model=GraphFragment,
    dependencies=[Depends(enforce_graph_rate_limit)],
)
async def expand_node(
    node: str = Query(
        min_length=3,
        max_length=200,
        description="A node id from a query response, e.g. 'Country:USA'.",
    ),
    runtime: Runtime = Depends(get_runtime),  # noqa: B008 - FastAPI's dependency idiom
) -> GraphFragment:
    try:
        label, key = kg_subgraph.parse_node_id(node)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    fragment = await kg_subgraph.expand(runtime.kg, label=label, key=key)
    return GraphFragment(
        nodes=[dataclasses.asdict(item) for item in fragment.nodes],
        edges=[dataclasses.asdict(item) for item in fragment.edges],
        truncated=fragment.truncated,
    )
