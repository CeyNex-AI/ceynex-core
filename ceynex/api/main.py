"""The FastAPI application — SAD §8 Presentation Layer, SRS 3.9.3.

    uvicorn ceynex.api.main:app --host 0.0.0.0 --port 8000

**Ownership.** M2 seeds this package with `/health` and `POST /api/query` so the
orchestrator is reachable for the mid-evaluation demo. Everything else on the
§4.5 surface — signup, login, logout, query history, help content, and the four
admin routes — is M3's, along with `web/`. New routers go in
`ceynex/api/routes/` and are included below; nothing here needs restructuring to
accommodate them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ceynex.api.api_keys import ensure_table as ensure_api_keys_table
from ceynex.api.deps import Runtime, set_runtime
from ceynex.api.history import ensure_table as ensure_history_table
from ceynex.api.preferences import ensure_table as ensure_preferences_table
from ceynex.api.routes import account, admin, auth, graph, health, history, news, query
from ceynex.news import refresh as news_refresh
from ceynex.news import snapshot as news_snapshot

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the Neo4j pool, compile the graph, warm the models, start the refresher.

    `warmup()` was previously defined and never called: `deps.py` explains that
    the ONNX sessions must be built before the first request or that request
    silently degrades against a 2 s budget and never reproduces once the process
    is warm. It is started here — as a task, not awaited, because on a cold
    `fastembed_cache` volume it downloads several hundred megabytes and the
    container healthcheck allows about 95 s before it starts killing us. An
    inline await would turn a latent bug into a crash loop.

    Both tasks are locals of this generator and so outlive the `yield`. That is
    the strong reference keeping them alive: a bare fire-and-forget
    `create_task` can be garbage-collected mid-flight, which is the kind of bug
    that only shows up under load and never reproduces.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    runtime = Runtime.build()
    set_runtime(runtime)
    ensure_history_table()
    ensure_preferences_table()
    ensure_api_keys_table()
    news_snapshot.ensure_table()

    warm = asyncio.create_task(runtime.warmup())
    # Waits for `warm` before its first pass — a refresh that starts ahead of the
    # embedder would index nothing and blame the wrong thing.
    refresher = asyncio.create_task(
        news_refresh.run_forever(runtime.gdelt, runtime.news, warm=warm)
    )
    try:
        yield
    finally:
        for task in (refresher, warm):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await runtime.aclose()
        set_runtime(None)


app = FastAPI(
    title="CeyNex",
    version="0.1.0",
    summary="Multi-agent decision intelligence for Sri Lanka's export economy",
    description=(
        "Ask a question about Sri Lanka's agriculture or apparel exports in plain English. "
        "The orchestrator routes it to the relevant specialist agents, merges their findings "
        "into one answer, and returns it with a confidence score and the evidence behind it.\n\n"
        "Forecasts, confidence scores and simulations are generated from available data and "
        "modelling techniques. They are not financial, legal, investment or official policy "
        "advice, and are not guaranteed outcomes (SRS 3.11.1)."
    ),
    lifespan=lifespan,
)

# The frontend VM serves the browser and calls this over the VPC. Restricted to
# the deployment's own origins once M3's web app has one; permissive here would
# be a real problem only if this service were internet-facing, which by design it
# is not (see ceynex-infra/gcp/01_firewall_setup.sh).
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|10\.160\.0\.\d+)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(query.router)
app.include_router(auth.router)
app.include_router(history.router)
app.include_router(admin.router)
app.include_router(account.router)
app.include_router(news.router)
app.include_router(graph.router)
