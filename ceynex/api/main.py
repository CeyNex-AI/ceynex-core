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

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ceynex.api.api_keys import ensure_table as ensure_api_keys_table
from ceynex.api.deps import Runtime, set_runtime
from ceynex.api.history import ensure_table as ensure_history_table
from ceynex.api.preferences import ensure_table as ensure_preferences_table
from ceynex.api.routes import account, admin, auth, health, history, query

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the Neo4j pool and compile the graph once, close the pool on shutdown."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    runtime = Runtime.build()
    set_runtime(runtime)
    ensure_history_table()
    ensure_preferences_table()
    ensure_api_keys_table()
    try:
        yield
    finally:
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
