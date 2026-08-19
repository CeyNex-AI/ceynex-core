"""Implements SRS 3.1.2 — deciding which agents answer a query.

Two routers, one interface. `keyword_route()` is the v0: deterministic, offline,
free, and it is what runs when the LLM is unavailable or its answer is unusable.
`llm_route()` upgrades it. **The keyword router is not a placeholder** — it is the
fallback the whole design leans on, and the system is required to keep working
when the LLM does not (SRS 3.4.3).

Three invariants, in the order they matter:

1. **The route is never empty.** A query nobody recognises still gets an answer;
   it defaults to Export Analytics, which can at least describe what the graph
   holds. Returning `[]` would mean the user gets silence.
2. **Out-of-scope questions are recognised, not routed away.** CeyNex covers
   agriculture and apparel (SRS 2.4). Asking about gems or tourism should produce
   "that is outside what I cover", which is a correct answer — so the router
   flags it rather than quietly picking an agent that will find nothing.
3. **Relevance is a number, not a flag.** Each routed agent gets a weight, and
   `confidence.py` uses it so a peripheral agent's low confidence does not drag
   down an answer it was barely part of.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from ceynex.contracts import ALL_AGENTS, DEFAULT_AGENT, AgentName, Sector

log = logging.getLogger(__name__)

AGRICULTURE_WORDS = (
    "agricultur", "tea", "cinnamon", "rubber", "coconut", "copra", "coir",
    "spice", "crop", "plantation", "commodity", "commodities", "harvest",
)
APPAREL_WORDS = (
    "apparel", "garment", "clothing", "textile", "knit", "woven", "t-shirt",
    "tshirt", "manufactur", "factory", "labour cost", "labor cost", "buyer",
)
SIMULATION_WORDS = (
    "deprecia", "apprecia", "exchange rate", "rupee", "lkr", "currency", "devalu",
    "tariff", "duty", "gsp", "fta", "trade agreement", "policy", "simulat",
    "what happens if", "what if", "impact of", "shock",
)
FORECAST_WORDS = (
    "forecast", "predict", "projection", "outlook", "next year", "next quarter",
    "next month", "will ", "future", "expected", "going to",
)
ANALYTICS_WORDS = (
    "trend", "growth", "cagr", "market share", "concentration", "district",
    "fastest", "largest", "top ", "which country", "which importing",
    "compare", "comparison", "versus", " vs ", "over the last", "historical",
)

# SRS 2.4 fixes scope. Naming one of these is a strong signal the question is
# outside it — but only when nothing in scope is named too, since "should we
# prioritise gems or tea" is still answerable about tea.
OUT_OF_SCOPE_WORDS = (
    "gem", "sapphire", "tourism", "tourist", "remittance", "fisheries", "fish ",
    "cement", "petroleum", "software export", "it export", "bpo",
)


@dataclass
class RouteDecision:
    """What the router decided, and how it decided it."""

    route: list[AgentName]
    sectors: list[Sector]
    relevance: dict[AgentName, float]
    method: str  # "keyword" | "llm" | "llm->keyword"
    out_of_scope: bool = False
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    def as_state_patch(self) -> dict[str, object]:
        return {"route": self.route, "sectors": self.sectors, "relevance": self.relevance}


# --- v0: keyword routing -------------------------------------------------


def keyword_route(query: str) -> RouteDecision:
    """Deterministic routing from the query text. Never empty, never raises."""
    lowered = f" {query.lower().strip()} "

    hits_agriculture = _any(lowered, AGRICULTURE_WORDS)
    hits_apparel = _any(lowered, APPAREL_WORDS)
    wants_simulation = _any(lowered, SIMULATION_WORDS)
    wants_forecast = _any(lowered, FORECAST_WORDS)
    wants_analytics = _any(lowered, ANALYTICS_WORDS)

    out_of_scope = _any(lowered, OUT_OF_SCOPE_WORDS) and not (hits_agriculture or hits_apparel)

    sectors: list[Sector] = []
    if hits_agriculture:
        sectors.append("agriculture")
    if hits_apparel:
        sectors.append("apparel")
    if len(sectors) == 2:
        sectors = ["cross_sector", "agriculture", "apparel"]
    if wants_simulation and "macro" not in sectors:
        sectors.append("macro")

    relevance: dict[AgentName, float] = {}

    if wants_simulation:
        relevance["trade_economics"] = 1.0
        # A shock lands on whichever sectors it touches, so their agents come too.
        if hits_agriculture:
            relevance["agriculture_commodity"] = 0.8
        if hits_apparel:
            relevance["apparel_manufacturing"] = 0.8
        if not hits_agriculture and not hits_apparel:
            relevance["agriculture_commodity"] = 0.6
            relevance["apparel_manufacturing"] = 0.6

    if hits_agriculture and not wants_simulation:
        relevance["agriculture_commodity"] = 1.0
    if hits_apparel and not wants_simulation:
        relevance["apparel_manufacturing"] = 1.0

    if wants_forecast:
        relevance["forecast"] = 0.9 if (hits_agriculture or hits_apparel) else 0.7

    if wants_analytics or not relevance:
        relevance["export_analytics"] = 1.0 if wants_analytics else 0.6

    # Invariant 1. Nothing below this line may produce an empty route.
    if not relevance:
        relevance[DEFAULT_AGENT] = 0.5

    route = [agent for agent in ALL_AGENTS if agent in relevance]
    reason = _explain(hits_agriculture, hits_apparel, wants_simulation, wants_forecast, wants_analytics)

    decision = RouteDecision(
        route=route,
        sectors=sectors or ["cross_sector"],
        relevance=relevance,
        method="keyword",
        out_of_scope=out_of_scope,
        reason=reason,
    )
    if out_of_scope:
        decision.notes.append(
            "The query names a sector CeyNex does not cover. Scope is agriculture "
            "(tea, cinnamon, rubber, coconut) and apparel (HS 61/62) — SRS 2.4."
        )
    return decision


def _any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)


def _explain(agri: bool, apparel: bool, sim: bool, forecast: bool, analytics: bool) -> str:
    parts = []
    if agri and apparel:
        parts.append("names both sectors")
    elif agri:
        parts.append("names agriculture")
    elif apparel:
        parts.append("names apparel")
    if sim:
        parts.append("asks about a policy or currency shock")
    if forecast:
        parts.append("asks about the future")
    if analytics:
        parts.append("asks for a trend or ranking")
    return "keyword router: " + (", ".join(parts) if parts else "no strong signal, defaulting")


# --- v1: LLM routing -----------------------------------------------------

ROUTER_SYSTEM = """You route questions about Sri Lanka's export economy to specialist agents.

The agents:
- export_analytics: trends, growth rates, market share, district concentration, rankings. Answers from a knowledge graph.
- agriculture_commodity: tea, cinnamon, rubber, coconut — prices, production, substitution.
- apparel_manufacturing: HS 61 knit and HS 62 woven garments — buyer demand, capacity, labour costs.
- trade_economics: simulating exchange-rate moves, tariffs, and trade-agreement changes on export revenue.
- forecast: forward-looking export volume and value projections with intervals.

Rules:
- Return at least one agent. Never an empty list.
- A question spanning both sectors gets both sector agents.
- A currency, tariff or agreement question gets trade_economics, plus the sector agents it affects.
- relevance is 0.0-1.0 per agent: how central it is to the question.
- CeyNex covers only agriculture (tea, cinnamon, rubber, coconut) and apparel. Set out_of_scope true if the question is about some other sector entirely.

Reply with JSON only:
{"route": ["..."], "sectors": ["agriculture"|"apparel"|"cross_sector"|"macro"], "relevance": {"agent": 0.0-1.0}, "out_of_scope": false, "reason": "one short sentence"}"""


async def llm_route(query: str, llm) -> RouteDecision:  # noqa: ANN001 - protocol, not a concrete type
    """LLM routing with `keyword_route` as the fallback for every failure mode.

    Falls back when: the LLM is unavailable, returns unparseable JSON, or returns
    a route containing no valid agent. The last one matters most — a plausible
    but wrong agent name would otherwise produce an empty graph invocation.
    """
    fallback = keyword_route(query)

    raw = await llm.generate("router", ROUTER_SYSTEM, query, json_mode=True)
    if not raw:
        fallback.notes.append("LLM unavailable; routed by keyword (SRS 3.4.3).")
        return fallback

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        log.warning("router returned unparseable JSON, falling back: %r", str(raw)[:200])
        fallback.method = "llm->keyword"
        fallback.notes.append("LLM routing output was not valid JSON; routed by keyword.")
        return fallback

    route = [a for a in parsed.get("route", []) if a in ALL_AGENTS]
    if not route:
        log.warning("router returned no valid agents (%s), falling back", parsed.get("route"))
        fallback.method = "llm->keyword"
        fallback.notes.append("LLM named no valid agent; routed by keyword.")
        return fallback

    relevance = {
        agent: float(weight)
        for agent, weight in (parsed.get("relevance") or {}).items()
        if agent in route
    }
    for agent in route:
        relevance.setdefault(agent, 1.0)

    sectors = [s for s in parsed.get("sectors", []) if s in ("agriculture", "apparel", "cross_sector", "macro")]

    return RouteDecision(
        route=[agent for agent in ALL_AGENTS if agent in route],
        sectors=sectors or fallback.sectors,
        relevance=relevance,
        method="llm",
        out_of_scope=bool(parsed.get("out_of_scope", False)),
        reason=str(parsed.get("reason", ""))[:200],
    )


__all__ = ["RouteDecision", "keyword_route", "llm_route"]
