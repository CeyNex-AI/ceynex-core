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

# Questions about what a destination market's policy *says*, which
# `trade_economics` answers from the D10 document corpus rather than by
# simulating anything. Kept as its own list rather than folded into
# SIMULATION_WORDS because these are not shocks, and a reader of this file
# should not have to infer that "priority market" is a simulation keyword.
#
# `export_analytics` holds only Sri Lanka's own trade flows, so a question about
# the UK's trade strategy routed to it alone returns no evidence at all —
# measured at 0 evidence and 0.15 confidence on P03/P04/P05 (EVALUATION.md §7).
POLICY_WORDS = (
    "trade strategy", "trade policy", "foreign trade", "export strategy",
    "preferential", "preference", "non-tariff", "market access",
    "priority market", "rules of origin", "duty-free", "duty free",
    "trade agreement", "quota", "licensing",
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
#
# The freight entries are deliberately two-word where the single word would
# over-match: a bare "shipping" collides with phrasings like "drop-shipping
# apparel", where the question really is about apparel. `keyword_route` pads the
# query with spaces, which is what lets the existing "fish " entry rely on its
# trailing space. There is no freight, shipping-cost or logistics data anywhere
# in CeyNex, so naming freight here is what makes the refusal say *why* rather
# than falling through to the generic "names nothing CeyNex covers" branch.
#
# Kept to the smallest set with the same matching power, because matching is
# substring-based and every match is named in the note: "shipping cost" already
# catches "shipping costs", and "freight" already catches "ocean/sea freight".
# Listing the longer forms too would only make the note name one exclusion twice
# ("the question is about freight, ocean freight").
OUT_OF_SCOPE_WORDS = (
    "gem", "sapphire", "tourism", "tourist", "remittance", "fisheries", "fish ",
    "cement", "petroleum", "software export", "it export", "bpo",
    "freight", "shipping cost", "container rate", "logistics cost",
)

# The sentence every out-of-scope note ends with, and the two whole-note shapes
# built from it. One copy, because two copies drift — and they did: `keyword_route`
# built these notes and `llm_route` built none, so every LLM-router out-of-scope
# verdict reached `merger.py` with an empty note and fell through to its generic
# fallback, telling the user a sector had been named when none was. Measured live
# 2026-09-03 on "how are shipping costs affecting Sri Lankan exporters?", "who was
# Leonhard Euler?" and "what were Sri Lanka's tea exports in 2035?" alike.
SCOPE_SENTENCE = (
    "Scope is agriculture (tea, cinnamon, rubber, coconut) and apparel "
    "(HS 61/62) exports — SRS 2.4"
)
NO_TOPIC_NOTE = f"the question does not name anything CeyNex covers. {SCOPE_SENTENCE}"
MIXED_SCOPE_NOTE = f"part of the question is outside what CeyNex covers. {SCOPE_SENTENCE}"


def named_out_of_scope_note(named: str, *, partly_in_scope: bool) -> str:
    """The most specific note of the three: it can name what was excluded.

    Only `keyword_route` can produce this — it knows *which* word matched. The
    LLM router only reports that something was out of scope, so it falls back to
    the two generic notes above.
    """
    lead = (
        f"the question also asks about {named}, which CeyNex does not cover"
        if partly_in_scope
        else f"the question is about {named}, which CeyNex does not cover"
    )
    return f"{lead}. {SCOPE_SENTENCE}"


@dataclass
class RouteDecision:
    """What the router decided, and how it decided it."""

    route: list[AgentName]
    sectors: list[Sector]
    relevance: dict[AgentName, float]
    method: str  # "keyword" | "llm" | "llm->keyword"
    out_of_scope: bool = False
    # Distinguishes two shapes `out_of_scope` used to conflate: a *mixed*
    # question naming both an in-scope sector and an excluded one ("tea vs
    # fisheries" -- still worth answering about tea) from one naming nothing
    # CeyNex covers at all ("who is Euler" -- worth answering nothing about).
    # merger.py uses this to decide whether a routed agent's output is a real
    # finding or noise that happens to have run.
    no_topic_recognized: bool = False
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
    # Both go to trade_economics; the agent's own `_classify_shock` decides
    # whether the question is a shock to simulate or a policy to describe.
    wants_simulation = _any(lowered, SIMULATION_WORDS) or _any(lowered, POLICY_WORDS)
    wants_forecast = _any(lowered, FORECAST_WORDS)
    wants_analytics = _any(lowered, ANALYTICS_WORDS)

    # Flagged whenever an uncovered sector is named, in-scope words present or
    # not. Suppressing the flag when the query also names tea or apparel is what
    # made "how does tea compare with fisheries" return a confident tea answer
    # with no mention of fisheries — the mixed question is the one that most
    # needs the limit stated, because half of it looks answered.
    named_out_of_scope = [word.strip() for word in OUT_OF_SCOPE_WORDS if word in lowered]
    partly_in_scope = bool(named_out_of_scope) and (hits_agriculture or hits_apparel)

    # Found live 2026-08-27: "who is Euler" names no excluded sector either --
    # it names nothing at all, in scope or out. The keyword lists above are
    # the only signal this router has, so if every one of them came back
    # empty, the question is not about Sri Lankan trade in any sense this
    # router can recognise, not merely "the wrong sector".
    no_topic_recognized = not (
        hits_agriculture
        or hits_apparel
        or wants_simulation
        or wants_forecast
        or wants_analytics
        or named_out_of_scope
    )
    out_of_scope = bool(named_out_of_scope) or no_topic_recognized

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

    # Export Analytics answers from Cypher (SRS 3.1.6), so it can contribute
    # market share, concentration and growth to any question naming a sector or
    # an item — not only to ones using an explicitly analytical word.
    #
    # Measured on the 30-question set: adding it only when an analytics keyword
    # matched left 9 of 30 questions routed exclusively to sector agents, which
    # returned in ~2 ms having never touched the graph, with no figures and no
    # evidence. "Which markets buy the most Sri Lankan knitted apparel?" is
    # answerable from the graph today and was answering nothing.
    #
    # Simulations are left alone: trade_economics resolves its own baselines
    # from the graph, so adding a second grounded agent there widens the route
    # without adding information.
    if wants_analytics:
        relevance["export_analytics"] = 1.0
    elif (hits_agriculture or hits_apparel) and not wants_simulation:
        relevance["export_analytics"] = 0.7
    elif not relevance:
        relevance["export_analytics"] = 0.6

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
        no_topic_recognized=no_topic_recognized,
        reason=reason,
    )
    if named_out_of_scope:
        decision.notes.append(
            named_out_of_scope_note(
                ", ".join(sorted(set(named_out_of_scope))), partly_in_scope=partly_in_scope
            )
        )
    elif no_topic_recognized:
        decision.notes.append(NO_TOPIC_NOTE)
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
- trade_economics: two jobs. (a) simulating exchange-rate moves, tariffs, and trade-agreement changes on export revenue; (b) answering what a DESTINATION MARKET's own trade policy says -- it is the only agent with the foreign policy-document corpus (US, UK, India, Canada, Italy, EU, and Sri Lanka's own export strategy).
- forecast: forward-looking export volume and value projections with intervals.

Rules:
- Return at least one agent. Never an empty list.
- A question spanning both sectors gets both sector agents.
- A currency, tariff or agreement question gets trade_economics, plus the sector agents it affects.
- ANY question about a foreign government's or trade bloc's trade policy, trade strategy, tariffs, non-tariff measures, preferences, market access or priority markets gets trade_economics -- INCLUDING when it only asks what that policy SAYS and simulates nothing. export_analytics holds only Sri Lanka's own trade flows and cannot answer any of them.
  ADD the sector agent too whenever such a question names goods: "What non-tariff measures does the EU apply to spices?" is trade_economics AND agriculture_commodity; "Which trade agreement gives Sri Lankan cinnamon preferential access?" is trade_economics AND agriculture_commodity. trade_economics alone is right only when no commodity or garment is named at all, as in "What does India's Foreign Trade Policy say about imports from Sri Lanka?".
- A trend, growth rate, ranking ("fastest", "largest", "top", "which country"), market share, or concentration question gets export_analytics, IN ADDITION TO the sector agent(s) it names -- not instead of them. "Which country is the fastest growing market for cinnamon?" is both agriculture_commodity (names cinnamon) and export_analytics (asks for a growth ranking) at once.
- relevance is 0.0-1.0 per agent: how central it is to the question.
- CeyNex covers only agriculture (tea, cinnamon, rubber, coconut) and apparel. Set out_of_scope true if
  the question is about some other sector entirely (gems, tourism, fisheries, ...), OR if it is not about
  Sri Lankan trade/exports at all (a general-knowledge question, small talk, anything unrelated). In the
  second case, sectors must not include "agriculture" or "apparel" -- naming one of those, even in a
  route picked only because a route can never be empty, would say the question is partly about a real
  sector when it is not about one at all.
  IN SCOPE, out_of_scope FALSE: the trade policy of any country Sri Lanka exports to, even when the
  question names no commodity at all. "What does India's Foreign Trade Policy say about imports from
  Sri Lanka?" and "Does the Netherlands identify Sri Lanka as a priority market?" are squarely in scope
  -- a destination market's rules govern Sri Lankan exports, and trade_economics holds documents for
  them. Use sectors ["macro"] for these. Marking one out_of_scope makes the merger discard the
  agent's findings as noise and the user gets "CeyNex does not cover that" about a question it does.
- Still return at least one agent even when out_of_scope is true (a route can never be empty) --
  export_analytics with a low relevance is the reasonable default when nothing else fits.

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
    out_of_scope = bool(parsed.get("out_of_scope", False))
    # ROUTER_SYSTEM tells the model to omit "agriculture"/"apparel" from
    # sectors when the question isn't about a real sector at all, only about
    # some excluded one -- so out_of_scope plus neither of those present is
    # the "no topic recognised" case, same distinction keyword_route makes.
    no_topic_recognized = out_of_scope and not any(s in ("agriculture", "apparel") for s in sectors)

    # Parity with `keyword_route`. `graph.py` turns `notes[0]` into the
    # `out_of_scope:` error the merger reads back, so leaving this empty is not a
    # missing nicety -- it is the difference between the user being told what
    # CeyNex actually covers and being told, falsely, that they named an excluded
    # sector. `keyword_route`'s more specific "names gems/tourism" note has no
    # equivalent here: the LLM reports *that* something is out of scope, never
    # which word did it.
    notes: list[str] = []
    if out_of_scope:
        notes.append(NO_TOPIC_NOTE if no_topic_recognized else MIXED_SCOPE_NOTE)

    return RouteDecision(
        route=[agent for agent in ALL_AGENTS if agent in route],
        sectors=sectors or fallback.sectors,
        relevance=relevance,
        method="llm",
        out_of_scope=out_of_scope,
        no_topic_recognized=no_topic_recognized,
        reason=str(parsed.get("reason", ""))[:200],
        notes=notes,
    )


__all__ = ["RouteDecision", "keyword_route", "llm_route"]
