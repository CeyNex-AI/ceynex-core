"""The short plan shown before the fan-out starts — SRS 3.1.2, deviation D12.

The orchestration trace reports what *did* happen. This reports what is *about*
to, which is the difference between a progress bar and an explanation: a reader
who sees "checking which markets took Sri Lankan cinnamon, then comparing against
last year" understands the answer they are about to get.

**It runs concurrently with routing, not before it.** Both take only the question
as input, and routing is a real LLM call — measured at 4.6s on the live stack, the
single largest fixed cost in a query. Running the planner alongside it makes the
plan effectively free; running it first would add its latency to a path
`docs/EVALUATION.md` already records as breaching SRS 3.4.1's budget.

**Degraded mode gets a real plan, not a blank space** (SRS 3.4.3). `keyword_route`
is deterministic, free and always available, so the fallback below describes the
same work the system is genuinely about to do — it is a plainer plan, not an
absent one. That also makes the fallback the *default* safety net rather than a
path only exercised when a key expires.

Nothing here is decorative. A step is emitted only when the work it names is
actually going to be attempted, because a plan that lists a step the system then
skips is the same defect as a trace that invents one.
"""

from __future__ import annotations

import asyncio
import json
import logging

from ceynex.agents.common import parse_intent
from ceynex.contracts import AgentName
from ceynex.orchestrator.router import keyword_route

log = logging.getLogger(__name__)

#: Kept tight because it overlaps with routing: past this the plan is not worth
#: delaying the fan-out for, and the deterministic plan is already good.
PLANNER_BUDGET_S = 3.0

MIN_STEPS = 2
MAX_STEPS = 6

PLANNER_SYSTEM = """You plan how to answer a question about Sri Lanka's export economy.

You are not answering it. You are stating, in order, what you are about to check.

Return JSON: {"steps": ["...", "..."]}

Rules:
1. Between 3 and 5 steps. Each one short — under 12 words, no trailing full stop.
2. Describe checks, never conclusions. "Compare 2024 and 2025 export value" is a
   step; "Exports grew 12%" is not, and you do not know it yet.
3. Only name things this system holds: a trade knowledge graph of commodities,
   apparel categories, partner countries and trade agreements; a unified dataset
   of export value, volume and price; registered forecasting models; and policy
   documents from destination markets.
4. Never promise a figure, a source or a country the question did not raise.
5. Plain English for a policymaker. No jargon, no numbering, no markdown."""


#: What each agent is about to do, in the reader's language rather than ours.
#: Used for the deterministic plan and as the vocabulary the LLM plan is checked
#: against — an agent that is not routed cannot appear in the plan.
_AGENT_STEP: dict[AgentName, str] = {
    "export_analytics": "Check export value and market concentration in the trade graph",
    "agriculture_commodity": "Read the agriculture price and volume series",
    "apparel_manufacturing": "Read the apparel export series for HS 61 and 62",
    "trade_economics": "Work through the tariff and trade-agreement effects",
    "forecast": "Run the forecast model and its prediction interval",
}


def _partner_name(iso3: str) -> str:
    """"Germany", not "DEU".

    `parse_intent` resolves a partner to its ISO-3166 alpha-3 code because that
    is what the Cypher needs. The plan is prose a policymaker reads, and a step
    saying "cinnamon and DEU" reads as a system talking to itself. Falls back to
    the code if the crosswalk does not know it — a slightly technical plan beats
    a plan that raises.
    """
    from ceynex.data.crosswalk import CrosswalkError, country_name

    try:
        return country_name(iso3)
    except (CrosswalkError, KeyError):
        return iso3


def deterministic_plan(query: str) -> list[str]:
    """A true plan with no LLM at all, from the keyword router and the intent.

    Free, instant and always available, so it is both the degraded path and the
    thing the LLM plan has to beat.
    """
    decision = keyword_route(query)
    intent = parse_intent(query)

    subject = intent.item or "the sectors named"
    opening = f"Work out what is being asked about {subject}"
    if intent.partner:
        opening = f"Work out what is being asked about {subject} and {_partner_name(intent.partner)}"

    steps = [opening]
    steps.extend(
        _AGENT_STEP[agent] for agent in decision.route if agent in _AGENT_STEP
    )
    steps.append("Merge the findings, cite each figure, and score confidence")
    return steps[:MAX_STEPS]


async def plan(query: str, llm) -> tuple[list[str], str]:
    """`(steps, method)` — the plan to show, and how it was produced.

    `method` is returned rather than logged because the UI says which it was. A
    reader who cannot tell a model's plan from a generated one cannot judge
    either, and "the plan was written by a keyword router" is a fact about the
    answer, not an implementation detail to hide.
    """
    fallback = deterministic_plan(query)

    if llm is None or not getattr(llm, "available", False):
        return fallback, "deterministic"

    try:
        raw = await asyncio.wait_for(
            llm.generate("planner", PLANNER_SYSTEM, query, json_mode=True),
            timeout=PLANNER_BUDGET_S,
        )
    except (TimeoutError, Exception):  # noqa: BLE001 - a plan must never fail a query
        log.info("planner unavailable; using the deterministic plan")
        return fallback, "deterministic"

    steps = _parse(raw)
    if steps is None:
        return fallback, "deterministic"
    return steps, "llm"


def _parse(raw: str | None) -> list[str] | None:
    """Steps from the model's JSON, or None to fall back.

    Rejects rather than repairs. A half-parsed plan is worse than the
    deterministic one: it is shown with the same authority and describes work
    nobody chose.
    """
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.info("planner returned unparseable JSON; falling back")
        return None

    steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(steps, list):
        return None

    cleaned = [
        step.strip().rstrip(".")
        for step in steps
        if isinstance(step, str) and step.strip()
    ]
    if not MIN_STEPS <= len(cleaned) <= MAX_STEPS:
        log.info("planner returned %d steps, outside %d-%d", len(cleaned), MIN_STEPS, MAX_STEPS)
        return None
    return cleaned


__all__ = ["MAX_STEPS", "MIN_STEPS", "PLANNER_BUDGET_S", "deterministic_plan", "plan"]
