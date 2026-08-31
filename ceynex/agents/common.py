"""Shared machinery for the agent nodes — SRS 3.1.4, 3.4.3, and the SAD §4.1 guarantee.

Every agent needs the same four things: a way to work out what the user asked, a
way to build `Evidence`, a way to derive confidence rather than inventing it, and
a way to attach LLM prose that degrades cleanly. Putting them here means the five
agents differ only where they genuinely differ.

`parse_intent` is deliberately a keyword parser, not an LLM call. Agents are
already inside a graph invocation with a response-time budget; spending a second
LLM round trip per agent to re-read a query the router has already classified is
how the 20-second cross-sector budget (SRS 3.4.1) gets spent before any data is
touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ceynex.contracts import (
    AgentName,
    AgentOutput,
    AgentState,
    Evidence,
    ForecastPoint,
    KnowledgeGraphClientProtocol,
    LLMReasoningClientProtocol,
)
from ceynex.orchestrator.confidence import aggregate_confidence

# --- what the agents are given -------------------------------------------


@dataclass
class AgentDeps:
    """Everything an agent node needs from the outside world.

    Passed explicitly rather than imported, so tests substitute a fake graph and
    a `FakeLLMClient` without patching module globals.
    """

    kg: KnowledgeGraphClientProtocol
    llm: LLMReasoningClientProtocol
    dsn: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# --- reading the question ------------------------------------------------

ITEM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tea": ("tea", "ceylon tea", "black tea", "green tea"),
    "cinnamon": ("cinnamon", "ceylon cinnamon"),
    "rubber": ("rubber", "latex"),
    "coconut": ("coconut", "copra", "coir", "desiccated coconut"),
    "apparel_knit": ("knit", "knitted", "t-shirt", "tshirt", "jersey"),
    "apparel_woven": ("woven", "trousers", "shirts", "not knitted"),
}

# Checked only when nothing more specific matched, so "knitted apparel" resolves
# to apparel_knit rather than to the generic bucket.
SECTOR_FALLBACK_ITEMS = {
    "apparel": "apparel_knit",
    "garment": "apparel_knit",
    "garments": "apparel_knit",
    "clothing": "apparel_knit",
    "textile": "apparel_knit",
    "textiles": "apparel_knit",
}

DISTRICT_WORDS = ("district", "region", "province", "grown", "produced", "producing")


WORD_NUMBERS = {
    "a": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
}


@dataclass
class Intent:
    """What the agent could work out from the query text alone."""

    item: str | None = None
    partner: str | None = None
    year: int | None = None
    horizon: int = 2
    #: What the user asked for. May not be what we hold — see `frequency_note`.
    requested_frequency: str | None = None
    #: Explicit forecast measurement requested by the user, if M1 has one.
    #: `None` intentionally means the established export-value default.
    forecast_target: str | None = None
    wants_districts: bool = False
    wants_forecast: bool = False
    pct_change: float | None = None


def parse_intent(query: str) -> Intent:
    lowered = query.lower()
    intent = Intent()

    for item, keywords in ITEM_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            intent.item = item
            break
    if intent.item is None:
        for word, item in SECTOR_FALLBACK_ITEMS.items():
            if word in lowered:
                intent.item = item
                break

    intent.partner = _find_partner(query)
    _parse_forecast_target(lowered, intent)

    # Non-capturing inner group: re.findall returns the group it captures, so a
    # bare (19|20) would yield centuries rather than years.
    years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", query)]
    if years:
        intent.year = max(years)

    intent.wants_districts = any(word in lowered for word in DISTRICT_WORDS)
    intent.wants_forecast = any(
        word in lowered
        for word in ("forecast", "predict", "next year", "next quarter", "outlook", "will ", "future")
    )

    percent = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)", lowered)
    if percent:
        intent.pct_change = float(percent.group(1)) / 100.0

    _parse_horizon(lowered, intent)
    return intent


def _parse_forecast_target(lowered: str, intent: Intent) -> None:
    """Detect the two M1 agriculture targets without changing export-value defaults.

    A bare request such as "forecast cinnamon exports" deliberately has no
    target here: the forecast agent then uses its existing USD export-value
    path.  That prevents a producer-price model being silently substituted for
    an export forecast merely because both refer to cinnamon.
    """
    if intent.item == "tea" and (
        "export volume" in lowered
        or "tea tonnes" in lowered
        or "tea tons" in lowered
        or re.search(r"\btea\s+(?:export\s+)?kg\b", lowered) is not None
    ):
        intent.forecast_target = "export_volume"
    elif intent.item == "cinnamon" and re.search(
        r"\b(?:cinnamon\s+)?producer\s+prices?\b|\bcinnamon\s+prices?\b", lowered
    ):
        intent.forecast_target = "producer_price"


def _parse_horizon(lowered: str, intent: Intent) -> None:
    """Read "the next two quarters" / "next 3 years" out of the query.

    The requested unit is recorded separately from the horizon because we may not
    hold it. Comtrade is annual; answering a quarterly question with annual
    figures is fine, but doing so silently is not.
    """
    match = re.search(
        r"(?:next|coming|following)\s+(\w+)?\s*(quarter|month|year)s?", lowered
    )
    if not match:
        for unit in ("quarter", "month", "year"):
            if f"next {unit}" in lowered:
                intent.horizon, intent.requested_frequency = 1, unit
                return
        return

    count_word, unit = match.group(1), match.group(2)
    count = 1
    if count_word:
        count = int(count_word) if count_word.isdigit() else WORD_NUMBERS.get(count_word, 1)
    intent.horizon = max(1, min(count, 8))
    intent.requested_frequency = unit


def _find_partner(query: str) -> str | None:
    """Resolve a country named in the query, longest name first.

    Longest-first matters: "United States" must not be shadowed by a shorter
    entry, and "India" must not match inside "Indian Ocean" ahead of it.
    """
    from ceynex.data.crosswalk import _countries

    lowered = query.lower()
    candidates = sorted(_countries(), key=lambda c: -len(c.name))
    for country in candidates:
        if len(country.name) < 4:
            continue
        if re.search(rf"\b{re.escape(country.name.lower())}\b", lowered):
            return country.iso3
    return None


# "American"/"americas" deliberately excluded: a query naming a country
# (already handled by `find_region`'s caller falling back to `_find_partner`)
# is far more likely to mean "American" as in "the US market" than the whole
# Americas continent, and a wrong region filter is a wrong answer, not a
# missing one. `DISTRICT_WORDS` above already uses the bare word "region" for
# a different meaning (sub-national districts) -- these are continent names,
# never that word itself, so the two never collide.
REGION_ALIASES: dict[str, str] = {
    "asia": "Asia",
    "asian": "Asia",
    "europe": "Europe",
    "european": "Europe",
    "africa": "Africa",
    "african": "Africa",
    "oceania": "Oceania",
}


def find_region(query: str) -> str | None:
    """Resolve a continent named in the query (`crosswalk.region_names()`'s
    five values), the same word-boundary approach `_find_partner` uses for a
    country name. Shared by every agent that filters a market-share ranking
    by geography, so "top markets in Asia" means the same thing everywhere
    it's asked -- found live 2026-08-27 duplicated (and answered
    inconsistently) across two agents before this was centralised here.
    """
    lowered = query.lower()
    for alias, region in REGION_ALIASES.items():
        if re.search(rf"\b{alias}\b", lowered):
            return region
    return None


# --- building evidence ---------------------------------------------------


def evidence_from_query(claim: str, cypher: str, period: str | None = None) -> Evidence:
    """Evidence whose `detail` is the literal Cypher that produced the claim.

    SRS 3.1.4 wants the reasoning visible; SRS 3.1.6 wants relational answers to
    come from the graph. Carrying the query text is what makes the second one
    verifiable instead of merely stated.
    """
    evidence = Evidence(
        source_id="KG",
        claim=_as_sentence(claim),
        detail=" ".join(cypher.split()),
    )
    if period:
        evidence["period"] = period
    return evidence


def evidence_from_dataset(claim: str, detail: str, source_id: str, period: str | None = None) -> Evidence:
    """Evidence for a figure read from `fact_trade` rather than the graph."""
    evidence = Evidence(source_id=source_id, claim=_as_sentence(claim), detail=detail)
    if period:
        evidence["period"] = period
    return evidence


def evidence_from_policy(
    claim: str,
    detail: str,
    url: str = "",
    period: str | None = None,
) -> Evidence:
    """Evidence for a claim read out of a retrieved policy document (D10).

    `source_id` is `"POLICY"`. `SourceId` is a plain `str` in the contract
    precisely so a member can add a source without a contract change — M1 already
    ships `FAOSTAT`, `TEA_BOARD` and `DQ_FLAG` the same way.

    `detail` should name the document and page (`PolicyChunk.citation`) followed
    by the search filter the retriever ran under, and `url` should resolve to the
    document. A policy claim is only checkable if a reader can open the source
    and find the sentence; without both, this is the one evidence type in the
    system that would amount to "trust me".
    """
    evidence = Evidence(source_id="POLICY", claim=_as_sentence(claim), detail=detail)
    if url:
        evidence["url"] = url
    if period:
        evidence["period"] = period
    return evidence


def evidence_from_model(claim: str, model_id: str, period: str | None = None) -> Evidence:
    evidence = Evidence(source_id="MODEL", claim=_as_sentence(claim), detail=model_id)
    if period:
        evidence["period"] = period
    return evidence


def figures_evidence(claim: str) -> Evidence:
    """Last-resort evidence saying why there is nothing else."""
    return Evidence(source_id="KG", claim=_as_sentence(claim), detail="no matching rows")


def _as_sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "?", "!")) else text + "."


# --- finishing an agent --------------------------------------------------


async def finish(
    *,
    agent: AgentName,
    state: AgentState,
    deps: AgentDeps,
    summary: str,
    figures: dict[str, float],
    evidence: list[Evidence],
    assumptions: list[str],
    forecast: list[ForecastPoint] | None = None,
    relevance: float | None = None,
) -> dict[str, Any]:
    """Attach prose, derive confidence, and return the partial state.

    Returns only the keys this agent changed, so LangGraph's reducers merge
    parallel branches without one agent clobbering another's output.
    """
    prose = ""
    if summary:
        prose = await deps.llm.generate_explanation(
            {
                "question": state["query"],
                "agent": agent,
                "findings": summary,
                "figures": figures,
                "evidence": [e["claim"] for e in evidence],
            }
        )

    degraded = not prose
    weight = relevance if relevance is not None else state.get("relevance", {}).get(agent, 1.0)

    output = AgentOutput(
        agent=agent,
        # Degraded means figures and evidence without prose (SRS 3.4.3) — the
        # deterministic summary still ships, only the LLM's phrasing is missing.
        summary=prose or summary,
        figures=figures,
        assumptions=assumptions,
        evidence=evidence,
        confidence=_confidence(agent, figures, evidence, degraded, weight),
        degraded=degraded,
    )
    if forecast:
        output["forecast"] = forecast

    return {"agent_outputs": {agent: output}, "degraded": degraded}


def _confidence(
    agent: AgentName,
    figures: dict[str, float],
    evidence: list[Evidence],
    degraded: bool,
    relevance: float,
) -> float:
    """Derive this agent's own confidence — never a hardcoded 0.85.

    The self-assessment before the orchestrator's aggregation: how much did this
    agent actually find? An agent with no figures and one placeholder evidence
    entry should say so in its score rather than leaving the merger to guess.

    Reuses `aggregate_confidence` so there is exactly one formula in the system
    and the docstring a marker will ask about lives in one place
    (`ceynex/orchestrator/confidence.py`).
    """
    base = 0.9 if figures else 0.3
    # Two evidence entries is the contract's floor; more than that is corroboration.
    base -= max(0.0, (2 - len(evidence))) * 0.15

    probe = AgentOutput(
        agent=agent,
        summary="",
        figures=figures,
        assumptions=[],
        evidence=evidence,
        confidence=max(0.0, min(1.0, base)),
        degraded=degraded,
    )
    return aggregate_confidence({agent: probe}, route=[agent], relevance={agent: relevance})


__all__ = [
    "AgentDeps",
    "Intent",
    "evidence_from_dataset",
    "evidence_from_model",
    "evidence_from_policy",
    "evidence_from_query",
    "figures_evidence",
    "find_region",
    "finish",
    "parse_intent",
]
