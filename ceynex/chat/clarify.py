"""Ask one question back, but only when the question genuinely cannot be read —
SRS 3.1.2, SRS 3.4.3, deviation D13.

Two stages, and the first one is free.

**Stage A is deterministic and runs on every turn.** It works over values the
system already computes — `parse_intent`'s keyword categories, `_find_partner`,
`find_region`, `keyword_route` — and answers one question: is this query
ambiguous in a way that will silently lose part of the user's meaning? On the
overwhelming majority of questions it says no, and says it without an LLM call,
a database read, or any added latency at all. That skip path is the reason this
feature will not become annoying, so it is the part most heavily asserted.

**Stage B only ever phrases.** When Stage A fires, one cheap `clarifier` call
turns the trigger into a readable question — and may *veto* it, when the phrasing
resolves an ambiguity the syntactic check missed. It never decides to ask. With no
key, a timeout, or an unparseable reply, a deterministic template says the same
thing in plainer words, so the degraded clarifier is the same feature with worse
prose rather than a missing one.

**The gate never runs on the resume path**, which is what makes the one-round cap
structural rather than a counter that could drift across the two uvicorn workers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from ceynex.agents.common import ITEM_KEYWORDS, find_region, parse_intent
from ceynex.orchestrator.router import POLICY_WORDS, SIMULATION_WORDS, keyword_route

log = logging.getLogger(__name__)

#: Bounded because the reader is waiting, but not so tight that the call can
#: never land. An earlier 2.5s never once succeeded against the real provider —
#: a comparable `router` call measures ~3.4s — so every clarification silently
#: used the template and the model's one real job, vetoing a gate the syntactic
#: check opened wrongly, never happened. Still far inside `STREAM_BUDGET_S`, and
#: the reader has the `start` frame throughout.
CLARIFIER_BUDGET_S = 5.0

#: Phrasings where naming several things is the point. The router already handles
#: these correctly, and re-asking about a case that works is how a clarifier turns
#: into an obstacle.
COMPARISON_MARKERS = (" vs ", " vs.", "versus", "compare", "comparison", "against each other")

#: A referent that stands in for a market without naming one. Only consulted when
#: `_find_partner` and `find_region` have both already failed.
VAGUE_DESTINATIONS = (
    "that country", "a country", "another country", "that market", "a market",
    "another market", "the region", "that region", "somewhere else", "elsewhere",
    "our biggest market", "their market",
)

MAX_OPTIONS = 4

CLARIFIER_SYSTEM = """You phrase one clarifying question for a Sri Lankan export intelligence platform.

A deterministic check has already decided the question is ambiguous and has given you
the ambiguity and the candidate options. Your job is only to word it well.

Return JSON: {"ask": true, "question": "...", "options": ["...", "..."]}

Rules:
1. You may set "ask": false — and should — if the user's phrasing already resolves the
   ambiguity and the check was too blunt. That is the one judgement you are asked for.
2. Never invent an option. Use the ones given, in the same wording.
3. One short question. No preamble, no explanation, no restating the user's question.
4. Never ask about anything except the ambiguity you were given."""


@dataclass(frozen=True)
class Trigger:
    """What Stage A found, before anyone has phrased anything."""

    kind: str  # "multi_item" | "vague_destination"
    original_query: str
    options: tuple[str, ...] = ()


@dataclass
class Clarification:
    """The question actually put to the reader."""

    question: str
    options: list[str] = field(default_factory=list)
    original_query: str = ""
    kind: str = ""
    method: str = "template"  # "template" | "llm"
    #: Never False. A gate with no way past it is a wall, not a gate.
    allow_skip: bool = True

    def as_payload(self) -> dict:
        return {
            "question": self.question,
            "options": list(self.options),
            "kind": self.kind,
            "method": self.method,
            "allow_skip": self.allow_skip,
            "original_query": self.original_query,
        }

    @staticmethod
    def compose(original_query: str, answers: list[str]) -> str:
        """Fold the reader's choice back into a standalone question.

        Appended rather than substituted: rewriting the user's own words risks
        changing what they asked, and the router reads the whole string anyway.
        """
        chosen = ", ".join(a.strip() for a in answers if a and a.strip())
        if not chosen:
            return original_query
        return f"{original_query.rstrip('?').strip()} — specifically: {chosen}"


def _matching_items(lowered: str) -> list[str]:
    """Every item category the query mentions, not just the first.

    `parse_intent` breaks at the first match, so "compare tea and cinnamon
    exports" resolves to `item="tea"` and cinnamon is dropped with no trace
    anywhere. This is the same loop without the `break`, which is precisely the
    information the gate needs and `Intent` has nowhere to put.
    """
    return [
        item
        for item, keywords in ITEM_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]


def clarification_needed(query: str) -> Trigger | None:
    """Stage A. Free, deterministic, and usually `None`."""
    lowered = query.lower()

    # An out-of-scope question gets today's honest refusal. Asking a clarifying
    # question about a domain CeyNex does not cover would be worse than useless:
    # it implies an answer exists behind the choice.
    decision = keyword_route(query)
    if decision.no_topic_recognized or decision.nothing_in_scope:
        return None

    items = _matching_items(lowered)
    if len(items) > 1 and not any(marker in lowered for marker in COMPARISON_MARKERS):
        return Trigger("multi_item", query, tuple(items[:MAX_OPTIONS]))

    wants_simulation = any(word in lowered for word in SIMULATION_WORDS)
    wants_policy = any(word in lowered for word in POLICY_WORDS)
    if wants_simulation or wants_policy:
        intent = parse_intent(query)
        if (
            intent.partner is None
            and find_region(query) is None
            and any(phrase in lowered for phrase in VAGUE_DESTINATIONS)
        ):
            return Trigger("vague_destination", query)

    return None


def template_clarification(trigger: Trigger) -> Clarification:
    """The deterministic phrasing. Also the fallback for every Stage B failure."""
    if trigger.kind == "multi_item":
        readable = [item.replace("_", " ") for item in trigger.options]
        named = (
            " and ".join(readable)
            if len(readable) == 2
            else ", ".join(readable[:-1]) + f" and {readable[-1]}"
        )
        return Clarification(
            question=f"Your question mentions {named}. Which would you like?",
            options=[*trigger.options, "both"] if len(trigger.options) == 2 else list(trigger.options),
            original_query=trigger.original_query,
            kind=trigger.kind,
        )
    return Clarification(
        question="Which market did you mean?",
        options=[],
        original_query=trigger.original_query,
        kind=trigger.kind,
    )


async def llm_clarify(trigger: Trigger, llm) -> Clarification | None:
    """Stage B. Phrases the question, or vetoes it. Never decides to ask.

    Returns `None` only when the model actively vetoes — every failure mode
    returns the template instead, because a gate that silently disappears when a
    key expires is a different feature on Tuesday than it was on Monday.
    """
    fallback = template_clarification(trigger)
    if llm is None or not getattr(llm, "available", False):
        return fallback

    payload = json.dumps(
        {
            "question": trigger.original_query,
            "ambiguity": trigger.kind,
            "options": list(trigger.options),
        },
        default=str,
    )
    try:
        raw = await asyncio.wait_for(
            llm.generate("clarifier", CLARIFIER_SYSTEM, payload, json_mode=True),
            timeout=CLARIFIER_BUDGET_S,
        )
    except (TimeoutError, Exception):  # noqa: BLE001 - a clarifier never fails a turn
        log.warning("clarifier unavailable; using the deterministic phrasing", exc_info=True)
        return fallback

    if not raw:
        return fallback
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback
    if not isinstance(parsed, dict):
        return fallback

    if parsed.get("ask") is False:
        log.info("clarifier vetoed the gate for: %s", trigger.original_query[:80])
        return None

    question = str(parsed.get("question") or "").strip()
    if not question:
        return fallback

    # The options are the deterministic check's, always — the model phrases and
    # nothing else. It may not widen the choice (an invented option is one the
    # check never found) and it may not narrow it either: asked about tea and
    # cinnamon it returned the question "tea, cinnamon, or both?" while dropping
    # "both" from the list, offering the reader a choice they could not make.
    return Clarification(
        question=question,
        options=fallback.options,
        original_query=trigger.original_query,
        kind=trigger.kind,
        method="llm",
    )


__all__ = [
    "CLARIFIER_BUDGET_S",
    "Clarification",
    "Trigger",
    "clarification_needed",
    "llm_clarify",
    "template_clarification",
]
