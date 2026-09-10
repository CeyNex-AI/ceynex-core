"""Does this follow-up need the graph again, or just the answer already on screen?

Deviation D13. This is the single most consequential decision in the
conversational layer, for both cost and latency:

- **`discuss`** — *"what does HHI mean?"*, *"say that in three bullets"*, *"why is
  the confidence low?"*. Answered from the previous turn's own answer, figures
  and evidence with one cheap call. No fan-out, no Cypher, ~1-2s.
- **`analyse`** — *"now do the same for rubber"*, *"what about 2024?"*. Rewritten
  into a standalone question and put through the full graph.

**Ambiguity resolves to `analyse`, deliberately.** The two failure modes are not
symmetric. Misrouting a discussion into `analyse` costs latency and a few cents.
Misrouting a new question into `discuss` answers it from the *previous* question's
evidence — a confident, well-cited answer to something the user did not ask,
which is precisely the failure `orchestrator/grounding.py` exists to prevent in
prose. `keyword_route` makes the same trade for the same reason.

Two tiers, exactly like `orchestrator/router.py`: `keyword_turn` is deterministic,
free and always available; `llm_turn` upgrades it and falls back to the keyword
decision on any failure. The keyword tier is not a placeholder — it is what runs
in degraded mode (SRS 3.4.3), and it is what the LLM tier is measured against.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from ceynex.agents.common import find_region, parse_intent

log = logging.getLogger(__name__)

TurnMode = Literal["discuss", "analyse"]

#: Phrases that redirect the question at new subject matter. Checked first,
#: because "what about rubber" contains "what about" *and* reads like a
#: question about the previous answer until you notice it names a new item.
ANALYSE_CUES = (
    "what about", "how about", "and for", "same for", "do the same",
    "instead", "compare", "versus", " vs ", "now do", "next year",
    "last year", "this year", "forecast", "predict", "project",
)

#: Phrases that ask about the answer already given rather than about the world.
DISCUSS_CUES = (
    "what does", "what do you mean", "what is that", "explain", "why",
    "how confident", "how did you", "where did", "which source", "source for",
    "summarise", "summarize", "shorter", "simpler", "in bullet", "bullet point",
    "rephrase", "elaborate", "more detail", "expand on", "tell me more",
    "break that down", "in plain english", "what does it mean",
)

CLASSIFY_SYSTEM = """You decide how to handle a follow-up in a conversation about
Sri Lanka's export economy.

Return JSON: {"mode": "discuss" | "analyse", "standalone_query": "..."}

"discuss" — the follow-up is about the answer that was just given: asking what a
term means, why the confidence is what it is, where a figure came from, or for it
to be reworded, shortened or expanded. No new data is needed.

"analyse" — the follow-up asks about something the previous answer does not
already contain: a different commodity, a different market, a different period, a
forecast that was not run. New data is needed.

When it could be either, choose "analyse". Answering a new question from the old
question's evidence is far worse than looking the answer up again.

For "analyse", `standalone_query` must be the follow-up rewritten so it makes
sense on its own, carrying forward whatever the follow-up left implicit. For
"discuss", set `standalone_query` to null."""


@dataclass
class TurnDecision:
    mode: TurnMode
    #: Populated for `analyse` only — the follow-up made self-contained.
    standalone_query: str | None
    #: "keyword" | "llm" | "llm->keyword". Surfaced in the trace so a reader can
    #: tell a model's judgement from a word list's.
    method: str
    reason: str = ""


def _mentions(text: str, cues: tuple[str, ...]) -> str | None:
    lowered = f" {text.lower()} "
    for cue in cues:
        if cue in lowered:
            return cue
    return None


def _new_subject(follow_up: str, prior_query: str) -> str | None:
    """What the follow-up names that the previous question did not.

    The most reliable deterministic signal there is: a follow-up that introduces
    a commodity, a market, a region or a year the last question never mentioned
    cannot be answered from the last question's evidence, whatever it sounds like.
    """
    new, old = parse_intent(follow_up), parse_intent(prior_query)

    if new.item and new.item != old.item:
        return f"names {new.item}, previously {old.item or 'nothing'}"
    if new.partner and new.partner != old.partner:
        return f"names partner {new.partner}, previously {old.partner or 'none'}"
    if new.year and new.year != old.year:
        return f"names {new.year}, previously {old.year or 'no year'}"

    region_new, region_old = find_region(follow_up), find_region(prior_query)
    if region_new and region_new != region_old:
        return f"names region {region_new}"
    return None


#: Words that carry no subject matter — pronouns, discourse markers, politeness.
#: A follow-up built entirely from these cannot be asking about anything new.
_DISCOURSE_ONLY = frozenset(
    # Written as prose rather than ~85 quoted list items on purpose: the whole
    # value of this list is that a reviewer can scan it for a word that should
    # not be here, and four screens of commas defeats that.
    """
    a an and again also anything back but can continue could do does else
    expand explain further go going he her him his how i is it its just keep
    like little me more much my no now of ok okay on one or please really
    right said say see she so some sorry sure tell than that thanks the their
    them then there these they thing things this those to too us was we well
    what when where which who why will with yes you your
    """.split()  # noqa: SIM905 - see above
)


def _is_discourse_only(text: str) -> bool:
    """Whether the follow-up names nothing at all — *"go on"*, *"and then?"*.

    Deliberately **not** `parse_intent`-based. That parser recognises the
    commodities and countries the graph holds, which makes it far too narrow a
    test for "could this be a new question": *"the EU figures"* resolves to no
    item, no partner and no region (`find_region` covers the five continents,
    and the EU is not one), so a `parse_intent` test reads it as contentless and
    answers it from the *previous* market's evidence. Found by an assertion, and
    it is exactly the failure this module exists to avoid.

    So the test is inverted — positive evidence of emptiness rather than absence
    of evidence of content. Anything with a real noun in it goes to `analyse`.
    """
    if re.search(r"\b(19|20)\d{2}\b", text):
        return False
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and all(word in _DISCOURSE_ONLY for word in words)


def keyword_turn(follow_up: str, prior_query: str) -> TurnDecision:
    """Deterministic, free, and what runs when there is no LLM key."""
    subject = _new_subject(follow_up, prior_query)
    if subject:
        return TurnDecision("analyse", follow_up, "keyword", f"new subject: {subject}")

    cue = _mentions(follow_up, ANALYSE_CUES)
    if cue:
        return TurnDecision("analyse", follow_up, "keyword", f"redirection cue: {cue!r}")

    discuss_cue = _mentions(follow_up, DISCUSS_CUES)
    if discuss_cue:
        return TurnDecision("discuss", None, "keyword", f"discussion cue: {discuss_cue!r}")

    if _is_discourse_only(follow_up):
        return TurnDecision("discuss", None, "keyword", "names nothing new to analyse")

    return TurnDecision("analyse", follow_up, "keyword", "ambiguous — defaulting to analyse")


async def llm_turn(follow_up: str, prior_query: str, prior_answer: str, llm) -> TurnDecision:
    """The LLM's judgement, or the keyword decision if it cannot be had.

    Routing *never* fails: every path below ends in a usable decision, matching
    `router.llm_route`'s contract. A conversation that stops because a classifier
    was unavailable would be a worse outcome than one that occasionally re-runs a
    query it did not need to.
    """
    fallback = keyword_turn(follow_up, prior_query)

    if llm is None or not getattr(llm, "available", False):
        return fallback

    user = (
        f"Previous question: {prior_query}\n"
        f"Previous answer: {prior_answer[:1500]}\n"
        f"Follow-up: {follow_up}"
    )
    try:
        raw = await llm.generate("turn_classify", CLASSIFY_SYSTEM, user, json_mode=True)
    except Exception as exc:  # noqa: BLE001 - classification must never fail a turn
        log.warning("turn classification raised, using keywords: %s", exc)
        fallback.method = "llm->keyword"
        return fallback

    decision = _parse(raw, follow_up)
    if decision is None:
        fallback.method = "llm->keyword"
        return fallback
    return decision


def _parse(raw: str | None, follow_up: str) -> TurnDecision | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.info("turn classifier returned unparseable JSON")
        return None
    if not isinstance(payload, dict):
        return None

    mode = payload.get("mode")
    if mode not in ("discuss", "analyse"):
        return None

    if mode == "discuss":
        return TurnDecision("discuss", None, "llm", "model judged it a discussion")

    standalone = payload.get("standalone_query")
    if not isinstance(standalone, str) or not standalone.strip():
        # An `analyse` with no rewritten question is unusable — the graph would
        # receive "what about that one?" and route on nothing. Falling back to
        # the raw follow-up loses conversational context, so let the caller use
        # the keyword decision instead.
        log.info("turn classifier chose analyse without a standalone query")
        return None
    return TurnDecision("analyse", standalone.strip(), "llm", "model judged it a new question")


__all__ = [
    "ANALYSE_CUES",
    "DISCUSS_CUES",
    "TurnDecision",
    "TurnMode",
    "keyword_turn",
    "llm_turn",
]
