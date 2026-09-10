"""Answering a follow-up from what is already on screen — deviation D13.

The `analyse` path re-runs the graph and is just `query_runner.run_query`. This
module is the other half: the `discuss` path, which answers from the previous
turn's own answer, figures and evidence without touching Neo4j, Postgres or the
five-agent fan-out.

**Grounding still applies, and this is the important part.** A discussion is
still prose containing figures, so it goes through the same
`orchestrator/grounding.py::ungrounded_figures` check the merger uses — against
the *previous turn's* corpus. Without it, "summarise that in three bullets" is an
invitation for a model to round 4.2 to 4, or to helpfully add a number nobody
sourced, and it would arrive wearing the confidence badge of an answer that was
properly sourced. The corpus is what was actually shown; anything outside it is
not a summary.

**Degraded mode returns the previous answer rather than nothing** (SRS 3.4.3).
That is a real, if plain, form of the feature: the evidence panel is unchanged
and the user is told the discussion needs a model. It is not a silent failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ceynex.chat.store import Message
from ceynex.observability import trace
from ceynex.orchestrator.grounding import ungrounded_figures

log = logging.getLogger(__name__)

#: How much of the prior answer and evidence the discuss call may see. Generous
#: — it is one cheap-model call and the whole point is that it has the context —
#: but bounded, because a conversation about a long answer runs this every turn.
MAX_CONTEXT_CHARS = 6000

DISCUSS_SYSTEM = """You are discussing an analysis of Sri Lanka's export economy
that has already been produced. The reader is looking at it.

You are given the question that was asked, the answer that was given, and the
evidence behind it. Answer the reader's follow-up about that material.

Absolute rules:
1. Every number you state must already appear in the material you were given.
   Never estimate, never round to a different figure, never add one from your own
   knowledge. If the follow-up asks for a number that is not there, say it was
   not part of this analysis.
2. Never contradict the answer you were given. If you think it is wrong, say what
   in the evidence makes you doubt it — do not quietly correct it.
3. If the follow-up cannot be answered from this material, say so plainly and say
   what would need to be looked up. Do not guess.
4. Answer only what was asked. A request to shorten something is not a request to
   re-analyse it.
5. Plain English for a policymaker who is not an economist.
6. You describe data. You do not give financial, legal or investment advice."""


@dataclass
class DiscussResult:
    """A follow-up answered without re-running the graph."""

    answer: str
    degraded: bool
    grounded: bool = True
    #: Carried forward unchanged: the discussion is *about* this evidence, so
    #: the panel beside it must stay the panel it is discussing.
    evidence: list[dict[str, Any]] = field(default_factory=list)
    rejected_figures: list[str] = field(default_factory=list)


def _corpus(prior: Message, prior_query: str = "") -> list[str]:
    """Everything a discussion is allowed to state a figure from.

    Exactly what the reader can see: the question they asked, the answer they
    got, and the evidence beside it. A figure outside that set was not in the
    analysis, so a summary containing it is not a summary of the analysis.

    **The question is part of the corpus**, and leaving it out was a real
    over-rejection found end to end. Asked "how did cinnamon exports to Germany
    change in 2025", the analysis declined without repeating the year; the
    follow-up "explain that more simply" then said "2025", and a corpus built
    only from the answer discarded a perfectly good reply. Any follow-up echoing
    the year or market the user typed would have hit the same wall.
    """
    corpus = [prior.content, prior_query]
    for item in prior.evidence or []:
        corpus.append(str(item.get("claim", "")))
        corpus.append(str(item.get("detail", "")))
    for point in prior.forecast or []:
        corpus.append(
            f"{point.get('period')} {point.get('point')} "
            f"{point.get('lower')} {point.get('upper')}"
        )
    if prior.confidence is not None:
        # So "why is the confidence 61%?" can restate the number it is about.
        corpus.append(f"confidence {prior.confidence} {round(prior.confidence * 100)}%")
    return corpus


def _context(prior: Message, prior_query: str) -> str:
    lines = [
        f"Question asked: {prior_query}",
        f"Answer given: {prior.content}",
    ]
    if prior.confidence is not None:
        lines.append(f"Confidence: {round(prior.confidence * 100)}% ({prior.confidence_band})")
    if prior.agents_used:
        lines.append(f"Analyses that contributed: {', '.join(prior.agents_used)}")
    if prior.unanswered:
        lines.append(f"Not answered: {'; '.join(prior.unanswered)}")
    if prior.evidence:
        lines.append("Evidence:")
        for item in prior.evidence:
            lines.append(
                f"  - [{item.get('source_id')}] {item.get('claim')} :: {item.get('detail', '')[:300]}"
            )
    if prior.forecast:
        lines.append("Forecast:")
        for point in prior.forecast:
            lines.append(
                f"  - {point.get('period')}: {point.get('point')} "
                f"({point.get('lower')}-{point.get('upper')} {point.get('unit', '')})"
            )
    return "\n".join(lines)[:MAX_CONTEXT_CHARS]


def _degraded_answer(prior: Message) -> str:
    return (
        "Answering follow-up questions needs the language model, which is not "
        "available right now. The analysis and its evidence are unchanged and "
        "still shown below — the figures in it were not produced by the model."
    )


async def discuss(follow_up: str, prior: Message, prior_query: str, llm) -> DiscussResult:
    """Answer a follow-up from the previous turn's material.

    Never raises. A failure here degrades to the previous answer, the same
    contract every agent node honours (SAD §4.1).
    """
    evidence = list(prior.evidence or [])

    if llm is None or not getattr(llm, "available", False):
        trace.emit("discuss", status="degraded", reason="no llm available")
        return DiscussResult(_degraded_answer(prior), degraded=True, evidence=evidence)

    user = f"{_context(prior, prior_query)}\n\nFollow-up: {follow_up}"
    try:
        text = await llm.generate("chat", DISCUSS_SYSTEM, user)
    except Exception as exc:  # noqa: BLE001 - a follow-up must never fail the conversation
        log.warning("discuss call raised: %s", exc)
        trace.emit("discuss", status="degraded", reason=str(exc))
        return DiscussResult(_degraded_answer(prior), degraded=True, evidence=evidence)

    if not text:
        trace.emit("discuss", status="degraded", reason="model returned nothing")
        return DiscussResult(_degraded_answer(prior), degraded=True, evidence=evidence)

    rejected = ungrounded_figures(text, _corpus(prior, prior_query))
    if rejected:
        # Same posture as `merger._reject_ungrounded_prose`: the prose is
        # discarded whole rather than patched. A summary that invented a figure
        # is not a summary with one bad number in it — it is evidence that the
        # model was not reading the material.
        log.warning("discuss prose stated ungrounded figures %s; discarding", rejected)
        trace.emit("discuss", status="rejected", ungrounded=rejected)
        return DiscussResult(
            "That follow-up could not be answered without stating figures this "
            "analysis did not produce, so the original answer is shown unchanged.",
            degraded=True,
            grounded=False,
            evidence=evidence,
            rejected_figures=rejected,
        )

    trace.emit("discuss", status="ok", chars=len(text))
    return DiscussResult(text, degraded=False, evidence=evidence)


__all__ = ["DISCUSS_SYSTEM", "MAX_CONTEXT_CHARS", "DiscussResult", "discuss"]
