"""Naming a conversation from its first exchange — deviation D13.

A sidebar of rows all reading "cinnamon export trend…" is a sidebar nobody can
navigate, so the first turn earns the conversation a short name.

**Never blocks a turn and never fails one.** The title is written after the
answer has already been delivered, and every failure path here ends in a name
derived from the question itself. A conversation with a plain title is fine; a
conversation that failed to answer because naming it went wrong is not.

**Only ever set once, and never over a name the user chose.** The write is
conditional in SQL (`WHERE title IS NULL`) rather than checked first and written
second — between a read and a write the user may have renamed it from another
tab, and quietly overwriting that is the kind of bug that looks like the app
losing data.
"""

from __future__ import annotations

import logging

from ceynex.chat.store import MAX_TITLE

log = logging.getLogger(__name__)

#: Cut a fallback title here rather than at MAX_TITLE, so it reads as a
#: deliberately short name instead of a sentence that ran out of room.
FALLBACK_CHARS = 48

TITLE_SYSTEM = """You name a conversation about Sri Lanka's export economy from
its first question and answer.

Return only the name. No quotes, no punctuation at the end, no preamble.

Rules:
1. Two to six words. It goes in a narrow sidebar.
2. Name the subject, not the activity: "Cinnamon exports to Germany", never
   "Analysis of cinnamon exports" or "A question about cinnamon".
3. Use the specific commodity, market or period if the question named one.
4. Never invent a figure, a country or a period that was not in the exchange."""


def fallback_title(question: str) -> str:
    """The question itself, trimmed to something sidebar-sized.

    What runs with no LLM key (SRS 3.4.3), and the safety net for every other
    failure. Cuts on a word boundary — a name ending mid-word reads as a bug.
    """
    cleaned = " ".join(question.strip().split())
    if len(cleaned) <= FALLBACK_CHARS:
        return cleaned or "New conversation"

    cut = cleaned[:FALLBACK_CHARS].rsplit(" ", 1)[0]
    return (cut or cleaned[:FALLBACK_CHARS]) + "…"


def _clean(raw: str) -> str | None:
    """A usable name from the model's reply, or None to fall back.

    Models like to answer a naming request with a sentence. One line, no
    surrounding quotes, no trailing full stop, and nothing long enough to be
    prose rather than a name.
    """
    name = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    name = name.strip("\"'").rstrip(".").strip()
    if not name or len(name) > MAX_TITLE:
        return None
    if len(name.split()) > 8:
        return None
    return name


async def title_for(question: str, answer: str, llm) -> str:
    """A short name for this conversation. Always returns something usable."""
    if llm is None or not getattr(llm, "available", False):
        return fallback_title(question)

    user = f"Question: {question}\nAnswer: {answer[:600]}"
    try:
        raw = await llm.generate("title", TITLE_SYSTEM, user)
    except Exception as exc:  # noqa: BLE001 - naming must never fail a turn
        log.info("title generation raised, using the question: %s", exc)
        return fallback_title(question)

    if not raw:
        return fallback_title(question)
    return _clean(raw) or fallback_title(question)


__all__ = ["FALLBACK_CHARS", "TITLE_SYSTEM", "fallback_title", "title_for"]
