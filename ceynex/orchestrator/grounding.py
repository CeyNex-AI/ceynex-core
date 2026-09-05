"""Implements SRS 3.1.3 — every figure in an answer must come from retrieved facts.

This is the schedule's activity 072, "add the validation pass constraining
answers to retrieved facts". Until an LLM key was configured the requirement was
satisfied for free: with no model available the merge fell back to
`compose_deterministic`, which can only restate summaries the agents wrote, so
there was nothing to constrain. Once prose generation is on, a composed sentence
can carry a number no agent ever reported, and nothing downstream would notice.

The rule this module enforces is narrow and checkable: **a digit string in the
prose must appear somewhere in what the composer was given.** It is not a
semantic check. It does not know whether a correctly-sourced figure is attached
to the wrong claim — `docs/EVALUATION.md` states that limitation and it still
holds. What it catches is the failure that matters most and is otherwise
invisible: a figure that exists nowhere but in the model's output.

Why the comparison lives here rather than in `eval/harness.py`, which had it
first: the harness measures a metric and the merger enforces a guarantee, and
those two must agree on what counts as "a figure" or the reported number would
be scoring something other than the thing being enforced. Same reasoning as
`confidence.py` owning the confidence formula outright — one definition, and
nothing else is allowed to invent its own.

The two callers deliberately differ in **what they compare against**, and only
in that:

- `eval.harness.ungrounded` uses the merged evidence alone. That is the stricter
  external claim — "every figure is traceable to a cited source" — and it is the
  number that belongs in the evaluation report.
- `merger._reject_ungrounded_prose` uses everything the merge LLM was actually
  shown: the question, each finding's summary, its figures, its assumptions, and
  the evidence. Holding the model to facts it was never given would reject
  correct prose. The clearest case is a shock magnitude quoted back from the
  question itself ("a 10% tariff") — it is in no evidence entry, it is not
  invented, and `docs/EVALUATION.md` records exactly that as 2 of the 2
  ungrounded figures in the 30-question run.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# An integer or decimal, optional thousands separators, optional sign.
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

# Digit strings this short are structural rather than claims: a 3-year horizon,
# 12 markets, a 10% shock. Requiring evidence for them would flag every answer
# and the check would stop discriminating.
STRUCTURAL_DIGIT_LIMIT = 2


def _normalise(raw: str) -> str:
    return raw.replace(",", "")


def numbers_in(text: str) -> set[str]:
    """Every figure in `text`, comma-stripped so 1,234 and 1234 compare equal."""
    return {_normalise(match) for match in NUMBER.findall(text or "")}


def corpus_texts(
    *,
    summary: str | None = None,
    figures: dict[str, object] | None = None,
    evidence: Iterable[dict[str, object]] | None = None,
    assumptions: Iterable[str] | None = None,
) -> list[str]:
    """Every text form a set of findings could restate a figure in.

    The one place this rendering lives, shared by `merger._grounding_corpus`
    (checks the merge LLM's prose against every contributing finding) and
    `agents.common.finish` (checks a single agent's own explanation prose
    against that same agent's own findings, before the explanation ever
    reaches the merger) — two different scopes, same figure-matching rule,
    so the rendering can't drift between them.

    Figures are rendered in multiple forms deliberately: a prompt shows a
    large number as `f"{value:,.4g}"`, so it can honestly come back as
    "1.235 billion", while a plain summary sentence would contain the same
    value's `f"{value:,.2f}"` form. Offering only one spelling would reject
    correct prose for restating a figure in the form it was actually given.
    """
    texts: list[str] = []
    if summary:
        texts.append(summary)
    for value in (figures or {}).values():
        texts.append(str(value))
        if isinstance(value, int | float):
            texts.append(f"{value:,.4g}")
            texts.append(f"{value:,.2f}")
    for item in evidence or []:
        texts.append(str(item.get("claim", "")))
        texts.append(str(item.get("detail", "")))
    texts.extend(assumptions or [])
    return texts


def ungrounded_figures(answer: str, corpus: Iterable[str]) -> list[str]:
    """Figures in `answer` that appear nowhere in `corpus`, in source order.

    Deliberately crude and deliberately generous, in that order:

    - **Crude** — it compares digit strings, not meanings. A figure rounded
      differently in the prose than in its source is reported as ungrounded.
    - **Generous** — a prose figure is accepted if any corpus figure merely
      *starts with* its integer part, so 1234.5 is grounded by 1234.52.

    Generosity is the right direction for a check that rejects an answer: a
    false alarm costs the reader plainer wording, a miss costs the claim. The
    same trade is why the harness over-reports rather than under-reports.
    """
    grounded: set[str] = set()
    for text in corpus:
        grounded |= numbers_in(text)

    missing: list[str] = []
    for raw in NUMBER.findall(answer or ""):
        value = _normalise(raw)
        if len(value.lstrip("-").replace(".", "")) <= STRUCTURAL_DIGIT_LIMIT:
            continue
        if value in grounded:
            continue
        if any(grounded_value.startswith(value.split(".")[0]) for grounded_value in grounded):
            continue
        missing.append(raw)
    return missing


__all__ = ["NUMBER", "corpus_texts", "numbers_in", "ungrounded_figures"]
