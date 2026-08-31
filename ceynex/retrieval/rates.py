"""Supports SRS 3.1.5 (deviation D10) — reading a tariff rate out of policy text.

The one place in this feature where a **number** is taken from retrieved prose
rather than from the graph, and therefore the one place that can produce the
failure `docs/EVALUATION.md` §5 names as the limitation the runtime guard cannot
catch: *a correctly-sourced figure attached to the wrong claim*.
`orchestrator/grounding.py` compares digit strings, so a real percentage lifted
from the wrong sentence passes every check the system has and is wrong anyway.
S06 is the live example of that failing in the other direction.

So this module is deliberately built to **refuse more often than it answers**.
Four conditions, all required:

1. the number is a percentage;
2. its sentence also carries a tariff word — a bare "12%" in a document about
   export growth is not a tariff;
3. its sentence names the goods in question, by HS code or by item word. This is
   the condition that does the real work: it is what stops a rate quoted for
   footwear being served as the rate for tea;
4. exactly **one** distinct rate survives. Two candidates means the passage is
   ambiguous, and picking the first is how a plausible wrong number gets into an
   answer with a real citation attached to it.

Failing any of them returns None, and the caller falls back to the documented
constant in `config/elasticities.yaml` — stating which it used, either way. A
refusal here costs a less precise answer. A wrong acceptance costs a wrong answer
that looks better sourced than the right one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ceynex.retrieval.schema import PolicyChunk
from ceynex.retrieval.tagging import APPAREL_WORDS, HS_FOR_ITEM, ITEM_WORDS

#: A percentage: "16.5%", "16.5 per cent", "16.5 percent".
PERCENT = re.compile(r"(\d{1,2}(?:\.\d{1,2})?)\s*(?:%|per\s?cent)", re.IGNORECASE)

#: Sentence-ish split. Deliberately crude — the unit only has to be small enough
#: that a rate and its subject being in the same one means something.
SENTENCE = re.compile(r"(?<=[.;:!?])\s+|\n+")

TARIFF_WORDS = (
    "tariff", "duty", "duties", "mfn", "most favoured nation", "most favored nation",
    "ad valorem", "customs", "bound rate", "applied rate",
)

#: Above this a "percentage" is almost certainly a share, a growth rate or a
#: target — not an import tariff on these goods. Under it, 0 is excluded because
#: "0% duty" is a statement about preference, not a rate to re-impose.
MAX_PLAUSIBLE_RATE = 60.0
MIN_PLAUSIBLE_RATE = 0.1


@dataclass(frozen=True)
class SourcedRate:
    """A tariff rate read from a document, with the sentence that stated it."""

    rate: float  #: fraction, e.g. 0.165
    sentence: str
    chunk: PolicyChunk

    @property
    def claim(self) -> str:
        return (
            f"{self.chunk.publisher} states a tariff of {self.rate * 100:.1f}% in "
            f"{self.chunk.title}: \"{self.sentence.strip()[:200]}\""
        )


def _goods_words(hs_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """Words and codes that mean "the goods this question is about"."""
    words: list[str] = list(hs_prefixes)
    for item, code in HS_FOR_ITEM.items():
        if any(code.startswith(p) or p.startswith(code) for p in hs_prefixes):
            words.extend(ITEM_WORDS.get(item, ()))
            if code in ("61", "62"):
                words.extend(APPAREL_WORDS)
    return tuple(dict.fromkeys(words))


def extract_tariff_rate(
    chunks: list[PolicyChunk], hs_prefixes: tuple[str, ...]
) -> SourcedRate | None:
    """The one unambiguous tariff rate for these goods, or None.

    None is the expected result most of the time and is not a failure — see the
    module docstring. The caller must fall back to the configured constant and
    say so rather than treating a missing rate as a reason not to answer.
    """
    goods = _goods_words(hs_prefixes)
    if not goods:
        return None

    found: dict[float, SourcedRate] = {}
    for chunk in chunks:
        for sentence in SENTENCE.split(chunk.text):
            lowered = sentence.lower()
            if not any(word in lowered for word in TARIFF_WORDS):
                continue
            if not any(word.lower() in lowered for word in goods):
                continue
            for raw in PERCENT.findall(sentence):
                value = float(raw)
                if not MIN_PLAUSIBLE_RATE <= value <= MAX_PLAUSIBLE_RATE:
                    continue
                found.setdefault(value, SourcedRate(value / 100.0, sentence, chunk))

    # Condition 4. Ambiguity is reported as "no rate", never resolved by picking.
    if len(found) != 1:
        return None
    return next(iter(found.values()))


__all__ = ["MAX_PLAUSIBLE_RATE", "MIN_PLAUSIBLE_RATE", "SourcedRate", "extract_tariff_rate"]
