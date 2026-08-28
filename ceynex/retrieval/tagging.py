"""Supports SRS 3.1.9 (deviation D10) — deterministic entity tagging for policy chunks.

What turns an ordinary vector store into a graph-anchored one. Every chunk is
tagged with the graph entities it is about — countries as `iso3`, goods as HS
prefixes, agreements by their `TradeAgreement.name` — so a search can be filtered
to the entities an agent has already resolved from Neo4j instead of run blind
over the whole corpus.

**Keyword rules, not an LLM.** Two reasons, and the second is the real one:

1. Tagging runs over every chunk of every document. An LLM call per chunk is
   minutes and money for a step that has to be re-runnable on a whim.
2. **A mislabelled chunk is worse than an untagged one.** An untagged chunk is
   merely never retrieved; a chunk wrongly tagged `iso3=USA` will be retrieved
   *for* US questions and cited as US policy. A rule that misses is recoverable,
   a rule that lies is not, so every rule here is written to under-claim.

Imported by the offline pipeline in `trade-data-pipeline/`, which is why it lives
in the installed package rather than beside those scripts: the tags the indexer
writes and the filters the agent queries with have to come from the same table.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ceynex.data.crosswalk import _countries

#: Item name -> the HS code that item's trade is recorded under. The single
#: definition for the whole system: `trade_economics._hs_for_item` delegates
#: here, and the offline enricher tags chunks with it, so a question about
#: cinnamon and a chunk about cinnamon resolve to the same 0906.
HS_FOR_ITEM: dict[str, str] = {
    "tea": "0902",
    "cinnamon": "0906",
    "rubber": "4001",
    "coconut": "1513",
    "apparel_knit": "61",
    "apparel_woven": "62",
}

#: Words that identify an item in running prose. Wider than
#: `agents.common.ITEM_KEYWORDS` because policy documents use formal register —
#: "natural rubber", "coconut oil" — where a user query says "rubber".
ITEM_WORDS: dict[str, tuple[str, ...]] = {
    "tea": ("tea", "ceylon tea", "black tea", "green tea"),
    "cinnamon": ("cinnamon", "cassia", "spices", "spice"),
    "rubber": ("rubber", "natural rubber", "latex"),
    "coconut": ("coconut", "copra", "coir", "desiccated coconut"),
    "apparel_knit": ("knitted", "knitwear", "t-shirt", "jersey", "hosiery"),
    "apparel_woven": ("woven", "trousers", "not knitted"),
}

#: Generic apparel words tag both chapters. A document saying "garments" means
#: 61 and 62 together, and picking one would answer half the question.
APPAREL_WORDS = ("apparel", "garment", "garments", "clothing", "textile", "textiles", "ready-made")

#: "HS 6109", "HS code 61", "Chapter 62", "heading 0902".
HS_MENTION = re.compile(
    r"\b(?:hs\s*(?:code)?|chapter|heading|tariff\s+line)\s*[:.]?\s*(\d{2,10})\b",
    re.IGNORECASE,
)

#: Alternative names for the agreements in `trade_agreements.csv`. A document
#: almost never uses the short name the graph stores: the EU writes "Generalised
#: Scheme of Preferences", the UK writes "Developing Countries Trading Scheme".
#: Without these the `agreement` tag would be empty on precisely the documents
#: that discuss the agreement in most depth.
AGREEMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "GSP+": ("gsp+", "gsp plus", "generalised scheme of preferences", "generalized scheme of preferences", "gsp"),
    "UK DCTS": ("dcts", "developing countries trading scheme"),
    "ISFTA": ("isfta", "india-sri lanka free trade agreement", "india–sri lanka free trade agreement"),
    "PSFTA": ("psfta", "pakistan-sri lanka free trade agreement", "pakistan–sri lanka free trade agreement"),
    "SAFTA": ("safta", "south asian free trade area"),
    "APTA": ("apta", "asia-pacific trade agreement", "asia pacific trade agreement"),
}

#: Keyword -> measure type, checked in this order. First match wins, so the more
#: specific categories are listed before the broad ones: a sentence about a
#: tariff *under* an FTA is a tariff statement, which is what a simulation needs.
MEASURE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tariff", ("tariff", "duty", "duties", "customs duty", "mfn", "most favoured nation",
                "most favored nation", "ad valorem", "tariff rate quota", "bound rate")),
    ("fta", ("free trade agreement", "preferential", "preference", "gsp", "dcts", "fta",
             "trade agreement", "duty-free", "duty free", "rules of origin")),
    ("ntm", ("sanitary", "phytosanitary", "sps", "technical barrier", "tbt", "non-tariff",
             "quota", "licensing", "standards", "certification", "traceability")),
    ("export_promotion", ("export promotion", "export development", "trade mission",
                          "export credit", "market access support", "exporter support")),
    ("investment", ("investment", "fdi", "foreign direct investment", "bilateral investment")),
)


@lru_cache(maxsize=1)
def _agreement_names() -> tuple[str, ...]:
    """Canonical agreement names, from the same CSV the KG loader reads.

    Imported rather than re-listed: a name that does not match
    `TradeAgreement.name` exactly produces no `DESCRIBES` edge and no usable
    filter, and it fails silently in both directions.
    """
    from ceynex.kg.loaders.trade_agreements import agreement_rows

    return tuple(row["name"] for row in agreement_rows())


@lru_cache(maxsize=1)
def _country_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """(iso3, word-boundary pattern), longest name first.

    Longest-first for the reason `agents.common._find_partner` documents: "India"
    must not match inside "Indian Ocean" ahead of a longer name, and short names
    are skipped entirely because a two-letter match in running prose is noise.
    """
    ordered = sorted(_countries(), key=lambda c: -len(c.name))
    return tuple(
        (country.iso3, re.compile(rf"\b{re.escape(country.name.lower())}\b"))
        for country in ordered
        if len(country.name) >= 4
    )


def countries_in(text: str) -> tuple[str, ...]:
    """ISO3 codes for every country named in the text, in first-seen order."""
    lowered = text.lower()
    found: list[str] = []
    for iso3, pattern in _country_patterns():
        if iso3 not in found and pattern.search(lowered):
            found.append(iso3)
    return tuple(found)


def hs_prefixes_in(text: str) -> tuple[str, ...]:
    """HS prefixes the text is about — explicit codes first, then item words.

    Explicit codes are expanded to every level of their hierarchy, the same way
    `kg.queries._hs_prefixes` does, because a document that names 610910 is also
    about 6109 and 61 and a question asked at either level should find it.
    """
    from ceynex.kg.queries import _hs_prefixes

    found: list[str] = []

    for raw in HS_MENTION.findall(text):
        # Odd digit counts are not HS levels — "chapter 615" is a section number
        # in someone's document, not a tariff line, and expanding it would tag
        # the chunk with HS 61.
        if len(raw) % 2:
            continue
        try:
            expanded = _hs_prefixes(raw)
        except Exception:  # noqa: BLE001 - a number that is not an HS code is ordinary
            continue
        for code in expanded:
            if code not in found:
                found.append(code)

    lowered = text.lower()
    for item, words in ITEM_WORDS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in words):
            code = HS_FOR_ITEM[item]
            if code not in found:
                found.append(code)

    if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in APPAREL_WORDS):
        for code in ("61", "62"):
            if code not in found:
                found.append(code)

    return tuple(found)


def agreements_in(text: str) -> tuple[str, ...]:
    """Canonical `TradeAgreement.name` values the text refers to."""
    lowered = text.lower()
    found: list[str] = []
    for name in _agreement_names():
        aliases = AGREEMENT_ALIASES.get(name, ()) + (name.lower(),)
        if any(re.search(rf"(?<![\w+]){re.escape(alias)}(?![\w])", lowered) for alias in aliases):
            found.append(name)
    return tuple(found)


def measure_type_of(text: str) -> str:
    """Which kind of measure the chunk is about. `other` when nothing matches.

    `other` is a real answer, not a failure: most of a trade strategy is context,
    objectives and prose. Those chunks stay searchable — nothing filters them
    out — they are simply never the answer to "what tariff applies".
    """
    lowered = text.lower()
    for measure, keywords in MEASURE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return measure
    return "other"


__all__ = [
    "AGREEMENT_ALIASES",
    "APPAREL_WORDS",
    "HS_FOR_ITEM",
    "ITEM_WORDS",
    "MEASURE_KEYWORDS",
    "agreements_in",
    "countries_in",
    "hs_prefixes_in",
    "measure_type_of",
]
