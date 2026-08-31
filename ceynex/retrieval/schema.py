"""Supports SRS 3.1.4 and 3.1.9 (deviation D10) — the policy-chunk payload contract.

The shape of a Qdrant point, written down once. The offline indexer in
`trade-data-pipeline/` produces it and `retrieval/client.py` consumes it, and if
those two ever disagree about a field name the search silently returns nothing —
a filter on a payload key that does not exist matches no points and raises
nothing. That failure is invisible, which is why this is a module rather than a
convention.

Same reasoning as `orchestrator/grounding.py` owning the definition of "a figure"
for both the merger and the harness: one definition, and nothing else is allowed
to invent its own.

**No qdrant imports here, deliberately.** `qdrant-client` is the optional
`[policy]` extra, so a module every test can import must not need it. The client
does the qdrant-specific work; this module is constants, a dataclass, and the
filter *description* that goes into `Evidence.detail`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

# --- the models ----------------------------------------------------------
#
# All three are served by `fastembed` (ONNX). Verified present in fastembed
# 0.8.0; the reranker lives at `fastembed.rerank.cross_encoder`, not at the
# package root.

DENSE_MODEL = "BAAI/bge-base-en-v1.5"
DENSE_DIM = 768
SPARSE_MODEL = "Qdrant/bm25"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# Named vectors on the point. Both are required: policy text turns on exact
# tokens — "GSP+", "HS 6109", "MFN", "DCTS" — and a dense embedding blurs
# precisely those into their neighbours, which is the failure mode that makes a
# tea question return apparel text. BM25 does not blur them. Neither retriever
# is good enough alone, so the collection carries both and fuses at query time.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# --- payload keys --------------------------------------------------------
#
# Named rather than spelled inline, because a typo in a filter key is a silent
# empty result rather than an error.

DOC_ID = "doc_id"
CHUNK_INDEX = "chunk_index"
TEXT = "text"
PAGE = "page"
SECTION = "section"
ISO3 = "iso3"
HS_PREFIX = "hs_prefix"
MEASURE_TYPE = "measure_type"
AGREEMENT = "agreement"
TITLE = "title"
PUBLISHER = "publisher"
URL = "url"
LANGUAGE = "language"

#: Payload fields that get a Qdrant index. Every one of these is filtered on at
#: query time; without an index Qdrant falls back to a full scan of the payload,
#: which at this corpus size is survivable but wastes the 2 s budget the agent
#: has to spend.
INDEXED_KEYWORD_FIELDS = (DOC_ID, ISO3, HS_PREFIX, MEASURE_TYPE, AGREEMENT, LANGUAGE)

#: What kind of measure a chunk is about. Assigned by keyword in the offline
#: `enrich.py`, never by an LLM — a mislabelled chunk is worse than an untagged
#: one, because the filter will then exclude it from the query that needed it.
#: `other` is the honest default and is never filtered *for*, only left in.
MEASURE_TYPES = (
    "tariff",
    "ntm",
    "fta",
    "export_promotion",
    "investment",
    "other",
)

#: The measures a preference-loss or tariff simulation cares about. Pulled out
#: here so the agent and any future caller ask the same question.
SIMULATION_MEASURES = ("tariff", "fta")


@dataclass(frozen=True)
class PolicyChunk:
    """One retrieved passage, with everything needed to cite it.

    `url` and `page` are not decoration: `Evidence` for a policy claim is only
    checkable if a reader can open the document at the right place. An entry
    that names a document without saying where in it is the paper equivalent of
    a figure with no source.
    """

    doc_id: str
    chunk_index: int
    text: str
    title: str
    publisher: str
    url: str
    page: int | None = None
    section: str = ""
    iso3: tuple[str, ...] = ()
    hs_prefix: tuple[str, ...] = ()
    measure_type: str = "other"
    agreement: tuple[str, ...] = ()
    language: str = "en"
    score: float = 0.0

    @property
    def citation(self) -> str:
        """The `Evidence.detail` half — which document, which page."""
        where = f", p. {self.page}" if self.page else ""
        return f"{self.doc_id} ({self.title}, {self.publisher}{where})"

    @classmethod
    def from_payload(cls, payload: dict, score: float = 0.0) -> PolicyChunk:
        return cls(
            doc_id=payload.get(DOC_ID, ""),
            chunk_index=int(payload.get(CHUNK_INDEX, 0)),
            text=payload.get(TEXT, ""),
            title=payload.get(TITLE, ""),
            publisher=payload.get(PUBLISHER, ""),
            url=payload.get(URL, ""),
            page=payload.get(PAGE),
            section=payload.get(SECTION, ""),
            iso3=tuple(payload.get(ISO3) or ()),
            hs_prefix=tuple(payload.get(HS_PREFIX) or ()),
            measure_type=payload.get(MEASURE_TYPE, "other"),
            agreement=tuple(payload.get(AGREEMENT) or ()),
            language=payload.get(LANGUAGE, "en"),
            score=score,
        )

    def to_payload(self) -> dict:
        return {
            DOC_ID: self.doc_id,
            CHUNK_INDEX: self.chunk_index,
            TEXT: self.text,
            TITLE: self.title,
            PUBLISHER: self.publisher,
            URL: self.url,
            PAGE: self.page,
            SECTION: self.section,
            ISO3: list(self.iso3),
            HS_PREFIX: list(self.hs_prefix),
            MEASURE_TYPE: self.measure_type,
            AGREEMENT: list(self.agreement),
            LANGUAGE: self.language,
        }


#: Fixed namespace so a re-index of the same chunk overwrites its own point
#: rather than adding a second copy. The indexer is re-run freely, exactly like
#: the `MERGE`-only KG loaders, and an autoincrementing id would duplicate the
#: whole corpus on every run.
_POINT_NAMESPACE = uuid.UUID("6f3a1c52-8f21-4b0e-9d64-4f0a2c1b7e55")


def point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic point id. Same (doc, chunk) always maps to the same point."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}:{chunk_index}"))


@dataclass
class RetrievalFilter:
    """What the graph decided the search is allowed to look at.

    This is the graph-anchoring step made explicit. `iso3` and `hs_prefixes` are
    resolved from Neo4j *before* any vector search runs, so retrieval cannot
    return a chunk about the wrong country however similar its wording. An
    unfiltered search over this corpus would happily answer a question about
    Germany with Canadian text, because trade-policy documents all read alike.
    """

    iso3: tuple[str, ...] = ()
    hs_prefixes: tuple[str, ...] = ()
    measure_types: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()
    language: str = "en"
    extra_notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """The string that goes into `Evidence.detail`.

        `KnowledgeGraphClient.run()` hands back the literal Cypher so an agent
        can cite the query that produced a figure (SRS 3.1.4). A vector search
        has no query text to hand back, so this is its equivalent: the exact
        constraints the search ran under. Without it, policy evidence would be
        the one kind in the system that says "trust me".
        """
        parts = [f"qdrant hybrid search (dense={DENSE_MODEL}, sparse={SPARSE_MODEL}, RRF)"]
        if self.iso3:
            parts.append(f"iso3 in {list(self.iso3)}")
        if self.hs_prefixes:
            parts.append(f"hs_prefix in {list(self.hs_prefixes)}")
        if self.measure_types:
            parts.append(f"measure_type in {list(self.measure_types)}")
        if self.doc_ids:
            parts.append(f"doc_id in {list(self.doc_ids)}")
        if self.language:
            parts.append(f"language = {self.language}")
        parts.extend(self.extra_notes)
        return "; ".join(parts)

    @property
    def is_anchored(self) -> bool:
        """True when the graph actually constrained this search.

        An unanchored search is allowed — it is what a broad policy question
        gets — but the agent says so in its assumptions rather than implying the
        result was scoped to the country in question.
        """
        return bool(self.iso3 or self.hs_prefixes or self.doc_ids)


__all__ = [
    "AGREEMENT",
    "CHUNK_INDEX",
    "DENSE_DIM",
    "DENSE_MODEL",
    "DENSE_VECTOR",
    "DOC_ID",
    "HS_PREFIX",
    "INDEXED_KEYWORD_FIELDS",
    "ISO3",
    "LANGUAGE",
    "MEASURE_TYPE",
    "MEASURE_TYPES",
    "PAGE",
    "PUBLISHER",
    "RERANK_MODEL",
    "SECTION",
    "SIMULATION_MEASURES",
    "SPARSE_MODEL",
    "SPARSE_VECTOR",
    "TEXT",
    "TITLE",
    "URL",
    "PolicyChunk",
    "RetrievalFilter",
    "point_id",
]
