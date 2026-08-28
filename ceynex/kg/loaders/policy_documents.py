"""Implements SRS 3.1.9 (deviation D10) — :PolicyDocument nodes and their edges.

**The node is a pointer, not the text.** Neo4j records that a document exists,
who issued it, what goods and agreements it covers, and which Qdrant collection
holds its chunks. The text itself lives only in Qdrant. That split is the whole
design: a Cypher query decides *which documents are eligible* before any vector
search runs, so retrieval is scoped by the graph rather than by cosine distance
alone.

Everything is `MERGE`, like every other loader here. Three members load into one
graph and re-run their loaders freely.

Reads `ceynex/data/reference/policy_documents.csv` — the same file the offline
pipeline in `trade-data-pipeline/` chunks from, so the graph and the vector store
cannot disagree about which documents are in the corpus.

`chunk_count` is not in the CSV. It is a property of what was actually indexed,
so it comes from Qdrant at load time when the collection is reachable and is left
at 0 when it is not: a stale count would claim coverage the corpus does not have.
"""

from __future__ import annotations

import csv
import logging
from typing import Any

from ceynex.kg.client import KnowledgeGraphClient
from ceynex.kg.loaders.trade_agreements import REFERENCE_DIR
from ceynex.settings import qdrant_collection

log = logging.getLogger(__name__)


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def document_rows() -> list[dict[str, Any]]:
    """Read the manifest, skipping the `#` provenance header block."""
    path = REFERENCE_DIR / "policy_documents.csv"
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))

    collection = qdrant_collection()
    return [
        {
            "doc_id": row["doc_id"],
            "iso3": _split(row["iso3"]),
            "title": row["title"],
            "publisher": row["publisher"],
            "url": row["url"],
            "doc_type": row["doc_type"],
            "hs_focus": _split(row["hs_focus"]),
            "agreements": _split(row["agreements"]),
            "language": row["language"],
            "published": row["published"],
            "retrieved_at": row["retrieved_at"],
            "sha256": row["sha256"],
            "verified": row["verified"],
            "qdrant_collection": collection,
            # English-only, because the dense embedding model is. A non-English
            # row is still a node — the document was found and consciously
            # skipped — but nothing will ever retrieve it, and an agent that
            # cites it would be citing text no search can reach.
            "indexed": row["language"] == "en",
            "chunk_count": 0,
        }
        for row in rows
    ]


MERGE_DOCUMENTS = """
UNWIND $rows AS row
MERGE (p:PolicyDocument {doc_id: row.doc_id})
  SET p.title             = row.title,
      p.publisher         = row.publisher,
      p.url               = row.url,
      p.doc_type          = row.doc_type,
      p.language          = row.language,
      p.published         = row.published,
      p.retrieved_at      = row.retrieved_at,
      p.sha256            = row.sha256,
      p.verified          = row.verified,
      p.qdrant_collection = row.qdrant_collection,
      p.indexed           = row.indexed,
      p.chunk_count       = row.chunk_count,
      p.iso3              = row.iso3
RETURN count(p) AS merged
"""

# ISSUED_BY only attaches to a Country that already exists. A policy manifest is
# not the place new countries enter the graph — `dim_country` and the trade-flow
# loaders own that, and MERGE-ing a Country here would create a bare node with no
# m49 and break the crosswalk's uniqueness constraint on the next real load.
MERGE_ISSUED_BY = """
UNWIND $rows AS row
UNWIND row.iso3 AS iso3
MATCH (p:PolicyDocument {doc_id: row.doc_id})
MATCH (c:Country {iso3: iso3})
MERGE (p)-[:ISSUED_BY]->(c)
RETURN count(*) AS merged
"""

# APPLIES_TO does MERGE its HSCode, matching what the trade-agreement loader
# already does for coverage rows: an HS code named by a document is a real code
# whether or not any flow has been ingested for it yet.
MERGE_APPLIES_TO = """
UNWIND $rows AS row
UNWIND row.hs_focus AS code
MATCH (p:PolicyDocument {doc_id: row.doc_id})
MERGE (h:HSCode {code: code})
MERGE (p)-[:APPLIES_TO]->(h)
RETURN count(*) AS merged
"""

MERGE_DESCRIBES = """
UNWIND $rows AS row
UNWIND row.agreements AS name
MATCH (p:PolicyDocument {doc_id: row.doc_id})
MATCH (t:TradeAgreement {name: name})
MERGE (p)-[:DESCRIBES]->(t)
RETURN count(*) AS merged
"""

# `indexed` means RETRIEVABLE, not "eligible in principle". A row can be English
# and still have no chunks — the URL 404'd, or it served a JavaScript shell that
# `extract.py` rejected. Leaving those marked `indexed` makes
# `policy_documents_for()` hand Qdrant a doc_id allow-list containing documents
# with nothing behind them, so the graph claims coverage the corpus does not
# have. Set from the count Qdrant actually reports, never from the manifest.
SET_CHUNK_COUNTS = """
UNWIND $counts AS row
MATCH (p:PolicyDocument {doc_id: row.doc_id})
  SET p.chunk_count = row.chunk_count,
      p.indexed     = (p.language = 'en' AND row.chunk_count > 0)
RETURN count(p) AS updated
"""


async def chunk_counts(collection: str | None = None) -> dict[str, int]:
    """How many chunks each document actually has in Qdrant.

    Returns `{}` — not an error — when Qdrant is unreachable or the client is not
    installed. The graph is still correct without it; `chunk_count` stays 0 and
    means "not known", which is the honest value when the collection cannot be
    read.
    """
    try:
        from qdrant_client import AsyncQdrantClient, models

        from ceynex.settings import qdrant_url
    except ImportError:
        log.info("qdrant-client not installed — chunk_count left at 0")
        return {}

    url = qdrant_url()
    if not url:
        return {}

    counts: dict[str, int] = {}
    client = AsyncQdrantClient(url=url, timeout=10)
    try:
        for row in document_rows():
            result = await client.count(
                collection_name=collection or qdrant_collection(),
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=row["doc_id"])
                        )
                    ]
                ),
                exact=True,
            )
            counts[row["doc_id"]] = result.count
    except Exception as exc:  # noqa: BLE001 - an unreachable vector store is not a load failure
        log.warning("could not read chunk counts from qdrant: %s", exc)
        return {}
    finally:
        await client.close()
    return counts


async def load(kg: KnowledgeGraphClient) -> dict[str, int]:
    """Merge policy documents and their edges. Idempotent."""
    rows = document_rows()

    await kg.write(MERGE_DOCUMENTS, {"rows": rows})
    await kg.write(MERGE_ISSUED_BY, {"rows": rows})
    await kg.write(MERGE_APPLIES_TO, {"rows": rows})
    await kg.write(MERGE_DESCRIBES, {"rows": rows})

    counts = await chunk_counts()
    if counts:
        await kg.write(
            SET_CHUNK_COUNTS,
            {"counts": [{"doc_id": k, "chunk_count": v} for k, v in counts.items()]},
        )

    # With Qdrant reachable, "indexed" is what it actually holds. Without it, the
    # optimistic language-only value from `document_rows()` stands and the count
    # stays 0 — stated here so a run against a stopped Qdrant is not mistaken for
    # a run against an empty one.
    if counts:
        indexed = sum(1 for row in rows if row["indexed"] and counts.get(row["doc_id"], 0) > 0)
    else:
        indexed = sum(1 for row in rows if row["indexed"])
        log.warning("qdrant not reachable — `indexed` reflects language only, not real coverage")

    log.info(
        "merged %d policy documents (%d retrievable), %d chunks",
        len(rows),
        indexed,
        sum(counts.values()),
    )
    return {
        "documents": len(rows),
        "indexable": indexed,
        "chunks": sum(counts.values()),
    }


__all__ = ["chunk_counts", "document_rows", "load"]
