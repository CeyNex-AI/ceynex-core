"""Supports SRS 3.1.4, 3.1.5 and 3.1.9 (deviation D10) — the only way into Qdrant.

Deliberately the same shape as `ceynex/kg/client.py`, because every agent already
knows that shape and a second retrieval client with different manners is a second
thing to learn:

1. **`search()` returns the results *and* a description of the query.** The KG
   client returns the literal Cypher so an agent can put it in `Evidence.detail`
   (SRS 3.1.4). A vector search has no query text, so it returns the exact filter
   it ran under instead — see `RetrievalFilter.describe()`. Policy evidence would
   otherwise be the one kind in this system that cannot be checked.

2. **Unreachable is a degraded answer, not a failure.** `PolicyRetrieverUnavailableError`
   after one retry; the agent catches it and answers exactly as it did before
   retrieval existed (SAD §4.1). Nothing here is allowed to be the reason a query
   fails.

**The 2-second budget is the load-bearing constraint.** `docs/EVALUATION.md` §1
records single-sector p95 at 14.6 s against SRS 3.4.1's 10 s budget, and
`orchestrator/graph.py` gives each agent a 12 s slice of it. Retrieval is being
added to a path that is already over. `RETRIEVAL_TIMEOUT_S` is therefore a hard
ceiling on the whole embed → search → rerank round trip, not a per-hop timeout:
past it, the agent gets nothing and says so, which costs an unsourced answer.
Blowing the budget would cost the request.

**Imports are lazy.** `qdrant-client` is the optional `[policy]` extra. Importing
this module must not fail for a teammate who did not install it; only
constructing a retriever does, and `from_settings()` returns None rather than
raising so the caller never has to care.
"""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

from ceynex.retrieval.schema import (
    DENSE_MODEL,
    DENSE_VECTOR,
    DOC_ID,
    HS_PREFIX,
    ISO3,
    LANGUAGE,
    MEASURE_TYPE,
    RERANK_MODEL,
    SPARSE_MODEL,
    SPARSE_VECTOR,
    PolicyChunk,
    RetrievalFilter,
)
from ceynex.settings import policy_retrieval_enabled, qdrant_collection, qdrant_url

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from qdrant_client import AsyncQdrantClient

log = logging.getLogger(__name__)

# One retry, then degrade — the same trade `kg/client.py` makes, and for the same
# reason: retrying harder eats the response-time budget SRS 3.4.1 spends on the
# LLM call.
MAX_ATTEMPTS = 2

#: Hard ceiling on embed + search + rerank, together. See the module docstring.
RETRIEVAL_TIMEOUT_S = 2.0

#: How many candidates the fused search returns before reranking. 30 into 5 is
#: the usual shape: wide enough that the cross-encoder has something to reorder,
#: narrow enough that scoring the pairs stays inside the budget.
CANDIDATE_LIMIT = 30
DEFAULT_LIMIT = 5


class PolicyRetrieverUnavailableError(RuntimeError):
    """Qdrant could not be reached, or the search failed.

    Agents catch this and degrade. They never let it propagate — the partial
    result guarantee (SAD §4.1) does not have an exception for the newest
    datastore.
    """


# Model handles are process-wide. Loading the ONNX weights takes on the order of
# a second, which is half the per-query budget; doing it per query would mean the
# budget is spent before a single vector is compared.
_models: dict[str, Any] = {}
_models_lock = asyncio.Lock()


def _load_models() -> dict[str, Any]:
    """Import fastembed and instantiate the three models. Blocking; call in a thread."""
    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return {
        "dense": TextEmbedding(model_name=DENSE_MODEL),
        "sparse": SparseTextEmbedding(model_name=SPARSE_MODEL),
        "rerank": TextCrossEncoder(model_name=RERANK_MODEL),
    }


async def _models_ready() -> dict[str, Any]:
    global _models
    if _models:
        return _models
    async with _models_lock:
        if not _models:
            _models = await asyncio.to_thread(_load_models)
    return _models


class PolicyRetriever:
    """Async hybrid search over the policy-document collection.

    One instance per process, shared by every agent, same as the KG client: the
    Qdrant client holds a connection pool and the fastembed models hold ONNX
    sessions, neither of which should be built per query.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        collection: str | None = None,
        timeout_s: float = RETRIEVAL_TIMEOUT_S,
    ) -> None:
        self._url = url or qdrant_url() or "http://localhost:6333"
        self._collection = collection or qdrant_collection()
        self._timeout_s = timeout_s
        self._client: AsyncQdrantClient | None = None

    # --- lifecycle -------------------------------------------------------

    @classmethod
    def from_settings(cls) -> PolicyRetriever | None:
        """Build one, or return None if retrieval is off or not installed.

        None is the "not configured" state the agent already knows how to handle,
        and returning it here means the wiring site does not need a try/except
        around an optional dependency. Three ways to get None, all of them
        ordinary rather than exceptional:

        - `CEYNEX_POLICY_RETRIEVAL=off`, which is how `make eval-policy-baseline`
          measures the system as it behaved before this feature;
        - no `QDRANT_URL` configured;
        - `qdrant-client` not installed, i.e. the `[policy]` extra was skipped.
        """
        if not policy_retrieval_enabled():
            log.info("policy retrieval disabled by CEYNEX_POLICY_RETRIEVAL")
            return None
        if not qdrant_url():
            log.info("no QDRANT_URL configured — policy retrieval is off")
            return None
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            log.info("qdrant-client not installed — policy retrieval is off (pip install '.[policy]')")
            return None
        return cls()

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(url=self._url, timeout=int(self._timeout_s) or 1)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def verify_connectivity(self) -> bool:
        """Cheap liveness probe for /health. Reports, never raises."""
        try:
            await self.client.get_collection(self._collection)
        except Exception as exc:  # noqa: BLE001 - liveness probe reports, never raises
            log.warning("qdrant collection %s unavailable at %s: %s", self._collection, self._url, exc)
            return False
        return True

    # --- searching -------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        filters: RetrievalFilter | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> tuple[list[PolicyChunk], str]:
        """Hybrid search, reranked. Returns `(chunks, filter_description)`.

        The description comes back with the results for the same reason the KG
        client returns its Cypher: the caller must be able to put the thing that
        actually ran into `Evidence.detail`, rather than reconstructing from
        memory what it thinks it asked for.
        """
        filters = filters or RetrievalFilter()
        description = filters.describe()

        try:
            chunks = await asyncio.wait_for(
                self._search(query, filters, limit), timeout=self._timeout_s
            )
        except TimeoutError as exc:
            raise PolicyRetrieverUnavailableError(
                f"policy retrieval exceeded its {self._timeout_s:.1f}s budget"
            ) from exc
        return chunks, description

    async def _search(
        self, query: str, filters: RetrievalFilter, limit: int
    ) -> list[PolicyChunk]:
        models = await _models_ready()
        dense, sparse = await asyncio.to_thread(_embed_query, models, query)
        query_filter = _build_filter(filters)

        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                from qdrant_client import models as qmodels

                response = await self.client.query_points(
                    collection_name=self._collection,
                    prefetch=[
                        qmodels.Prefetch(
                            query=dense, using=DENSE_VECTOR, limit=CANDIDATE_LIMIT,
                            filter=query_filter,
                        ),
                        qmodels.Prefetch(
                            query=qmodels.SparseVector(**sparse), using=SPARSE_VECTOR,
                            limit=CANDIDATE_LIMIT, filter=query_filter,
                        ),
                    ],
                    query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                    limit=CANDIDATE_LIMIT,
                    with_payload=True,
                )
                break
            except Exception as exc:  # noqa: BLE001 - retried once, then degraded
                last_error = exc
                log.warning("qdrant attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc)
        else:
            raise PolicyRetrieverUnavailableError(
                f"qdrant unreachable at {self._url} after {MAX_ATTEMPTS} attempts: {last_error}"
            ) from last_error

        candidates = [
            PolicyChunk.from_payload(point.payload or {}, score=point.score or 0.0)
            for point in response.points
        ]
        if not candidates:
            return []
        return await asyncio.to_thread(_rerank, models, query, candidates, limit)


def _embed_query(models: dict[str, Any], query: str) -> tuple[list[float], dict[str, list]]:
    """Embed once for both vectors. Blocking — fastembed is synchronous ONNX."""
    dense = next(iter(models["dense"].query_embed(query))).tolist()
    raw = next(iter(models["sparse"].query_embed(query)))
    return dense, {"indices": raw.indices.tolist(), "values": raw.values.tolist()}


def _rerank(
    models: dict[str, Any], query: str, candidates: list[PolicyChunk], limit: int
) -> list[PolicyChunk]:
    """Cross-encoder rerank of the fused candidates.

    RRF fuses two *rankings*; it never reads the query and a document together.
    The cross-encoder does, which is why it reorders the top of the list so much
    better than either retriever produced it — and why it is worth ~50 ms on 30
    pairs. Scores are replaced, not blended: an RRF score and a cross-encoder
    score are not on the same scale and averaging them means nothing.
    """
    scores = list(models["rerank"].rerank(query, [c.text for c in candidates]))
    ranked = sorted(
        (
            PolicyChunk(**{**vars(candidate), "score": float(score)})
            for candidate, score in zip(candidates, scores, strict=False)
        ),
        key=lambda c: c.score,
        reverse=True,
    )
    return ranked[:limit]


def _build_filter(filters: RetrievalFilter) -> Any:
    """Translate the graph-resolved constraints into a Qdrant filter.

    Every clause is `must`. A policy chunk is only eligible if it is about the
    right country *and* the right goods *and* the right kind of measure — an
    `should`/`or` here would let a chunk match on country alone and reintroduce
    exactly the cross-contamination that anchoring exists to prevent.

    `hs_prefix`, `iso3` and `agreement` are list-valued in the payload;
    `MatchAny` on a list field matches when the lists intersect, which is the
    semantics wanted: a chunk tagged for both HS 61 and 62 answers a question
    about either.
    """
    from qdrant_client import models as qmodels

    conditions = []
    if filters.iso3:
        conditions.append(
            qmodels.FieldCondition(key=ISO3, match=qmodels.MatchAny(any=list(filters.iso3)))
        )
    if filters.hs_prefixes:
        conditions.append(
            qmodels.FieldCondition(
                key=HS_PREFIX, match=qmodels.MatchAny(any=list(filters.hs_prefixes))
            )
        )
    if filters.measure_types:
        conditions.append(
            qmodels.FieldCondition(
                key=MEASURE_TYPE, match=qmodels.MatchAny(any=list(filters.measure_types))
            )
        )
    if filters.doc_ids:
        conditions.append(
            qmodels.FieldCondition(key=DOC_ID, match=qmodels.MatchAny(any=list(filters.doc_ids)))
        )
    if filters.language:
        conditions.append(
            qmodels.FieldCondition(key=LANGUAGE, match=qmodels.MatchValue(value=filters.language))
        )
    return qmodels.Filter(must=conditions) if conditions else None


__all__ = [
    "CANDIDATE_LIMIT",
    "DEFAULT_LIMIT",
    "RETRIEVAL_TIMEOUT_S",
    "PolicyRetriever",
    "PolicyRetrieverUnavailableError",
]
