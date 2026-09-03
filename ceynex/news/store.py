"""Supports docs/ARCHITECTURE_DELTA.md D11 — the only way into `ceynex_news`.

Same shape as `retrieval/client.py`, because every reader of this codebase
already knows that shape: a `from_settings()` that returns None rather than
raising, a lazy `qdrant-client` import so the optional `[policy]` extra is not
needed to import the module, and a hybrid dense+sparse search fused with RRF.

Why a second collection and not a `doc_type` field
--------------------------------------------------
The policy corpus is fifteen hand-verified documents whose whole purpose is to be
citable (SRS 3.1.4). News is thousands of unvetted headlines a week. Sharing one
collection would mean every existing retrieval path had to remember to exclude
news — and `retrieval/client.py` already widens its own filters on an empty
result, so "remembering" would have to survive a code path specifically designed
to relax constraints. Forgetting once puts a headline in an evidence panel, and
that failure is silent.

Two collections cost nothing at query time: same embedding model, same
dimensions, so the ONNX sessions `shared_models()` holds are reused as-is.

`__init__` refuses to open the policy collection at all. A mis-set
`QDRANT_NEWS_COLLECTION` is a configuration error and should be loud at startup,
not a runtime condition discovered by a marker.

Retention
---------
Nothing else in this system prunes anything, because nothing else grows without
bound. This does: ~24 refreshes a day, forever. `prune()` runs at the end of
every cycle and deletes by `seen_ts` range, which is why that field carries an
integer payload index rather than a keyword one.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

from ceynex import settings
from ceynex.news.schema import (
    INDEXED_INTEGER_FIELDS,
    INDEXED_KEYWORD_FIELDS,
    KIND,
    NEWS_KIND,
    SCOPE,
    SEEN_TS,
    TOPIC,
    NewsArticle,
)
from ceynex.retrieval.schema import DENSE_DIM, DENSE_VECTOR, SPARSE_VECTOR

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qdrant_client import AsyncQdrantClient

log = logging.getLogger(__name__)

#: One retry then degrade, the same trade `kg/client.py` and `retrieval/client.py`
#: both make: retrying harder eats the response-time budget.
MAX_ATTEMPTS = 2

#: The search budget. Looser than retrieval's 2.0 s because this runs on its own
#: endpoint rather than inside the orchestrator's per-node slice, and tighter
#: than the route's 8 s so a slow Qdrant still leaves room to answer.
SEARCH_TIMEOUT_S = 3.0

#: Fused candidates before truncation. No cross-encoder pass here — the caller
#: scores what comes back, so this only has to be wide enough to score.
CANDIDATE_LIMIT = 40


class NewsStoreUnavailableError(RuntimeError):
    """Qdrant could not be reached. Callers degrade; this never reaches a client."""


class NewsStore:
    """Async hybrid search and idempotent upsert over the news collection."""

    def __init__(
        self,
        url: str | None = None,
        *,
        collection: str | None = None,
        timeout_s: float = SEARCH_TIMEOUT_S,
    ) -> None:
        self._url = url or settings.qdrant_url() or "http://localhost:6333"
        self._collection = collection or settings.news_collection()
        if self._collection == settings.qdrant_collection():
            raise ValueError(
                f"the news collection may not be the policy collection "
                f"({self._collection!r}). Unvetted headlines behind the policy "
                f"filters would become citable evidence — see D11."
            )
        self._timeout_s = timeout_s
        self._client: AsyncQdrantClient | None = None

    # --- lifecycle -------------------------------------------------------

    @classmethod
    def from_settings(cls) -> NewsStore | None:
        """Build one, or None. Three ordinary ways to get None, none exceptional.

        Mirrors `PolicyRetriever.from_settings()` exactly: the news sidecar off,
        no Qdrant configured, or the `[policy]` extra not installed. Without a
        store, `/api/news/search` still works — it just loses its fallback when
        GDELT is down, and stops accumulating headlines.
        """
        if not settings.news_enabled():
            log.info("news disabled by CEYNEX_NEWS — nothing will be indexed")
            return None
        if not settings.qdrant_url():
            log.info("no QDRANT_URL configured — news is not indexed")
            return None
        try:
            import qdrant_client  # noqa: F401, PLC0415 - probing the optional extra
        except ImportError:
            log.info("qdrant-client not installed — news is not indexed (pip install '.[policy]')")
            return None
        return cls()

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            from qdrant_client import AsyncQdrantClient  # noqa: PLC0415 - optional extra

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

    # --- schema ----------------------------------------------------------

    async def ensure_collection(self) -> bool:
        """Create the collection and its payload indexes. Idempotent; reports.

        Returns False rather than raising when Qdrant is unreachable: this runs
        at startup, and a vector store that is not up yet must not stop the API
        from serving everything else.
        """
        from qdrant_client import models as qmodels  # noqa: PLC0415 - optional extra

        try:
            if not await self.client.collection_exists(self._collection):
                await self.client.create_collection(
                    collection_name=self._collection,
                    vectors_config={
                        DENSE_VECTOR: qmodels.VectorParams(
                            size=DENSE_DIM, distance=qmodels.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)
                    },
                )
                log.info("created qdrant collection %s", self._collection)

            for field in INDEXED_KEYWORD_FIELDS:
                await self._ensure_index(field, qmodels.PayloadSchemaType.KEYWORD)
            for field in INDEXED_INTEGER_FIELDS:
                await self._ensure_index(field, qmodels.PayloadSchemaType.INTEGER)
        except Exception:  # noqa: BLE001 - startup reports, it does not fail
            log.warning("could not prepare %s at %s", self._collection, self._url, exc_info=True)
            return False
        return True

    async def _ensure_index(self, field: str, schema: Any) -> None:
        try:
            await self.client.create_payload_index(
                collection_name=self._collection, field_name=field, field_schema=schema
            )
        except Exception as exc:  # noqa: BLE001 - "already exists" is the happy path here
            if "already exists" not in str(exc).lower():
                log.debug("payload index on %s: %s", field, exc)

    async def verify_connectivity(self) -> bool:
        """Cheap liveness probe. Reports, never raises."""
        try:
            await self.client.get_collection(self._collection)
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
            log.warning("news collection %s unavailable: %s", self._collection, exc)
            return False
        return True

    async def count(self) -> int:
        """How many articles are indexed. For `/health` and the CLI."""
        try:
            result = await self.client.count(collection_name=self._collection, exact=False)
        except Exception:  # noqa: BLE001 - a count nobody can get is reported as zero
            return 0
        return int(getattr(result, "count", 0))

    # --- writing ---------------------------------------------------------

    async def upsert(self, articles: list[NewsArticle], *, batch: int = 64) -> int:
        """Index articles. Returns how many points were written.

        Idempotent by construction: `article_id` is a uuid5 of the canonical URL,
        so re-fetching the same story every hour overwrites one point rather than
        adding twenty-four.

        Never raises. This is called from a background task after a response has
        already gone out; there is nobody left to tell.
        """
        articles = [a for a in articles if a.url and a.embed_text()]
        if not articles:
            return 0

        from qdrant_client import models as qmodels  # noqa: PLC0415 - optional extra

        try:
            dense, sparse = await self._embed([a.embed_text() for a in articles])
        except Exception:  # noqa: BLE001 - indexing is opportunistic, never load-bearing
            log.warning("could not embed %d headlines; not indexing them", len(articles), exc_info=True)
            return 0

        ingested_ts = int(time.time())
        points = [
            qmodels.PointStruct(
                id=article.article_id,
                vector={
                    DENSE_VECTOR: dense_vector,
                    SPARSE_VECTOR: qmodels.SparseVector(
                        indices=sparse_vector["indices"], values=sparse_vector["values"]
                    ),
                },
                payload=article.to_payload(ingested_ts=ingested_ts),
            )
            for article, dense_vector, sparse_vector in zip(articles, dense, sparse, strict=True)
        ]

        written = 0
        for start in range(0, len(points), batch):
            chunk = points[start : start + batch]
            try:
                await self.client.upsert(collection_name=self._collection, points=chunk)
            except Exception:  # noqa: BLE001 - one bad batch must not lose the rest
                log.warning("could not upsert %d news points", len(chunk), exc_info=True)
                continue
            written += len(chunk)
        return written

    async def prune(self, older_than_days: int) -> int:
        """Delete articles first seen more than `older_than_days` ago.

        Reports rather than raises, and returns the cutoff-matching count it
        asked to remove. Nothing depends on this succeeding — a failed prune
        costs disk, not correctness — so the refresher logs and moves on.
        """
        if older_than_days <= 0:
            return 0

        from qdrant_client import models as qmodels  # noqa: PLC0415 - optional extra

        cutoff = int(time.time()) - older_than_days * 86_400
        stale = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key=SEEN_TS, range=qmodels.Range(lt=cutoff)),
                # `seen_ts` is 0 for an article whose date would not parse.
                # Those are not old, they are undated, and deleting them on the
                # first sweep would quietly discard every malformed-date row.
                qmodels.FieldCondition(key=SEEN_TS, range=qmodels.Range(gt=0)),
            ]
        )
        try:
            counted = await self.client.count(
                collection_name=self._collection, count_filter=stale, exact=True
            )
            await self.client.delete(
                collection_name=self._collection,
                points_selector=qmodels.FilterSelector(filter=stale),
            )
        except Exception:  # noqa: BLE001 - a failed prune costs disk, not correctness
            log.warning("could not prune %s", self._collection, exc_info=True)
            return 0

        removed = int(getattr(counted, "count", 0))
        if removed:
            log.info("pruned %d news articles older than %d days", removed, older_than_days)
        return removed

    # --- reading ---------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        limit: int = 24,
        topic: str = "",
        scope: str = "",
    ) -> list[NewsArticle]:
        """Hybrid search over indexed headlines. The fallback when GDELT is down.

        Returns articles carrying the fused retrieval score in `relevance`. The
        caller re-scores with the cross-encoder, exactly as it does for a live
        GDELT response, so both paths produce comparable numbers and one UI.
        """
        from qdrant_client import models as qmodels  # noqa: PLC0415 - optional extra

        dense, sparse = await self._embed([query], as_query=True)
        query_filter = self._build_filter(topic=topic, scope=scope)

        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await self.client.query_points(
                    collection_name=self._collection,
                    prefetch=[
                        qmodels.Prefetch(
                            query=dense[0],
                            using=DENSE_VECTOR,
                            limit=CANDIDATE_LIMIT,
                            filter=query_filter,
                        ),
                        qmodels.Prefetch(
                            query=qmodels.SparseVector(**sparse[0]),
                            using=SPARSE_VECTOR,
                            limit=CANDIDATE_LIMIT,
                            filter=query_filter,
                        ),
                    ],
                    query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                    limit=max(limit, CANDIDATE_LIMIT),
                    with_payload=True,
                )
                break
            except Exception as exc:  # noqa: BLE001 - retried once, then degraded
                last_error = exc
                log.warning("news search attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc)
        else:
            raise NewsStoreUnavailableError(
                f"qdrant unreachable at {self._url} after {MAX_ATTEMPTS} attempts: {last_error}"
            ) from last_error

        return [
            NewsArticle.from_payload(point.payload or {})
            for point in response.points
            if (point.payload or {}).get(KIND) == NEWS_KIND
        ][:limit]

    def _build_filter(self, *, topic: str, scope: str) -> Any:
        from qdrant_client import models as qmodels  # noqa: PLC0415 - optional extra

        conditions = [
            qmodels.FieldCondition(key=KIND, match=qmodels.MatchValue(value=NEWS_KIND))
        ]
        if topic:
            conditions.append(
                qmodels.FieldCondition(key=TOPIC, match=qmodels.MatchValue(value=topic))
            )
        if scope:
            conditions.append(
                qmodels.FieldCondition(key=SCOPE, match=qmodels.MatchValue(value=scope))
            )
        return qmodels.Filter(must=conditions)

    # --- embedding -------------------------------------------------------

    async def _embed(
        self, texts: list[str], *, as_query: bool = False
    ) -> tuple[list[list[float]], list[dict[str, list]]]:
        """Dense and sparse vectors for `texts`, off the shared ONNX sessions."""
        from ceynex.retrieval.client import shared_models  # noqa: PLC0415 - optional extra

        models = await shared_models()
        return await asyncio.to_thread(_embed_texts, models, texts, as_query)


def _embed_texts(
    models: dict[str, Any], texts: list[str], as_query: bool
) -> tuple[list[list[float]], list[dict[str, list]]]:
    """Blocking — fastembed is synchronous ONNX.

    `query_embed` and `embed` are different calls on purpose: BGE prepends a
    retrieval instruction to a query and not to a document, and using the wrong
    one silently costs recall rather than raising.
    """
    if as_query:
        dense = [vector.tolist() for vector in models["dense"].query_embed(texts)]
        raw_sparse = list(models["sparse"].query_embed(texts))
    else:
        dense = [vector.tolist() for vector in models["dense"].embed(texts)]
        raw_sparse = list(models["sparse"].embed(texts))

    sparse = [
        {"indices": vector.indices.tolist(), "values": vector.values.tolist()}
        for vector in raw_sparse
    ]
    return dense, sparse


__all__ = [
    "CANDIDATE_LIMIT",
    "MAX_ATTEMPTS",
    "SEARCH_TIMEOUT_S",
    "NewsStore",
    "NewsStoreUnavailableError",
]
