"""Implements docs/ARCHITECTURE_DELTA.md D11 — GET /api/news/search and /trending.

Two read-only endpoints beside the query flow, never inside it.

**Why this is not folded into `POST /api/query`.** It would be one less request
and it would be wrong. `retrieval/client.py`'s docstring records single-sector
p95 at 14.6 s against SRS 3.4.1's 10 s budget — that path is already over, and
adding a GDELT round trip to it makes a documented problem worse for every query,
including the ones nobody wanted news for. It would also mean changing
`QueryResponse`, whose own docstring calls its field names "as frozen in practice
as the Python contracts". The browser fires this in parallel instead, so news
renders in about a second while the orchestrator is still working.

That parallelism is a property of the *frontend*. If a future change calls this
from inside the graph for convenience, the numbers in `docs/EVALUATION.md`
silently regress. This paragraph is the warning.

**News is never evidence.** The response shape has no `source_id`, no `claim` and
no `detail`; `NewsArticle` has no `.to_evidence()`; `SourceId` in the frontend is
a closed union behind a three-reviewer PR. Four independent things would have to
change before a headline could be cited, which is the point.

**Always 200.** A dead GDELT, a dead Qdrant and an empty snapshot are all
ordinary states here. A 5xx from the sidecar would make a working answer page
look broken.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response

from ceynex import settings
from ceynex.api import rate_limit
from ceynex.api.deps import Runtime, get_runtime
from ceynex.api.routes.auth import TokenPayload, get_optional_user
from ceynex.api.schemas import (
    NewsArticleItem,
    NewsSearchResponse,
    TrendingArticleItem,
    TrendingResponse,
    TrendingTopicItem,
)
from ceynex.news import snapshot
from ceynex.news.gdelt import GdeltUnavailableError
from ceynex.news.relevance import score_articles
from ceynex.news.schema import SCOPES, NewsArticle, relevance_label, title_key
from ceynex.news.store import NewsStore

log = logging.getLogger(__name__)

router = APIRouter(tags=["news"])

#: The whole outbound budget for one search: throttle wait, fetch, score. Well
#: inside what a user will wait beside an answer that itself takes seconds.
NEWS_SEARCH_BUDGET_S = 8.0

#: How long a caller will hold for a GDELT slot before giving up and answering
#: from the store. Shorter than the budget so there is time left to do that.
THROTTLE_WAIT_S = 3.0

#: Served-response cache. Small and in-process on purpose: this exists for the
#: demo's re-ask path (the same question asked twice in a minute), not as a tier.
_CACHE_TTL_S = 300.0
_CACHE_MAX = 64
_cache: OrderedDict[tuple[str, int], tuple[float, list[NewsArticle], str]] = OrderedDict()


# --- rate limiting --------------------------------------------------------

_window_singleton: rate_limit.Window | None = None


def _window() -> rate_limit.Window:
    """Built on first use, not at import — `build_window` reads REDIS_URL."""
    global _window_singleton  # noqa: PLW0603 - one process-lifetime object
    if _window_singleton is None:
        _window_singleton = rate_limit.build_window()
    return _window_singleton


def set_window(window: rate_limit.Window | None) -> None:
    """Test seam. Production never calls this."""
    global _window_singleton  # noqa: PLW0603
    _window_singleton = window


async def enforce_news_rate_limit(
    http_request: Request,
    user: TokenPayload | None = Depends(get_optional_user),  # noqa: B008
) -> None:
    """SRS 3.4.6's shape, applied to the sidecar's own allowance.

    The identity is prefixed with `news:` rather than reusing `identity_of()`'s
    output directly. `rate_limit.KEY_PREFIX` is a module constant shared by every
    `Window` built from that module, so an unprefixed identity would write to the
    *same* Redis keys as `POST /api/query` — and since the browser fires a news
    search on every submitted query, that would silently halve the query
    allowance. Fixed here rather than in `rate_limit.py`, which is M3's and whose
    current behaviour is correct for its one existing caller.
    """
    config = settings.load_config("api").get("news_rate_limit", {})
    if not config.get("enabled", True):
        return

    limit = int(config.get("search_per_minute", 20))
    window_s = int(config.get("window_seconds", 60))
    identity = "news:" + rate_limit.identity_of(
        user.email if user else None,
        http_request.client.host if http_request.client else None,
    )

    decision = await _window().check(identity, limit, window_s)
    if decision.allowed:
        return

    raise HTTPException(
        status_code=429,
        detail=f"rate limit exceeded: at most {limit} news searches per {window_s} seconds.",
        headers={"Retry-After": str(decision.retry_after_s)},
    )


# --- search ---------------------------------------------------------------


@router.get(
    "/api/news/search",
    response_model=NewsSearchResponse,
    dependencies=[Depends(enforce_news_rate_limit)],
)
async def search_news(
    response: Response,
    background: BackgroundTasks,
    q: str = Query(min_length=3, max_length=300, description="The question to find coverage for."),
    limit: int = Query(default=8, ge=1, le=25),
    runtime: Runtime = Depends(get_runtime),  # noqa: B008 - FastAPI's dependency idiom
) -> NewsSearchResponse:
    started = time.perf_counter()
    query = q.strip()

    cached = _cache_get(query, limit)
    if cached is not None:
        articles, source = cached
    else:
        try:
            articles, source = await asyncio.wait_for(
                _fetch(runtime, query, limit), timeout=NEWS_SEARCH_BUDGET_S
            )
        except TimeoutError:
            log.info("news search for %r exceeded its %.0fs budget", query, NEWS_SEARCH_BUDGET_S)
            articles, source = [], "unavailable"
        except Exception:  # noqa: BLE001 - the sidecar never fails the page
            log.warning("news search for %r failed", query, exc_info=True)
            articles, source = [], "unavailable"

        if source == "gdelt" and runtime.news is not None and articles:
            # After the response is flushed, in the same loop. Not a bare
            # `asyncio.create_task`: a task with no strong reference can be
            # garbage-collected mid-flight, which is an intermittent bug that
            # never reproduces. Only for live results — re-indexing what came
            # out of the store is pure write amplification.
            background.add_task(_index, runtime.news, list(articles))

        _cache_put(query, limit, articles, source)

    response.headers["Cache-Control"] = "public, max-age=300"
    return NewsSearchResponse(
        query=query,
        articles=[_to_item(article) for article in articles],
        source=source,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )


async def _fetch(runtime: Runtime, query: str, limit: int) -> tuple[list[NewsArticle], str]:
    """Live GDELT, falling back to indexed headlines, falling back to nothing."""
    config = settings.news_config()
    gdelt_config = config.get("gdelt", {})
    blocked = {d.lower() for d in config.get("store", {}).get("blocked_domains", []) or []}

    articles: list[NewsArticle] = []
    source = "unavailable"

    if runtime.gdelt is not None:
        try:
            articles = await runtime.gdelt.articles(
                query,
                timespan=str(gdelt_config.get("timespan_search", "7d")),
                max_records=int(gdelt_config.get("max_records_search", 75)),
                sort=str(gdelt_config.get("sort", "HybridRel")),
                attempts=2,
                backoff_base_s=0.5,
                max_wait_s=THROTTLE_WAIT_S,
            )
            source = "gdelt"
        except GdeltUnavailableError as exc:
            log.info("live news search unavailable (%s); falling back to the index", exc)

    if not articles and runtime.news is not None:
        try:
            articles = await runtime.news.search(query, limit=limit * 3)
            source = "cache" if articles else source
        except Exception:  # noqa: BLE001 - both sources down is an ordinary state
            log.info("indexed news search unavailable too", exc_info=True)

    if not articles:
        return [], "unavailable"

    if blocked:
        articles = [a for a in articles if a.domain.lower() not in blocked]

    scored = await score_articles(query, articles, limit=limit * 2)
    return _dedupe(scored)[:limit], source


def _dedupe(articles: list[NewsArticle]) -> list[NewsArticle]:
    """Drop syndication twins, keeping the highest-scoring spelling of each.

    In the response only — the store keeps them, because they genuinely are
    different articles and collapsing them there would be irreversible.
    """
    seen: set[str] = set()
    kept: list[NewsArticle] = []
    for article in articles:
        key = title_key(article.title)
        if key and key in seen:
            continue
        seen.add(key)
        kept.append(article)
    return kept


async def _index(store: NewsStore, articles: list[NewsArticle]) -> None:
    """Opportunistic. Same posture as `history.record()` — losing a write here
    costs a row in a cache, not an answer anyone was waiting for."""
    try:
        await store.upsert(articles)
    except Exception:  # noqa: BLE001 - the response has already gone out
        log.warning("could not index %d news articles", len(articles), exc_info=True)


def _to_item(article: NewsArticle) -> NewsArticleItem:
    return NewsArticleItem(
        url=article.url,
        title=article.title,
        domain=article.domain,
        source_country=article.source_country,
        seen_at=article.seen_at.isoformat() if article.seen_at else None,
        relevance_score=round(article.relevance, 2) if article.relevance is not None else None,
        relevance=relevance_label(article.relevance),
    )


# --- the served-result cache ---------------------------------------------


def _cache_get(query: str, limit: int) -> tuple[list[NewsArticle], str] | None:
    key = (query.casefold(), limit)
    entry = _cache.get(key)
    if entry is None:
        return None
    stored_at, articles, source = entry
    if time.time() - stored_at > _CACHE_TTL_S:
        _cache.pop(key, None)
        return None
    _cache.move_to_end(key)
    return articles, source


def _cache_put(query: str, limit: int, articles: list[NewsArticle], source: str) -> None:
    if source == "unavailable":
        # Caching a failure would keep the panel empty for five minutes after
        # GDELT came back.
        return
    _cache[(query.casefold(), limit)] = (time.time(), articles, source)
    _cache.move_to_end((query.casefold(), limit))
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def clear_cache() -> None:
    """Test seam."""
    _cache.clear()


# --- trending -------------------------------------------------------------


@router.get("/api/news/trending", response_model=TrendingResponse)
async def trending(response: Response) -> TrendingResponse:
    """The precomputed panel. Always 200, whatever state the refresher is in."""
    response.headers["Cache-Control"] = "public, max-age=120"

    if not settings.news_enabled():
        return TrendingResponse(status="unavailable", scopes={scope: [] for scope in SCOPES})

    config = settings.news_config()
    interval_s = float(config.get("refresh", {}).get("interval_minutes", 60)) * 60.0

    scopes: dict[str, list[TrendingTopicItem]] = {}
    newest: datetime | None = None
    partial = False
    oldest_age: float | None = None

    for scope in SCOPES:
        stored = await asyncio.to_thread(snapshot.latest, scope)
        scopes[scope] = [_to_topic(topic) for topic in (stored.topics if stored else [])]
        if stored is None:
            continue
        partial = partial or stored.partial
        newest = stored.computed_at if newest is None else max(newest, stored.computed_at)
        age = (datetime.now(tz=UTC) - stored.computed_at).total_seconds()
        oldest_age = age if oldest_age is None else max(oldest_age, age)

    if newest is None:
        # Never computed. The UI renders nothing and lets the static examples
        # carry the page, rather than showing an error for a panel that is
        # simply not ready yet.
        return TrendingResponse(status="warming", scopes=scopes)

    # Two intervals' grace. A six-hour-old panel labelled "trending now" is a
    # lie, so past that the UI says "as of Nh ago" instead.
    status = "stale" if (oldest_age or 0.0) > interval_s * 2 else "ready"
    return TrendingResponse(
        status=status, computed_at=newest.isoformat(), partial=partial, scopes=scopes
    )


def _to_topic(raw: dict) -> TrendingTopicItem:
    return TrendingTopicItem(
        topic_id=raw.get("topic_id", ""),
        label=raw.get("label", ""),
        prompt=raw.get("prompt", ""),
        scope=raw.get("scope", ""),
        articles_24h=int(raw.get("articles_24h", 0)),
        baseline_24h=float(raw.get("baseline_24h", 0.0)),
        delta_pct=float(raw.get("delta_pct", 0.0)),
        direction=raw.get("direction", "steady"),
        top_articles=[
            TrendingArticleItem(
                url=article.get("url", ""),
                title=article.get("title", ""),
                domain=article.get("domain", ""),
                seen_at=article.get("seen_at"),
            )
            for article in raw.get("top_articles") or []
        ],
    )
