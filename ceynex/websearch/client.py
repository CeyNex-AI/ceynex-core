"""Deciding whether to search, and surviving it when it goes wrong — D14.

Two rules govern everything here.

**Most questions never make a call.** `wants_current_context()` is a cheap
keyword test in the same shape as the existing `FORECAST_WORDS`, and a question
about 2019 fails it. That bounds cost and injection surface in one decision,
before any network exists.

**No web result is ever worth failing an answer.** `safe_search` has a hard
ceiling and swallows everything, in the same spirit as `RETRIEVAL_TIMEOUT_S` — the
graph has already produced a real answer by the time these results are wanted, so
the worst outcome of a bad day at the provider is the answer the system would
have given anyway.
"""

from __future__ import annotations

import asyncio
import logging

from ceynex import settings
from ceynex.observability import trace
from ceynex.websearch.providers import TavilyProvider, WebSearchProvider
from ceynex.websearch.schema import WebResult

log = logging.getLogger(__name__)

#: A hard ceiling on the whole round trip, not a per-hop budget — the same
#: reasoning as `RETRIEVAL_TIMEOUT_S = 2.0`. It runs concurrently with a fan-out
#: that takes seconds, so this is generous without costing anything.
WEBSEARCH_TIMEOUT_S = 4.0

MAX_RESULTS = 4

#: Recency words, deliberately narrow. A false negative costs a question its
#: related coverage; a false positive costs an outbound call and widens the
#: untrusted surface on a question that had no use for it.
CURRENT_WORDS = (
    "recent", "recently", "latest", "current", "currently", "today", "this week",
    "this month", "right now", "news", "headline", "just announced", "so far",
    "up to date", "new ", "newly", "at the moment", "these days",
)


def wants_current_context(query: str) -> bool:
    """Whether today's web could plausibly add anything to this question."""
    lowered = query.lower()
    return any(word in lowered for word in CURRENT_WORDS)


def from_settings() -> WebSearchProvider | None:
    """Build a provider, or return `None` — which is an ordinary state.

    Two ways to get `None`, both unexceptional, mirroring
    `PolicyRetriever.from_settings()`:

    - `CEYNEX_WEB_SEARCH=off`, which is what makes "the answers are byte-identical
      to the pre-web-search system" a claim anyone can check;
    - no `TAVILY_API_KEY`.

    **There is deliberately no keyless fallback.** An earlier note promised one;
    a scraped keyless provider is fragile and widens the untrusted surface for
    very little, so absence of a key means the feature is simply off — which is
    also the behaviour the design's own "answers exactly as it does today"
    guarantee describes.
    """
    if not settings.web_search_enabled():
        log.info("web search disabled by CEYNEX_WEB_SEARCH")
        return None
    key = settings.tavily_api_key()
    if not key:
        log.info("no TAVILY_API_KEY configured — web search is off")
        return None
    return TavilyProvider(key)


#: One key for everybody, on purpose. See config/api.yaml's websearch_rate_limit
#: block: the per-user limits bound fairness, this bounds the bill.
GLOBAL_IDENTITY = "websearch:global"

_window = None


def _global_window():
    """Built on first use, not at import — `build_window` reads REDIS_URL."""
    global _window  # noqa: PLW0603 - one process-lifetime object
    if _window is None:
        from ceynex.api import rate_limit

        _window = rate_limit.build_window()
    return _window


def set_global_window(window) -> None:
    """Test seam. Production never calls this."""
    global _window  # noqa: PLW0603
    _window = window


async def _within_global_cap() -> bool:
    """Whether the deployment as a whole may make another outbound call.

    Fails *open* on any error, exactly as `rate_limit.RedisWindow` does: a
    throttle that takes the feature down when Redis blinks is worse than the
    spend it was protecting against.
    """
    config = settings.load_config("api").get("websearch_rate_limit", {})
    if not config.get("enabled", True):
        return True
    try:
        decision = await _global_window().check(
            GLOBAL_IDENTITY,
            int(config.get("searches_per_minute", 60)),
            int(config.get("window_seconds", 60)),
        )
    except Exception:  # noqa: BLE001 - never let the throttle be the failure
        return True
    if not decision.allowed:
        log.info("global web-search cap reached; answering without it")
    return decision.allowed


async def safe_search(
    provider: WebSearchProvider | None,
    query: str,
    *,
    limit: int = MAX_RESULTS,
    timeout_s: float = WEBSEARCH_TIMEOUT_S,
) -> list[WebResult]:
    """Search, or return nothing. Never raises, never delays past `timeout_s`."""
    if provider is None:
        return []
    if not await _within_global_cap():
        trace.emit("web_search", query=query, results=0, status="capped")
        return []
    try:
        results = await asyncio.wait_for(provider.search(query, limit=limit), timeout=timeout_s)
    except TimeoutError:
        log.info("web search exceeded %.1fs; answering without it", timeout_s)
        trace.emit("web_search", query=query, results=0, status="timeout")
        return []
    except Exception as exc:  # noqa: BLE001 - a web result never fails an answer
        log.warning("web search failed; answering without it: %s", exc)
        trace.emit("web_search", query=query, results=0, status="failed")
        return []

    trace.emit(
        "web_search",
        query=query,
        results=len(results),
        status="ok",
        domains=[r.domain for r in results if r.domain],
    )
    return list(results)


__all__ = [
    "CURRENT_WORDS",
    "GLOBAL_IDENTITY",
    "set_global_window",
    "MAX_RESULTS",
    "WEBSEARCH_TIMEOUT_S",
    "from_settings",
    "safe_search",
    "wants_current_context",
]
