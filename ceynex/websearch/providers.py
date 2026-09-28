"""Where web results come from — deviation D14.

`TavilyProvider` is reached over `httpx`, which this package already requires,
rather than by adding `tavily-python`. The dependency list is a tax on all three
members, and one POST to one documented endpoint does not justify a package.

Tavily is preferred where a key exists because it returns extracted page
*content* rather than link snippets, which is the difference between a result
that can be cited and one that can only be linked.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import httpx

from ceynex.websearch.schema import WebResult

log = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"


@runtime_checkable
class WebSearchProvider(Protocol):
    async def search(self, query: str, *, limit: int = 5) -> list[WebResult]: ...


#: A result's URL becomes an `href` in the evidence panel, so anything but a web
#: page — `javascript:`, `data:`, `file:` — is dropped here, at the boundary
#: where untrusted data enters, rather than trusted to every renderer.
ALLOWED_SCHEMES = ("http", "https")


class TavilyProvider:
    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = TAVILY_ENDPOINT,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._api_key = api_key
        self._endpoint = endpoint
        #: Test seam, the same idea as `set_global_window`. Production passes none.
        self._transport = transport

    async def search(self, query: str, *, limit: int = 5) -> list[WebResult]:
        # The key travels as a Bearer header, the only form Tavily's API reference
        # documents. It used to ride in the JSON body, which an older API
        # accepted; were that dropped, every call would 401 and `safe_search`
        # would swallow it as `failed`, leaving the feature on in name only.
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "query": query,
            "max_results": limit,
            "search_depth": "basic",
            # No raw page content: v1 keeps the untrusted surface to a snippet.
            "include_raw_content": False,
            "include_answer": False,
        }
        async with httpx.AsyncClient(timeout=10.0, transport=self._transport) as client:
            response = await client.post(self._endpoint, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()

        results = []
        for hit in body.get("results", [])[:limit]:
            url = str(hit.get("url") or "")
            if not url:
                continue
            try:
                parsed = httpx.URL(url)
            except httpx.InvalidURL:
                continue
            if parsed.scheme not in ALLOWED_SCHEMES or not parsed.host:
                log.info("dropped a web result with a %r URL", parsed.scheme)
                continue
            results.append(
                WebResult(
                    title=str(hit.get("title") or url),
                    url=url,
                    snippet=str(hit.get("content") or "")[:600],
                    published=str(hit["published_date"]) if hit.get("published_date") else None,
                    domain=parsed.host,
                )
            )
        return results


class FixtureProvider:
    """Committed results for tests. Never the network — the house rule."""

    def __init__(self, results: Sequence[WebResult]):
        self._results = list(results)
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int = 5) -> list[WebResult]:
        self.queries.append(query)
        return self._results[:limit]


__all__ = ["ALLOWED_SCHEMES", "FixtureProvider", "TavilyProvider", "WebSearchProvider"]
