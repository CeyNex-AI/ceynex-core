"""General web search as enrichment, never as an agent — deviation D14."""

from ceynex.websearch.client import (
    WEBSEARCH_TIMEOUT_S,
    from_settings,
    safe_search,
    wants_current_context,
)
from ceynex.websearch.providers import FixtureProvider, TavilyProvider, WebSearchProvider
from ceynex.websearch.schema import WebResult

__all__ = [
    "WEBSEARCH_TIMEOUT_S",
    "FixtureProvider",
    "TavilyProvider",
    "WebResult",
    "WebSearchProvider",
    "from_settings",
    "safe_search",
    "wants_current_context",
]
