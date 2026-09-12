"""Process-lifetime resources for the API — SAD §7 Web/API Server Node.

The Neo4j driver holds a connection pool and the LLM client holds a prompt cache.
Building either per request would open a pool per request and throw the cache
away between them, which is most of the response-time budget (SRS 3.4.1) spent on
setup. Both are built once at startup and shared.

The compiled LangGraph graph is also built once. Compilation is not free and the
graph is stateless between invocations — state lives in `AgentState`, which is
passed in per query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ceynex.agents.common import AgentDeps
from ceynex.kg.client import KnowledgeGraphClient
from ceynex.llm import LLMReasoningClient
from ceynex.news.gdelt import GdeltClient
from ceynex.news.store import NewsStore
from ceynex.orchestrator.graph import build_graph
from ceynex.retrieval.client import PolicyRetriever
from ceynex.websearch import WebSearchProvider
from ceynex.websearch import from_settings as websearch_from_settings

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Everything the API holds open for its lifetime."""

    kg: KnowledgeGraphClient
    llm: LLMReasoningClient
    deps: AgentDeps
    graph: Any
    policy: PolicyRetriever | None = None
    #: The news sidecar (D11). Both None-tolerant: without them the endpoints
    #: still answer, they just answer "unavailable". Deliberately *not* in
    #: `AgentDeps` — no agent may reach news, and the way to guarantee that is
    #: for it never to be handed to one.
    gdelt: GdeltClient | None = None
    news: NewsStore | None = None
    #: General web search (D14). Here for exactly the reason `news` is here and
    #: not in `AgentDeps`: no agent may reach it. Web results are appended after
    #: `merge()` has returned, so they cannot reach the merge LLM, cannot enter
    #: grounding, and cannot move confidence — and an agent holding the provider
    #: could undo all three without anyone noticing.
    websearch: WebSearchProvider | None = None

    @classmethod
    def build(cls) -> Runtime:
        kg = KnowledgeGraphClient()
        llm = LLMReasoningClient()
        # None when retrieval is off, unconfigured, or the [policy] extra was not
        # installed. The agent treats all three the same as an absent LLM key:
        # answer without it (D10).
        policy = PolicyRetriever.from_settings()
        deps = AgentDeps(kg=kg, llm=llm, extras={"policy": policy})
        # Routing through the LLM only when there is a key. Without one the
        # keyword router runs and the system still answers (SRS 3.4.3).
        graph = build_graph(deps, use_llm_router=llm.available)
        gdelt = GdeltClient.from_settings()
        news = NewsStore.from_settings()
        # Not passed to `AgentDeps` above, and that omission is the enforcement.
        websearch = websearch_from_settings()
        log.info(
            "runtime built: llm=%s router=%s policy_retrieval=%s news=%s news_index=%s websearch=%s",
            "available" if llm.available else "degraded",
            "llm" if llm.available else "keyword",
            "on" if policy else "off",
            "on" if gdelt else "off",
            "on" if news else "off",
            "on" if websearch else "off",
        )
        return cls(
            kg=kg,
            llm=llm,
            deps=deps,
            graph=graph,
            policy=policy,
            gdelt=gdelt,
            news=news,
            websearch=websearch,
        )

    async def warmup(self) -> None:
        """Load the retrieval models and prepare the news collection.

        Their ONNX sessions take about a second to build — half again the whole
        per-query retrieval budget — so paying it here rather than inside the
        first query is the difference between one slow startup and one query that
        mysteriously degrades and never reproduces.

        Started as a task rather than awaited by the lifespan: on a cold
        `fastembed_cache` volume this downloads several hundred megabytes, and
        the container healthcheck allows roughly 95 s before it starts killing
        the process. `_models_ready()` inside `search()` remains the safety net
        for a query that arrives first.
        """
        if self.policy is not None:
            await self.policy.warmup()
        if self.news is not None:
            await self.news.ensure_collection()

    async def aclose(self) -> None:
        await self.kg.close()
        if self.policy is not None:
            await self.policy.close()
        if self.gdelt is not None:
            await self.gdelt.close()
        if self.news is not None:
            await self.news.close()


_runtime: Runtime | None = None


def set_runtime(runtime: Runtime | None) -> None:
    global _runtime  # noqa: PLW0603 - one process-lifetime object, set at startup
    _runtime = runtime


def get_runtime() -> Runtime:
    """FastAPI dependency. Raises rather than lazily building a second runtime."""
    if _runtime is None:
        raise RuntimeError("runtime not initialised — the app lifespan did not run")
    return _runtime
