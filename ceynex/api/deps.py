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
from ceynex.orchestrator.graph import build_graph
from ceynex.retrieval.client import PolicyRetriever

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Everything the API holds open for its lifetime."""

    kg: KnowledgeGraphClient
    llm: LLMReasoningClient
    deps: AgentDeps
    graph: Any
    policy: PolicyRetriever | None = None

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
        log.info(
            "runtime built: llm=%s router=%s policy_retrieval=%s",
            "available" if llm.available else "degraded",
            "llm" if llm.available else "keyword",
            "on" if policy else "off",
        )
        return cls(kg=kg, llm=llm, deps=deps, graph=graph, policy=policy)

    async def warmup(self) -> None:
        """Load the retrieval models before the first request.

        Their ONNX sessions take about a second to build — half again the whole
        per-query retrieval budget — so paying it here rather than inside the
        first query is the difference between one slow startup and one query that
        mysteriously degrades and never reproduces.
        """
        if self.policy is not None:
            await self.policy.warmup()

    async def aclose(self) -> None:
        await self.kg.close()
        if self.policy is not None:
            await self.policy.close()


_runtime: Runtime | None = None


def set_runtime(runtime: Runtime | None) -> None:
    global _runtime  # noqa: PLW0603 - one process-lifetime object, set at startup
    _runtime = runtime


def get_runtime() -> Runtime:
    """FastAPI dependency. Raises rather than lazily building a second runtime."""
    if _runtime is None:
        raise RuntimeError("runtime not initialised — the app lifespan did not run")
    return _runtime
