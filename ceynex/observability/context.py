"""Per-request ambient state — the carrier for SRS 3.1.4's transparency, live.

SRS 3.1.4 requires every answer to be accompanied by the evidence behind it. The
system already satisfies that *after the fact*: `Evidence.detail` carries the
literal Cypher, the vector filter, the model id. What it cannot do today is say
any of it **while the answer is being computed**, and a measured p95 of 14.6s
against a 10s budget (docs/EVALUATION.md) is a long time to show nothing.

Reporting from the call sites that already hold the payload means those call
sites need a per-request place to write to. There are only two candidates:

- thread a sink parameter through every signature — `kg.run`, `llm.generate`,
  `retriever.search`, every agent node. That is a cross-team diff touching M1's
  and M3's files to add a parameter none of their code cares about.
- an ambient, context-scoped variable that is a no-op when absent.

This module is the second. `AgentDeps` cannot serve: `api/deps.py` builds one
per process and shares it across every request, so it has nowhere to put
per-request state.

**Why this survives the parallel fan-out.** `langgraph/pregel/_executor.py`'s
`AsyncBackgroundExecutor.submit()` calls `copy_context()` once per dispatched
node and threads it into `loop.create_task(coro, context=...)`. Each of the 2-5
parallel agents therefore gets an independent *copy* of the context live at
`ainvoke()`. Two consequences that shape everything below:

1. A binding installed before `ainvoke()` is visible in every node.
2. A `set()` *inside* a node cannot propagate back out — the context is a copy.

So the value stored here is a **reference to a mutable object**. Siblings hold
isolated contexts but the same underlying sink, and appending reaches everyone.
Never re-`set()` these vars from inside a node.

Those are private langgraph modules, so `tests/observability/test_context_propagation.py`
is a canary and `pyproject.toml` pins `langgraph>=1.2,<1.3`. If a future version
ever breaks it, the fallback is langgraph's public `RunnableConfig` mechanism —
larger, because it means editing every agent signature.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from ceynex.observability.trace import TraceSink


@dataclass
class LLMCall:
    """One `generate()` attempt, as it will be written to the ledger.

    A row per *call* rather than per request: a request that times out halfway
    has still spent real money, and that spend is exactly what a cost ledger
    exists to record. It also makes every rollup a plain GROUP BY instead of a
    nested-JSON format nobody can query.
    """

    role: str
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    elapsed_ms: float = 0.0
    cache_hit: bool = False
    fallback: bool = False
    failed: bool = False


@dataclass
class RequestLLMUsage:
    """What this one request spent, separate from the process-lifetime total.

    `LLMReasoningClient.usage` stays exactly as it is — it is what
    `_cap_reached()` reads, and repurposing it per-request would silently
    disable the R5 spend cap. This accumulates alongside it.

    Mutated only from `generate()`, which never awaits between reading and
    writing, so no other coroutine can interleave mid-update on the loop thread.
    Unlike the trace sink, `generate()` is never called from a worker thread, so
    this needs no cross-thread marshalling.
    """

    calls: list[LLMCall] = field(default_factory=list)

    def record(self, call: LLMCall) -> None:
        self.calls.append(call)

    @property
    def tokens_in(self) -> int:
        return sum(call.tokens_in for call in self.calls)

    @property
    def tokens_out(self) -> int:
        return sum(call.tokens_out for call in self.calls)

    @property
    def cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)

    @property
    def cache_hits(self) -> int:
        return sum(1 for call in self.calls if call.cache_hit)

    def as_summary(self) -> dict[str, float | int]:
        """The per-message footer: "1,240 in / 890 out · $0.0021 · 4 calls"."""
        return {
            "calls": len(self.calls),
            "cache_hits": self.cache_hits,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class RequestObservability:
    """Everything ambient to one request.

    `trace` is None for `POST /api/query`: that transport has nowhere to put
    events, but its LLM spend still belongs in the ledger. Usage collection and
    event streaming are therefore independent — one can be on without the other.
    """

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trace: TraceSink | None = None
    usage: RequestLLMUsage = field(default_factory=RequestLLMUsage)
    user_email: str | None = None
    conversation_id: int | None = None
    #: This reader's standing instruction about how answers should read (D15).
    #: Carried here rather than on `AgentState` for the same reason the trace
    #: sink is: it is per-request ambient context, the contract is frozen, and
    #: `graph.py` stays exactly `route -> fan-out -> merge -> END`. Empty is the
    #: overwhelmingly common case and reproduces the original prompt exactly.
    instruction: str = ""
    #: The SRS 3.1.4 confidence terms, written by `merge_node` and read by the
    #: response assembler. Here for the same reason `instruction` is: the state
    #: contract is frozen at three keys out of merge, and this is per-request
    #: ambient data rather than something the graph should carry.
    confidence_breakdown: dict | None = None


_current: ContextVar[RequestObservability | None] = ContextVar(
    "ceynex_observability", default=None
)
#: Which graph node is executing, so an event emitted three layers down inside
#: `kg/client.py` can still be attributed to the agent that caused it.
_current_node: ContextVar[str | None] = ContextVar("ceynex_trace_node", default=None)


def install(obs: RequestObservability) -> Token[RequestObservability | None]:
    """Bind `obs` for this context. Call before `graph.ainvoke()`, never inside a node."""
    return _current.set(obs)


def reset(token: Token[RequestObservability | None]) -> None:
    _current.reset(token)


def current() -> RequestObservability | None:
    """The active request's ambient state, or None outside one.

    None is the normal case, not an error: `demo.py`, `eval/harness.py` and every
    existing test invoke the graph without installing anything, and must keep
    behaving exactly as they did.
    """
    return _current.get()


def current_node() -> str | None:
    return _current_node.get()


def set_node(name: str | None) -> Token[str | None]:
    return _current_node.set(name)


def reset_node(token: Token[str | None]) -> None:
    _current_node.reset(token)


def current_instruction() -> str:
    """This request's user instruction, or `""` outside a request."""
    obs = current()
    return obs.instruction if obs is not None else ""


def record_llm_call(call: LLMCall) -> None:
    """Ambient, no-op when no request is active. Called from `llm/client.py`."""
    obs = _current.get()
    if obs is None:
        return
    obs.usage.record(call)


__all__ = [
    "LLMCall",
    "RequestLLMUsage",
    "RequestObservability",
    "current",
    "current_instruction",
    "current_node",
    "install",
    "record_llm_call",
    "reset",
    "reset_node",
    "set_node",
]
