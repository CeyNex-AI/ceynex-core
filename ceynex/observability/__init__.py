"""Per-request tracing and LLM spend accounting (SRS 3.1.4, SRS 3.4.6).

A leaf package on purpose: it imports nothing from `ceynex` except `settings`,
so `kg/`, `llm/`, `retrieval/`, `data/` and `orchestrator/` can all emit into it
without an import cycle.

Import `trace.emit` at a call site; import `context` only where a request is
being set up or torn down.
"""

from ceynex.observability.context import (
    LLMCall,
    RequestLLMUsage,
    RequestObservability,
    current,
    install,
    record_llm_call,
    reset,
)
from ceynex.observability.trace import TraceEvent, TraceSink, active, emit, node

__all__ = [
    "LLMCall",
    "RequestLLMUsage",
    "RequestObservability",
    "TraceEvent",
    "TraceSink",
    "active",
    "current",
    "emit",
    "install",
    "node",
    "record_llm_call",
    "reset",
]
