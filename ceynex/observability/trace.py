"""The trace event bus — SRS 3.1.4's evidence, reported while it is being gathered.

Every event here is emitted from a call site that actually ran, carrying its real
payload and its real duration. Nothing is synthesised to fill a gap in the
timeline: the project's central claim is that every figure traces to its source,
and a fabricated progress step is the one thing that would undermine it. A step
that took 40ms is reported as 40ms; the *interface* may hold it on screen long
enough to read, which is rendering, not invention.

**Why not `astream_events`.** LangGraph's own streaming sees node boundaries. The
events worth showing — *this Cypher returned 14 rows in 41ms*, *the reranker
rejected every passage*, *the merge call cost 1,840 tokens* — happen three layers
below a node, inside `kg/client.py`, `retrieval/client.py` and `llm/client.py`.

**The no-op guarantee.** `emit()` with no sink installed costs one `ContextVar`
lookup — tens of nanoseconds against calls that already cost milliseconds. That
is what lets `demo.py`, `eval/harness.py` and ~1,220 existing tests keep their
current behaviour without a single change.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from ceynex.observability import context

log = logging.getLogger(__name__)

#: Past this, a pathological instrumentation point cannot grow queue memory
#: without bound in a request whose producer outruns its consumer. Events past
#: it are counted and dropped, never blocked on — an answer must not wait on its
#: own progress reporting.
DEFAULT_MAX_QUEUED = 512

#: Cypher, SQL and snippets go into `detail` for display. Long ones are cut here
#: rather than at each call site, so no instrumentation point can accidentally
#: put a megabyte on the wire.
MAX_TEXT = 600


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_TEXT:
        return value[:MAX_TEXT] + " …"
    return value


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One thing that happened, in the order it happened."""

    seq: int
    request_id: str
    ts: float
    kind: str
    node: str | None
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "kind": self.kind,
            "node": self.node,
            **self.payload,
        }


@dataclass
class TraceSink:
    """A bounded queue for the live stream, plus the full ordered history.

    Two consumers with different needs: the SSE generator drains `queue` as
    events arrive, and `history` is written to `chat_trace_event` once at the end
    of the turn so reopening a past conversation replays the real trace. History
    is never dropped — only the live queue is, and only under backpressure.
    """

    request_id: str
    loop: asyncio.AbstractEventLoop
    max_queued: int = DEFAULT_MAX_QUEUED
    history: list[TraceEvent] = field(default_factory=list)
    dropped: int = 0
    _seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _queue: asyncio.Queue[TraceEvent] | None = field(default=None, repr=False)

    @property
    def queue(self) -> asyncio.Queue[TraceEvent]:
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queued)
        return self._queue

    def emit(
        self, kind: str, node: str | None, payload: Mapping[str, Any], *, persist: bool = True
    ) -> None:
        """Record an event. Never blocks, never raises into the caller.

        `seq` is assigned under a lock in the calling thread, so ordering is true
        arrival order even though producers span the event loop (`kg/client.py`)
        and worker threads (`retrieval/client.py` runs its ONNX embed and rerank
        through `asyncio.to_thread`). The queue put is then marshalled onto the
        loop when the caller is not already on it.

        `persist=False` is for the answer text itself (`answer_delta`): it is
        streamed and kept for a resume, but not written to the stored trace —
        the message row is the durable copy of the answer, and a second one in
        `chat_trace_event` would be a transcript nobody reconciles. Nor is it
        clipped: `MAX_TEXT` exists to keep a pathological Cypher string off the
        wire, and cutting a sentence of the answer mid-word would show the
        reader a draft the model never wrote.
        """
        values = dict(payload) if not persist else {k: _clip(v) for k, v in payload.items()}
        with self._lock:
            self._seq += 1
            event = TraceEvent(
                seq=self._seq,
                request_id=self.request_id,
                ts=time.time(),
                kind=kind,
                node=node,
                payload=values,
            )
            if persist:
                self.history.append(event)

        try:
            on_loop = asyncio.get_running_loop() is self.loop
        except RuntimeError:  # no running loop — we are in a worker thread
            on_loop = False

        if on_loop:
            self._enqueue(event)
        else:
            with contextlib.suppress(RuntimeError):  # loop already closed
                self.loop.call_soon_threadsafe(self._enqueue, event)

    def _enqueue(self, event: TraceEvent) -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1


def emit(kind: str, /, **payload: Any) -> None:
    """Record one event, or do nothing at all.

    The only function `kg/client.py`, `data/reader.py`, `retrieval/client.py`,
    `llm/client.py`, `websearch/client.py` and `orchestrator/graph.py` import.
    """
    obs = context.current()
    if obs is None or obs.trace is None:
        return
    obs.trace.emit(kind, context.current_node(), payload)


def emit_live(kind: str, /, **payload: Any) -> None:
    """Like `emit`, for the live stream and a resume only — never the stored trace.

    The answer text as it arrives (`orchestrator/answer_stream.py`). See
    `TraceSink.emit` for why it is kept out of `history`.
    """
    obs = context.current()
    if obs is None or obs.trace is None:
        return
    obs.trace.emit(kind, context.current_node(), payload, persist=False)


def active() -> bool:
    """Whether anything is listening. For skipping expensive payload assembly."""
    obs = context.current()
    return obs is not None and obs.trace is not None


@contextlib.contextmanager
def node(name: str, **start_payload: Any) -> Iterator[None]:
    """Attribute everything emitted inside to `name`, and time it.

    Used by `orchestrator/graph.py` around the route node, each wrapped agent and
    the merge node, so a Cypher query emitted deep inside an agent still carries
    the name of the agent that caused it.
    """
    token = context.set_node(name)
    started = time.perf_counter()
    emit("node_start", **start_payload)
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 - re-raised below; we only annotate
        emit(
            "node_error",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            error=str(exc),
        )
        raise
    else:
        emit("node_end", elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
    finally:
        context.reset_node(token)


__all__ = ["MAX_TEXT", "TraceEvent", "TraceSink", "active", "emit", "emit_live", "node"]
