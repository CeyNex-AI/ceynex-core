"""Streams an answer while it is being produced — deviation D12.

The SRS says *"no persistent socket based or streaming protocol is required for
standard query submission and response."* Not required, not forbidden — so this
is an addition beyond the written design, in the same class as D10 (Qdrant) and
D11 (the news sidecar), and recorded the same way.

**Why it is worth the deviation.** `docs/EVALUATION.md` records single-sector p95
at 14.6s against SRS 3.4.1's 10s budget, with a 29.0s cold tail. Streaming does
not make the system faster. It moves time-to-first-paint from ~15s to under a
second and turns a documented budget breach into progressive disclosure, while
showing the Cypher, the vector filter and the model spend that the system was
already producing and discarding.

**SSE, not WebSocket.** One direction, no new dependency, testable with
`TestClient`. WebSocket is ruled out by infrastructure anyway:
`ceynex-infra/frontend/nginx.conf.template`'s `location /api/` sets no
`Upgrade`/`Connection` headers, so an upgrade cannot cross the proxy today.

Two nginx defaults would each silently break this, and both are handled here
rather than left to a deploy:

- `proxy_buffering` is **on** by default, so nginx would hold the whole response
  until the connection closed — the exact opposite of the feature. Setting
  `X-Accel-Buffering: no` on the response makes nginx stream it, with no
  infrastructure change needed. The explicit `location` block is still worth
  adding, but this works without waiting for it.
- `proxy_read_timeout` defaults to **60s**, so a stream silent for a minute gets
  a 504. `HEARTBEAT_INTERVAL_S` keeps the connection audibly alive well inside
  that, and `STREAM_BUDGET_S` finishes before nginx would give up so *this* code
  produces the failure rather than the proxy dropping the socket.

**No HTTP error status once streaming has begun.** Headers go out at 200 with the
first byte, so a graph failure, a timeout and a disconnect must each become an
in-band `error` frame followed by `done`. That is a property of SSE, not a
choice, and every failure path below is written for it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ceynex import settings
from ceynex.api import rate_limit
from ceynex.api.deps import Runtime, get_runtime
from ceynex.api.query_runner import OrchestrationError, QueryOutcome, run_query
from ceynex.api.routes.auth import TokenPayload, get_optional_user, require_user
from ceynex.api.schemas import ChatStreamRequest, ClarifyAnswerRequest
from ceynex.chat import clarify, classify, instructions, store, titles, turn
from ceynex.observability import context as obs
from ceynex.observability import ledger, trace

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

#: Worst realistic case is route (8s LLM timeout) + the slowest parallel agent
#: (NODE_TIMEOUT_S = 12s) + merge (8s primary, then a 10s OpenRouter failsafe)
#: ~= 38s. This clears that with margin while staying meaningfully under nginx's
#: 60s read timeout, so a pathological request ends in our own `error` frame
#: rather than the proxy dropping the socket with nothing on the wire.
STREAM_BUDGET_S = 45.0

#: Comment frames keep nginx's 60s `proxy_read_timeout` from ever firing. Well
#: inside it rather than just under, because the timeout counts silence and a
#: heartbeat that races the deadline is a heartbeat that sometimes loses.
HEARTBEAT_INTERVAL_S = 15.0

#: How often the loop wakes to check for a disconnect and to consider a
#: heartbeat. Also the tail latency between the graph finishing and the `done`
#: frame, which is why it is well under a second.
POLL_INTERVAL_S = 0.25


async def enforce_chat_rate_limit(
    http_request: Request,
    user: TokenPayload | None = Depends(get_optional_user),  # noqa: B008
) -> None:
    """SRS 3.4.6's shape, applied to the conversational surface's own allowance.

    Prefixed with `chat:` for the same reason `news.py` prefixes with `news:` —
    `rate_limit.KEY_PREFIX` is shared by every `Window` from that module, so an
    unprefixed identity would write to the *same* Redis keys as `/api/query` and
    a conversation would silently spend the user's ability to ask a fresh
    question from the classic page.

    A separate allowance, not an exemption: this endpoint invokes the identical
    five-agent fan-out, so leaving it unlimited would be a documented bypass
    around the requirement for the most expensive call in the system.
    """
    config = settings.load_config("api").get("chat_rate_limit", {})
    if not config.get("enabled", True):
        return

    limit = int(config.get("turns_per_minute", 45))
    window_s = int(config.get("window_seconds", 60))
    identity = "chat:" + rate_limit.identity_of(
        user.email if user else None,
        http_request.client.host if http_request.client else None,
    )

    decision = await _chat_window().check(identity, limit, window_s)
    if decision.allowed:
        return

    log.info("chat rate limit hit by %s (%d/%ds)", identity, limit, window_s)
    raise HTTPException(
        status_code=429,
        detail=(
            f"rate limit exceeded: at most {limit} conversation turns per "
            f"{window_s} seconds. Try again in {decision.retry_after_s}s."
        ),
        headers={"Retry-After": str(decision.retry_after_s)},
    )


_chat_window_singleton: rate_limit.Window | None = None


def _chat_window() -> rate_limit.Window:
    """Built on first use, not at import — `build_window` reads REDIS_URL."""
    global _chat_window_singleton  # noqa: PLW0603 - one process-lifetime object
    if _chat_window_singleton is None:
        _chat_window_singleton = rate_limit.build_window()
    return _chat_window_singleton


def set_chat_window(window: rate_limit.Window | None) -> None:
    """Test seam. Production never calls this."""
    global _chat_window_singleton  # noqa: PLW0603
    _chat_window_singleton = window


def _frame(event: str, data: dict[str, Any]) -> str:
    """One SSE frame. `data` is one line — json.dumps never emits a raw newline."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _drain(sink: trace.TraceSink) -> list[trace.TraceEvent]:
    """Whatever is queued right now, without waiting.

    A yield point first: events emitted from a worker thread (the retriever runs
    its ONNX embed and rerank through `asyncio.to_thread`) are enqueued via
    `call_soon_threadsafe`, so their callbacks may still be pending when the
    graph task completes. Without this, the last few steps of a retrieval-heavy
    query would be missing from the live stream and present in the replay — the
    two telling different stories about the same request.
    """
    await asyncio.sleep(0)
    events = []
    while True:
        try:
            events.append(sink.queue.get_nowait())
        except asyncio.QueueEmpty:
            return events


async def _resolve_turn(
    runtime: Runtime,
    query: str,
    conversation_id: int | None,
    user_email: str | None,
) -> tuple[classify.TurnDecision | None, store.Message | None, str]:
    """Decide whether this turn re-runs the graph, and against what.

    Returns `(decision, prior_answer, prior_query)`. A `None` decision means
    there is nothing to follow up — a first turn, or a conversation whose
    transcript could not be read — so the caller runs the graph, which is the
    safe default in both cases.
    """
    if conversation_id is None or user_email is None:
        return None, None, ""

    try:
        transcript = await store.messages(conversation_id, user_email)
    except store.ChatStoreUnavailableError as exc:
        log.warning("could not read conversation %s: %s", conversation_id, exc)
        return None, None, ""

    if not transcript:
        return None, None, ""

    # Any prior assistant turn, not only one that produced evidence. Requiring
    # evidence meant that when an analysis *declined* — no data loaded, agents
    # unavailable — "explain that more simply" found no prior answer, fell
    # through to the analyse path, and re-ran the whole fan-out to produce the
    # identical decline. A decline is exactly when someone asks what you meant,
    # and `turn.discuss` handles an empty evidence list fine.
    prior_answer = next(
        (m for m in reversed(transcript) if m.role == "assistant"), None
    )
    if prior_answer is None:
        return None, None, ""

    # `asked`, not `content`: the transcript keeps what the reader typed, but a
    # follow-up is about the question the system actually ran.
    prior_query = next(
        (m.asked for m in reversed(transcript) if m.role == "user" and m.seq < prior_answer.seq),
        "",
    )
    decision = await classify.llm_turn(query, prior_query, prior_answer.content, runtime.llm)
    return decision, prior_answer, prior_query


async def _maybe_clarify(
    runtime: Runtime,
    query: str,
    conversation_id: int | None,
    user_email: str | None,
    *,
    typed: str | None = None,
) -> str | None:
    """One question back, or nothing at all. Never raises.

    Stage A is free and silent on every question in `eval/questions.yaml`, so the
    common path costs one keyword pass and returns here immediately. Only when it
    fires does anything else happen — and only then is an LLM call made.

    Needs somewhere to remember the pending question, so an anonymous turn (no
    conversation) is never clarified: there would be nowhere to resume to.
    """
    if not settings.clarify_enabled() or conversation_id is None or user_email is None:
        return None

    trigger = clarify.clarification_needed(query)
    if trigger is None:
        return None

    clarification = await clarify.llm_clarify(trigger, runtime.llm)
    if clarification is None:  # the model vetoed a gate the syntax check opened
        trace.emit("clarify", kind=trigger.kind, asked=False, method="llm-veto")
        return None

    try:
        # `typed` rides along in the stored payload (never in the frame) so the
        # resume can record the reader's own words even when the gate fired on
        # a follow-up that had already been rewritten into `query`.
        pending_id = await store.record_clarification(
            conversation_id, user_email, query,
            {**clarification.as_payload(), "typed": typed or query},
        )
    except store.ChatStoreUnavailableError as exc:
        # Without somewhere to resume from, asking would strand the reader on a
        # question whose answer goes nowhere. Answering the original is strictly
        # better than that.
        log.warning("could not store a clarification; answering as asked: %s", exc)
        return None

    trace.emit("clarify", kind=trigger.kind, asked=True, method=clarification.method,
               question=clarification.question)
    return _frame("clarify", {"pending_id": pending_id, **clarification.as_payload()})


async def _stream(
    http_request: Request,
    runtime: Runtime,
    query: str,
    user: TokenPayload | None,
    conversation_id: int | None = None,
    *,
    skip_clarify: bool = False,
    typed: str | None = None,
) -> AsyncIterator[str]:
    """`typed` is what the reader actually wrote, when `query` is not it — the
    clarify resume passes the original question and a composed `query`. The
    transcript stores `typed`; the graph runs `query`."""
    loop = asyncio.get_running_loop()
    sink = trace.TraceSink(request_id="pending", loop=loop)
    user_email = user.email if user else None
    typed = typed if typed is not None else query

    yield _frame("start", {"query": query, "budget_s": STREAM_BUDGET_S,
                           "conversation_id": conversation_id})
    last_write = time.monotonic()

    decision, prior, prior_query = await _resolve_turn(
        runtime, query, conversation_id, user_email
    )

    if decision is not None and decision.mode == "discuss":
        # The cheap path: answered from the evidence already on screen, with no
        # fan-out at all. Most follow-ups are this, which is what keeps a
        # conversation from costing five agent runs per turn.
        yield _frame("turn", {"mode": "discuss", "method": decision.method,
                              "reason": decision.reason})
        async for chunk in _discuss_stream(
            runtime, query, prior, prior_query, conversation_id, user_email
        ):
            yield chunk
        return

    if decision is not None:
        yield _frame("turn", {"mode": "analyse", "method": decision.method,
                              "reason": decision.reason,
                              "standalone_query": decision.standalone_query})
        query = decision.standalone_query or query

    if not skip_clarify:
        asked = await _maybe_clarify(runtime, query, conversation_id, user_email, typed=typed)
        if asked is not None:
            yield asked
            yield _frame("done", {"failed": False, "clarify": True,
                                  "conversation_id": conversation_id})
            return

    task = asyncio.create_task(
        run_query(
            runtime,
            query,
            user_email=user_email,
            overall_timeout_s=STREAM_BUDGET_S,
            trace_sink=sink,
            conversation_id=conversation_id,
        )
    )

    try:
        while True:
            if await http_request.is_disconnected():
                # The client is gone. Cancelling stops the fan-out rather than
                # letting five agents finish an answer nobody will read.
                log.info("chat stream client disconnected; cancelling the graph")
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                return

            try:
                event = await asyncio.wait_for(sink.queue.get(), timeout=POLL_INTERVAL_S)
            except TimeoutError:
                if task.done():
                    break
                now = time.monotonic()
                if now - last_write >= HEARTBEAT_INTERVAL_S:
                    yield ": heartbeat\n\n"
                    last_write = now
                continue

            yield _frame(event.kind, event.as_dict())
            last_write = time.monotonic()

        for event in await _drain(sink):
            yield _frame(event.kind, event.as_dict())

        outcome: QueryOutcome = task.result()
    except OrchestrationError as exc:
        yield _frame("error", {"message": str(exc)})
        yield _frame("done", {"failed": True})
        return
    except Exception as exc:  # noqa: BLE001 - a stream never becomes an HTTP status
        log.exception("chat stream failed")
        yield _frame("error", {"message": f"unexpected failure: {exc}"})
        yield _frame("done", {"failed": True})
        return

    answer = outcome.response.model_dump(mode="json")
    usage = outcome.usage.as_summary()

    # Persistence after the answer is assembled and before `done`, so a client
    # that reloads immediately finds the turn already there. Failures here are
    # logged, never surfaced: the answer has been produced, and losing its
    # transcript row is not worth turning a success into an error frame.
    ids: list[int] = []
    if conversation_id is not None and user_email is not None:
        ids = await _persist_turn(
            conversation_id, user_email, typed, answer, outcome, usage, sink,
            runtime.llm, mode="analyse", effective_query=query,
        )

    yield _frame(
        "done",
        {
            "failed": False,
            "request_id": outcome.request_id,
            "conversation_id": conversation_id,
            "usage": usage,
            "dropped_events": sink.dropped,
            "query_history_id": outcome.history_id,
            # What actually ran, when it is not what was typed — the same value
            # the transcript stores as the user turn's `effective_query`.
            "effective_query": query if query != typed else None,
            **_message_ids(ids),
            "answer": answer,
        },
    )


def _message_ids(ids: list[int]) -> dict[str, int | None]:
    """The rows a finished turn was stored as, so the client can fold the live
    turn into its transcript without a reload — and rate it, save it or
    regenerate it straight away. Both None when the turn was not persisted."""
    if len(ids) >= 2:
        return {"user_message_id": ids[0], "message_id": ids[1]}
    return {"user_message_id": None, "message_id": ids[0] if ids else None}


async def _discuss_stream(
    runtime: Runtime,
    follow_up: str,
    prior: store.Message,
    prior_query: str,
    conversation_id: int | None,
    user_email: str | None,
) -> AsyncIterator[str]:
    """A follow-up answered from the previous turn — no graph, no Cypher.

    Still a stream rather than a plain response, so the client has one transport
    and one set of frame handlers regardless of which path a turn takes.
    """
    # The reader's standing instruction, read the way `run_query` reads it for
    # the analyse path — without it a reader who asked for bullet points got them
    # on the first answer and plain prose on every follow-up.
    instruction, instruction_on = await instructions.get(user_email)
    observation = obs.RequestObservability(
        user_email=user_email,
        conversation_id=conversation_id,
        instruction=instruction if instruction_on else "",
    )
    token = obs.install(observation)
    try:
        result = await turn.discuss(follow_up, prior, prior_query, runtime.llm)
    except Exception as exc:  # noqa: BLE001 - a stream never becomes an HTTP status
        log.exception("discuss turn failed")
        yield _frame("error", {"message": f"unexpected failure: {exc}"})
        yield _frame("done", {"failed": True})
        return
    finally:
        obs.reset(token)

    await ledger.record(
        request_id=observation.request_id,
        user_email=user_email,
        conversation_id=conversation_id,
        calls=observation.usage.calls,
    )

    usage = observation.usage.as_summary()
    answer = {
        "answer": result.answer,
        # The confidence, band and panels are the *previous* turn's, carried
        # forward unchanged — a discussion does not produce a new analysis, and
        # inventing a fresh confidence score for it would be a number with
        # nothing behind it.
        "confidence": prior.confidence,
        "confidence_band": prior.confidence_band,
        "confidence_breakdown": prior.confidence_breakdown,
        "agents_used": prior.agents_used,
        "evidence": result.evidence,
        "forecast": prior.forecast,
        "graph": prior.graph,
        "degraded": result.degraded,
        "route": prior.route,
        "sectors": prior.sectors,
        "unanswered": prior.unanswered,
        "elapsed_ms": None,
        "grounded": result.grounded,
    }

    ids: list[int] = []
    if conversation_id is not None and user_email is not None:
        ids = await _persist_turn(
            conversation_id, user_email, follow_up, answer, None, usage, None,
            runtime.llm, mode="discuss"
        )

    yield _frame(
        "done",
        {
            "failed": False,
            "request_id": observation.request_id,
            "conversation_id": conversation_id,
            "usage": usage,
            "dropped_events": 0,
            # A discussion is not a new analysis and writes no history row.
            "query_history_id": None,
            **_message_ids(ids),
            "answer": answer,
        },
    )


async def _persist_turn(
    conversation_id: int,
    user_email: str,
    question: str,
    answer: dict[str, Any],
    outcome: QueryOutcome | None,
    usage: dict[str, Any],
    sink: trace.TraceSink | None,
    llm: Any,
    *,
    mode: str,
    effective_query: str | None = None,
) -> list[int]:
    """Write the question, the answer and the trace. Never raises.

    Returns the new `[user, assistant]` message ids, or `[]` when nothing could
    be written. The turn has already been delivered by the time this runs, so a
    database outage here costs a transcript row, not an answer — the same trade
    `history.record` makes and for the same reason.

    `question` is what the reader typed. `effective_query` is what the graph
    ran, stored only when the two differ, so a reopened transcript shows the
    reader's own words and the trace still says what was run.
    """
    request_id = outcome.request_id if outcome else None
    ran = effective_query if effective_query and effective_query != question else None
    try:
        ids = await store.append(
            conversation_id,
            user_email,
            [
                store.Message(role="user", content=question, effective_query=ran),
                store.Message(
                    role="assistant",
                    content=answer.get("answer", ""),
                    mode=mode,
                    request_id=request_id,
                    confidence=answer.get("confidence"),
                    confidence_band=answer.get("confidence_band"),
                    degraded=bool(answer.get("degraded", False)),
                    agents_used=answer.get("agents_used") or [],
                    route=answer.get("route") or [],
                    sectors=answer.get("sectors") or [],
                    unanswered=answer.get("unanswered") or [],
                    evidence=answer.get("evidence") or [],
                    forecast=answer.get("forecast"),
                    graph=answer.get("graph"),
                    elapsed_ms=answer.get("elapsed_ms"),
                    usage=usage,
                    query_history_id=outcome.history_id if outcome else None,
                    confidence_breakdown=answer.get("confidence_breakdown"),
                    grounded=answer.get("grounded"),
                ),
            ],
        )
    except store.ChatStoreUnavailableError as exc:
        log.warning("could not persist turn in conversation %s: %s", conversation_id, exc)
        return []

    if sink is not None and request_id:
        await store.save_trace(request_id, conversation_id, sink.history)

    # After the answer is delivered, and conditional in SQL, so a slow or absent
    # model costs a plainer name rather than a slower turn.
    name = await titles.title_for(question, answer.get("answer", ""), llm)
    await store.set_title_if_unset(conversation_id, user_email, name)
    return list(ids or [])


@router.post("/api/chat/stream", dependencies=[Depends(enforce_chat_rate_limit)])
async def chat_stream(
    request: ChatStreamRequest,
    http_request: Request,
    runtime: Runtime = Depends(get_runtime),  # noqa: B008 - FastAPI's dependency idiom
    user: TokenPayload | None = Depends(get_optional_user),  # noqa: B008
) -> StreamingResponse:
    """The same answer as `POST /api/query`, reported as it is assembled.

    Rate-limited by the *same* dependency as `/api/query`, deliberately: this
    invokes the identical five-agent fan-out, so leaving it off would make this
    endpoint a bypass around SRS 3.4.6 for the most expensive call in the system.

    `conversation_id` is optional and requires a signed-in owner. Without one the
    turn is a stateless question, exactly as `/api/query` is — which keeps the
    streaming transport usable for a demo before any account exists.
    """
    if not settings.chat_enabled():
        raise HTTPException(status_code=404, detail="chat is not enabled on this deployment")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    conversation_id = request.conversation_id
    if conversation_id is not None:
        if user is None:
            raise HTTPException(status_code=401, detail="a conversation needs a signed-in user")
        try:
            if not await store.owns(conversation_id, user.email):
                raise HTTPException(status_code=404, detail="conversation not found")
        except store.ChatStoreUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return StreamingResponse(
        _stream(http_request, runtime, query, user, conversation_id),
        media_type="text/event-stream",
        headers={
            # nginx honours this and stops buffering — the difference between a
            # live trace and one whole response arriving at the end.
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post(
    "/api/chat/clarify/{pending_id}/answer",
    dependencies=[Depends(enforce_chat_rate_limit)],
)
async def answer_clarification(
    pending_id: int,
    request: ClarifyAnswerRequest,
    http_request: Request,
    runtime: Runtime = Depends(get_runtime),  # noqa: B008 - FastAPI's dependency idiom
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> StreamingResponse:
    """Resume a turn the gate paused, and stream the answer.

    **Rate-limited by the same dependency as the stream itself.** This runs the
    identical five-agent fan-out, so omitting it would leave a bypass around
    SRS 3.4.6 through the one route added last.

    **`skip_clarify=True`, always.** The gate is not on this path at all, which is
    what makes the one-round cap structural rather than a counter — there is
    nothing to keep in sync across the two uvicorn workers. Claiming the pending
    row is a conditional `UPDATE ... RETURNING`, so a double submission resolves
    once and 404s the second time rather than running the turn twice.
    """
    if not settings.chat_enabled():
        raise HTTPException(status_code=404, detail="chat is not enabled on this deployment")

    try:
        pending = await store.resolve_clarification(pending_id, user.email)
    except store.ChatStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if pending is None:
        raise HTTPException(
            status_code=404, detail="that question is no longer waiting for an answer"
        )

    original = str(pending["original_query"])
    query = original if request.skip else clarify.Clarification.compose(original, request.answers)
    typed = str((pending.get("payload") or {}).get("typed") or original)

    return StreamingResponse(
        _stream(
            http_request,
            runtime,
            query,
            user,
            int(pending["conversation_id"]),
            skip_clarify=True,
            typed=typed,
        ),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
