"""Running one conversational turn, whoever is watching it — D12 (amended), D13.

SRS 3.1.1-3.1.4 and 3.4.1. The turn and the connection used to be one thing:
`routes/chat.py` ran the orchestration inside the HTTP response's generator, so
whatever happened to the socket happened to the answer. A connection that
dropped mid-turn cancelled the fan-out, and the reader was told "it may have
completed" about a turn the server had in fact just thrown away.

They are two things now:

- **`start_turn()`** runs a turn to completion as a task of its own —
  classification, the clarification gate, the discuss path or `run_query`, and
  persistence — publishing every frame into the turn's log
  (`api/turn_log.py`). It is bounded by `STREAM_BUDGET_S`, exactly as before.
- **`follow()`** is an HTTP response's view of that log: frames past a `seq`,
  heartbeats while it is quiet, and nothing else. Any number of them may follow
  one turn; the first one is simply the connection that started it.

**What a disconnect does now.** For a signed-in reader, nothing to the turn: it
finishes, it is persisted, and the reader can resume it (same `request_id`,
last `seq`) or find it in the reloaded transcript. The explicit Stop is
`cancel()`, reached through its own endpoint, because a browser's abort and a
dropped network look identical from here. An **anonymous** turn keeps the old
behaviour — cancelled on disconnect — because nobody can resume it and nothing
is stored, so finishing it would be spending money on an answer no one can
ever read.

**No HTTP error status once streaming has begun.** Headers go out at 200 with
the first byte, so a graph failure, a timeout and a cancellation each become
frames in the log — `error`, then `done` — never a status change.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from starlette.requests import Request

from ceynex import settings
from ceynex.api.deps import Runtime
from ceynex.api.query_runner import OrchestrationError, QueryOutcome, run_query
from ceynex.api.turn_log import LocalTurn, TurnFrame, mirror, registry
from ceynex.chat import clarify, classify, instructions, store, titles, turn
from ceynex.observability import context as obs
from ceynex.observability import ledger, trace

log = logging.getLogger(__name__)

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

#: How often a reader wakes to check its connection and consider a heartbeat,
#: and how often a running turn checks for a cross-worker cancel. Also the tail
#: latency between the graph finishing and the `done` frame.
POLL_INTERVAL_S = 0.25

#: A reader following a turn on *another* worker through the Redis mirror has
#: no task to watch, only frames. Past a whole budget plus a heartbeat of
#: silence, the turn is over whether or not its `done` frame survived — the
#: mirror may have expired, or the other worker may have died with it.
REMOTE_IDLE_LIMIT_S = STREAM_BUDGET_S + HEARTBEAT_INTERVAL_S


@dataclass
class TurnRequest:
    """Everything a turn needs, captured before the request handler returns."""

    runtime: Runtime
    #: What the graph runs — the typed question, or one composed with a
    #: clarification answer.
    query: str
    #: What the reader typed. The transcript stores this, never a rewrite.
    typed: str
    user_email: str | None
    conversation_id: int | None = None
    skip_clarify: bool = False
    #: Set for Regenerate: the answer being replaced, and what it answered.
    regenerate: RegeneratePlan | None = None


@dataclass(frozen=True)
class RegeneratePlan:
    """What a Regenerate re-runs, worked out from the transcript up front.

    Regenerate asks for a fresh answer to a question already asked, not a new
    turn: the mode is the original's (an analysis re-runs the graph, a
    discussion re-discusses the same prior answer), there is no classification
    and no clarifying question, and the new answer is stored beside the old one
    rather than over it.
    """

    #: The answer being replaced — always the conversation's latest.
    target: store.Message
    #: The user turn it answered.
    question: store.Message
    #: A discussion's subject: the answer before `question`. None for analysis.
    prior: store.Message | None
    prior_query: str

    @property
    def mode(self) -> str:
        return self.target.mode or "analyse"


def plan_regeneration(transcript: list[store.Message], message_id: int) -> RegeneratePlan | None:
    """The plan for regenerating `message_id`, or None if it cannot be.

    Only the latest answer, deliberately. Regenerating one further back would
    fork the conversation — every later turn was classified and grounded against
    the answer that is being replaced — and a transcript that silently branches
    is harder to trust than one that simply says "only the last answer".
    """
    answers = [m for m in transcript if m.role == "assistant"]
    if not answers or answers[-1].id != message_id:
        return None
    target = answers[-1]
    question = next(
        (m for m in reversed(transcript) if m.role == "user" and m.seq < target.seq), None
    )
    if question is None:
        return None
    prior = next(
        (m for m in reversed(transcript) if m.role == "assistant" and m.seq < question.seq), None
    )
    if (target.mode or "analyse") == "discuss" and prior is None:
        return None  # a discussion with nothing before it to discuss
    prior_query = ""
    if prior is not None:
        prior_query = next(
            (m.asked for m in reversed(transcript) if m.role == "user" and m.seq < prior.seq), ""
        )
    return RegeneratePlan(target=target, question=question, prior=prior, prior_query=prior_query)


def start_turn(request: TurnRequest) -> LocalTurn:
    """Start a turn as its own task and return its (live, empty) log."""
    request_id = uuid.uuid4().hex
    log_ = registry().start(request_id, owner=request.user_email,
                            conversation_id=request.conversation_id)
    log_.task = asyncio.create_task(_run(log_, request), name=f"chat-turn-{request_id[:8]}")
    return log_


async def cancel(turn_log: LocalTurn) -> bool:
    """Stop a running turn. False when it had already finished."""
    if turn_log.task is None or turn_log.task.done():
        return False
    turn_log.task.cancel()
    return True


async def follow(
    request_id: str,
    after: int,
    http_request: Request,
    *,
    cancel_on_disconnect: bool = False,
) -> AsyncIterator[str]:
    """Frames of a turn past `after`, as SSE, until its `done` frame.

    Served from this process's log when the turn runs here — always the case
    for the connection that started it — and from the Redis mirror otherwise.
    `cancel_on_disconnect` is for the anonymous path only; see the module
    docstring for why a signed-in turn outlives its reader.
    """
    local = registry().get(request_id)
    seq = after
    last_write = last_frame = time.monotonic()
    finished = False
    try:
        while True:
            if await http_request.is_disconnected():
                return

            if local is not None:
                frames = await local.wait_after(seq, POLL_INTERVAL_S)
            else:
                replica = mirror()
                frames = (
                    await replica.read(request_id, seq, int(POLL_INTERVAL_S * 1000))
                    if replica is not None
                    else []
                )

            for frame in frames:
                yield frame.as_sse()
                seq = frame.seq
                last_write = last_frame = time.monotonic()
                if frame.event == "done":
                    finished = True
                    return

            if local is None and time.monotonic() - last_frame > REMOTE_IDLE_LIMIT_S:
                finished = True
                yield _synthetic_done(seq).as_sse()
                return

            if local is not None and local.done and local.last_seq <= seq:
                # The task ended without a `done` frame — it died before it
                # could write one. Say so rather than leaving the reader waiting.
                finished = True
                yield _synthetic_done(seq).as_sse()
                return

            if not frames and time.monotonic() - last_write >= HEARTBEAT_INTERVAL_S:
                yield ": heartbeat\n\n"
                last_write = time.monotonic()
    finally:
        if not finished and cancel_on_disconnect and local is not None:
            log.info("anonymous chat reader went away; cancelling turn %s", request_id)
            await cancel(local)


def _synthetic_done(seq: int) -> TurnFrame:
    return TurnFrame(seq=seq + 1, event="done",
                     data={"failed": True, "reason": "the turn ended without finishing"})


# --- producing a turn --------------------------------------------------------


async def _run(turn_log: LocalTurn, request: TurnRequest) -> None:
    """The whole turn, publishing as it goes. Never raises out of the task."""
    replica = mirror()
    if replica is not None and request.user_email is not None:
        await replica.open(turn_log.request_id, request.user_email)

    async def publish(event: str, data: dict[str, Any]) -> None:
        frame = await turn_log.append(event, data)
        if replica is not None and request.user_email is not None:
            await replica.append(turn_log.request_id, frame)

    try:
        await _produce(turn_log, request, publish)
    except asyncio.CancelledError:
        # Stop, or an anonymous reader leaving. Nothing is persisted: the reader
        # chose not to have this answer, and a half-written turn in the
        # transcript would be a record of something that did not happen.
        log.info("chat turn %s cancelled", turn_log.request_id)
        with contextlib.suppress(Exception):
            await publish("done", {"failed": False, "cancelled": True,
                                   "request_id": turn_log.request_id,
                                   "conversation_id": request.conversation_id})
    except Exception as exc:  # noqa: BLE001 - a turn never escapes its log
        log.exception("chat turn %s failed", turn_log.request_id)
        with contextlib.suppress(Exception):
            await publish("error", {"message": f"unexpected failure: {exc}"})
            await publish("done", {"failed": True})
    finally:
        await turn_log.finish()
        if replica is not None and request.user_email is not None:
            await replica.close(turn_log.request_id)


async def _produce(turn_log: LocalTurn, request: TurnRequest, publish) -> None:
    runtime, query, typed = request.runtime, request.query, request.typed
    user_email, conversation_id = request.user_email, request.conversation_id

    loop = asyncio.get_running_loop()
    sink = trace.TraceSink(request_id=turn_log.request_id, loop=loop)

    await publish("start", {
        "query": query,
        "budget_s": STREAM_BUDGET_S,
        "conversation_id": conversation_id,
        "request_id": turn_log.request_id,
        # Only a signed-in turn can be resumed: the token is the request id
        # *and* the owner, and an anonymous turn has no owner to check.
        "resumable": user_email is not None,
    })

    # The sink is installed for the whole turn, not only the graph, so the
    # classifier's and the clarifier's model calls and the gate's own decision
    # reach the trace too. `run_query` installs its own observation for the
    # graph; this one covers everything around it.
    instruction, instruction_on = await instructions.get(user_email)
    observation = obs.RequestObservability(
        request_id=turn_log.request_id,
        trace=sink,
        user_email=user_email,
        conversation_id=conversation_id,
        instruction=instruction if instruction_on else "",
    )
    token = obs.install(observation)
    try:
        if request.regenerate is not None:
            await _regenerate(turn_log, request, request.regenerate, sink, publish, observation)
            return

        decision, prior, prior_query = await _resolve_turn(
            runtime, query, conversation_id, user_email
        )

        if decision is not None and decision.mode == "discuss":
            # The cheap path: answered from the evidence already on screen, with
            # no fan-out at all. Most follow-ups are this, which is what keeps a
            # conversation from costing five agent runs per turn.
            await _flush(sink, publish)
            await publish("turn", {"mode": "discuss", "method": decision.method,
                                   "reason": decision.reason})
            await _discuss(runtime, typed, prior, prior_query, conversation_id, user_email,
                           sink, observation, publish)
            return

        if decision is not None:
            await _flush(sink, publish)
            await publish("turn", {"mode": "analyse", "method": decision.method,
                                   "reason": decision.reason,
                                   "standalone_query": decision.standalone_query})
            query = decision.standalone_query or query

        if not request.skip_clarify:
            asked = await _maybe_clarify(runtime, query, conversation_id, user_email, typed=typed)
            await _flush(sink, publish)
            if asked is not None:
                await publish("clarify", asked)
                await publish("done", {"failed": False, "clarify": True,
                                       "conversation_id": conversation_id,
                                       "request_id": turn_log.request_id})
                return

        # `prior is None` is a conversation's first exchange — the only turn
        # worth naming it from. Naming it on every turn paid for a model call
        # whose result `set_title_if_unset` then threw away.
        await _analyse(turn_log, request, query, typed, sink, publish,
                       outer=observation, first_exchange=prior is None)
    finally:
        obs.reset(token)
        # The turn's own model calls — classification, clarification, a title,
        # a discussion — in one batch. `run_query` records the graph's calls
        # itself, under the same request id, so a turn is one request in the
        # ledger however its spend was split.
        await ledger.record(
            request_id=turn_log.request_id,
            user_email=user_email,
            conversation_id=conversation_id,
            calls=observation.usage.calls,
        )


async def _regenerate(turn_log, request, plan: RegeneratePlan, sink, publish,
                      observation: obs.RequestObservability) -> None:
    """A fresh answer to the latest question, kept beside the one it replaces."""
    await publish("turn", {"mode": plan.mode, "method": "regenerate",
                           "reason": "a fresh answer to the same question",
                           "regenerates": plan.target.id,
                           # The question the graph re-runs, as an ordinary analyse
                           # turn's frame names it. The page keys related news off
                           # this; without it a regenerate searched for "".
                           "standalone_query": plan.question.asked
                           if plan.mode == "analyse" else None})
    if plan.mode == "discuss":
        # The prompt cache would hand back the discussion being replaced.
        observation.bypass_cache_roles = frozenset({"chat"})
        await _discuss(request.runtime, plan.question.content, plan.prior, plan.prior_query,
                       request.conversation_id, request.user_email, sink, observation,
                       publish, regenerated_from=plan.target.id)
        return
    await _analyse(turn_log, request, plan.question.asked, plan.question.content, sink,
                   publish, outer=observation, first_exchange=False,
                   regenerated_from=plan.target.id)


async def _analyse(turn_log, request, query, typed, sink, publish, *,
                   outer: obs.RequestObservability, first_exchange: bool,
                   regenerated_from: int | None = None) -> None:
    """The graph path: `run_query`, with its trace pumped into the log live."""
    runtime, user_email = request.runtime, request.user_email
    conversation_id = request.conversation_id

    task = asyncio.create_task(
        run_query(
            runtime,
            query,
            user_email=user_email,
            overall_timeout_s=STREAM_BUDGET_S,
            trace_sink=sink,
            conversation_id=conversation_id,
            request_id=turn_log.request_id,
            # Regenerate re-runs the graph for real — the trace is real — but
            # asks the merge for new wording rather than the cached answer.
            bypass_cache_roles=frozenset({"merge"}) if regenerated_from else frozenset(),
        )
    )
    try:
        outcome: QueryOutcome = await _pump(
            task, sink, publish, request_id=turn_log.request_id, user_email=user_email
        )
    except OrchestrationError as exc:
        await publish("error", {"message": str(exc)})
        await publish("done", {"failed": True, "request_id": turn_log.request_id})
        return

    answer = outcome.response.model_dump(mode="json")

    # Persistence after the answer is assembled and before `done`, so a client
    # that reloads immediately finds the turn already there. Failures here are
    # logged, never surfaced: the answer has been produced, and losing its
    # transcript row is not worth turning a success into an error frame.
    ids: list[int] = []
    if conversation_id is not None and user_email is not None:
        ids = await _persist_turn(
            conversation_id, user_email, typed, answer, outcome,
            _usage(outer, outcome), sink, runtime.llm, mode="analyse",
            effective_query=query, name_it=first_exchange,
            regenerated_from=regenerated_from,
        )

    await publish("done", {
        "failed": False,
        "request_id": outcome.request_id,
        "conversation_id": conversation_id,
        # Everything the turn spent: the graph's calls and the turn's own
        # (classification, clarification, a title) — the footer's one number.
        "usage": _usage(outer, outcome),
        "dropped_events": sink.dropped,
        "query_history_id": outcome.history_id,
        # What actually ran, when it is not what was typed — the same value the
        # transcript stores as the user turn's `effective_query`.
        "effective_query": query if query != typed else None,
        "regenerated_from": regenerated_from,
        **_message_ids(ids, regenerated=regenerated_from is not None),
        "answer": answer,
    })


async def _discuss(runtime, follow_up, prior, prior_query, conversation_id, user_email,
                   sink, observation, publish, *, regenerated_from: int | None = None) -> None:
    """A follow-up answered from the previous turn — no graph, no Cypher.

    Still streamed rather than returned whole, so the client has one transport
    and one set of frame handlers regardless of which path a turn takes — and,
    since it shares the turn's sink, its model call and its grounding verdict
    appear in the trace like any other step.

    Pumped while it runs, exactly as the graph path is. Awaiting `discuss()` to
    completion and flushing afterwards — which is what this did until it was
    measured — delivered every grounded sentence in one burst just before
    `done`, so the sentence gate was doing its work for nobody.
    """
    task = asyncio.create_task(turn.discuss(follow_up, prior, prior_query, runtime.llm))
    result = await _pump(
        task, sink, publish, request_id=observation.request_id, user_email=user_email
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
            conversation_id, user_email, follow_up, answer, None, usage, sink,
            runtime.llm, mode="discuss", request_id=observation.request_id,
            name_it=False, regenerated_from=regenerated_from,
        )

    await publish("done", {
        "failed": False,
        "request_id": observation.request_id,
        "conversation_id": conversation_id,
        "usage": usage,
        "dropped_events": sink.dropped,
        # A discussion is not a new analysis and writes no history row.
        "query_history_id": None,
        "effective_query": None,
        "regenerated_from": regenerated_from,
        **_message_ids(ids, regenerated=regenerated_from is not None),
        "answer": answer,
    })


async def _pump(task: asyncio.Task, sink: trace.TraceSink, publish, *,
                request_id: str, user_email: str | None):
    """Publish the sink's events live while `task` runs, then return its result.

    One loop for both the graph path and the discuss path, so a sentence the
    gate releases reaches the wire when it is released rather than when the
    call that produced it returns. Between events it checks for a Stop pressed
    on a reader served by the other worker. A cancellation from outside — the
    Stop endpoint on this worker, or an anonymous reader leaving — is passed to
    the task, waited out, and re-raised for `_run` to record.
    """
    replica = mirror()
    try:
        while not task.done():
            try:
                event = await asyncio.wait_for(sink.queue.get(), timeout=POLL_INTERVAL_S)
            except TimeoutError:
                if (replica is not None and user_email is not None
                        and await replica.cancel_requested(request_id)):
                    task.cancel()
                continue
            await publish(event.kind, event.as_dict())

        await _flush(sink, publish)
        return task.result()
    except asyncio.CancelledError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        raise


async def _flush(sink: trace.TraceSink, publish) -> None:
    """Publish whatever the sink holds right now, without waiting.

    A yield point first: events emitted from a worker thread (the retriever runs
    its ONNX embed and rerank through `asyncio.to_thread`) are enqueued via
    `call_soon_threadsafe`, so their callbacks may still be pending when the
    graph task completes. Without this, the last few steps of a retrieval-heavy
    query would be missing from the live stream and present in the replay — the
    two telling different stories about the same request.
    """
    await asyncio.sleep(0)
    while True:
        try:
            event = sink.queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        await publish(event.kind, event.as_dict())


# --- deciding what kind of turn this is ---------------------------------------


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
) -> dict[str, Any] | None:
    """One question back, or nothing at all. Never raises.

    Stage A is free and silent on every question in `eval/questions.yaml`, so the
    common path costs one keyword pass and returns here immediately. Only when it
    fires does anything else happen — and only then is an LLM call made.

    Needs somewhere to remember the pending question, so an anonymous turn (no
    conversation) is never clarified: there would be nowhere to resume to.

    Returns the `clarify` frame's payload, or None to answer as asked.
    """
    if not settings.clarify_enabled() or conversation_id is None or user_email is None:
        return None

    trigger = clarify.clarification_needed(query)
    if trigger is None:
        return None

    clarification = await clarify.llm_clarify(trigger, runtime.llm)
    if clarification is None:  # the model vetoed a gate the syntax check opened
        trace.emit("clarify_gate", kind=trigger.kind, asked=False, method="llm-veto")
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

    # `clarify_gate`, not `clarify`: the latter is the prompt frame itself, and
    # a trace event sharing its SSE name would reach the client's prompt
    # handler. Invisible until the gate ran inside the turn's sink.
    trace.emit("clarify_gate", kind=trigger.kind, asked=True, method=clarification.method,
               question=clarification.question)
    return {"pending_id": pending_id, **clarification.as_payload()}


# --- persistence -------------------------------------------------------------


def _message_ids(ids: list[int], *, regenerated: bool = False) -> dict[str, int | None]:
    """The rows a finished turn was stored as, so the client can fold the live
    turn into its transcript without a reload — and rate it, save it or
    regenerate it straight away. Both None when the turn was not persisted.

    A regenerated turn stores only its answer: the question is the one already
    in the transcript, and storing it again would show it twice."""
    if regenerated:
        return {"user_message_id": None, "message_id": ids[0] if ids else None}
    if len(ids) >= 2:
        return {"user_message_id": ids[0], "message_id": ids[1]}
    return {"user_message_id": None, "message_id": ids[0] if ids else None}


def _usage(outer: obs.RequestObservability, outcome: QueryOutcome) -> dict[str, Any]:
    """The turn's spend as one footer: the graph's calls plus the turn's own."""
    return obs.RequestLLMUsage(calls=[*outer.usage.calls, *outcome.usage.calls]).as_summary()


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
    request_id: str | None = None,
    name_it: bool = True,
    regenerated_from: int | None = None,
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
    request_id = outcome.request_id if outcome else request_id
    ran = effective_query if effective_query and effective_query != question else None
    # A regenerated answer joins the transcript alone: its question is already
    # there. The version it replaces stays, linked by `regenerated_from`.
    rows = [] if regenerated_from is not None else [
        store.Message(role="user", content=question, effective_query=ran)
    ]
    try:
        ids = await store.append(
            conversation_id,
            user_email,
            [
                *rows,
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
                    regenerated_from=regenerated_from,
                ),
            ],
        )
    except store.ChatStoreUnavailableError as exc:
        log.warning("could not persist turn in conversation %s: %s", conversation_id, exc)
        return []

    if sink is not None and request_id:
        await store.save_trace(request_id, conversation_id, sink.history)

    # After the answer is delivered, and conditional in SQL, so a slow or absent
    # model costs a plainer name rather than a slower turn. Only on the first
    # exchange: the write was always once-only, the model call was not.
    if name_it:
        name = await titles.title_for(question, answer.get("answer", ""), llm)
        await store.set_title_if_unset(conversation_id, user_email, name)
    return list(ids or [])


__all__ = [
    "HEARTBEAT_INTERVAL_S",
    "RegeneratePlan",
    "POLL_INTERVAL_S",
    "REMOTE_IDLE_LIMIT_S",
    "STREAM_BUDGET_S",
    "TurnRequest",
    "cancel",
    "follow",
    "plan_regeneration",
    "start_turn",
]
