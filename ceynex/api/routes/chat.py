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
choice, and every failure path is written for it.

**The routes here only start and follow turns.** The turn itself runs in
`ceynex/api/turn_runner.py`, as a task that outlives the connection watching it,
writing every frame into a numbered log (`api/turn_log.py`). That is what makes
a dropped stream resumable — `GET /api/chat/turns/{request_id}/events` picks up
after the last frame the reader saw — and why Stop is its own endpoint rather
than "close the socket": a browser's abort and a network drop look identical
from the server.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from ceynex import settings
from ceynex.api import rate_limit, turn_log, turn_runner
from ceynex.api.deps import Runtime, get_runtime
from ceynex.api.routes.auth import TokenPayload, get_optional_user, require_user
from ceynex.api.schemas import ChatStreamRequest, ClarifyAnswerRequest
from ceynex.chat import clarify, store

# Re-exported: the budget and heartbeat are properties of the transport this
# module serves, and the docstring above names them.
STREAM_BUDGET_S = turn_runner.STREAM_BUDGET_S
HEARTBEAT_INTERVAL_S = turn_runner.HEARTBEAT_INTERVAL_S

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


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


_SSE_HEADERS = {
    # nginx honours this and stops buffering — the difference between a live
    # trace and one whole response arriving at the end.
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


def _follow_response(
    request_id: str, http_request: Request, *, after: int = 0, anonymous: bool = False
) -> StreamingResponse:
    return StreamingResponse(
        turn_runner.follow(request_id, after, http_request, cancel_on_disconnect=anonymous),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


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

    started = turn_runner.start_turn(
        turn_runner.TurnRequest(
            runtime=runtime,
            query=query,
            typed=query,
            user_email=user.email if user else None,
            conversation_id=conversation_id,
        )
    )
    return _follow_response(started.request_id, http_request, anonymous=user is None)


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

    started = turn_runner.start_turn(
        turn_runner.TurnRequest(
            runtime=runtime,
            query=query,
            typed=typed,
            user_email=user.email,
            conversation_id=int(pending["conversation_id"]),
            skip_clarify=True,
        )
    )
    return _follow_response(started.request_id, http_request)


@router.post(
    "/api/chat/messages/{message_id}/regenerate",
    dependencies=[Depends(enforce_chat_rate_limit)],
)
async def regenerate(
    message_id: int,
    http_request: Request,
    runtime: Runtime = Depends(get_runtime),  # noqa: B008 - FastAPI's dependency idiom
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> StreamingResponse:
    """A fresh answer to the latest question, streamed like any other turn.

    **Rate-limited like a turn**, because an analysis regenerate runs the whole
    fan-out again — the trace is real, not replayed — and only the merge's
    wording is asked for anew (its prompt cache is skipped for this request).

    Only the conversation's latest answer, and a 409 otherwise: every later turn
    was classified and grounded against an answer, and replacing one further
    back would fork the conversation under the reader's feet. The answer being
    replaced is kept; the new one is stored beside it with `regenerated_from`.
    """
    if not settings.chat_enabled():
        raise HTTPException(status_code=404, detail="chat is not enabled on this deployment")

    try:
        conversation_id = await store.conversation_of(message_id, user.email)
        transcript = (
            await store.messages(conversation_id, user.email)
            if conversation_id is not None
            else None
        )
    except store.ChatStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if conversation_id is None or transcript is None:
        raise HTTPException(status_code=404, detail="message not found")

    plan = turn_runner.plan_regeneration(transcript, message_id)
    if plan is None:
        raise HTTPException(status_code=409, detail="only the latest answer can be regenerated")

    started = turn_runner.start_turn(
        turn_runner.TurnRequest(
            runtime=runtime,
            query=plan.question.asked,
            typed=plan.question.content,
            user_email=user.email,
            conversation_id=conversation_id,
            skip_clarify=True,
            regenerate=plan,
        )
    )
    return _follow_response(started.request_id, http_request)


async def _owned_turn(request_id: str, user: TokenPayload) -> str:
    """Where this caller's turn lives — "local" or "remote" — or a 404.

    The request id alone is not the token; the owner is checked too, and "no
    such turn" and "not yours" are the same 404, as everywhere else in chat.
    """
    local = turn_log.registry().get(request_id)
    if local is not None:
        if local.owner != user.email:
            raise HTTPException(status_code=404, detail="turn not found")
        return "local"
    replica = turn_log.mirror()
    if replica is not None and await replica.owner(request_id) == user.email:
        return "remote"
    raise HTTPException(status_code=404, detail="turn not found")


@router.get("/api/chat/turns/{request_id}/events")
async def resume_turn(
    request_id: str,
    http_request: Request,
    after: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None),  # noqa: B008
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> StreamingResponse:
    """Pick a turn back up after the last frame the reader saw.

    `after` or the standard `Last-Event-ID` header, whichever is larger — a
    reader that lost its connection reports the last `id:` it received, and
    gets every frame after it exactly once, then the live remainder.

    Not rate-limited: it runs nothing. Like reading your own history, it only
    reads back what was already produced for you. A 404 means the turn is not
    running here, not mirrored, not yours, or finished long enough ago to have
    expired — in every case the transcript is where its answer now lives.
    """
    if not settings.chat_enabled():
        raise HTTPException(status_code=404, detail="chat is not enabled on this deployment")
    await _owned_turn(request_id, user)
    header_seq = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
    return _follow_response(request_id, http_request, after=max(after, header_seq))


@router.post("/api/chat/turns/{request_id}/cancel", status_code=202)
async def cancel_turn(
    request_id: str,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> dict[str, Any]:
    """Stop a turn — the Stop button, made explicit.

    A browser that aborts its fetch looks, from here, exactly like one whose
    network dropped, and since a signed-in turn deliberately survives a drop,
    Stop has to say what it means. A cancelled turn persists nothing and ends
    with `done` and `cancelled: true` for any reader still following it.

    Served by either worker: a turn running on this one is cancelled directly,
    one running on the other is flagged through the mirror, which the running
    turn checks between frames.
    """
    where = await _owned_turn(request_id, user)
    if where == "local":
        local = turn_log.registry().get(request_id)
        cancelled = await turn_runner.cancel(local) if local is not None else False
    else:
        replica = turn_log.mirror()
        cancelled = await replica.request_cancel(request_id) if replica is not None else False
    return {"request_id": request_id, "cancelled": cancelled, "where": where}
