"""Past chats — deviation D13, extending SRS 3.5.2's query history.

SRS 3.5.2 gives a user their past queries and lets them bookmark one. A
conversation is the same idea with turns in it, so these routes sit beside
`/api/history` rather than replacing it: every turn that runs the graph still
writes its `query_history` row, and `ChatMessageItem.query_history_id` links the
two so the chat UI's save button calls the *existing* `/api/history/{id}/save`.

**Signed in, always.** `/api/query` answers anonymous callers by a documented
deferred-scope decision, and that is fine for a stateless question. A
conversation is stateful, higher-value, and addressed by a `BIGSERIAL` id that
anyone can guess — so every route here takes `require_user` and every store call
filters on `user_email` in the statement itself. Not found and not yours return
the same 404, the same posture as `history.set_saved`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from ceynex import settings
from ceynex.api.routes.auth import TokenPayload, require_user
from ceynex.api.schemas import (
    ChatMessageItem,
    ConversationCreateRequest,
    ConversationDetail,
    ConversationPatchRequest,
    ConversationSummary,
    TraceEventItem,
)
from ceynex.chat import store

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"], prefix="/api/chat")


def _require_enabled() -> None:
    if not settings.chat_enabled():
        raise HTTPException(status_code=404, detail="chat is not enabled on this deployment")


def _summary(conversation: store.Conversation) -> ConversationSummary:
    return ConversationSummary(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        pinned=conversation.pinned,
        archived=conversation.archived,
        message_count=conversation.message_count,
    )


def _message(message: store.Message) -> ChatMessageItem:
    return ChatMessageItem(
        seq=message.seq,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        mode=message.mode,
        request_id=message.request_id,
        confidence=message.confidence,
        confidence_band=message.confidence_band,
        degraded=message.degraded,
        agents_used=message.agents_used,
        route=message.route,
        sectors=message.sectors,
        unanswered=message.unanswered,
        evidence=message.evidence,
        forecast=message.forecast,
        graph=message.graph,
        elapsed_ms=message.elapsed_ms,
        usage=message.usage,
        query_history_id=message.query_history_id,
    )


@router.post("/conversations", response_model=ConversationSummary, status_code=201)
async def create_conversation(
    request: ConversationCreateRequest,
    user: TokenPayload = Depends(require_user),  # noqa: B008 - FastAPI's dependency idiom
) -> ConversationSummary:
    _require_enabled()
    try:
        conversation_id = await store.create(user.email, request.title)
        conversations = await store.list_for_user(user.email, limit=1, include_archived=True)
    except store.ChatStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    for conversation in conversations:
        if conversation.id == conversation_id:
            return _summary(conversation)
    raise HTTPException(status_code=503, detail="conversation was created but could not be read back")


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    include_archived: bool = Query(default=False),
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> list[ConversationSummary]:
    """Pinned first, then most recently active.

    503 rather than an empty list on a database outage — a user shown an empty
    sidebar would reasonably conclude their conversations were gone.
    """
    _require_enabled()
    try:
        conversations = await store.list_for_user(
            user.email, limit=limit, include_archived=include_archived
        )
    except store.ChatStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [_summary(conversation) for conversation in conversations]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> ConversationDetail:
    _require_enabled()
    try:
        messages = await store.messages(conversation_id, user.email)
        conversations = await store.list_for_user(user.email, limit=200, include_archived=True)
    except store.ChatStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if messages is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    summary = next((c for c in conversations if c.id == conversation_id), None)
    if summary is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    return ConversationDetail(
        conversation=_summary(summary),
        messages=[_message(message) for message in messages],
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
async def patch_conversation(
    conversation_id: int,
    request: ConversationPatchRequest,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> ConversationSummary:
    """Rename, pin or archive. Only the fields present in the body change."""
    _require_enabled()
    if request.title is None and request.pinned is None and request.archived is None:
        raise HTTPException(status_code=422, detail="nothing to change")

    try:
        updated = await store.update(
            conversation_id,
            user.email,
            title=request.title,
            pinned=request.pinned,
            archived=request.archived,
        )
        conversations = await store.list_for_user(user.email, limit=200, include_archived=True)
    except store.ChatStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not updated:
        raise HTTPException(status_code=404, detail="conversation not found")

    summary = next((c for c in conversations if c.id == conversation_id), None)
    if summary is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return _summary(summary)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: int,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> None:
    """A real delete, cascading to its messages.

    SRS 3.10's data-protection principles say collect only what is needed to
    operate the feature. A "deleted" conversation still sitting in the table is
    still collected, so this is a `DELETE`, not an archive flag — archiving is a
    separate, explicit action on PATCH.
    """
    _require_enabled()
    try:
        deleted = await store.delete(conversation_id, user.email)
    except store.ChatStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="conversation not found")


@router.get("/conversations/{conversation_id}/trace/{request_id}",
            response_model=list[TraceEventItem])
async def get_trace(
    conversation_id: int,
    request_id: str,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> list[TraceEventItem]:
    """Replay one turn's reasoning trace, in the order it happened.

    Ownership is checked on the *conversation* before the trace is read: a
    `request_id` is a UUID rather than a serial, but that is obscurity, and this
    endpoint would otherwise let anyone holding one read the Cypher and token
    counts of somebody else's analysis.
    """
    _require_enabled()
    try:
        if not await store.owns(conversation_id, user.email):
            raise HTTPException(status_code=404, detail="conversation not found")
        events = await store.trace_for(request_id)
    except store.ChatStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return [
        TraceEventItem(
            seq=event["seq"],
            kind=event["kind"],
            node=event.get("node"),
            ts=event["ts"],
            payload={
                key: value
                for key, value in event.items()
                if key not in ("seq", "kind", "node", "ts")
            },
        )
        for event in events
    ]
