"""Conversations, turns and their replayed traces — deviation D13.

Additive to the frozen `ceynex-contracts` schema, for the reason
`api/history.py` sets out for `query_history`: nobody else's code reads or writes
these, so they have no business behind that repo's three-way-approval gate.

**`query_history` is not replaced.** Every turn that runs the graph still writes
its `query_history` row exactly as it always did, so `GET /api/history`, the
`saved` star and the frontend's History panel keep working untouched.
`chat_message.query_history_id` cross-links the two, which means a chat UI's save
button calls the *existing* `/api/history/{id}/save` rather than a parallel one.

**Everything here is scoped to a user, and 404s rather than 403s.** A
`BIGSERIAL` conversation id is trivially guessable, so every read and write
filters on `user_email` in the statement itself — never a separate ownership
check that a later refactor could drop. Not found and not yours are deliberately
indistinguishable, the same posture as `history.set_saved`.

**Every call goes through `asyncio.to_thread`.** `history.py` calls
`psycopg.connect()` inline, which is fine for one write at the end of a bounded
request. It is not fine here: these run while an SSE connection is expected to be
emitting heartbeats, and with `uvicorn --workers 2` a blocking connect stalls
every other request on the process. A named departure from that template.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

#: Long enough to be useful in a sidebar, short enough not to wrap.
MAX_TITLE = 80

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_conversation (
    id BIGSERIAL PRIMARY KEY,
    user_email TEXT NOT NULL,
    title TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    pinned BOOLEAN NOT NULL DEFAULT false,
    archived BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS chat_conversation_user_idx
    ON chat_conversation (user_email, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_message (
    id BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES chat_conversation(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    mode TEXT,
    request_id TEXT,
    confidence DOUBLE PRECISION,
    confidence_band TEXT,
    degraded BOOLEAN,
    agents_used JSONB,
    route JSONB,
    sectors JSONB,
    unanswered JSONB,
    evidence JSONB,
    forecast JSONB,
    graph JSONB,
    elapsed_ms DOUBLE PRECISION,
    usage JSONB,
    query_history_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_message_conv_idx ON chat_message (conversation_id, seq);

CREATE TABLE IF NOT EXISTS chat_trace_event (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    conversation_id BIGINT REFERENCES chat_conversation(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    node TEXT,
    payload JSONB NOT NULL,
    ts DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_trace_event_request_idx ON chat_trace_event (request_id, seq);
CREATE INDEX IF NOT EXISTS chat_trace_event_conv_idx ON chat_trace_event (conversation_id);

-- The cascade above only applies to a table created after it was added, and
-- `CREATE TABLE IF NOT EXISTS` is a no-op against one that already exists —
-- the same gap `history.py` covers with `ADD COLUMN IF NOT EXISTS`. Postgres
-- has no `ADD CONSTRAINT IF NOT EXISTS`, so this is the idempotent form.
--
-- It is not cosmetic. Without it, deleting a conversation left every Cypher
-- query, token count and timing from its analyses in the database for good,
-- which is precisely what the delete endpoint claims not to do (SRS 3.10).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chat_trace_event_conversation_fk'
    ) THEN
        DELETE FROM chat_trace_event e
        WHERE e.conversation_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM chat_conversation c WHERE c.id = e.conversation_id);

        ALTER TABLE chat_trace_event
            ADD CONSTRAINT chat_trace_event_conversation_fk
            FOREIGN KEY (conversation_id) REFERENCES chat_conversation(id) ON DELETE CASCADE;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS chat_pending_clarification (
    id BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES chat_conversation(id) ON DELETE CASCADE,
    user_email TEXT NOT NULL,
    original_query TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    resolved BOOLEAN NOT NULL DEFAULT false
);

-- Only ever read by "is this pending row still open", so the index carries the
-- filter rather than just the key.
CREATE INDEX IF NOT EXISTS chat_pending_open_idx
    ON chat_pending_clarification (conversation_id, resolved, expires_at DESC);

-- Answer feedback. On its own table rather than columns on chat_message,
-- because it is written by a different action at a different time and read for a
-- different purpose: turning real usage into eval data (eval/questions.yaml has
-- no growth path otherwise). One row per message per user, so changing your mind
-- replaces rather than accumulates.
CREATE TABLE IF NOT EXISTS chat_feedback (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT NOT NULL REFERENCES chat_message(id) ON DELETE CASCADE,
    user_email TEXT NOT NULL,
    rating SMALLINT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (message_id, user_email)
);

-- A shareable read-only link to a finished conversation (§5). Nullable and
-- unique: NULL means never shared, which is the default and the safe state.
ALTER TABLE chat_conversation ADD COLUMN IF NOT EXISTS share_token TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS chat_conversation_share_idx
    ON chat_conversation (share_token) WHERE share_token IS NOT NULL;

-- So a reopened conversation shows what the live turn showed: the working behind
-- the confidence score, whether a discussion's prose survived grounding, and —
-- on a user turn — the query actually run when it differs from what was typed.
ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS confidence_breakdown JSONB;
ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS grounded BOOLEAN;
ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS effective_query TEXT;

-- Regenerate keeps every version. SET NULL rather than CASCADE: no path deletes
-- a single message today, but if one ever does, removing an old version must
-- not take the newer one with it. Deleting the conversation still removes all
-- of them through `conversation_id`.
ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS regenerated_from BIGINT
    REFERENCES chat_message(id) ON DELETE SET NULL;
"""


def ensure_table() -> None:
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()
    except psycopg.Error as exc:
        log.warning("chat tables not ensured (postgres unreachable?): %s", exc)


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str | None
    created_at: str
    updated_at: str
    pinned: bool
    archived: bool
    message_count: int = 0


@dataclass
class Message:
    role: str
    content: str
    seq: int = 0
    #: The database row id, present on a message read back from the store and
    #: None on one being written. Feedback attaches to this, not to `seq`.
    id: int | None = None
    mode: str | None = None
    request_id: str | None = None
    confidence: float | None = None
    confidence_band: str | None = None
    degraded: bool | None = None
    agents_used: list[str] = field(default_factory=list)
    route: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    unanswered: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    forecast: list[dict[str, Any]] | None = None
    graph: dict[str, Any] | None = None
    elapsed_ms: float | None = None
    usage: dict[str, Any] | None = None
    query_history_id: int | None = None
    created_at: str | None = None
    #: SRS 3.1.4's working behind the score. Stored so "why this confidence?"
    #: survives a reload — without it the panel beside a reopened answer was
    #: silently missing, which reads as "there was no working".
    confidence_breakdown: dict[str, Any] | None = None
    #: A `discuss` turn only: False when its prose was withheld for stating a
    #: figure the analysis never produced. None on everything else.
    grounded: bool | None = None
    #: A *user* turn only: the query the system actually ran, when it differs
    #: from `content` — a follow-up rewritten as a standalone question, or a
    #: clarified question composed with the reader's choice. `content` stays
    #: what the reader typed, so a reopened transcript never puts words in their
    #: mouth. None means the two are the same.
    effective_query: str | None = None
    #: An assistant turn produced by Regenerate: the id of the version it
    #: replaces. Older versions are kept, never overwritten.
    regenerated_from: int | None = None
    #: Read-only, joined from `query_history.saved` through `query_history_id`,
    #: so the chat's save star shows the state the History panel shows.
    saved: bool = False

    @property
    def asked(self) -> str:
        """The question as the system ran it: the rewrite if there was one."""
        return self.effective_query or self.content


class ChatStoreUnavailableError(RuntimeError):
    """Postgres could not be reached. Routes turn this into a 503."""


def _connect():
    return psycopg.connect(postgres_dsn(), connect_timeout=3)


# --- conversations --------------------------------------------------------


def _create_sync(user_email: str, title: str | None) -> int:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_conversation (user_email, title) VALUES (%s, %s) RETURNING id",
            (user_email, (title or "")[:MAX_TITLE] or None),
        )
        row = cur.fetchone()
        conn.commit()
    return int(row[0])


async def create(user_email: str, title: str | None = None) -> int:
    try:
        return await asyncio.to_thread(_create_sync, user_email, title)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not create conversation: {exc}") from exc


def _list_sync(user_email: str, limit: int, include_archived: bool) -> list[Conversation]:
    clause = "" if include_archived else " AND NOT c.archived"
    with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT c.id, c.title, c.created_at, c.updated_at, c.pinned, c.archived,
                   count(m.id) FILTER (WHERE m.role = 'user') AS message_count
            FROM chat_conversation c
            LEFT JOIN chat_message m ON m.conversation_id = c.id
            WHERE c.user_email = %s{clause}
            GROUP BY c.id
            ORDER BY c.pinned DESC, c.updated_at DESC
            LIMIT %s
            """,  # noqa: S608 - `clause` is one of two hardcoded literals, no input
            (user_email, limit),
        )
        rows = cur.fetchall()
    return [
        Conversation(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"].isoformat(),
            updated_at=row["updated_at"].isoformat(),
            pinned=row["pinned"],
            archived=row["archived"],
            message_count=row["message_count"],
        )
        for row in rows
    ]


async def list_for_user(
    user_email: str, *, limit: int = 50, include_archived: bool = False
) -> list[Conversation]:
    """Pinned first, then most recently active.

    Raises rather than returning `[]` on a database outage: a user shown an empty
    sidebar would reasonably conclude their conversations were gone. Same
    distinction `history.list_for_user` draws between recording (opportunistic)
    and listing (not).
    """
    try:
        return await asyncio.to_thread(_list_sync, user_email, limit, include_archived)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not list conversations: {exc}") from exc


def _update_sync(conversation_id: int, user_email: str, fields: dict[str, Any]) -> bool:
    assignments = ", ".join(f"{name} = %s" for name in fields)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE chat_conversation SET {assignments}, updated_at = now() "  # noqa: S608
            "WHERE id = %s AND user_email = %s",
            (*fields.values(), conversation_id, user_email),
        )
        updated = cur.rowcount > 0
        conn.commit()
    return updated


async def update(
    conversation_id: int,
    user_email: str,
    *,
    title: str | None = None,
    pinned: bool | None = None,
    archived: bool | None = None,
) -> bool:
    """Rename, pin or archive. False when it does not exist *or* is not yours.

    Column names come from this function's own keyword arguments, never from the
    caller — the f-string above interpolates a fixed set of identifiers, and the
    values stay parameterised.
    """
    fields: dict[str, Any] = {}
    if title is not None:
        fields["title"] = title[:MAX_TITLE]
    if pinned is not None:
        fields["pinned"] = pinned
    if archived is not None:
        fields["archived"] = archived
    if not fields:
        return False

    try:
        return await asyncio.to_thread(_update_sync, conversation_id, user_email, fields)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not update conversation: {exc}") from exc


def _set_title_if_unset_sync(conversation_id: int, user_email: str, title: str) -> bool:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE chat_conversation SET title = %s "
            "WHERE id = %s AND user_email = %s AND title IS NULL",
            (title[:MAX_TITLE], conversation_id, user_email),
        )
        updated = cur.rowcount > 0
        conn.commit()
    return updated


async def set_title_if_unset(conversation_id: int, user_email: str, title: str) -> bool:
    """Name a conversation, but never over a name the user chose.

    Conditional in SQL rather than read-then-write: between the two the user may
    have renamed it from another tab, and silently overwriting that reads as the
    application losing their data.
    """
    try:
        return await asyncio.to_thread(
            _set_title_if_unset_sync, conversation_id, user_email, title
        )
    except psycopg.Error as exc:
        log.warning("could not title conversation %s: %s", conversation_id, exc)
        return False


def _delete_sync(conversation_id: int, user_email: str) -> bool:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chat_conversation WHERE id = %s AND user_email = %s",
            (conversation_id, user_email),
        )
        deleted = cur.rowcount > 0
        conn.commit()
    return deleted


async def delete(conversation_id: int, user_email: str) -> bool:
    """A real delete, not a flag.

    SRS 3.10's data-protection principles say collect only what is needed to
    operate the feature; a "deleted" conversation that is still in the table is
    still collected. Messages go with it via `ON DELETE CASCADE`.
    """
    try:
        return await asyncio.to_thread(_delete_sync, conversation_id, user_email)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not delete conversation: {exc}") from exc


def _owns_sync(conversation_id: int, user_email: str) -> bool:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM chat_conversation WHERE id = %s AND user_email = %s",
            (conversation_id, user_email),
        )
        return cur.fetchone() is not None


async def owns(conversation_id: int, user_email: str) -> bool:
    """Whether this user may write to this conversation. Checked before every turn."""
    try:
        return await asyncio.to_thread(_owns_sync, conversation_id, user_email)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not verify conversation: {exc}") from exc


def _conversation_of_sync(message_id: int, user_email: str) -> int | None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT m.conversation_id FROM chat_message m "
            "JOIN chat_conversation c ON c.id = m.conversation_id "
            "WHERE m.id = %s AND c.user_email = %s",
            (message_id, user_email),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None


async def conversation_of(message_id: int, user_email: str) -> int | None:
    """The conversation a message belongs to — if it belongs to this user.

    Scoped in the statement, like every read here: a message id is a guessable
    serial, and "no such message" and "not yours" are the same None.
    """
    try:
        return await asyncio.to_thread(_conversation_of_sync, message_id, user_email)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not find message: {exc}") from exc


# --- messages -------------------------------------------------------------


_MESSAGE_COLUMNS = (
    # `id` travels so a reader can rate an answer (§5). `seq` orders a
    # transcript; it does not identify a row across conversations, and the
    # feedback endpoint needs something that does.
    "m.id, m.seq, m.role, m.content, m.mode, m.request_id, m.confidence, "
    "m.confidence_band, m.degraded, m.agents_used, m.route, m.sectors, m.unanswered, "
    "m.evidence, m.forecast, m.graph, m.elapsed_ms, m.usage, m.query_history_id, "
    "m.created_at, m.confidence_breakdown, m.grounded, m.effective_query, "
    "m.regenerated_from, coalesce(qh.saved, false) AS saved"
)


def _append_sync(conversation_id: int, user_email: str, messages: list[Message]) -> list[int]:
    ids: list[int] = []
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM chat_conversation WHERE id = %s AND user_email = %s",
            (conversation_id, user_email),
        )
        if cur.fetchone() is None:
            return []

        cur.execute(
            "SELECT coalesce(max(seq), 0) FROM chat_message WHERE conversation_id = %s",
            (conversation_id,),
        )
        seq = int(cur.fetchone()[0])

        for message in messages:
            seq += 1
            cur.execute(
                """
                INSERT INTO chat_message (
                    conversation_id, seq, role, content, mode, request_id,
                    confidence, confidence_band, degraded, agents_used, route, sectors,
                    unanswered, evidence, forecast, graph, elapsed_ms, usage, query_history_id,
                    confidence_breakdown, grounded, effective_query, regenerated_from
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                          %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    conversation_id, seq, message.role, message.content, message.mode,
                    message.request_id, message.confidence, message.confidence_band,
                    message.degraded, Jsonb(message.agents_used), Jsonb(message.route),
                    Jsonb(message.sectors), Jsonb(message.unanswered), Jsonb(message.evidence),
                    Jsonb(message.forecast) if message.forecast is not None else None,
                    Jsonb(message.graph) if message.graph is not None else None,
                    message.elapsed_ms,
                    Jsonb(message.usage) if message.usage is not None else None,
                    message.query_history_id,
                    Jsonb(message.confidence_breakdown)
                    if message.confidence_breakdown is not None
                    else None,
                    message.grounded,
                    message.effective_query,
                    message.regenerated_from,
                ),
            )
            ids.append(int(cur.fetchone()[0]))

        cur.execute(
            "UPDATE chat_conversation SET updated_at = now() WHERE id = %s", (conversation_id,)
        )
        conn.commit()
    return ids


async def append(conversation_id: int, user_email: str, messages: list[Message]) -> list[int]:
    """Write a turn. Returns the new message ids, or `[]` if the conversation
    is not this user's.

    Both halves of a turn go in one transaction and one `seq` sequence, so a
    crash between them cannot leave a question with no answer beside it.
    """
    try:
        return await asyncio.to_thread(_append_sync, conversation_id, user_email, messages)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not append to conversation: {exc}") from exc


def _messages_sync(conversation_id: int, user_email: str) -> list[Message] | None:
    with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT 1 FROM chat_conversation WHERE id = %s AND user_email = %s",
            (conversation_id, user_email),
        )
        if cur.fetchone() is None:
            return None

        # The join is the documented cross-link (see the module docstring), read
        # so the save star in chat and the one in the History panel can never
        # disagree about the same row.
        cur.execute(
            f"SELECT {_MESSAGE_COLUMNS} FROM chat_message m "  # noqa: S608 - fixed column list
            "LEFT JOIN query_history qh ON qh.id = m.query_history_id "
            "WHERE m.conversation_id = %s ORDER BY m.seq",
            (conversation_id,),
        )
        rows = cur.fetchall()

    return [
        Message(
            id=row.get("id"),
            seq=row["seq"],
            role=row["role"],
            content=row["content"],
            mode=row["mode"],
            request_id=row["request_id"],
            confidence=row["confidence"],
            confidence_band=row["confidence_band"],
            degraded=row["degraded"],
            agents_used=row["agents_used"] or [],
            route=row["route"] or [],
            sectors=row["sectors"] or [],
            unanswered=row["unanswered"] or [],
            evidence=row["evidence"] or [],
            forecast=row["forecast"],
            graph=row["graph"],
            elapsed_ms=row["elapsed_ms"],
            usage=row["usage"],
            query_history_id=row["query_history_id"],
            created_at=row["created_at"].isoformat(),
            confidence_breakdown=row.get("confidence_breakdown"),
            grounded=row.get("grounded"),
            effective_query=row.get("effective_query"),
            regenerated_from=row.get("regenerated_from"),
            saved=bool(row.get("saved")),
        )
        for row in rows
    ]


async def messages(conversation_id: int, user_email: str) -> list[Message] | None:
    """The whole transcript, or None when it is not this user's conversation."""
    try:
        return await asyncio.to_thread(_messages_sync, conversation_id, user_email)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not read conversation: {exc}") from exc


async def last_answer(conversation_id: int, user_email: str) -> Message | None:
    """The most recent assistant turn — what a follow-up is asking *about*.

    A `discuss` follow-up is answered from this message's evidence rather than by
    re-running the graph, so this is the single read that makes the cheap path
    possible.
    """
    transcript = await messages(conversation_id, user_email)
    if not transcript:
        return None
    # Not gated on evidence: an analysis that declined is still something a
    # follow-up can ask about, and gating on it makes "what do you mean?"
    # re-run the fan-out for the same decline. See `api/turn_runner.py::_resolve_turn`.
    for message in reversed(transcript):
        if message.role == "assistant":
            return message
    return None


# --- the replayed trace ---------------------------------------------------


def _save_trace_sync(request_id: str, conversation_id: int | None, events: list[Any]) -> None:
    if not events:
        return
    rows = [
        (request_id, conversation_id, event.seq, event.kind, event.node,
         Jsonb(event.payload), event.ts)
        for event in events
    ]
    with _connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chat_trace_event
                (request_id, conversation_id, seq, kind, node, payload, ts)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        conn.commit()


async def save_trace(request_id: str, conversation_id: int | None, events: list[Any]) -> None:
    """Persist a finished trace in one batch, so reopening a chat replays it.

    One `executemany` at the end of the turn rather than a write per event: the
    live stream reads from an in-memory queue precisely so the database is never
    in the hot path, and a row-per-event would put it right back there.

    Best-effort. Losing a replay log is not worth failing an answer that has
    already been delivered.
    """
    try:
        await asyncio.to_thread(_save_trace_sync, request_id, conversation_id, events)
    except psycopg.Error as exc:
        log.warning("failed to persist %d trace events for %s: %s", len(events), request_id, exc)


def _trace_sync(request_id: str) -> list[dict[str, Any]]:
    with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT seq, kind, node, payload, ts FROM chat_trace_event "
            "WHERE request_id = %s ORDER BY seq",
            (request_id,),
        )
        rows = cur.fetchall()
    return [
        {"seq": row["seq"], "kind": row["kind"], "node": row["node"], "ts": row["ts"],
         **(row["payload"] or {})}
        for row in rows
    ]


async def trace_for(request_id: str) -> list[dict[str, Any]]:
    """The stored trace for one turn, in the order it happened."""
    try:
        return await asyncio.to_thread(_trace_sync, request_id)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not read trace: {exc}") from exc



# --- pending clarifications (D13) --------------------------------------------

#: A clarifying question the reader never answered is stale within the hour. It
#: is a row rather than process memory precisely so it survives a page reload and
#: is visible to both uvicorn workers — an in-memory pending state would resolve
#: on one worker and still be open on the other.
PENDING_TTL_MINUTES = 60


def _open_clarification_sync(conversation_id: int, user_email: str) -> dict[str, Any] | None:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor(
        row_factory=dict_row
    ) as cur:
        cur.execute(
            """
            SELECT id, original_query, payload
            FROM chat_pending_clarification
            WHERE conversation_id = %s AND user_email = %s
              AND NOT resolved AND expires_at > now()
            ORDER BY id DESC
            LIMIT 1
            """,
            (conversation_id, user_email),
        )
        return cur.fetchone()


def _record_clarification_sync(
    conversation_id: int, user_email: str, original_query: str, payload: dict[str, Any]
) -> int:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_pending_clarification
                (conversation_id, user_email, original_query, payload, expires_at)
            VALUES (%s, %s, %s, %s, now() + make_interval(mins => %s))
            RETURNING id
            """,
            (conversation_id, user_email, original_query, Jsonb(payload), PENDING_TTL_MINUTES),
        )
        row = cur.fetchone()
        conn.commit()
    if row is None:
        raise ChatStoreUnavailableError("could not record the pending clarification")
    return int(row[0])


def _resolve_clarification_sync(pending_id: int, user_email: str) -> dict[str, Any] | None:
    """Claim the row and return it, in one statement.

    `RETURNING` on a conditional `UPDATE` is what makes the one-round cap hold
    under two workers: whichever request claims the row first gets the query
    back, and the loser gets `None` rather than a second run of the same turn.
    """
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor(
        row_factory=dict_row
    ) as cur:
        cur.execute(
            """
            UPDATE chat_pending_clarification
            SET resolved = true
            WHERE id = %s AND user_email = %s AND NOT resolved AND expires_at > now()
            RETURNING id, conversation_id, original_query, payload
            """,
            (pending_id, user_email),
        )
        row = cur.fetchone()
        conn.commit()
        return row


async def open_clarification(conversation_id: int, user_email: str) -> dict[str, Any] | None:
    """The unanswered question for this conversation, if there is one."""
    try:
        return await asyncio.to_thread(_open_clarification_sync, conversation_id, user_email)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not read the pending clarification: {exc}") from exc


async def record_clarification(
    conversation_id: int, user_email: str, original_query: str, payload: dict[str, Any]
) -> int:
    try:
        return await asyncio.to_thread(
            _record_clarification_sync, conversation_id, user_email, original_query, payload
        )
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not record the clarification: {exc}") from exc


async def resolve_clarification(pending_id: int, user_email: str) -> dict[str, Any] | None:
    try:
        return await asyncio.to_thread(_resolve_clarification_sync, pending_id, user_email)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not resolve the clarification: {exc}") from exc



# --- answer feedback and sharing (§5) ----------------------------------------


def _feedback_sync(message_id: int, user_email: str, rating: int, reason: str) -> bool:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        # Scoped through the join, not by trusting message_id: a bare insert
        # would let anyone rate any message by guessing a BIGSERIAL.
        cur.execute(
            """
            INSERT INTO chat_feedback (message_id, user_email, rating, reason)
            SELECT m.id, %s, %s, %s
            FROM chat_message m
            JOIN chat_conversation c ON c.id = m.conversation_id
            WHERE m.id = %s AND c.user_email = %s
            ON CONFLICT (message_id, user_email) DO UPDATE
            SET rating = EXCLUDED.rating, reason = EXCLUDED.reason, created_at = now()
            RETURNING id
            """,
            (user_email, rating, reason[:2000], message_id, user_email),
        )
        row = cur.fetchone()
        conn.commit()
    return row is not None


def _share_sync(conversation_id: int, user_email: str, token: str | None) -> str | None:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE chat_conversation SET share_token = %s WHERE id = %s AND user_email = %s "
            "RETURNING share_token",
            (token, conversation_id, user_email),
        )
        row = cur.fetchone()
        conn.commit()
    return row[0] if row else None


def _shared_sync(token: str) -> dict[str, Any] | None:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor(
        row_factory=dict_row
    ) as cur:
        cur.execute(
            "SELECT id, title, created_at FROM chat_conversation WHERE share_token = %s",
            (token,),
        )
        conversation = cur.fetchone()
        if conversation is None:
            return None
        # `user_email` is deliberately not selected. A shared link shows what the
        # system did, not who asked.
        cur.execute(
            """
            SELECT id, seq, role, content, mode, request_id, confidence, confidence_band,
                   degraded, agents_used, route, sectors, unanswered, evidence,
                   forecast, graph, elapsed_ms, confidence_breakdown, grounded,
                   effective_query, regenerated_from
            FROM chat_message WHERE conversation_id = %s ORDER BY seq
            """,
            (conversation["id"],),
        )
        conversation["messages"] = _latest_versions(cur.fetchall())
        return conversation


def _latest_versions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each question's latest answer, for a reader who cannot switch versions.

    A regenerated answer is kept beside the one it replaced (the owner can page
    between them), but a shared transcript that showed both would read as two
    answers to one question. So a replaced answer is dropped, and its
    replacement says it is one — hiding the fact of a regenerate would be its
    own small misstatement of what the system did. Row ids stay out of it.
    """
    replaced = {row["regenerated_from"] for row in rows if row.get("regenerated_from")}
    latest = []
    for row in rows:
        if row["id"] in replaced:
            continue
        shown = {key: value for key, value in row.items() if key not in ("id", "regenerated_from")}
        if row.get("regenerated_from"):
            shown["regenerated"] = True
        latest.append(shown)
    return latest


async def record_feedback(message_id: int, user_email: str, rating: int, reason: str = "") -> bool:
    """Rate an answer. Returns False when the message is not this user's."""
    try:
        return await asyncio.to_thread(_feedback_sync, message_id, user_email, rating, reason)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not record feedback: {exc}") from exc


async def set_share_token(conversation_id: int, user_email: str, token: str | None) -> str | None:
    try:
        return await asyncio.to_thread(_share_sync, conversation_id, user_email, token)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not share conversation: {exc}") from exc


async def shared_conversation(token: str) -> dict[str, Any] | None:
    try:
        return await asyncio.to_thread(_shared_sync, token)
    except psycopg.Error as exc:
        raise ChatStoreUnavailableError(f"could not read shared conversation: {exc}") from exc


__all__ = [
    "CREATE_TABLE_SQL",
    "MAX_TITLE",
    "ChatStoreUnavailableError",
    "Conversation",
    "Message",
    "open_clarification",
    "record_clarification",
    "resolve_clarification",
    "PENDING_TTL_MINUTES",
    "record_feedback",
    "set_share_token",
    "shared_conversation",
    "append",
    "create",
    "delete",
    "ensure_table",
    "last_answer",
    "list_for_user",
    "messages",
    "conversation_of",
    "owns",
    "set_title_if_unset",
    "save_trace",
    "trace_for",
    "update",
]
