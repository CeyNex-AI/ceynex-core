"""Test doubles shared by the conversational surface's route tests (D12/D13).

Not a test module — pytest collects `test_*.py` only — so importing from here
never re-runs anyone's tests, and fixtures stay declared in the modules that use
them rather than imported across test files.

`install_fake_store` patches the `ceynex.chat.store` *module*, so every route and
runner that calls `store.<fn>(...)` sees the fake, however many modules sit
between the HTTP handler and the store.
"""

from __future__ import annotations

from typing import Any

from ceynex.api import deps as deps_module
from ceynex.api.auth import DemoUser, issue_token
from ceynex.chat import store as real_store
from tests.api.test_query import ANSWERED, FakeGraph, FakeKG, FakeLLM

OWNER = "policymaker@ceynex.dev"
OTHER = "researcher@ceynex.dev"

#: The store functions a route or runner may call, and the fake's attribute
#: that stands in for each.
_STORE_SEAMS = {
    "create": "create",
    "list_for_user": "list_for_user",
    "messages": "messages_for",
    "update": "update",
    "delete": "delete",
    "owns": "owns",
    "conversation_of": "conversation_of",
    "append": "append",
    "save_trace": "save_trace",
    "trace_for": "trace_for",
}


class FakeStore:
    """An in-memory stand-in that enforces ownership the way the real one does.

    It records every `user_email` it is asked for, so a route that forgets to
    scope its read is caught by an assertion on `seen_emails` rather than by
    hoping the fixture happens to produce a 404.
    """

    def __init__(self):
        self.conversations: dict[int, dict] = {}
        self.messages: dict[int, list] = {}
        self.traces: dict[str, list] = {}
        self.seen_emails: list[str] = []
        self._next = 1
        self._next_message_id = 0

    async def create(self, user_email, title=None):
        self.seen_emails.append(user_email)
        cid = self._next
        self._next += 1
        self.conversations[cid] = {
            "id": cid, "user_email": user_email, "title": title,
            "pinned": False, "archived": False,
        }
        self.messages[cid] = []
        return cid

    async def list_for_user(self, user_email, *, limit=50, include_archived=False):
        self.seen_emails.append(user_email)
        return [
            real_store.Conversation(
                id=c["id"], title=c["title"], created_at="2026-09-10T00:00:00+00:00",
                updated_at="2026-09-10T00:00:00+00:00", pinned=c["pinned"],
                archived=c["archived"],
                message_count=sum(1 for m in self.messages[c["id"]] if m.role == "user"),
            )
            for c in self.conversations.values()
            if c["user_email"] == user_email and (include_archived or not c["archived"])
        ]

    async def messages_for(self, conversation_id, user_email):
        self.seen_emails.append(user_email)
        c = self.conversations.get(conversation_id)
        if c is None or c["user_email"] != user_email:
            return None
        return list(self.messages[conversation_id])

    async def update(self, conversation_id, user_email, *, title=None, pinned=None, archived=None):
        self.seen_emails.append(user_email)
        c = self.conversations.get(conversation_id)
        if c is None or c["user_email"] != user_email:
            return False
        if title is not None:
            c["title"] = title
        if pinned is not None:
            c["pinned"] = pinned
        if archived is not None:
            c["archived"] = archived
        return True

    async def delete(self, conversation_id, user_email):
        self.seen_emails.append(user_email)
        c = self.conversations.get(conversation_id)
        if c is None or c["user_email"] != user_email:
            return False
        del self.conversations[conversation_id]
        del self.messages[conversation_id]
        return True

    async def conversation_of(self, message_id, user_email):
        self.seen_emails.append(user_email)
        for cid, conversation in self.conversations.items():
            if conversation["user_email"] != user_email:
                continue
            if any(m.id == message_id for m in self.messages[cid]):
                return cid
        return None

    async def owns(self, conversation_id, user_email):
        self.seen_emails.append(user_email)
        c = self.conversations.get(conversation_id)
        return c is not None and c["user_email"] == user_email

    async def append(self, conversation_id, user_email, messages):
        self.seen_emails.append(user_email)
        c = self.conversations.get(conversation_id)
        if c is None or c["user_email"] != user_email:
            return []
        seq = len(self.messages[conversation_id])
        ids = []
        for message in messages:
            seq += 1
            self._next_message_id += 1
            message.seq = seq
            message.id = self._next_message_id
            ids.append(message.id)
            self.messages[conversation_id].append(message)
        return ids

    async def save_trace(self, request_id, conversation_id, events):
        self.traces[request_id] = list(events)

    async def trace_for(self, request_id):
        return [
            {"seq": e.seq, "kind": e.kind, "node": e.node, "ts": e.ts, **e.payload}
            for e in self.traces.get(request_id, [])
        ]


def install_fake_store(monkeypatch) -> FakeStore:
    """Replace every store seam with an in-memory `FakeStore`, and return it."""
    fake = FakeStore()
    for name, attribute in _STORE_SEAMS.items():
        monkeypatch.setattr(real_store, name, getattr(fake, attribute))
    return fake


class TracingGraph(FakeGraph):
    """A fake graph that also emits a trace event.

    `FakeGraph` alone produces an empty sink, which would let the
    trace-persistence tests below pass while persisting nothing. Standing in for
    the graph means standing in for what the graph reports, too.
    """

    async def ainvoke(self, state):
        from ceynex.observability import trace

        with trace.node("export_analytics"):
            trace.emit("kg_query", cypher="MATCH (n) RETURN n", row_count=1, status="ok")
        return await super().ainvoke(state)


def auth(email=OWNER, role="policymaker"):
    token = issue_token(DemoUser(email=email, role=role, password_hash=b""))
    return {"authorization": f"Bearer {token}"}


def conversation_runtime(llm: Any = None, final: dict | None = None) -> deps_module.Runtime:
    """A runtime whose graph reports a real-looking trace and returns `final`."""
    return deps_module.Runtime(
        kg=FakeKG(), llm=llm or FakeLLM(), deps=None, graph=TracingGraph(final or ANSWERED)
    )


class ScriptedChatLLM:
    """A model that answers each role from a script, and remembers the prompts.

    `available` is True so the classifier and the discuss path take their model
    branches — the unavailable `FakeLLM` would keep both on the keyword path and
    none of the prompt-level behaviour would be exercised.
    """

    available = True

    def __init__(self, script: dict[str, str | None]):
        self.script = script
        self.systems: dict[str, list[str]] = {}
        self.users: dict[str, list[str]] = {}

    async def generate(self, role, system, user, *, json_mode=False, **_):
        self.systems.setdefault(role, []).append(system)
        self.users.setdefault(role, []).append(user)
        return self.script.get(role)


def frames(response) -> dict[str, dict]:
    """Frames by event name — the last of each kind wins."""
    from tests.api.test_chat_stream import parse_frames

    return dict(parse_frames(response.text))


def stream(client, query, conversation_id=None, headers=None):
    body = {"query": query}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    return client.post("/api/chat/stream", json=body, headers=headers or auth())
