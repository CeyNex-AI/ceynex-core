"""Assertions for the clarification gate on the wire (deviation D13).

Three properties, and the last two are the ones that would be expensive to get
wrong: the resume route carries the same rate limit as the stream it replaces,
and claiming a pending row is atomic, so a double submission cannot run the
five-agent fan-out twice.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ceynex.api import deps as deps_module
from ceynex.api.auth import issue_token
from ceynex.api.main import app
from ceynex.api.routes import chat as chat_routes
from ceynex.chat import store as real_store
from tests.api.test_query import ANSWERED, FakeGraph, FakeKG, FakeLLM


class CountingGraph(FakeGraph):
    """`FakeGraph` emits no trace events, so asserting a `merge` frame is absent
    would pass whether or not the fan-out ran. Counting invocations is the only
    assertion here that actually distinguishes "the gate stopped it" from "the
    fake never says anything anyway"."""

    def __init__(self, final=None):
        super().__init__(final)
        self.invocations = 0

    async def ainvoke(self, state):
        self.invocations += 1
        return await super().ainvoke(state)

OWNER = "policymaker@ceynex.dev"
AMBIGUOUS = "How did tea and cinnamon exports do last year?"


class PendingStore:
    """Just the clarification half of the store, in memory.

    `resolve` mirrors the real `UPDATE ... WHERE NOT resolved ... RETURNING`:
    the first caller gets the row and every later one gets `None`. That is the
    property the one-round cap rests on, so a fake that always returned the row
    would let the test pass while production ran the turn twice.
    """

    def __init__(self):
        self.rows: dict[int, dict[str, Any]] = {}
        self._next = 1

    async def owns(self, conversation_id, user_email):
        return True

    async def messages(self, conversation_id, user_email):
        return []

    async def record_clarification(self, conversation_id, user_email, original_query, payload):
        pid = self._next
        self._next += 1
        self.rows[pid] = {
            "id": pid, "conversation_id": conversation_id, "user_email": user_email,
            "original_query": original_query, "payload": payload, "resolved": False,
        }
        return pid

    async def resolve_clarification(self, pending_id, user_email):
        row = self.rows.get(pending_id)
        if row is None or row["resolved"] or row["user_email"] != user_email:
            return None
        row["resolved"] = True
        return row

    async def append(self, *a, **k):
        return [1, 2]

    async def save_trace(self, *a, **k):
        return None

    async def set_title_if_unset(self, *a, **k):
        return True


def parse_frames(body: str) -> list[tuple[str, dict[str, Any]]]:
    frames = []
    for block in body.split("\n\n"):
        block = block.strip("\n")
        if not block or block.startswith(":"):
            continue
        event, data = None, []
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        if event:
            frames.append((event, json.loads("".join(data)) if data else {}))
    return frames


@pytest.fixture
def pending(monkeypatch):
    store = PendingStore()
    for name in (
        "owns", "messages", "record_clarification", "resolve_clarification",
        "append", "save_trace", "set_title_if_unset",
    ):
        monkeypatch.setattr(chat_routes.store, name, getattr(store, name))
    monkeypatch.setattr(real_store, "ChatStoreUnavailableError", real_store.ChatStoreUnavailableError)
    graph = CountingGraph(ANSWERED)
    store.graph = graph
    deps_module.set_runtime(
        deps_module.Runtime(kg=FakeKG(), llm=FakeLLM(), deps=None, graph=graph)
    )
    try:
        yield store
    finally:
        deps_module.set_runtime(None)


def auth():
    return {"Authorization": f"Bearer {issue_token(OWNER, 'policymaker')}"}


def test_an_ambiguous_question_is_asked_about_rather_than_guessed(pending):
    client = TestClient(app)
    body = client.post(
        "/api/chat/stream", json={"query": AMBIGUOUS, "conversation_id": 1}, headers=auth()
    ).text
    kinds = [k for k, _ in parse_frames(body)]
    assert "clarify" in kinds
    assert kinds[-1] == "done"

    clar = next(p for k, p in parse_frames(body) if k == "clarify")
    assert clar["allow_skip"] is True
    assert {"tea", "cinnamon"} <= set(clar["options"])
    # The graph must not have run: a clarified turn is a question, not an answer,
    # and paying for a five-agent fan-out before knowing what was asked is the
    # cost this gate exists to avoid.
    assert pending.graph.invocations == 0


def test_a_clear_question_is_never_interrupted(pending):
    client = TestClient(app)
    body = client.post(
        "/api/chat/stream",
        json={"query": "What is the current price trend for cinnamon?", "conversation_id": 1},
        headers=auth(),
    ).text
    assert "clarify" not in [k for k, _ in parse_frames(body)]


def test_the_gate_can_be_switched_off_without_losing_chat(pending, monkeypatch):
    """A reviewer comparing answers against queries.md needs exactly this."""
    monkeypatch.setattr("ceynex.settings.clarify_enabled", lambda: False)
    client = TestClient(app)
    body = client.post(
        "/api/chat/stream", json={"query": AMBIGUOUS, "conversation_id": 1}, headers=auth()
    ).text
    kinds = [k for k, _ in parse_frames(body)]
    assert "clarify" not in kinds
    assert kinds[-1] == "done"


def test_answering_resumes_the_turn_and_never_re_asks(pending):
    client = TestClient(app)
    first = client.post(
        "/api/chat/stream", json={"query": AMBIGUOUS, "conversation_id": 1}, headers=auth()
    ).text
    pid = next(p for k, p in parse_frames(first) if k == "clarify")["pending_id"]

    body = client.post(
        f"/api/chat/clarify/{pid}/answer", json={"answers": ["cinnamon"]}, headers=auth()
    ).text
    kinds = [k for k, _ in parse_frames(body)]
    assert "clarify" not in kinds, "the gate must not run on the resume path"
    assert kinds[-1] == "done"
    # ...and this time it really did run, exactly once.
    assert pending.graph.invocations == 1


def test_a_pending_question_can_only_be_answered_once(pending):
    """The one-round cap is a claimed row, not a counter — so a double submit
    cannot run the fan-out twice across two uvicorn workers."""
    client = TestClient(app)
    first = client.post(
        "/api/chat/stream", json={"query": AMBIGUOUS, "conversation_id": 1}, headers=auth()
    ).text
    pid = next(p for k, p in parse_frames(first) if k == "clarify")["pending_id"]

    assert client.post(
        f"/api/chat/clarify/{pid}/answer", json={"answers": ["tea"]}, headers=auth()
    ).status_code == 200
    assert client.post(
        f"/api/chat/clarify/{pid}/answer", json={"answers": ["tea"]}, headers=auth()
    ).status_code == 404


def test_the_resume_route_is_rate_limited_like_the_stream(pending):
    """The plan named this route as the one remaining bypass. It invokes the
    identical fan-out, so it carries the identical allowance."""
    seen: list[str] = []

    class Blocked:
        async def check(self, identity, limit, window_s):
            from ceynex.api.rate_limit import Decision

            seen.append(identity)
            return Decision(allowed=False, limit=limit, remaining=0, retry_after_s=30)

    chat_routes.set_chat_window(Blocked())
    try:
        response = TestClient(app).post(
            "/api/chat/clarify/1/answer", json={"answers": ["tea"]}, headers=auth()
        )
        assert response.status_code == 429
        assert all(identity.startswith("chat:") for identity in seen), seen
    finally:
        chat_routes.set_chat_window(None)


def test_answering_someone_elses_pending_question_is_a_404(pending):
    client = TestClient(app)
    first = client.post(
        "/api/chat/stream", json={"query": AMBIGUOUS, "conversation_id": 1}, headers=auth()
    ).text
    pid = next(p for k, p in parse_frames(first) if k == "clarify")["pending_id"]

    other = {"Authorization": f"Bearer {issue_token('someone@else.dev', 'policymaker')}"}
    assert client.post(
        f"/api/chat/clarify/{pid}/answer", json={"answers": ["tea"]}, headers=other
    ).status_code == 404
