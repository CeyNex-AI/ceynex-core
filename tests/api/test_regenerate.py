"""Regenerate: a fresh answer to the latest question, kept beside the old one.

What is pinned here is what makes Regenerate honest rather than a slot machine:

- **It asks the same question the same way.** An analysis re-runs the graph —
  the trace is real — and only the merge is asked for new wording, by skipping
  its prompt cache; a discussion re-discusses the same prior answer. No
  re-classification, no clarifying question.
- **Nothing is overwritten.** The new answer is stored beside the one it
  replaces, linked by `regenerated_from`, and the question is not stored twice.
- **Only the latest answer.** Every later turn was classified and grounded
  against an answer, so replacing one further back would fork the conversation.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ceynex.api import deps as deps_module
from ceynex.api import turn_log
from ceynex.api.main import app
from ceynex.api.turn_log import LocalTurnRegistry
from ceynex.api.turn_runner import plan_regeneration
from ceynex.chat.store import Message
from ceynex.observability import context
from tests.api.chat_doubles import (
    OTHER,
    ScriptedChatLLM,
    TracingGraph,
    auth,
    install_fake_store,
)
from tests.api.chat_doubles import frames as _frames
from tests.api.chat_doubles import stream as _stream
from tests.api.test_query import ANSWERED, FakeKG

# --- the plan, from a transcript ------------------------------------------------


def _m(id_, seq, role, content="x", mode=None, effective_query=None):
    return Message(id=id_, seq=seq, role=role, content=content, mode=mode,
                   effective_query=effective_query)


TRANSCRIPT = [
    _m(1, 1, "user", "cinnamon exports"),
    _m(2, 2, "assistant", "Cinnamon rose.", mode="analyse"),
    _m(3, 3, "user", "explain that"),
    _m(4, 4, "assistant", "It went up.", mode="discuss"),
]


def test_only_the_latest_answer_can_be_regenerated():
    assert plan_regeneration(TRANSCRIPT, 4) is not None
    assert plan_regeneration(TRANSCRIPT, 2) is None, "an earlier answer would fork the thread"
    assert plan_regeneration(TRANSCRIPT, 3) is None, "a question is not an answer"
    assert plan_regeneration(TRANSCRIPT, 99) is None


def test_a_discussion_regenerates_against_the_answer_it_discussed():
    plan = plan_regeneration(TRANSCRIPT, 4)
    assert plan.mode == "discuss"
    assert plan.question.content == "explain that"
    assert plan.prior.id == 2 and plan.prior_query == "cinnamon exports"


def test_an_analysis_regenerates_the_question_that_actually_ran():
    transcript = [
        *TRANSCRIPT,
        _m(5, 5, "user", "now do rubber", effective_query="Rubber export trends?"),
        _m(6, 6, "assistant", "Rubber fell.", mode="analyse"),
    ]
    plan = plan_regeneration(transcript, 6)
    assert plan.mode == "analyse"
    assert plan.question.asked == "Rubber export trends?"
    assert plan.question.content == "now do rubber"


def test_after_a_regenerate_the_new_version_is_the_latest():
    regenerated = [*TRANSCRIPT, _m(5, 5, "assistant", "It rose, again.", mode="discuss")]
    regenerated[-1].regenerated_from = 4
    assert plan_regeneration(regenerated, 5) is not None
    assert plan_regeneration(regenerated, 4) is None
    # The question of the new version is still the one question asked.
    assert plan_regeneration(regenerated, 5).question.id == 3


# --- the route ----------------------------------------------------------------------


class RecordingGraph(TracingGraph):
    """Counts its runs, and records which LLM roles skipped the cache in each."""

    def __init__(self, final=None):
        super().__init__(final or ANSWERED)
        self.runs = 0
        self.bypassed: list[frozenset] = []

    async def ainvoke(self, state):
        self.runs += 1
        observation = context.current()
        self.bypassed.append(observation.bypass_cache_roles if observation else frozenset())
        return await super().ainvoke(state)


class BypassAwareLLM(ScriptedChatLLM):
    """Records, per role, whether the request asked it to skip the cache."""

    def __init__(self, script):
        super().__init__(script)
        self.bypassed: dict[str, list[bool]] = {}

    async def generate(self, role, system, user, *, json_mode=False, **kwargs):
        observation = context.current()
        skip = bool(observation and role in observation.bypass_cache_roles)
        self.bypassed.setdefault(role, []).append(skip)
        return await super().generate(role, system, user, json_mode=json_mode, **kwargs)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    turn_log.set_registry(LocalTurnRegistry())
    turn_log.set_mirror(None)

    async def no_instruction(_email):
        return "", True

    async def no_ledger(**_kwargs):
        return None

    monkeypatch.setattr("ceynex.chat.instructions.get", no_instruction)
    monkeypatch.setattr("ceynex.observability.ledger.record", no_ledger)
    yield
    turn_log.set_registry(None)


@pytest.fixture
def history_rows(monkeypatch):
    rows: list[dict] = []

    def record(**kwargs):
        rows.append(kwargs)
        return 7000 + len(rows)

    monkeypatch.setattr("ceynex.api.history.record", record)
    return rows


@pytest.fixture
def fake_store(monkeypatch):
    return install_fake_store(monkeypatch)


@pytest.fixture
def graph():
    return RecordingGraph()


@pytest.fixture
def llm():
    return BypassAwareLLM({
        "turn_classify": json.dumps({"mode": "discuss"}),
        "chat": "It went up.",
        "title": "Cinnamon",
    })


@pytest.fixture
def client(graph, llm):
    deps_module.set_runtime(deps_module.Runtime(kg=FakeKG(), llm=llm, deps=None, graph=graph))
    try:
        yield TestClient(app)
    finally:
        deps_module.set_runtime(None)


def _regenerate(client, message_id, headers=None):
    return client.post(f"/api/chat/messages/{message_id}/regenerate",
                       headers=headers if headers is not None else auth())


def _first_turn(client, fake_store):
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])
    return created["id"], fake_store.messages[created["id"]]


def test_an_analysis_is_regenerated_by_running_the_graph_again(client, fake_store, graph,
                                                               history_rows):
    conversation_id, stored = _first_turn(client, fake_store)
    first_answer = stored[1]

    response = _regenerate(client, first_answer.id)
    assert response.status_code == 200
    done = _frames(response)["done"]

    assert graph.runs == 2, "the regenerate must run the graph, not replay it"
    assert graph.bypassed[-1] == frozenset({"merge"}), "the merge must not reuse its answer"
    assert graph.bypassed[0] == frozenset(), "an ordinary turn reads the cache as always"
    assert done["regenerated_from"] == first_answer.id
    assert done["user_message_id"] is None, "the question is not stored twice"

    transcript = fake_store.messages[conversation_id]
    assert [m.role for m in transcript] == ["user", "assistant", "assistant"]
    assert transcript[2].regenerated_from == first_answer.id
    assert transcript[1].content == first_answer.content, "the old version was overwritten"
    assert len(history_rows) == 2, "a regenerated analysis is an analysis: it is recorded"


def test_a_discussion_is_regenerated_without_the_graph(client, fake_store, graph, llm,
                                                       history_rows):
    conversation_id, _ = _first_turn(client, fake_store)
    _stream(client, "explain that", conversation_id)
    discussion = fake_store.messages[conversation_id][-1]
    assert discussion.mode == "discuss"
    runs_before = graph.runs

    done = _frames(_regenerate(client, discussion.id))["done"]

    assert graph.runs == runs_before, "a discussion regenerate must not re-run the fan-out"
    assert llm.bypassed["chat"][-1] is True, "the chat answer must not come from the cache"
    assert done["regenerated_from"] == discussion.id
    assert fake_store.messages[conversation_id][-1].regenerated_from == discussion.id
    assert len(history_rows) == 1, "a discussion writes no history row, regenerated or not"


def test_a_regenerate_asks_no_clarifying_question_and_reclassifies_nothing(
    client, fake_store, llm, history_rows
):
    conversation_id, stored = _first_turn(client, fake_store)
    classified = len(llm.bypassed.get("turn_classify", []))

    frames = _frames(_regenerate(client, stored[1].id))

    assert "clarify" not in frames
    assert frames["turn"]["method"] == "regenerate"
    assert len(llm.bypassed.get("turn_classify", [])) == classified


def test_a_regenerate_does_not_rename_the_conversation(client, fake_store, llm, history_rows):
    conversation_id, stored = _first_turn(client, fake_store)
    titles_before = len(llm.users.get("title", []))
    _regenerate(client, stored[1].id)
    assert len(llm.users.get("title", [])) == titles_before


def test_an_earlier_answer_cannot_be_regenerated(client, fake_store, history_rows):
    conversation_id, stored = _first_turn(client, fake_store)
    _stream(client, "explain that", conversation_id)
    response = _regenerate(client, stored[1].id)
    assert response.status_code == 409


def test_someone_elses_answer_is_not_found(client, fake_store, history_rows):
    _, stored = _first_turn(client, fake_store)
    response = _regenerate(client, stored[1].id, headers=auth(OTHER, "researcher"))
    assert response.status_code == 404


def test_regenerate_needs_a_signed_in_user(client, fake_store):
    assert _regenerate(client, 1, headers={}).status_code == 401


def test_regenerate_is_rate_limited_like_a_turn(client, fake_store, history_rows):
    """An analysis regenerate runs the whole fan-out; without the chat limit it
    would be a bypass around SRS 3.4.6 for the most expensive call there is."""
    from ceynex.api.rate_limit import Decision
    from ceynex.api.routes import chat as chat_routes

    seen: list[str] = []

    class Blocked:
        async def check(self, identity, limit, window_s):
            seen.append(identity)
            return Decision(allowed=False, limit=limit, remaining=0, retry_after_s=9)

    _, stored = _first_turn(client, fake_store)
    chat_routes.set_chat_window(Blocked())
    try:
        response = _regenerate(client, stored[1].id)
        assert response.status_code == 429
        assert seen and all(identity.startswith("chat:") for identity in seen)
    finally:
        chat_routes.set_chat_window(None)
