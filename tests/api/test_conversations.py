"""Assertions for the past-chats surface (deviation D13, SRS 3.5.2, SRS 3.10).

Two things carry most of the weight:

**Ownership.** A conversation is addressed by a `BIGSERIAL` id, so every route
must scope to the caller in the statement itself. These tests drive a fake store
that records the `user_email` it was asked for, which catches a route that reads
by id alone far more reliably than a test that merely expects a 404.

**`query_history` is untouched.** The existing History panel and its save button
must keep working, so a chat turn still writes a `query_history` row and links to
it rather than replacing it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ceynex.api import deps as deps_module
from ceynex.api.main import app
from tests.api.chat_doubles import (
    OTHER,
    OWNER,
    TracingGraph,
    auth,
    install_fake_store,
)
from tests.api.chat_doubles import frames as _frames
from tests.api.chat_doubles import stream as _stream
from tests.api.test_query import ANSWERED, FakeKG, FakeLLM


@pytest.fixture
def fake_store(monkeypatch):
    return install_fake_store(monkeypatch)


@pytest.fixture
def client():
    deps_module.set_runtime(
        deps_module.Runtime(
            kg=FakeKG(), llm=FakeLLM(), deps=None, graph=TracingGraph(ANSWERED)
        )
    )
    try:
        yield TestClient(app)
    finally:
        deps_module.set_runtime(None)


# --- authentication is required -------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/chat/conversations"),
        ("post", "/api/chat/conversations"),
        ("get", "/api/chat/conversations/1"),
        ("patch", "/api/chat/conversations/1"),
        ("delete", "/api/chat/conversations/1"),
    ],
)
def test_every_conversation_route_needs_a_signed_in_user(client, fake_store, method, path):
    """`/api/query` answers anonymous callers by a documented decision. A
    conversation is stateful and addressed by a guessable serial, so it does not."""
    kwargs = {"json": {}} if method in ("post", "patch") else {}
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 401


# --- ownership ------------------------------------------------------------


def test_a_conversation_is_scoped_to_its_owner_on_every_read(client, fake_store):
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    fake_store.seen_emails.clear()

    client.get(f"/api/chat/conversations/{created['id']}", headers=auth())
    assert fake_store.seen_emails, "the route read without scoping to a user"
    assert all(email == OWNER for email in fake_store.seen_emails)


def test_another_users_conversation_is_not_found_rather_than_forbidden(client, fake_store):
    """404, not 403 — distinguishing them tells a stranger the id exists."""
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()

    for method in ("get", "patch", "delete"):
        kwargs = {"json": {"title": "mine now"}} if method == "patch" else {}
        response = getattr(client, method)(
            f"/api/chat/conversations/{created['id']}",
            headers=auth(OTHER, "researcher"),
            **kwargs,
        )
        assert response.status_code == 404, method


def test_listing_shows_only_your_own(client, fake_store):
    client.post("/api/chat/conversations", json={"title": "mine"}, headers=auth())
    client.post("/api/chat/conversations", json={"title": "theirs"},
                headers=auth(OTHER, "researcher"))

    mine = client.get("/api/chat/conversations", headers=auth()).json()
    assert [c["title"] for c in mine] == ["mine"]


# --- the lifecycle --------------------------------------------------------


def test_a_conversation_can_be_created_renamed_pinned_and_deleted(client, fake_store):
    created = client.post("/api/chat/conversations", json={"title": "draft"}, headers=auth())
    assert created.status_code == 201
    cid = created.json()["id"]

    renamed = client.patch(
        f"/api/chat/conversations/{cid}", json={"title": "cinnamon markets"}, headers=auth()
    )
    assert renamed.json()["title"] == "cinnamon markets"

    pinned = client.patch(f"/api/chat/conversations/{cid}", json={"pinned": True}, headers=auth())
    assert pinned.json()["pinned"] is True
    assert pinned.json()["title"] == "cinnamon markets", "a PATCH must not clear what it omits"

    assert client.delete(f"/api/chat/conversations/{cid}", headers=auth()).status_code == 204
    assert client.get(f"/api/chat/conversations/{cid}", headers=auth()).status_code == 404


def test_an_empty_patch_is_rejected(client, fake_store):
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    response = client.patch(f"/api/chat/conversations/{created['id']}", json={}, headers=auth())
    assert response.status_code == 422


def test_archived_conversations_are_hidden_unless_asked_for(client, fake_store):
    created = client.post("/api/chat/conversations", json={"title": "old"}, headers=auth()).json()
    client.patch(f"/api/chat/conversations/{created['id']}", json={"archived": True}, headers=auth())

    assert client.get("/api/chat/conversations", headers=auth()).json() == []
    included = client.get(
        "/api/chat/conversations?include_archived=true", headers=auth()
    ).json()
    assert [c["title"] for c in included] == ["old"]


def test_a_deleted_conversation_is_really_gone(client, fake_store):
    """SRS 3.10 — a "deleted" row still in the table is still collected."""
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    client.delete(f"/api/chat/conversations/{created['id']}", headers=auth())
    assert created["id"] not in fake_store.conversations


def test_chat_routes_disappear_when_chat_is_off(client, fake_store, monkeypatch):
    monkeypatch.setattr("ceynex.settings.chat_enabled", lambda: False)
    assert client.get("/api/chat/conversations", headers=auth()).status_code == 404


# --- turns inside a conversation ------------------------------------------


def test_a_conversation_id_requires_a_signed_in_user(client, fake_store):
    """Anonymous streaming still works — but not against someone's transcript."""
    assert client.post("/api/chat/stream", json={"query": "cinnamon"}).status_code == 200
    response = client.post("/api/chat/stream", json={"query": "cinnamon", "conversation_id": 1})
    assert response.status_code == 401


def test_streaming_into_another_users_conversation_is_not_found(client, fake_store):
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    response = _stream(client, "cinnamon", created["id"], auth(OTHER, "researcher"))
    assert response.status_code == 404


def test_a_first_turn_runs_the_graph_and_stores_both_halves(client, fake_store):
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    response = _stream(client, "cinnamon export trend", created["id"])

    assert response.status_code == 200
    stored = fake_store.messages[created["id"]]
    assert [m.role for m in stored] == ["user", "assistant"]
    assert stored[0].content == "cinnamon export trend"
    assert stored[1].mode == "analyse"
    assert stored[1].evidence, "the answer payload must survive, not just the prose"


def test_a_first_turn_emits_no_turn_frame(client, fake_store):
    """There is nothing to follow up on, so nothing to classify."""
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    frames = _frames(_stream(client, "cinnamon export trend", created["id"]))
    assert "turn" not in frames


def test_a_follow_up_naming_something_new_re_runs_the_graph(client, fake_store):
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])

    frames = _frames(_stream(client, "what about rubber", created["id"]))
    assert frames["turn"]["mode"] == "analyse"
    assert frames["done"]["failed"] is False


def test_a_follow_up_about_the_answer_skips_the_graph(client, fake_store):
    """The cheap path. Most follow-ups are this, which is what keeps a
    conversation from costing five agent runs per turn."""
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])

    response = _stream(client, "explain that more simply", created["id"])
    frames = _frames(response)

    assert frames["turn"]["mode"] == "discuss"
    # No fan-out happened: the analyse path always emits node events, this must not.
    assert "node_start" not in frames
    assert "kg_query" not in frames


def test_a_discussion_carries_the_previous_confidence_forward(client, fake_store):
    """A discussion produces no new analysis, so inventing a fresh confidence
    score for it would be a number with nothing behind it."""
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    first = _frames(_stream(client, "cinnamon export trend", created["id"]))["done"]["answer"]

    follow = _frames(_stream(client, "explain that", created["id"]))["done"]["answer"]

    assert follow["confidence"] == first["confidence"]
    assert follow["evidence"] == first["evidence"]


def test_a_discussion_is_stored_as_a_discussion(client, fake_store):
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])
    _stream(client, "explain that", created["id"])

    stored = fake_store.messages[created["id"]]
    assert [m.mode for m in stored if m.role == "assistant"] == ["analyse", "discuss"]


def test_the_trace_is_persisted_and_replayable(client, fake_store):
    """Reopening a past chat replays the real trace, not a summary of it."""
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    done = _frames(_stream(client, "cinnamon export trend", created["id"]))["done"]

    replayed = client.get(
        f"/api/chat/conversations/{created['id']}/trace/{done['request_id']}", headers=auth()
    )
    assert replayed.status_code == 200
    events = replayed.json()
    assert events, "no trace was stored"
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)


def test_a_trace_cannot_be_read_through_someone_elses_conversation(client, fake_store):
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    done = _frames(_stream(client, "cinnamon export trend", created["id"]))["done"]

    response = client.get(
        f"/api/chat/conversations/{created['id']}/trace/{done['request_id']}",
        headers=auth(OTHER, "researcher"),
    )
    assert response.status_code == 404


def test_reopening_a_conversation_returns_the_whole_transcript(client, fake_store):
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])
    _stream(client, "explain that", created["id"])

    detail = client.get(f"/api/chat/conversations/{created['id']}", headers=auth()).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert detail["conversation"]["message_count"] == 2


def test_a_declined_analysis_can_still_be_discussed(client, fake_store, monkeypatch):
    """Found end-to-end: with no databases loaded the first turn declines, and
    "explain that more simply" then re-ran the entire five-agent fan-out to
    produce a byte-identical decline.

    The cause was requiring the prior assistant turn to carry evidence before a
    follow-up counted as a follow-up. A decline is precisely when someone asks
    what you meant, and re-running cannot tell them.
    """
    empty_answer = {
        **ANSWERED,
        "merged_evidence": [],
        "final_answer": "This question could not be answered from the data currently loaded.",
    }
    deps_module.set_runtime(
        deps_module.Runtime(
            kg=FakeKG(), llm=FakeLLM(), deps=None, graph=TracingGraph(empty_answer)
        )
    )
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])

    frames = _frames(_stream(client, "explain that more simply", created["id"]))

    assert frames["turn"]["mode"] == "discuss"
    assert "node_start" not in frames, "the fan-out ran again for a decline"
