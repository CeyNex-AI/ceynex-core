"""What a finished turn hands back, and what a reopened one still shows (D13).

The live turn and the stored turn are two renderings of the same answer, and for
a while they disagreed: the stored one had no working behind its confidence, no
link to the history row it produced, and a question the reader never typed. The
client also had no way to name the rows a turn was stored as, so it could not
fold a finished turn into its transcript, rate it, save it or regenerate it
without reloading. These tests pin both halves.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ceynex.api import deps as deps_module
from ceynex.api.main import app
from tests.api.chat_doubles import (
    ScriptedChatLLM,
    auth,
    conversation_runtime,
    install_fake_store,
)
from tests.api.chat_doubles import frames as _frames
from tests.api.chat_doubles import stream as _stream

BREAKDOWN = {"weighted": 0.7, "staleness": 0.0, "dq": 0.0, "coverage": 0.02, "final": 0.72}


@pytest.fixture
def fake_store(monkeypatch):
    return install_fake_store(monkeypatch)


@pytest.fixture
def client():
    deps_module.set_runtime(conversation_runtime())
    try:
        yield TestClient(app)
    finally:
        deps_module.set_runtime(None)


def _use(llm, final=None):
    deps_module.set_runtime(conversation_runtime(llm, final))


@pytest.fixture
def history_ids(monkeypatch):
    """`history.record` returning real-looking ids, without a database."""
    issued: list[dict] = []

    def record(**kwargs):
        issued.append(kwargs)
        return 4000 + len(issued)

    monkeypatch.setattr("ceynex.api.history.record", record)
    return issued


@pytest.fixture
def no_instruction(monkeypatch):
    async def none(_email):
        return "", True

    monkeypatch.setattr("ceynex.chat.instructions.get", none)


# --- the done frame names what it stored ------------------------------------


def test_done_names_the_rows_the_turn_was_stored_as(client, fake_store, history_ids,
                                                    no_instruction):
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    done = _frames(_stream(client, "cinnamon export trend", created["id"]))["done"]

    stored = fake_store.messages[created["id"]]
    assert done["user_message_id"] is not None and done["message_id"] is not None
    assert [done["user_message_id"], done["message_id"]] == [m.seq for m in stored]


def test_done_links_the_history_row_the_analysis_wrote(client, fake_store, history_ids,
                                                      no_instruction):
    """The chat save star calls the existing `/api/history/{id}/save`, so the
    turn has to know which row is its own — at once, not after a reload."""
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    done = _frames(_stream(client, "cinnamon export trend", created["id"]))["done"]

    assert done["query_history_id"] == 4001
    assert fake_store.messages[created["id"]][1].query_history_id == 4001


def test_a_discussion_writes_and_links_no_history_row(client, fake_store, history_ids,
                                                     no_instruction):
    """A discussion is not a new analysis. Linking it to a history row would
    offer to save a paraphrase as though it were a finding."""
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])
    done = _frames(_stream(client, "explain that", created["id"]))["done"]

    assert done["query_history_id"] is None
    assert done["message_id"] is not None
    assert len(history_ids) == 1, "only the analysis wrote a history row"


def test_an_unstored_turn_names_no_rows(client, fake_store, no_instruction):
    """Anonymous and stateless: nothing was written, so nothing is claimed."""
    done = _frames(client.post("/api/chat/stream", json={"query": "cinnamon"}))["done"]
    assert done["message_id"] is None and done["user_message_id"] is None


# --- a reopened turn shows what the live one showed -------------------------


def test_the_confidence_working_is_stored_with_the_answer(client, fake_store, history_ids,
                                                         no_instruction, monkeypatch):
    from ceynex.api import query_runner

    original = query_runner._assemble

    async def with_breakdown(runtime, query, final, started, web_results=(), observation=None):
        if observation is not None:
            observation.confidence_breakdown = BREAKDOWN
        return await original(runtime, query, final, started, web_results, observation)

    monkeypatch.setattr(query_runner, "_assemble", with_breakdown)
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])

    assert fake_store.messages[created["id"]][1].confidence_breakdown == BREAKDOWN

    # ...and a discussion of it carries the same working forward, as it carries
    # the score: it produced no new analysis, so it has no new working either.
    done = _frames(_stream(client, "explain that", created["id"]))["done"]
    assert done["answer"]["confidence_breakdown"] == BREAKDOWN


def test_a_rewritten_follow_up_is_stored_as_typed_beside_what_ran(client, fake_store,
                                                                history_ids, no_instruction):
    rewrite = "What are Sri Lanka's rubber export trends?"
    _use(ScriptedChatLLM({
        "turn_classify": json.dumps({"mode": "analyse", "standalone_query": rewrite}),
        "title": "Rubber",
    }))
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])
    _stream(client, "now do rubber", created["id"])

    follow_up = fake_store.messages[created["id"]][2]
    assert follow_up.role == "user"
    assert follow_up.content == "now do rubber", "the transcript must keep the reader's words"
    assert follow_up.effective_query == rewrite
    assert history_ids[-1]["query"] == rewrite, "history records the question that ran"


def test_an_unrewritten_question_stores_no_effective_query(client, fake_store, history_ids,
                                                         no_instruction):
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])
    assert fake_store.messages[created["id"]][0].effective_query is None


def test_a_follow_up_is_classified_against_the_question_that_ran(client, fake_store,
                                                                history_ids, no_instruction):
    """The classifier and the discuss context read `effective_query`: a follow-up
    to "now do rubber" is about rubber exports, not about the words "now do"."""
    rewrite = "What are Sri Lanka's rubber export trends?"
    llm = ScriptedChatLLM({
        "turn_classify": json.dumps({"mode": "analyse", "standalone_query": rewrite}),
    })
    _use(llm)
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])
    _stream(client, "now do rubber", created["id"])

    llm.script["turn_classify"] = json.dumps({"mode": "discuss"})
    _stream(client, "explain that", created["id"])
    assert f"Previous question: {rewrite}" in llm.users["turn_classify"][-1]


# --- the reader's instruction reaches a follow-up ---------------------------


def test_a_discussion_is_written_to_the_readers_instruction(client, fake_store, history_ids,
                                                           monkeypatch):
    async def instruction(_email):
        return "Answer in exactly three bullet points.", True

    monkeypatch.setattr("ceynex.chat.instructions.get", instruction)
    llm = ScriptedChatLLM({
        "turn_classify": json.dumps({"mode": "discuss"}),
        "chat": "Exports grew steadily.",
    })
    _use(llm)
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])
    _stream(client, "summarise that", created["id"])

    assert "Answer in exactly three bullet points." in llm.systems["chat"][-1]


def test_a_disabled_instruction_does_not_reach_a_discussion(client, fake_store, history_ids,
                                                           monkeypatch):
    async def disabled(_email):
        return "Answer in exactly three bullet points.", False

    monkeypatch.setattr("ceynex.chat.instructions.get", disabled)
    llm = ScriptedChatLLM({
        "turn_classify": json.dumps({"mode": "discuss"}),
        "chat": "Exports grew steadily.",
    })
    _use(llm)
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])
    _stream(client, "summarise that", created["id"])

    assert "three bullet points" not in llm.systems["chat"][-1]


def test_done_says_what_ran_when_it_is_not_what_was_typed(client, fake_store, history_ids,
                                                         no_instruction):
    """So a live turn folded into the transcript can show "interpreted as" at
    once, exactly as the stored turn will after a reload."""
    rewrite = "What are Sri Lanka's rubber export trends?"
    _use(ScriptedChatLLM({
        "turn_classify": json.dumps({"mode": "analyse", "standalone_query": rewrite}),
    }))
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    first = _frames(_stream(client, "cinnamon export trend", created["id"]))["done"]
    second = _frames(_stream(client, "now do rubber", created["id"]))["done"]

    assert first["effective_query"] is None
    assert second["effective_query"] == rewrite


# --- a conversation is named once ---------------------------------------------


def test_a_conversation_is_named_from_its_first_exchange_only(client, fake_store, history_ids,
                                                             no_instruction):
    """The title was always *written* once (`set_title_if_unset`), but the model
    was asked for one on every turn and the answer thrown away — a paid call per
    turn, for nothing, on the path that is supposed to be cheap."""
    llm = ScriptedChatLLM({
        "turn_classify": json.dumps({"mode": "analyse", "standalone_query": "rubber exports"}),
        "title": "Cinnamon exports",
    })
    _use(llm)
    created = client.post("/api/chat/conversations", json={}, headers=auth()).json()
    _stream(client, "cinnamon export trend", created["id"])
    _stream(client, "now do rubber", created["id"])
    llm.script["turn_classify"] = json.dumps({"mode": "discuss"})
    llm.script["chat"] = "Exports grew steadily."
    _stream(client, "explain that", created["id"])

    assert len(llm.users.get("title", [])) == 1
