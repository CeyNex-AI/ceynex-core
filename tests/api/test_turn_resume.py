"""A turn outlives the connection watching it (D12, amended).

Three properties, and the second is the one that closes a recorded risk:

**Resume is exact.** Every frame is numbered, and a reader that drops and comes
back with the last number it saw gets every later frame exactly once, in order —
whether the turn has finished or is still running.

**Stop means stop, cleanly.** `EXECUTION_PLAN_CONVERSATIONAL.md` §6 left one
risk open: that cancelling mid-turn might leak a Neo4j session or leave a task
failing unobserved. It is tested here against the *real* compiled graph and the
*real* `KnowledgeGraphClient`, over a stub session whose query never returns —
cancelled from outside, mid-query, the way the Stop endpoint does it.

**A disconnect is not a Stop.** A signed-in turn finishes and is stored when its
reader leaves; an anonymous one, which nobody could ever resume, is cancelled.
"""

from __future__ import annotations

import asyncio
import gc
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from ceynex.agents.common import AgentDeps
from ceynex.api import deps as deps_module
from ceynex.api import turn_log, turn_runner
from ceynex.api.main import app
from ceynex.api.turn_log import LocalTurnRegistry, RedisTurnMirror, TurnFrame
from ceynex.chat.store import Message
from ceynex.llm import FakeLLMClient
from ceynex.orchestrator.graph import build_graph
from tests.api.chat_doubles import (
    OTHER,
    OWNER,
    ScriptedChatLLM,
    auth,
    conversation_runtime,
    install_fake_store,
)
from tests.api.test_chat_stream import parse_frames
from tests.api.test_query import ANSWERED, FakeGraph, FakeKG, FakeLLM

# --- fixtures ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """A fresh turn registry, no Redis mirror, and none of the best-effort
    writes a turn makes reaching a real database."""
    turn_log.set_registry(LocalTurnRegistry())
    turn_log.set_mirror(None)

    async def no_instruction(_email):
        return "", True

    async def no_ledger(**_kwargs):
        return None

    monkeypatch.setattr("ceynex.chat.instructions.get", no_instruction)
    monkeypatch.setattr("ceynex.observability.ledger.record", no_ledger)
    monkeypatch.setattr("ceynex.api.history.record", lambda **_kwargs: None)
    yield
    turn_log.set_registry(None)
    turn_log.set_mirror(None)


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


class GatedGraph(FakeGraph):
    """A graph that reports one step, then waits for the test to let it finish.

    Records whether it was cancelled, because "the turn was cancelled" is only a
    claim until the fan-out itself stops.
    """

    def __init__(self, final=None):
        super().__init__(final or ANSWERED)
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.cancelled = False
        self.completed = False

    async def ainvoke(self, state):
        from ceynex.observability import trace

        with trace.node("export_analytics"):
            trace.emit("kg_query", cypher="MATCH (n) RETURN n", row_count=1, status="ok")
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.completed = True
        return await super().ainvoke(state)


def _gated_runtime(graph):
    return deps_module.Runtime(kg=FakeKG(), llm=FakeLLM(), deps=None, graph=graph)


class Disconnectable:
    """Stands in for a Starlette `Request`: connected until told otherwise."""

    def __init__(self):
        self.gone = False

    async def is_disconnected(self):
        return self.gone


def _numbered(body: str) -> list[tuple[int, str]]:
    """(`id`, `event`) for every frame, the way a browser's parser sees them."""
    frames = []
    for block in body.split("\n\n"):
        lines = [line for line in block.strip("\n").splitlines() if not line.startswith(":")]
        if not lines:
            continue
        ids = [line[len("id: "):] for line in lines if line.startswith("id: ")]
        events = [line[len("event: "):] for line in lines if line.startswith("event: ")]
        frames.append((int(ids[0]) if ids else -1, events[0] if events else "?"))
    return frames


def _stream_turn(client, headers=None, conversation_id=None):
    body = {"query": "cinnamon export trend"}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    return client.post("/api/chat/stream", json=body, headers=headers or {})


async def _drain_turn(started: turn_log.LocalTurn) -> None:
    """Wait for the turn to end on its own. Deliberately not `wait_for`, which
    cancels on timeout — and would perform the very cancellation a test is
    checking the code under test performs."""
    if started.task is not None:
        done, _ = await asyncio.wait({started.task}, timeout=5)
        assert done, "the turn never finished"


# --- every frame is numbered ----------------------------------------------------


def test_every_frame_is_numbered_from_one_without_gaps(client, fake_store):
    """The number is the resume token's other half. A gap or a repeat would make
    "everything after N" ambiguous."""
    frames = _numbered(_stream_turn(client).text)
    assert [seq for seq, _ in frames] == list(range(1, len(frames) + 1))
    assert frames[0][1] == "start" and frames[-1][1] == "done"


def test_the_start_frame_names_the_turn_and_whether_it_can_be_resumed(client, fake_store):
    signed_in = dict(parse_frames(_stream_turn(client, auth()).text))["start"]
    anonymous = dict(parse_frames(_stream_turn(client).text))["start"]

    assert signed_in["request_id"] and signed_in["resumable"] is True
    # No owner, so nothing to check a resume against — and nothing is stored.
    assert anonymous["resumable"] is False


def test_the_done_frame_and_the_start_frame_name_the_same_turn(client, fake_store):
    frames = dict(parse_frames(_stream_turn(client, auth()).text))
    assert frames["start"]["request_id"] == frames["done"]["request_id"]


# --- resuming a finished turn -----------------------------------------------------


def test_a_reader_resumes_after_the_last_frame_it_saw(client, fake_store):
    original = _stream_turn(client, auth()).text
    request_id = dict(parse_frames(original))["start"]["request_id"]
    everything = _numbered(original)

    resumed = client.get(f"/api/chat/turns/{request_id}/events?after=2", headers=auth())
    assert resumed.status_code == 200
    assert _numbered(resumed.text) == everything[2:], "not exactly the frames after 2"


def test_the_standard_last_event_id_header_is_honoured(client, fake_store):
    """What a browser's own EventSource would send; ours sends `after`, but a
    server that ignored the standard header would replay from the start."""
    original = _stream_turn(client, auth()).text
    request_id = dict(parse_frames(original))["start"]["request_id"]

    resumed = client.get(
        f"/api/chat/turns/{request_id}/events", headers={**auth(), "last-event-id": "3"}
    )
    assert _numbered(resumed.text) == _numbered(original)[3:]


def test_resuming_from_the_start_replays_the_turn_as_it_happened(client, fake_store):
    original = _stream_turn(client, auth()).text
    request_id = dict(parse_frames(original))["start"]["request_id"]

    resumed = client.get(f"/api/chat/turns/{request_id}/events?after=0", headers=auth())
    assert parse_frames(resumed.text) == parse_frames(original)


def test_someone_elses_turn_cannot_be_resumed(client, fake_store):
    """The request id is unguessable, but it is not the whole token: a leaked
    one must not hand someone else the answer."""
    original = _stream_turn(client, auth()).text
    request_id = dict(parse_frames(original))["start"]["request_id"]

    stolen = client.get(f"/api/chat/turns/{request_id}/events", headers=auth(OTHER, "researcher"))
    assert stolen.status_code == 404


def test_an_anonymous_turn_cannot_be_resumed_by_anyone(client, fake_store):
    original = _stream_turn(client).text
    request_id = dict(parse_frames(original))["start"]["request_id"]
    assert client.get(f"/api/chat/turns/{request_id}/events", headers=auth()).status_code == 404


def test_an_unknown_turn_is_not_found(client, fake_store):
    assert client.get("/api/chat/turns/nope/events", headers=auth()).status_code == 404


def test_resuming_needs_a_signed_in_reader(client, fake_store):
    assert client.get("/api/chat/turns/nope/events").status_code == 401
    assert client.post("/api/chat/turns/nope/cancel").status_code == 401


# --- a reader joining a turn that is still running ---------------------------------


async def test_a_reader_joining_mid_turn_gets_the_past_then_the_live_frames(fake_store):
    graph = GatedGraph()
    deps_module.set_runtime(_gated_runtime(graph))
    try:
        started = turn_runner.start_turn(
            turn_runner.TurnRequest(runtime=deps_module.get_runtime(), query="cinnamon",
                                    typed="cinnamon", user_email=OWNER)
        )
        await asyncio.wait_for(graph.started.wait(), timeout=5)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            joined = asyncio.create_task(
                ac.get(f"/api/chat/turns/{started.request_id}/events?after=1", headers=auth())
            )
            await asyncio.sleep(0.3)  # the reader is attached and waiting
            graph.release.set()
            response = await asyncio.wait_for(joined, timeout=5)

        frames = _numbered(response.text)
        assert frames[0][0] == 2, "frame 1 was already seen and must not repeat"
        assert frames[-1][1] == "done"
        assert [seq for seq, _ in frames] == list(range(2, 2 + len(frames)))
    finally:
        deps_module.set_runtime(None)


# --- a disconnect is not a stop ------------------------------------------------------


async def test_a_signed_in_turn_outlives_a_reader_that_leaves(fake_store):
    """The reader's network drops mid-fan-out. The turn finishes and is stored,
    so the answer they were paying for is waiting when they come back."""
    graph = GatedGraph()
    runtime = _gated_runtime(graph)
    conversation_id = await fake_store.create(OWNER)
    started = turn_runner.start_turn(
        turn_runner.TurnRequest(runtime=runtime, query="cinnamon", typed="cinnamon",
                                user_email=OWNER, conversation_id=conversation_id)
    )
    reader = Disconnectable()
    stream = turn_runner.follow(started.request_id, 0, reader, cancel_on_disconnect=False)
    first = await anext(stream)
    assert "event: start" in first

    await asyncio.wait_for(graph.started.wait(), timeout=5)
    reader.gone = True
    with pytest.raises(StopAsyncIteration):
        while True:
            await anext(stream)

    assert not started.task.done(), "the turn was cancelled with its reader"
    graph.release.set()
    await _drain_turn(started)

    assert graph.completed and not graph.cancelled
    assert started.frames[-1].event == "done" and started.frames[-1].data["failed"] is False
    assert [m.role for m in fake_store.messages[conversation_id]] == ["user", "assistant"]


async def test_an_anonymous_turn_is_cancelled_when_its_reader_leaves(fake_store):
    """No owner, so no resume and nothing stored: finishing it would be spending
    money on an answer that nobody could ever read."""
    graph = GatedGraph()
    started = turn_runner.start_turn(
        turn_runner.TurnRequest(runtime=_gated_runtime(graph), query="cinnamon",
                                typed="cinnamon", user_email=None)
    )
    reader = Disconnectable()
    stream = turn_runner.follow(started.request_id, 0, reader, cancel_on_disconnect=True)
    await anext(stream)
    await asyncio.wait_for(graph.started.wait(), timeout=5)
    reader.gone = True
    with pytest.raises(StopAsyncIteration):
        while True:
            await anext(stream)

    await _drain_turn(started)
    assert graph.cancelled and not graph.completed, "the fan-out kept running"
    assert started.frames[-1].data.get("cancelled") is True


def test_only_an_anonymous_reader_leaving_cancels_its_turn(client, fake_store, monkeypatch):
    """The routes decide which turns die with their reader. Pinned here, because
    the runner tests above take the flag as given."""
    seen: list[bool] = []
    real_follow = turn_runner.follow

    def spy(request_id, after, http_request, *, cancel_on_disconnect=False):
        seen.append(cancel_on_disconnect)
        return real_follow(request_id, after, http_request,
                           cancel_on_disconnect=cancel_on_disconnect)

    monkeypatch.setattr(turn_runner, "follow", spy)
    _stream_turn(client, auth())
    _stream_turn(client)
    assert seen == [False, True]


# --- a discussion streams while it is being written ------------------------------------


class StallingChatLLM(ScriptedChatLLM):
    """A chat model that writes one sentence, then stops until released.

    The probe that found the defect: with the model stalled after its first
    sentence, the analyse path had already published that sentence and the
    discuss path had published nothing, because it waited for the whole reply
    before flushing. The gate needs the start of the next sentence to know the
    first one is complete, so the first chunk carries it.
    """

    def __init__(self, script, first="Exports reached USD 4.2m. That", rest=" was all."):
        super().__init__(script)
        self.first, self.rest = first, rest
        self.fed_first = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, role, system, user, *, json_mode=False, stream=None, **kwargs):
        if role != "chat" or stream is None:
            return await super().generate(role, system, user, json_mode=json_mode, **kwargs)
        stream.restart()
        stream.feed(self.first)
        self.fed_first.set()
        await self.release.wait()
        stream.feed(self.rest)
        return self.first + self.rest


async def _seed_prior_exchange(fake_store) -> int:
    conversation_id = await fake_store.create(OWNER)
    await fake_store.append(conversation_id, OWNER, [
        Message(role="user", content="cinnamon exports"),
        Message(role="assistant", content="Exports reached USD 4.2m.", mode="analyse",
                evidence=[{"source_id": "KG", "claim": "USD 4.2m in 2024", "detail": ""}]),
    ])
    return conversation_id


async def _wait_for_frame(started: turn_log.LocalTurn, event: str, timeout_s: float = 2.0):
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if any(f.event == event for f in started.frames):
            return True
        await asyncio.sleep(0.02)
    return False


async def test_a_discussions_first_sentence_reaches_the_wire_before_its_last(fake_store):
    """The sentence gate releases a grounded sentence as soon as it is complete.
    That is worth nothing unless the runner publishes it then, rather than after
    the whole reply has arrived — which is what it did until this test."""
    llm = StallingChatLLM({"turn_classify": json.dumps({"mode": "discuss"})})
    runtime = deps_module.Runtime(kg=FakeKG(), llm=llm, deps=None, graph=FakeGraph(ANSWERED))
    conversation_id = await _seed_prior_exchange(fake_store)

    started = turn_runner.start_turn(
        turn_runner.TurnRequest(runtime=runtime, query="explain that", typed="explain that",
                                user_email=OWNER, conversation_id=conversation_id)
    )
    try:
        await asyncio.wait_for(llm.fed_first.wait(), timeout=5)
        assert await _wait_for_frame(started, "answer_delta"), (
            "the first sentence was released by the gate but never published while the "
            "model was still writing"
        )
        assert not any(f.event == "done" for f in started.frames)
        shown = [f.data["text"] for f in started.frames if f.event == "answer_delta"]
        assert shown == ["Exports reached USD 4.2m. "]
    finally:
        llm.release.set()
    await _drain_turn(started)

    assert started.frames[-1].event == "done"
    assert started.frames[-1].data["answer"]["answer"] == "Exports reached USD 4.2m. That was all."
    assert started.frames[-1].data["answer"]["grounded"] is True


async def test_a_regenerated_discussion_streams_the_same_way(fake_store):
    """Regenerate takes the discuss path with the cache bypassed; it pumps too."""
    llm = StallingChatLLM({"turn_classify": json.dumps({"mode": "discuss"})})
    runtime = deps_module.Runtime(kg=FakeKG(), llm=llm, deps=None, graph=FakeGraph(ANSWERED))
    conversation_id = await _seed_prior_exchange(fake_store)
    await fake_store.append(conversation_id, OWNER, [
        Message(role="user", content="explain that"),
        Message(role="assistant", content="It went up.", mode="discuss"),
    ])
    transcript = await fake_store.messages_for(conversation_id, OWNER)
    plan = turn_runner.plan_regeneration(transcript, transcript[-1].id)
    assert plan is not None and plan.mode == "discuss"

    started = turn_runner.start_turn(
        turn_runner.TurnRequest(runtime=runtime, query=plan.question.content,
                                typed=plan.question.content, user_email=OWNER,
                                conversation_id=conversation_id, skip_clarify=True,
                                regenerate=plan)
    )
    try:
        await asyncio.wait_for(llm.fed_first.wait(), timeout=5)
        assert await _wait_for_frame(started, "answer_delta")
        assert not any(f.event == "done" for f in started.frames)
    finally:
        llm.release.set()
    await _drain_turn(started)
    assert started.frames[-1].data["regenerated_from"] == plan.target.id


# --- stop -----------------------------------------------------------------------------


async def test_stop_cancels_a_running_turn_and_stores_nothing(fake_store):
    graph = GatedGraph()
    deps_module.set_runtime(_gated_runtime(graph))
    owner = OWNER
    conversation_id = await fake_store.create(owner)
    try:
        started = turn_runner.start_turn(
            turn_runner.TurnRequest(runtime=deps_module.get_runtime(), query="cinnamon",
                                    typed="cinnamon", user_email=owner,
                                    conversation_id=conversation_id)
        )
        await asyncio.wait_for(graph.started.wait(), timeout=5)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(f"/api/chat/turns/{started.request_id}/cancel",
                                     headers=auth())
        assert response.status_code == 202
        assert response.json() == {"request_id": started.request_id, "cancelled": True,
                                   "where": "local"}

        await _drain_turn(started)
        assert graph.cancelled and not graph.completed
        assert started.frames[-1].event == "done"
        assert started.frames[-1].data["cancelled"] is True
        # The reader chose not to have this answer; a half-written turn in the
        # transcript would record something that did not happen.
        assert fake_store.messages[conversation_id] == []
    finally:
        deps_module.set_runtime(None)


def test_stopping_someone_elses_turn_is_not_found(client, fake_store):
    original = _stream_turn(client, auth()).text
    request_id = dict(parse_frames(original))["start"]["request_id"]
    response = client.post(f"/api/chat/turns/{request_id}/cancel",
                           headers=auth(OTHER, "researcher"))
    assert response.status_code == 404


def test_stopping_a_finished_turn_says_there_was_nothing_to_stop(client, fake_store):
    original = _stream_turn(client, auth()).text
    request_id = dict(parse_frames(original))["start"]["request_id"]
    response = client.post(f"/api/chat/turns/{request_id}/cancel", headers=auth())
    assert response.status_code == 202
    assert response.json()["cancelled"] is False


# --- the recorded §6 risk: cancellation mid-Neo4j-query --------------------------------


class _BlockingSession:
    """A neo4j session whose query never returns — until it is cancelled."""

    opened = 0
    closed = 0

    async def __aenter__(self):
        type(self).opened += 1
        return self

    async def __aexit__(self, *exc):
        type(self).closed += 1
        return False

    async def run(self, cypher, params):
        await asyncio.Event().wait()


class _BlockingDriver:
    def session(self, database=None):
        return _BlockingSession()


def _real_kg_over_a_blocking_driver():
    from ceynex.kg.client import KnowledgeGraphClient

    client = KnowledgeGraphClient.__new__(KnowledgeGraphClient)
    client._driver = _BlockingDriver()  # noqa: SLF001 - standing in for the pool
    client._database = None  # noqa: SLF001
    client._uri = "bolt://test"  # noqa: SLF001
    return client


async def test_stop_mid_neo4j_query_leaks_no_session_and_no_task(fake_store):
    """§6's open risk, closed: the real graph and the real KG client, cancelled
    from outside while every agent is blocked inside a Cypher query."""
    _BlockingSession.opened = _BlockingSession.closed = 0
    unhandled: list[dict] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    kg = _real_kg_over_a_blocking_driver()
    deps = AgentDeps(kg=kg, llm=FakeLLMClient(available=False),
                     dsn="postgresql://ceynex@127.0.0.1:1/nonexistent")
    graph = build_graph(deps, use_llm_router=False)
    runtime = deps_module.Runtime(kg=kg, llm=FakeLLMClient(available=False), deps=deps,
                                  graph=graph)
    before = {task for task in asyncio.all_tasks() if not task.done()}
    try:
        started = turn_runner.start_turn(
            turn_runner.TurnRequest(runtime=runtime, query="cinnamon exports to Germany",
                                    typed="cinnamon exports to Germany",
                                    user_email=OWNER)
        )
        for _ in range(100):  # until at least one agent is inside a query
            if _BlockingSession.opened:
                break
            await asyncio.sleep(0.02)
        assert _BlockingSession.opened, "no agent reached Neo4j; the test proves nothing"

        assert await turn_runner.cancel(started) is True
        await _drain_turn(started)
        await asyncio.sleep(0.05)
        gc.collect()

        assert _BlockingSession.closed == _BlockingSession.opened, "a Neo4j session leaked"
        leftover = {t for t in asyncio.all_tasks() if not t.done()} - before
        leftover.discard(asyncio.current_task())
        assert not leftover, f"tasks outlived the cancelled turn: {leftover}"
        assert not unhandled, f"an exception went unobserved: {unhandled}"
        assert started.frames[-1].data.get("cancelled") is True
    finally:
        loop.set_exception_handler(previous_handler)


# --- heartbeats ------------------------------------------------------------------------


async def test_a_quiet_turn_is_kept_alive_with_heartbeats(fake_store, monkeypatch):
    """nginx drops a connection silent for 60s. The comments that prevent it
    must appear while the graph is quiet — and a parser must ignore them."""
    monkeypatch.setattr(turn_runner, "HEARTBEAT_INTERVAL_S", 0.05)
    monkeypatch.setattr(turn_runner, "POLL_INTERVAL_S", 0.02)
    graph = GatedGraph()
    started = turn_runner.start_turn(
        turn_runner.TurnRequest(runtime=_gated_runtime(graph), query="cinnamon",
                                typed="cinnamon", user_email=OWNER)
    )
    reader = Disconnectable()
    chunks: list[str] = []

    async def read_all():
        async for chunk in turn_runner.follow(started.request_id, 0, reader):
            chunks.append(chunk)

    reading = asyncio.create_task(read_all())
    await asyncio.wait_for(graph.started.wait(), timeout=5)
    await asyncio.sleep(0.3)
    graph.release.set()
    await asyncio.wait_for(reading, timeout=5)

    body = "".join(chunks)
    assert body.count(": heartbeat") >= 2
    assert [name for name, _ in parse_frames(body)][-1] == "done"


# --- the Redis mirror, for a reader on the other worker ----------------------------------


class FakeRedis:
    """The handful of Redis commands the mirror uses, in memory."""

    def __init__(self):
        self.streams: dict[str, list[tuple[str, dict]]] = {}
        self.hashes: dict[str, dict] = {}
        self.values: dict[str, str] = {}

    async def hset(self, key, field=None, value=None, mapping=None):
        target = self.hashes.setdefault(key, {})
        if mapping:
            target.update(mapping)
        if field is not None:
            target[field] = value

    async def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    async def expire(self, key, seconds):
        return True

    async def xadd(self, key, fields, id, maxlen=None, approximate=True):  # noqa: A002
        self.streams.setdefault(key, []).append((id, dict(fields)))

    async def xread(self, streams, count=None, block=None):
        out = []
        for key, last in streams.items():
            after = int(last.split("-")[1])
            entries = [(i, f) for i, f in self.streams.get(key, []) if int(i.split("-")[1]) > after]
            if entries:
                out.append((key, entries[:count]))
        return out

    async def set(self, key, value, ex=None):
        self.values[key] = value

    async def get(self, key):
        return self.values.get(key)


def test_a_turn_is_mirrored_frame_for_frame(client, fake_store):
    redis = FakeRedis()
    turn_log.set_mirror(RedisTurnMirror(redis))
    original = _stream_turn(client, auth()).text
    request_id = dict(parse_frames(original))["start"]["request_id"]

    mirrored = redis.streams[f"ceynex:turn:{request_id}"]
    assert [int(entry_id.split("-")[1]) for entry_id, _ in mirrored] == [
        seq for seq, _ in _numbered(original)
    ]
    assert redis.hashes[f"ceynex:turn:{request_id}:meta"] == {
        "owner": OWNER, "status": "done"}


def test_a_reader_on_the_other_worker_resumes_from_the_mirror(client, fake_store):
    """The kernel, not us, decides which uvicorn worker accepts a reconnect. A
    worker that never ran the turn must still serve it, from the mirror."""
    redis = FakeRedis()
    turn_log.set_mirror(RedisTurnMirror(redis))
    original = _stream_turn(client, auth()).text
    request_id = dict(parse_frames(original))["start"]["request_id"]

    turn_log.set_registry(LocalTurnRegistry())  # "the other worker": no local copy
    resumed = client.get(f"/api/chat/turns/{request_id}/events?after=1", headers=auth())
    assert resumed.status_code == 200
    assert _numbered(resumed.text) == _numbered(original)[1:]

    stolen = client.get(f"/api/chat/turns/{request_id}/events", headers=auth(OTHER, "researcher"))
    assert stolen.status_code == 404


def test_stop_reaches_a_turn_on_the_other_worker_through_the_mirror(client, fake_store):
    redis = FakeRedis()
    turn_log.set_mirror(RedisTurnMirror(redis))
    redis.hashes["ceynex:turn:elsewhere:meta"] = {"owner": OWNER,
                                                  "status": "running"}
    response = client.post("/api/chat/turns/elsewhere/cancel", headers=auth())
    assert response.json() == {"request_id": "elsewhere", "cancelled": True, "where": "remote"}
    assert redis.values["ceynex:turn:elsewhere:cancel"] == "1"


def test_a_mirror_that_is_down_never_fails_the_turn(client, fake_store):
    """The replica is best-effort. Its failures are logged, and the answer the
    reader is watching arrives exactly as it would with no replica at all."""

    class DownRedis(FakeRedis):
        async def xadd(self, *args, **kwargs):
            raise ConnectionError("redis is down")

        async def hset(self, *args, **kwargs):
            raise ConnectionError("redis is down")

    turn_log.set_mirror(RedisTurnMirror(DownRedis()))
    frames = parse_frames(_stream_turn(client, auth()).text)
    assert frames[-1][0] == "done" and frames[-1][1]["failed"] is False


# --- the log itself ----------------------------------------------------------------------


def test_a_frame_is_one_sse_block_with_its_number():
    frame = TurnFrame(seq=7, event="kg_query", data={"cypher": "MATCH (n)\nRETURN n"})
    wire = frame.as_sse()
    assert wire.startswith("id: 7\nevent: kg_query\ndata: ")
    assert wire.endswith("\n\n") and wire.count("\n") == 4, "a raw newline escaped the JSON"
    assert json.loads(wire.split("data: ", 1)[1]) == {"cypher": "MATCH (n)\nRETURN n"}


async def test_finished_turns_expire_and_running_ones_never_do():
    registry = LocalTurnRegistry(ttl_s=0.0, max_turns=2)
    running = registry.start("running", owner="a", conversation_id=None)
    finished = registry.start("finished", owner="a", conversation_id=None)
    await finished.append("done", {"failed": False})
    await asyncio.sleep(0.01)

    assert registry.get("finished") is None, "a finished turn outlived its ttl"
    assert registry.get("running") is running, "a running turn was evicted"
