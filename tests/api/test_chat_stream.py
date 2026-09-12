"""Assertions for POST /api/chat/stream (deviation D12).

Two things are being pinned here, and the second is the one that matters most.

**The wire format**, because a browser's SSE parser is unforgiving: frames are
`event:`/`data:` pairs separated by a blank line, the stream always terminates in
a `done` frame, and a failure after the first byte is an in-band `error` frame
rather than an HTTP status — headers are long gone by then.

**That streaming changed nothing about the answer.** `/api/query` and
`/api/chat/stream` run one implementation (`api/query_runner.py`). The day they
diverge, `make eval` measures one system and users get another, and every number
in the report describes something that no longer exists.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ceynex.api import deps as deps_module
from ceynex.api.main import app
from tests.api.test_query import ANSWERED, FakeGraph, FakeKG, FakeLLM


def runtime(final=None, raises=None, kg=None):
    return deps_module.Runtime(
        kg=kg or FakeKG(), llm=FakeLLM(), deps=None, graph=FakeGraph(final, raises)
    )


@pytest.fixture
def client(request):
    final = getattr(request, "param", ANSWERED)
    deps_module.set_runtime(runtime(final))
    try:
        yield TestClient(app)
    finally:
        deps_module.set_runtime(None)


def parse_frames(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE body the way a browser would, so the test fails when a
    browser would fail rather than when a lenient split would."""
    frames = []
    for block in body.split("\n\n"):
        block = block.strip("\n")
        if not block or block.startswith(":"):
            continue  # heartbeat comment
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        assert event is not None, f"frame with no event name: {block!r}"
        frames.append((event, data or {}))
    return frames


def stream(client, query="cinnamon export trend"):
    response = client.post("/api/chat/stream", json={"query": query})
    return response, parse_frames(response.text)


# --- the wire format ------------------------------------------------------


def test_the_response_is_an_event_stream(client):
    response, _ = stream(client)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_nginx_is_told_not_to_buffer(client):
    """Without this header nginx holds the whole response until the connection
    closes, which is precisely the behaviour this endpoint exists to avoid — and
    it fails silently, looking like a slow backend rather than a proxy setting."""
    response, _ = stream(client)
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["cache-control"] == "no-cache"


def test_the_stream_opens_with_start_and_closes_with_done(client):
    _, frames = stream(client)
    assert frames[0][0] == "start"
    assert frames[-1][0] == "done"


def test_exactly_one_done_frame_is_sent(client):
    """A second `done` would leave a client that stops on the first one hanging
    on a socket that never closes."""
    _, frames = stream(client)
    assert [name for name, _ in frames].count("done") == 1


# --- the answer is the same answer ----------------------------------------


def test_the_done_frame_carries_the_same_answer_as_the_json_endpoint(client):
    """One orchestration path, two transports. This is the assertion that keeps
    it that way."""
    plain = client.post("/api/query", json={"query": "cinnamon export trend"}).json()
    _, frames = stream(client)
    streamed = dict(frames)["done"]["answer"]

    for field in ("answer", "confidence", "confidence_band", "agents_used",
                  "evidence", "forecast", "degraded", "route", "sectors", "unanswered"):
        assert streamed[field] == plain[field], f"{field} differs between transports"


def test_the_done_frame_reports_what_the_turn_spent(client):
    _, frames = stream(client)
    done = dict(frames)["done"]
    assert done["failed"] is False
    assert set(done["usage"]) == {"calls", "cache_hits", "tokens_in", "tokens_out", "cost_usd"}
    assert done["request_id"]


def test_dropped_events_are_reported_rather_than_hidden(client):
    """If backpressure ever discards events, the client is told. A trace that
    silently omits steps is the same defect as one that invents them."""
    _, frames = stream(client)
    assert dict(frames)["done"]["dropped_events"] == 0


# --- failure after the first byte -----------------------------------------


@pytest.mark.parametrize("client", [ANSWERED], indirect=True)
def test_a_graph_failure_becomes_an_error_frame_not_a_500(client):
    """Once headers are sent the status is already 200. A failure has to arrive
    in-band or the client sees a truncated success."""
    deps_module.set_runtime(runtime(raises=RuntimeError("neo4j exploded")))
    response, frames = stream(client)

    assert response.status_code == 200
    names = [name for name, _ in frames]
    assert "error" in names
    assert names[-1] == "done"
    assert dict(frames)["done"]["failed"] is True
    assert "neo4j exploded" in dict(frames)["error"]["message"]


def test_an_empty_query_is_rejected_before_the_stream_opens(client):
    """This one *can* be a status code: nothing has been written yet."""
    response = client.post("/api/chat/stream", json={"query": "   "})
    assert response.status_code == 422


def test_chat_can_be_switched_off_entirely(client, monkeypatch):
    """`CEYNEX_CHAT=off` must leave the deployment exactly as it was before this
    feature — the same discipline D10 applies to policy retrieval."""
    monkeypatch.setattr("ceynex.settings.chat_enabled", lambda: False)
    response = client.post("/api/chat/stream", json={"query": "cinnamon"})
    assert response.status_code == 404

    # ...and the endpoint it does not replace still answers.
    assert client.post("/api/query", json={"query": "cinnamon"}).status_code == 200


# --- the trace itself -----------------------------------------------------


def test_the_stream_reports_the_route_it_took(client):
    """The fake graph does not run the real router, so this asserts the frame
    plumbing rather than the routing; `tests/observability/` owns the content."""
    _, frames = stream(client)
    names = [name for name, _ in frames]
    assert names.count("start") == 1
    assert all(isinstance(data, dict) for _, data in frames)


def test_every_frame_is_a_single_data_line(client):
    """A raw newline inside `data:` splits one frame into two and corrupts every
    frame after it. `json.dumps` escapes them; this is the regression guard."""
    response, _ = stream(client)
    for block in response.text.split("\n\n"):
        data_lines = [line for line in block.splitlines() if line.startswith("data: ")]
        assert len(data_lines) <= 1, f"multi-line data payload: {block!r}"


def test_rate_limiting_applies_to_the_stream_too(client, monkeypatch):
    """`/api/chat/stream` runs the identical five-agent fan-out. Without the same
    dependency it would be a documented bypass around SRS 3.4.6 for the most
    expensive call in the system."""
    from ceynex.api.routes import chat as chat_routes

    seen: list[str] = []

    class Blocked:
        async def check(self, identity, limit, window_s):
            from ceynex.api.rate_limit import Decision

            seen.append(identity)
            return Decision(allowed=False, limit=limit, remaining=0, retry_after_s=30)

    chat_routes.set_chat_window(Blocked())
    try:
        response = client.post("/api/chat/stream", json={"query": "cinnamon"})
        assert response.status_code == 429
        assert response.headers["retry-after"] == "30"
        # Its own namespace, or a conversation silently spends the allowance for
        # asking a fresh question from the classic page.
        assert all(identity.startswith("chat:") for identity in seen), seen
    finally:
        chat_routes.set_chat_window(None)
