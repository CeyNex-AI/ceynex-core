"""Assertions for query history (SRS 3.5.2): recording via POST /api/query and
listing via GET /api/history.

No real Postgres here — `ceynex.api.history`'s `record()`/`list_for_user()` are
monkeypatched, matching `test_query.py`'s "no network and no database" style.
The graph is faked the same way `test_query.py` fakes it, since a query has to
go through `POST /api/query` first to produce something worth recording.
`tests/api/test_history_integration.py` (docker-marked) proves the real
Postgres round trip.
"""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from ceynex.api import deps as deps_module
from ceynex.api import history as history_module
from ceynex.api.auth import issue_token
from ceynex.api.main import app


class FakeGraph:
    def __init__(self, final: dict[str, Any]):
        self._final = final

    async def ainvoke(self, state):
        return {**state, **self._final}


class FakeKG:
    async def verify_connectivity(self):
        return True

    async def close(self):
        return None


class FakeLLM:
    available = False


ANSWERED = {
    "final_answer": "Exports grew steadily.",
    "final_confidence": 0.72,
    "merged_evidence": [],
    "route": [],
    "sectors": [],
    "degraded": False,
    "agent_outputs": {},
}


@pytest.fixture
def client():
    deps_module.set_runtime(
        deps_module.Runtime(kg=FakeKG(), llm=FakeLLM(), deps=None, graph=FakeGraph(ANSWERED))
    )
    try:
        # No `with` block: see test_query.py — avoids running the real lifespan.
        yield TestClient(app)
    finally:
        deps_module.set_runtime(None)


@pytest.fixture
def recorded(monkeypatch):
    """Captures history.record() calls instead of hitting Postgres."""
    calls: list[dict] = []
    monkeypatch.setattr(history_module, "record", lambda **kwargs: calls.append(kwargs))
    return calls


def token_for(email="researcher@ceynex.dev", role="researcher"):
    return issue_token(email, role)


# --- recording via POST /api/query -----------------------------------------


def test_an_authenticated_query_is_recorded(client, recorded):
    client.post(
        "/api/query",
        json={"query": "cinnamon export trend"},
        headers={"Authorization": f"Bearer {token_for()}"},
    )
    assert len(recorded) == 1
    assert recorded[0]["user_email"] == "researcher@ceynex.dev"
    assert recorded[0]["query"] == "cinnamon export trend"
    assert recorded[0]["answer"] == "Exports grew steadily."
    assert recorded[0]["confidence"] == pytest.approx(0.72)
    assert recorded[0]["degraded"] is False


def test_an_anonymous_query_is_not_recorded(client, recorded):
    client.post("/api/query", json={"query": "cinnamon export trend"})
    assert recorded == []


def test_an_invalid_token_is_treated_as_anonymous_not_rejected(client, recorded):
    """POST /api/query stays open to anyone — an expired/garbage token just
    means the query isn't attributed to anyone, it does not fail the request."""
    response = client.post(
        "/api/query",
        json={"query": "cinnamon export trend"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert response.status_code == 200
    assert recorded == []


# --- GET /api/history --------------------------------------------------


def test_get_history_returns_the_signed_in_users_entries(client, monkeypatch):
    fake_entries = [
        history_module.HistoryEntry(
            id=1,
            query="q1",
            answer="a1",
            confidence=0.5,
            degraded=False,
            asked_at="2026-08-21T00:00:00+00:00",
            saved=False,
        )
    ]
    monkeypatch.setattr(
        history_module, "list_for_user", lambda user_email, limit=20, saved=None: fake_entries
    )

    response = client.get("/api/history", headers={"Authorization": f"Bearer {token_for()}"})

    assert response.status_code == 200
    assert response.json()["items"] == [
        {
            "id": 1,
            "query": "q1",
            "answer": "a1",
            "confidence": 0.5,
            "degraded": False,
            "asked_at": "2026-08-21T00:00:00+00:00",
            "saved": False,
        }
    ]


def test_get_history_passes_the_saved_filter_through(client, monkeypatch):
    captured = {}

    def fake_list(user_email, limit=20, saved=None):
        captured["saved"] = saved
        return []

    monkeypatch.setattr(history_module, "list_for_user", fake_list)

    client.get("/api/history?saved=true", headers={"Authorization": f"Bearer {token_for()}"})
    assert captured["saved"] is True


def test_get_history_without_a_token_is_rejected(client):
    assert client.get("/api/history").status_code == 401


def test_get_history_surfaces_a_real_postgres_outage_as_503(client, monkeypatch):
    def boom(user_email, limit=20, saved=None):
        raise psycopg.OperationalError("db down")

    monkeypatch.setattr(history_module, "list_for_user", boom)

    response = client.get("/api/history", headers={"Authorization": f"Bearer {token_for()}"})
    assert response.status_code == 503


# --- save / unsave -------------------------------------------------------


def test_saving_an_owned_entry_marks_it_saved(client, monkeypatch):
    monkeypatch.setattr(history_module, "set_saved", lambda entry_id, user_email, saved: True)

    response = client.post(
        "/api/history/1/save", headers={"Authorization": f"Bearer {token_for()}"}
    )

    assert response.status_code == 200
    assert response.json() == {"id": 1, "saved": True}


def test_unsaving_an_owned_entry_marks_it_unsaved(client, monkeypatch):
    monkeypatch.setattr(history_module, "set_saved", lambda entry_id, user_email, saved: True)

    response = client.post(
        "/api/history/1/unsave", headers={"Authorization": f"Bearer {token_for()}"}
    )

    assert response.status_code == 200
    assert response.json() == {"id": 1, "saved": False}


def test_saving_someone_elses_or_a_nonexistent_entry_is_a_404(client, monkeypatch):
    """set_saved's ownership check happens inside the UPDATE itself (see
    ceynex/api/history.py) -- "not yours" and "doesn't exist" both come back
    as False from there, and both must look identical from the outside."""
    monkeypatch.setattr(history_module, "set_saved", lambda entry_id, user_email, saved: False)

    response = client.post(
        "/api/history/999/save", headers={"Authorization": f"Bearer {token_for()}"}
    )
    assert response.status_code == 404


def test_saving_without_a_token_is_rejected(client):
    assert client.post("/api/history/1/save").status_code == 401


def test_saving_during_a_postgres_outage_is_a_503(client, monkeypatch):
    def boom(entry_id, user_email, saved):
        raise psycopg.OperationalError("db down")

    monkeypatch.setattr(history_module, "set_saved", boom)

    response = client.post(
        "/api/history/1/save", headers={"Authorization": f"Bearer {token_for()}"}
    )
    assert response.status_code == 503
