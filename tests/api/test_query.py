"""Assertions for the REST surface (SRS 3.9.3, team overview §4.5).

`QueryResponse`'s field names are what M3's web application binds to, so they are
frozen in practice — JSON has no type checker on the wire, and a rename breaks the
frontend silently. These tests are what notices.

No network and no database: the runtime is replaced with a fake graph.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from ceynex.api import deps as deps_module
from ceynex.api.main import app


class FakeGraph:
    def __init__(self, final: dict[str, Any] | None = None, raises: Exception | None = None):
        self._final = final or {}
        self._raises = raises

    async def ainvoke(self, state):
        if self._raises:
            raise self._raises
        return {**state, **self._final}


class FakeKG:
    async def verify_connectivity(self):
        return True

    async def close(self):
        return None


class FakeLLM:
    available = False


def runtime(final=None, raises=None):
    return deps_module.Runtime(kg=FakeKG(), llm=FakeLLM(), deps=None, graph=FakeGraph(final, raises))


ANSWERED = {
    "final_answer": "Exports grew steadily.",
    "final_confidence": 0.72,
    "merged_evidence": [
        {"source_id": "KG", "claim": "Germany took 12%.", "detail": "MATCH (n) RETURN n", "period": "2024"}
    ],
    "route": ["export_analytics", "forecast"],
    "sectors": ["agriculture"],
    "degraded": False,
    "agent_outputs": {
        "export_analytics": {"agent": "export_analytics", "summary": "s", "figures": {},
                             "assumptions": [], "evidence": [], "confidence": 0.8, "degraded": False},
        "forecast": {
            "agent": "forecast", "summary": "s", "figures": {}, "assumptions": [], "evidence": [],
            "confidence": 0.7, "degraded": False,
            "forecast": [{"period": "2025", "point": 10.0, "lower": 8.0, "upper": 12.0, "unit": "USD"}],
        },
    },
}


@pytest.fixture
def client(request):
    """A client whose runtime is a fake, with the real lifespan bypassed."""
    final = getattr(request, "param", ANSWERED)
    deps_module.set_runtime(runtime(final))
    try:
        # No `with` block: entering TestClient's context would run the app
        # lifespan, which builds a real Neo4j pool.
        yield TestClient(app)
    finally:
        deps_module.set_runtime(None)


def post(client, query="cinnamon export trend"):
    return client.post("/api/query", json={"query": query})


# --- the response contract ------------------------------------------------


def test_a_query_returns_every_contracted_field(client):
    body = post(client).json()
    for field in (
        "answer", "confidence", "confidence_band", "agents_used",
        "evidence", "forecast", "degraded", "elapsed_ms",
    ):
        assert field in body, f"{field} missing — M3's frontend binds to this"


def test_the_answer_and_confidence_come_through(client):
    body = post(client).json()
    assert body["answer"] == "Exports grew steadily."
    assert body["confidence"] == pytest.approx(0.72)
    assert body["confidence_band"] == "Moderate"


def test_evidence_keeps_its_source_and_detail(client):
    """SRS 3.1.4 — the evidence panel renders exactly these fields."""
    item = post(client).json()["evidence"][0]
    assert item["source_id"] == "KG"
    assert item["detail"].startswith("MATCH")
    assert item["period"] == "2024"


def test_a_forecast_carries_its_interval(client):
    """SRS 3.1.3 forbids an unqualified number reaching the user."""
    point = post(client).json()["forecast"][0]
    assert point["lower"] < point["point"] < point["upper"]
    assert point["unit"] == "USD"


def test_the_route_is_exposed_for_the_demo(client):
    body = post(client).json()
    assert body["route"] == ["export_analytics", "forecast"]
    assert body["sectors"] == ["agriculture"]


def test_elapsed_ms_is_reported(client):
    assert post(client).json()["elapsed_ms"] >= 0


# --- partial results and degradation -------------------------------------


PARTIAL = {
    **ANSWERED,
    "degraded": True,
    "agent_outputs": {
        "export_analytics": {"agent": "export_analytics", "summary": "s", "figures": {},
                             "assumptions": [], "evidence": [], "confidence": 0.8, "degraded": True},
        "forecast": {"agent": "forecast", "summary": "s", "figures": {}, "assumptions": [],
                     "evidence": [], "confidence": 0.0, "degraded": True, "error": "no model"},
    },
}


@pytest.mark.parametrize("client", [PARTIAL], indirect=True)
def test_a_failed_agent_is_reported_not_hidden(client):
    """SAD §4.1 — the user is told which part could not be answered.

    `unanswered` is a human-readable reason, not the bare agent name (the
    same list merger.unanswered_from_outputs computes internally for the
    prose) -- the field is additive to §4.5, not frozen there, and no real
    consumer (ceynex-web) parses its contents; only its presence as
    list[str] matters for the wire contract.
    """
    body = post(client).json()
    assert body["agents_used"] == ["export_analytics"]
    assert len(body["unanswered"]) == 1
    assert "could not be covered" in body["unanswered"][0]
    assert "no model" in body["unanswered"][0]
    assert body["degraded"] is True
    assert body["answer"], "a partial result is still an answer"


NO_FORECAST = {**ANSWERED, "agent_outputs": {
    "export_analytics": {"agent": "export_analytics", "summary": "s", "figures": {},
                         "assumptions": [], "evidence": [], "confidence": 0.8, "degraded": False},
}}


@pytest.mark.parametrize("client", [NO_FORECAST], indirect=True)
def test_a_query_without_a_forecast_returns_null_not_an_empty_list(client):
    assert post(client).json()["forecast"] is None


# --- validation and failure ----------------------------------------------


def test_an_empty_query_is_rejected(client):
    assert client.post("/api/query", json={"query": ""}).status_code == 422


def test_a_whitespace_query_is_rejected(client):
    assert client.post("/api/query", json={"query": "   "}).status_code == 422


def test_a_missing_body_is_rejected(client):
    assert client.post("/api/query", json={}).status_code == 422


def test_an_over_long_query_is_rejected(client):
    assert client.post("/api/query", json={"query": "x" * 5000}).status_code == 422


def test_an_orchestration_failure_becomes_a_500_with_a_reason():
    deps_module.set_runtime(runtime(raises=RuntimeError("graph exploded")))
    try:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/api/query", json={"query": "anything"}
        )
        assert response.status_code == 500
        assert "graph exploded" in response.json()["detail"]
    finally:
        deps_module.set_runtime(None)


# --- health ---------------------------------------------------------------


def test_health_reports_each_dependency_separately(client):
    body = client.get("/health").json()
    assert set(body) >= {"status", "neo4j", "postgres", "llm"}


def test_a_missing_llm_key_does_not_make_the_service_unhealthy(client):
    """SRS 3.4.3 — degrading is designed behaviour, not a reason to restart."""
    body = client.get("/health").json()
    assert body["llm"] is False
    assert body["status"] in {"ok", "degraded"}


def test_the_runtime_must_be_initialised():
    deps_module.set_runtime(None)
    with pytest.raises(RuntimeError, match="lifespan"):
        deps_module.get_runtime()
