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
from ceynex.orchestrator.merger import NO_TOPIC_MARKER


class FakeGraph:
    def __init__(self, final: dict[str, Any] | None = None, raises: Exception | None = None):
        self._final = final or {}
        self._raises = raises

    async def ainvoke(self, state):
        if self._raises:
            raise self._raises
        return {**state, **self._final}


class FakeKG:
    """Answers the subgraph facets with one triple each.

    `run` exists because `POST /api/query` now draws the graph behind an answer
    (`_answer_graph`). It is deliberately not a full graph: these tests are about
    the endpoint's contract, and `tests/kg/test_subgraph.py` owns the assembly.
    """

    def __init__(self, raises: Exception | None = None):
        self._raises = raises

    async def verify_connectivity(self):
        return True

    async def run(self, cypher, params=None):
        if self._raises:
            raise self._raises
        if "EXPORTS_TO" in cypher:
            return [
                {
                    "source_label": "Commodity", "source_key": "cinnamon",
                    "source_name": "cinnamon", "rel_type": "EXPORTS_TO",
                    "rel_props": {"year": 2024}, "target_label": "Country",
                    "target_key": "DEU", "target_name": "Germany", "weight": 12.0,
                }
            ], cypher
        return [], cypher

    async def run_one(self, cypher, params=None):
        if self._raises:
            raise self._raises
        return {"latest_year": 2024}, cypher

    async def close(self):
        return None


class FakeLLM:
    available = False


def runtime(final=None, raises=None, kg=None):
    return deps_module.Runtime(
        kg=kg or FakeKG(), llm=FakeLLM(), deps=None, graph=FakeGraph(final, raises)
    )


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


NO_TOPIC = {
    "final_answer": "This question could not be answered from the data currently loaded. "
    "The question does not name anything CeyNex covers.",
    "final_confidence": 0.15,
    "merged_evidence": [],
    "degraded": True,
    "route": ["export_analytics"],
    "sectors": ["cross_sector"],
    "errors": ["out_of_scope: the question does not name anything CeyNex covers.", "out_of_scope_no_topic: true"],
    "agent_outputs": {
        "export_analytics": {
            "agent": "export_analytics", "summary": "Sri Lanka exported tea worth USD 1.4bn.",
            "figures": {"total_export_value_usd": 1_431_567_471.0}, "assumptions": [], "evidence": [],
            "confidence": 0.9, "degraded": False,
            "forecast": [{"period": "2025", "point": 10.0, "lower": 8.0, "upper": 12.0, "unit": "USD"}],
        },
    },
}


@pytest.mark.parametrize("client", [NO_TOPIC], indirect=True)
def test_a_no_topic_question_does_not_leak_the_routed_agents_real_answer(client):
    """Regression, found live 2026-08-27 from "whats 4+4": merge()'s own
    no-topic suppression correctly emptied `final_answer`/`merged_evidence`,
    but this endpoint separately recomputes `agents_used` and `unanswered`
    from the raw, unsuppressed `agent_outputs` -- so the response still said
    `agents_used: ["export_analytics"]` and listed a full, irrelevant tea
    report under `unanswered`, contradicting its own "could not be answered"
    answer. A forecast attached to that same irrelevant agent output must not
    leak through either.
    """
    body = post(client, query="whats 4+4").json()

    assert body["agents_used"] == []
    assert body["forecast"] is None
    assert body["unanswered"] == ["the question does not name anything CeyNex covers."]
    assert "1.4" not in " ".join(body["unanswered"])
    assert "tea" not in " ".join(body["unanswered"]).lower()


NO_FORECAST = {**ANSWERED, "agent_outputs": {
    "export_analytics": {"agent": "export_analytics", "summary": "s", "figures": {},
                         "assumptions": [], "evidence": [], "confidence": 0.8, "degraded": False},
}}


@pytest.mark.parametrize("client", [NO_FORECAST], indirect=True)
def test_a_query_without_a_forecast_returns_null_not_an_empty_list(client):
    assert post(client).json()["forecast"] is None


# --- the drawable graph (SRS 3.1.4) --------------------------------------


def test_a_kg_grounded_answer_carries_a_graph(client):
    graph = post(client).json()["graph"]
    assert graph is not None
    assert {node["id"] for node in graph["nodes"]} == {"Commodity:cinnamon", "Country:DEU"}
    assert graph["edges"][0]["type"] == "EXPORTS_TO"
    assert graph["focus_id"] == "Commodity:cinnamon"


def test_the_graph_carries_the_cypher_that_drew_it(client):
    """The picture gets the provenance the figures already have (SRS 3.1.4)."""
    graph = post(client).json()["graph"]
    assert graph["queries"]
    assert any("EXPORTS_TO" in text for text in graph["queries"])


SIMULATED = {
    **ANSWERED,
    "agent_outputs": {
        **ANSWERED["agent_outputs"],
        "trade_economics": {
            "agent": "trade_economics", "summary": "s", "assumptions": [], "evidence": [],
            "confidence": 0.8, "degraded": False,
            "figures": {"agriculture_impact_usd": -8589405.0, "agriculture_impact_pct": -0.006},
        },
    },
}


@pytest.mark.parametrize("client", [SIMULATED], indirect=True)
def test_a_simulated_answer_carries_its_result_on_the_graph(client):
    """A shock simulation's own working, not just its output in the evidence
    panel — `trade_economics` and the graph are independent agents that share
    a subject, and without this join a reader asking "how would a 6%
    depreciation affect cinnamon" saw the simulated change nowhere near the
    picture of the market it was simulated against.

    This is the regression case for the bug this fix's own first draft had:
    `merge()`'s `as_state_patch()` never writes a merged `figures` dict onto
    state, only `final_answer`/`final_confidence`/`merged_evidence` -- an
    earlier version of `_annotate_focus` read `final["figures"]`, which is
    always empty, and silently annotated nothing. This test posts through the
    real endpoint rather than calling `_annotate_focus` directly, so it fails
    the same way that bug did.
    """
    node = next(
        n for n in post(client).json()["graph"]["nodes"] if n["id"] == "Commodity:cinnamon"
    )
    assert node["properties"]["simulated_revenue_change_pct"] == pytest.approx(-0.6)
    assert node["properties"]["simulated_revenue_change_usd"] == pytest.approx(-8589405.0)


@pytest.mark.parametrize("client", [ANSWERED], indirect=True)
def test_an_unsimulated_answer_carries_no_simulation_properties(client):
    """No trade_economics figures in this fixture -- the annotation must be
    silent, not a KeyError, and must not invent properties on a node nobody
    simulated."""
    node = next(
        n for n in post(client).json()["graph"]["nodes"] if n["id"] == "Commodity:cinnamon"
    )
    assert "simulated_revenue_change_pct" not in node["properties"]


NO_KG_EVIDENCE = {
    **ANSWERED,
    "merged_evidence": [
        {"source_id": "MODEL", "claim": "The model projects growth.", "detail": "arima:tea:v3"}
    ],
}


@pytest.mark.parametrize("client", [NO_KG_EVIDENCE], indirect=True)
def test_an_answer_the_graph_did_not_produce_has_no_graph(client):
    """A node-link diagram is a claim about where a number came from. Beside a
    model-derived answer it would assert a provenance that isn't there — the
    same failure `orchestrator/grounding.py` catches in prose."""
    assert post(client).json()["graph"] is None


NO_TOPIC = {**ANSWERED, "errors": [NO_TOPIC_MARKER]}


@pytest.mark.parametrize("client", [NO_TOPIC], indirect=True)
def test_an_out_of_scope_question_has_no_graph(client):
    """Same suppression `forecast` already gets — a routed agent's graph is
    noise, not an answer, for a question that named nothing CeyNex covers.

    Asked with a query that *does* name an item, so this fails if the marker
    check is removed. A question naming nothing has no graph anyway, via the
    no-item path below, and would pass either way.
    """
    assert post(client, "cinnamon export trend").json()["graph"] is None


def test_a_question_naming_no_item_has_no_graph(client):
    """There is no subject to centre on, and a graph of everything is not an
    answer to anything."""
    assert post(client, "how are exports doing").json()["graph"] is None


def test_a_dead_graph_still_answers():
    """The illustration must never fail the answer. This is the assertion that
    keeps `_answer_graph`'s bare `except` honest."""
    from ceynex.kg.client import KnowledgeGraphUnavailableError

    deps_module.set_runtime(
        runtime(ANSWERED, kg=FakeKG(raises=KnowledgeGraphUnavailableError("neo4j down")))
    )
    try:
        response = TestClient(app).post("/api/query", json={"query": "cinnamon export trend"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Exports grew steadily."
        assert body["graph"] is None
    finally:
        deps_module.set_runtime(None)


def test_the_graph_year_follows_the_agents_own_precedence():
    """`AgentState` has nowhere to record the year an agent settled on, so the
    route re-derives it. That is only sound if it derives the same one: an
    explicit year in the question wins, and otherwise it is
    `latest_observation_year(item)` — scoped to the item, which is the
    precedence the 2026-08-26 bug in that query's docstring was about."""
    asked: list[str] = []

    class RecordingKG(FakeKG):
        async def run(self, cypher, params=None):
            if "EXPORTS_TO" in cypher:
                asked.append(str(params.get("year")))
            return await super().run(cypher, params)

    deps_module.set_runtime(runtime(ANSWERED, kg=RecordingKG()))
    try:
        client = TestClient(app)
        client.post("/api/query", json={"query": "cinnamon exports in 2019"})
        assert asked == ["2019"], "an explicit year in the question must win"

        asked.clear()
        client.post("/api/query", json={"query": "cinnamon export trend"})
        assert asked == ["2024"], "otherwise the item's own latest year"
    finally:
        deps_module.set_runtime(None)


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
