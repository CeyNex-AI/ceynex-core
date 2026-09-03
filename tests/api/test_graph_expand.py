"""Assertions for GET /api/graph/expand (SRS 3.1.4, 3.1.6, 3.4.6).

No database. The endpoint's own job is small — validate a node id, run one
bounded hop, degrade quietly — so these are about the boundary rather than the
traversal, which `tests/kg/test_subgraph.py` owns.

The distinction the endpoint has to get right is between a caller bug and an
unavailable dependency: a malformed node id is a 422, an unreachable Neo4j is an
empty 200. Getting that backwards makes either a dead database look like the
user's fault or a typo look like an outage.
"""

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from ceynex.api import deps as deps_module
from ceynex.api.main import app
from ceynex.api.routes import graph as graph_route
from ceynex.kg.client import KnowledgeGraphUnavailableError


class FakeKG:
    def __init__(self, rows=None, raises=None):
        self._rows = rows if rows is not None else []
        self._raises = raises
        self.calls: list[tuple[str, dict]] = []

    async def run(self, cypher, params=None):
        self.calls.append((cypher, params or {}))
        if self._raises:
            raise self._raises
        return self._rows, cypher

    async def verify_connectivity(self):
        return True

    async def close(self):
        return None


class FakeLLM:
    available = False


NEIGHBOUR_ROWS = [
    {
        "source_label": "Commodity",
        "source_key": "tea",
        "source_name": "tea",
        "rel_type": "EXPORTS_TO",
        "rel_props": {"year": 2024},
        "target_label": "Country",
        "target_key": "USA",
        "target_name": "United States",
        "weight": 412_000_000.0,
    }
]


@contextmanager
def serving(kg):
    """A client over one specific fake, so a test can inspect what was asked."""
    deps_module.set_runtime(
        deps_module.Runtime(kg=kg, llm=FakeLLM(), deps=None, graph=None)
    )
    # The limiter is a process-lifetime singleton; without resetting it these
    # tests would spend — and eventually exhaust — whatever an earlier file left.
    graph_route.set_window(None)
    try:
        yield TestClient(app)
    finally:
        deps_module.set_runtime(None)
        graph_route.set_window(None)


@pytest.fixture
def client():
    with serving(FakeKG(NEIGHBOUR_ROWS)) as test_client:
        yield test_client


def expand(client, node="Country:USA"):
    return client.get("/api/graph/expand", params={"node": node})


# --- the happy path -------------------------------------------------------


def test_expanding_a_node_returns_its_neighbourhood(client):
    body = expand(client).json()
    assert {node["id"] for node in body["nodes"]} == {"Commodity:tea", "Country:USA"}
    assert body["edges"][0]["type"] == "EXPORTS_TO"
    assert body["edges"][0]["source"] == "Commodity:tea"


def test_the_fragment_shape_is_what_the_canvas_merges(client):
    """Deliberately not AnswerGraph: a fragment is added to a drawing that
    already has a focus, and reusing the answer shape would invite a caller to
    replace the drawing with it."""
    assert set(expand(client).json()) == {"nodes", "edges", "truncated"}


def test_the_key_reaches_neo4j_as_a_parameter():
    """The label is substituted into the query text; the value never is."""
    kg = FakeKG(NEIGHBOUR_ROWS)
    with serving(kg) as client:
        expand(client, "Country:USA")
    cypher, params = kg.calls[0]
    assert params["key"] == "USA"
    assert "(n:Country {iso3: $key})" in cypher


def test_the_hop_is_bounded():
    """One hop, with a limit, and no parameter to raise either."""
    kg = FakeKG(NEIGHBOUR_ROWS)
    with serving(kg) as client:
        expand(client, "Country:USA")
    cypher, params = kg.calls[0]
    assert "LIMIT $limit" in cypher
    assert params["limit"] > 0
    assert "*" not in cypher, "a variable-length pattern is not one hop"


def test_no_depth_can_be_requested(client):
    """An unbounded traversal wearing a number. FastAPI ignores the unknown
    query parameter rather than honouring it, which is the point."""
    response = client.get("/api/graph/expand", params={"node": "Country:USA", "depth": "5"})
    assert response.status_code == 200


# --- validation -----------------------------------------------------------


@pytest.mark.parametrize(
    "node",
    [
        "Country",  # no key
        "Country:",  # empty key
        "Bogus:x",  # not a label in this graph
        "User:admin",  # a label from a different system entirely
        "Country) MATCH (n) DETACH DELETE n //:x",  # structure, not a label
    ],
)
def test_a_node_id_that_is_not_in_the_graph_is_rejected(client, node):
    """422, not 500 and not an empty 200: the caller sent something wrong, and
    the allowlist is what keeps a query string from becoming query structure."""
    assert expand(client, node).status_code == 422


def test_a_rejected_node_id_never_reaches_neo4j():
    kg = FakeKG(NEIGHBOUR_ROWS)
    with serving(kg) as client:
        expand(client, "User:admin")
    assert kg.calls == []


def test_a_missing_node_parameter_is_rejected(client):
    assert client.get("/api/graph/expand").status_code == 422


# --- degradation ----------------------------------------------------------


def test_an_unreachable_graph_is_an_empty_200():
    """A 5xx would make a working answer page look broken over a click that was
    optional. The canvas keeps what it is already showing."""
    with serving(FakeKG(raises=KnowledgeGraphUnavailableError("neo4j down"))) as client:
        response = expand(client)
    assert response.status_code == 200
    assert response.json() == {"nodes": [], "edges": [], "truncated": False}


def test_a_node_with_no_neighbours_is_an_empty_200():
    with serving(FakeKG([])) as client:
        response = expand(client)
    assert response.status_code == 200
    assert response.json()["nodes"] == []


# --- rate limiting (SRS 3.4.6) -------------------------------------------


def test_expansions_are_limited_on_their_own_allowance(monkeypatch):
    """Prefixed `graph:` so exploring does not spend the allowance for asking.
    `rate_limit.KEY_PREFIX` is shared by every Window built from that module, so
    an unprefixed identity would write to `POST /api/query`'s own keys."""
    from ceynex import settings

    real_load = settings.load_config

    def tiny_limit(name):
        config = dict(real_load(name))
        if name == "api":
            config["graph_rate_limit"] = {
                "enabled": True,
                "expand_per_minute": 2,
                "window_seconds": 60,
            }
        return config

    monkeypatch.setattr(settings, "load_config", tiny_limit)

    with serving(FakeKG(NEIGHBOUR_ROWS)) as client:
        assert expand(client).status_code == 200
        assert expand(client).status_code == 200
        blocked = expand(client)
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers
