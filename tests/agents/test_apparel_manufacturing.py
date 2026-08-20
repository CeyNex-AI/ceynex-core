import pytest

from ceynex.agents.apparel_manufacturing import apparel_manufacturing_node
from ceynex.agents.common import AgentDeps
from ceynex.contracts.state import new_state
from ceynex.kg.client import KnowledgeGraphClient
from ceynex.llm import FakeLLMClient


class FakeKGClient:
    def __init__(self, edb_rows=None, jaaf_rows=None, overview_rows=None, raise_exc=None):
        self.edb_rows = edb_rows if edb_rows is not None else []
        self.jaaf_rows = jaaf_rows if jaaf_rows is not None else []
        self.overview_rows = overview_rows if overview_rows is not None else []
        self.raise_exc = raise_exc

    async def run(self, cypher, params=None):
        if self.raise_exc:
            raise self.raise_exc
        if "date.truncate" in cypher:
            return self.jaaf_rows, cypher
        if "latest_period" in cypher:
            return self.overview_rows, cypher
        return self.edb_rows, cypher


def _deps(kg=None, llm=None):
    return AgentDeps(kg=kg or FakeKGClient(), llm=llm or FakeLLMClient())


_EDB_ROWS = [
    {"period": "2023-01-01", "value": 1782720000.0, "product_name": "APPREL"},
    {"period": "2022-01-01", "value": 2300240000.0, "product_name": "APPREL"},
]
# Real EDB Apparel sub-category exports to the USA -- enough years (>=
# MIN_OBSERVATIONS_FOR_FORECAST) to exercise the naive forecast path.
_EDB_ROWS_FORECASTABLE = [
    {"period": "2024-01-01", "value": 1875850000.0, "product_name": "APPREL"},
    {"period": "2023-01-01", "value": 1782720000.0, "product_name": "APPREL"},
    {"period": "2022-01-01", "value": 2300240000.0, "product_name": "APPREL"},
    {"period": "2021-01-01", "value": 2082600000.0, "product_name": "APPREL"},
    {"period": "2020-01-01", "value": 1649270000.0, "product_name": "APPREL"},
]
_JAAF_ROWS = [{"year": "2025-01-01", "total": 1947370000.0, "latest_month": "2025-05-01"}]
_OVERVIEW_ROWS = [
    {"partner": "USA", "value": 1782720000.0, "period": "2023-01-01"},
    {"partner": "GBR", "value": 614620000.0, "period": "2023-01-01"},
]


async def test_happy_path_partner_query_validates_against_agent_output():
    deps = _deps(kg=FakeKGClient(edb_rows=_EDB_ROWS, jaaf_rows=_JAAF_ROWS))

    state = new_state(query="How are apparel exports to the United States doing?", user_id="u1")
    result = await apparel_manufacturing_node(state, deps)
    output = result["agent_outputs"]["apparel_manufacturing"]

    assert output["agent"] == "apparel_manufacturing"
    assert isinstance(output["summary"], str) and output["summary"]
    assert isinstance(output["figures"], dict) and output["figures"]
    assert 0.0 < output["confidence"] <= 0.95
    assert output["degraded"] is False
    assert "error" not in output
    assert len(output["evidence"]) >= 2
    assert {e["source_id"] for e in output["evidence"]} == {"EDB", "JAAF"}
    assert output["summary"] == "A canned explanation."
    # Only 2 years of EDB history here -- below MIN_OBSERVATIONS_FOR_FORECAST,
    # so no forecast should be attempted (see test below for the >=3-year case).
    assert "forecast" not in output


async def test_forecast_added_when_enough_edb_history():
    deps = _deps(kg=FakeKGClient(edb_rows=_EDB_ROWS_FORECASTABLE))

    state = new_state(query="apparel exports to the United States", user_id="u1")
    result = await apparel_manufacturing_node(state, deps)
    output = result["agent_outputs"]["apparel_manufacturing"]

    assert "forecast" in output
    assert len(output["forecast"]) == 2  # agent_module._FORECAST_HORIZON
    for point in output["forecast"]:
        assert point["lower"] <= point["point"] <= point["upper"]
        assert point["unit"] == "USD"
    # Naive baseline == last observed year's value (2024), flat across horizon.
    assert output["forecast"][0]["point"] == pytest.approx(1875850000.0)

    model_evidence = [e for e in output["evidence"] if e["source_id"] == "MODEL"]
    assert len(model_evidence) == 1
    assert "MAPE" in model_evidence[0]["claim"]
    assert any("naive" in a.lower() for a in output["assumptions"])


async def test_overview_query_when_no_partner_named():
    deps = _deps(kg=FakeKGClient(overview_rows=_OVERVIEW_ROWS))

    state = new_state(query="How are apparel exports doing overall?", user_id="u1")
    result = await apparel_manufacturing_node(state, deps)
    output = result["agent_outputs"]["apparel_manufacturing"]

    assert output["figures"] == {"USA": 1782720000.0, "GBR": 614620000.0}
    assert len(output["evidence"]) >= 1
    assert output["degraded"] is False


async def test_llm_failure_degrades_but_node_still_returns():
    # `generate_explanation` never raises (SRS 3.4.3) — a provider failure is
    # signalled by an empty string, which `FakeLLMClient(available=False)` gives.
    deps = _deps(
        kg=FakeKGClient(edb_rows=_EDB_ROWS, jaaf_rows=_JAAF_ROWS),
        llm=FakeLLMClient(available=False),
    )

    state = new_state(query="apparel exports to the United States", user_id="u1")
    result = await apparel_manufacturing_node(state, deps)
    output = result["agent_outputs"]["apparel_manufacturing"]

    assert output["degraded"] is True
    assert len(output["evidence"]) >= 2  # figures/evidence survive the LLM failure
    assert "error" not in output


async def test_kg_failure_sets_error_and_zero_confidence_without_raising():
    deps = _deps(kg=FakeKGClient(raise_exc=RuntimeError("neo4j unreachable")))

    state = new_state(query="apparel exports to the United States", user_id="u1")
    result = await apparel_manufacturing_node(state, deps)  # must not raise
    output = result["agent_outputs"]["apparel_manufacturing"]

    assert output["error"] == "neo4j unreachable"
    assert output["confidence"] == pytest.approx(0.0)
    assert output["degraded"] is True
    assert result["errors"] == ["neo4j unreachable"]


async def test_no_data_found_is_a_failed_output_not_a_crash():
    deps = _deps(kg=FakeKGClient())  # empty everywhere

    state = new_state(query="how are apparel exports doing overall", user_id="u1")
    result = await apparel_manufacturing_node(state, deps)
    output = result["agent_outputs"]["apparel_manufacturing"]

    assert output["confidence"] == pytest.approx(0.0)
    assert output["degraded"] is True
    assert "error" in output


@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known gap, PR #1 comment "
        "(github.com/CeyNex-AI/ceynex-core/pull/1#issuecomment-5345438470): "
        "there is no ceynex/kg/loaders/apparel.py yet, so nothing populates the "
        "frozen (:Commodity|:ApparelCategory)-[:EXPORTS_TO]->(:Country) shape "
        "for EDB/JAAF. This node's own Cypher still targets the *retired* "
        "(:Country)-[:REPORTED]->(:ExportRecord)-[:OF]->(:Product) shape, which "
        "nothing in the current codebase writes either -- kg/schema.py and its "
        "loader were deleted in the PR #1 merge-conflict resolution. Once the "
        "apparel loader exists AND this node's queries are updated to match the "
        "frozen schema, this test should start passing (and xfail(strict=True) "
        "will fail the suite until the marker above is removed, so the fix "
        "can't be missed)."
    ),
)
async def test_partner_query_against_the_real_merged_graph_finds_nothing_yet():
    """Turns the gap flagged on PR #1 into something the suite tracks automatically.

    Run against the actual local stack (`make up`, real EDB/JAAF data ingested,
    `make kg-load` applied) rather than FakeKGClient, so this reflects what the
    live deployed backend was independently observed doing on 2026-08-20: a real
    "apparel exports to the United States" query, live-verified via curl against
    http://35.200.228.142/api/query, came back with the apparel_manufacturing
    agent listed under `"unanswered"` even though PR #1 (EDB/JAAF connectors +
    this agent) was already merged to main.
    """
    async with KnowledgeGraphClient() as kg:
        deps = AgentDeps(kg=kg, llm=FakeLLMClient())
        state = new_state(query="How are apparel exports to the United States doing?", user_id="u1")
        result = await apparel_manufacturing_node(state, deps)
        output = result["agent_outputs"]["apparel_manufacturing"]

        assert "error" not in output
        assert output["evidence"], "expected real EDB/JAAF evidence once kg/loaders/apparel.py exists"
