import pytest

from ceynex.agents import apparel_manufacturing as agent_module
from ceynex.agents.apparel_manufacturing import apparel_manufacturing_node
from ceynex.contracts.state import new_state


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


class FakeLLMClient:
    def __init__(self, raise_exc=None, text="Explained by LLM."):
        self.raise_exc = raise_exc
        self.text = text

    async def generate_explanation(self, context):
        if self.raise_exc:
            raise self.raise_exc
        return self.text


_EDB_ROWS = [
    {"period": "2023-01-01", "value": 1782720000.0, "product_name": "APPREL"},
    {"period": "2022-01-01", "value": 2300240000.0, "product_name": "APPREL"},
]
_JAAF_ROWS = [{"year": "2025-01-01", "total": 1947370000.0, "latest_month": "2025-05-01"}]
_OVERVIEW_ROWS = [
    {"partner": "USA", "value": 1782720000.0, "period": "2023-01-01"},
    {"partner": "GBR", "value": 614620000.0, "period": "2023-01-01"},
]


async def test_happy_path_partner_query_validates_against_agent_output(monkeypatch):
    monkeypatch.setattr(
        agent_module,
        "_get_kg_client",
        lambda: FakeKGClient(edb_rows=_EDB_ROWS, jaaf_rows=_JAAF_ROWS),
    )
    monkeypatch.setattr(agent_module, "_get_llm_client", lambda: FakeLLMClient())

    state = new_state(query="How are apparel exports to the United States doing?", user_id="u1")
    result = await apparel_manufacturing_node(state)
    output = result["agent_outputs"]["apparel_manufacturing"]

    assert output["agent"] == "apparel_manufacturing"
    assert isinstance(output["summary"], str) and output["summary"]
    assert isinstance(output["figures"], dict) and output["figures"]
    assert 0.0 < output["confidence"] <= 0.95
    assert output["degraded"] is False
    assert "error" not in output
    assert len(output["evidence"]) >= 2
    assert {e["source_id"] for e in output["evidence"]} == {"EDB", "JAAF"}
    assert output["summary"] == "Explained by LLM."


async def test_overview_query_when_no_partner_named(monkeypatch):
    monkeypatch.setattr(
        agent_module, "_get_kg_client", lambda: FakeKGClient(overview_rows=_OVERVIEW_ROWS)
    )
    monkeypatch.setattr(agent_module, "_get_llm_client", lambda: FakeLLMClient())

    state = new_state(query="How are apparel exports doing overall?", user_id="u1")
    result = await apparel_manufacturing_node(state)
    output = result["agent_outputs"]["apparel_manufacturing"]

    assert output["figures"] == {"USA": 1782720000.0, "GBR": 614620000.0}
    assert len(output["evidence"]) >= 1
    assert output["degraded"] is False


async def test_llm_failure_degrades_but_node_still_returns(monkeypatch):
    monkeypatch.setattr(
        agent_module,
        "_get_kg_client",
        lambda: FakeKGClient(edb_rows=_EDB_ROWS, jaaf_rows=_JAAF_ROWS),
    )
    monkeypatch.setattr(
        agent_module, "_get_llm_client", lambda: FakeLLMClient(raise_exc=RuntimeError("no quota"))
    )

    state = new_state(query="apparel exports to the United States", user_id="u1")
    result = await apparel_manufacturing_node(state)
    output = result["agent_outputs"]["apparel_manufacturing"]

    assert output["degraded"] is True
    assert len(output["evidence"]) >= 2  # figures/evidence survive the LLM failure
    assert "error" not in output


async def test_missing_llm_client_degrades_by_default(monkeypatch):
    monkeypatch.setattr(
        agent_module,
        "_get_kg_client",
        lambda: FakeKGClient(edb_rows=_EDB_ROWS, jaaf_rows=_JAAF_ROWS),
    )
    monkeypatch.setattr(agent_module, "_get_llm_client", lambda: None)

    state = new_state(query="apparel exports to the United States", user_id="u1")
    result = await apparel_manufacturing_node(state)
    output = result["agent_outputs"]["apparel_manufacturing"]

    assert output["degraded"] is True


async def test_kg_failure_sets_error_and_zero_confidence_without_raising(monkeypatch):
    monkeypatch.setattr(
        agent_module,
        "_get_kg_client",
        lambda: FakeKGClient(raise_exc=RuntimeError("neo4j unreachable")),
    )
    monkeypatch.setattr(agent_module, "_get_llm_client", lambda: FakeLLMClient())

    state = new_state(query="apparel exports to the United States", user_id="u1")
    result = await apparel_manufacturing_node(state)  # must not raise
    output = result["agent_outputs"]["apparel_manufacturing"]

    assert output["error"] == "neo4j unreachable"
    assert output["confidence"] == pytest.approx(0.0)
    assert output["degraded"] is True
    assert result["errors"] == ["neo4j unreachable"]


async def test_no_data_found_is_a_failed_output_not_a_crash(monkeypatch):
    monkeypatch.setattr(agent_module, "_get_kg_client", lambda: FakeKGClient())  # empty everywhere
    monkeypatch.setattr(agent_module, "_get_llm_client", lambda: FakeLLMClient())

    state = new_state(query="how are apparel exports doing overall", user_id="u1")
    result = await apparel_manufacturing_node(state)
    output = result["agent_outputs"]["apparel_manufacturing"]

    assert output["confidence"] == pytest.approx(0.0)
    assert output["degraded"] is True
    assert "error" in output
