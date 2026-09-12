"""The scenario workbench route (deviation D17, SRS 3.1.5).

Three things matter, in this order: the numbers are the agent's numbers (one
formula, `models/shocks.py`); the route refuses the way the agent refuses when
the graph cannot support a simulation (SAD §4.1); and every parameter arrives
with its provenance, `TBD` included, because the page shows it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ceynex.api import deps as deps_module
from ceynex.api.auth import issue_token
from ceynex.api.main import app
from ceynex.api.routes import scenario as scenario_routes
from ceynex.models import shocks
from ceynex.settings import elasticity_config
from tests.api.test_query import FakeLLM

USER = "policymaker@ceynex.dev"

COVERAGE = [
    {"agreement": "GSP+", "agreement_type": "unilateral_preference", "matched_on": "61",
     "agreement_verified": "verified"},
]


def auth(email: str = USER, role: str = "policymaker") -> dict[str, str]:
    token = issue_token(email, role)
    return {"Authorization": f"Bearer {token}"}


class ScenarioKG:
    """Answers the three reads the route makes, and records them."""

    def __init__(self, *, baseline: float | None = 1_000_000.0, coverage=None,
                 raises: Exception | None = None):
        self.baseline = baseline
        self.coverage = COVERAGE if coverage is None else coverage
        self.raises = raises
        self.queries: list[str] = []

    async def run(self, cypher, params=None):
        if self.raises:
            raise self.raises
        self.queries.append(cypher)
        if "max(e.year)" in cypher:
            return [{"latest_year": 2024}], cypher
        if "COVERED_BY" in cypher:
            return list(self.coverage), cypher
        if "sum(e.value)" in cypher:
            if self.baseline is None:
                return [], cypher
            return [{"total_export_value_usd": self.baseline}], cypher
        return [], cypher


@pytest.fixture
def kg():
    return ScenarioKG()


@pytest.fixture
def client(kg):
    deps_module.set_runtime(deps_module.Runtime(kg=kg, llm=FakeLLM(), deps=None, graph=None))
    try:
        yield TestClient(app)
    finally:
        deps_module.set_runtime(None)


def run(client, body, headers=None):
    return client.post(
        "/api/scenario/run", json=body, headers=headers if headers is not None else auth()
    )


# --- the numbers are the agent's ----------------------------------------------


def test_an_fx_run_reports_the_shared_formulas_figure(client):
    body = run(client, {"shock": "fx", "sector": "agriculture", "magnitude": 0.05}).json()
    expected = shocks.fx_shock("agriculture", 1_000_000.0, 0.05, elasticity_config())
    assert body["refused"] is False
    assert body["item"] == "tea"
    assert body["baseline_usd"] == 1_000_000.0 and body["baseline_year"] == 2024
    assert body["outcome"]["revenue_change_usd"] == pytest.approx(expected.revenue_change_usd)
    assert body["outcome"]["detail"] == expected.detail
    assert body["baseline_cypher"].startswith("MATCH")


def test_a_tariff_run_takes_its_rate_from_the_magnitude(client):
    body = run(client, {"shock": "tariff", "sector": "apparel", "magnitude": 0.10}).json()
    expected = shocks.tariff_shock("apparel", 1_000_000.0, 0.10, elasticity_config())
    assert body["outcome"]["revenue_change_pct"] == pytest.approx(expected.revenue_change_pct)


def test_an_agreement_run_uses_the_graphs_coverage_and_cites_it(client, kg):
    body = run(client, {"shock": "agreement", "sector": "apparel"}).json()
    expected = shocks.agreement_loss_shock(
        "apparel", 1_000_000.0, elasticity_config(), coverage=shocks.describe_coverage(COVERAGE)
    )
    assert body["refused"] is False
    assert body["outcome"]["revenue_change_usd"] == pytest.approx(expected.revenue_change_usd)
    assert body["outcome"]["detail"] == expected.detail
    assert any("COVERED_BY" in query for query in kg.queries)
    assert [e["source_id"] for e in body["evidence"]] == ["KG", "KG"]
    assert "MFN tariff of 9.5%" in body["evidence"][1]["claim"]


# --- it refuses the way the agent refuses --------------------------------------


def test_no_baseline_in_the_graph_is_a_refusal_not_a_number():
    kg = ScenarioKG(baseline=None)
    deps_module.set_runtime(deps_module.Runtime(kg=kg, llm=FakeLLM(), deps=None, graph=None))
    try:
        body = run(TestClient(app), {"shock": "fx", "sector": "agriculture"}).json()
    finally:
        deps_module.set_runtime(None)
    assert body["refused"] is True and body["outcome"] is None
    assert "No baseline" in body["reason"]
    assert body["assumptions"], "SRS 3.1.5: assumptions are stated even on a refusal"
    assert body["evidence"][0]["detail"].startswith("MATCH")


def test_no_preference_coverage_refuses_an_agreement_shock():
    kg = ScenarioKG(coverage=[{"agreement": "SAFTA", "agreement_type": "free_trade_agreement",
                               "matched_on": "61"}])
    deps_module.set_runtime(deps_module.Runtime(kg=kg, llm=FakeLLM(), deps=None, graph=None))
    try:
        body = run(TestClient(app), {"shock": "agreement", "sector": "apparel"}).json()
    finally:
        deps_module.set_runtime(None)
    assert body["refused"] is True
    assert "preference coverage" in body["reason"]
    assert body["baseline_usd"] == 1_000_000.0, "the known half of the picture is still stated"


def test_a_graph_outage_is_a_503():
    kg = ScenarioKG(raises=RuntimeError("neo4j down"))
    deps_module.set_runtime(deps_module.Runtime(kg=kg, llm=FakeLLM(), deps=None, graph=None))
    try:
        assert run(TestClient(app), {"shock": "fx", "sector": "agriculture"}).status_code == 503
    finally:
        deps_module.set_runtime(None)


# --- provenance travels ----------------------------------------------------------


def test_every_parameter_carries_its_source_including_tbd(client):
    body = run(client, {"shock": "fx", "sector": "agriculture"}).json()
    by_name = {p["name"]: p for p in body["outcome"]["parameters"]}
    assert by_name["fx_pass_through"]["basis"] == "literature_range"
    assert by_name["fx_pass_through"]["source"].startswith("TBD")
    assert all(p["overridden"] is False for p in by_name.values())


def test_an_override_is_applied_and_echoed_with_its_default(client):
    body = run(client, {"shock": "fx", "sector": "agriculture", "magnitude": 0.05,
                        "overrides": {"fx_pass_through": 1.0}}).json()
    moved = {p["name"]: p for p in body["outcome"]["parameters"]}["fx_pass_through"]
    assert moved["overridden"] is True and moved["value"] == 1.0 and moved["default"] == 0.6
    assert body["outcome"]["revenue_change_pct"] == pytest.approx(-0.01)


def test_an_agreement_run_names_the_rate_as_its_magnitude(client):
    body = run(client, {"shock": "agreement", "sector": "apparel",
                        "overrides": {"agreement_loss_mfn_tariff": 0.15}}).json()
    assert body["assumptions"][0] == "Shock modelled: agreement, magnitude 15.0%."
    assert "set in the scenario workbench" in body["outcome"]["detail"]


def test_every_run_states_its_assumptions_and_the_workbench_names_itself(client):
    body = run(client, {"shock": "tariff", "sector": "apparel", "magnitude": 0.1}).json()
    assert body["assumptions"][0] == "Shock modelled: tariff, magnitude 10.0%."
    assert any("scenario workbench" in a for a in body["assumptions"])
    assert body["assumptions"][-1] == body["outcome"]["detail"]


# --- the edges of the request -----------------------------------------------------


def test_the_workbench_needs_a_signed_in_user(client):
    assert run(client, {"shock": "fx", "sector": "agriculture"}, headers={}).status_code == 401


@pytest.mark.parametrize("body", [
    {"shock": "fx", "sector": "agriculture", "magnitude": 1.5},
    {"shock": "fx", "sector": "agriculture", "magnitude": -2},
    {"shock": "devaluation", "sector": "agriculture"},
    {"shock": "fx", "sector": "fisheries"},
    {"shock": "fx", "sector": "agriculture", "overrides": {"export_demand_elasticity": 2.0}},
])
def test_out_of_range_inputs_are_rejected_before_anything_runs(client, kg, body):
    assert run(client, body).status_code == 422
    assert kg.queries == []


def test_an_item_from_the_other_sector_is_rejected(client):
    response = run(client, {"shock": "fx", "sector": "agriculture", "item": "apparel_knit"})
    assert response.status_code == 422
    assert "agriculture" in response.json()["detail"]


def test_a_named_item_is_the_one_simulated(client, kg):
    body = run(client, {"shock": "fx", "sector": "agriculture", "item": "cinnamon"}).json()
    assert body["item"] == "cinnamon"


def test_runs_are_limited_under_their_own_namespace(client, monkeypatch):
    class Refusing:
        seen: list[str] = []

        async def check(self, identity, limit, window_s):
            self.seen.append(identity)
            from ceynex.api.rate_limit import Decision
            return Decision(allowed=False, limit=limit, remaining=0, retry_after_s=7)

    window = Refusing()
    scenario_routes.set_window(window)
    response = run(client, {"shock": "fx", "sector": "agriculture"})
    assert response.status_code == 429 and response.headers["Retry-After"] == "7"
    assert window.seen and window.seen[0].startswith("scenario:")


def test_the_switch_removes_the_surface(client, monkeypatch):
    monkeypatch.setenv("CEYNEX_SCENARIO", "off")
    assert run(client, {"shock": "fx", "sector": "agriculture"}).status_code == 404


def test_the_usage_page_now_discloses_the_scenario_allowance(client):
    body = client.get("/api/usage/limits", headers=auth()).json()
    assert any(item["key"].startswith("Scenario runs") for item in body["limits"])
