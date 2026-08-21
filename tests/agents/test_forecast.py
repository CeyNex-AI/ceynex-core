"""Assertions for SRS 3.1.3 and 3.1.10 — the forecast node.

This node has two jobs: serve whatever M1 or M3 registered, and stay useful
before they register anything. Both paths must produce an interval, because a
bare point is what SRS 3.1.3 forbids.
"""

import asyncio

import pandas as pd
import pytest

from ceynex.agents.common import AgentDeps
from ceynex.agents.forecast import AGENT, forecast_node
from ceynex.contracts import new_state
from ceynex.llm import FakeLLMClient
from ceynex.models import registry
from ceynex.models.timeseries import TimeSeriesModel

YEARS = [2015, 2016, 2017, 2019, 2020, 2021, 2022, 2023, 2024]
VALUES = [100.0, 112.0, 121.0, 133.0, 129.0, 145.0, 158.0, 166.0, 181.0]


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Never read or write the developer's real `models/` directory."""
    monkeypatch.setenv("CEYNEX_MODELS_DIR", str(tmp_path / "models"))


class KG:
    def __init__(self, *, history=None, raises=None):
        self.history = history if history is not None else list(zip(YEARS, VALUES, strict=True))
        self.raises = raises

    async def run(self, cypher, params=None):
        if self.raises:
            raise self.raises
        if "EXPORTS_TO" in cypher and "year" in cypher:
            return [{"year": y, "value": v} for y, v in self.history], cypher
        return [], cypher


def run(query="forecast cinnamon exports for the next 3 years", kg=None):
    patch = asyncio.run(
        forecast_node(new_state(query, "test"), AgentDeps(kg=kg or KG(), llm=FakeLLMClient(available=False)))
    )
    return patch["agent_outputs"][AGENT]


# --- the interval requirement --------------------------------------------


def test_the_baseline_forecast_still_carries_an_interval():
    """SRS 3.1.3 applies to the fallback too, not only to registered models."""
    points = run()["forecast"]
    assert points
    for point in points:
        assert point["lower"] <= point["point"] <= point["upper"]


def test_the_baseline_says_it_is_a_baseline():
    """A naive forecast presented as a model is worse than no forecast."""
    out = run()
    assert any("baseline" in a.lower() or "naive" in a.lower() for a in out["assumptions"])


def test_the_interval_widens_with_the_horizon():
    points = run()["forecast"]
    widths = [p["upper"] - p["lower"] for p in points]
    assert widths[-1] > widths[0]


def test_a_lower_bound_is_never_negative():
    falling = list(zip(YEARS, [900.0, 800.0, 700.0, 600.0, 500.0, 400.0, 300.0, 200.0, 100.0], strict=True))
    for point in run(kg=KG(history=falling))["forecast"]:
        assert point["lower"] >= 0.0


# --- serving the registry ------------------------------------------------


def test_a_registered_model_is_served_instead_of_the_baseline():
    frame = pd.DataFrame({"period": YEARS, "value": VALUES})
    registry.save(
        TimeSeriesModel(sector="agriculture", item="cinnamon").fit(frame),
        metrics={"mape": 0.05, "rmse": 1.0, "coverage": 0.8},
    )

    out = run()
    assert any("model registry" in a for a in out["assumptions"])
    assert not any("baseline" in a.lower() for a in out["assumptions"])


def test_the_served_model_is_named_with_its_version_in_evidence():
    """'A registered model produced this' is not traceable without the version."""
    frame = pd.DataFrame({"period": YEARS, "value": VALUES})
    saved = registry.save(
        TimeSeriesModel(sector="agriculture", item="cinnamon").fit(frame),
        metrics={"mape": 0.05, "rmse": 1.0, "coverage": 0.8},
    )

    details = [e["detail"] for e in run()["evidence"] if e["source_id"] == "MODEL"]
    assert details, "no evidence names the model that produced the forecast"
    assert saved.version in details[0]


def test_a_broken_registry_falls_back_rather_than_failing_the_node():
    frame = pd.DataFrame({"period": YEARS, "value": VALUES})
    saved = registry.save(TimeSeriesModel(sector="agriculture", item="cinnamon").fit(frame))
    # Corrupt the artifact the way a half-written file would be.
    from ceynex.settings import models_dir

    artifact = models_dir() / "agriculture" / "cinnamon" / saved.target / saved.version / "model.pkl"
    artifact.write_bytes(b"not a pickle")

    out = run()
    assert out["forecast"], "a corrupt registry entry took the forecast down with it"


# --- the agent node contract ---------------------------------------------


def test_the_node_writes_exactly_one_output_key():
    patch = asyncio.run(
        forecast_node(new_state("forecast tea", "test"), AgentDeps(kg=KG(), llm=FakeLLMClient(available=False)))
    )
    assert set(patch["agent_outputs"]) == {AGENT}


def test_at_least_two_evidence_entries_are_attached():
    assert len(run()["evidence"]) >= 2


def test_a_series_too_short_to_forecast_is_reported_not_extrapolated():
    out = run(kg=KG(history=[(2023, 100.0), (2024, 110.0)]))
    assert out["assumptions"], "the agent forecast from two points and said nothing"


def test_an_empty_history_produces_no_forecast_rather_than_a_guess():
    out = run(kg=KG(history=[]))
    assert not out.get("forecast")
    assert out["assumptions"]


def test_the_node_never_raises_when_the_graph_is_down():
    """SAD §4.1 partial-result guarantee."""
    out = run(kg=KG(raises=RuntimeError("neo4j unreachable")))
    assert out["agent"] == AGENT
