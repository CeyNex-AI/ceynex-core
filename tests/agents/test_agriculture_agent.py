"""Tests for SRS 3.1.4/3.1.6 agriculture evidence and degraded behaviour."""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from ceynex.agents import agriculture_commodity as agriculture
from ceynex.agents.common import AgentDeps
from ceynex.contracts import new_state
from ceynex.kg.client import KnowledgeGraphUnavailableError
from ceynex.llm import FakeLLMClient
from ceynex.models import registry
from ceynex.models.agriculture.baseline import AnnualNaiveModel

YEARS = list(range(2017, 2026))
VALUES = [100.0, 105.0, 109.0, 115.0, 121.0, 125.0, 131.0, 140.0, 147.0]


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("CEYNEX_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(agriculture, "relevant_dq_flags", lambda *_args, **_kwargs: [])


class KG:
    async def run(self, cypher, params=None):
        if "PRODUCED_IN" in cypher:
            return [
                {"district": "Matara", "share": None},
                {"district": "Galle", "share": None},
                {"district": "Ratnapura", "share": None},
            ], cypher
        if "SUBSTITUTES_WITH" in cypher:
            return [], cypher
        raise AssertionError(f"unexpected Cypher: {cypher}")


class DownKG:
    async def run(self, _cypher, _params=None):
        raise KnowledgeGraphUnavailableError("neo4j unavailable")


class RaisingLLM:
    async def generate_explanation(self, _payload):
        raise RuntimeError("LLM unavailable")


def _series(item, *, target, **_kwargs):
    assert item in {"tea", "cinnamon"}
    assert target in {"price", "export_volume"}
    return pd.DataFrame({"period": YEARS, "value": VALUES})


def run(query, *, kg=None, llm=None):
    patch = asyncio.run(
        agriculture.agriculture_commodity_node(
            new_state(query, "test"), AgentDeps(kg=kg or KG(), llm=llm or FakeLLMClient(available=False))
        )
    )
    return patch["agent_outputs"][agriculture.AGENT]


def test_current_cinnamon_price_trend_has_readable_faostat_evidence(monkeypatch):
    monkeypatch.setattr(agriculture, "annual_series", _series)

    out = run("What is the current price trend for cinnamon?")

    assert out["figures"]["latest_price"] == 147.0
    assert "rising" in out["summary"].lower()
    assert len(out["evidence"]) >= 2
    assert all(evidence["source_id"] == "FAOSTAT" for evidence in out["evidence"])
    assert all(evidence.get("period") == "2017-2025" for evidence in out["evidence"])
    assert 0.05 <= out["confidence"] <= 0.95
    assert not any(evidence["source_id"] == "DQ_FLAG" for evidence in out["evidence"])


def test_a_material_dq_flag_is_evidence_and_reduces_confidence_without_changing_values(monkeypatch):
    monkeypatch.setattr(agriculture, "annual_series", _series)
    baseline = run("What is the current price trend for cinnamon?")
    monkeypatch.setattr(
        agriculture,
        "relevant_dq_flags",
        lambda *_args, **_kwargs: [
            {
                "period_start": "2024-01-01",
                "metric": "price",
                "source_a": "FAOSTAT",
                "value_a": 10.05,
                "source_b": "PINK_SHEET",
                "value_b": 11.06,
                "pct_diff": 10.05,
                "severity": "material",
            }
        ],
    )

    out = run("What is the current price trend for cinnamon?")

    assert out["figures"] == baseline["figures"]
    assert out["confidence"] == pytest.approx(baseline["confidence"] - 0.05)
    assert any("material=1" in assumption for assumption in out["assumptions"])
    flag = next(evidence for evidence in out["evidence"] if evidence["source_id"] == "DQ_FLAG")
    assert "FAOSTAT" in flag["claim"] and "PINK_SHEET" in flag["claim"] and "10.1%" in flag["claim"]


def test_a_severe_dq_flag_uses_the_shared_severe_penalty(monkeypatch):
    monkeypatch.setattr(agriculture, "annual_series", _series)
    baseline = run("What is the current price trend for cinnamon?")
    monkeypatch.setattr(
        agriculture,
        "relevant_dq_flags",
        lambda *_args, **_kwargs: [
            {
                "period_start": "2024-01-01",
                "metric": "price",
                "source_a": "FAOSTAT",
                "value_a": 10.05,
                "source_b": "PINK_SHEET",
                "value_b": 15.08,
                "pct_diff": 50.0,
                "severity": "severe",
            }
        ],
    )

    out = run("What is the current price trend for cinnamon?")

    assert out["figures"] == baseline["figures"]
    assert out["confidence"] == pytest.approx(baseline["confidence"] - 0.10)
    assert any("severe=1" in assumption for assumption in out["assumptions"])


def test_cinnamon_forecast_uses_only_registered_price_model_with_80_percent_interval():
    frame = pd.DataFrame({"period": YEARS, "value": VALUES})
    registry.save(
        AnnualNaiveModel(
            sector="agriculture", item="cinnamon", target="producer_price", unit="USD/kg"
        ).fit(frame),
        training_rows=len(frame),
        training_window={"period_start": 2017, "period_end": 2025},
        source="FAOSTAT annual Sri Lanka cinnamon producer price",
        metrics={"mape": 0.12, "rmse": 1.0, "coverage": 0.8, "folds": 3.0},
        git_sha="a" * 40,
    )

    out = run("Will cinnamon prices rise or fall over the next two quarters?")

    point = out["forecast"][0]
    assert point["unit"] == "USD/kg"
    assert point["lower"] <= point["point"] <= point["upper"]
    assert "flat" in out["summary"].lower()
    assert any("annual" in assumption for assumption in out["assumptions"])
    assert all("producer_price" in evidence["detail"] for evidence in out["evidence"])


def test_cinnamon_district_question_refuses_to_invent_a_largest_share():
    out = run("Which district contributes the largest share of cinnamon exports?")

    assert "no sourced numerical district share" in out["summary"].lower()
    assert "Matara" in out["summary"]
    assert "largest_district_share" not in out["figures"]
    assert len(out["evidence"]) >= 2


def test_tea_export_volume_trend_uses_tea_board_series(monkeypatch):
    monkeypatch.setattr(agriculture, "annual_series", _series)

    out = run("How have tea export volumes changed over the last five years?")

    assert out["figures"]["latest_export_volume"] == 147.0
    assert "tea export volume" in out["summary"].lower()
    assert all(evidence["source_id"] == "TEA_BOARD" for evidence in out["evidence"])


def test_bare_forecast_question_defers_instead_of_answering_the_wrong_commodity(monkeypatch):
    """Regression: a real "cinnamon exports outlook" query returned TEA_BOARD
    evidence (this node's "volume" trend branch is hardcoded to tea) because
    wants_forecast=True with no M1-specific forecast_target fell through past
    the "forecast" kind into the default "volume" kind instead of deferring
    to the separate export-value forecast agent, per parse_intent's own
    documented intent. Found live 2026-08-26 via a real query against the
    deployed backend, before this fix.
    """
    monkeypatch.setattr(agriculture, "annual_series", _series)

    out = run("What's the outlook for cinnamon exports next year?")

    assert all(evidence["source_id"] != "TEA_BOARD" for evidence in out["evidence"])
    assert "forecast" not in out
    assert out["confidence"] == pytest.approx(0.20)


def test_substitution_question_is_an_honest_insufficient_data_response():
    out = run("If tea prices rise, what happens to demand for rubber?")

    assert "cannot be estimated responsibly" in out["summary"]
    assert not out["figures"]
    assert len(out["evidence"]) >= 2
    assert "substitution" in out["assumptions"][0].lower()


def test_llm_failure_keeps_grounded_figures_and_evidence(monkeypatch):
    monkeypatch.setattr(agriculture, "annual_series", _series)

    out = run("What is the current price trend for cinnamon?", llm=RaisingLLM())

    assert out["degraded"] is True
    assert out["figures"]["latest_price"] == 147.0
    assert len(out["evidence"]) >= 2


def test_knowledge_graph_failure_returns_a_contract_conformant_error():
    out = run("Which district contributes the largest share of cinnamon exports?", kg=DownKG())

    assert out["error"]
    assert out["degraded"] is True
