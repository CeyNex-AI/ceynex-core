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


#: Which source each SERIES branch must read from. The agent names the source in
#: every evidence entry it emits, so reading an unfiltered series would cite one
#: source over another's numbers -- live 2026-09-03, a Comtrade unit-value blend
#: was cited as the FAOSTAT producer-price series.
_SOURCE_FOR_TARGET = {"price": "FAOSTAT", "export_volume": "TEA_BOARD"}


def _series(item, *, target, source_id=None, **_kwargs):
    assert item in {"tea", "cinnamon"}
    assert target in {"price", "export_volume"}
    assert source_id == _SOURCE_FOR_TARGET[target], (
        f"{target} must be read from {_SOURCE_FOR_TARGET[target]} only, not blended across sources"
    )
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


def test_a_material_dq_flag_is_evidence_without_changing_the_agents_own_confidence(monkeypatch):
    """The DQ penalty is applied exactly once, centrally, by
    orchestrator.merger's scan of the merged evidence for DQ_FLAG entries --
    not here too. Double-applying it (once per-agent, once at merge) would
    silently over-penalize the final score. See merger._dq_severities_from_evidence.
    """
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
    assert out["confidence"] == pytest.approx(baseline["confidence"])
    assert any("material=1" in assumption for assumption in out["assumptions"])
    flag = next(evidence for evidence in out["evidence"] if evidence["source_id"] == "DQ_FLAG")
    assert "FAOSTAT" in flag["claim"] and "PINK_SHEET" in flag["claim"] and "10.1%" in flag["claim"]
    assert "severity=material" in flag["detail"]


def test_a_severe_dq_flag_is_evidence_without_changing_the_agents_own_confidence(monkeypatch):
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
    assert out["confidence"] == pytest.approx(baseline["confidence"])
    assert any("severe=1" in assumption for assumption in out["assumptions"])
    flag = next(evidence for evidence in out["evidence"] if evidence["source_id"] == "DQ_FLAG")
    assert "severity=severe" in flag["detail"]


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


def test_undercovered_cinnamon_interval_reduces_confidence_and_is_explained_in_evidence():
    frame = pd.DataFrame({"period": YEARS, "value": VALUES})
    registry.save(
        AnnualNaiveModel(
            sector="agriculture", item="cinnamon", target="producer_price", unit="USD/kg"
        ).fit(frame),
        training_rows=len(frame),
        training_window={"period_start": 2017, "period_end": 2025},
        source="FAOSTAT annual Sri Lanka cinnamon producer price",
        metrics={"mape": 0.12, "rmse": 1.0, "coverage": 1 / 3, "folds": 3.0},
        git_sha="a" * 40,
    )

    out = run("Will cinnamon prices rise or fall over the next two quarters?")

    metadata = registry.list_models(sector="agriculture", item="cinnamon")[0]
    assert agriculture._interval_coverage_penalty(metadata.metrics) == pytest.approx(0.14)
    assert out["confidence"] == pytest.approx(agriculture._model_confidence(metadata))
    assert any("coverage was 33%" in assumption for assumption in out["assumptions"])
    assert any("below nominal 80%" in evidence["claim"] for evidence in out["evidence"])


def test_missing_interval_coverage_is_penalised_instead_of_assumed_calibrated():
    assert agriculture._interval_coverage_penalty({"coverage": 0.8}) == 0.0
    assert agriculture._interval_coverage_penalty({"coverage": 0.33}) == pytest.approx(0.141)
    assert agriculture._interval_coverage_penalty({}) == pytest.approx(0.10)
    assert agriculture._interval_coverage_penalty({"coverage": 1.2}) == pytest.approx(0.10)


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


def test_rubber_volume_question_declines_instead_of_answering_about_tea(monkeypatch):
    """Guard rail: SERIES["volume"] only has a sourced series for tea. A
    query naming a different item must decline honestly, not silently
    substitute tea's data -- this is the exact class of bug the cinnamon
    forecast regression above caught, generalized to any other item.
    """
    monkeypatch.setattr(agriculture, "annual_series", _series)

    out = run("How have rubber export volumes changed over the last five years?")

    assert all(evidence["source_id"] != "TEA_BOARD" for evidence in out["evidence"])
    assert "rubber" in out["summary"].lower()
    assert "figures" not in out or out["figures"] == {}
    assert out["confidence"] == pytest.approx(0.20)


def test_the_item_mismatch_decline_does_not_claim_only_tea_is_answerable(monkeypatch):
    """Found live 2026-09-03. The decline used to end "only tea is covered for
    this question shape", which states this agent's coverage as the system's --
    and it is false: the graph answers rubber concentration, share and growth,
    and did so in the very same response (S03). The merger drops this line when
    another finding covered the question, but the line itself must be true on the
    occasions it does surface.
    """
    monkeypatch.setattr(agriculture, "annual_series", _series)

    summary = run("How have rubber export volumes changed over the last five years?")["summary"]

    assert "only tea" not in summary.lower()
    assert "rubber" in summary.lower(), "the decline must still say which item it holds nothing for"


def test_a_bare_cinnamon_data_question_answers_with_the_real_price_series(monkeypatch):
    """Regression, found live 2026-08-27: "do we have cinnamon data?" has no
    "price"/"production"/forecast signal for _question_kind to key off, so it
    fell through to the hardcoded "volume" default -- tea's series, not
    cinnamon's -- and declined, even though cinnamon's own sourced price
    series was one line away and would have honestly answered the question.
    """
    monkeypatch.setattr(agriculture, "annual_series", _series)

    out = run("do we have cinnamon data?")

    assert out["figures"]["latest_price"] == 147.0
    assert all(evidence["source_id"] == "FAOSTAT" for evidence in out["evidence"])
    assert out["confidence"] > 0.20, "must not read as the generic data-gap decline"


def test_unsupported_target_evidence_says_no_compatible_series_was_found(monkeypatch):
    """The generic second evidence entry attached to every _unsupported_target
    decline claimed "no incompatible ... series was substituted" -- backwards,
    since of course nothing incompatible was substituted. Found live
    2026-08-27 from a user-reported "do we have cinnamon data?" answer that
    (correctly) said only tea has volume data, justified with this
    nonsensical claim.
    """
    monkeypatch.setattr(agriculture, "annual_series", _series)

    out = run("How have rubber export volumes changed over the last five years?")

    claims = " ".join(e["claim"] for e in out["evidence"]).lower()
    assert "incompatible" not in claims
    assert "compatible" in claims


def test_coconut_price_question_declines_instead_of_answering_about_cinnamon(monkeypatch):
    """Same guard rail, the price side: SERIES["price"] only has a sourced
    series for cinnamon.
    """
    monkeypatch.setattr(agriculture, "annual_series", _series)

    out = run("What is the current price trend for coconut?")

    assert all(evidence["source_id"] != "FAOSTAT" for evidence in out["evidence"])
    assert "coconut" in out["summary"].lower()
    assert out["confidence"] == pytest.approx(0.20)


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


def test_annual_series_and_dq_flags_run_off_the_event_loop_thread(monkeypatch):
    """Regression: annual_series/relevant_dq_flags are synchronous psycopg calls
    (ceynex/data/reader.py). Calling them inline from this async node blocks the
    whole event loop -- not just this request -- for as long as Postgres takes
    to answer, and (found live 2026-08-26 via a stack-dump timer) an unreachable
    Postgres hangs well past graph.py's NODE_TIMEOUT_S with nothing able to
    cancel it, since a blocking call has no await point to receive the
    cancellation. Both call sites must go through asyncio.to_thread.
    """
    import threading

    main_thread = threading.current_thread()
    seen_threads: list[threading.Thread] = []

    def recording_series(*_args, **_kwargs):
        seen_threads.append(threading.current_thread())
        return _series(*_args, target="price")

    def recording_dq_flags(*_args, **_kwargs):
        seen_threads.append(threading.current_thread())
        return []

    monkeypatch.setattr(agriculture, "annual_series", recording_series)
    monkeypatch.setattr(agriculture, "relevant_dq_flags", recording_dq_flags)

    run("What is the current price trend for cinnamon?")

    assert seen_threads, "annual_series/relevant_dq_flags were never called"
    assert all(t is not main_thread for t in seen_threads), (
        "a mocked reader call ran on the event loop thread -- the real "
        "psycopg call would have blocked it"
    )
