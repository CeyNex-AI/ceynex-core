"""Assertions first: these encode the documented formula in ceynex/orchestrator/confidence.py."""

import pytest

from ceynex.contracts import AgentOutput
from ceynex.orchestrator.confidence import (
    CEILING,
    FLOOR,
    aggregate_confidence,
    confidence_band,
    coverage_penalty,
    dq_penalty,
    staleness_penalty,
    weighted_agent_confidence,
)


def _output(agent, confidence, degraded=False, error=None) -> AgentOutput:
    out = AgentOutput(
        agent=agent,
        summary="s",
        figures={},
        assumptions=[],
        evidence=[],
        confidence=confidence,
        degraded=degraded,
    )
    if error:
        out["error"] = error
    return out


def test_weighting_favours_the_more_relevant_agent():
    outputs = {
        "export_analytics": _output("export_analytics", 0.9),
        "trade_economics": _output("trade_economics", 0.5),
    }
    flat = weighted_agent_confidence(outputs)
    weighted = weighted_agent_confidence(
        outputs, {"export_analytics": 3.0, "trade_economics": 1.0}
    )
    assert flat == pytest.approx(0.7)
    assert weighted == pytest.approx(0.8)


def test_failed_agent_drags_the_score_down_rather_than_being_ignored():
    outputs = {
        "export_analytics": _output("export_analytics", 0.9),
        "forecast": _output("forecast", 0.0, degraded=True, error="no model"),
    }
    assert weighted_agent_confidence(outputs) == pytest.approx(0.45)


@pytest.mark.parametrize(
    ("months", "expected"),
    [(None, 0.0), (0, 0.0), (5, 0.10), (10, 0.20), (36, 0.20)],
)
def test_staleness_accrues_then_caps(months, expected):
    assert staleness_penalty(months) == pytest.approx(expected)


def test_dq_penalises_severe_twice_as_hard_as_material_and_ignores_minor():
    assert dq_penalty(["minor", "minor"]) == pytest.approx(0.0)
    assert dq_penalty(["material"]) == pytest.approx(0.05)
    assert dq_penalty(["severe"]) == pytest.approx(0.10)
    assert dq_penalty(["severe"] * 10) == pytest.approx(0.25)  # capped


def test_coverage_penalty_fires_when_a_routed_agent_never_reported():
    outputs = {"export_analytics": _output("export_analytics", 0.9)}
    assert coverage_penalty(["export_analytics"], outputs) == 0.0
    assert coverage_penalty(["export_analytics", "forecast"], outputs) == pytest.approx(0.15)


def test_full_formula_matches_the_documented_arithmetic():
    outputs = {
        "export_analytics": _output("export_analytics", 0.9),
        "trade_economics": _output("trade_economics", 0.7),
    }
    # weighted = (2*0.9 + 1*0.7)/3 = 0.8333...; staleness = 0.06; dq = 0.05; coverage = 0
    score = aggregate_confidence(
        outputs,
        route=["export_analytics", "trade_economics"],
        relevance={"export_analytics": 2.0, "trade_economics": 1.0},
        months_since_latest_observation=3,
        dq_severities=["material"],
    )
    assert score == pytest.approx(0.8333333 - 0.06 - 0.05, abs=1e-6)


def test_score_never_reaches_zero_or_one():
    perfect = {"export_analytics": _output("export_analytics", 1.0)}
    hopeless = {"export_analytics": _output("export_analytics", 0.0, degraded=True)}
    assert aggregate_confidence(perfect) == pytest.approx(CEILING)
    assert aggregate_confidence(hopeless) == pytest.approx(FLOOR)


def test_empty_outputs_do_not_divide_by_zero():
    assert aggregate_confidence({}) == pytest.approx(FLOOR)


@pytest.mark.parametrize(
    ("score", "band"),
    [(0.90, "High"), (0.60, "Moderate"), (0.35, "Low"), (0.10, "Very low")],
)
def test_bands(score, band):
    assert confidence_band(score) == band
