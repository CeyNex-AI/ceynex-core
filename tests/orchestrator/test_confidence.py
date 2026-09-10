"""Assertions first: these encode the documented formula in ceynex/orchestrator/confidence.py."""

import pytest

from ceynex.contracts import AgentOutput
from ceynex.orchestrator.confidence import (
    CEILING,
    FLOOR,
    aggregate_confidence,
    aggregate_confidence_breakdown,
    clamp,
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


def test_coverage_penalty_fires_when_a_routed_agent_errored():
    """Coverage gap in the original suite: only the "never reported" disjunct
    of `output is None or output.get("error") or output["degraded"]` had a
    test. The other two conditions were completely unexercised.
    """
    outputs = {
        "export_analytics": _output("export_analytics", 0.9),
        "forecast": _output("forecast", 0.0, error="no model"),
    }
    assert coverage_penalty(["export_analytics", "forecast"], outputs) == pytest.approx(0.15)


def test_coverage_penalty_fires_when_a_routed_agent_is_degraded():
    outputs = {
        "export_analytics": _output("export_analytics", 0.9),
        "forecast": _output("forecast", 0.6, degraded=True),
    }
    assert coverage_penalty(["export_analytics", "forecast"], outputs) == pytest.approx(0.15)


def test_weight_of_zero_excludes_an_agent_rather_than_dividing_by_it():
    """Coverage gap: `weight <= 0.0: continue` was never exercised. Real
    exposure -- the LLM router is explicitly allowed to return relevance
    0.0 for a routed agent (router.py's ROUTER_SYSTEM: "relevance is
    0.0-1.0 per agent"), not just the keyword router's fixed 0.6-1.0 set.
    """
    outputs = {
        "export_analytics": _output("export_analytics", 0.9),
        "trade_economics": _output("trade_economics", 0.1),
    }
    assert weighted_agent_confidence(
        outputs, {"export_analytics": 1.0, "trade_economics": 0.0}
    ) == pytest.approx(0.9)


def test_dq_penalty_combines_material_and_severe_in_the_same_answer():
    """Coverage gap: material and severe were only ever tested in isolation,
    never combined -- the cross-term in `0.05*material + 0.10*severe` was
    unexercised.
    """
    assert dq_penalty(["material", "material", "severe"]) == pytest.approx(0.20)


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


def test_full_formula_includes_the_coverage_penalty_when_it_fires():
    """Coverage gap: the one existing "full formula" test has route exactly
    matching outputs, so coverage_penalty is always 0 there -- its
    subtraction inside aggregate_confidence's composite score was never
    actually exercised with a nonzero value, only unit-tested in isolation.
    A copy-paste omission of that term from the sum would not have been
    caught.
    """
    outputs = {"export_analytics": _output("export_analytics", 0.9)}
    # weighted = 0.9; staleness = 0; dq = 0; coverage = 0.15 (forecast never reported)
    score = aggregate_confidence(outputs, route=["export_analytics", "forecast"])
    assert score == pytest.approx(0.9 - 0.15)


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


@pytest.mark.parametrize(
    ("score", "band"),
    [
        # Coverage gap: none of the four cases above sit on a threshold --
        # exactly where an off-by-one (>= vs >) would hide. Confirmed live
        # 2026-08-26 (0.75 -> "High", 0.7312 -> "Moderate") but never pinned
        # as a regression test until now.
        (0.75, "High"),
        (0.7499, "Moderate"),
        (0.50, "Moderate"),
        (0.4999, "Low"),
        (0.30, "Low"),
        (0.2999, "Very low"),
    ],
)
def test_bands_at_the_exact_thresholds(score, band):
    assert confidence_band(score) == band


def test_the_breakdown_always_adds_up_to_the_score_beside_it():
    """A waterfall that disagreed with the number it explains would be worse
    than no waterfall — it would make the most predictable question this project
    is asked (SRS 3.1.4) look like it has two answers. `aggregate_confidence`
    delegates to the breakdown so the two cannot drift apart."""
    outputs = {
        "export_analytics": {"confidence": 0.8, "degraded": False},
        "forecast": {"confidence": 0.6, "degraded": True},
    }
    route = ["export_analytics", "forecast"]
    breakdown = aggregate_confidence_breakdown(
        outputs, route, None, 18.0, ["material", "severe"]
    )
    assert breakdown.final == aggregate_confidence(outputs, route, None, 18.0, ["material", "severe"])
    # And the terms are the formula, not a re-derivation of it.
    assert breakdown.final == clamp(
        breakdown.weighted - breakdown.staleness - breakdown.dq - breakdown.coverage
    )


def test_the_breakdown_names_the_penalty_that_actually_bit():
    """The point of showing the working is that a reader can see *which* term
    cost them, not just that something did."""
    outputs = {"export_analytics": {"confidence": 0.9, "degraded": True}}
    breakdown = aggregate_confidence_breakdown(outputs, ["export_analytics"], None, 0.0, [])
    assert breakdown.coverage > 0
    assert breakdown.staleness == 0
    assert breakdown.dq == 0
