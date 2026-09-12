"""Unit checks for the M1 real-service evaluation acceptance criteria."""

from __future__ import annotations

from eval.agriculture_agent_e2e import CHECKS, assess, report


def _output(**overrides):
    output = {
        "summary": "A grounded response.",
        "figures": {},
        "forecast": [],
        "evidence": [{}, {}],
        "confidence": 0.5,
        "degraded": True,
    }
    output.update(overrides)
    return output


def test_trend_check_requires_its_expected_figure():
    check = CHECKS[0]

    assert assess(check, _output(figures={"latest_price": 10.05})) == []
    assert "missing figure: latest_price" in assess(check, _output())


def test_refusal_check_accepts_the_explicit_data_limit():
    check = CHECKS[4]
    output = _output(summary="The effect cannot be estimated responsibly from the available evidence.")

    assert assess(check, output) == []


def test_cinnamon_forecast_check_requires_the_undercoverage_evidence():
    check = CHECKS[1]
    output = _output(
        forecast=[{"point": 10.05}],
        evidence=[{"claim": "The interval is below nominal 80% coverage."}, {}],
    )

    assert assess(check, output) == []
    assert "missing forecast-evidence limitation: 'below nominal 80%'" in assess(
        check, _output(forecast=[{"point": 10.05}])
    )


def test_report_keeps_the_question_denominator():
    summary = report(
        [
            {"passed": True, "degraded": True, "evidence_count": 2},
            {"passed": False, "degraded": False, "evidence_count": 1},
        ]
    )

    assert summary == {
        "questions": 2,
        "passed": 1,
        "failed": 1,
        "degraded": 1,
        "mean_evidence_count": 1.5,
    }
