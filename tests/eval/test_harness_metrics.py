"""Assertions for the evaluation harness's own metrics.

A metric that measures the wrong thing is worse than no metric, because it gets
quoted. `is_refusal` did exactly that: on 2026-08-28 the same 30 questions
scored 100% refusal in degraded mode and 33% with the LLM writing the prose,
purely because the deterministic composer uses the vocabulary the marker list
was built from and the LLM paraphrases. These tests hold the fix in place.
"""

from eval.harness import is_refusal, ungrounded

# --- refusal detection ---------------------------------------------------


def test_a_structural_gap_counts_as_a_refusal_whatever_the_wording():
    """The real X11 answer, which matched none of the REFUSAL_MARKERS."""
    answer = (
        "In 2024, Sri Lanka exported USD 1,373,467,019 worth of tea to 145 different "
        "markets. Unfortunately, data for the fisheries sector is not available for "
        "comparison, so a direct comparison cannot be made."
    )
    result = {"unanswered": ["the fisheries sector is not covered by CeyNex"]}
    assert is_refusal(answer, result)


def test_an_answer_with_no_stated_gap_is_not_a_refusal():
    answer = "In 2024, Sri Lanka exported USD 1,373,467,019 worth of tea to 145 markets."
    assert not is_refusal(answer, {"unanswered": []})


def test_an_empty_answer_is_a_refusal_regardless_of_the_fields():
    """Producing nothing at all is a refusal whatever `unanswered` says."""
    assert is_refusal("", {"unanswered": []})


def test_the_marker_list_still_serves_results_that_predate_unanswered():
    """Older --json dumps replayed through `report` have no `unanswered` key."""
    assert is_refusal("This question could not be answered from the data loaded.", {})
    assert not is_refusal("Tea exports reached USD 1,373,467,019 in 2024.", {})


def test_the_structural_signal_wins_over_the_marker_list():
    """A phrase from the marker list appearing incidentally must not create a
    refusal the orchestrator never reported."""
    answer = "Rubber exports show no data gaps and reached USD 25,787,076 in 2024."
    assert not is_refusal(answer, {"unanswered": []})


# --- grounding metric ----------------------------------------------------


def test_the_metric_scores_against_evidence_only():
    """Deliberately stricter than the runtime guard in `merger`.

    Both figures here are ungrounded *against evidence*: the export value
    because it is invented, and the tariff rate because a magnitude the user
    supplied appears in no source. The runtime guard accepts the second (it
    sees the question) and rejects the first. That asymmetry is the whole
    reason the two callers pass different corpora to the same comparison, and
    it is why docs/EVALUATION.md's 2 ungrounded figures were both shock
    magnitudes rather than fabrications.
    """
    evidence = [{"claim": "Tea exports 2024: USD 1,373,467,019.", "detail": "MATCH (n) RETURN n"}]
    assert ungrounded("Tea exports were USD 1,373,467,019.", evidence) == []
    assert ungrounded("A 10.5% tariff costs USD 999,888,777.", evidence) == ["10.5", "999,888,777."]
