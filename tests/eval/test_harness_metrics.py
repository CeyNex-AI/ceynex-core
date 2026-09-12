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


# --- citation discipline (CEYNEX_CITATIONS) -------------------------------------


def test_every_marker_is_checked_against_the_evidence_list():
    from eval.harness import citation_counts

    text = "Tea exports reached USD 1,373,467,019 in 2024 [1]. Germany took 12% [2]. See [9]."
    counts = citation_counts(text, evidence_count=2)
    assert counts["citations_total"] == 3
    assert counts["citations_valid"] == 2, "[9] indexes an entry that does not exist"


def test_figure_sentences_are_counted_and_cited_ones_recognised():
    from eval.harness import citation_counts

    text = (
        "Tea exports reached USD 1,373,467,019 in 2024 [1]. "  # figure, cited
        "That was a rise over the year. "  # no figure
        "Germany took USD 84,000,000 of it. "  # figure, not cited
        "Over 12 markets bought some."  # structural digits only
    )
    counts = citation_counts(text, evidence_count=3)
    assert counts["figure_sentences"] == 2
    assert counts["figure_sentences_cited"] == 1


def test_a_two_digit_marker_is_never_mistaken_for_a_figure():
    """`[12]` is a citation, not a claim — in the sentence count here and in
    the grounding check the runtime shares with the harness."""
    from eval.harness import citation_counts, ungrounded

    counts = citation_counts("Exports rose sharply [12].", evidence_count=12)
    assert counts["figure_sentences"] == 0 and counts["citations_valid"] == 1
    assert ungrounded("Exports rose sharply [12].", []) == []


def test_no_markers_means_zeros_not_an_error():
    from eval.harness import citation_counts

    assert citation_counts("Exports reached USD 4,200,000.", evidence_count=1) == {
        "citations_total": 0,
        "citations_valid": 0,
        "figure_sentences": 1,
        "figure_sentences_cited": 0,
    }


def test_the_report_carries_a_citations_block_even_with_the_flag_off():
    """An off run and an on run must summarise to the same shape."""
    from eval.harness import Result, report

    plain = Result(
        id="S01", category="single_sector", question="q", expected_route=["a"],
        actual_route=["a"], agents_used=["a"], answerable=True, partial=False,
        elapsed_ms=10.0, confidence=0.5, degraded=False, evidence_count=2,
        answer="USD 4,200,000 in 2024.", figure_sentences=1,
    )
    summary = report([plain])
    assert summary["citations"]["answers_with_markers"] == 0
    assert summary["citations"]["marker_valid_rate"] == {"rate": None, "of": 0}
    assert summary["citations"]["figure_sentences_cited_rate"] == {"rate": 0.0, "of": 1}


# --- repeated runs (eval/repeat.py) ----------------------------------------------


def _summary(exact, ungrounded_total, p95, *, cited=None):
    return {
        "questions": 30,
        "crashed": 0,
        "routing": {"exact_match": {"rate": exact, "of": 30}, "recall": 0.9,
                    "never_empty": {"rate": 1.0, "of": 30}},
        "evidence": {"answers_fully_grounded": {"rate": 0.85, "of": 27},
                     "ungrounded_figures_total": ungrounded_total,
                     "mean_evidence_per_answer": 4.3, "answers_with_no_evidence": 0},
        "refusal": {"unanswerable_correctly_refused": {"rate": 1.0, "of": 3}},
        "latency_ms": {"single_sector": {"p50": 3000.0, "p95": p95}},
        "degraded_answers": 0,
        "citations": {"answers_with_markers": 0 if cited is None else 27,
                      "marker_valid_rate": {"rate": cited, "of": 0 if cited is None else 80},
                      "figure_sentences_cited_rate": {"rate": None, "of": 0}},
    }


def test_repeated_runs_report_the_median_and_the_spread():
    from eval.repeat import summarize_runs

    out = summarize_runs([_summary(0.60, 4, 8568.0), _summary(0.5667, 5, 5582.0),
                          _summary(0.60, 5, 5867.0)])
    assert out["runs"] == 3
    exact = out["metrics"]["routing.exact_match"]
    assert exact == {"median": 0.6, "min": 0.5667, "max": 0.6,
                     "values": [0.6, 0.5667, 0.6], "runs": 3}
    assert out["metrics"]["evidence.ungrounded_figures_total"]["median"] == 5
    assert out["latency_ms"]["single_sector"]["p95"]["min"] == 5582.0
    assert out["latency_ms"]["single_sector"]["p95"]["max"] == 8568.0


def test_a_metric_no_run_measured_is_omitted_rather_than_reported_as_zero():
    from eval.repeat import summarize_runs

    out = summarize_runs([_summary(0.6, 4, 8000.0)])
    assert "citations.marker_valid_rate" not in out["metrics"], "None is not a measurement"
    assert "latency_ms" in out and "cross_sector" not in out["latency_ms"]


def test_a_cited_run_summarises_its_marker_rate():
    from eval.repeat import summarize_runs

    out = summarize_runs([_summary(0.6, 4, 8000.0, cited=0.99), _summary(0.6, 4, 8000.0, cited=1.0)])
    assert out["metrics"]["citations.marker_valid_rate"]["median"] == 0.995


def test_questions_that_disagree_with_themselves_are_listed():
    from eval.repeat import disagreements

    def result(qid, route, ungrounded):
        return {"id": qid, "question": qid, "actual_route": route,
                "ungrounded_figures": ungrounded}

    runs = [
        [result("S07", ["export_analytics", "apparel_manufacturing"], []),
         result("M01", ["agriculture_commodity"], ["4.2"])],
        [result("S07", ["apparel_manufacturing"], []),
         result("M01", ["agriculture_commodity"], ["4.2"])],
    ]
    moved = disagreements(runs)
    assert [m["id"] for m in moved] == ["S07"]
    assert moved[0]["route_moved"] and not moved[0]["grounding_moved"]


def test_clearing_the_cache_removes_every_entry_and_reports_the_count(tmp_path):
    from eval.repeat import clear_prompt_cache

    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    assert clear_prompt_cache(tmp_path) == 2
    assert list(tmp_path.iterdir()) == []
    assert clear_prompt_cache(tmp_path / "missing") == 0


def test_the_repeat_summary_is_written_beside_the_runs(tmp_path):
    import json

    from eval.repeat import write_repeat_summary

    out = write_repeat_summary(tmp_path, [_summary(0.6, 4, 8000.0)], [[]], degraded=False)
    payload = json.loads(out.read_text())
    assert payload["repeat"]["runs"] == 1 and payload["disagreements"] == []


def test_both_grounding_definitions_are_reported_on_the_same_answer():
    """EVALUATION.md §13: the published series stays strict, whichever rule the
    runtime used, and what only the direction rule accepts is listed with its
    sentence for the audit."""
    from eval.harness import direction_accepted

    evidence = [{"claim": "GSP+ loss changes export value by USD -161,815,198.", "detail": ""}]
    answer = "Export value would decrease by USD 161,815,198 a year. It is a large share."
    assert ungrounded(answer, evidence) == ["161,815,198"]
    assert ungrounded(answer, evidence, direction_aware=True) == []
    assert direction_accepted(answer, evidence) == [
        "161,815,198 :: Export value would decrease by USD 161,815,198 a year."
    ]


def test_the_strict_metric_ignores_the_runtime_switch(monkeypatch):
    monkeypatch.setenv("CEYNEX_GROUNDING", "direction")
    evidence = [{"claim": "USD -161,815,198", "detail": ""}]
    assert ungrounded("It would decrease by USD 161,815,198 a year.", evidence) == ["161,815,198"]
