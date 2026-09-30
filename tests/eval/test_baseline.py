"""The baseline comparison's scoring: what counts as a figure, what counts as
unsupported, and how repeats are summarised. The runners need the full stack
and an API key, so they are exercised by running the harness, not here."""

from eval import baseline


def test_years_and_short_numbers_are_not_figures():
    assert baseline.figures("In 2024, 12 markets bought tea; by 2025 it rose.") == []


def test_a_figure_absent_from_the_reference_is_unsupported():
    result = baseline.score(
        {"answer": "Russia took 23.5%, worth USD 1,234.5 million; Iraq took 18.2%."},
        ["Which market is largest?", "Russia 23.5% of tea exports, USD 1,234.5 million"],
    )
    assert result["figures"] == ["23.5", "1,234.5", "18.2"]
    assert result["unsupported"] == ["18.2"]


def test_a_figure_quoted_from_the_question_is_supported():
    result = baseline.score({"answer": "A 7.5% depreciation would lift volumes."}, ["What if the rupee falls 7.5%?"])
    assert result["unsupported"] == []


def _row(repeat, qid, *, answerable=True, a_unsupported=True, a_refused=False):
    row = {"repeat": repeat, "id": qid, "category": "single_sector", "question": "q",
           "answerable": answerable, "reference_size": 1}
    for c in baseline.CONDITIONS:
        unsupported = ["18.2"] if (c == "A" and a_unsupported) else []
        row[c] = {"figures": ["18.2", "23.5"], "unsupported": unsupported,
                  "refused": (not answerable and c != "A") or (c == "A" and a_refused),
                  "error": None, "elapsed_ms": 1000.0 * (repeat + 1), "guard_discards": []}
    return row


def test_summary_reports_rates_per_condition_with_spread_over_repeats():
    records = [_row(1, "S1"), _row(1, "U1", answerable=False),
               _row(2, "S1", a_unsupported=False), _row(2, "U1", answerable=False, a_refused=True)]
    summary = baseline.summarise(records)

    assert summary["repeats"] == 2 and summary["questions"] == 2
    a = summary["A"]
    assert a["unsupported_figure_rate"]["mean"] == 0.25  # 50% then 0%
    assert a["unsupported_figure_rate"]["sd"] > 0
    assert a["unanswerable_refused"]["mean"] == 0.5
    assert summary["C"]["unsupported_figure_rate"]["mean"] == 0.0
    assert summary["C"]["unanswerable_refused"]["mean"] == 1.0


def test_render_lists_every_condition():
    table = baseline.render(baseline.summarise([_row(1, "S1"), _row(1, "U1", answerable=False)]))
    for c in "ABCD":
        assert baseline.LABELS[c] in table
