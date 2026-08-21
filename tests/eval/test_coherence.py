"""Assertions for the merge-coherence rating tool.

The tool's whole value is that raters cannot game it, so the tests are about
blindness and about the spread being reported rather than averaged away.
"""

import csv
import json

import pytest

from eval.coherence import build_sheet, score, write_sheet

RESULTS = [
    {"id": "S01", "question": "q1", "answer": "One unified answer about tea.", "agents_used": ["export_analytics"]},
    {"id": "X01", "question": "q2", "answer": "The tea agent says X. The apparel agent says Y.", "agents_used": ["a", "b"]},
    {"id": "M01", "question": "q3", "answer": "A simulation answer.", "agents_used": ["trade_economics"]},
    {"id": "S99", "question": "q4", "answer": "   ", "agents_used": []},
]


def test_the_sheet_carries_no_agent_attribution():
    """A rater who can count the agents is rating the machinery, not the prose."""
    sheet = build_sheet(RESULTS)
    for row in sheet.rows:
        assert "agents_used" not in row
        assert "agent" not in " ".join(str(v) for v in row).lower()


def test_the_sheet_hides_the_question_id():
    sheet = build_sheet(RESULTS)
    labels = {row["label"] for row in sheet.rows}
    assert labels.isdisjoint({"S01", "X01", "M01"})


def test_the_key_maps_every_label_back():
    sheet = build_sheet(RESULTS)
    assert set(sheet.key) == {row["label"] for row in sheet.rows}
    assert set(sheet.key.values()) <= {"S01", "X01", "M01"}


def test_answers_with_no_content_are_not_put_up_for_rating():
    """Rating an empty answer measures nothing and wastes a rater's attention."""
    assert len(build_sheet(RESULTS).rows) == 3


def test_the_shuffle_is_reproducible_so_the_key_stays_valid():
    assert build_sheet(RESULTS).key == build_sheet(RESULTS).key


def test_a_different_seed_produces_a_different_order():
    """Enough items that two seeds colliding is not the coin flip 3! would give."""
    many = [
        {"id": f"Q{i:02d}", "question": f"q{i}", "answer": f"answer {i}", "agents_used": []}
        for i in range(20)
    ]
    assert build_sheet(many, seed=1).key != build_sheet(many, seed=2).key


# --- scoring -------------------------------------------------------------


def sheet_file(tmp_path, name, ratings):
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "question", "answer", "rating_1_to_5", "comment"])
        writer.writeheader()
        for label, value in ratings.items():
            writer.writerow({"label": label, "question": "q", "answer": "a", "rating_1_to_5": value, "comment": ""})
    return path


def test_agreement_and_disagreement_do_not_score_the_same(tmp_path):
    """Three raters at 4 and raters at 2/4/5 share a mean and mean different things."""
    agreed = score([sheet_file(tmp_path, f"r{i}", {"A01": 4}) for i in range(3)])
    split = score(
        [
            sheet_file(tmp_path, "s1", {"A01": 2}),
            sheet_file(tmp_path, "s2", {"A01": 4}),
            sheet_file(tmp_path, "s3", {"A01": 5}),
        ]
    )

    assert agreed["mean_coherence"] == pytest.approx(4.0)
    assert split["mean_coherence"] == pytest.approx(3.67, abs=0.01)
    assert agreed["mean_spread_between_raters"] == 0
    assert split["mean_spread_between_raters"] == 3


def test_disputed_answers_are_named_for_follow_up(tmp_path):
    result = score(
        [
            sheet_file(tmp_path, "r1", {"A01": 1, "A02": 4}),
            sheet_file(tmp_path, "r2", {"A01": 5, "A02": 4}),
        ]
    )
    assert result["disputed_answers"] == ["A01"]


def test_a_rating_outside_the_scale_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        score([sheet_file(tmp_path, "r1", {"A01": 7})])


def test_an_unfilled_sheet_is_rejected_rather_than_scored_as_zero(tmp_path):
    with pytest.raises(ValueError, match="no ratings"):
        score([sheet_file(tmp_path, "r1", {"A01": ""})])


def test_scores_map_back_to_question_ids_when_the_key_is_given(tmp_path):
    sheet = build_sheet(RESULTS)
    label = sheet.rows[0]["label"]
    result = score([sheet_file(tmp_path, "r1", {label: 4})], key=sheet.key)
    assert result["per_answer"][0]["question_id"] == sheet.key[label]


def test_writing_a_sheet_also_writes_the_key_and_the_rubric(tmp_path):
    out = tmp_path / "sheet.csv"
    write_sheet(build_sheet(RESULTS), out)

    assert out.exists()
    assert json.loads(out.with_suffix(".key.json").read_text())
    assert "coherence" in out.with_suffix(".rubric.txt").read_text().lower()
