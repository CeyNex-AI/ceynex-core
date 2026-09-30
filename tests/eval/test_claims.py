"""The claim-level check's parsing, summary, audit sample and agreement. The
judge itself needs an API key, so it is exercised by running the module."""

import csv

from eval import claims


def test_the_prompt_carries_question_records_and_answer():
    prompt = claims.judge_prompt("Top tea market?", ["Iraq 12.4% of tea value, 2025"], "Russia.")
    assert "Top tea market?" in prompt and "- Iraq 12.4%" in prompt and prompt.endswith("Russia.")


def test_long_references_are_cut_with_a_marker():
    prompt = claims.judge_prompt("q", ["x" * 9000, "y" * 9000], "a")
    assert "further records omitted" in prompt and "y" * 100 not in prompt


def test_unknown_labels_become_unverifiable_and_bad_json_is_recorded():
    parsed = claims.parse_judgement('{"claims": [{"claim": "c", "label": "maybe"}, {"claim": "d", "label": "contradicted"}]}')
    assert [c["label"] for c in parsed["claims"]] == ["UNVERIFIABLE", "CONTRADICTED"]
    assert "parse_error" in claims.parse_judgement("not json")


def _j(condition, repeat, labels):
    return {"repeat": repeat, "id": "S01", "condition": condition, "question": "q", "answer": "a",
            "claims": [{"claim": f"c{i}", "label": label, "reference_quote": ""} for i, label in enumerate(labels)]}


def test_summary_rates_per_condition():
    judgements = [_j("A", 1, ["CONTRADICTED", "UNVERIFIABLE"]), _j("C", 1, ["SUPPORTED", "SUPPORTED"]),
                  _j("D", 1, ["SUPPORTED"]), _j("A", 2, ["SUPPORTED", "CONTRADICTED"]),
                  _j("C", 2, ["SUPPORTED"]), _j("D", 2, [])]
    s = claims.summarise(judgements)
    assert s["A"]["contradicted"]["mean"] == 0.5
    assert s["A"]["answers_with_contradiction"]["mean"] == 1.0
    assert s["C"]["supported"]["mean"] == 1.0
    assert "Contradicted" in claims.render(s)


def test_audit_is_blind_and_agreement_is_measured(tmp_path):
    judgements = [_j(c, 1, ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]) for c in "ACD"]
    n = claims.write_audit(judgements, tmp_path, size=6)
    assert n == 6
    header = (tmp_path / "audit_blind.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "condition" not in header and "judge" not in header

    with (tmp_path / "audit_key.csv").open(encoding="utf-8") as f:
        key = {r["item"]: r["judge_label"] for r in csv.DictReader(f)}
    with (tmp_path / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "question", "claim", "human_label (SUPPORTED/CONTRADICTED/UNVERIFIABLE)"])
        for item, label in key.items():
            w.writerow([item, "q", "c", label])
    result = claims.agreement(tmp_path / "labels.csv", tmp_path / "audit_key.csv")
    assert result["items"] == 6 and result["agreement"] == 1.0 and result["kappa"] == 1.0


def test_kappa_is_zero_for_chance_agreement():
    assert claims.cohens_kappa(["SUPPORTED", "CONTRADICTED"], ["CONTRADICTED", "SUPPORTED"]) < 0
