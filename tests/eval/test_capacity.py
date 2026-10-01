"""The pre-registered capacity rule (EVALUATION.md §15), on synthetic runs."""

from __future__ import annotations

import json

import yaml

from eval import capacity


def run(single_ms, grounded=True, evidence=3, route_ok=True):
    results = [
        {"category": "single_sector", "elapsed_ms": ms, "answerable": True,
         "ungrounded_figures": [] if grounded else ["9%"], "evidence_count": evidence,
         "actual_route": ["a"], "expected_route": ["a"] if route_ok else ["b"]}
        for ms in single_ms
    ]
    results.append({"category": "cross_sector", "elapsed_ms": 50_000.0, "answerable": True,
                    "ungrounded_figures": [], "evidence_count": 4,
                    "actual_route": ["a", "b"], "expected_route": ["a", "b"]})
    return results


def write(directory, runs):
    directory.mkdir()
    for i, results in enumerate(runs, start=1):
        (directory / f"run-{i}.json").write_text(json.dumps({"results": results}))
    return directory


def test_the_p95_is_pooled_across_runs_and_ignores_other_categories():
    m = capacity.measure([run([1000.0] * 11 + [20000.0]), run([1000.0] * 12), run([1000.0] * 12)])
    assert m["single_sector_samples"] == 36
    assert m["single_sector_p95_ms"] == 1000.0  # one slow sample in 36 is under the 95th


def test_b_is_adopted_when_faster_and_no_worse():
    a = capacity.measure([run([12000.0] * 12)] * 3)
    b = capacity.measure([run([8000.0] * 12)] * 3)
    assert capacity.verdict(a, b)["adopt_b"] is True


def test_b_is_refused_when_it_stays_over_budget():
    a = capacity.measure([run([12000.0] * 12)] * 3)
    b = capacity.measure([run([10500.0] * 12)] * 3)
    v = capacity.verdict(a, b)
    assert v["adopt_b"] is False and v["checks"]["p95_within_budget"] is False


def test_b_is_refused_when_grounding_drops_past_the_noise_floor():
    a = capacity.measure([run([12000.0] * 12)] * 3)
    # 13 fully grounded answers per run in A; 11 in B: two past the floor of one.
    b = capacity.measure([run([8000.0] * 9) + run([8000.0] * 3, grounded=False)] * 3)
    assert capacity.verdict(a, b)["checks"]["grounding_held"] is False


def test_b_is_refused_for_a_single_answer_without_evidence():
    a = capacity.measure([run([12000.0] * 12)] * 3)
    b = capacity.measure([run([8000.0] * 12), run([8000.0] * 11) + run([8000.0], evidence=0),
                          run([8000.0] * 12)])
    assert capacity.verdict(a, b)["checks"]["no_answer_without_evidence"] is False


def test_arm_b_changes_the_merge_role_and_nothing_else(tmp_path):
    out = capacity.write_arm_b_config(tmp_path / "config")
    before = yaml.safe_load((capacity.settings.config_dir() / "llm.yaml").read_text())
    after = yaml.safe_load((out / "llm.yaml").read_text())
    assert after["models"]["merge"]["model"] == "gpt-4o-mini"
    assert after["models"]["merge"]["cost_per_1k_output_tokens"] == 0.0006
    before["models"]["merge"].update(model="gpt-4o-mini", cost_per_1k_input_tokens=0.00015,
                                     cost_per_1k_output_tokens=0.0006)
    assert after == before
    assert (out / "api.yaml").read_text() == (capacity.settings.config_dir() / "api.yaml").read_text()


def test_verdict_reads_run_files(tmp_path, capsys):
    a = write(tmp_path / "a", [run([12000.0] * 12)] * 3)
    b = write(tmp_path / "b", [run([8000.0] * 12)] * 3)
    assert capacity.main(["verdict", str(a), str(b)]) == 0
    assert json.loads(capsys.readouterr().out)["adopt_b"] is True
