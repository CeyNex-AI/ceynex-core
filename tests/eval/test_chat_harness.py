"""The multi-turn harness's own scoring, over scripted frame logs.

The set itself is checked too: a conversation whose first turn expects a
discussion, or whose expectation names a mode the classifier cannot return,
would score nothing and look like a pass.
"""

from __future__ import annotations

import pytest

from eval.chat_harness import CONVERSATIONS, load_conversations, report, score_turn


def _frames(*events):
    return [{"event": event, "data": data} for event, data in events]


DONE_ANALYSE = ("done", {
    "failed": False,
    "answer": {"answer": "Iraq took 12.4% of tea exports, USD 170,000,000.", "route": ["export_analytics"],
               "unanswered": [], "degraded": False, "confidence": 0.7,
               "evidence": [{"source_id": "KG", "claim": "USD 170,000,000", "detail": "MATCH"}]},
    "usage": {"calls": 3, "cost_usd": 0.0021},
})


# --- the set is well formed ------------------------------------------------------


def test_the_set_loads_with_unique_ids_and_no_discussion_first():
    conversations = load_conversations(CONVERSATIONS)
    ids = [c["id"] for c in conversations]
    assert len(ids) == len(set(ids))
    for conversation in conversations:
        first = conversation["turns"][0].get("expect", {})
        assert first.get("mode") != "discuss", f"{conversation['id']} discusses nothing"
        assert len(conversation["turns"]) >= 2, f"{conversation['id']} is not a conversation"


def test_an_unknown_expectation_key_is_rejected(tmp_path):
    bad = tmp_path / "c.yaml"
    bad.write_text("conversations:\n  - id: X\n    turns:\n      - query: q\n        expect: {mood: happy}\n")
    with pytest.raises(ValueError, match="unknown expectation"):
        load_conversations(bad)


def test_an_unknown_mode_is_rejected(tmp_path):
    bad = tmp_path / "c.yaml"
    bad.write_text("conversations:\n  - id: X\n    turns:\n      - query: q\n        expect: {mode: chat}\n")
    with pytest.raises(ValueError, match="mode must be"):
        load_conversations(bad)


# --- scoring ------------------------------------------------------------------------


def test_a_first_turn_counts_as_an_analysis_without_a_turn_frame():
    result = score_turn("C01", 0, {"query": "q", "expect": {"mode": "analyse", "route": ["export_analytics"]}},
                        _frames(("start", {}), ("kg_query", {}), DONE_ANALYSE), 1200.0)
    assert result.mode == "first"
    assert result.checks == {"mode": True, "no_clarify": True, "route": True, "answered": True}
    assert result.passed and result.frames == 3 and result.cost_usd == 0.0021


def test_a_discussion_is_scored_on_its_mode_and_its_grounding():
    frames = _frames(
        ("start", {}),
        ("turn", {"mode": "discuss", "method": "llm"}),
        ("answer_delta", {"text": "It means one buyer dominates. "}),
        ("done", {"failed": False, "answer": {"answer": "It means one buyer dominates.", "grounded": True,
                                              "route": ["export_analytics"], "unanswered": []}}),
    )
    result = score_turn("C01", 1, {"query": "explain", "expect": {"mode": "discuss", "grounded": True}},
                        frames, 900.0)
    assert result.mode == "discuss" and result.answer_deltas == 1
    assert result.checks == {"mode": True, "no_clarify": True, "grounded": True}


def test_a_withdrawn_discussion_fails_the_grounding_check():
    frames = _frames(
        ("turn", {"mode": "discuss", "method": "llm"}),
        ("answer_reset", {"reason": "ungrounded"}),
        ("done", {"failed": False, "answer": {"answer": "original shown", "grounded": False}}),
    )
    result = score_turn("C01", 1, {"query": "explain", "expect": {"mode": "discuss", "grounded": True}},
                        frames, 900.0)
    assert result.checks["grounded"] is False and not result.passed


def test_a_rewrite_must_carry_what_the_reader_named():
    frames = _frames(
        ("turn", {"mode": "analyse", "method": "llm",
                  "standalone_query": "How fast have rubber exports grown over the last five years?"}),
        DONE_ANALYSE,
    )
    expect = {"mode": "analyse", "standalone_contains": ["Rubber"], "route": ["export_analytics"]}
    result = score_turn("C02", 1, {"query": "now rubber", "expect": expect}, frames, 5000.0)
    assert result.checks["standalone_contains"] is True
    assert result.standalone_query.startswith("How fast have rubber")


def test_a_discussion_where_an_analysis_was_expected_is_a_miss():
    frames = _frames(("turn", {"mode": "discuss", "method": "keyword"}),
                     ("done", {"failed": False, "answer": {"answer": "x", "grounded": True}}))
    result = score_turn("C02", 1, {"query": "now rubber", "expect": {"mode": "analyse"}}, frames, 100.0)
    assert result.checks["mode"] is False


def test_the_gate_is_scored_both_ways():
    asked = _frames(("clarify_gate", {"asked": True}), ("clarify", {"pending_id": 4, "options": ["tea", "cinnamon", "both"]}),
                    ("done", {"failed": False, "clarify": True}))
    result = score_turn("C05", 0, {"query": "tea and cinnamon", "expect": {"clarify": True}}, asked, 50.0)
    assert result.mode == "clarify" and result.checks == {"clarify": True}

    silent = score_turn("C05", 1, {"query": "q", "expect": {"mode": "analyse"}},
                        _frames(("turn", {"mode": "analyse"}), DONE_ANALYSE), 50.0)
    assert silent.checks["no_clarify"] is True

    unexpected = score_turn("C01", 1, {"query": "q", "expect": {"mode": "analyse"}}, asked, 50.0)
    assert unexpected.checks["no_clarify"] is False


def test_a_partial_answer_that_states_a_limit_is_still_answered():
    """SAD §4.1: naming what could not be covered while answering the rest is
    the required behaviour, not a refusal. The first draft scored it as one."""
    partial = ("done", {"failed": False, "answer": {
        "answer": "Iraq took 10.9%, USD 149,621,564. Tea volume data is not available.",
        "route": ["export_analytics"], "confidence": 0.62,
        "unanswered": ["tea export volume has no usable observations"],
        "evidence": [{"source_id": "KG", "claim": "USD 149,621,564", "detail": "MATCH"}]}})
    result = score_turn("C01", 0, {"query": "q", "expect": {"mode": "analyse"}}, _frames(partial), 10.0)
    assert result.checks["answered"] is True and result.stated_limit and not result.refused


def test_a_refusal_is_a_miss_unless_the_set_allows_it():
    declined = ("done", {"failed": False, "answer": {"answer": "That period is not in the record.",
                                                     "route": ["export_analytics"], "confidence": 0.15,
                                                     "unanswered": ["2035 is not in the record"],
                                                     "evidence": []}})
    strict = score_turn("C06", 0, {"query": "2035", "expect": {"mode": "analyse"}}, _frames(declined), 10.0)
    assert strict.checks["answered"] is False
    allowed = score_turn("C06", 0, {"query": "2035", "expect": {"mode": "analyse", "refused_ok": True}},
                         _frames(declined), 10.0)
    assert allowed.passed and allowed.refused


def test_a_failed_turn_is_a_result_with_its_error():
    frames = _frames(("error", {"message": "neo4j down"}), ("done", {"failed": True}))
    result = score_turn("C01", 0, {"query": "q", "expect": {"mode": "analyse"}}, frames, 5.0)
    assert result.mode == "failed" and result.error == "neo4j down" and not result.passed


def test_the_report_carries_every_rate_with_its_denominator():
    results = [
        score_turn("C01", 0, {"query": "q", "expect": {"mode": "analyse"}}, _frames(DONE_ANALYSE), 1000.0),
        score_turn("C01", 1, {"query": "explain", "expect": {"mode": "discuss", "grounded": True}},
                   _frames(("turn", {"mode": "discuss"}),
                           ("done", {"failed": False, "answer": {"answer": "x", "grounded": True}})), 800.0),
        score_turn("C02", 1, {"query": "rubber", "expect": {"mode": "analyse", "standalone_contains": ["rubber"]}},
                   _frames(("turn", {"mode": "analyse", "standalone_query": "tea?"}), DONE_ANALYSE), 4000.0),
    ]
    summary = report(results)
    assert summary["turns"] == 3 and summary["conversations"] == 2
    assert summary["classifier"]["mode_accuracy"] == {"rate": 1.0, "of": 2}
    assert summary["classifier"]["rewrite_fidelity"] == {"rate": 0.0, "of": 1}
    assert summary["discuss"]["grounded"] == {"rate": 1.0, "of": 1}
    assert summary["gate"]["asked_where_expected"] == {"rate": None, "of": 0}
    assert summary["by_mode"]["discuss"]["turns"] == 1
    assert summary["turns_passing_every_check"] == {"rate": pytest.approx(0.6667, abs=1e-4), "of": 3}
