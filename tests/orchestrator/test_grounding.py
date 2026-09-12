"""Assertions for SRS 3.1.3 — a composed answer may not state a figure nothing gave it.

Two layers are covered here: the comparison itself
(`ceynex.orchestrator.grounding`) and the guarantee it backs in `merge()`.

The guarantee only has teeth now that an LLM key is configured. With no model
available the merge falls back to `compose_deterministic`, which can only
restate what agents wrote, so the interesting failure was unreachable and every
number in `docs/EVALUATION.md` was measured on that path. These tests exercise
the path a live key opens up.
"""

from ceynex.contracts import AgentOutput, Evidence, new_state
from ceynex.llm import FakeLLMClient
from ceynex.orchestrator.grounding import numbers_in, ungrounded_figures
from ceynex.orchestrator.merger import compose_deterministic, merge


def ev(source, claim, detail="MATCH (n) RETURN n"):
    return Evidence(source_id=source, claim=claim, detail=detail)


def output(
    agent, *, summary="A finding.", figures=None, evidence=None, confidence=0.8, assumptions=None
):
    return AgentOutput(
        agent=agent,
        summary=summary,
        figures=figures or {},
        assumptions=assumptions or [],
        evidence=evidence if evidence is not None else [ev("KG", "A claim.")],
        confidence=confidence,
        degraded=False,
    )


def state(query="a question", outputs=None):
    st = new_state(query, "tester")
    st["agent_outputs"] = outputs or {}
    st["route"] = list(st["agent_outputs"])
    st["relevance"] = dict.fromkeys(st["route"], 1.0)
    return st


# --- the comparison ------------------------------------------------------


def test_a_figure_present_in_the_corpus_is_grounded():
    assert ungrounded_figures("Exports reached USD 1,234,567.", ["1234567"]) == []


def test_a_figure_present_nowhere_is_reported():
    assert ungrounded_figures("Exports reached USD 999,888", ["1234567"]) == ["999,888"]


def test_a_sentence_ending_period_is_absorbed_into_the_reported_figure():
    """Documenting a quirk rather than fixing it: `NUMBER`'s trailing `\\.?\\d*`
    swallows a full stop, so the reported string is "999,888." not "999,888".

    Harmless — the comparison strips the dot before matching, so only the
    human-readable report differs — and the regex is shared with
    `eval.harness.ungrounded`, whose measured figures in docs/EVALUATION.md
    were produced by exactly this behaviour. Tightening it here would silently
    change what a published number means.
    """
    assert ungrounded_figures("Exports reached USD 999,888.", ["1234567"]) == ["999,888."]


def test_commas_do_not_decide_the_match():
    """1,234,567 in prose and 1234567 in a figures dict are the same number."""
    assert ungrounded_figures("USD 1,234,567", ["USD 1234567"]) == []
    assert ungrounded_figures("USD 1234567", ["USD 1,234,567"]) == []


def test_short_digit_strings_are_structural_not_claims():
    """A 3-year horizon or a 10% shock needs no evidence line of its own.

    Without this the check would fire on nearly every answer and stop
    discriminating between a horizon and a fabricated export value.
    """
    assert ungrounded_figures("Over 3 years a 10% shock affects 12 markets.", []) == []


def test_a_rounded_restatement_is_accepted():
    """1234.5 in prose against 1234.52 in evidence is a rounding, not an invention."""
    assert ungrounded_figures("about 1234.5", ["1234.52"]) == []


def test_numbers_in_ignores_empty_text():
    assert numbers_in("") == set()
    assert numbers_in(None) == set()


# --- the guarantee -------------------------------------------------------


async def test_prose_inventing_a_figure_is_discarded():
    """The whole point: a fluent sentence carrying a number no agent reported
    must not reach the reader, however well it reads."""
    outputs = {
        "export_analytics": output(
            "export_analytics",
            summary="Tea exports reached USD 1,270,000,000 in 2023.",
            figures={"export_value_usd": 1_270_000_000},
            evidence=[ev("COMTRADE", "Tea exports 2023: USD 1,270,000,000.")],
        )
    }
    llm = FakeLLMClient(response="Tea exports reached USD 4,555,666,777 in 2023.")

    result = await merge(state(outputs=outputs), llm)

    assert result.ungrounded == ["4,555,666,777"]
    assert "4,555,666,777" not in result.answer
    assert "1,270,000,000" in result.answer


async def test_the_discarded_answer_falls_back_to_the_deterministic_composition():
    outputs = {
        "export_analytics": output(
            "export_analytics",
            summary="Tea exports reached USD 1,270,000,000.",
            figures={"export_value_usd": 1_270_000_000},
        )
    }
    llm = FakeLLMClient(response="Exports were USD 9,876,543,210.")
    st = state(outputs=outputs)

    result = await merge(st, llm)

    expected = compose_deterministic(st["query"], outputs, [], [])
    assert result.answer == expected.strip()


async def test_grounded_prose_survives_untouched():
    outputs = {
        "export_analytics": output(
            "export_analytics",
            summary="Tea exports reached USD 1,270,000,000 in 2023.",
            figures={"export_value_usd": 1_270_000_000},
            evidence=[ev("COMTRADE", "Tea exports 2023: USD 1,270,000,000.")],
        )
    }
    llm = FakeLLMClient(response="Tea exports were worth USD 1,270,000,000 in 2023.")

    result = await merge(state(outputs=outputs), llm)

    assert result.ungrounded == []
    assert "1,270,000,000" in result.answer


async def test_a_magnitude_quoted_from_the_question_is_not_an_invention():
    """docs/EVALUATION.md records exactly this as 2 of its 2 ungrounded figures.

    "If the EU raises tariffs by 10%" puts the number in the question, not in
    any evidence entry. The eval metric counts that as ungrounded on purpose —
    it measures traceability to a *source* — but rejecting the answer over it
    would be wrong, so the runtime corpus includes the query.
    """
    outputs = {
        "trade_economics": output(
            "trade_economics",
            summary="The simulated impact is USD 45,600,000.",
            figures={"tariff_impact_usd": -45_600_000},
        )
    }
    llm = FakeLLMClient(response="A 10.5% tariff would cost about USD 45,600,000.")

    result = await merge(
        state(query="what if the EU raises tariffs by 10.5%", outputs=outputs), llm
    )

    assert result.ungrounded == []
    assert "10.5" in result.answer


async def test_a_figure_stated_only_in_an_assumption_is_grounded():
    """`_merge_prompt` forwards assumptions, so the model may legitimately quote one."""
    outputs = {
        "trade_economics": output(
            "trade_economics",
            summary="Preference loss was simulated.",
            assumptions=["MFN tariff re-imposed at 9.5% (config constant, WITS not ingested)"],
        )
    }
    llm = FakeLLMClient(response="Losing GSP+ re-imposes an MFN rate of 9.5%.")

    result = await merge(state(outputs=outputs), llm)

    assert result.ungrounded == []


async def test_the_degraded_path_never_reports_ungrounded_figures():
    """No model, no invention — the deterministic composition restates summaries."""
    outputs = {
        "export_analytics": output(
            "export_analytics", summary="Exports grew 4.7% to USD 1,270,000,000."
        )
    }

    result = await merge(state(outputs=outputs), FakeLLMClient(available=False))

    assert result.ungrounded == []
    assert result.degraded


# --- direction-aware grounding (CEYNEX_GROUNDING=direction, EVALUATION.md §13) --
#
# Trade economics reports an impact signed ("USD -161,815,198"), and the model
# states it unsigned with a word for the fall ("a decrease of USD 161,815,198").
# That is §1's grounding class 1 and §9's X09 misses. The rule accepts it only
# when the figure's own sentence says the value fell.

from ceynex.orchestrator.grounding import FELL, split_sentences  # noqa: E402

SIGNED = ["Losing GSP+ changes export value by USD -161,815,198 (-5.7%)."]


def test_a_fall_stated_without_its_sign_is_grounded_when_its_sentence_says_so():
    answer = "Export value would decrease by USD 161,815,198 a year."
    assert ungrounded_figures(answer, SIGNED, direction_aware=True) == []
    assert ungrounded_figures(answer, SIGNED, direction_aware=False) == ["161,815,198"]


def test_without_a_word_for_the_fall_the_unsigned_figure_stays_ungrounded():
    """The sign flip the strict check exists to catch: a fall told as a rise."""
    answer = "Export value would grow by USD 161,815,198 a year."
    assert ungrounded_figures(answer, SIGNED, direction_aware=True) == ["161,815,198"]


def test_the_word_must_be_in_the_figures_own_sentence():
    answer = "Export value would decrease. The change is USD 161,815,198 a year."
    assert ungrounded_figures(answer, SIGNED, direction_aware=True) == ["161,815,198"]


def test_a_negative_figure_is_never_grounded_by_a_positive_one():
    assert ungrounded_figures("It fell by -1,234,567 in all.", ["It was 1,234,567 in all."],
                              direction_aware=True) == ["-1,234,567"]


def test_direction_is_the_default_and_strict_restores_the_old_check(monkeypatch):
    """The default flipped when EVALUATION.md §13's rule held (2026-09-12)."""
    answer = "Export value would decrease by USD 161,815,198 a year."
    monkeypatch.delenv("CEYNEX_GROUNDING", raising=False)
    assert ungrounded_figures(answer, SIGNED) == []
    monkeypatch.setenv("CEYNEX_GROUNDING", "strict")
    assert ungrounded_figures(answer, SIGNED) == ["161,815,198"]


def test_the_words_that_say_a_value_fell_are_whole_words():
    assert FELL.search("a decline of") and FELL.search("would be cut") and FELL.search("down from")
    assert not FELL.search("the breakdown by market") and not FELL.search("a downturn-free year")


def test_splitting_loses_and_moves_nothing():
    text = "Tea rose 4.25 percent. The U.S. was not covered! Why? 2024 was the last year."
    assert "".join(split_sentences(text)) == text
    assert len(split_sentences(text)) == 4
