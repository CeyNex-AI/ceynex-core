"""Assertions for the discuss path (deviation D13, SRS 3.1.3, SRS 3.4.3).

The cheap path is the one most likely to quietly lower the project's standards:
it produces prose containing figures, beside a confidence badge, without any
agent having run. So the tests that matter here are the grounding ones — a
discussion may restate what was found and may not add to it.
"""

from __future__ import annotations

from ceynex.chat.store import Message
from ceynex.chat.turn import discuss
from ceynex.llm import FakeLLMClient

PRIOR_QUERY = "how did cinnamon exports to Germany change in 2025"

PRIOR = Message(
    role="assistant",
    content="Cinnamon exports to Germany rose 12% to USD 4.2m in 2025.",
    confidence=0.61,
    confidence_band="Moderate",
    agents_used=["export_analytics"],
    evidence=[
        {
            "source_id": "KG",
            "claim": "Germany took USD 4.2m of cinnamon in 2025.",
            "detail": "MATCH (c:Commodity)-[e:EXPORTS_TO]->(p:Country) RETURN e.value",
            "period": "2025",
        }
    ],
    forecast=[{"period": "2026", "point": 4.8, "lower": 4.1, "upper": 5.5, "unit": "USD m"}],
    unanswered=["district-level breakdown"],
)


class ScriptedLLM:
    available = True

    def __init__(self, text):
        self.text = text
        self.roles: list[str] = []
        self.user: str = ""

    async def generate(self, role, system, user, *, json_mode=False):
        self.roles.append(role)
        self.user = user
        return self.text


# --- grounding ------------------------------------------------------------


async def test_a_discussion_may_restate_the_figures_it_was_given():
    llm = ScriptedLLM("Exports reached USD 4.2m, a rise of 12% on the year.")
    result = await discuss("summarise that", PRIOR, PRIOR_QUERY, llm)

    assert result.grounded
    assert not result.degraded
    assert "4.2" in result.answer


async def test_a_discussion_that_invents_a_figure_is_discarded_whole():
    """Same posture as the merger: prose that states an unsourced number is not
    prose with one bad number in it, it is evidence the model was not reading."""
    llm = ScriptedLLM("Exports reached USD 4.2m, and 8,400,000 went to France.")
    result = await discuss("summarise that", PRIOR, PRIOR_QUERY, llm)

    assert not result.grounded
    assert "8,400,000" not in result.answer
    assert "8,400,000" in " ".join(result.rejected_figures)


async def test_the_discuss_path_inherits_the_grounding_check_it_does_not_fork_it():
    """Including its documented weakness, which is the point.

    `grounding.STRUCTURAL_DIGIT_LIMIT` is 2, so a short figure like "9.9" is
    treated as structural — the same class as "12%" or "a 3-year horizon" — and
    passes unchecked. That is a real gap, already recorded in
    `docs/ANSWERABLE_QUESTIONS.md` §9 as "string-based grounding that can't catch
    a right number on a wrong claim".

    It is inherited rather than patched here deliberately. `grounding.py`'s own
    docstring exists so the merger and the eval harness "can't silently measure
    different things"; a stronger private check in the chat path would mean the
    discussion is held to a standard the analysis beside it is not, and the
    published grounding figures would stop describing the whole system. Fix it in
    `grounding.py`, for everyone, or not at all.
    """
    from ceynex.orchestrator.grounding import STRUCTURAL_DIGIT_LIMIT

    assert STRUCTURAL_DIGIT_LIMIT == 2

    llm = ScriptedLLM("Exports reached USD 4.2m, and will hit USD 9.9m by 2029.")
    result = await discuss("summarise that", PRIOR, PRIOR_QUERY, llm)

    # "2029" is caught; "9.9" is not. Asserted so the gap is visible in the
    # suite rather than discovered by a reader of the output.
    assert not result.grounded
    assert "2029" in " ".join(result.rejected_figures)
    assert "9.9" not in " ".join(result.rejected_figures)


async def test_the_evidence_panel_survives_a_rejected_discussion():
    """The analysis was fine — only the discussion of it was not."""
    llm = ScriptedLLM("It will be USD 9.9m by 2030.")
    result = await discuss("what next", PRIOR, PRIOR_QUERY, llm)

    assert result.evidence == PRIOR.evidence


async def test_the_confidence_figure_may_be_discussed():
    """"why is the confidence 61%?" must be answerable — the number is on screen,
    so a grounding check that rejected it would make the question unanswerable."""
    llm = ScriptedLLM("The confidence is 61% because only one analysis contributed.")
    result = await discuss("why is the confidence what it is", PRIOR, PRIOR_QUERY, llm)

    assert result.grounded, result.rejected_figures


async def test_forecast_figures_may_be_discussed():
    llm = ScriptedLLM("The 2026 projection is 4.8, within a range of 4.1 to 5.5.")
    result = await discuss("explain the forecast", PRIOR, PRIOR_QUERY, llm)
    assert result.grounded, result.rejected_figures


# --- what the model is given ----------------------------------------------


async def test_the_model_sees_the_evidence_not_just_the_answer():
    llm = ScriptedLLM("Fine.")
    await discuss("where did that come from", PRIOR, PRIOR_QUERY, llm)

    assert "MATCH (c:Commodity)" in llm.user
    assert "Germany took USD 4.2m" in llm.user


async def test_the_model_is_told_what_was_not_answered():
    """So "what about districts?" gets "that was not part of this analysis"
    rather than an invented breakdown."""
    llm = ScriptedLLM("Fine.")
    await discuss("what about districts", PRIOR, PRIOR_QUERY, llm)
    assert "district-level breakdown" in llm.user


async def test_it_uses_its_own_role_so_the_spend_is_separable():
    llm = ScriptedLLM("Fine.")
    await discuss("summarise", PRIOR, PRIOR_QUERY, llm)
    assert llm.roles == ["chat"]


async def test_the_context_is_bounded():
    """This runs every turn of a conversation; an unbounded prompt would make a
    long conversation quadratically expensive."""
    from ceynex.chat.turn import MAX_CONTEXT_CHARS

    huge = Message(role="assistant", content="x" * 50_000, evidence=[])
    llm = ScriptedLLM("Fine.")
    await discuss("summarise", huge, PRIOR_QUERY, llm)
    assert len(llm.user) <= MAX_CONTEXT_CHARS + len("\n\nFollow-up: summarise")


# --- degraded mode --------------------------------------------------------


async def test_no_llm_returns_the_analysis_rather_than_an_error():
    """SRS 3.4.3. A plain form of the feature, not a silent failure."""
    result = await discuss("summarise that", PRIOR, PRIOR_QUERY, FakeLLMClient(available=False))

    assert result.degraded
    assert result.evidence == PRIOR.evidence
    assert "not available" in result.answer


async def test_a_raising_model_degrades_rather_than_failing_the_conversation():
    class Exploding:
        available = True

        async def generate(self, *args, **kwargs):
            raise RuntimeError("provider on fire")

    result = await discuss("summarise that", PRIOR, PRIOR_QUERY, Exploding())
    assert result.degraded
    assert result.evidence == PRIOR.evidence


async def test_an_empty_response_degrades():
    result = await discuss("summarise that", PRIOR, PRIOR_QUERY, ScriptedLLM(None))
    assert result.degraded


async def test_a_degraded_discussion_never_claims_the_model_produced_the_figures():
    """The distinction a reader needs: the analysis is still trustworthy, the
    discussion of it is what is missing."""
    result = await discuss("summarise", PRIOR, PRIOR_QUERY, FakeLLMClient(available=False))
    assert "not produced by the model" in result.answer


async def test_a_discussion_may_restate_the_question_it_is_about():
    """Found end to end. The question is on screen — the user typed it — so a
    follow-up echoing its year or market must not be treated as inventing one.

    Before the fix, a corpus built only from the answer discarded a good reply
    the moment it said "2025", because the analysis had declined without
    repeating the year back.
    """
    declined = Message(
        role="assistant",
        content="This question could not be answered from the data currently loaded.",
        confidence=0.05,
        evidence=[],
    )
    llm = ScriptedLLM("The 2025 figures for Germany were not available to look up.")
    result = await discuss("explain that more simply", declined, PRIOR_QUERY, llm)

    assert result.grounded, result.rejected_figures
    assert "2025" in result.answer


async def test_a_web_figure_cannot_be_laundered_into_a_follow_up():
    """D14's ordering protects the *first* turn. This protects every one after.

    Web evidence is appended after `merge()`, so it never reaches the merge LLM
    and never enters grounding on the turn that produced it. But it is persisted
    on the message, and a `discuss` turn grounds against that stored list — so
    without an explicit exclusion, "summarise that" could restate a scraped
    figure as though the analysis had produced it.
    """
    prior = Message(
        role="assistant",
        content="Cinnamon exports were steady.",
        evidence=[
            {"source_id": "KG", "claim": "Exports were USD 100.", "detail": "MATCH ..."},
            {
                "source_id": "WEB",
                "claim": "A blog says exports hit USD 987,654,321.",
                "detail": "https://example.test/post",
                "url": "https://example.test/post",
            },
        ],
    )
    result = await discuss(
        "restate that with the figure",
        prior,
        "how did cinnamon exports do?",
        FakeLLMClient("Exports reached USD 987,654,321 according to the analysis."),
    )
    assert result.grounded is False, "a web figure must not pass the grounding check"
