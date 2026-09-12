"""Assertions for the clarification gate (SRS 3.1.2, 3.4.3; deviation D13).

The gate's whole justification is that it is *rare and free*. So the assertions
that matter are the negative ones: on the overwhelming majority of questions it
must decide not to ask, and it must decide that without spending an LLM call.
A clarifier that interrupts a clear question is worse than no clarifier at all.
"""

from __future__ import annotations

from ceynex.chat.clarify import (
    Clarification,
    clarification_needed,
    llm_clarify,
    template_clarification,
)
from ceynex.llm import FakeLLMClient

# --- Stage A: free, deterministic, and mostly silent -------------------------


def test_a_plain_single_item_question_is_never_clarified():
    assert clarification_needed("What is the current price trend for cinnamon?") is None


def test_a_question_naming_nothing_we_cover_is_not_clarified():
    """An out-of-scope question gets today's honest refusal, not a clarifying
    question about a domain CeyNex does not cover."""
    assert clarification_needed("What is the outlook for Sri Lankan gem exports?") is None


def test_two_commodities_in_one_question_are_caught():
    """`parse_intent` stops at the first matching category, so "tea and cinnamon"
    silently becomes tea and cinnamon is dropped with no trace anywhere. That
    dropped half is the strongest trigger the gate has."""
    trigger = clarification_needed("How did tea and cinnamon exports do last year?")
    assert trigger is not None
    assert set(trigger.options) >= {"tea", "cinnamon"}


def test_an_explicit_comparison_is_left_alone():
    """The router already handles a comparison correctly. Re-litigating a case
    that works is exactly how a clarifier becomes an obstacle."""
    for phrasing in (
        "Compare tea and cinnamon exports",
        "tea versus cinnamon export value",
        "tea vs cinnamon",
    ):
        assert clarification_needed(phrasing) is None, phrasing


def test_a_simulation_with_no_resolvable_destination_is_caught():
    trigger = clarification_needed("What if that country raised tariffs on our tea?")
    assert trigger is not None


def test_a_simulation_naming_a_real_market_is_left_alone():
    assert clarification_needed("What if the European Union raised tariffs on tea by 10%?") is None


def test_a_simulation_naming_a_region_is_left_alone():
    assert clarification_needed("What if Asia raised tariffs on Sri Lankan tea?") is None


# --- Stage B: phrasing only, never the decision ------------------------------


async def test_no_llm_still_asks_a_real_question():
    """SRS 3.4.3. The degraded clarifier is the same detection with templated
    phrasing — a plainer feature, not an absent one."""
    trigger = clarification_needed("How did tea and cinnamon exports do last year?")
    assert trigger is not None
    result = await llm_clarify(trigger, FakeLLMClient(available=False))
    assert result is not None
    assert result.question
    assert set(result.options) >= {"tea", "cinnamon"}
    assert result.method == "template"


async def test_the_model_may_veto_the_gate():
    """The phrasing may resolve an ambiguity the syntactic check missed, and the
    clarifier is allowed to say so rather than ask anyway."""
    trigger = clarification_needed("How did tea and cinnamon exports do last year?")
    assert trigger is not None
    result = await llm_clarify(trigger, FakeLLMClient('{"ask": false}'))
    assert result is None


async def test_an_unparseable_model_reply_falls_back_to_the_template():
    trigger = clarification_needed("How did tea and cinnamon exports do last year?")
    assert trigger is not None
    result = await llm_clarify(trigger, FakeLLMClient("not json at all"))
    assert result is not None
    assert result.method == "template"


async def test_a_raising_model_never_fails_the_turn():
    class Exploding:
        available = True

        async def generate(self, *args, **kwargs):
            raise RuntimeError("provider on fire")

    trigger = clarification_needed("How did tea and cinnamon exports do last year?")
    assert trigger is not None
    result = await llm_clarify(trigger, Exploding())
    assert result is not None
    assert result.method == "template"


def test_the_template_always_offers_an_escape():
    """"Just answer it" is not optional. A gate with no way past it is a wall."""
    trigger = clarification_needed("How did tea and cinnamon exports do last year?")
    assert trigger is not None
    assert template_clarification(trigger).allow_skip


def test_composing_an_answer_produces_a_standalone_query():
    trigger = clarification_needed("How did tea and cinnamon exports do last year?")
    assert trigger is not None
    clar = template_clarification(trigger)
    composed = Clarification.compose(clar.original_query, ["cinnamon"])
    assert "cinnamon" in composed.lower()
    assert composed != clar.original_query


def test_the_gate_is_silent_on_every_question_the_project_actually_asks():
    """The feature's entire justification is that it is rare.

    A clarifier that interrupts real questions is worse than no clarifier: it
    adds a round trip to the common case to serve the uncommon one. The
    30-question evaluation set is the closest thing this project has to a corpus
    of questions a real user asks, so silence across all of it is the assertion
    that matters — and it is checked here rather than asserted in prose, because
    a widened keyword list is exactly the kind of change that would quietly break
    it.
    """
    import pathlib

    import yaml

    path = pathlib.Path(__file__).resolve().parents[2] / "eval" / "questions.yaml"
    loaded = yaml.safe_load(path.read_text())
    questions = loaded["questions"] if isinstance(loaded, dict) else loaded

    fired = [q["id"] for q in questions if clarification_needed(q["question"]) is not None]
    assert fired == [], f"the gate interrupted questions it should have answered: {fired}"


async def test_the_model_phrases_but_never_changes_the_choice():
    """Asked about tea and cinnamon, the real provider once returned a question
    while dropping an option from its list — offering the reader a choice they
    could not then make. The options are the deterministic check's, always; the
    model supplies wording and nothing else.
    """
    trigger = clarification_needed("How did tea and cinnamon exports do last year?")
    assert trigger is not None
    narrowed = FakeLLMClient(
        '{"ask": true, "question": "Tea or cinnamon?", "options": ["tea"]}'
    )
    result = await llm_clarify(trigger, narrowed)
    assert result is not None
    assert result.method == "llm"
    assert result.question == "Tea or cinnamon?"
    assert result.options == template_clarification(trigger).options

    widened = FakeLLMClient(
        '{"ask": true, "question": "which?", "options": ["tea", "sapphires"]}'
    )
    result = await llm_clarify(trigger, widened)
    assert result is not None
    assert "sapphires" not in result.options


# --- "both" is not offered (docs/DEFERRED.md, the owner's call of 2026-09-12) --
#
# "both" composed a query the single-item export agent answered for the first
# item only, stating the other was unavailable: a choice that promised more than
# the analysis delivers. The owner's call was to stop offering it.


def test_the_template_offers_the_items_and_nothing_else():
    for query, items in [
        ("How did tea and cinnamon exports do last year?", ["tea", "cinnamon"]),
        ("How are tea, cinnamon and rubber exports doing?", ["tea", "cinnamon", "rubber"]),
    ]:
        trigger = clarification_needed(query)
        assert trigger is not None, query
        options = template_clarification(trigger).options
        assert sorted(options) == sorted(items), query
        assert "both" not in options


async def test_a_phrasing_that_offers_a_combination_is_replaced_by_the_template():
    """The real provider's own wording, before rule 5: it promised "both"."""
    trigger = clarification_needed("How did tea and cinnamon exports do last year?")
    assert trigger is not None
    for question in ("tea, cinnamon, or both?", "Would you like all of them?",
                     "Tea, cinnamon or each of them?"):
        llm = FakeLLMClient(f'{{"ask": true, "question": "{question}", "options": ["tea"]}}')
        result = await llm_clarify(trigger, llm)
        assert result is not None
        assert result.question == template_clarification(trigger).question, question
        assert result.method == "template"
        assert "both" not in result.options


async def test_a_phrasing_that_offers_one_item_at_a_time_is_kept():
    """The guard is narrow: "either" and "which" are choices, not combinations."""
    trigger = clarification_needed("How did tea and cinnamon exports do last year?")
    assert trigger is not None
    llm = FakeLLMClient('{"ask": true, "question": "Which, tea or cinnamon? Either is fine.", "options": ["tea", "cinnamon"]}')
    result = await llm_clarify(trigger, llm)
    assert result is not None
    assert result.method == "llm"
    assert result.question.startswith("Which, tea or cinnamon?")

