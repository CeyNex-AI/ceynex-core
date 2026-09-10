"""Assertions for follow-up classification (deviation D13, SRS 3.4.3).

The asymmetry is the whole point and most of these tests are about it. Sending a
discussion through the graph wastes a few seconds and a few cents. Sending a *new
question* to the discuss path answers it from the previous question's evidence —
a confident, fully-cited answer to something nobody asked. So the bar for
`discuss` is high and every ambiguous case is expected to land on `analyse`.
"""

from __future__ import annotations

import json

import pytest

from ceynex.chat.classify import TurnDecision, keyword_turn, llm_turn
from ceynex.llm import FakeLLMClient

PRIOR = "how did cinnamon exports to Germany change in 2025"
PRIOR_ANSWER = "Cinnamon exports to Germany rose 12% to USD 4.2m in 2025."


class ScriptedLLM:
    available = True

    def __init__(self, payload):
        self.payload = payload
        self.roles: list[str] = []

    async def generate(self, role, system, user, *, json_mode=False):
        self.roles.append(role)
        return self.payload


# --- a new subject always means a new analysis ----------------------------


@pytest.mark.parametrize(
    "follow_up",
    [
        "what about rubber",
        "now do the same for tea",
        "and for the United States?",
        "what did it look like in 2023",
        "compare that with coconut",
    ],
)
def test_a_follow_up_naming_something_new_re_runs_the_graph(follow_up):
    """The dangerous direction. Answering any of these from the cinnamon/Germany
    evidence would produce a well-cited answer to a question nobody asked."""
    decision = keyword_turn(follow_up, PRIOR)
    assert decision.mode == "analyse", f"{follow_up!r} -> {decision.reason}"


def test_the_reason_names_what_changed():
    """Surfaced in the trace, so a user who disagrees with the routing can see
    what the system thought it heard."""
    decision = keyword_turn("what about rubber", PRIOR)
    assert "rubber" in decision.reason


def test_repeating_the_same_subject_is_not_a_new_subject():
    """"why was cinnamon down in Germany" is still about the answer on screen."""
    decision = keyword_turn("why did cinnamon fall in Germany", PRIOR)
    assert decision.mode == "discuss"


# --- discussion of the answer already given -------------------------------


@pytest.mark.parametrize(
    "follow_up",
    [
        "what does HHI mean",
        "explain that more simply",
        "why is the confidence low",
        "summarise that in three bullets",
        "where did that figure come from",
        "can you say that in plain english",
        "tell me more",
        "why?",
    ],
)
def test_a_question_about_the_answer_does_not_re_run_the_graph(follow_up):
    decision = keyword_turn(follow_up, PRIOR)
    assert decision.mode == "discuss", f"{follow_up!r} -> {decision.reason}"
    assert decision.standalone_query is None


def test_a_follow_up_naming_nothing_is_a_discussion():
    """Catches the phrasings no word list anticipates: if there is nothing in it
    for the graph to answer, it can only be about what was already said."""
    decision = keyword_turn("go on", PRIOR)
    assert decision.mode == "discuss"
    assert "nothing new" in decision.reason


# --- the asymmetry --------------------------------------------------------


def test_an_ambiguous_follow_up_defaults_to_re_running():
    """A wrong answer is worse than a slow one."""
    decision = keyword_turn("the EU figures", PRIOR)
    assert decision.mode == "analyse"


def test_a_redirection_cue_wins_over_a_discussion_cue():
    """"why not do the same for rubber" contains "why" — but it is a redirection,
    and reading only the first cue found would send it to the wrong path."""
    decision = keyword_turn("why not do the same for rubber", PRIOR)
    assert decision.mode == "analyse"


# --- degraded mode --------------------------------------------------------


async def test_no_llm_key_still_classifies():
    """SRS 3.4.3. Conversation works with no provider at all."""
    decision = await llm_turn("what about rubber", PRIOR, PRIOR_ANSWER, FakeLLMClient(available=False))
    assert decision.mode == "analyse"
    assert decision.method == "keyword"


async def test_a_raising_classifier_falls_back_rather_than_failing_the_turn():
    class Exploding:
        available = True

        async def generate(self, *args, **kwargs):
            raise RuntimeError("provider on fire")

    decision = await llm_turn("what about rubber", PRIOR, PRIOR_ANSWER, Exploding())
    assert decision.mode == "analyse"
    assert decision.method == "llm->keyword"


# --- the LLM tier ---------------------------------------------------------


async def test_a_well_formed_discuss_verdict_is_used():
    llm = ScriptedLLM(json.dumps({"mode": "discuss", "standalone_query": None}))
    decision = await llm_turn("what does that mean", PRIOR, PRIOR_ANSWER, llm)

    assert decision.mode == "discuss"
    assert decision.method == "llm"
    assert llm.roles == ["turn_classify"], "own role, so its spend is separable"


async def test_an_analyse_verdict_carries_the_rewritten_question():
    llm = ScriptedLLM(json.dumps({
        "mode": "analyse",
        "standalone_query": "how did rubber exports to Germany change in 2025",
    }))
    decision = await llm_turn("what about rubber", PRIOR, PRIOR_ANSWER, llm)

    assert decision.mode == "analyse"
    assert decision.standalone_query == "how did rubber exports to Germany change in 2025"


async def test_an_analyse_verdict_without_a_rewrite_is_rejected():
    """The graph would receive "what about that one?" and route on nothing. The
    keyword decision at least carries the raw follow-up."""
    llm = ScriptedLLM(json.dumps({"mode": "analyse", "standalone_query": ""}))
    decision = await llm_turn("what about rubber", PRIOR, PRIOR_ANSWER, llm)

    assert decision.method == "llm->keyword"
    assert decision.standalone_query


async def test_an_unrecognised_mode_falls_back():
    llm = ScriptedLLM(json.dumps({"mode": "ponder", "standalone_query": None}))
    decision = await llm_turn("what about rubber", PRIOR, PRIOR_ANSWER, llm)
    assert decision.method == "llm->keyword"


async def test_unparseable_json_falls_back():
    decision = await llm_turn("what about rubber", PRIOR, PRIOR_ANSWER, ScriptedLLM("nope"))
    assert decision.method == "llm->keyword"
    assert decision.mode == "analyse"


async def test_the_prior_answer_is_given_to_the_classifier_but_bounded():
    """It needs the answer to judge "is this already covered", but a long answer
    must not blow the prompt budget on a call that runs every single turn."""
    captured = {}

    class Capturing:
        available = True

        async def generate(self, role, system, user, *, json_mode=False):
            captured["user"] = user
            return json.dumps({"mode": "discuss", "standalone_query": None})

    await llm_turn("why", PRIOR, "x" * 5000, Capturing())
    assert len(captured["user"]) < 2500


def test_a_decision_is_always_actionable():
    """Every path produces something the caller can execute — an `analyse` with
    no query, or a `discuss` with one, would both be unusable."""
    for follow_up in ("what about rubber", "why", "go on", "the EU figures"):
        decision: TurnDecision = keyword_turn(follow_up, PRIOR)
        if decision.mode == "analyse":
            assert decision.standalone_query
        else:
            assert decision.standalone_query is None
