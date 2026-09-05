"""Tests for `agents.common.finish` -- SRS 3.1.3's grounding guarantee applied
at the per-agent layer, not just the merge layer.

`orchestrator/merger.py`'s `_reject_ungrounded_prose` only ever checked the
merge LLM's own prose. An agent's own explanation prose (`generate_explanation`)
was protected by a prompt instruction alone, with nothing downstream verifying
it -- and it reaches the user unchecked via `merger.compose_deterministic`
(which uses `summary` verbatim) whenever the merge LLM is unavailable or its
own prose gets rejected. These tests pin the fix: `finish` now runs the same
`ungrounded_figures` check against a corpus built from *this agent's own*
figures/evidence/assumptions before trusting its explanation prose.
"""

from __future__ import annotations

from ceynex.agents.common import AgentDeps, finish
from ceynex.contracts import Evidence, new_state
from ceynex.llm import FakeLLMClient


class FakeKG:
    async def run(self, cypher, params=None):
        raise AssertionError("finish() should not touch the knowledge graph")


def _deps(response: str | None) -> AgentDeps:
    return AgentDeps(kg=FakeKG(), llm=FakeLLMClient(response=response))


async def test_explanation_prose_is_used_when_it_states_only_grounded_figures():
    deps = _deps("Tea exports reached USD 1,431,567,471 in 2025, up from prior years.")
    patch = await finish(
        agent="agriculture_commodity",
        state=new_state("How is tea exporting?", "tester"),
        deps=deps,
        summary="Tea export value was USD 1,431,567,471 in 2025.",
        figures={"export_value_usd": 1431567471.0},
        evidence=[Evidence(source_id="KG", claim="Tea export value was USD 1,431,567,471 in 2025.", detail="")],
        assumptions=[],
    )
    output = patch["agent_outputs"]["agriculture_commodity"]
    assert output["summary"] == "Tea exports reached USD 1,431,567,471 in 2025, up from prior years."
    assert not output["degraded"]


async def test_explanation_prose_stating_an_ungrounded_figure_is_discarded():
    """Regression: found live-reasoning, not from a bug report -- the
    explanation LLM is told never to invent a number (`EXPLANATION_SYSTEM`),
    but nothing verified it actually didn't. This is the one figure in this
    test that appears in no figure, no evidence claim, and no evidence detail.
    """
    deps = _deps("Tea exports reached USD 999,999,999 in 2025, a record high.")
    patch = await finish(
        agent="agriculture_commodity",
        state=new_state("How is tea exporting?", "tester"),
        deps=deps,
        summary="Tea export value was USD 1,431,567,471 in 2025.",
        figures={"export_value_usd": 1431567471.0},
        evidence=[Evidence(source_id="KG", claim="Tea export value was USD 1,431,567,471 in 2025.", detail="")],
        assumptions=[],
    )
    output = patch["agent_outputs"]["agriculture_commodity"]
    assert "999,999,999" not in output["summary"]
    # Falls back to the deterministic summary this agent itself computed --
    # never a blank answer, and never a figure from nowhere.
    assert output["summary"] == "Tea export value was USD 1,431,567,471 in 2025."
    assert output["degraded"] is True
    assert patch["degraded"] is True


async def test_a_figure_only_present_in_assumptions_is_still_grounded():
    """The corpus is the agent's whole finding, not just `figures`/`evidence`
    narrowly -- an assumption stating a rate the explanation then restates
    must not be flagged, the same way the merge-level check already treats
    assumptions as legitimate corpus (`merger._grounding_corpus`)."""
    deps = _deps("At the assumed 9.5% MFN rate, the effect would be smaller.")
    patch = await finish(
        agent="trade_economics",
        state=new_state("What if GSP+ were lost?", "tester"),
        deps=deps,
        summary="Losing GSP+ would raise the effective tariff.",
        figures={},
        evidence=[Evidence(source_id="KG", claim="No TradeAgreement coverage found for this HS code.", detail="")],
        assumptions=["Reverts to the documented MFN rate of 9.5% absent a retrieved document rate."],
    )
    output = patch["agent_outputs"]["trade_economics"]
    assert "9.5%" in output["summary"]
    assert not output["degraded"]


async def test_a_short_structural_number_is_never_flagged():
    """`orchestrator.grounding.STRUCTURAL_DIGIT_LIMIT` -- a horizon/count this
    short isn't a claim needing its own evidence. Same floor the merge-level
    check already relies on; this pins that agents get it too."""
    deps = _deps("Reported across 12 partner markets this year.")
    patch = await finish(
        agent="export_analytics",
        state=new_state("How many markets?", "tester"),
        deps=deps,
        summary="12 markets reported data.",
        figures={"market_count": 12.0},
        evidence=[Evidence(source_id="KG", claim="12 markets reported data.", detail="")],
        assumptions=[],
    )
    output = patch["agent_outputs"]["export_analytics"]
    assert not output["degraded"]


async def test_no_explanation_call_when_summary_is_empty():
    """Unrelated to grounding, but a real edge the new corpus-building must
    not choke on: `finish` skips the LLM call entirely when there is nothing
    to explain (an agent that found nothing)."""
    llm = FakeLLMClient(response="This should never be used.")
    patch = await finish(
        agent="forecast",
        state=new_state("Forecast rubber?", "tester"),
        deps=AgentDeps(kg=FakeKG(), llm=llm),
        summary="",
        figures={},
        evidence=[],
        assumptions=[],
    )
    assert llm.usage.calls == 0
    assert patch["agent_outputs"]["forecast"]["summary"] == ""
