"""Assertions for SRS 3.1.2 merging — the difference between orchestration and concatenation.

The first test in this file is the one that matters. If the answer names the
agents, the SRS is violated no matter how good the rest of it is.
"""

import pytest

from ceynex.contracts import AgentOutput, Evidence, failed_output, new_state
from ceynex.llm import FakeLLMClient
from ceynex.orchestrator.merger import (
    NO_TOPIC_MARKER,
    OUT_OF_SCOPE_PREFIX,
    _dq_severities_from_evidence,
    compose_deterministic,
    dedupe_evidence,
    detect_conflicts,
    merge,
    unanswered_from_outputs,
)

AGENT_NAMES = (
    "export analytics",
    "agriculture commodity",
    "apparel manufacturing",
    "trade economics",
    "export_analytics",
    "trade_economics",
)


def output(agent, *, summary="A finding.", figures=None, evidence=None, confidence=0.8,
           degraded=False, assumptions=None, error=None):
    out = AgentOutput(
        agent=agent,
        summary=summary,
        figures=figures or {},
        assumptions=assumptions or [],
        evidence=evidence if evidence is not None else [ev("KG", "A claim.")],
        confidence=confidence,
        degraded=degraded,
    )
    if error:
        out["error"] = error
    return out


def ev(source, claim, detail="MATCH (n) RETURN n"):
    return Evidence(source_id=source, claim=claim, detail=detail)


def state(query="a question", outputs=None, route=None, relevance=None, errors=None):
    st = new_state(query, "tester")
    st["agent_outputs"] = outputs or {}
    st["route"] = route or list(st["agent_outputs"])
    st["relevance"] = relevance or dict.fromkeys(st["route"], 1.0)
    if errors is not None:
        st["errors"] = errors
    return st


# --- the rule the SRS states outright -----------------------------------


async def test_the_answer_never_names_the_agents():
    """SRS 3.1.2 forbids presenting outputs as separate, disconnected responses."""
    outputs = {
        "export_analytics": output("export_analytics", summary="Exports grew 4%."),
        "trade_economics": output("trade_economics", summary="A shock costs USD 2m."),
    }
    result = await merge(state(outputs=outputs), FakeLLMClient(available=False))

    lowered = result.answer.lower()
    for name in AGENT_NAMES:
        assert f"the {name} agent" not in lowered, f"answer names an agent: {name}"
    assert "agent says" not in lowered


async def test_the_degraded_answer_is_also_organised_by_finding():
    """Degrading drops the prose, not the structure."""
    outputs = {
        "export_analytics": output("export_analytics", summary="Exports grew 4%."),
        "forecast": output("forecast", summary="Next year is projected flat."),
    }
    result = await merge(state(outputs=outputs), FakeLLMClient(available=False))
    assert "Exports grew 4%." in result.answer
    assert "Next year is projected flat." in result.answer
    assert result.degraded


async def test_higher_confidence_findings_lead():
    outputs = {
        "export_analytics": output("export_analytics", summary="Weak finding.", confidence=0.3),
        "trade_economics": output("trade_economics", summary="Strong finding.", confidence=0.9),
    }
    answer = (await merge(state(outputs=outputs), FakeLLMClient(available=False))).answer
    assert answer.index("Strong finding.") < answer.index("Weak finding.")


# --- conflicts are surfaced, never averaged -----------------------------


def test_a_direction_disagreement_is_a_conflict():
    outputs = {
        "trade_economics": output("trade_economics", figures={"apparel_impact_pct": 0.04}),
        "forecast": output("forecast", figures={"apparel_impact_pct": -0.03}),
    }
    conflicts = detect_conflicts(outputs)
    assert len(conflicts) == 1
    assert conflicts[0].kind == "direction"


def test_a_large_magnitude_gap_is_a_conflict():
    outputs = {
        "a": output("export_analytics", figures={"total_export_value_usd": 100.0}),
        "b": output("forecast", figures={"total_export_value_usd": 180.0}),
    }
    assert detect_conflicts(outputs)[0].kind == "magnitude"


def test_close_figures_are_agreement_not_conflict():
    outputs = {
        "a": output("export_analytics", figures={"total_export_value_usd": 100.0}),
        "b": output("forecast", figures={"total_export_value_usd": 103.0}),
    }
    assert detect_conflicts(outputs) == []


def test_a_figure_only_one_agent_reports_is_never_a_conflict():
    outputs = {"a": output("export_analytics", figures={"cagr": 0.05})}
    assert detect_conflicts(outputs) == []


async def test_both_conflicting_figures_reach_the_answer():
    """Averaging destroys the information. The SRS wants the disagreement shown."""
    outputs = {
        "trade_economics": output("trade_economics", figures={"apparel_impact_pct": 0.04}),
        "forecast": output("forecast", figures={"apparel_impact_pct": -0.03}),
    }
    result = await merge(state(outputs=outputs), FakeLLMClient(available=False))

    assert result.conflicts
    assert "0.04" in result.answer
    assert "-0.03" in result.answer
    assert "0.005" not in result.answer, "the midpoint appeared — something averaged"


async def test_conflicts_are_stated_even_if_the_llm_omits_them():
    """Surfacing a disagreement is a requirement, not the model's judgement call."""
    outputs = {
        "trade_economics": output("trade_economics", figures={"apparel_impact_pct": 0.04}),
        "forecast": output("forecast", figures={"apparel_impact_pct": -0.03}),
    }
    llm = FakeLLMClient(response="Exports look broadly stable this year.")
    result = await merge(state(outputs=outputs), llm)
    assert "disagree" in result.answer.lower()


# --- partial results ------------------------------------------------------


async def test_one_failure_still_produces_an_answer():
    """SAD §4.1 — answer with what succeeded and say what could not be answered."""
    outputs = {
        "export_analytics": output("export_analytics", summary="Exports grew 4%."),
        "agriculture_commodity": failed_output("agriculture_commodity", "neo4j down"),
    }
    result = await merge(
        state(outputs=outputs, route=["export_analytics", "agriculture_commodity"]),
        FakeLLMClient(available=False),
    )
    assert "Exports grew 4%." in result.answer
    assert result.unanswered
    assert result.agents_used == ["export_analytics"]


async def test_an_agent_that_never_reported_is_named_too():
    outputs = {"export_analytics": output("export_analytics")}
    result = await merge(
        state(outputs=outputs, route=["export_analytics", "forecast"]),
        FakeLLMClient(available=False),
    )
    assert any("projection" in gap for gap in result.unanswered)


async def test_gaps_are_stated_even_if_the_llm_omits_them():
    outputs = {
        "export_analytics": output("export_analytics", summary="Exports grew 4%."),
        "forecast": failed_output("forecast", "no model"),
    }
    llm = FakeLLMClient(response="Exports grew four percent last year.")
    result = await merge(
        state(outputs=outputs, route=["export_analytics", "forecast"]), llm
    )
    assert "not covered" in result.answer.lower()


async def test_a_gap_already_phrased_as_cannot_or_not_available_is_not_duplicated():
    """Regression: the LLM's own phrasing for a decline ("cannot be
    provided", "is not available") matched none of the original marker
    words, so the deterministic append fired anyway and visibly duplicated
    the same reason a second time. Found live 2026-08-26.
    """
    outputs = {
        "export_analytics": output("export_analytics", summary="Exports grew 4%."),
        "forecast": failed_output("forecast", "no model"),
    }
    llm = FakeLLMClient(
        response="Exports grew four percent. A forecast cannot be provided for this item."
    )
    result = await merge(
        state(outputs=outputs, route=["export_analytics", "forecast"]), llm
    )
    assert result.answer.lower().count("cannot be provided") == 1


async def test_a_gap_restated_in_different_words_is_not_duplicated():
    """Regression, found live 2026-08-27 from a real user query ("do we have
    cinnamon data?"): the decline's own text ("No sourced export volume
    series is available for cinnamon; only tea is covered for this question.
    There are no compatible price, volume, or export-value series available
    to substitute for the requested target.") got restated a second,
    near-verbatim time as "Not covered: <the same text>" -- the merge LLM's
    own independent paraphrase used none of the marker words from the
    2026-08-26 fix above, a second recurrence of the same bug class with
    different wording.
    """
    summary = (
        "No sourced export volume series is available for cinnamon; only tea is covered "
        "for this question. There are no compatible price, volume, or export-value series "
        "available to substitute for the requested target."
    )
    outputs = {
        "agriculture_commodity": output(
            "agriculture_commodity", summary=summary, figures={}, confidence=0.20,
        ),
    }
    llm = FakeLLMClient(
        response=(
            "There is no available sourced export volume series specifically for cinnamon. "
            "The data provided only covers tea for this question, and there are no compatible "
            "price, volume, or export-value series that can be substituted for cinnamon."
        )
    )
    result = await merge(state(outputs=outputs, route=["agriculture_commodity"]), llm)

    assert "Not covered:" not in result.answer
    assert result.answer == llm._response


async def test_a_gap_using_entirely_different_words_is_still_appended():
    """The word-overlap check above must not become so permissive that a
    genuinely unstated gap silently disappears."""
    outputs = {
        "export_analytics": output("export_analytics", summary="Exports grew 4%.", confidence=0.8),
        "apparel_manufacturing": failed_output("apparel_manufacturing", "no model"),
    }
    llm = FakeLLMClient(response="Exports grew four percent last year.")
    result = await merge(
        state(outputs=outputs, route=["export_analytics", "apparel_manufacturing"]), llm
    )
    assert "Not covered:" in result.answer


# --- list-shaped evidence the merge LLM never saw --------------------------


def test_is_list_shaped_recognizes_a_name_list_not_a_number_heavy_claim():
    from ceynex.orchestrator.merger import _is_list_shaped

    names = ", ".join(
        ["United States", "United Kingdom", "Italy", "Germany", "Netherlands",
         "Canada", "France", "Belgium", "Australia", "China", "India"]
    )
    assert _is_list_shaped(f"All 11 destination countries, ranked by export value: {names}.")

    numeric = (
        "United States took 38.5% of Sri Lanka's apparel export value in 2025, "
        "USD 1,149,365,364 of USD 2,984,685,707."
    )
    assert not _is_list_shaped(numeric)


async def test_a_list_shaped_evidence_claim_is_referenced_when_the_prose_omits_it():
    """Regression, found live 2026-08-27 from "give the names of all
    countries...": _merge_prompt never forwards evidence to the merge LLM,
    only each finding's summary/figures/assumptions, so it correctly reported
    "15 countries" as a count but had no way to know the names existed --
    and said so falsely ("were not provided") even though the full list was
    already sitting in the evidence panel the whole time.
    """
    names = ", ".join(
        ["United States", "United Kingdom", "Italy", "Germany", "Netherlands", "Canada",
         "France", "Belgium", "Australia", "China", "India", "Ireland", "Mexico", "Spain", "Japan"]
    )
    outputs = {
        "export_analytics": output(
            "export_analytics",
            summary="Sri Lanka exports apparel to 15 countries.",
            figures={"partner_count": 15.0},
            evidence=[ev("KG", f"All 15 destination countries, ranked by export value: {names}.")],
            confidence=0.8,
        ),
    }
    llm = FakeLLMClient(
        response="Sri Lanka exports apparel to 15 countries. The specific names were not provided."
    )
    result = await merge(state(outputs=outputs, route=["export_analytics"]), llm)

    assert "evidence panel" in result.answer.lower()


async def test_a_list_already_referenced_is_not_duplicated():
    names = ", ".join(
        ["United States", "United Kingdom", "Italy", "Germany", "Netherlands", "Canada",
         "France", "Belgium", "Australia", "China", "India", "Ireland", "Mexico", "Spain", "Japan"]
    )
    outputs = {
        "export_analytics": output(
            "export_analytics",
            summary="x",
            figures={},
            evidence=[ev("KG", f"All 15 destination countries, ranked by export value: {names}.")],
            confidence=0.8,
        ),
    }
    llm = FakeLLMClient(response="See the evidence panel for the full list of 15 countries.")
    result = await merge(state(outputs=outputs, route=["export_analytics"]), llm)

    assert result.answer.lower().count("evidence panel") == 1


async def test_a_short_claim_never_triggers_the_list_note():
    outputs = {
        "export_analytics": output(
            "export_analytics", summary="Exports grew 4%.", figures={"cagr": 0.04}, confidence=0.8,
        ),
    }
    llm = FakeLLMClient(response="Exports grew 4% last year.")
    result = await merge(state(outputs=outputs, route=["export_analytics"]), llm)

    assert "evidence panel" not in result.answer.lower()


# --- fully out-of-topic questions -----------------------------------------


async def test_a_no_topic_question_does_not_carry_the_agents_confident_answer():
    """Regression, found live 2026-08-27 from a real user query ("who is
    Euler"): export_analytics' own `item = intent.item or "tea"` default
    answered with a confident (0.9), fully unrelated tea report, and the
    merge LLM's paraphrase of it read as "does not relate to ... but here is
    a tea report anyway" -- the router's out_of_scope flag only ever added a
    disclaimer note, it never stopped the irrelevant finding from being
    merged in as if it were real content.
    """
    outputs = {
        "export_analytics": output(
            "export_analytics",
            summary="Sri Lanka exported tea worth USD 1.4bn to 141 markets in 2025.",
            figures={"total_export_value_usd": 1_431_567_471.0},
            confidence=0.9,
        ),
    }
    llm = FakeLLMClient(response="This should never be reached.")
    result = await merge(
        state(
            outputs=outputs,
            route=["export_analytics"],
            errors=[f"{OUT_OF_SCOPE_PREFIX}the question does not name anything CeyNex covers.", NO_TOPIC_MARKER],
        ),
        llm,
    )

    assert "tea" not in result.answer.lower()
    assert "1.4" not in result.answer
    assert result.agents_used == []
    assert result.evidence == []
    assert result.confidence < 0.20
    assert "does not name anything ceynex covers" in result.answer.lower()


async def test_a_mixed_out_of_scope_question_still_gets_its_real_answer():
    """The suppression above must not fire for the existing "gems or tea"
    shape -- only a question naming nothing in scope at all is topic-less.
    """
    outputs = {
        "agriculture_commodity": output(
            "agriculture_commodity", summary="Tea prices rose 4% this year.", confidence=0.8,
        ),
    }
    llm = FakeLLMClient(response="Tea prices rose 4% this year.")
    result = await merge(
        state(
            outputs=outputs,
            route=["agriculture_commodity"],
            errors=[f"{OUT_OF_SCOPE_PREFIX}the question also asks about gem, which CeyNex does not cover."],
        ),
        llm,
    )

    assert "tea" in result.answer.lower()
    assert result.agents_used == ["agriculture_commodity"]


async def test_an_empty_out_of_scope_note_no_longer_claims_a_sector_was_named():
    """Belt to the router's braces (both routers now populate `notes`). When the
    fallback does fire it must not invent a reason: found live 2026-09-03, an
    empty note made "who was Leonhard Euler?" and "how are shipping costs
    affecting exporters?" both answer "part of the question names a sector CeyNex
    does not cover" -- neither names any sector.
    """
    llm = FakeLLMClient(response="")
    result = await merge(state(outputs={}, route=[], errors=[OUT_OF_SCOPE_PREFIX]), llm)

    assert result.unanswered == ["part of the question is outside what CeyNex covers"]
    assert "names a sector" not in result.answer


async def test_an_honest_refusal_reads_as_a_gap_not_a_conflicting_finding():
    """Regression: a real "cinnamon exports outlook" question was narrated as
    "uncertain due to conflicting findings" -- one analysis gave a real
    forecast, the other (a confidence=0.20 SAD Section 4.1 refusal, no
    error key set, empty figures) was presented as a peer FINDING that
    appeared to disagree with it, even though detect_conflicts never found
    an actual numeric conflict (a refusal has no figures to conflict with).
    Found live 2026-08-26.
    """
    outputs = {
        "forecast": output(
            "forecast", summary="Exports are projected at USD 224.7m.",
            figures={"forecast_next_usd": 224_700_000.0}, confidence=0.68,
        ),
        "agriculture_commodity": output(
            "agriculture_commodity",
            summary="No registered national export-value model was requested for this item.",
            figures={}, confidence=0.20,
        ),
    }
    result = await merge(
        state(outputs=outputs, route=["forecast", "agriculture_commodity"]), FakeLLMClient(available=False)
    )

    assert not result.conflicts
    assert "Exports are projected at USD 224.7m." in result.answer
    assert any("No registered national export-value model" in gap for gap in result.unanswered)
    assert "Not covered: No registered national export-value model" in result.answer


async def test_a_decline_another_finding_already_answered_is_not_a_gap():
    """Regression, found live 2026-09-03: S03 came back correct and complete from
    the graph -- rubber concentration, HHI, market count -- and still carried "No
    sourced export volume series is held for rubber" in `unanswered`, because
    agriculture_commodity was routed too and holds only tea volume. The same shape
    hit CO1 (coconut), X02 and M01. A correct answer read as half-failed.
    """
    outputs = {
        "export_analytics": output(
            "export_analytics",
            summary="Rubber export destinations are moderately concentrated: Germany takes 18.2%, HHI 0.09 across 62 markets in 2025.",
            figures={"hhi": 0.09},
            evidence=[ev("KG", "Rubber export value to Germany was USD 62,113,000 in 2025.")],
            confidence=0.82,
        ),
        "agriculture_commodity": output(
            "agriculture_commodity",
            summary="No sourced export volume series is held for rubber.",
            figures={}, confidence=0.20,
        ),
    }
    result = await merge(
        state(
            query="How concentrated are Sri Lanka's rubber export destinations?",
            outputs=outputs,
            route=["export_analytics", "agriculture_commodity"],
        ),
        FakeLLMClient(available=False),
    )

    assert result.unanswered == []
    assert "not covered" not in result.answer.lower()
    assert "Germany takes 18.2%" in result.answer


async def test_a_decline_nothing_else_covered_is_still_a_gap():
    """The other half of the rule. `agriculture_commodity` correctly refuses a
    district question (no sourced district share exists), and the market-share
    answer alongside it does not cover districts -- so the reader must be told.
    Suppressing this would be the failure `_out_of_scope_gaps` was written to
    prevent: omitting a real limit is as wrong as inventing a figure.
    """
    outputs = {
        "export_analytics": output(
            "export_analytics",
            summary="Cinnamon export value was USD 312m in 2025, with Mexico the largest market at 21.4%.",
            figures={"share": 0.214},
            evidence=[ev("KG", "Cinnamon export value to Mexico was USD 66,768,000 in 2025.")],
            confidence=0.82,
        ),
        "agriculture_commodity": output(
            "agriculture_commodity",
            summary="The graph records cinnamon producing districts, but has no sourced numerical district share.",
            figures={}, confidence=0.20,
        ),
    }
    result = await merge(
        state(
            query="Which district contributes the largest share of cinnamon production?",
            outputs=outputs,
            route=["export_analytics", "agriculture_commodity"],
        ),
        FakeLLMClient(available=False),
    )

    assert any("district share" in gap for gap in result.unanswered)
    assert "district" in result.answer.lower()


async def test_unanswered_from_outputs_suppresses_the_same_declines_as_merge():
    """`merge()` and this helper compute the same list by two paths, and the API
    route uses the helper. They diverged once already -- the bug this function's
    own docstring records -- so the suppression has to land in both or a correct
    answer keeps its spurious caveat in the API response only.
    """
    outputs = {
        "export_analytics": output(
            "export_analytics",
            summary="Rubber export destinations are moderately concentrated: Germany takes 18.2%.",
            figures={"hhi": 0.09},
            evidence=[ev("KG", "Rubber export value to Germany was USD 62,113,000 in 2025.")],
            confidence=0.82,
        ),
        "agriculture_commodity": output(
            "agriculture_commodity",
            summary="No sourced export volume series is held for rubber.",
            figures={}, confidence=0.20,
        ),
    }
    final = state(
        query="How concentrated are Sri Lanka's rubber export destinations?",
        outputs=outputs,
        route=["export_analytics", "agriculture_commodity"],
    )
    result = await merge(final, FakeLLMClient(available=False))

    assert unanswered_from_outputs(final) == result.unanswered == []


async def test_everything_failing_says_so_rather_than_returning_nothing():
    outputs = {
        "export_analytics": failed_output("export_analytics", "neo4j down"),
        "forecast": failed_output("forecast", "neo4j down"),
    }
    result = await merge(state(outputs=outputs), FakeLLMClient(available=False))
    assert result.answer
    assert "could not be answered" in result.answer.lower()
    assert result.agents_used == []
    assert result.degraded


# --- evidence -------------------------------------------------------------


def test_identical_evidence_from_two_agents_collapses_to_one():
    same = ev("KG", "Germany took 12%.", "MATCH (a) RETURN a")
    outputs = {
        "export_analytics": output("export_analytics", evidence=[same]),
        "trade_economics": output("trade_economics", evidence=[dict(same)]),
    }
    assert len(dedupe_evidence(outputs)) == 1


def test_the_same_claim_from_two_sources_is_kept_as_corroboration():
    outputs = {
        "export_analytics": output("export_analytics", evidence=[ev("KG", "Exports rose.")]),
        "forecast": output("forecast", evidence=[ev("MODEL", "Exports rose.")]),
    }
    merged = dedupe_evidence(outputs)
    assert len(merged) == 2
    assert {e["source_id"] for e in merged} == {"KG", "MODEL"}


def test_evidence_attribution_survives_deduplication():
    outputs = {"a": output("export_analytics", evidence=[ev("UN_COMTRADE", "A claim.")])}
    assert dedupe_evidence(outputs)[0]["source_id"] == "UN_COMTRADE"


async def test_merged_evidence_reaches_the_state_patch():
    outputs = {"export_analytics": output("export_analytics")}
    result = await merge(state(outputs=outputs), FakeLLMClient(available=False))
    patch = result.as_state_patch()
    assert set(patch) == {"final_answer", "final_confidence", "merged_evidence"}
    assert patch["merged_evidence"]


# --- confidence -----------------------------------------------------------


async def test_confidence_is_derived_and_banded():
    outputs = {"export_analytics": output("export_analytics", confidence=0.9)}
    result = await merge(state(outputs=outputs), FakeLLMClient(available=False))
    assert 0.0 < result.confidence < 1.0
    assert result.band in {"High", "Moderate", "Low", "Very low"}


async def test_a_failed_agent_lowers_confidence():
    complete = {"export_analytics": output("export_analytics", confidence=0.9)}
    partial = {
        "export_analytics": output("export_analytics", confidence=0.9),
        "forecast": failed_output("forecast", "no data"),
    }
    high = await merge(state(outputs=complete), FakeLLMClient(available=False))
    low = await merge(
        state(outputs=partial, route=["export_analytics", "forecast"]),
        FakeLLMClient(available=False),
    )
    assert low.confidence < high.confidence


# --- the DQ penalty is now actually wired to real data --------------------


async def test_a_dq_flag_in_the_evidence_lowers_the_aggregate_confidence():
    """Regression: aggregate_confidence's `dq` term had a correct, tested
    formula, but nothing in the real orchestrator graph ever called merge()
    with a non-default dq_severities -- it always contributed 0 in
    production regardless of real dq_flag data. merge() now reads severity
    back out of any DQ_FLAG evidence entry in the merged set (the shape
    agriculture_commodity._with_dq_flags already emits) instead of relying
    on an external caller to supply it.
    """
    dq_evidence = ev(
        "DQ_FLAG", "Material data-quality flag.",
        "dq_flag: source_a=FAOSTAT; source_b=PINK_SHEET; metric=price; pct_diff=10.05; severity=material",
    )
    clean = {"export_analytics": output("export_analytics", confidence=0.9)}
    flagged = {
        "export_analytics": output(
            "export_analytics", confidence=0.9, evidence=[ev("KG", "A claim."), dq_evidence]
        ),
    }
    high = await merge(state(outputs=clean), FakeLLMClient(available=False))
    lower = await merge(state(outputs=flagged), FakeLLMClient(available=False))
    assert lower.confidence == pytest.approx(high.confidence - 0.05)


async def test_severe_dq_flags_from_different_agents_both_count():
    """DQ_FLAG evidence is read from the whole merged set, not per-agent --
    two agents each surfacing their own real flag both contribute.
    """
    flag_a = ev(
        "DQ_FLAG", "Severe flag A.",
        "dq_flag: source_a=A; source_b=B; metric=price; pct_diff=50.0; severity=severe",
    )
    flag_b = ev(
        "DQ_FLAG", "Severe flag B.",
        "dq_flag: source_a=C; source_b=D; metric=export_volume; pct_diff=60.0; severity=severe",
    )
    outputs = {
        "agriculture_commodity": output(
            "agriculture_commodity", confidence=0.9, evidence=[ev("KG", "A."), flag_a]
        ),
        "apparel_manufacturing": output(
            "apparel_manufacturing", confidence=0.9, evidence=[ev("EDB", "B."), flag_b]
        ),
    }
    result = await merge(state(outputs=outputs), FakeLLMClient(available=False))
    baseline = await merge(
        state(outputs={"agriculture_commodity": output("agriculture_commodity", confidence=0.9)}),
        FakeLLMClient(available=False),
    )
    assert result.confidence == pytest.approx(baseline.confidence - 0.20)  # 2 * severe (0.10 each)


def test_dq_severities_from_evidence_ignores_non_flag_and_malformed_entries():
    good = ev(
        "DQ_FLAG", "A flag.",
        "dq_flag: source_a=A; source_b=B; metric=price; pct_diff=10.0; severity=material",
    )
    not_a_flag = ev("KG", "Unrelated.", "MATCH (n) RETURN n")
    malformed = Evidence(source_id="DQ_FLAG", claim="No severity field.", detail="dq_flag: metric=price")
    assert _dq_severities_from_evidence([good, not_a_flag, malformed]) == ["material"]


# --- the deterministic composer --------------------------------------------


def test_compose_deterministic_handles_having_nothing():
    assert compose_deterministic("q", {}, [], []).strip()


@pytest.mark.parametrize("marker", ["Not covered"])
def test_compose_deterministic_names_the_gaps(marker):
    text = compose_deterministic("q", {"a": output("export_analytics")}, [], ["the forecast failed"])
    assert marker in text
