"""Assertions for SRS 3.1.2 routing.

The v0 keyword router is the fallback the whole design leans on, so it is tested
harder than the LLM router. Its three invariants — never empty, recognises
out-of-scope, weights relevance — are what the orchestrator assumes downstream.
"""

import json

import pytest

from ceynex.contracts import ALL_AGENTS
from ceynex.llm import FakeLLMClient
from ceynex.orchestrator.router import (
    MIXED_SCOPE_NOTE,
    NO_TOPIC_NOTE,
    SCOPE_SENTENCE,
    keyword_route,
    llm_route,
)

# --- invariant 1: never empty -------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "hello",
        "asdfghjkl",
        "what",
        "tell me something",
        "?",
    ],
)
def test_the_route_is_never_empty(query):
    """An unrecognised query still gets an answer; silence is not a response."""
    decision = keyword_route(query)
    assert decision.route
    assert all(agent in ALL_AGENTS for agent in decision.route)


def test_an_unrecognised_query_defaults_to_export_analytics():
    assert keyword_route("hello").route == ["export_analytics"]


# --- the shapes the plan names ------------------------------------------


def test_a_single_sector_query_routes_to_that_sector():
    decision = keyword_route("What is the current price trend for cinnamon?")
    assert "agriculture_commodity" in decision.route
    assert "apparel_manufacturing" not in decision.route
    assert decision.sectors == ["agriculture"]


def test_a_cross_sector_query_routes_to_both_sectors():
    decision = keyword_route(
        "Compare the impact of rising shipping costs on tea exports versus apparel exports"
    )
    assert "agriculture_commodity" in decision.route
    assert "apparel_manufacturing" in decision.route
    assert "cross_sector" in decision.sectors


def test_a_simulation_query_routes_to_trade_economics_and_the_affected_sectors():
    decision = keyword_route(
        "How would a 5% depreciation of the Sri Lankan rupee affect apparel exports "
        "compared to agriculture?"
    )
    assert "trade_economics" in decision.route
    assert "agriculture_commodity" in decision.route
    assert "apparel_manufacturing" in decision.route
    assert "macro" in decision.sectors


def test_the_gsp_question_hands_off_to_trade_economics():
    """M3 verifies exactly this against the orchestrator on Day 6."""
    decision = keyword_route("How exposed is apparel to a loss of GSP+ status?")
    assert "trade_economics" in decision.route
    assert "apparel_manufacturing" in decision.route


def test_a_forward_looking_query_pulls_in_the_forecast_agent():
    decision = keyword_route("What will Ceylon tea export volumes look like next year?")
    assert "forecast" in decision.route
    assert "agriculture_commodity" in decision.route


def test_a_trend_query_pulls_in_export_analytics():
    decision = keyword_route(
        "Which importing country has shown the fastest growing demand for Sri Lankan apparel?"
    )
    assert "export_analytics" in decision.route


# --- invariant 2: out of scope is recognised, not routed away -----------


@pytest.mark.parametrize("query", [
    "What is the outlook for Sri Lankan gem exports?",
    "How is tourism revenue trending?",
    "What about fisheries exports?",
])
def test_an_out_of_scope_sector_is_flagged(query):
    """SRS 2.4 fixes scope. "I do not cover that" is a correct answer."""
    decision = keyword_route(query)
    assert decision.out_of_scope
    assert decision.notes, "an out-of-scope route must explain itself"
    assert decision.route, "even out-of-scope queries get an agent, so the user hears back"


def test_a_partly_covered_question_is_answered_and_its_gap_is_still_flagged():
    """"Prioritise gems or tea" is answerable about tea, and gems is still named.

    This previously asserted `not out_of_scope`, treating the flag as "the whole
    query is unanswerable". That let the mixed case through silently: the
    evaluation set showed "how does tea compare with fisheries" returning a
    confident tea answer that never mentioned fisheries. The flag now means
    "something here is not covered", which is the half the reader needs.
    """
    decision = keyword_route("Should Sri Lanka prioritise gems or tea next year?")

    assert "agriculture_commodity" in decision.route, "the tea half must still be answered"
    assert decision.out_of_scope, "the gems half must not be dropped silently"
    assert not decision.no_topic_recognized, "tea is named -- this is a mixed question, not a topic-less one"
    assert "gem" in decision.notes[0].lower()
    assert "also asks" in decision.notes[0], "a partial gap must read differently from a total one"


def test_a_question_naming_nothing_ceynex_covers_is_flagged_no_topic():
    """Regression, found live 2026-08-27: "who is Euler" names no excluded
    sector either -- it names nothing at all -- so it fell through every
    keyword list with no "out of scope" flag raised, and export_analytics'
    own `item = intent.item or "tea"` default answered with a confident,
    unrelated tea report. This is a different shape from "gems or tea": there
    is no in-scope half here to still answer.
    """
    decision = keyword_route("who is Euler")

    assert decision.out_of_scope
    assert decision.no_topic_recognized
    assert decision.notes


@pytest.mark.parametrize("query", [
    "What is the outlook for Sri Lankan gem exports?",
    "How is tourism revenue trending?",
])
def test_a_wholly_excluded_sector_is_out_of_scope_but_not_topic_less(query):
    """Distinct from the "who is Euler" case above: an excluded sector was
    still named, so this is "the wrong sector", not "no sector at all" --
    keeps `_out_of_scope_gaps`'s existing note text (naming what was asked
    about) rather than the generic no-topic one.
    """
    decision = keyword_route(query)
    assert decision.out_of_scope
    assert not decision.no_topic_recognized


@pytest.mark.parametrize("query", [
    "What is the outlook for Sri Lankan gem exports?",
    "How is tourism revenue trending?",
    "How are shipping costs affecting Sri Lankan exporters?",
    "who is Euler",
])
def test_a_question_naming_nothing_in_scope_leaves_no_half_to_answer(query):
    """`no_topic_recognized` is only one way to have no in-scope half. Naming an
    excluded topic and nothing else is the other, and the merger has to treat the
    two identically -- keyed on the narrower flag, "what is the outlook for Sri
    Lankan gem exports?" was answered with a confident tea forecast and a scope
    note stapled to the end.
    """
    assert keyword_route(query).nothing_in_scope


@pytest.mark.parametrize("query", [
    "Should Sri Lanka prioritise gems or tea next year?",
    "How does tea compare with fisheries?",
    "Are logistics costs hurting Sri Lankan tea exporters?",
])
def test_a_mixed_question_keeps_its_in_scope_half(query):
    """The flag must stay off wherever there is something real to answer,
    otherwise the mixed case regresses to a bare refusal.
    """
    decision = keyword_route(query)

    assert decision.out_of_scope, "the excluded half is still flagged"
    assert not decision.nothing_in_scope, "tea is named -- there is a half to answer"


# --- a sector comparison naming no goods still spans both sectors -------


@pytest.mark.parametrize(
    ("case", "query"),
    [
        ("X04", "Which of Sri Lanka's export sectors is most concentrated in a single market?"),
        ("X06", "Is Sri Lanka's export base becoming more or less diversified across sectors?"),
    ],
)
def test_a_sector_comparison_naming_no_commodity_fans_to_both_sector_agents(case, query):
    """`eval/questions.yaml` labels both of these
    [export_analytics, agriculture_commodity, apparel_manufacturing].

    Measured live 2026-09-03: X04 routed to export_analytics alone and answered
    about tea -- "Iraq accounts for 12.4% … HHI 0.05, relatively diversified" --
    to a question asking which sector is *most* concentrated, at confidence 0.9.
    Under-fanning is the dangerous direction: a confident answer to half the
    question reads exactly like a whole one.
    """
    decision = keyword_route(query)

    assert set(decision.route) == {
        "export_analytics", "agriculture_commodity", "apparel_manufacturing",
    }, case
    assert decision.sectors == ["cross_sector", "agriculture", "apparel"]


@pytest.mark.parametrize(
    "query",
    [
        "Which of Sri Lanka's export sectors is most concentrated in a single market?",
        "Is Sri Lanka's export base becoming more or less diversified across sectors?",
    ],
)
def test_a_sector_comparison_is_in_scope(query):
    """The half of this bug the live run did not show, because the LLM router was
    up. `keyword_route` matched no keyword group at all for these -- "concentrated"
    missed the "concentration" entry and nothing matched "diversified" -- so both
    were flagged `no_topic_recognized` and the degraded path (SRS 3.4.3) refused an
    in-scope question at the 0.15 floor.
    """
    decision = keyword_route(query)

    assert not decision.out_of_scope
    assert not decision.no_topic_recognized
    assert not decision.notes


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How exposed is Sri Lanka's apparel sector to a single buyer market?", "apparel_manufacturing"),
        ("How is the tea sector performing?", "agriculture_commodity"),
    ],
)
def test_a_single_sector_question_is_not_dragged_across_both(query, expected):
    """"sectors" is plural in CROSS_SECTOR_WORDS on purpose. A question about one
    named sector is answered by that sector's agent, not fanned to both.
    """
    decision = keyword_route(query)

    assert expected in decision.route
    other = (
        "agriculture_commodity" if expected == "apparel_manufacturing" else "apparel_manufacturing"
    )
    assert other not in decision.route


def test_a_named_two_sector_comparison_is_unchanged():
    """X07 already worked -- it names both sectors outright. Guard that the new
    group did not change its route or its weights.
    """
    decision = keyword_route("Which sector recovered faster after 2020, agriculture or apparel?")

    assert set(decision.route) == {
        "export_analytics", "agriculture_commodity", "apparel_manufacturing",
    }
    assert decision.relevance["agriculture_commodity"] == 1.0, "a named sector still outweighs an inferred one"


def test_an_inferred_sector_weighs_less_than_a_named_one():
    """Relevance is a number, not a flag (invariant 3): a question that never
    named agriculture should not weight its agent as heavily as one that did.
    """
    inferred = keyword_route("Which of Sri Lanka's export sectors is most concentrated?")
    named = keyword_route("How concentrated are Sri Lanka's tea export destinations?")

    assert inferred.relevance["agriculture_commodity"] < named.relevance["agriculture_commodity"]


# --- invariant 3: relevance is a weight ---------------------------------


def test_every_routed_agent_has_a_relevance_weight():
    decision = keyword_route(
        "How would a 10% tariff affect tea exports compared to garments?"
    )
    assert set(decision.relevance) == set(decision.route)
    assert all(0.0 < w <= 1.0 for w in decision.relevance.values())


def test_the_central_agent_outweighs_the_peripheral_ones():
    decision = keyword_route("How would losing GSP+ affect apparel revenue?")
    assert decision.relevance["trade_economics"] >= decision.relevance["apparel_manufacturing"]


def test_the_route_is_ordered_consistently():
    """Two equivalent queries must not produce differently-ordered routes."""
    a = keyword_route("tea and apparel exports trend")
    b = keyword_route("apparel and tea exports trend")
    assert a.route == b.route


def test_the_decision_converts_to_a_state_patch():
    patch = keyword_route("cinnamon price trend").as_state_patch()
    assert set(patch) == {"route", "sectors", "relevance"}


# --- the LLM router and its fallbacks -----------------------------------


async def test_llm_routing_is_used_when_it_returns_valid_json():
    llm = FakeLLMClient(
        response='{"route": ["trade_economics", "forecast"], "sectors": ["macro"], '
        '"relevance": {"trade_economics": 1.0, "forecast": 0.5}, '
        '"out_of_scope": false, "reason": "a policy simulation"}'
    )
    decision = await llm_route("what if tariffs rise?", llm)
    assert decision.method == "llm"
    assert decision.route == ["trade_economics", "forecast"]
    assert decision.relevance["forecast"] == 0.5


async def test_an_unavailable_llm_falls_back_to_keywords():
    """SRS 3.4.3 — the system keeps routing when the provider does not."""
    decision = await llm_route("cinnamon price trend", FakeLLMClient(available=False))
    assert decision.method == "keyword"
    assert "agriculture_commodity" in decision.route
    assert decision.notes


async def test_unparseable_json_falls_back_to_keywords():
    decision = await llm_route("cinnamon price trend", FakeLLMClient(response="not json at all"))
    assert decision.method == "llm->keyword"
    assert "agriculture_commodity" in decision.route


async def test_a_hallucinated_agent_name_falls_back_rather_than_routing_nowhere():
    """A plausible but wrong name would otherwise produce an empty fan-out."""
    llm = FakeLLMClient(response='{"route": ["shipping_agent", "customs_agent"], "sectors": []}')
    decision = await llm_route("cinnamon price trend", llm)
    assert decision.method == "llm->keyword"
    assert all(agent in ALL_AGENTS for agent in decision.route)


async def test_partially_valid_routes_keep_only_the_real_agents():
    llm = FakeLLMClient(
        response='{"route": ["forecast", "made_up_agent"], "sectors": ["agriculture"], '
        '"relevance": {"forecast": 0.9, "made_up_agent": 1.0}}'
    )
    decision = await llm_route("will tea rise?", llm)
    assert decision.route == ["forecast"]
    assert "made_up_agent" not in decision.relevance


async def test_llm_route_flags_no_topic_when_out_of_scope_names_no_sector():
    """ROUTER_SYSTEM tells the model to omit agriculture/apparel from sectors
    when the question isn't about a real sector at all -- llm_route must read
    that the same way keyword_route derives it from its own keyword hits.
    """
    llm = FakeLLMClient(
        response='{"route": ["export_analytics"], "sectors": [], '
        '"relevance": {"export_analytics": 0.3}, '
        '"out_of_scope": true, "reason": "not about Sri Lankan trade"}'
    )
    decision = await llm_route("who is Euler?", llm)
    assert decision.out_of_scope
    assert decision.no_topic_recognized


async def test_llm_route_does_not_flag_no_topic_for_a_mixed_question():
    llm = FakeLLMClient(
        response='{"route": ["agriculture_commodity"], "sectors": ["agriculture"], '
        '"relevance": {"agriculture_commodity": 0.7}, '
        '"out_of_scope": true, "reason": "also asks about fisheries"}'
    )
    decision = await llm_route("how does tea compare with fisheries?", llm)
    assert decision.out_of_scope
    assert not decision.no_topic_recognized


# --- both routers explain themselves the same way ------------------------


@pytest.mark.parametrize(
    ("sectors", "expected_note"),
    [
        ([], NO_TOPIC_NOTE),
        (["agriculture"], MIXED_SCOPE_NOTE),
    ],
)
async def test_llm_route_populates_notes_for_both_out_of_scope_shapes(sectors, expected_note):
    """Regression, found live 2026-09-03: `llm_route` built its RouteDecision with
    no `notes`, so `graph.py` produced `errors = ["out_of_scope: "]` with an empty
    note and `merger.py` substituted its generic fallback -- telling the user they
    had named a sector CeyNex does not cover when they had named no sector at all.
    Wrong for every LLM-router out-of-scope verdict, not just the shipping query
    that surfaced it.
    """
    llm = FakeLLMClient(
        response=json.dumps(
            {
                "route": ["export_analytics"],
                "sectors": sectors,
                "relevance": {"export_analytics": 0.3},
                "out_of_scope": True,
                "reason": "x",
            }
        )
    )
    decision = await llm_route("how are shipping costs affecting exporters?", llm)

    assert decision.notes == [expected_note]
    assert "names a sector" not in decision.notes[0]


async def test_llm_route_leaves_notes_empty_when_the_question_is_in_scope():
    """The note is an explanation of a refusal. An answered question has none."""
    llm = FakeLLMClient(
        response='{"route": ["agriculture_commodity"], "sectors": ["agriculture"], '
        '"relevance": {"agriculture_commodity": 1.0}, '
        '"out_of_scope": false, "reason": "names tea"}'
    )
    decision = await llm_route("how have tea exports grown?", llm)

    assert decision.notes == []


def test_both_routers_end_an_out_of_scope_note_with_the_same_scope_sentence():
    """The two routers built their own copies of this sentence and drifted --
    `keyword_route` said "apparel (HS 61/62)" where the no-topic branch said
    "apparel (HS 61/62) exports", and `llm_route` said nothing at all. One
    constant now, so a future edit cannot desynchronise them again.
    """
    assert keyword_route("who is Euler").notes[0].endswith(SCOPE_SENTENCE)
    assert keyword_route("how is tourism revenue trending?").notes[0].endswith(SCOPE_SENTENCE)
    assert NO_TOPIC_NOTE.endswith(SCOPE_SENTENCE)
    assert MIXED_SCOPE_NOTE.endswith(SCOPE_SENTENCE)


# --- freight / logistics is named as excluded, not left topic-less -------


@pytest.mark.parametrize(
    "query",
    [
        "How are shipping costs affecting Sri Lankan exporters?",
        "What are ocean freight rates doing?",
        "How have container rates moved this year?",
    ],
)
def test_a_freight_question_names_freight_as_the_reason(query):
    """CeyNex holds no freight, shipping-cost or logistics data (SRS 2.4 scope is
    agriculture and apparel). Before this these questions matched no keyword group
    at all and fell to the "names nothing CeyNex covers" branch, which is a worse
    explanation than the true one: the topic is recognised, it is just not held.
    """
    decision = keyword_route(query)

    assert decision.out_of_scope
    assert not decision.no_topic_recognized, "freight is a recognised exclusion, not an unrecognised topic"
    assert any(word in decision.notes[0] for word in ("freight", "shipping cost", "container rate"))
    assert decision.route, "an out-of-scope query still gets an agent so the user hears back"


def test_freight_alongside_an_in_scope_sector_still_answers_the_in_scope_half():
    """Same mixed-question rule the gems/tea case established: naming an excluded
    topic does not throw away the half CeyNex can answer.
    """
    decision = keyword_route("Are logistics costs hurting Sri Lankan tea exporters?")

    assert "agriculture_commodity" in decision.route, "the tea half must still be answered"
    assert decision.out_of_scope
    assert not decision.no_topic_recognized
    assert "also asks" in decision.notes[0]


def test_shipping_as_a_business_model_is_not_a_freight_question():
    """`OUT_OF_SCOPE_WORDS` matches on substrings, so a bare "shipping" would
    exclude "drop-shipping apparel" -- a question squarely about apparel. The
    two-word entries are what keep that in scope.
    """
    decision = keyword_route("How is our drop-shipping apparel channel performing?")

    assert not decision.out_of_scope
    assert "apparel_manufacturing" in decision.route


# --- policy questions reach the agent that holds the corpus (D10) --------


@pytest.mark.parametrize(
    "query",
    [
        "What does India's Foreign Trade Policy say about imports from Sri Lanka?",
        "What non-tariff measures does the European Union apply to imported spices?",
        "Does the Netherlands' foreign trade policy identify Sri Lanka as a priority market?",
        "Does the United Kingdom's trade strategy keep preferential access for Sri Lankan tea?",
        "Compare the preferential access Sri Lankan apparel receives in the UK and the EU.",
    ],
)
def test_a_foreign_policy_question_routes_to_trade_economics(query):
    """`export_analytics` holds only Sri Lanka's own trade flows.

    Measured on the policy set before this: these questions routed to
    `export_analytics` alone and returned 0 evidence at 0.15 confidence, because
    the agent they reached has nothing to say about another government's policy
    (EVALUATION.md §7).
    """
    assert "trade_economics" in keyword_route(query).route


def test_an_ordinary_analytics_question_still_does_not_get_trade_economics():
    """The policy words must not drag the simulation agent into every query."""
    route = keyword_route("Which country takes the largest share of Sri Lanka's tea exports?").route

    assert "trade_economics" not in route
    assert "export_analytics" in route
