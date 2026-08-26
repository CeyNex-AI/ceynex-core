"""Assertions for SRS 3.1.2 routing.

The v0 keyword router is the fallback the whole design leans on, so it is tested
harder than the LLM router. Its three invariants — never empty, recognises
out-of-scope, weights relevance — are what the orchestrator assumes downstream.
"""

import pytest

from ceynex.contracts import ALL_AGENTS
from ceynex.llm import FakeLLMClient
from ceynex.orchestrator.router import keyword_route, llm_route

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
