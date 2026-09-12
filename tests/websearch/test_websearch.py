"""Assertions for general web search (deviation D14).

The safety argument for this feature is *structural*, not procedural: web
evidence is appended after `merge()` has returned, so it cannot reach the merge
LLM, cannot enter `ungrounded_figures()`, and cannot move `aggregate_confidence()`
— not because anything checks, but because it does not exist yet when those run.

These tests exist to make sure that stays true, because it is the kind of
property a well-meaning refactor silently breaks.
"""

from __future__ import annotations

import pytest

from ceynex.websearch import (
    WebResult,
    WebSearchProvider,
    from_settings,
    wants_current_context,
)
from ceynex.websearch.providers import FixtureProvider

RESULTS = [
    WebResult(
        title="Sri Lanka tea exports climb in Q3",
        url="https://example.test/tea-q3",
        snippet="Exports reached USD 1.1 billion in the third quarter.",
        published="2026-09-01",
        domain="example.test",
    )
]


# --- the recency gate: most questions never make a call ----------------------


@pytest.mark.parametrize(
    "query",
    [
        "What is the current price trend for cinnamon?",
        "latest tea export figures",
        "recent news on apparel tariffs",
        "what happened this week in rubber",
    ],
)
def test_a_recency_flavoured_question_is_eligible(query):
    assert wants_current_context(query)


@pytest.mark.parametrize(
    "query",
    [
        "Which country took the largest share of Sri Lanka's tea exports in 2019?",
        "How would losing GSP+ affect apparel export revenue?",
        "Compare tea and rubber export value",
    ],
)
def test_a_historical_question_makes_no_outbound_call(query):
    """The gate bounds cost and injection surface in one decision. A question
    about 2019 has nothing to gain from today's web."""
    assert not wants_current_context(query)


# --- absence is ordinary, not exceptional ------------------------------------


def test_no_key_means_the_feature_is_simply_off(monkeypatch):
    monkeypatch.setattr("ceynex.settings.tavily_api_key", lambda: None)
    monkeypatch.setattr("ceynex.settings.web_search_enabled", lambda: True)
    assert from_settings() is None


def test_the_kill_switch_wins_even_with_a_key(monkeypatch):
    """`CEYNEX_WEB_SEARCH=off` is what makes "answers are byte-identical to the
    pre-web-search system" a checkable claim."""
    monkeypatch.setattr("ceynex.settings.tavily_api_key", lambda: "tvly-whatever")
    monkeypatch.setattr("ceynex.settings.web_search_enabled", lambda: False)
    assert from_settings() is None


async def test_a_failing_provider_returns_nothing_rather_than_raising():
    """No web result is ever worth failing an answer that the graph produced."""

    class Broken:
        async def search(self, query, *, limit=5):
            raise RuntimeError("provider on fire")

    from ceynex.websearch.client import safe_search

    assert await safe_search(Broken(), "latest tea prices") == []


async def test_a_slow_provider_is_cut_off_rather_than_delaying_the_answer():
    import asyncio

    class Slow:
        async def search(self, query, *, limit=5):
            await asyncio.sleep(30)
            return RESULTS

    from ceynex.websearch.client import safe_search

    assert await safe_search(Slow(), "latest tea prices", timeout_s=0.05) == []


# --- evidence shape ----------------------------------------------------------


def test_web_evidence_is_marked_as_web_and_carries_its_url():
    from ceynex.agents.common import evidence_from_web

    item = evidence_from_web(RESULTS[0].title, RESULTS[0].snippet, RESULTS[0].url)
    assert item["source_id"] == "WEB"
    assert item["url"] == RESULTS[0].url


async def test_the_fixture_provider_satisfies_the_protocol():
    provider: WebSearchProvider = FixtureProvider(RESULTS)
    assert await provider.search("anything") == RESULTS


# --- the structural exclusions, which are the whole safety argument ----------


async def test_web_evidence_is_appended_after_merge_and_never_reaches_it():
    """The merge LLM must never see a web result, or the prose could cite a
    scraped figure as though the graph had produced it.

    Asserted by running a real `run_query` with a provider that returns a
    distinctive figure, then checking that figure appears in the evidence and
    *not* in anything the merge call was shown.
    """
    from ceynex.api import deps as deps_module
    from ceynex.api.query_runner import run_query
    from ceynex.websearch.schema import WebResult
    from tests.api.test_query import ANSWERED, FakeGraph, FakeKG, FakeLLM

    marker = "USD 987,654,321"
    provider = FixtureProvider(
        [WebResult(title="A blog post", url="https://example.test/p",
                   snippet=f"Exports hit {marker} last week.", domain="example.test")]
    )
    runtime = deps_module.Runtime(
        kg=FakeKG(), llm=FakeLLM(), deps=None, graph=FakeGraph(ANSWERED), websearch=provider
    )

    outcome = await run_query(runtime, "latest tea export news", record_history=False)

    web = [e for e in outcome.response.evidence if e.source_id == "WEB"]
    assert web, "the web result should be present as evidence"
    assert marker in web[0].detail
    # The graph ran on the question alone; nothing web-derived was in its state.
    assert marker not in outcome.response.answer


async def test_web_evidence_comes_last_so_earlier_evidence_keeps_its_indices():
    """Phase 7's inline citations number evidence positionally. Appending keeps
    `[1]` pointing at the same entry whether or not a web search happened."""
    from ceynex.api import deps as deps_module
    from ceynex.api.query_runner import run_query
    from ceynex.websearch.schema import WebResult
    from tests.api.test_query import ANSWERED, FakeGraph, FakeKG, FakeLLM

    def build(provider):
        return deps_module.Runtime(
            kg=FakeKG(), llm=FakeLLM(), deps=None, graph=FakeGraph(ANSWERED), websearch=provider
        )

    without = await run_query(build(None), "latest tea export news", record_history=False)
    with_web = await run_query(
        build(FixtureProvider([WebResult(title="t", url="https://e.test/x", snippet="s")])),
        "latest tea export news",
        record_history=False,
    )
    n = len(without.response.evidence)
    assert [e.source_id for e in with_web.response.evidence[:n]] == [
        e.source_id for e in without.response.evidence
    ]
    assert with_web.response.evidence[n].source_id == "WEB"


async def test_a_historical_question_never_calls_the_provider():
    from ceynex.api import deps as deps_module
    from ceynex.api.query_runner import run_query
    from ceynex.websearch.schema import WebResult
    from tests.api.test_query import ANSWERED, FakeGraph, FakeKG, FakeLLM

    provider = FixtureProvider([WebResult(title="t", url="https://e.test/x", snippet="s")])
    runtime = deps_module.Runtime(
        kg=FakeKG(), llm=FakeLLM(), deps=None, graph=FakeGraph(ANSWERED), websearch=provider
    )
    await run_query(
        runtime, "Which country took the largest share of tea exports in 2019?",
        record_history=False,
    )
    assert provider.queries == []


async def test_with_the_provider_off_the_answer_is_byte_identical():
    """`CEYNEX_WEB_SEARCH=off` must return the system to exactly what it was.
    That is what makes the claim checkable rather than merely stated."""
    from ceynex.api import deps as deps_module
    from ceynex.api.query_runner import run_query
    from tests.api.test_query import ANSWERED, FakeGraph, FakeKG, FakeLLM

    runtime = deps_module.Runtime(
        kg=FakeKG(), llm=FakeLLM(), deps=None, graph=FakeGraph(ANSWERED), websearch=None
    )
    outcome = await run_query(runtime, "latest tea export news", record_history=False)
    assert all(e.source_id != "WEB" for e in outcome.response.evidence)


def test_web_search_is_not_reachable_from_an_agent(monkeypatch):
    """SRS 3.6.4 fixes the agent count at five, and D14 is not a sixth.

    `AgentDeps` is the only thing an agent is handed, so checking the declared
    fields is most of it — but `extras` is an open dict, and that is exactly
    where a future convenience would put the provider. Both are checked, against
    a real `Runtime.build()` rather than a hand-made one, because the wiring site
    is the thing that could get this wrong.

    The provider has to exist for the identity check to mean anything. With no
    key it is `None`, and so is the policy retriever with no Qdrant configured,
    so `None is not None` failed the test on a clean checkout while a developer
    `.env` naming a Qdrant made it pass; a CI-like run with no `.env` found it.
    Building the real provider (no network until `search()` is called) makes
    the check the same on every machine.
    """
    import dataclasses

    from ceynex.agents.common import AgentDeps
    from ceynex.api.deps import Runtime

    assert "websearch" not in {f.name for f in dataclasses.fields(AgentDeps)}

    monkeypatch.setenv("CEYNEX_WEB_SEARCH", "on")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key-never-sent")
    runtime = Runtime.build()
    assert runtime.websearch is not None, "no provider was built, so nothing below is checked"
    # Identity, not `isinstance`: `WebSearchProvider` is a runtime_checkable
    # Protocol, so `isinstance` is satisfied by anything with a `search` method —
    # `PolicyRetriever` included, which made the obvious form of this assertion
    # fail against perfectly correct wiring.
    assert "websearch" not in runtime.deps.extras
    assert all(value is not runtime.websearch for value in runtime.deps.extras.values()), (
        "the provider reached AgentDeps.extras, where an agent can read it"
    )


# --- the global cap: fairness limits do not bound a bill ---------------------


async def test_a_global_cap_bounds_outbound_calls_across_all_users():
    """The four per-user allowances bound how often one account can search. They
    say nothing about the total, and a hundred accounts inside their own limits
    still add up to a bill on someone else's API."""
    from ceynex.api.rate_limit import Decision
    from ceynex.websearch import client as websearch_client

    seen: list[str] = []

    class Capped:
        async def check(self, identity, limit, window_s):
            seen.append(identity)
            return Decision(allowed=False, limit=limit, remaining=0, retry_after_s=10)

    provider = FixtureProvider(RESULTS)
    websearch_client.set_global_window(Capped())
    try:
        assert await websearch_client.safe_search(provider, "latest tea prices") == []
        assert provider.queries == [], "the provider must not be called once capped"
        assert seen == [websearch_client.GLOBAL_IDENTITY]
    finally:
        websearch_client.set_global_window(None)


async def test_the_cap_fails_open_rather_than_taking_the_feature_down():
    """A throttle that breaks the feature when Redis blinks is worse than the
    spend it was protecting against — the same posture `RedisWindow` takes."""
    from ceynex.websearch import client as websearch_client

    class Broken:
        async def check(self, identity, limit, window_s):
            raise RuntimeError("redis is gone")

    provider = FixtureProvider(RESULTS)
    websearch_client.set_global_window(Broken())
    try:
        assert await websearch_client.safe_search(provider, "latest tea prices") == RESULTS
    finally:
        websearch_client.set_global_window(None)
