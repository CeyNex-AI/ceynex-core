"""Assertions for the policy retriever — filter construction and the degrade path.

No network and no docker. What is asserted here is the part that fails silently
in production: a filter built on a payload key that does not exist matches no
points and raises nothing, so the retriever returns an empty list and the agent
reports "no policy evidence" forever without anyone seeing an error.
"""

import asyncio

import pytest

from ceynex.retrieval.client import (
    MIN_RERANK_SCORE,
    PolicyRetriever,
    PolicyRetrieverUnavailableError,
    _build_filter,
    _rerank,
)
from ceynex.retrieval.schema import (
    DOC_ID,
    HS_PREFIX,
    ISO3,
    MEASURE_TYPE,
    PolicyChunk,
    RetrievalFilter,
)


def keys_of(built) -> list[str]:
    return [condition.key for condition in built.must]


# --- filter construction -------------------------------------------------


def test_every_graph_resolved_constraint_becomes_a_filter_condition():
    built = _build_filter(
        RetrievalFilter(
            iso3=("USA",),
            hs_prefixes=("61", "6109"),
            measure_types=("tariff",),
            doc_ids=("USA-USTR-TARIFF-ACTIONS",),
        )
    )

    assert set(keys_of(built)) == {ISO3, HS_PREFIX, MEASURE_TYPE, DOC_ID, "language"}


def test_conditions_are_all_must_never_should():
    """`should` would let a chunk match on country alone.

    That reintroduces exactly the cross-contamination anchoring exists to
    prevent: a passage about US footwear qualifying for a question about US tea
    because the country matched and nothing else had to.
    """
    built = _build_filter(RetrievalFilter(iso3=("USA",), hs_prefixes=("61",)))

    assert built.should is None
    assert len(built.must) == 3  # iso3, hs_prefix, language


def test_an_empty_filter_is_no_filter_rather_than_an_impossible_one():
    """A filter with zero conditions must be None, not `Filter(must=[])`.

    An empty `must` list is a filter that matches nothing in Qdrant, so building
    one would turn "search everything" into "return nothing".
    """
    assert _build_filter(RetrievalFilter(language="")) is None


def test_hs_prefixes_match_any_level_of_the_hierarchy():
    built = _build_filter(RetrievalFilter(hs_prefixes=("610910", "6109", "61")))
    condition = next(c for c in built.must if c.key == HS_PREFIX)

    assert set(condition.match.any) == {"610910", "6109", "61"}


# --- the description that becomes Evidence.detail ------------------------


def test_the_description_names_every_constraint_the_search_ran_under():
    """Policy evidence is only checkable if it says what was searched.

    `KnowledgeGraphClient.run()` hands its Cypher back for the same reason. A
    vector search has no query text, so this string is the whole of its
    provenance.
    """
    described = RetrievalFilter(
        iso3=("GBR",), hs_prefixes=("0902",), measure_types=("tariff", "fta")
    ).describe()

    assert "GBR" in described
    assert "0902" in described
    assert "tariff" in described
    assert "bge-base-en-v1.5" in described, "the embedding model is part of the provenance"


def test_a_filter_with_no_graph_constraints_reports_itself_as_unanchored():
    assert not RetrievalFilter().is_anchored
    assert RetrievalFilter(iso3=("USA",)).is_anchored


# --- the relevance floor -------------------------------------------------


def chunk(text: str, index: int) -> PolicyChunk:
    return PolicyChunk(
        doc_id="D", chunk_index=index, text=text, title="T", publisher="P", url="u"
    )


class FakeReranker:
    def __init__(self, scores):
        self._scores = scores

    def rerank(self, query, texts):  # noqa: ARG002 - signature mirrors fastembed
        return self._scores


def test_chunks_below_the_relevance_floor_are_dropped():
    """Measured case: the corpus ABBREVIATIONS page scored -10.05 on a tariff
    question because it contains "United States dollars", while responsive
    passages score +0.85 to +3.64."""
    models = {"rerank": FakeReranker([3.6, -10.05, 0.9])}
    kept = _rerank(models, "q", [chunk("good", 0), chunk("abbreviations", 1), chunk("ok", 2)], 5)

    assert [c.chunk_index for c in kept] == [0, 2]


def test_returning_nothing_beats_returning_the_least_bad_chunk():
    models = {"rerank": FakeReranker([-4.0, -10.0])}

    assert _rerank(models, "q", [chunk("a", 0), chunk("b", 1)], 5) == []


def test_the_floor_is_the_models_own_relevance_boundary():
    assert MIN_RERANK_SCORE == 0.0


# --- degradation ---------------------------------------------------------


def test_a_search_that_overruns_its_budget_raises_rather_than_hanging():
    """The 2-second ceiling is a requirement, not a nicety.

    `orchestrator/graph.py` gives each agent a 12 s slice of a budget
    single-sector queries already breach (EVALUATION.md §1). A retriever that
    blocks instead of raising spends the agent's whole slice and turns an
    enhancement into a timeout.
    """

    async def scenario():
        retriever = PolicyRetriever(url="http://localhost:1", timeout_s=0.05)
        retriever._search = _never_returns  # noqa: SLF001 - substituting the slow part
        with pytest.raises(PolicyRetrieverUnavailableError, match="budget"):
            await retriever.search("anything")

    asyncio.run(scenario())


async def _never_returns(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
    await asyncio.sleep(60)


def test_from_settings_returns_none_when_retrieval_is_switched_off(monkeypatch):
    """`None` is the configured-off state, not an error.

    `make eval-policy-baseline` sets this to measure the system as it behaved
    before retrieval existed, and the agent treats None exactly as it treats an
    absent LLM key.
    """
    monkeypatch.setenv("CEYNEX_POLICY_RETRIEVAL", "off")

    assert PolicyRetriever.from_settings() is None


def test_from_settings_returns_none_when_no_qdrant_is_configured(monkeypatch):
    monkeypatch.setenv("CEYNEX_POLICY_RETRIEVAL", "on")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_HOST", raising=False)
    monkeypatch.delenv("QDRANT_PORT", raising=False)

    assert PolicyRetriever.from_settings() is None


def test_a_typo_in_the_kill_switch_leaves_retrieval_on(monkeypatch):
    """Anything other than off/0/false/no means on.

    A misspelled env var silently disabling a feature is how a measurement gets
    taken against the wrong system.
    """
    from ceynex.settings import policy_retrieval_enabled

    monkeypatch.setenv("CEYNEX_POLICY_RETRIEVAL", "offf")
    assert policy_retrieval_enabled()

    monkeypatch.setenv("CEYNEX_POLICY_RETRIEVAL", "off")
    assert not policy_retrieval_enabled()
