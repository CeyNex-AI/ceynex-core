"""Assertions for headline scoring — ordering, the floor, and the degrade path.

The cross-encoder is faked through `set_models_provider()` rather than by
monkeypatching fastembed. Tests never load ONNX weights, and the seam is the same
one `routes/query.py` already establishes with `set_window()`.
"""

import pytest

from ceynex.news import relevance
from ceynex.news.schema import NewsArticle
from ceynex.retrieval.client import MIN_RERANK_SCORE


class FakeCrossEncoder:
    """Returns a canned score per title, in the order it was handed them."""

    def __init__(self, scores):
        self._scores = scores
        self.calls = 0

    def rerank(self, query, texts):  # noqa: ARG002 - mirrors fastembed's signature
        self.calls += 1
        return [self._scores[text] for text in texts]


@pytest.fixture
def provider():
    """Installs a fake cross-encoder and removes it afterwards."""
    holder = {}

    def install(scores):
        encoder = FakeCrossEncoder(scores)
        holder["encoder"] = encoder

        async def provide():
            return {"rerank": encoder}

        relevance.set_models_provider(provide)
        return encoder

    yield install
    relevance.set_models_provider(None)


def article(title: str) -> NewsArticle:
    return NewsArticle(url=f"https://example.com/{title.replace(' ', '-')}", title=title)


# --- ordering ------------------------------------------------------------


async def test_headlines_come_back_most_relevant_first(provider):
    provider({"tea prices rise": 3.0, "unrelated football result": -2.0, "tea exports grow": 1.0})
    articles = [article("unrelated football result"), article("tea prices rise"), article("tea exports grow")]

    ranked = await relevance.score_articles("ceylon tea", articles, limit=5, floor=-10.0)

    assert [a.title for a in ranked] == ["tea prices rise", "tea exports grow", "unrelated football result"]


async def test_the_score_is_attached_to_the_article(provider):
    provider({"tea prices rise": 3.0})

    (ranked,) = await relevance.score_articles("tea", [article("tea prices rise")], limit=5)

    assert ranked.relevance == 3.0


async def test_limit_is_respected(provider):
    provider({f"story {i}": float(i) for i in range(10)})
    articles = [article(f"story {i}") for i in range(10)]

    ranked = await relevance.score_articles("q", articles, limit=3, floor=-10.0)

    assert len(ranked) == 3


async def test_no_articles_means_the_model_is_never_touched(provider):
    encoder = provider({})

    assert await relevance.score_articles("q", [], limit=5) == []
    assert encoder.calls == 0


# --- the floor -----------------------------------------------------------


async def test_headlines_below_the_floor_are_dropped(provider):
    """Returning nothing is a correct outcome; returning the least-bad row is not."""
    provider({"tea prices rise": 2.0, "celebrity gossip": -20.0})
    articles = [article("tea prices rise"), article("celebrity gossip")]

    ranked = await relevance.score_articles("ceylon tea", articles, limit=5, floor=-6.0)

    assert [a.title for a in ranked] == ["tea prices rise"]


async def test_the_floor_comes_from_config_not_a_module_constant(provider, monkeypatch):
    """It is a judgement, so it lives where a diff shows who changed it."""
    provider({"borderline story": -5.0})
    monkeypatch.setattr(
        relevance.settings, "news_config", lambda: {"relevance": {"min_score": -1.0}}
    )

    ranked = await relevance.score_articles("q", [article("borderline story")], limit=5)

    assert ranked == []


async def test_the_news_floor_is_looser_than_the_policy_floor():
    """A headline is a fifth the length of a passage and scores lower on the same model.

    Reusing retrieval's 0.0 here would empty the panel for most real questions,
    and this asserts the two never silently converge.
    """
    assert relevance.min_score() < MIN_RERANK_SCORE


async def test_an_unreadable_config_falls_back_rather_than_raising(monkeypatch):
    def boom():
        raise FileNotFoundError("config/news.yaml")

    monkeypatch.setattr(relevance.settings, "news_config", boom)

    assert relevance.min_score() == relevance.FALLBACK_MIN_SCORE


# --- degrading -----------------------------------------------------------


async def test_without_the_policy_extra_articles_come_back_in_gdelts_own_order():
    """Unranked is a usable answer; an empty panel is not.

    This is what keeps `[policy]` defensibly optional — the news feature still
    returns real, linkable headlines without fastembed installed.
    """

    async def no_fastembed():
        raise ImportError("No module named 'fastembed'")

    relevance.set_models_provider(no_fastembed)
    try:
        articles = [article("first"), article("second"), article("third")]

        ranked = await relevance.score_articles("q", articles, limit=2)

        assert [a.title for a in ranked] == ["first", "second"]
        assert all(a.relevance is None for a in ranked)
    finally:
        relevance.set_models_provider(None)


async def test_a_scoring_failure_costs_the_ranking_not_the_panel():
    class BrokenEncoder:
        def rerank(self, query, texts):
            raise RuntimeError("onnx session died")

    async def provide():
        return {"rerank": BrokenEncoder()}

    relevance.set_models_provider(provide)
    try:
        ranked = await relevance.score_articles("q", [article("a"), article("b")], limit=5)

        assert [a.title for a in ranked] == ["a", "b"]
    finally:
        relevance.set_models_provider(None)
