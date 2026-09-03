"""Assertions for the news store.

The guard tests need no docker and run everywhere; the round-trip tests need a
live Qdrant and are marked `integration`, same convention as the rest of the
suite. Run the whole file with `make test`, the rest with `make test-unit`.
"""

import pytest

from ceynex import settings
from ceynex.news.schema import NewsArticle
from ceynex.news.store import NewsStore

pytest.importorskip("qdrant_client", reason="the [policy] extra is optional")


# --- the guard that needs nothing running --------------------------------


def test_the_store_refuses_to_open_the_policy_collection():
    """A mis-set QDRANT_NEWS_COLLECTION must be loud at startup.

    Sharing the collection would put unvetted headlines behind the filters that
    produce citable policy evidence, and `retrieval/client.py` widens its own
    filters on an empty result — so the mistake would surface as a news headline
    in an evidence panel rather than as an error.
    """
    with pytest.raises(ValueError, match="policy collection"):
        NewsStore(collection=settings.qdrant_collection())


def test_a_distinct_collection_is_accepted():
    store = NewsStore(collection="ceynex_news_test")

    assert store is not None


# --- round trips ---------------------------------------------------------


@pytest.fixture
async def store():
    store = NewsStore(collection="ceynex_news_test")
    ready = await store.ensure_collection()
    if not ready:
        await store.close()
        pytest.skip("qdrant is not reachable")
    yield store
    try:
        await store.client.delete_collection("ceynex_news_test")
    finally:
        await store.close()


@pytest.mark.integration
async def test_ensure_collection_is_idempotent(store):
    """It runs on every startup, so the second run must be a no-op."""
    assert await store.ensure_collection() is True


@pytest.mark.integration
async def test_indexing_the_same_article_twice_leaves_one_point(store):
    """The refresher re-fetches the same stories hourly, forever.

    Without deterministic ids this would add twenty-four copies of every article
    every day, and the panel would show each headline once per refresh.
    """
    article = NewsArticle(url="https://example.com/tea", title="Ceylon tea exports rise")

    await store.upsert([article])
    await store.upsert([article])

    assert await store.count() == 1


@pytest.mark.integration
async def test_the_same_story_under_two_url_spellings_is_one_point(store):
    """GDELT returns both `asiaone.com` and `asiaone.com:443` for one article."""
    await store.upsert(
        [
            NewsArticle(url="https://www.asiaone.com:443/money/x", title="Iran oil exports stall"),
            NewsArticle(url="http://asiaone.com/money/x/?utm_source=tw", title="Iran oil exports stall"),
        ]
    )

    assert await store.count() == 1


@pytest.mark.integration
async def test_search_finds_an_indexed_headline(store):
    await store.upsert(
        [
            NewsArticle(url="https://example.com/tea", title="Ceylon tea auction prices climb"),
            NewsArticle(url="https://example.com/cricket", title="Cricket final draws record crowd"),
        ]
    )

    found = await store.search("ceylon tea prices", limit=5)

    assert found
    assert found[0].title == "Ceylon tea auction prices climb"


@pytest.mark.integration
async def test_prune_removes_only_what_is_actually_old(store):
    """An undated article has seen_ts 0, which is not the same as being ancient.

    Treating 0 as "older than the cutoff" would silently discard every article
    whose GDELT timestamp failed to parse, on the very first sweep.
    """
    from datetime import UTC, datetime, timedelta

    old = NewsArticle(
        url="https://example.com/old",
        title="An old story",
        seen_at=datetime.now(tz=UTC) - timedelta(days=90),
    )
    recent = NewsArticle(
        url="https://example.com/recent", title="A recent story", seen_at=datetime.now(tz=UTC)
    )
    undated = NewsArticle(url="https://example.com/undated", title="An undated story")
    await store.upsert([old, recent, undated])

    removed = await store.prune(older_than_days=30)

    assert removed == 1
    assert await store.count() == 2
