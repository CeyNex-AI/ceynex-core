"""Assertions for the news endpoints — the degrade guarantee above all.

The one thing that must hold: a dead GDELT, a dead Qdrant and an empty snapshot
are all HTTP 200. A 5xx from the sidecar would make a working answer page look
broken, which is a worse outcome than showing no news.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from ceynex.api import rate_limit
from ceynex.api.deps import Runtime, set_runtime
from ceynex.api.routes import news as news_routes
from ceynex.news import snapshot
from ceynex.news.gdelt import GdeltUnavailableError
from ceynex.news.schema import NewsArticle


class FakeGdelt:
    def __init__(self, articles=None, *, fails=False):
        self._articles = articles or []
        self._fails = fails

    async def articles(self, query, **kwargs):  # noqa: ARG002 - client signature
        if self._fails:
            raise GdeltUnavailableError("down")
        return list(self._articles)

    async def close(self):
        return None


class FakeStore:
    def __init__(self, articles=None, *, fails=False):
        self._articles = articles or []
        self._fails = fails
        self.upserted: list[NewsArticle] = []

    async def search(self, query, **kwargs):  # noqa: ARG002 - store signature
        if self._fails:
            raise RuntimeError("qdrant is down")
        return list(self._articles)

    async def upsert(self, articles, batch=64):  # noqa: ARG002
        self.upserted.extend(articles)
        return len(articles)

    async def close(self):
        return None


def article(title: str, domain: str = "example.com") -> NewsArticle:
    return NewsArticle(
        url=f"https://{domain}/{title.replace(' ', '-')}",
        title=title,
        domain=domain,
        source_country="Sri Lanka",
        seen_at=datetime.now(tz=UTC),
    )


@pytest.fixture(autouse=True)
def isolated_news(monkeypatch):
    """A fresh window and cache per test, and no cross-encoder.

    Scoring is faked to the identity so these tests assert routing and
    degradation rather than re-testing `test_relevance.py`.
    """
    news_routes.set_window(rate_limit.InProcessWindow())
    news_routes.clear_cache()

    async def unranked(query, articles, *, limit, floor=None):  # noqa: ARG001
        return list(articles)[:limit]

    monkeypatch.setattr(news_routes, "score_articles", unranked)
    yield
    news_routes.set_window(None)
    news_routes.clear_cache()
    set_runtime(None)


def client_with(gdelt=None, store=None) -> TestClient:
    """A TestClient over a runtime carrying only what the news routes touch."""
    runtime = Runtime(
        kg=None, llm=None, deps=None, graph=None, policy=None, gdelt=gdelt, news=store
    )
    set_runtime(runtime)
    from ceynex.api.main import app

    return TestClient(app)


# --- the happy path ------------------------------------------------------


def test_a_live_search_reports_that_it_was_live():
    client = client_with(gdelt=FakeGdelt([article("Ceylon tea exports rise")]))

    body = client.get("/api/news/search", params={"q": "ceylon tea"}).json()

    assert body["source"] == "gdelt"
    assert body["articles"][0]["title"] == "Ceylon tea exports rise"
    assert body["articles"][0]["url"].startswith("https://")


def test_every_article_carries_a_relevance_label_even_when_unscored():
    """The UI switches on the label, so it may never be absent."""
    client = client_with(gdelt=FakeGdelt([article("Tea")]))

    body = client.get("/api/news/search", params={"q": "ceylon tea"}).json()

    assert body["articles"][0]["relevance"] == "unscored"


def test_syndication_twins_collapse_in_the_response():
    """One wire story on three sites is one row, not three."""
    client = client_with(
        gdelt=FakeGdelt(
            [
                article("Iran oil exports stall", "asiaone.com"),
                article("iran oil exports stall", "reuters.com"),
                article("Tea auction prices climb", "ft.lk"),
            ]
        )
    )

    body = client.get("/api/news/search", params={"q": "oil"}).json()

    assert len(body["articles"]) == 2


# --- degrading -----------------------------------------------------------


def test_a_dead_gdelt_falls_back_to_indexed_headlines():
    client = client_with(
        gdelt=FakeGdelt(fails=True), store=FakeStore([article("An older indexed story")])
    )

    body = client.get("/api/news/search", params={"q": "ceylon tea"}).json()

    assert body["source"] == "cache"
    assert body["articles"]


def test_both_sources_down_is_a_200_with_an_empty_list_not_a_5xx():
    """The degrade guarantee. A broken sidecar must not break the page."""
    client = client_with(gdelt=FakeGdelt(fails=True), store=FakeStore(fails=True))

    response = client.get("/api/news/search", params={"q": "ceylon tea"})

    assert response.status_code == 200
    assert response.json() == {
        "query": "ceylon tea",
        "articles": [],
        "source": "unavailable",
        "elapsed_ms": pytest.approx(response.json()["elapsed_ms"]),
    }


def test_no_news_configured_at_all_is_still_a_200():
    client = client_with(gdelt=None, store=None)

    response = client.get("/api/news/search", params={"q": "ceylon tea"})

    assert response.status_code == 200
    assert response.json()["source"] == "unavailable"


def test_a_failure_is_not_cached():
    """Otherwise the panel stays empty for five minutes after GDELT recovers."""
    store = FakeStore(fails=True)
    client = client_with(gdelt=FakeGdelt(fails=True), store=store)
    client.get("/api/news/search", params={"q": "ceylon tea"})

    client = client_with(gdelt=FakeGdelt([article("Back online")]), store=store)
    body = client.get("/api/news/search", params={"q": "ceylon tea"}).json()

    assert body["source"] == "gdelt"


# --- input ---------------------------------------------------------------


def test_a_query_too_short_to_mean_anything_is_refused():
    """GDELT answers a one-character query with a plain-text complaint anyway."""
    client = client_with(gdelt=FakeGdelt([]))

    assert client.get("/api/news/search", params={"q": "a"}).status_code == 422


def test_the_limit_is_bounded():
    client = client_with(gdelt=FakeGdelt([]))

    assert client.get("/api/news/search", params={"q": "tea", "limit": 500}).status_code == 422


# --- indexing ------------------------------------------------------------


def test_live_results_are_indexed_after_the_response():
    store = FakeStore()
    client = client_with(gdelt=FakeGdelt([article("Ceylon tea exports rise")]), store=store)

    client.get("/api/news/search", params={"q": "ceylon tea"})

    assert [a.title for a in store.upserted] == ["Ceylon tea exports rise"]


def test_results_that_came_out_of_the_store_are_not_written_back():
    """Re-upserting what we just read is pure write amplification."""
    store = FakeStore([article("An older indexed story")])
    client = client_with(gdelt=FakeGdelt(fails=True), store=store)

    client.get("/api/news/search", params={"q": "ceylon tea"})

    assert store.upserted == []


# --- rate limiting -------------------------------------------------------


async def test_news_searches_do_not_consume_the_query_allowance():
    """`rate_limit.KEY_PREFIX` is one constant shared by every Window.

    Without the `news:` identity prefix both endpoints would count into the same
    keys — and since the browser fires a news search on every submitted query,
    the query limit would silently become half of what config/api.yaml says.

    Asserted against a window the two endpoints share, which is the deployed
    shape: one Redis, one key prefix, two callers.
    """
    shared = rate_limit.InProcessWindow()
    news_routes.set_window(shared)
    client = client_with(gdelt=FakeGdelt([]))

    for i in range(21):
        client.get("/api/news/search", params={"q": f"question number {i}"})

    # The query endpoint's own identity, unprefixed, must be untouched.
    decision = await shared.check(rate_limit.identity_of(None, "testclient"), 30, 60)

    assert decision.allowed
    assert decision.remaining == 29, "news requests leaked into the query allowance"


def test_the_news_allowance_is_enforced():
    client = client_with(gdelt=FakeGdelt([]))

    statuses = [
        client.get("/api/news/search", params={"q": f"question number {i}"}).status_code
        for i in range(22)
    ]

    assert 429 in statuses


# --- trending ------------------------------------------------------------


def test_before_the_first_refresh_trending_is_warming_not_broken(monkeypatch):
    monkeypatch.setattr(snapshot, "latest", lambda scope: None)
    client = client_with()

    response = client.get("/api/news/trending")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "warming"
    assert set(body["scopes"]) == {"sri_lanka", "global"}
    assert body["scopes"]["global"] == []


def test_a_fresh_snapshot_is_ready(monkeypatch):
    monkeypatch.setattr(
        snapshot,
        "latest",
        lambda scope: snapshot.Snapshot(
            scope=scope,
            computed_at=datetime.now(tz=UTC),
            topics=[
                {
                    "topic_id": "lk_tea",
                    "label": "Ceylon tea",
                    "prompt": "How are Ceylon tea exports performing right now?",
                    "scope": scope,
                    "articles_24h": 47,
                    "baseline_24h": 16.8,
                    "delta_pct": 180.0,
                    "direction": "surging",
                    "top_articles": [],
                }
            ],
        ),
    )
    client = client_with()

    body = client.get("/api/news/trending").json()

    assert body["status"] == "ready"
    assert body["scopes"]["sri_lanka"][0]["direction"] == "surging"
    assert body["scopes"]["sri_lanka"][0]["prompt"].startswith("How are")


def test_an_old_snapshot_says_it_is_stale(monkeypatch):
    """A six-hour-old panel labelled "trending now" is a lie."""
    monkeypatch.setattr(
        snapshot,
        "latest",
        lambda scope: snapshot.Snapshot(
            scope=scope,
            computed_at=datetime.now(tz=UTC) - timedelta(hours=6),
            topics=[],
        ),
    )
    client = client_with()

    assert client.get("/api/news/trending").json()["status"] == "stale"


def test_an_unreachable_database_still_returns_200(monkeypatch):
    """`snapshot.latest` already swallows psycopg errors and returns None."""
    monkeypatch.setattr(snapshot, "latest", lambda scope: None)
    client = client_with()

    assert client.get("/api/news/trending").status_code == 200
