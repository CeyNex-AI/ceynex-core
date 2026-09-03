"""Assertions for the refresher — the lock, and what happens when GDELT is down.

Redis is a dict-backed fake rather than a real server: what is worth asserting is
the *protocol* (set-if-absent, check-before-delete), and a real Redis would test
redis rather than us.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ceynex.news import refresh, snapshot
from ceynex.news.gdelt import GdeltUnavailableError
from ceynex.news.refresh import NoopLock, RedisLock, RefreshReport, build_lock, refresh_once
from ceynex.news.schema import NewsArticle
from ceynex.news.trending import WatchTopic

FIXTURES = Path(__file__).parent / "fixtures"


class FakeRedis:
    """Just enough of redis.asyncio for the lock protocol."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def set(self, key, value, nx=False, ex=None, px=None):  # noqa: ARG002 - redis signature
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def get(self, key):
        return self.store.get(key)

    async def delete(self, key):
        return int(self.store.pop(key, None) is not None)

    async def pexpire(self, key, ms):  # noqa: ARG002 - no clock in the fake
        return key in self.store


# --- the lock ------------------------------------------------------------


async def test_without_redis_the_worker_assumes_it_is_the_only_one(monkeypatch):
    """Correct for a single-worker development run, and said out loud in the log."""
    monkeypatch.setattr(refresh.settings, "redis_url", lambda: None)

    lock = build_lock()

    assert isinstance(lock, NoopLock)
    assert await lock.acquire() is True


async def test_only_one_worker_refreshes_per_cycle():
    redis = FakeRedis()
    first = RedisLock(redis)
    second = RedisLock(redis)

    assert await first.acquire() is True
    assert await second.acquire() is False


async def test_releasing_a_lock_we_no_longer_hold_does_not_delete_it():
    """The bug this prevents produces two refreshers and raises nothing.

    Sequence: worker A takes the lock, A's cycle overruns its TTL, the key
    expires, worker B takes it. A then finishes and releases. With a bare `DEL`
    that release removes *B's* lock, and the next cycle has two refreshers.
    """
    redis = FakeRedis()
    slow = RedisLock(redis)
    await slow.acquire()

    redis.store.clear()  # the TTL expires mid-cycle
    other = RedisLock(redis)
    await other.acquire()
    other_token = redis.store[refresh.LOCK_KEY]

    await slow.release()

    assert redis.store.get(refresh.LOCK_KEY) == other_token


async def test_releasing_our_own_lock_frees_it_for_the_next_cycle():
    redis = FakeRedis()
    lock = RedisLock(redis)
    await lock.acquire()

    await lock.release()

    assert refresh.LOCK_KEY not in redis.store
    assert await RedisLock(redis).acquire() is True


async def test_a_heartbeat_from_a_worker_that_lost_the_lock_is_a_no_op():
    redis = FakeRedis()
    slow = RedisLock(redis)
    await slow.acquire()
    redis.store[refresh.LOCK_KEY] = "someone-elses-token"

    await slow.heartbeat()

    assert redis.store[refresh.LOCK_KEY] == "someone-elses-token"


async def test_an_unreachable_lock_store_skips_the_cycle_rather_than_crashing():
    class BrokenRedis:
        async def set(self, *args, **kwargs):
            raise ConnectionError("redis is down")

    assert await RedisLock(BrokenRedis()).acquire() is False


# --- one pass ------------------------------------------------------------


class FakeGdelt:
    """A GDELT client that answers from the fixtures, or fails on demand."""

    def __init__(self, *, fail_topics=(), fail_all=False):
        self._fail = set(fail_topics)
        self._fail_all = fail_all
        from ceynex.news.gdelt import _parse_timeline

        self.points = _parse_timeline(json.loads((FIXTURES / "gdelt_timeline_7d.json").read_text()))
        self.volume_calls = 0
        self.article_calls = 0

    async def volume(self, query, **kwargs):  # noqa: ARG002 - client signature
        self.volume_calls += 1
        if self._fail_all or query in self._fail:
            raise GdeltUnavailableError("down")
        return self.points

    async def articles(self, query, *, topic="", scope="", **kwargs):  # noqa: ARG002
        self.article_calls += 1
        if self._fail_all or query in self._fail:
            raise GdeltUnavailableError("down")
        return [
            NewsArticle(
                url=f"https://example.com/{topic}-{i}",
                title=f"{topic} story {i}",
                topic=topic,
                scope=scope,
                seen_at=datetime.now(tz=UTC),
            )
            for i in range(3)
        ]


def watchlist():
    return [
        WatchTopic(id="lk_tea", scope="sri_lanka", label="Tea", prompt="p", query="tea"),
        WatchTopic(id="gl_tariffs", scope="global", label="Tariffs", prompt="p", query="tariffs"),
    ]


CONFIG = {
    "gdelt": {"timespan_trending": "7d", "timespan_topic_articles": "24h", "max_records_topic": 20},
    "trending": {"window_hours": 24, "min_articles": 3, "top_n": 6},
    "store": {"retention_days": 30, "upsert_batch": 64},
}


@pytest.fixture
def captured_snapshots(monkeypatch):
    """Intercept snapshot writes. `refresh.snapshot` is this same module object."""
    written: list[tuple[str, list, bool]] = []

    def capture(scope, topics, *, partial=False):
        written.append((scope, topics, partial))
        return True

    monkeypatch.setattr(snapshot, "write", capture)
    return written


async def test_a_full_pass_writes_one_snapshot_per_scope(captured_snapshots):
    report = await refresh_once(
        FakeGdelt(), None, config=CONFIG, topics=watchlist(), index=False
    )

    assert report.topics_ok == 2
    assert report.topics_failed == []
    assert {scope for scope, _, _ in captured_snapshots} == {"sri_lanka", "global"}


async def test_one_failing_topic_does_not_stop_the_others(captured_snapshots):
    report = await refresh_once(
        FakeGdelt(fail_topics=("tea",)), None, config=CONFIG, topics=watchlist(), index=False
    )

    assert report.topics_ok == 1
    assert report.topics_failed == ["lk_tea"]
    assert report.partial is True
    assert all(partial for _, _, partial in captured_snapshots)


async def test_when_everything_fails_the_previous_panel_is_left_alone(captured_snapshots):
    """Replacing a good panel with an empty one on a transient outage is worse
    than showing a stale one and saying so."""
    report = await refresh_once(
        FakeGdelt(fail_all=True), None, config=CONFIG, topics=watchlist(), index=False
    )

    assert report.topics_ok == 0
    assert captured_snapshots == []


async def test_two_gdelt_calls_per_topic_and_no_more(captured_snapshots):
    """The call budget in config/news.yaml assumes exactly this."""
    gdelt = FakeGdelt()

    await refresh_once(gdelt, None, config=CONFIG, topics=watchlist(), index=False)

    assert gdelt.volume_calls == 2
    assert gdelt.article_calls == 2


async def test_the_lock_is_heartbeaten_as_the_pass_proceeds(captured_snapshots):
    """A cycle slower than the TTL would otherwise lose its lock mid-flight."""

    class CountingLock(NoopLock):
        beats = 0

        async def heartbeat(self):
            CountingLock.beats += 1

    await refresh_once(
        FakeGdelt(), None, config=CONFIG, topics=watchlist(), index=False, lock=CountingLock()
    )

    assert CountingLock.beats == 2


async def test_articles_are_indexed_when_a_store_is_present(captured_snapshots):
    class RecordingStore:
        def __init__(self):
            self.indexed = []

        async def upsert(self, articles, batch=64):  # noqa: ARG002 - store signature
            self.indexed.extend(articles)
            return len(articles)

        async def prune(self, days):  # noqa: ARG002
            return 0

    store = RecordingStore()

    report = await refresh_once(FakeGdelt(), store, config=CONFIG, topics=watchlist())

    assert report.articles_indexed == 6
    assert {a.topic for a in store.indexed} == {"lk_tea", "gl_tariffs"}


# --- reporting -----------------------------------------------------------


def test_the_report_names_the_topics_that_failed():
    """`docker logs` has to answer "did it run, and what broke" on its own."""
    report = RefreshReport(topics_ok=12, topics_failed=["lk_tea"], articles_indexed=200)

    assert "lk_tea" in str(report)
    assert "12 topics ok" in str(report)


# --- the CLI -------------------------------------------------------------


def test_dry_run_selects_topics_without_calling_anything():
    from ceynex.settings import news_config

    args = refresh._parse_args(["--dry-run", "--scope", "sri_lanka"])
    selected = refresh._selected(news_config(), args)

    assert selected
    assert {topic.scope for topic in selected} == {"sri_lanka"}


def test_a_single_topic_can_be_selected_by_id():
    from ceynex.settings import news_config

    args = refresh._parse_args(["--topic", "lk_tea"])
    selected = refresh._selected(news_config(), args)

    assert [topic.id for topic in selected] == ["lk_tea"]
