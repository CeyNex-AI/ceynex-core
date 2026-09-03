"""Supports docs/ARCHITECTURE_DELTA.md D11 — the hourly trending refresher.

    python -m ceynex.news.refresh --once          # one pass, by hand
    python -m ceynex.news.refresh --dry-run       # print the calls, make none

Two entry points into the same `refresh_once()`: this CLI, and a background task
the API's lifespan starts. The CLI is not only a convenience — it populates the
collection before a demo without waiting an hour, and `--dry-run` is how a
watchlist query gets checked before it is trusted. `data/pipeline.py`,
`kg/load.py` and `data/bootstrap.py` all have this shape already.

No scheduler
------------
There is no Celery, no APScheduler and no cron anywhere in this project, and
adding one for fourteen HTTP calls an hour would be a component the SAD does not
have. An asyncio task in the lifespan is the smallest thing that works. What it
costs is the lock below, because `ceynex-infra/backend/Dockerfile` runs
`uvicorn --workers 2` and both workers run the same lifespan.

The lock
--------
`SET <key> <token> NX EX ttl`. A worker that does not get it is simply not the
refresher this cycle — it logs at debug and sleeps to the next tick. It never
waits for the lock: the goal is that exactly one worker refreshes per cycle, not
that both eventually do.

Release is a token-checked delete, never a bare `DEL`. A bare delete is precisely
how a slow worker removes the *other* worker's lock after its own TTL expired,
producing two refreshers and no error. The heartbeat has the same check.

`GET`-then-`DEL` is not atomic; a Lua CAS would be. The race window is the few
milliseconds between the two calls, against a 15-minute TTL, and losing it costs
one duplicated refresh cycle — deliberately accepted rather than unnoticed.

With no `REDIS_URL` the lock is a no-op that always succeeds, which is correct
for a single-worker development run and stated in the log rather than assumed.

Degrading
---------
Failures are handled per topic. If every topic fails, **no snapshot is written**
— the previous one stays and ages into `stale`. Overwriting a good panel with an
empty one because GDELT had a bad minute is the opposite of degrading gracefully.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import random
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from ceynex import settings
from ceynex.news import snapshot
from ceynex.news.gdelt import GdeltClient, GdeltUnavailableError
from ceynex.news.schema import SCOPES
from ceynex.news.store import NewsStore
from ceynex.news.trending import TrendingTopic, WatchTopic, build_topic, load_watchlist, rank

log = logging.getLogger(__name__)

LOCK_KEY = "ceynex:news:refresh:lock"


# --- the lock -------------------------------------------------------------


class Lock(Protocol):
    async def acquire(self) -> bool: ...
    async def heartbeat(self) -> None: ...
    async def release(self) -> None: ...


class NoopLock:
    """Always the refresher. Correct for one worker, wrong for two."""

    async def acquire(self) -> bool:
        return True

    async def heartbeat(self) -> None:
        return None

    async def release(self) -> None:
        return None


class RedisLock:
    """One refresher per cycle across workers, with the TTL as the safety net."""

    def __init__(self, client: Any, *, key: str = LOCK_KEY, ttl_s: int = 900) -> None:
        self._client = client
        self._key = key
        self._ttl_s = max(1, int(ttl_s))
        self._token = uuid.uuid4().hex

    async def acquire(self) -> bool:
        try:
            return bool(await self._client.set(self._key, self._token, nx=True, ex=self._ttl_s))
        except Exception:  # noqa: BLE001 - an unreachable lock store must not fail the refresh
            log.warning("news refresh lock unavailable; skipping this cycle", exc_info=True)
            return False

    async def _holds(self) -> bool:
        current = await self._client.get(self._key)
        if isinstance(current, bytes):
            current = current.decode("utf-8", errors="replace")
        return current == self._token

    async def heartbeat(self) -> None:
        """Extend the lease, but only while we still hold it."""
        try:
            if await self._holds():
                await self._client.pexpire(self._key, self._ttl_s * 1000)
        except Exception:  # noqa: BLE001 - a missed heartbeat costs a lease, not correctness
            log.debug("could not extend the news refresh lock", exc_info=True)

    async def release(self) -> None:
        """Delete our own lock and nobody else's.

        The check is the whole point. A bare `DEL` after our TTL had already
        expired would remove the lock a *different* worker is now holding, and
        the next cycle would have two refreshers with nothing raised anywhere.
        """
        try:
            if await self._holds():
                await self._client.delete(self._key)
        except Exception:  # noqa: BLE001 - the TTL releases it anyway
            log.debug("could not release the news refresh lock", exc_info=True)


def build_lock(ttl_s: int = 900) -> Lock:
    """Redis when configured, else a no-op. Never raises."""
    url = settings.redis_url()
    if not url:
        log.info(
            "REDIS_URL unset — the news refresher assumes it is the only worker "
            "(correct for one worker, not for the deployed two)"
        )
        return NoopLock()
    try:
        from redis.asyncio import from_url  # noqa: PLC0415 - optional at import time

        return RedisLock(from_url(url, encoding="utf-8", decode_responses=True), ttl_s=ttl_s)
    except Exception:  # noqa: BLE001 - a lock we cannot build must not fail startup
        log.warning("could not build the news refresh lock; assuming one worker", exc_info=True)
        return NoopLock()


# --- one pass -------------------------------------------------------------


@dataclass
class RefreshReport:
    """What one cycle actually did, for the log and for the CLI.

    Logged at INFO so `docker logs` answers "has the refresher been running"
    without attaching a debugger to a container.
    """

    topics_ok: int = 0
    topics_failed: list[str] = field(default_factory=list)
    articles_indexed: int = 0
    articles_pruned: int = 0
    snapshots_written: int = 0
    elapsed_s: float = 0.0

    @property
    def partial(self) -> bool:
        return bool(self.topics_failed) and self.topics_ok > 0

    def __str__(self) -> str:
        return (
            f"{self.topics_ok} topics ok, {len(self.topics_failed)} failed"
            f"{' (' + ', '.join(self.topics_failed) + ')' if self.topics_failed else ''}; "
            f"{self.articles_indexed} indexed, {self.articles_pruned} pruned, "
            f"{self.snapshots_written} snapshots in {self.elapsed_s:.1f}s"
        )


async def refresh_once(
    gdelt: GdeltClient,
    store: NewsStore | None = None,
    *,
    config: dict | None = None,
    lock: Lock | None = None,
    topics: list[WatchTopic] | None = None,
    index: bool = True,
) -> RefreshReport:
    """Fetch every watchlist topic, compute the panel, store it, prune.

    Never raises. The caller is a background task with nobody to report to.
    """
    started = asyncio.get_running_loop().time()
    config = config if config is not None else settings.news_config()
    gdelt_config = config.get("gdelt", {})
    trending_config = config.get("trending", {})
    watchlist = topics if topics is not None else load_watchlist(config)

    report = RefreshReport()
    computed: list[TrendingTopic] = []

    for topic in watchlist:
        try:
            points = await gdelt.volume(
                topic.query, timespan=str(gdelt_config.get("timespan_trending", "7d"))
            )
            articles = await gdelt.articles(
                topic.query,
                timespan=str(gdelt_config.get("timespan_topic_articles", "24h")),
                max_records=int(gdelt_config.get("max_records_topic", 20)),
                sort="DateDesc",
                topic=topic.id,
                scope=topic.scope,
                attempts=4,
                backoff_base_s=2.0,
                keep_forensic=True,
            )
        except GdeltUnavailableError as exc:
            log.warning("news topic %s failed: %s", topic.id, exc)
            report.topics_failed.append(topic.id)
            continue
        except Exception:  # noqa: BLE001 - one bad topic must not stop the rest
            log.warning("news topic %s failed unexpectedly", topic.id, exc_info=True)
            report.topics_failed.append(topic.id)
            continue

        computed.append(
            build_topic(
                topic,
                points,
                articles,
                window_hours=int(trending_config.get("window_hours", 24)),
            )
        )
        report.topics_ok += 1

        if index and store is not None and articles:
            report.articles_indexed += await store.upsert(
                articles, batch=int(config.get("store", {}).get("upsert_batch", 64))
            )

        if lock is not None:
            await lock.heartbeat()

    report.elapsed_s = asyncio.get_running_loop().time() - started

    if not computed:
        # Every topic failed. Leave the previous panel alone and let it age into
        # `stale` rather than replacing a good panel with an empty one.
        log.warning("no news topic succeeded; keeping the previous snapshot")
        return report

    for scope in SCOPES:
        ranked = rank(
            computed,
            min_articles=int(trending_config.get("min_articles", 3)),
            top_n=int(trending_config.get("top_n", 6)),
            scope=scope,
        )
        if snapshot.write(scope, [topic.to_json() for topic in ranked], partial=report.partial):
            report.snapshots_written += 1

    if store is not None:
        report.articles_pruned = await store.prune(
            int(config.get("store", {}).get("retention_days", 30))
        )

    return report


# --- the loop -------------------------------------------------------------


async def run_forever(
    gdelt: GdeltClient | None,
    store: NewsStore | None = None,
    *,
    warm: asyncio.Task | None = None,
    config: dict | None = None,
) -> None:
    """The background task the API lifespan starts. Cancelled at shutdown.

    Awaits `warm` before the first pass: the ONNX sessions take about a second to
    build, and a refresh that starts before them would index nothing and blame
    the embedder.
    """
    if gdelt is None or not settings.news_refresh_enabled():
        log.info("news refresher is off (CEYNEX_NEWS_REFRESH or CEYNEX_NEWS)")
        return

    config = config if config is not None else settings.news_config()
    refresh_config = config.get("refresh", {})
    if not refresh_config.get("enabled", True):
        log.info("news refresher disabled in config/news.yaml")
        return

    interval_s = max(60.0, float(refresh_config.get("interval_minutes", 60)) * 60.0)
    jitter_s = max(0.0, float(refresh_config.get("jitter_seconds", 300)))
    ttl_s = int(refresh_config.get("lock_ttl_seconds", 900))
    lock = build_lock(ttl_s)

    await asyncio.sleep(float(refresh_config.get("startup_delay_seconds", 20)))
    if warm is not None:
        with contextlib.suppress(Exception):
            await warm

    while True:
        try:
            if await _should_refresh(interval_s) and await lock.acquire():
                try:
                    report = await refresh_once(gdelt, store, config=config, lock=lock)
                    log.info("news refresh: %s", report)
                finally:
                    await lock.release()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the loop outlives any one bad cycle
            log.warning("news refresh cycle failed", exc_info=True)

        # Jitter so two workers that started together stop contending for the
        # lock at the same instant every hour, forever.
        await asyncio.sleep(interval_s + random.uniform(0.0, jitter_s))  # noqa: S311 - not crypto


async def _should_refresh(interval_s: float) -> bool:
    """True when the stored panel is missing or older than one interval.

    Checked before taking the lock so a restarting worker warms a cold panel in
    minutes rather than waiting a full hour — the difference between a demo
    working and a demo showing an empty box.
    """
    for scope in SCOPES:
        age = snapshot.age_seconds(await asyncio.to_thread(snapshot.latest, scope))
        if age is None or age >= interval_s:
            return True
    return False


# --- the CLI --------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the CeyNex news sidecar.")
    parser.add_argument("--once", action="store_true", default=True, help="one pass (the default)")
    parser.add_argument(
        "--ensure-collection",
        action="store_true",
        help="create the Qdrant collection and its payload indexes, then continue",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the GDELT URLs this would call and exit without calling them",
    )
    parser.add_argument("--topic", action="append", default=[], help="only this topic id (repeatable)")
    parser.add_argument("--scope", choices=SCOPES, help="only topics in this scope")
    parser.add_argument("--no-index", action="store_true", help="compute trending, index nothing")
    return parser.parse_args(argv)


def _selected(config: dict, args: argparse.Namespace) -> list[WatchTopic]:
    topics = load_watchlist(config)
    if args.topic:
        wanted = set(args.topic)
        topics = [topic for topic in topics if topic.id in wanted]
    if args.scope:
        topics = [topic for topic in topics if topic.scope == args.scope]
    return topics


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = settings.news_config()
    gdelt_config = config.get("gdelt", {})
    topics = _selected(config, args)

    if not topics:
        log.error("no topics selected")
        return 1

    if args.dry_run:
        # The single most useful flag here: a watchlist query GDELT silently
        # rejects looks identical to one that simply found nothing.
        # The configured endpoint, not the default: --dry-run exists to show the
        # calls that would actually be made, and this deployment's may be http.
        client = GdeltClient(base_url=settings.news_base_url())
        for topic in topics:
            print(f"# {topic.id} ({topic.scope}) — {topic.label}")
            print(
                client.describe(
                    client.volume_params(
                        topic.query, timespan=str(gdelt_config.get("timespan_trending", "7d"))
                    )
                )
            )
            print(
                client.describe(
                    client.article_params(
                        topic.query,
                        timespan=str(gdelt_config.get("timespan_topic_articles", "24h")),
                        max_records=int(gdelt_config.get("max_records_topic", 20)),
                        sort="DateDesc",
                    )
                )
            )
            print()
        await client.close()
        return 0

    gdelt = GdeltClient.from_settings()
    if gdelt is None:
        log.error("news is disabled (CEYNEX_NEWS) — nothing to do")
        return 1

    store = None if args.no_index else NewsStore.from_settings()
    if store is not None and args.ensure_collection:
        await store.ensure_collection()

    snapshot.ensure_table()

    try:
        report = await refresh_once(
            gdelt, store, config=config, topics=topics, index=not args.no_index
        )
    finally:
        await gdelt.close()
        if store is not None:
            await store.close()

    log.info("news refresh: %s", report)
    print(report)
    return 0 if report.topics_ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(_main(argv))


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())


__all__ = [
    "LOCK_KEY",
    "Lock",
    "NoopLock",
    "RedisLock",
    "RefreshReport",
    "build_lock",
    "main",
    "refresh_once",
    "run_forever",
]
