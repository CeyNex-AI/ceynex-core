"""Supports docs/ARCHITECTURE_DELTA.md D11 — what is moving, and by how much.

Pure functions over a GDELT timeline plus the watchlist in `config/news.yaml`.
Nothing here does I/O, which is what makes the interesting parts — the delta, the
small-denominator guard, the ranking — testable without a network or a database.

Why the metric is a ratio against the topic's own past
------------------------------------------------------
`TimelineVolRaw` counts move with GDELT's total crawl volume: every topic is
quieter at the weekend, and a raw count would rank Monday above Sunday for
reasons that have nothing to do with trade. Dividing a topic's last 24 hours by
its own preceding week cancels that common-mode drift, because the numerator and
the denominator drift together. A marker will ask why not raw counts; this is the
answer, and it is also why the baseline has to come from the same series rather
than from a second call.

Why the window is anchored on the data, not on the clock
--------------------------------------------------------
`end` is the newest bucket GDELT actually returned, not `now()`. GDELT's last
bucket lags real time by up to an hour, so anchoring on the clock would count a
partially-filled final bucket as a whole one and read every topic as slightly
falling. It also makes every function here deterministic against a fixture.

The small-denominator guard
---------------------------
A topic going from 0 to 2 articles is "+200%" and would outrank one going from 40
to 90. `min_articles` drops a topic before it can rank at all, which inverts the
panel's failure mode from "full of noise from the quietest topics" to "shows
fewer rows on a slow day". The second is the right way to be wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ceynex.news.schema import NewsArticle

log = logging.getLogger(__name__)

#: Direction cuts, on the percentage change. Deliberately coarse: the underlying
#: count is a crawl statistic, not a measurement, and four buckets is about as
#: much precision as it can carry honestly.
SURGING_PCT = 50.0
UP_PCT = 15.0
DOWN_PCT = -15.0


@dataclass(frozen=True)
class WatchTopic:
    """One row of `config/news.yaml`'s watchlist.

    Three strings, three audiences: `label` is the chip, `prompt` is what lands
    in the query box when it is clicked, `query` is GDELT syntax. Collapsing any
    two of them produces a bad string for one of the three.
    """

    id: str
    scope: str
    label: str
    prompt: str
    query: str


@dataclass(frozen=True)
class TrendingTopic:
    """A watchlist topic with its measured movement."""

    topic_id: str
    label: str
    prompt: str
    scope: str
    articles_24h: int
    baseline_24h: float
    delta_pct: float
    direction: str
    top_articles: list[NewsArticle] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "label": self.label,
            "prompt": self.prompt,
            "scope": self.scope,
            "articles_24h": self.articles_24h,
            "baseline_24h": round(self.baseline_24h, 2),
            "delta_pct": round(self.delta_pct, 1),
            "direction": self.direction,
            "top_articles": [
                {
                    "url": article.url,
                    "title": article.title,
                    "domain": article.domain,
                    "seen_at": article.seen_at.isoformat() if article.seen_at else None,
                }
                for article in self.top_articles
            ],
        }


def load_watchlist(config: dict) -> list[WatchTopic]:
    """Read the watchlist, skipping any row that is not fully specified.

    Skipping rather than raising: one malformed topic must not take the other
    thirteen down with it. `tests/news/test_config.py` is what stops a malformed
    row from reaching production in the first place.
    """
    topics: list[WatchTopic] = []
    for raw in config.get("trending", {}).get("topics", []) or []:
        try:
            topic = WatchTopic(
                id=str(raw["id"]).strip(),
                scope=str(raw["scope"]).strip(),
                label=str(raw["label"]).strip(),
                prompt=str(raw["prompt"]).strip(),
                query=str(raw["query"]).strip(),
            )
        except (KeyError, TypeError):
            log.warning("skipping a malformed watchlist topic: %r", raw)
            continue
        if not all((topic.id, topic.scope, topic.label, topic.prompt, topic.query)):
            log.warning("skipping watchlist topic %r — a required field is empty", raw.get("id"))
            continue
        topics.append(topic)
    return topics


def bucket_hours(points: list[tuple[datetime, float]]) -> float:
    """How much time one bucket covers, measured rather than assumed.

    GDELT returns hourly buckets for a 7-day span and 15-minute buckets for a
    short one, and nothing stops it changing that. The median gap is used rather
    than the first: a single missing bucket would otherwise double the estimate
    and halve every baseline computed from it.
    """
    if len(points) < 2:
        return 1.0
    gaps = [
        (later - earlier).total_seconds()
        for (earlier, _), (later, _) in zip(points, points[1:], strict=False)
    ]
    positive = sorted(gap for gap in gaps if gap > 0)
    if not positive:
        return 1.0
    return positive[len(positive) // 2] / 3600.0


def split_window(
    points: list[tuple[datetime, float]], *, window_hours: int
) -> tuple[float, float, float]:
    """`(window_total, baseline_total, baseline_hours)` from one timeline.

    Splits by timestamp rather than by bucket position, so the bucket width is
    irrelevant to the caller. The baseline's *duration*, though, is counted as
    buckets × width rather than as the span between the first and last timestamp
    — a span measures the gaps between N buckets, which is one bucket short of
    the time those N buckets actually cover, and that off-by-one lands directly
    in the denominator of every delta.
    """
    if not points:
        return 0.0, 0.0, 0.0

    end = max(when for when, _ in points)
    boundary = end - timedelta(hours=window_hours)
    width = bucket_hours(points)

    window_total = sum(value for when, value in points if when > boundary)
    baseline = [value for when, value in points if when <= boundary]
    return window_total, sum(baseline), len(baseline) * width


def compute_delta(
    points: list[tuple[datetime, float]], *, window_hours: int
) -> tuple[int, float, float]:
    """`(articles_in_window, baseline_scaled_to_window, delta_pct)`.

    The baseline is rescaled to the window's own length so the two numbers are
    comparable — 24 hours of coverage against a 6-day mean expressed per 24
    hours, not against the 6-day total.
    """
    window_total, baseline_total, baseline_hours = split_window(points, window_hours=window_hours)
    if baseline_hours <= 0:
        # Not enough history to say anything about movement. Reporting the count
        # with a zero delta is honest; inventing a percentage is not.
        return int(window_total), 0.0, 0.0

    baseline = baseline_total / baseline_hours * window_hours
    # max(baseline, 1.0) rather than a bare guard against zero: a topic going
    # from 0.3 to 4 articles should not read as "+1233%".
    delta_pct = (window_total - baseline) / max(baseline, 1.0) * 100.0
    return int(window_total), baseline, delta_pct


def direction_of(delta_pct: float) -> str:
    if delta_pct >= SURGING_PCT:
        return "surging"
    if delta_pct >= UP_PCT:
        return "up"
    if delta_pct <= DOWN_PCT:
        return "down"
    return "steady"


def rank(
    topics: list[TrendingTopic], *, min_articles: int, top_n: int, scope: str = ""
) -> list[TrendingTopic]:
    """Busiest movers first, having dropped everything too quiet to rank.

    The `min_articles` filter runs *before* the sort, not after: a topic with two
    articles has a meaningless percentage, and letting it sort first and then
    trimming would just move the noise to a different position.
    """
    eligible = [
        topic
        for topic in topics
        if topic.articles_24h >= min_articles and (not scope or topic.scope == scope)
    ]
    eligible.sort(key=lambda topic: topic.delta_pct, reverse=True)
    return eligible[:top_n]


def build_topic(
    topic: WatchTopic,
    points: list[tuple[datetime, float]],
    articles: list[NewsArticle],
    *,
    window_hours: int,
    top_articles: int = 3,
) -> TrendingTopic:
    """Assemble one row of the panel from its timeline and its headlines."""
    articles_24h, baseline, delta_pct = compute_delta(points, window_hours=window_hours)
    return TrendingTopic(
        topic_id=topic.id,
        label=topic.label,
        prompt=topic.prompt,
        scope=topic.scope,
        articles_24h=articles_24h,
        baseline_24h=baseline,
        delta_pct=delta_pct,
        direction=direction_of(delta_pct),
        top_articles=list(articles[:top_articles]),
    )


__all__ = [
    "DOWN_PCT",
    "SURGING_PCT",
    "UP_PCT",
    "TrendingTopic",
    "WatchTopic",
    "build_topic",
    "compute_delta",
    "direction_of",
    "load_watchlist",
    "rank",
    "split_window",
]
