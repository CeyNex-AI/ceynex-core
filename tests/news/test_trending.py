"""Assertions for the trending maths — the delta, the guard, and the ranking.

All pure functions over a captured `TimelineVolRaw` response, so none of this
needs a network, a database or a model. These are the highest-value tests in the
package: a wrong delta produces a plausible-looking panel that is quietly
meaningless, which is exactly the failure nobody notices in a demo.

The fixture is 149 hourly buckets ending 2026-09-02T11:00Z — the same shape and
length the live API returned for `timespan=7d`. The last 24 hours carry 5
articles each (120 total); the preceding 125 hours carry 2 each (250 total,
which is 48 per 24 hours). So the expected delta is exactly +150%.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ceynex.news.gdelt import _parse_timeline
from ceynex.news.schema import NewsArticle
from ceynex.news.trending import (
    TrendingTopic,
    WatchTopic,
    build_topic,
    compute_delta,
    direction_of,
    load_watchlist,
    rank,
    split_window,
)

FIXTURES = Path(__file__).parent / "fixtures"


def timeline() -> list[tuple[datetime, float]]:
    return _parse_timeline(json.loads((FIXTURES / "gdelt_timeline_7d.json").read_text()))


def series(*, hours: int, per_bucket: float, bucket_minutes: int, end: datetime):
    """A synthetic series at an arbitrary bucket width."""
    step = timedelta(minutes=bucket_minutes)
    count = int(hours * 60 / bucket_minutes)
    return [(end - step * i, per_bucket) for i in range(count - 1, -1, -1)]


# --- the delta -----------------------------------------------------------


def test_the_window_and_baseline_come_out_of_one_timeline():
    window_total, baseline_total, baseline_hours = split_window(timeline(), window_hours=24)

    assert window_total == 120
    assert baseline_total == 250
    # 125 hourly buckets cover 125 hours. The span between the first and last of
    # them is 124 hours, and using that would inflate every baseline by ~1%.
    assert baseline_hours == 125.0


def test_the_delta_is_the_window_against_the_topics_own_baseline():
    articles_24h, baseline, delta_pct = compute_delta(timeline(), window_hours=24)

    assert articles_24h == 120
    assert baseline == 48.0
    assert delta_pct == 150.0


def test_the_baseline_is_rescaled_to_the_windows_own_length():
    """A 24h window against a 6-day *total* would read as a 600% collapse."""
    _, baseline, _ = compute_delta(timeline(), window_hours=24)

    assert baseline < 250, "the baseline is a per-window rate, not the raw total"


def test_bucket_width_does_not_change_the_answer():
    """GDELT returns hourly buckets for 7d and 15-minute buckets for short spans.

    Summing by timestamp rather than by bucket position makes the width
    irrelevant — this asserts the two agree rather than trusting that they do.
    """
    end = datetime(2026, 9, 2, 11, 0, tzinfo=UTC)
    hourly = series(hours=48, per_bucket=4.0, bucket_minutes=60, end=end)
    quarterly = series(hours=48, per_bucket=1.0, bucket_minutes=15, end=end)

    assert compute_delta(hourly, window_hours=24) == compute_delta(quarterly, window_hours=24)


def test_a_timeline_with_no_history_reports_a_count_and_no_movement():
    """Inventing a percentage from one day of data would be a made-up number."""
    end = datetime(2026, 9, 2, 11, 0, tzinfo=UTC)

    articles, baseline, delta = compute_delta(
        series(hours=24, per_bucket=3.0, bucket_minutes=60, end=end), window_hours=24
    )

    assert articles == 72
    assert baseline == 0.0
    assert delta == 0.0


def test_an_empty_timeline_does_not_divide_by_zero():
    assert compute_delta([], window_hours=24) == (0, 0.0, 0.0)


# --- the small-denominator guard -----------------------------------------


def topic(topic_id: str, *, articles: int, delta: float, scope: str = "sri_lanka") -> TrendingTopic:
    return TrendingTopic(
        topic_id=topic_id,
        label=topic_id,
        prompt=f"tell me about {topic_id}",
        scope=scope,
        articles_24h=articles,
        baseline_24h=1.0,
        delta_pct=delta,
        direction=direction_of(delta),
    )


def test_a_topic_too_quiet_to_measure_cannot_rank_at_all():
    """0 -> 2 articles is "+200%" and would otherwise beat 40 -> 90.

    Without this the panel fills with noise from the quietest topics, which
    inverts what it is for.
    """
    ranked = rank(
        [topic("noise", articles=2, delta=200.0), topic("real", articles=90, delta=125.0)],
        min_articles=3,
        top_n=6,
    )

    assert [t.topic_id for t in ranked] == ["real"]


def test_the_guard_runs_before_the_sort_not_after():
    """Trimming after sorting would just move the noise to a different row."""
    ranked = rank(
        [
            topic("noise-a", articles=1, delta=900.0),
            topic("noise-b", articles=2, delta=800.0),
            topic("real", articles=40, delta=60.0),
        ],
        min_articles=3,
        top_n=2,
    )

    assert [t.topic_id for t in ranked] == ["real"]


def test_ranking_is_by_movement_and_capped():
    ranked = rank(
        [topic(f"t{i}", articles=10, delta=float(i * 10)) for i in range(10)],
        min_articles=3,
        top_n=3,
    )

    assert [t.topic_id for t in ranked] == ["t9", "t8", "t7"]


def test_ranking_can_be_confined_to_one_scope():
    """The panel is two columns; each is ranked within itself."""
    ranked = rank(
        [
            topic("lk", articles=10, delta=10.0, scope="sri_lanka"),
            topic("gl", articles=10, delta=90.0, scope="global"),
        ],
        min_articles=3,
        top_n=6,
        scope="sri_lanka",
    )

    assert [t.topic_id for t in ranked] == ["lk"]


# --- direction -----------------------------------------------------------


def test_direction_buckets():
    assert direction_of(200.0) == "surging"
    assert direction_of(20.0) == "up"
    assert direction_of(0.0) == "steady"
    assert direction_of(-40.0) == "down"


# --- assembly ------------------------------------------------------------


def test_a_built_topic_carries_the_prompt_not_just_the_label():
    """The chip's click handler uses `prompt`; a label in the query box is useless."""
    watch = WatchTopic(
        id="lk_tea",
        scope="sri_lanka",
        label="Ceylon tea",
        prompt="How are Ceylon tea exports performing right now?",
        query='"Ceylon tea"',
    )

    built = build_topic(watch, timeline(), [], window_hours=24)

    assert built.prompt.startswith("How are")
    assert built.delta_pct == 150.0


def test_only_a_few_headlines_ride_along_with_each_topic():
    watch = WatchTopic(id="t", scope="global", label="T", prompt="p", query="q")
    articles = [NewsArticle(url=f"https://e.com/{i}", title=f"story {i}") for i in range(10)]

    built = build_topic(watch, timeline(), articles, window_hours=24, top_articles=3)

    assert len(built.top_articles) == 3


def test_the_snapshot_json_is_flat_enough_to_render_without_lookups():
    watch = WatchTopic(id="lk_tea", scope="sri_lanka", label="Ceylon tea", prompt="p", query="q")
    article = NewsArticle(
        url="https://e.com/a", title="Tea", seen_at=datetime(2026, 9, 1, tzinfo=UTC)
    )

    payload = build_topic(watch, timeline(), [article], window_hours=24).to_json()

    assert payload["label"] == "Ceylon tea"
    assert payload["direction"] == "surging"
    assert payload["top_articles"][0]["url"] == "https://e.com/a"


# --- the watchlist -------------------------------------------------------


def test_a_malformed_topic_is_skipped_rather_than_taking_the_others_down():
    watchlist = load_watchlist(
        {
            "trending": {
                "topics": [
                    {"id": "good", "scope": "global", "label": "G", "prompt": "p", "query": "q"},
                    {"id": "missing-query", "scope": "global", "label": "M", "prompt": "p"},
                    {"id": "", "scope": "global", "label": "E", "prompt": "p", "query": "q"},
                ]
            }
        }
    )

    assert [t.id for t in watchlist] == ["good"]


def test_the_real_watchlist_loads_completely():
    from ceynex.settings import news_config

    assert len(load_watchlist(news_config())) == len(news_config()["trending"]["topics"])
