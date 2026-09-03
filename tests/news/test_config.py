"""Assertions for config/news.yaml — the watchlist is data, so it needs a guard.

Cheap tests that catch the failure the demo would otherwise find: a YAML typo,
a duplicated topic id (which silently overwrites a topic in the snapshot), or a
scope the frontend has no column for (which drops the topic on the floor with
nothing raised).
"""

from ceynex.news.schema import SCOPES
from ceynex.settings import news_config

REQUIRED_TOPIC_KEYS = {"id", "scope", "label", "prompt", "query"}


def test_the_watchlist_loads():
    config = news_config()

    assert config["trending"]["topics"]


def test_every_topic_carries_all_three_strings():
    """label, prompt and query serve three different audiences.

    A topic missing `prompt` puts its `label` in the query box instead — a chip
    reading "Ceylon tea" becomes the literal question "Ceylon tea", which the
    router cannot do anything useful with.
    """
    for topic in news_config()["trending"]["topics"]:
        assert set(topic) >= REQUIRED_TOPIC_KEYS, f"{topic.get('id')} is missing keys"
        for key in REQUIRED_TOPIC_KEYS:
            assert str(topic[key]).strip(), f"{topic.get('id')}.{key} is empty"


def test_topic_ids_are_unique():
    """Two topics sharing an id means one silently replaces the other."""
    ids = [topic["id"] for topic in news_config()["trending"]["topics"]]

    assert len(ids) == len(set(ids))


def test_every_scope_has_a_column_in_the_ui():
    for topic in news_config()["trending"]["topics"]:
        assert topic["scope"] in SCOPES, f"{topic['id']} has scope {topic['scope']!r}"


def test_both_scopes_are_actually_populated():
    """An empty column renders as a missing section rather than an empty one."""
    scopes = {topic["scope"] for topic in news_config()["trending"]["topics"]}

    assert scopes == set(SCOPES)


def test_the_trending_window_fits_inside_the_span_that_gets_fetched():
    """The baseline is computed from the same single call the window comes from.

    A `baseline_days` longer than `timespan_trending` would average over days
    that were never fetched, quietly producing a baseline of zero for the
    missing part and inflating every delta.
    """
    config = news_config()
    fetched_days = int(str(config["gdelt"]["timespan_trending"]).rstrip("d"))

    assert config["trending"]["baseline_days"] <= fetched_days
    assert config["trending"]["window_hours"] <= fetched_days * 24


def test_every_query_obeys_both_of_gdelts_parenthesis_rules():
    """GDELT enforces two rules that pull in opposite directions.

        1. "Queries containing OR'd terms must be surrounded by ()."
        2. "Parentheses may only be used around OR'd statements."

    Satisfying only the second is what broke four of these queries; "fixing"
    them into bare OR lists then broke seven more on the first. Both violations
    come back as HTTP **200** with the rule as plain text, which reads exactly
    like a topic with no coverage — so nothing surfaces them except a human
    reading container logs. Hence a static check.

    The grammar that satisfies both: a sequence of quoted phrases and
    parenthesised OR groups, implicitly ANDed. `("a" OR "b") "c" (d OR e)`.
    """
    for topic in news_config()["trending"]["topics"]:
        query = topic["query"]

        # Rule 2: nothing but ORs inside a group.
        for group in _paren_groups(query):
            assert " AND " not in group.upper(), f"{topic['id']}: AND inside parentheses"
            assert "(" not in group, f"{topic['id']}: nested parentheses"
            if _term_count(group) >= 2:
                assert " OR " in f" {group} ".upper(), (
                    f"{topic['id']}: implicit AND inside parentheses — {group!r}"
                )

        # Rule 1: no OR left outside a group.
        assert " OR " not in _outside_parens(query).upper(), (
            f"{topic['id']}: an OR outside parentheses — {query!r}"
        )


def _outside_parens(query: str) -> str:
    """Everything at depth 0, with each `(...)` group blanked out."""
    out: list[str] = []
    depth = 0
    for char in query:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0:
            out.append(char)
    return "".join(out)


def _paren_groups(query: str) -> list[str]:
    """The text inside each top-level `(...)`, without the outer parentheses."""
    groups: list[str] = []
    depth = 0
    start = 0
    for i, char in enumerate(query):
        if char == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                groups.append(query[start:i])
    return groups


def _term_count(group: str) -> int:
    """Quoted phrases and bare words, counted as single terms."""
    import re

    return len(re.findall(r'"[^"]*"|\S+', re.sub(r'"[^"]*"', '"x"', group)))


def test_the_relevance_floor_is_not_the_policy_floor():
    """retrieval's 0.0 is tuned for passages; reusing it here empties the panel."""
    from ceynex.retrieval.client import MIN_RERANK_SCORE

    assert news_config()["relevance"]["min_score"] < MIN_RERANK_SCORE


def test_the_refresher_stays_well_inside_its_self_imposed_call_budget():
    """Two GDELT calls per topic per cycle, gated at `min_interval_s`.

    This is the check that a future watchlist of eighty topics fails loudly here
    rather than by quietly taking longer than the interval to complete a cycle.
    """
    config = news_config()
    calls = len(config["trending"]["topics"]) * 2
    seconds_of_calls = calls * config["gdelt"]["min_interval_s"]
    cycle_seconds = config["refresh"]["interval_minutes"] * 60

    assert seconds_of_calls < cycle_seconds / 4
