"""Assertions for the news payload contract — identity, dates, and labels.

No network, no qdrant, no fastembed. What is asserted here is the part that
fails silently: an article whose identity is computed two different ways becomes
two points, and the panel shows the same headline twice with nothing raised.
"""

from datetime import UTC, datetime

from ceynex.news.schema import (
    KIND,
    NEWS_KIND,
    NewsArticle,
    article_id,
    canonical_url,
    parse_seendate,
    relevance_label,
    title_key,
)
from ceynex.settings import news_collection, qdrant_collection

# --- identity ------------------------------------------------------------


def test_the_spellings_gdelt_actually_returns_collapse_to_one_article():
    """`asiaone.com` and `asiaone.com:443` were both in the first live response.

    Not hypothetical: the capture in fixtures/gdelt_artlist_tea.json contains
    the pair. Left alone they are two points and two identical rows in the panel.
    """
    spellings = [
        "https://www.asiaone.com:443/money/blockade-iran-oil-exports-stall",
        "http://asiaone.com/money/blockade-iran-oil-exports-stall/",
        "https://asiaone.com/money/blockade-iran-oil-exports-stall?utm_source=twitter#top",
        "https://ASIAONE.com/money/blockade-iran-oil-exports-stall?fbclid=abc123",
    ]

    assert len({canonical_url(url) for url in spellings}) == 1
    assert len({article_id(url) for url in spellings}) == 1


def test_an_identifying_query_parameter_survives_canonicalisation():
    """The deny-list-not-allow-list decision, pinned.

    Plenty of outlets carry the article id in a query parameter. An allow-list
    would drop it and collapse every article on such a site into one point —
    silently, and only on the sites that do it. This test exists so that
    "simplifying" the deny-list into an allow-list fails loudly instead.
    """
    first = canonical_url("https://example.com/news?id=482&utm_source=twitter")
    second = canonical_url("https://example.com/news?id=915&utm_source=twitter")

    assert "id=482" in first
    assert "utm_source" not in first
    assert first != second


def test_parameter_order_does_not_change_identity():
    a = canonical_url("https://example.com/x?b=2&a=1")
    b = canonical_url("https://example.com/x?a=1&b=2")

    assert a == b


def test_article_id_is_stable_across_runs():
    """Re-fetching an article must overwrite its own point, not add a second.

    The refresher re-runs hourly over largely the same articles. A changed
    namespace or a changed canonicalisation silently duplicates the corpus every
    cycle, so the expected value is hardcoded rather than recomputed.
    """
    assert article_id("https://example.com/a") == "8d54d8ea-62cc-557e-8b7b-30da12a3b4cb"


def test_a_url_too_malformed_to_parse_still_gets_an_identity():
    """A crash here would cost the whole batch for one bad row."""
    assert article_id("not a url at all")
    assert canonical_url("") == ""


# --- dates ---------------------------------------------------------------


def test_seendate_parses_to_an_aware_utc_datetime():
    parsed = parse_seendate("20260901T041500Z")

    assert parsed == datetime(2026, 9, 1, 4, 15, tzinfo=UTC)


def test_a_malformed_seendate_is_none_rather_than_an_exception():
    """One bad date in a batch of seventy-five must not cost the other seventy-four."""
    assert parse_seendate("garbage") is None
    assert parse_seendate("") is None
    assert parse_seendate(None) is None


# --- relevance labels ----------------------------------------------------


def test_relevance_labels_bucket_on_the_models_own_boundary():
    assert relevance_label(2.0) == "strong"
    assert relevance_label(0.0) == "strong"
    assert relevance_label(-1.0) == "related"
    assert relevance_label(-5.0) == "loose"


def test_an_unscored_article_says_so_rather_than_scoring_zero():
    """No cross-encoder ran, so we have no judgement — not a judgement of zero."""
    assert relevance_label(None) == "unscored"


def test_an_extreme_logit_does_not_overflow():
    assert relevance_label(-1000.0) == "loose"
    assert relevance_label(1000.0) == "strong"


# --- the article ---------------------------------------------------------


def test_from_gdelt_tolerates_every_field_being_absent():
    """GDELT's article object is not a versioned contract."""
    article = NewsArticle.from_gdelt({})

    assert article.url == ""
    assert article.seen_at is None
    assert article.relevance is None


def test_gdelt_tokenizer_spacing_is_undone_in_titles():
    article = NewsArticle.from_gdelt({"title": "Commerce Ministry discusses India - U . S . trade"})

    assert " . " not in article.title


def test_only_the_title_is_embedded():
    """Domain and country in a dense vector cluster headlines by source, not subject."""
    article = NewsArticle(
        url="https://example.com/a", title="Tea exports rise", domain="example.com",
        source_country="Sri Lanka",
    )

    assert article.embed_text() == "Tea exports rise"


def test_every_point_carries_the_news_discriminator():
    """So a reader that finds itself in the wrong collection can tell."""
    payload = NewsArticle(url="https://example.com/a", title="x").to_payload()

    assert payload[KIND] == NEWS_KIND


def test_the_payload_round_trips():
    original = NewsArticle(
        url="https://example.com/a",
        title="Tea exports rise",
        domain="example.com",
        language="English",
        source_country="Sri Lanka",
        seen_at=datetime(2026, 9, 1, 4, 15, tzinfo=UTC),
        topic="lk_tea",
        scope="sri_lanka",
    )

    restored = NewsArticle.from_payload(original.to_payload())

    assert restored.title == original.title
    assert restored.seen_at == original.seen_at
    assert restored.topic == original.topic


def test_a_news_article_cannot_be_turned_into_evidence():
    """SRS 3.1.4 — every claim beside an answer must be checkable.

    The absence of `.citation` / `.to_evidence()` is the contract, not an
    oversight. Adding either one should fail here first and make the author read
    the module docstring.
    """
    assert not hasattr(NewsArticle, "citation")
    assert not hasattr(NewsArticle, "to_evidence")


def test_the_news_collection_is_never_the_policy_collection():
    """A shared collection would put unvetted headlines behind the policy filters."""
    assert news_collection() != qdrant_collection()


# --- syndication ---------------------------------------------------------


def test_near_identical_syndicated_headlines_share_a_title_key():
    a = title_key("Sri Lanka's Export Earnings Surpass US$10 Billion")
    b = title_key("sri lankas export earnings surpass us 10 billion")

    assert a == b
