"""Assertions for the GDELT client — parsing, degradation, throttling, caching.

No network: every request is answered by an `httpx.MockTransport` the client
takes as a constructor argument. The transport seam is designed in rather than
monkeypatched, because the thing most worth asserting here is what happens on a
*response*, and reaching that through a patched module is indirection with no
payoff.
"""

import json
from pathlib import Path

import httpx
import pytest

from ceynex.news.gdelt import (
    LANGUAGE_CLAUSE,
    MAX_RECORDS_CEILING,
    GdeltClient,
    GdeltUnavailableError,
)
from ceynex.news.throttle import NoThrottle

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def client_returning(
    *responses: httpx.Response, tmp_path: Path, throttle=None, **kwargs
) -> tuple[GdeltClient, list[httpx.Request]]:
    """A client whose transport replays `responses` in order, then repeats the last."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    client = GdeltClient(
        throttle=throttle if throttle is not None else NoThrottle(),
        cache_dir=tmp_path / "cache",
        forensic_dir=tmp_path / "forensic",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )
    return client, seen


def ok(content: bytes) -> httpx.Response:
    return httpx.Response(200, content=content)


# --- parsing -------------------------------------------------------------


async def test_a_real_artlist_response_parses_into_articles(tmp_path):
    client, _ = client_returning(ok(fixture_bytes("gdelt_artlist_tea.json")), tmp_path=tmp_path)

    articles = await client.articles("ceylon tea")

    assert len(articles) == 9  # ten records, one of which has no url
    assert all(article.url for article in articles)
    await client.close()


async def test_a_record_without_a_url_is_dropped_rather_than_kept_unlinkable(tmp_path):
    """A headline with no link is not something the panel can render."""
    client, _ = client_returning(ok(fixture_bytes("gdelt_artlist_tea.json")), tmp_path=tmp_path)

    articles = await client.articles("ceylon tea")

    assert not any(article.url == "" for article in articles)
    await client.close()


async def test_an_empty_response_is_an_ordinary_answer_not_an_error(tmp_path):
    """GDELT returns `{}` with no `articles` key when nothing matched."""
    client, _ = client_returning(ok(fixture_bytes("gdelt_artlist_empty.json")), tmp_path=tmp_path)

    assert await client.articles("something nobody wrote about") == []
    await client.close()


async def test_the_timeline_parses_into_dated_counts(tmp_path):
    client, _ = client_returning(ok(fixture_bytes("gdelt_timeline_7d.json")), tmp_path=tmp_path)

    points = await client.volume("ceylon tea")

    assert len(points) == 149
    assert points == sorted(points), "callers assume chronological order"
    await client.close()


async def test_an_empty_timeline_is_an_ordinary_answer(tmp_path):
    client, _ = client_returning(ok(b'{"timeline": []}'), tmp_path=tmp_path)

    assert await client.volume("nothing") == []
    await client.close()


# --- the failure that actually happens -----------------------------------


async def test_a_200_carrying_plain_text_degrades_rather_than_raising_a_decode_error(tmp_path):
    """GDELT answers 200 with a text complaint for a query it dislikes.

    `response.json()` then raises JSONDecodeError on a 2xx, which is not a shape
    any caller expects. The captured body is the literal string GDELT sent for
    `query=a`.
    """
    client, _ = client_returning(ok(fixture_bytes("gdelt_error_body.txt")), tmp_path=tmp_path)

    with pytest.raises(GdeltUnavailableError):
        await client.articles("a")
    await client.close()


async def test_an_unparseable_body_is_always_kept_on_disk(tmp_path):
    """The interactive path skips forensic copies, except for this case.

    When a watchlist query silently stops returning anything, the only useful
    question is what the API actually sent.
    """
    client, _ = client_returning(ok(fixture_bytes("gdelt_error_body.txt")), tmp_path=tmp_path)

    with pytest.raises(GdeltUnavailableError):
        await client.articles("a", keep_forensic=False)

    written = list((tmp_path / "forensic").rglob("*.json"))
    assert len(written) == 1
    assert b"too short or too long" in written[0].read_bytes()
    await client.close()


async def test_a_json_body_of_the_wrong_shape_degrades(tmp_path):
    client, _ = client_returning(ok(b'["not", "an", "object"]'), tmp_path=tmp_path)

    with pytest.raises(GdeltUnavailableError):
        await client.articles("tea")
    await client.close()


# --- retry and refusal ---------------------------------------------------


async def test_a_transport_error_is_retried_then_degrades(tmp_path):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise httpx.ConnectError("no route to host")

    client = GdeltClient(
        throttle=NoThrottle(),
        cache_dir=tmp_path / "cache",
        forensic_dir=tmp_path / "forensic",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(GdeltUnavailableError):
        await client.articles("tea", attempts=2, backoff_base_s=0.0)

    assert len(seen) == 2
    await client.close()


async def test_a_429_is_not_retried_into(tmp_path):
    """Retrying into a courtesy limit is how it becomes a ban."""
    client, seen = client_returning(
        httpx.Response(429, headers={"retry-after": "60"}), tmp_path=tmp_path
    )

    with pytest.raises(GdeltUnavailableError, match="rate limited"):
        await client.articles("tea", attempts=4, backoff_base_s=0.0)

    assert len(seen) == 1
    await client.close()


async def test_a_refused_throttle_slot_makes_no_http_call_at_all(tmp_path):
    """The whole point of refusing is not to spend the call."""

    class AlwaysBusy:
        async def acquire(self, *, max_wait_s):
            return False

    client, seen = client_returning(
        ok(fixture_bytes("gdelt_artlist_tea.json")), tmp_path=tmp_path, throttle=AlwaysBusy()
    )

    with pytest.raises(GdeltUnavailableError, match="throttle"):
        await client.articles("tea")

    assert seen == []
    await client.close()


# --- caching -------------------------------------------------------------


async def test_an_identical_call_inside_the_ttl_does_not_hit_the_network_twice(tmp_path):
    client, seen = client_returning(
        ok(fixture_bytes("gdelt_artlist_tea.json")), tmp_path=tmp_path, cache_ttl_s=900.0
    )

    first = await client.articles("ceylon tea")
    second = await client.articles("ceylon tea")

    assert len(seen) == 1
    assert [a.url for a in first] == [a.url for a in second]
    await client.close()


async def test_a_corrupt_cache_entry_heals_instead_of_poisoning_every_later_call(tmp_path):
    """A truncated entry from a killed process must not be permanent."""
    client, seen = client_returning(
        ok(fixture_bytes("gdelt_artlist_tea.json")), tmp_path=tmp_path, cache_ttl_s=900.0
    )
    await client.articles("ceylon tea")
    (cached,) = list((tmp_path / "cache").glob("*.json"))
    cached.write_bytes(b'{"articles": [trunca')

    articles = await client.articles("ceylon tea")

    assert len(seen) == 2
    assert articles
    await client.close()


async def test_an_expired_cache_entry_is_refetched(tmp_path):
    client, seen = client_returning(
        ok(fixture_bytes("gdelt_artlist_tea.json")), tmp_path=tmp_path, cache_ttl_s=0.0
    )

    await client.articles("ceylon tea")
    await client.articles("ceylon tea")

    assert len(seen) == 2
    await client.close()


# --- request construction ------------------------------------------------


async def test_the_language_clause_is_appended_once(tmp_path):
    """Appending it is idempotent, so a caller may or may not have set it.

    The two spellings normalise to identical params, which is why the second
    call is served from the first one's cache entry rather than spending a
    second GDELT slot on the same question.
    """
    client, seen = client_returning(ok(b'{"articles": []}'), tmp_path=tmp_path)

    await client.articles("ceylon tea")
    await client.articles(f"ceylon tea {LANGUAGE_CLAUSE}")

    assert len(seen) == 1
    assert seen[0].url.params["query"].count("sourcelang") == 1
    await client.close()


async def test_maxrecords_is_clamped_to_the_documented_ceiling(tmp_path):
    """Asking for more than 250 does not get more than 250; it just looks wrong."""
    client, seen = client_returning(ok(b'{"articles": []}'), tmp_path=tmp_path)

    await client.articles("tea", max_records=5000)

    assert seen[0].url.params["maxrecords"] == str(MAX_RECORDS_CEILING)
    await client.close()


async def test_trending_volume_uses_timelinevolraw_not_artlist(tmp_path):
    """ArtList saturates at maxrecords, so it cannot rank the busy topics."""
    client, seen = client_returning(ok(fixture_bytes("gdelt_timeline_7d.json")), tmp_path=tmp_path)

    await client.volume("tea")

    assert seen[0].url.params["mode"] == "TimelineVolRaw"
    assert "maxrecords" not in seen[0].url.params
    await client.close()


async def test_the_fixture_still_contains_the_traps_it_guards():
    """A fixture tidied of its awkward rows stops testing anything.

    Every one of these was in a real GDELT response and every one breaks
    something if unhandled: the `:443` twin becomes a duplicate point, the
    tracking parameters become a third, the malformed date crashes the batch,
    and the space-padded title renders as tokenizer output.
    """
    raw = json.loads(fixture_bytes("gdelt_artlist_tea.json"))
    urls = [record["url"] for record in raw["articles"]]

    assert any(":443" in url for url in urls), "the port-twin trap"
    assert any("utm_source" in url for url in urls), "the tracking-parameter trap"
    assert any(url == "" for url in urls), "the missing-url trap"
    assert any(
        record["seendate"] == "not-a-real-timestamp" for record in raw["articles"]
    ), "the malformed-date trap"
    assert any(" . " in record["title"] for record in raw["articles"]), "the tokenizer trap"
    assert any("socialimage" not in record for record in raw["articles"]), "the absent-field trap"


# --- the endpoint ---------------------------------------------------------


def test_the_endpoint_defaults_to_https():
    from ceynex.settings import news_base_url

    assert news_base_url().startswith("https://")


def test_the_endpoint_can_be_overridden_per_deployment(monkeypatch):
    """`api.gdeltproject.org` refuses TLS from some networks.

    The deployed backend VM is one of them: DNS resolves, port 80 answers
    normally, and every connection to :443 is reset — while github.com and
    api.openai.com over TLS are fine from the same host. `config/` ships baked
    into the image, so the value has to be settable per machine.
    """
    from ceynex.settings import news_base_url

    monkeypatch.setenv("CEYNEX_GDELT_BASE_URL", "http://api.gdeltproject.org/api/v2/doc/doc")

    assert news_base_url() == "http://api.gdeltproject.org/api/v2/doc/doc"


async def test_the_client_calls_whatever_endpoint_it_was_given(tmp_path):
    client, seen = client_returning(
        ok(b'{"articles": []}'), tmp_path=tmp_path, base_url="http://example.test/doc"
    )

    await client.articles("tea")

    assert str(seen[0].url).startswith("http://example.test/doc")
    await client.close()
