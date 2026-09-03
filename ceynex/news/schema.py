"""Supports docs/ARCHITECTURE_DELTA.md D11 — the news-article payload contract.

The shape of a `ceynex_news` Qdrant point, written down once, for the same reason
`retrieval/schema.py` exists: the refresher writes this payload and `store.py`
filters on it, and a filter on a key that does not exist matches no points and
raises nothing.

**News is not evidence, and this module is where that is enforced.**
`PolicyChunk` carries a `.citation` property because a policy passage is only
usable if a reader can open the document at the right page. `NewsArticle`
deliberately has no `.citation` and no `.to_evidence()`. The absence is the
design: SRS 3.1.4 says every claim must be checkable against a verified source,
and a headline from an outlet nobody vetted is not one. It sits beside the
answer, labelled, and never inside it.

Three further defences, because one convention in one file is not enough:

1. `NEWS_KIND` is on every point. `PolicyChunk.from_payload` uses `.get(..., "")`
   throughout and raises nothing, so a mis-set `QDRANT_COLLECTION` would happily
   yield chunks with empty text and cite them. `store.py` refuses to open the
   policy collection at all.
2. `SourceId` in `ceynex-web/src/types/contracts.ts` is a closed union, and
   `ceynex-contracts` is frozen behind a three-reviewer PR. Emitting news as
   `Evidence` is structurally impossible without that PR.
3. The web panel carries the sentence "Not used to produce the answer above" on
   screen, not only in a docstring.

**Stdlib only, deliberately.** `qdrant-client` is the optional `[policy]` extra
and `httpx` belongs to the client; a module every test can import must need
neither. Same discipline `retrieval/schema.py` keeps.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# --- payload keys ---------------------------------------------------------
#
# Named rather than spelled inline: a typo in a filter key is a silent empty
# result rather than an error.

ARTICLE_ID = "article_id"
URL = "url"
CANONICAL_URL = "canonical_url"
TITLE = "title"
DOMAIN = "domain"
LANGUAGE = "language"
SOURCE_COUNTRY = "sourcecountry"
SEEN_AT = "seen_at"
SEEN_TS = "seen_ts"
SOCIAL_IMAGE = "social_image"
TOPIC = "topic"
SCOPE = "scope"
INGESTED_TS = "ingested_ts"
KIND = "kind"

#: The value of `KIND` on every point this package writes. A discriminator, so a
#: reader that finds itself in the wrong collection can tell, rather than
#: silently treating a headline as a policy passage.
NEWS_KIND = "news"

#: Payload fields that get a Qdrant keyword index. Each is filtered on at query
#: time; without an index Qdrant falls back to scanning the payload.
INDEXED_KEYWORD_FIELDS = (
    ARTICLE_ID,
    CANONICAL_URL,
    DOMAIN,
    LANGUAGE,
    SOURCE_COUNTRY,
    TOPIC,
    SCOPE,
    KIND,
)

#: Filtered by *range*, for the retention sweep, so it needs an integer index
#: rather than a keyword one.
INDEXED_INTEGER_FIELDS = (SEEN_TS,)

#: Watchlist scopes. Two, because the trending panel shows two columns.
SCOPES = ("sri_lanka", "global")


# --- url canonicalisation -------------------------------------------------

#: Query parameters dropped before an article's identity is computed.
#:
#: **A deny-list, never an allow-list.** An allow-list is the tempting shape and
#: it is destructive: plenty of real outlets carry the article id in a query
#: parameter (`?id=12345`, `?p=98`, `?storyid=...`), and an allow-list would
#: collapse every article on such a site into a single point — silently, and
#: only on the sites it happens to. Dropping a tracking parameter we failed to
#: list costs one duplicate point; dropping an identifying one costs an entire
#: outlet's coverage. `tests/news/test_schema.py` pins this decision.
_TRACKING_PARAMS = frozenset(
    {
        "amp",
        "at_campaign",
        "at_medium",
        "cmp",
        "fbclid",
        "gclid",
        "ito",
        "mc_cid",
        "mc_eid",
        "msclkid",
        "ref",
        "referrer",
        "source",
        "spm",
        "twclid",
        "__twitter_impression",
    }
)

_TRACKING_PREFIXES = ("utm_",)


def canonical_url(url: str) -> str:
    """The identity of an article, independent of how GDELT happened to spell it.

    GDELT returns the same story under several spellings — `asiaone.com` and
    `asiaone.com:443` both appeared in the first live response we captured — and
    syndicated links arrive with tracking parameters attached. Without
    normalisation each spelling becomes its own point and the panel shows the
    same headline three times.

    Scheme is forced to https because a scheme is not identity: GDELT emits both
    for the same article, and no reader distinguishes them.
    """
    raw = (url or "").strip()
    if not raw:
        return ""

    try:
        parts = urlsplit(raw)
    except ValueError:
        # A URL malformed enough to fail parsing is not worth a crash; it just
        # gets its own identity, which is the same outcome as not recognising it.
        return raw

    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return raw

    # Port only survives when it is non-default. `urlsplit.port` raises on a
    # malformed port, which a hostile URL can carry.
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port and port not in (80, 443) else host

    params = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
        and not key.lower().startswith(_TRACKING_PREFIXES)
    ]
    query = urlencode(sorted(params))

    path = parts.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    return urlunsplit(("https", netloc, path, query, ""))


#: Fixed namespace so re-fetching the same article overwrites its own point
#: rather than adding a second copy. The refresher re-runs hourly and most of
#: what it fetches it has already seen; an autoincrementing id would duplicate
#: the whole corpus every cycle. Never regenerate this — it *is* the dedup.
_NEWS_NAMESPACE = uuid.UUID("15f2c660-6659-475a-8dab-1dc942efe3b8")


def article_id(url: str) -> str:
    """Deterministic point id. Same article, however spelled, maps to one point."""
    return str(uuid.uuid5(_NEWS_NAMESPACE, canonical_url(url)))


#: Apostrophes are *deleted*, everything else non-alphanumeric becomes a space.
#: The distinction matters: outlets disagree about whether it is "Sri Lanka's
#: exports" or "Sri Lankas exports", and turning the apostrophe into a space
#: makes those two differ by a word boundary — which is precisely the difference
#: this function exists to erase.
_TITLE_APOSTROPHE = re.compile(r"[’'`]")
_TITLE_NOISE = re.compile(r"[^a-z0-9]+")


def title_key(title: str) -> str:
    """A normalised title, for suppressing syndication duplicates in a response.

    One wire story runs on forty sites under near-identical headlines. Collapsing
    them is worth doing *in a response*, where three rows saying the same thing
    is a worse panel — and worth **not** doing in the store, where they are
    genuinely distinct articles and where a collapse would be irreversible.
    Cross-outlet entity resolution is not attempted; see docs/DEFERRED.md.
    """
    folded = _TITLE_APOSTROPHE.sub("", (title or "").casefold())
    return _TITLE_NOISE.sub(" ", folded).strip()


# --- dates ----------------------------------------------------------------


def parse_seendate(raw: str | None) -> datetime | None:
    """`20260901T041500Z` -> an aware UTC datetime, or None.

    None rather than an exception: one malformed date in a batch of seventy-five
    must not cost the other seventy-four.

    **This is when GDELT first *saw* the article, not when it was published.**
    Everything downstream says "seen" for that reason. In a project whose whole
    claim is that figures are checkable, quietly relabelling a crawl timestamp as
    a publication date would be the one unsourced number on the page.
    """
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


# --- relevance labels -----------------------------------------------------

#: Cuts on `sigmoid(cross-encoder logit)`. 0.5 is the model's own "more relevant
#: than not" boundary; 0.1 is where the tail stops being worth a row.
_STRONG_P = 0.5
_RELATED_P = 0.1


def relevance_label(score: float | None) -> str:
    """`strong` | `related` | `loose` | `unscored`.

    Three buckets, and the number itself is never rendered. `sigmoid(logit)`
    shown as "73% relevant" would be a calibration claim this project cannot
    support, sitting on the same page as `ConfidenceBadge`, which does real
    calibrated work derived in `orchestrator/confidence.py`. Two things that look
    like probabilities, one of them meaning nothing, is worse than one.

    `unscored` is the honest answer when the `[policy]` extra is absent and no
    cross-encoder ran — the articles are still real and still ordered by GDELT's
    own relevance sort, they just carry no judgement of ours.
    """
    if score is None:
        return "unscored"
    probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(score)))))
    if probability >= _STRONG_P:
        return "strong"
    if probability >= _RELATED_P:
        return "related"
    return "loose"


# --- the article ----------------------------------------------------------


@dataclass(frozen=True)
class NewsArticle:
    """One headline, with everything needed to link to it and rank it.

    No `.citation`, and no `.to_evidence()`. See the module docstring — the
    absence is the contract, not an oversight, and a future reader adding either
    one should have to delete this paragraph first.
    """

    url: str
    title: str
    domain: str = ""
    language: str = ""
    source_country: str = ""
    seen_at: datetime | None = None
    social_image: str = ""
    topic: str = ""
    scope: str = ""
    relevance: float | None = None

    @property
    def article_id(self) -> str:
        return article_id(self.url)

    @property
    def canonical_url(self) -> str:
        return canonical_url(self.url)

    def embed_text(self) -> str:
        """What goes into the dense and sparse vectors: the title, alone.

        Not the domain, country or date. Those pull headlines toward each other
        by *source* rather than by subject — every Reuters story becoming a
        neighbour of every other Reuters story is the opposite of what a
        relevance search wants. Filtering on them is what the keyword payload
        indexes are for.
        """
        return self.title.strip()

    @classmethod
    def from_gdelt(cls, record: dict[str, Any], *, topic: str = "", scope: str = "") -> NewsArticle:
        """Build one from a GDELT `ArtList` record.

        Tolerant of every field being absent. GDELT's article objects are not a
        versioned contract, and a missing `socialimage` must not cost the row.
        """
        return cls(
            url=str(record.get("url") or "").strip(),
            # GDELT space-pads punctuation in titles ("U . S . trade"), which is
            # an artefact of its tokenizer rather than the outlet's headline.
            title=_tidy_title(str(record.get("title") or "")),
            domain=str(record.get("domain") or "").strip().lower(),
            language=str(record.get("language") or "").strip(),
            source_country=str(record.get("sourcecountry") or "").strip(),
            seen_at=parse_seendate(record.get("seendate")),
            social_image=str(record.get("socialimage") or "").strip(),
            topic=topic,
            scope=scope,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any], relevance: float | None = None) -> NewsArticle:
        seen_ts = payload.get(SEEN_TS)
        return cls(
            url=payload.get(URL, ""),
            title=payload.get(TITLE, ""),
            domain=payload.get(DOMAIN, ""),
            language=payload.get(LANGUAGE, ""),
            source_country=payload.get(SOURCE_COUNTRY, ""),
            seen_at=datetime.fromtimestamp(seen_ts, tz=UTC) if seen_ts else None,
            social_image=payload.get(SOCIAL_IMAGE, ""),
            topic=payload.get(TOPIC, ""),
            scope=payload.get(SCOPE, ""),
            relevance=relevance,
        )

    def to_payload(self, *, ingested_ts: int | None = None) -> dict[str, Any]:
        return {
            ARTICLE_ID: self.article_id,
            URL: self.url,
            CANONICAL_URL: self.canonical_url,
            TITLE: self.title,
            DOMAIN: self.domain,
            LANGUAGE: self.language,
            SOURCE_COUNTRY: self.source_country,
            SEEN_AT: self.seen_at.isoformat() if self.seen_at else "",
            SEEN_TS: int(self.seen_at.timestamp()) if self.seen_at else 0,
            SOCIAL_IMAGE: self.social_image,
            TOPIC: self.topic,
            SCOPE: self.scope,
            INGESTED_TS: ingested_ts if ingested_ts is not None else int(_now().timestamp()),
            KIND: NEWS_KIND,
        }


_TITLE_SPACING = re.compile(r"\s+([,.;:!?%])")
_TITLE_WHITESPACE = re.compile(r"\s{2,}")


def _tidy_title(raw: str) -> str:
    """Undo GDELT's tokenizer spacing. Cosmetic, and confined to one place."""
    tidied = _TITLE_SPACING.sub(r"\1", raw.strip())
    return _TITLE_WHITESPACE.sub(" ", tidied)


def _now() -> datetime:
    return datetime.now(tz=UTC)


__all__ = [
    "ARTICLE_ID",
    "CANONICAL_URL",
    "DOMAIN",
    "INDEXED_INTEGER_FIELDS",
    "INDEXED_KEYWORD_FIELDS",
    "INGESTED_TS",
    "KIND",
    "LANGUAGE",
    "NEWS_KIND",
    "SCOPE",
    "SCOPES",
    "SEEN_AT",
    "SEEN_TS",
    "SOCIAL_IMAGE",
    "SOURCE_COUNTRY",
    "TITLE",
    "TOPIC",
    "URL",
    "NewsArticle",
    "article_id",
    "canonical_url",
    "parse_seendate",
    "relevance_label",
    "title_key",
]
