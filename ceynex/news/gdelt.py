"""Supports docs/ARCHITECTURE_DELTA.md D11 — the only way out to GDELT DOC 2.0.

Deliberately the same manners as `data/connectors/comtrade.py`: throttle, retry,
write the raw bytes to disk *before* parsing them, and never let a transport
detail escape as anything but this module's own exception. Two things differ,
both because this one runs inside a web request:

1. **Async.** `comtrade.py` uses sync `httpx.get` because it runs in a batch
   pipeline where blocking is free. A sync call here blocks the event loop and
   takes SRS 3.4.2's fifty concurrent users with it.

2. **Two retry budgets, not one.** comtrade's four attempts with exponential
   backoff to 30 s is ~40 s worst case — correct for a job nobody is waiting on,
   badly wrong inside a request that has 8 s in total. `attempts` and
   `backoff_base_s` are parameters rather than a decorator so the refresher and
   the route can each ask for the budget that suits them.

Two failures that actually happen
---------------------------------
**HTTP 200 with a non-JSON body**, when GDELT dislikes a query — too short, an
unbalanced quote, an operator it does not know. The captured body for `query=a`
is literally `Your query was too short or too long.`. `response.json()` then
raises `JSONDecodeError` on a 2xx, which is not a shape any caller expects. It is
caught here and re-raised as `GdeltUnavailableError`, and it is exactly why the
raw bytes are written to disk before parsing: when a watchlist topic silently
stops returning anything three weeks from now, the question is always "what did
the API actually send".

**HTTP 429.** GDELT publishes no rate limit but it does enforce one — confirmed
the hard way while validating the watchlist, by issuing three requests at once
and getting a 429 back for the trouble. This is why `throttle.py` exists and why
a 429 is never retried into: backing off is the only response that helps, and
retrying a courtesy limit is how it becomes a ban.

Two caches, doing different jobs
--------------------------------
- **Forensic** (`data/raw/gdelt/<date>/`): written before parsing, never read
  back on the hot path. Always for the refresher; for an interactive search
  only when parsing failed, because a file per user query is unbounded growth
  driven by user input.
- **Response** (`.cache/news/`): content-addressed, read on the hot path, TTL
  900 s. That TTL is GDELT's own minimum `timespan` granularity — a shorter one
  cannot return fresher data, it can only spend rate budget re-asking.

Both directories already sit inside volumes the backend mounts (`llm_cache` and
`dataset_data`), so persistence is free and no compose change is needed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ceynex import settings
from ceynex.news.schema import NewsArticle, parse_seendate
from ceynex.news.throttle import (
    MIN_INTERVAL_S,
    RATE_LIMIT_PENALTY_S,
    Throttle,
    build_throttle,
)

log = logging.getLogger(__name__)

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

#: GDELT indexes many languages; the watchlist queries and the cross-encoder are
#: both English-only, so this is appended to every query rather than repeated on
#: every line of config/news.yaml. Not a quality filter — see `blocked_domains`.
LANGUAGE_CLAUSE = "sourcelang:english"

#: GDELT's hard ceiling on `maxrecords`, whatever we ask for.
MAX_RECORDS_CEILING = 250

#: How long a caller will wait for a throttle slot before giving up and letting
#: the caller degrade. Overridden by the route, which fails fast to the store.
#:
#: Longer than `RATE_LIMIT_PENALTY_S` on purpose: the refresher should *wait out*
#: a rate-limit penalty rather than fail every remaining topic against it. Set
#: below the penalty, a single 429 costs the rest of the cycle.
DEFAULT_MAX_WAIT_S = 90.0


class GdeltUnavailableError(RuntimeError):
    """GDELT could not be reached, refused us, or sent something unparseable.

    Callers degrade on this; it never reaches the client. The news panel falling
    back to indexed headlines is a normal outcome, not an error path.
    """


class GdeltClient:
    """Async GDELT DOC 2.0 client. One per process, shared."""

    def __init__(
        self,
        *,
        base_url: str = GDELT_DOC_URL,
        timeout_s: float = 6.0,
        throttle: Throttle | None = None,
        cache_dir: Path | None = None,
        forensic_dir: Path | None = None,
        cache_ttl_s: float = 900.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._throttle = throttle if throttle is not None else build_throttle()
        self._cache_dir = cache_dir if cache_dir is not None else settings.news_cache_dir()
        self._forensic_dir = (
            forensic_dir if forensic_dir is not None else settings.data_dir() / "raw" / "gdelt"
        )
        self._cache_ttl_s = cache_ttl_s
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    # --- lifecycle -------------------------------------------------------

    @classmethod
    def from_settings(cls) -> GdeltClient | None:
        """Build one, or None when the sidecar is switched off.

        None is the "not configured" state every caller already handles, the same
        contract as `PolicyRetriever.from_settings()`. There is no key to check
        because GDELT DOC 2.0 does not have one.
        """
        if not settings.news_enabled():
            log.info("news disabled by CEYNEX_NEWS")
            return None
        config = settings.news_config().get("gdelt", {})
        base_url = settings.news_base_url()
        if base_url.startswith("http://"):
            # Loud, once, at startup. A plaintext call to a third party is a
            # decision someone should be able to find in the logs rather than
            # only in a config file — see config/news.yaml for why it is allowed.
            log.warning(
                "GDELT endpoint is plain HTTP (%s) — queries and headlines cross "
                "in clear text. Intended only where api.gdeltproject.org refuses TLS.",
                base_url,
            )
        return cls(
            base_url=base_url,
            timeout_s=float(config.get("request_timeout_s", 6.0)),
            throttle=build_throttle(float(config.get("min_interval_s", MIN_INTERVAL_S))),
            cache_ttl_s=float(config.get("response_cache_ttl_s", 900.0)),
        )

    @property
    def client(self) -> httpx.AsyncClient:
        """One client, held open.

        Connection reuse matters more than usual here: when calls are five
        seconds apart, a fresh TLS handshake each time is a visible fraction of
        the budget rather than a rounding error.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout_s,
                transport=self._transport,
                headers={"user-agent": "CeyNex/0.1 (University of Moratuwa; research project)"},
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- the two calls ---------------------------------------------------

    async def articles(
        self,
        query: str,
        *,
        timespan: str = "7d",
        max_records: int = 75,
        sort: str = "HybridRel",
        topic: str = "",
        scope: str = "",
        attempts: int = 2,
        backoff_base_s: float = 0.5,
        max_wait_s: float = DEFAULT_MAX_WAIT_S,
        keep_forensic: bool = False,
    ) -> list[NewsArticle]:
        """`mode=ArtList` — the headlines themselves."""
        params = {
            "query": _with_language(query),
            "mode": "ArtList",
            "format": "json",
            "timespan": timespan,
            "maxrecords": str(min(int(max_records), MAX_RECORDS_CEILING)),
            "sort": sort,
        }
        payload = await self._get(
            params,
            attempts=attempts,
            backoff_base_s=backoff_base_s,
            max_wait_s=max_wait_s,
            keep_forensic=keep_forensic,
        )
        records = payload.get("articles") or []
        if not isinstance(records, list):
            raise GdeltUnavailableError(f"unexpected `articles` shape: {type(records).__name__}")
        return [
            NewsArticle.from_gdelt(record, topic=topic, scope=scope)
            for record in records
            if isinstance(record, dict) and record.get("url")
        ]

    async def volume(
        self,
        query: str,
        *,
        timespan: str = "7d",
        attempts: int = 4,
        backoff_base_s: float = 2.0,
        max_wait_s: float = DEFAULT_MAX_WAIT_S,
        keep_forensic: bool = True,
    ) -> list[tuple[datetime, float]]:
        """`mode=TimelineVolRaw` — a bucketed article count, for trending.

        Not `ArtList` with a count of the records. `maxrecords` caps at 250, so
        every topic busier than that returns exactly 250 and the ranking
        degenerates to "everything is at the ceiling" — the busy topics being
        precisely the ones a trending panel exists to distinguish. This mode has
        no ceiling, returns a few hundred bytes instead of a few hundred
        kilobytes, and one `timespan=7d` call carries both the recent window and
        the baseline it is compared against.
        """
        params = {
            "query": _with_language(query),
            "mode": "TimelineVolRaw",
            "format": "json",
            "timespan": timespan,
        }
        payload = await self._get(
            params,
            attempts=attempts,
            backoff_base_s=backoff_base_s,
            max_wait_s=max_wait_s,
            keep_forensic=keep_forensic,
        )
        return _parse_timeline(payload)

    def describe(self, params: dict[str, str]) -> str:
        """The URL a call *would* use. For `refresh.py --dry-run`."""
        return str(httpx.URL(self._base_url, params=params))

    def article_params(self, query: str, *, timespan: str, max_records: int, sort: str) -> dict:
        """The ArtList params, exposed so `--dry-run` prints the real thing."""
        return {
            "query": _with_language(query),
            "mode": "ArtList",
            "format": "json",
            "timespan": timespan,
            "maxrecords": str(min(int(max_records), MAX_RECORDS_CEILING)),
            "sort": sort,
        }

    def volume_params(self, query: str, *, timespan: str) -> dict:
        return {
            "query": _with_language(query),
            "mode": "TimelineVolRaw",
            "format": "json",
            "timespan": timespan,
        }

    # --- the request -----------------------------------------------------

    async def _get(
        self,
        params: dict[str, str],
        *,
        attempts: int,
        backoff_base_s: float,
        max_wait_s: float,
        keep_forensic: bool,
    ) -> dict[str, Any]:
        cached = self._read_cache(params)
        if cached is not None:
            return cached

        raw = await self._fetch(
            params, attempts=attempts, backoff_base_s=backoff_base_s, max_wait_s=max_wait_s
        )

        if keep_forensic:
            self._write_forensic(params, raw)

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # A 200 carrying a plain-text complaint. Keep the body regardless of
            # `keep_forensic` — this is the one case worth a file every time.
            if not keep_forensic:
                self._write_forensic(params, raw)
            preview = raw[:200].decode("utf-8", errors="replace")
            raise GdeltUnavailableError(
                f"GDELT returned 200 with a body that is not JSON: {preview!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise GdeltUnavailableError(f"unexpected top-level shape: {type(payload).__name__}")

        self._write_cache(params, raw)
        return payload

    async def _fetch(
        self, params: dict[str, str], *, attempts: int, backoff_base_s: float, max_wait_s: float
    ) -> bytes:
        """Every attempt takes its own throttle slot.

        Acquiring once per logical request and then retrying inside that slot is
        the obvious shape and it is wrong: the retries fire at the backoff
        interval (2 s, then 4 s) rather than the gate's 5 s, so a run of
        transient failures turns into exactly the burst the gate exists to
        prevent. Measured on the deployed box — the refresher tripped GDELT's
        429 on six consecutive topics that way, having "respected" a 5 s limit
        the whole time.
        """
        last_error: Exception | None = None
        for attempt in range(1, max(1, attempts) + 1):
            if not await self._throttle.acquire(max_wait_s=max_wait_s):
                # Refusal, not an error: the caller has a store to fall back to
                # and would rather have stale headlines now than fresh later.
                raise GdeltUnavailableError("no GDELT throttle slot inside the budget")
            try:
                response = await self.client.get(self._base_url, params=params)
            except httpx.HTTPError as exc:
                last_error = exc
                log.warning("gdelt attempt %d/%d failed: %s", attempt, attempts, exc)
            else:
                if response.status_code == 429:
                    # Retrying into a 429 is how a courtesy limit becomes a ban.
                    # Shut the gate for everyone first — a 429 says our interval
                    # is wrong for current conditions, so the next caller through
                    # would hit it too — then let this one degrade.
                    await self._throttle.penalise(RATE_LIMIT_PENALTY_S)
                    raise GdeltUnavailableError(
                        f"GDELT rate limited us (retry-after "
                        f"{response.headers.get('retry-after', 'unset')})"
                    )
                if response.is_success:
                    return response.content
                last_error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )
                log.warning(
                    "gdelt attempt %d/%d returned HTTP %d", attempt, attempts, response.status_code
                )

            # No extra backoff sleep here: the gate above already spaces every
            # attempt by `min_interval_s`, and stacking an exponential wait on
            # top of it made a four-attempt topic take half a minute for no
            # added politeness. `backoff_base_s` survives only as the floor for
            # a caller whose throttle is a no-op (tests, `--dry-run`).
            if attempt < attempts and backoff_base_s:
                await asyncio.sleep(min(backoff_base_s, 1.0))

        raise GdeltUnavailableError(
            f"GDELT unreachable after {attempts} attempts: {last_error}"
        ) from last_error

    # --- caches ----------------------------------------------------------

    def _cache_path(self, params: dict[str, str]) -> Path:
        return self._cache_dir / f"{_params_key(params)}.json"

    def _read_cache(self, params: dict[str, str]) -> dict[str, Any] | None:
        path = self._cache_path(params)
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return None
        if age > self._cache_ttl_s:
            return None
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            # Self-healing, same as `PromptCache`: a truncated entry from a
            # killed process must not poison this query and every later one.
            path.unlink(missing_ok=True)
            return None
        return payload if isinstance(payload, dict) else None

    def _write_cache(self, params: dict[str, str], raw: bytes) -> None:
        path = self._cache_path(params)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        except OSError:  # noqa: BLE001 is not needed - OSError is specific
            log.debug("could not write the GDELT response cache at %s", path, exc_info=True)

    def _write_forensic(self, params: dict[str, str], raw: bytes) -> None:
        """Whatever GDELT sent, before anyone tried to interpret it."""
        day = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        path = self._forensic_dir / day / f"{_params_key(params)[:16]}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        except OSError:
            log.debug("could not write the GDELT forensic copy at %s", path, exc_info=True)


# --- parsing --------------------------------------------------------------


def _with_language(query: str) -> str:
    """Append the language clause unless the caller already set one."""
    text = query.strip()
    if "sourcelang:" in text.lower():
        return text
    return f"{text} {LANGUAGE_CLAUSE}"


def _params_key(params: dict[str, str]) -> str:
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:32]


def _parse_timeline(payload: dict[str, Any]) -> list[tuple[datetime, float]]:
    """`{"timeline": [{"series": ..., "data": [{"date": ..., "value": N}]}]}`.

    Tolerant of an empty timeline: a watchlist topic with no coverage in the
    window is an ordinary answer, not a malformed response.
    """
    timeline = payload.get("timeline") or []
    if not isinstance(timeline, list) or not timeline:
        return []
    series = timeline[0]
    if not isinstance(series, dict):
        return []

    points: list[tuple[datetime, float]] = []
    for entry in series.get("data") or []:
        if not isinstance(entry, dict):
            continue
        when = _parse_timeline_date(entry.get("date"))
        if when is None:
            continue
        try:
            points.append((when, float(entry.get("value") or 0.0)))
        except (TypeError, ValueError):
            continue
    return sorted(points)


def _parse_timeline_date(raw: Any) -> datetime | None:
    """Timeline dates use the same `seendate` stamp, occasionally date-only."""
    if not isinstance(raw, str):
        return None
    parsed = parse_seendate(raw)
    if parsed is not None:
        return parsed
    for pattern in ("%Y-%m-%dT%H:%M:%SZ", "%Y%m%d%H%M%S", "%Y%m%d"):
        try:
            return datetime.strptime(raw.strip(), pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


__all__ = [
    "DEFAULT_MAX_WAIT_S",
    "GDELT_DOC_URL",
    "LANGUAGE_CLAUSE",
    "MAX_RECORDS_CEILING",
    "GdeltClient",
    "GdeltUnavailableError",
]
