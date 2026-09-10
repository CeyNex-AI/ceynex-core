"""Implements SRS 3.4.3 and 3.6.5 — the reasoning client and its degraded path.

The LLM is a *purchased component* (SRS 3.6.5) and the SRS treats it as vendor
neutral: model ids live in `config/llm.yaml`, never in code, so cost can be
traded against quality without a rebuild. It is also the system's single largest
availability risk, which is why the degraded path is built here on day one rather
than bolted on before the demo.

**Degrading is required behaviour, not failure.** When the provider is
unreachable, the key is missing, or the call times out, `generate()` returns
`None` and the caller sets `degraded=True` and answers with raw knowledge-graph
and forecast figures. An answer with numbers and evidence but no prose is a
conforming answer. An exception reaching the orchestrator is not.

Three things keep the response-time budget (SRS 3.4.1: 10s single-sector, 20s
cross-sector) reachable:

- a **prompt-hash cache**, so a repeated question costs nothing and the demo is
  fast and repeatable;
- a **timeout below the budget** (8s per `config/llm.yaml`), because a call that
  overruns has already lost — degrading at 8s beats answering at 30;
- **one retry, then degrade.** Retrying harder spends the budget on hope.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ceynex.observability import context as obs_context
from ceynex.observability import spend as spend_limits
from ceynex.observability import trace
from ceynex.settings import llm_config, openai_api_key, openrouter_api_key

log = logging.getLogger(__name__)


@dataclass
class LLMUsage:
    """What the client did, so the orchestrator can report it and we can cost it.

    Process-lifetime, one instance per `LLMReasoningClient`, and `_cap_reached()`
    reads `cost_usd` off it — so this must stay a running total and must not be
    repurposed per request. Per-request accounting lives in
    `ceynex/observability/context.py::RequestLLMUsage`, accumulated alongside.

    `tokens_in`/`tokens_out` were added when the usage ledger landed: `_cost()`
    had been reading them off every provider response and discarding them, so the
    system was paying to learn a number it then threw away.
    """

    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    fallback_calls: int = 0
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def merge(self, other: LLMUsage) -> None:
        self.calls += other.calls
        self.cache_hits += other.cache_hits
        self.failures += other.failures
        self.fallback_calls += other.fallback_calls
        self.elapsed_s += other.elapsed_s
        self.cost_usd += other.cost_usd
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out


class TextStream(Protocol):
    """Somewhere to send a response's text as it arrives (D12, amended).

    `feed` receives each piece of text in order; `restart` says the attempt that
    produced the pieces so far has been abandoned — a timeout before a retry, or
    the primary before the failsafe — so whatever was shown from it must go.
    `orchestrator/answer_stream.py::SentenceGate` is the one implementation, and
    it is what stands between these pieces and the screen.
    """

    def feed(self, chunk: str) -> None: ...

    def restart(self) -> None: ...


@dataclass(frozen=True)
class _CallOutcome:
    """One provider round trip. Internal to this module."""

    text: str | None
    cost_usd: float
    tokens_in: int
    tokens_out: int


def _observe(
    *,
    role: str,
    model: str,
    provider: str,
    elapsed_ms: float,
    outcome: _CallOutcome | None = None,
    cache_hit: bool = False,
    fallback: bool = False,
    failed: bool = False,
    original: dict[str, Any] | None = None,
) -> None:
    """Report one call to the live trace and the per-request usage accumulator.

    Both are ambient and both no-op outside a request, which is what keeps
    `demo.py`, `eval/harness.py` and the existing test suite unaffected. This is
    additive to `self.usage`, never a replacement: that counter is what
    `_cap_reached()` reads, and moving it per-request would silently disable the
    R5 spend cap.
    """
    tokens_in = outcome.tokens_in if outcome else 0
    tokens_out = outcome.tokens_out if outcome else 0
    cost_usd = outcome.cost_usd if outcome else 0.0

    obs_context.record_llm_call(
        obs_context.LLMCall(
            role=role,
            model=model,
            provider=provider,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            elapsed_ms=round(elapsed_ms, 1),
            cache_hit=cache_hit,
            fallback=fallback,
            failed=failed,
        )
    )

    payload: dict[str, Any] = {
        "role": role,
        "model": model,
        "provider": provider,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost_usd, 6),
        "elapsed_ms": round(elapsed_ms, 1),
        "cache_hit": cache_hit,
        "fallback": fallback,
        "status": "failed" if failed else "ok",
    }
    if original:
        # Only present when the cache entry predates nothing — i.e. was written
        # by a build that stored it. Absent for up to the 168h TTL after deploy.
        payload["original_tokens_in"] = original.get("tokens_in", 0)
        payload["original_tokens_out"] = original.get("tokens_out", 0)
    trace.emit("llm_call", **payload)


@dataclass
class ProviderStatus:
    """One provider's health, for the admin dashboard (SRS 3.5.4).

    Derived from the outcome of the most recent real `generate()` attempt,
    not a live ping — free (no extra API spend just to check), but only as
    fresh as the last actual query that reached this provider. An admin
    reading "openai: down" alongside "openrouter: ok" is the point: GPT-4o
    needs attention, and the free failsafe is covering in the meantime.
    """

    configured: bool
    status: str  # "not_configured" | "cap_reached" | "unknown" | "ok" | "down"
    last_error: str | None = None
    last_checked_at: float | None = None  # unix epoch seconds; None if never attempted


class PromptCache:
    """Content-addressed cache on disk, keyed by a hash of the exact request.

    On disk rather than in memory because the demo is run more than once, and
    because two processes (the API and a CLI invocation) should share it.
    """

    def __init__(self, path: Path, ttl_hours: float = 168.0, enabled: bool = True) -> None:
        self.path = path
        self.ttl_s = ttl_hours * 3600
        self.enabled = enabled
        if self.enabled:
            self.path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(model: str, system: str, user: str, temperature: float) -> str:
        payload = json.dumps(
            {"model": model, "system": system, "user": user, "temperature": temperature},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> str | None:
        entry = self._read(key)
        return None if entry is None else entry.get("response")

    def get_meta(self, key: str) -> dict[str, Any]:
        """What the original, uncached call cost — for display only.

        A cache hit costs nothing, so the ledger correctly records zero. This is
        the separate, informational number: "reused a cached response (originally
        ~340 tokens)".

        Returns `{}` for an entry written before this metadata existed. That is
        not a hypothetical: the TTL is 168 hours, so for a full week after this
        ships the new `get_meta()` is reading entries the old `put()` wrote. Every
        field is fetched with `.get()` for exactly that reason.
        """
        entry = self._read(key)
        if entry is None:
            return {}
        return {
            "tokens_in": entry.get("tokens_in", 0),
            "tokens_out": entry.get("tokens_out", 0),
            "cost_usd": entry.get("cost_usd", 0.0),
            "model": entry.get("model"),
        }

    def _read(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        entry = self.path / f"{key}.json"
        if not entry.is_file():
            return None
        if time.time() - entry.stat().st_mtime > self.ttl_s:
            entry.unlink(missing_ok=True)
            return None
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
            if "response" not in payload:
                raise KeyError("response")
        except (json.JSONDecodeError, KeyError, TypeError):
            entry.unlink(missing_ok=True)
            return None
        return payload

    def put(
        self,
        key: str,
        response: str,
        *,
        model: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        entry = self.path / f"{key}.json"
        entry.write_text(
            json.dumps(
                {
                    "response": response,
                    "cached_at": time.time(),
                    "model": model,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cost_usd": cost_usd,
                }
            ),
            encoding="utf-8",
        )


@dataclass
class LLMReasoningClient:
    """Structured context in, natural-language explanation out — or None.

    Satisfies `LLMReasoningClientProtocol` from the contracts. Every agent and the
    orchestrator's merger share one instance.
    """

    config: dict[str, Any] = field(default_factory=llm_config)
    api_key: str | None = field(default_factory=openai_api_key)
    fallback_api_key: str | None = field(default_factory=openrouter_api_key)
    usage: LLMUsage = field(default_factory=LLMUsage)
    #: Today's spend, deployment-wide and per reader — what the daily limits are
    #: enforced against (D16). Shared through Redis by both uvicorn workers when
    #: `REDIS_URL` is set; this worker's own count otherwise.
    spend: spend_limits.SpendCounter = field(default_factory=spend_limits.shared_counter)
    _cache: PromptCache = field(init=False)
    #: The deployment-wide figure the last check saw, for the synchronous
    #: `provider_status()` — which must not make a network call to report.
    _last_total_spend: float = field(init=False, default=0.0)
    _client: Any = field(init=False, default=None)
    _fallback_client: Any = field(init=False, default=None)
    # Last-attempt outcome per provider, for provider_status() below. None
    # means "never attempted since this process started", distinct from a
    # real success or failure.
    _primary_last_ok: bool | None = field(init=False, default=None)
    _primary_last_error: str | None = field(init=False, default=None)
    _primary_last_checked_at: float | None = field(init=False, default=None)
    _fallback_last_ok: bool | None = field(init=False, default=None)
    _fallback_last_error: str | None = field(init=False, default=None)
    _fallback_last_checked_at: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        cache_cfg = self.config.get("cache", {})
        self._cache = PromptCache(
            path=Path(cache_cfg.get("path", ".cache/llm")),
            ttl_hours=float(cache_cfg.get("ttl_hours", 168)),
            enabled=bool(cache_cfg.get("enabled", True)),
        )

    # --- availability ----------------------------------------------------

    @property
    def _fallback_enabled(self) -> bool:
        fb = self.config.get("fallback") or {}
        return bool(fb.get("enabled")) and bool(self.fallback_api_key)

    @property
    def available(self) -> bool:
        """False means every answer this run is degraded, and that is fine.

        True if either the primary or the failsafe provider (R5) could plausibly
        answer — callers (health check, router mode) care whether prose is
        possible this run, not which provider supplies it.
        """
        return bool(self.api_key) or self._fallback_enabled

    def _cap_reached(self) -> bool:
        """Whether the deployment's daily cap was spent, as of the last check."""
        cap = float(self._limits().get("daily_spend_cap_usd", 0) or 0)
        return bool(cap and self._last_total_spend >= cap)

    async def _over_budget(self, user_email: str | None) -> tuple[str, float, float] | None:
        """`(scope, cap, spent)` when a daily limit is spent, else None.

        `scope` is "global" — the deployment's `daily_spend_cap_usd` — or "user",
        this reader's `per_user_daily_cap_usd`. Read from the shared counter, so
        the cap is one number across both workers and resets at 00:00 UTC (D16).
        A call with no reader (the eval harness, the demo CLI) meets only the
        global cap.
        """
        limits = self._limits()
        cap = float(limits.get("daily_spend_cap_usd", 0) or 0)
        per_user = float(limits.get("per_user_daily_cap_usd", 0) or 0)
        if not cap and not per_user:
            return None
        spent = await self.spend.spent(user_email)
        self._last_total_spend = spent.total_usd
        if cap and spent.total_usd >= cap:
            return "global", cap, spent.total_usd
        if per_user and user_email and spent.user_usd >= per_user:
            return "user", per_user, spent.user_usd
        return None

    def provider_status(self) -> dict[str, ProviderStatus]:
        """Per-provider health for the admin dashboard (SRS 3.5.4).

        Keyed "openai"/"openrouter" rather than "primary"/"fallback" -- the
        provider names are what an admin recognises and needs to go fix,
        not this client's internal role labels.
        """
        cap_reached = self._cap_reached()
        return {
            "openai": self._describe(
                configured=bool(self.api_key),
                last_ok=self._primary_last_ok,
                last_error=self._primary_last_error,
                last_checked_at=self._primary_last_checked_at,
                cap_reached=cap_reached,
            ),
            "openrouter": self._describe(
                configured=self._fallback_enabled,
                last_ok=self._fallback_last_ok,
                last_error=self._fallback_last_error,
                last_checked_at=self._fallback_last_checked_at,
            ),
        }

    @staticmethod
    def _describe(
        *,
        configured: bool,
        last_ok: bool | None,
        last_error: str | None,
        last_checked_at: float | None,
        cap_reached: bool = False,
    ) -> ProviderStatus:
        if not configured:
            return ProviderStatus(configured=False, status="not_configured")
        if cap_reached:
            # Calls are being deliberately skipped to protect the budget
            # (R5) -- distinct from a real failure: nothing here says GPT-4o
            # itself is broken, only that today's spend cap is spent.
            return ProviderStatus(configured=True, status="cap_reached")
        if last_ok is None:
            return ProviderStatus(configured=True, status="unknown")
        if last_ok:
            return ProviderStatus(configured=True, status="ok", last_checked_at=last_checked_at)
        return ProviderStatus(
            configured=True, status="down", last_error=last_error, last_checked_at=last_checked_at
        )

    def _model(self, role: str) -> dict[str, Any]:
        """Config for a role: `router`, `merge`, or `explanation`."""
        models = self.config.get("models", {})
        if role not in models:
            raise KeyError(f"config/llm.yaml has no models.{role}")
        return models[role]

    def _fallback_model(self, role: str) -> dict[str, Any] | None:
        """Failsafe provider config for a role, or None if it isn't usable.

        `models` is intentionally per-role and optional — a role with no entry
        here just has no failsafe and degrades normally once the primary fails.
        """
        if not self._fallback_enabled:
            return None
        fb = self.config.get("fallback") or {}
        models = fb.get("models", {})
        if role not in models:
            return None
        return {
            "model": models[role],
            "base_url": fb.get("base_url"),
            "timeout_s": float(fb.get("timeout_s", 5.0)),
        }

    def _limits(self) -> dict[str, Any]:
        return self.config.get("limits", {})

    # --- the one call ----------------------------------------------------

    async def generate(
        self,
        role: str,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        stream: TextStream | None = None,
    ) -> str | None:
        """Return the model's text, or None if it could not be obtained.

        Never raises. A caller that has to wrap this in try/except has been given
        the wrong interface — degrading is the normal path, not the exceptional
        one.

        `stream` receives the text as it arrives, and is only ever passed when
        someone is watching (`trace.active()`): with no stream this is exactly
        the call it always was, so `POST /api/query` and `make eval` measure the
        same path they always did. The returned text is identical either way —
        the pieces fed to `stream` join to it.
        """
        model_cfg = self._model(role)
        model = model_cfg["model"]
        temperature = float(model_cfg.get("temperature", 0.2))
        max_tokens = int(model_cfg.get("max_tokens", 600))
        if json_mode:
            stream = None  # structured output is parsed whole, never shown

        cache_key = self._cache.key(model, system, user, temperature)
        # Regenerate asks for a fresh answer from the same findings; serving the
        # cached one would hand back the answer it is replacing. Only the read
        # is skipped — the new response is still cached, and a regenerated
        # answer is what an identical request gets next.
        observation = obs_context.current()
        bypass = observation is not None and role in observation.bypass_cache_roles
        cached = None if bypass else self._cache.get(cache_key)
        if cached is not None:
            if stream is not None:
                # It arrives all at once, so it is shown all at once.
                stream.feed(cached)
            self.usage.cache_hits += 1
            # 0 tokens and $0 is the correct accounting, not a gap: no API call
            # happened. `original_*` is the separate, informational number, and
            # is absent for entries written before the cache stored it.
            original = self._cache.get_meta(cache_key)
            _observe(
                role=role,
                model=model,
                provider="cache",
                cache_hit=True,
                elapsed_ms=0.0,
                original=original,
            )
            return cached

        fallback = self._fallback_model(role)

        if not self.available and fallback is None:
            log.info("no OPENAI_API_KEY and no usable failsafe — degrading (SRS 3.4.3)")
            self.usage.failures += 1
            _observe(role=role, model=model, provider="none", failed=True, elapsed_ms=0.0)
            return None

        limits = self._limits()
        timeout_s = float(limits.get("request_timeout_s", 8.0))
        attempts = int(limits.get("max_retries", 1)) + 1
        user_email = observation.user_email if observation is not None else None
        over = await self._over_budget(user_email) if self.api_key else None
        cap_reached = over is not None
        if over is not None:
            scope, limit, spent = over
            log.warning(
                "%s daily spend limit ($%.2f, spent $%.2f) reached — %s (SRS 3.4.3, R5)",
                "the deployment's" if scope == "global" else "this reader's",
                limit,
                spent,
                "trying the free failsafe" if fallback else "degrading",
            )
            # Said in the trace, not only the log: a reader whose answer arrives
            # without prose should be able to see why.
            trace.emit("budget", scope=scope, cap_usd=limit, spent_usd=round(spent, 6),
                       resets_at=spend_limits.resets_at().isoformat())

        started = time.perf_counter()

        if self.api_key and not cap_reached:
            primary_error: str | None = None
            for attempt in range(1, attempts + 1):
                if stream is not None:
                    stream.restart()
                try:
                    outcome = await asyncio.wait_for(
                        self._call(model, system, user, temperature, max_tokens, json_mode,
                                   model_cfg, stream=stream),
                        timeout=timeout_s,
                    )
                except TimeoutError:
                    primary_error = f"timed out after {timeout_s:.1f}s"
                    log.warning("llm timed out after %.1fs (attempt %d/%d)", timeout_s, attempt, attempts)
                except Exception as exc:  # noqa: BLE001 - degrading is the contract
                    primary_error = str(exc)
                    log.warning("llm call failed (attempt %d/%d): %s", attempt, attempts, exc)
                else:
                    elapsed_s = time.perf_counter() - started
                    self.usage.calls += 1
                    self.usage.elapsed_s += elapsed_s
                    self.usage.cost_usd += outcome.cost_usd
                    self.usage.tokens_in += outcome.tokens_in
                    self.usage.tokens_out += outcome.tokens_out
                    # Counted against today's limits the moment it is spent, so
                    # the next check — on either worker — already sees it.
                    await self.spend.add(outcome.cost_usd, user_email)
                    self._primary_last_ok = True
                    self._primary_last_error = None
                    self._primary_last_checked_at = time.time()
                    _observe(
                        role=role,
                        model=model,
                        provider="openai",
                        outcome=outcome,
                        elapsed_ms=elapsed_s * 1000,
                    )
                    if outcome.text:
                        self._cache.put(
                            cache_key,
                            outcome.text,
                            model=model,
                            tokens_in=outcome.tokens_in,
                            tokens_out=outcome.tokens_out,
                            cost_usd=outcome.cost_usd,
                        )
                    return outcome.text
            # Every attempt failed -- record once per generate() call, not
            # per retry, so provider_status() reflects "is the primary
            # working right now" rather than flapping on individual retries.
            self._primary_last_ok = False
            self._primary_last_error = primary_error
            self._primary_last_checked_at = time.time()

        if fallback is not None:
            fallback_cfg = {"model": fallback["model"]}  # no cost fields — free tier, costs 0
            if stream is not None:
                stream.restart()
            try:
                outcome = await asyncio.wait_for(
                    self._call(
                        fallback["model"],
                        system,
                        user,
                        temperature,
                        max_tokens,
                        json_mode,
                        fallback_cfg,
                        base_url=fallback["base_url"],
                        api_key=self.fallback_api_key,
                        stream=stream,
                    ),
                    timeout=fallback["timeout_s"],
                )
            except TimeoutError:
                self._fallback_last_ok = False
                self._fallback_last_error = f"timed out after {fallback['timeout_s']:.1f}s"
                self._fallback_last_checked_at = time.time()
                log.warning("failsafe llm timed out after %.1fs — degrading", fallback["timeout_s"])
            except Exception as exc:  # noqa: BLE001 - degrading is the contract
                self._fallback_last_ok = False
                self._fallback_last_error = str(exc)
                self._fallback_last_checked_at = time.time()
                log.warning("failsafe llm call failed: %s — degrading", exc)
            else:
                elapsed_s = time.perf_counter() - started
                self.usage.calls += 1
                self.usage.fallback_calls += 1
                self.usage.elapsed_s += elapsed_s
                self.usage.cost_usd += outcome.cost_usd
                self.usage.tokens_in += outcome.tokens_in
                self.usage.tokens_out += outcome.tokens_out
                self._fallback_last_ok = True
                self._fallback_last_error = None
                self._fallback_last_checked_at = time.time()
                _observe(
                    role=role,
                    model=fallback["model"],
                    provider="openrouter",
                    outcome=outcome,
                    fallback=True,
                    elapsed_ms=elapsed_s * 1000,
                )
                if outcome.text:
                    self._cache.put(
                        cache_key,
                        outcome.text,
                        model=fallback["model"],
                        tokens_in=outcome.tokens_in,
                        tokens_out=outcome.tokens_out,
                        cost_usd=outcome.cost_usd,
                    )
                return outcome.text

        elapsed_s = time.perf_counter() - started
        self.usage.failures += 1
        self.usage.elapsed_s += elapsed_s
        _observe(
            role=role, model=model, provider="none", failed=True, elapsed_ms=elapsed_s * 1000
        )
        log.warning("llm unavailable after primary and failsafe — degrading")
        return None

    def _client_for(self, base_url: str | None, api_key: str | None) -> Any:
        """The primary and failsafe providers each get one lazily-built, cached
        client — OpenRouter speaks the OpenAI API, so only base_url/api_key differ.
        """
        from openai import AsyncOpenAI

        if base_url is None:
            if self._client is None:
                self._client = AsyncOpenAI(api_key=self.api_key)
            return self._client
        if self._fallback_client is None:
            self._fallback_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        return self._fallback_client

    async def _call(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        model_cfg: dict[str, Any],
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        stream: TextStream | None = None,
    ) -> _CallOutcome:
        """The provider-specific part. Swapping vendors means changing this method.

        Returns the text, what it cost and the token counts, so `generate()` can
        charge it against `daily_spend_cap_usd` (R5) and record it in the usage
        ledger without knowing anything provider-specific. `base_url`/`api_key`
        select the failsafe provider (R5); omitted, this calls the primary.
        """
        client = self._client_for(base_url, api_key)

        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if stream is not None:
            return await self._call_streaming(client, kwargs, model_cfg, stream)

        response = await client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        return _CallOutcome(
            text=text,
            cost_usd=self._cost(model_cfg, usage),
            tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    async def _call_streaming(
        self, client: Any, kwargs: dict[str, Any], model_cfg: dict[str, Any], stream: TextStream
    ) -> _CallOutcome:
        """The same call, delivered in pieces.

        `include_usage` asks for a final chunk carrying the token counts — the
        chunk has no choices, only usage — so a streamed call is costed exactly
        like a whole one (verified against the provider 2026-09-10). A provider
        that omits it is costed as free, the same posture `_cost` takes with a
        missing `usage` on a whole response.

        The response is always closed, even on a timeout: an unclosed stream
        holds its connection until the garbage collector finds it.
        """
        response = await client.chat.completions.create(
            **kwargs, stream=True, stream_options={"include_usage": True}
        )
        parts: list[str] = []
        usage = None
        try:
            async for chunk in response:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if chunk.choices:
                    piece = chunk.choices[0].delta.content
                    if piece:
                        parts.append(piece)
                        stream.feed(piece)
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                await close()
        return _CallOutcome(
            text="".join(parts) or None,
            cost_usd=self._cost(model_cfg, usage),
            tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    @staticmethod
    def _cost(model_cfg: dict[str, Any], usage: Any) -> float:
        """Dollar cost of one call, from config rates and the provider's token count.

        `usage` can be None (some providers omit it, or a mock in tests) — a call
        we can't cost is treated as free rather than raising, same "degrade, don't
        break" posture as the rest of this client.
        """
        if usage is None:
            return 0.0
        input_rate = float(model_cfg.get("cost_per_1k_input_tokens", 0.0))
        output_rate = float(model_cfg.get("cost_per_1k_output_tokens", 0.0))
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        return (prompt_tokens / 1000) * input_rate + (completion_tokens / 1000) * output_rate

    # --- protocol surface ------------------------------------------------

    async def generate_explanation(self, context: dict[str, Any]) -> str:
        """`LLMReasoningClientProtocol`. Returns "" when degraded, never raises.

        The empty string is the signal to the agent that it is in degraded mode:
        emit figures and evidence, skip the prose (SRS 3.4.3).
        """
        system = context.pop("system", EXPLANATION_SYSTEM)
        text = await self.generate(
            "explanation",
            system=system,
            user=json.dumps(context, default=str, indent=2),
        )
        return text or ""


EXPLANATION_SYSTEM = """You explain export-trade figures to Sri Lankan policymakers and exporters.

Rules, in order of importance:
1. Every number you state must appear in the context you were given. Never
   estimate, extrapolate, or supply a figure from your own knowledge. If the
   context lacks something, say it is not available.
2. Two to four sentences. Plain English. No preamble, no bullet points.
3. No hedging padding ("it is worth noting that"). Say the thing.
4. You are describing data, not giving financial or policy advice."""


class FakeLLMClient:
    """Deterministic stand-in for tests. No network, ever.

    Tests that need the degraded path construct this with `available=False`;
    tests that need prose construct it with a canned response.
    """

    def __init__(
        self,
        response: str | None = "A canned explanation.",
        available: bool = True,
        chunk_size: int = 7,
    ):
        self._response = response
        self.available = available
        self.usage = LLMUsage()
        self.calls: list[tuple[str, str, str]] = []
        #: Whether each call was given a stream — so a test can assert that the
        #: non-streaming path stayed non-streaming.
        self.streamed: list[bool] = []
        #: Streamed responses arrive in pieces this long, so a test exercises
        #: boundaries that fall inside words and figures, as real ones do.
        self.chunk_size = chunk_size

    async def generate(
        self,
        role: str,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        stream: TextStream | None = None,
    ) -> str | None:
        self.calls.append((role, system, user))
        self.streamed.append(stream is not None)
        if not self.available:
            self.usage.failures += 1
            return None
        self.usage.calls += 1
        if stream is not None and self._response:
            stream.restart()
            for start in range(0, len(self._response), self.chunk_size):
                stream.feed(self._response[start:start + self.chunk_size])
        return self._response

    async def generate_explanation(self, context: dict[str, Any]) -> str:
        text = await self.generate("explanation", EXPLANATION_SYSTEM, json.dumps(context, default=str))
        return text or ""

    def provider_status(self) -> dict[str, ProviderStatus]:
        """Never real network, so "ok"/"down" from a fake response would be a
        fabricated status -- "unknown" (configured but never really checked)
        is the honest answer regardless of `available`.
        """
        status = ProviderStatus(configured=self.available, status="unknown" if self.available else "not_configured")
        return {"openai": status, "openrouter": ProviderStatus(configured=False, status="not_configured")}
