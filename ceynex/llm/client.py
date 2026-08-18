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
from typing import Any

from ceynex.settings import llm_config, openai_api_key

log = logging.getLogger(__name__)


@dataclass
class LLMUsage:
    """What the client did, so the orchestrator can report it and we can cost it."""

    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    elapsed_s: float = 0.0

    def merge(self, other: LLMUsage) -> None:
        self.calls += other.calls
        self.cache_hits += other.cache_hits
        self.failures += other.failures
        self.elapsed_s += other.elapsed_s


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
        if not self.enabled:
            return None
        entry = self.path / f"{key}.json"
        if not entry.is_file():
            return None
        if time.time() - entry.stat().st_mtime > self.ttl_s:
            entry.unlink(missing_ok=True)
            return None
        try:
            return json.loads(entry.read_text(encoding="utf-8"))["response"]
        except (json.JSONDecodeError, KeyError):
            entry.unlink(missing_ok=True)
            return None

    def put(self, key: str, response: str) -> None:
        if not self.enabled:
            return
        entry = self.path / f"{key}.json"
        entry.write_text(
            json.dumps({"response": response, "cached_at": time.time()}),
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
    usage: LLMUsage = field(default_factory=LLMUsage)
    _cache: PromptCache = field(init=False)
    _client: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        cache_cfg = self.config.get("cache", {})
        self._cache = PromptCache(
            path=Path(cache_cfg.get("path", ".cache/llm")),
            ttl_hours=float(cache_cfg.get("ttl_hours", 168)),
            enabled=bool(cache_cfg.get("enabled", True)),
        )

    # --- availability ----------------------------------------------------

    @property
    def available(self) -> bool:
        """False means every answer this run is degraded, and that is fine."""
        return bool(self.api_key)

    def _model(self, role: str) -> dict[str, Any]:
        """Config for a role: `router`, `merge`, or `explanation`."""
        models = self.config.get("models", {})
        if role not in models:
            raise KeyError(f"config/llm.yaml has no models.{role}")
        return models[role]

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
    ) -> str | None:
        """Return the model's text, or None if it could not be obtained.

        Never raises. A caller that has to wrap this in try/except has been given
        the wrong interface — degrading is the normal path, not the exceptional
        one.
        """
        model_cfg = self._model(role)
        model = model_cfg["model"]
        temperature = float(model_cfg.get("temperature", 0.2))
        max_tokens = int(model_cfg.get("max_tokens", 600))

        cache_key = self._cache.key(model, system, user, temperature)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.usage.cache_hits += 1
            return cached

        if not self.available:
            log.info("no OPENAI_API_KEY — degrading (SRS 3.4.3)")
            self.usage.failures += 1
            return None

        limits = self._limits()
        timeout_s = float(limits.get("request_timeout_s", 8.0))
        attempts = int(limits.get("max_retries", 1)) + 1

        started = time.perf_counter()
        for attempt in range(1, attempts + 1):
            try:
                text = await asyncio.wait_for(
                    self._call(model, system, user, temperature, max_tokens, json_mode),
                    timeout=timeout_s,
                )
            except TimeoutError:
                log.warning("llm timed out after %.1fs (attempt %d/%d)", timeout_s, attempt, attempts)
            except Exception as exc:  # noqa: BLE001 - degrading is the contract
                log.warning("llm call failed (attempt %d/%d): %s", attempt, attempts, exc)
            else:
                self.usage.calls += 1
                self.usage.elapsed_s += time.perf_counter() - started
                if text:
                    self._cache.put(cache_key, text)
                return text

        self.usage.failures += 1
        self.usage.elapsed_s += time.perf_counter() - started
        log.warning("llm unavailable after %d attempts — degrading", attempts)
        return None

    async def _call(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str | None:
        """The provider-specific part. Swapping vendors means changing this method."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key)

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

        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

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

    def __init__(self, response: str | None = "A canned explanation.", available: bool = True):
        self._response = response
        self.available = available
        self.usage = LLMUsage()
        self.calls: list[tuple[str, str, str]] = []

    async def generate(self, role: str, system: str, user: str, *, json_mode: bool = False) -> str | None:
        self.calls.append((role, system, user))
        if not self.available:
            self.usage.failures += 1
            return None
        self.usage.calls += 1
        return self._response

    async def generate_explanation(self, context: dict[str, Any]) -> str:
        text = await self.generate("explanation", EXPLANATION_SYSTEM, json.dumps(context, default=str))
        return text or ""
