"""Assertions for SRS 3.4.3 — degrading is the required behaviour, not an error.

Nothing here touches the network. The one thing these tests exist to guarantee is
that no code path in the reasoning client can raise into an agent.
"""

import json

import pytest

from ceynex.contracts import LLMReasoningClientProtocol
from ceynex.llm import FakeLLMClient, LLMReasoningClient, PromptCache

CONFIG = {
    "provider": "openai",
    "models": {
        "router": {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 300,
            "cost_per_1k_input_tokens": 0.00015,
            "cost_per_1k_output_tokens": 0.0006,
        },
        "merge": {
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 1200,
            "cost_per_1k_input_tokens": 0.0025,
            "cost_per_1k_output_tokens": 0.01,
        },
        "explanation": {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 600,
            "cost_per_1k_input_tokens": 0.00015,
            "cost_per_1k_output_tokens": 0.0006,
        },
    },
    "limits": {"request_timeout_s": 8.0, "max_retries": 1, "daily_spend_cap_usd": 5.0},
    "cache": {"enabled": False, "path": ".cache/llm-test", "ttl_hours": 1},
}


FALLBACK_CONFIG = json.loads(json.dumps(CONFIG))
FALLBACK_CONFIG["fallback"] = {
    "enabled": True,
    "base_url": "https://openrouter.ai/api/v1",
    "timeout_s": 5.0,
    # openrouter/free is OpenRouter's own Free Models Router — it picks from
    # the live free-tier catalog itself, so there's no specific free model id
    # to pin (or go stale) here.
    "models": {
        "router": "openrouter/free",
        "merge": "openrouter/free",
        "explanation": "openrouter/free",
    },
}


def client(tmp_path, *, api_key=None, cache=False, fallback_api_key=None, config=None):
    config = json.loads(json.dumps(config if config is not None else CONFIG))
    config["cache"] = {"enabled": cache, "path": str(tmp_path / "cache"), "ttl_hours": 1}
    return LLMReasoningClient(config=config, api_key=api_key, fallback_api_key=fallback_api_key)


# --- degrading -----------------------------------------------------------


async def test_missing_api_key_degrades_instead_of_raising(tmp_path):
    """The single most likely production state, and it must not be an exception."""
    llm = client(tmp_path, api_key=None)
    assert llm.available is False
    assert await llm.generate("explanation", "sys", "user") is None
    assert llm.usage.failures == 1


async def test_generate_explanation_returns_empty_string_when_degraded(tmp_path):
    """Agents branch on falsiness, so the protocol never hands back None here."""
    llm = client(tmp_path, api_key=None)
    assert await llm.generate_explanation({"figures": {"cagr": 0.07}}) == ""


async def test_a_provider_error_degrades_rather_than_propagating(tmp_path, monkeypatch):
    llm = client(tmp_path, api_key="sk-test")

    async def boom(*args, **kwargs):
        raise RuntimeError("provider is on fire")

    monkeypatch.setattr(llm, "_call", boom)
    assert await llm.generate("explanation", "sys", "user") is None
    assert llm.usage.failures == 1


async def test_a_timeout_degrades_within_the_budget(tmp_path, monkeypatch):
    """SRS 3.4.1 gives 10s for a single-sector answer; the call gets 8 of them."""
    import asyncio

    llm = client(tmp_path, api_key="sk-test")
    llm.config["limits"] = {"request_timeout_s": 0.05, "max_retries": 0}

    async def hang(*args, **kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(llm, "_call", hang)
    assert await llm.generate("explanation", "sys", "user") is None


async def test_it_retries_once_then_stops(tmp_path, monkeypatch):
    """Retrying harder spends the response-time budget on hope."""
    llm = client(tmp_path, api_key="sk-test")
    attempts = []

    async def failing(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("nope")

    monkeypatch.setattr(llm, "_call", failing)
    await llm.generate("explanation", "sys", "user")
    assert len(attempts) == 2, "max_retries: 1 means two attempts total"


# --- failsafe provider (R5) -----------------------------------------------


async def test_failsafe_is_tried_after_the_primary_is_exhausted(tmp_path, monkeypatch):
    llm = client(tmp_path, api_key="sk-test", fallback_api_key="or-test", config=FALLBACK_CONFIG)

    async def primary_or_fallback(*args, base_url=None, **kwargs):
        if base_url is None:
            raise RuntimeError("openai is down")
        return "failsafe prose", 0.0

    monkeypatch.setattr(llm, "_call", primary_or_fallback)
    assert await llm.generate("explanation", "sys", "user") == "failsafe prose"
    assert llm.usage.calls == 1
    assert llm.usage.fallback_calls == 1


async def test_failsafe_is_used_when_there_is_no_primary_key(tmp_path, monkeypatch):
    """No OPENAI_API_KEY must not skip straight to degrading if the free
    failsafe can answer instead."""
    llm = client(tmp_path, api_key=None, fallback_api_key="or-test", config=FALLBACK_CONFIG)
    assert llm.available is True

    calls = []

    async def fallback_only(*args, base_url=None, **kwargs):
        calls.append(base_url)
        return "failsafe prose", 0.0

    monkeypatch.setattr(llm, "_call", fallback_only)
    assert await llm.generate("explanation", "sys", "user") == "failsafe prose"
    assert calls == ["https://openrouter.ai/api/v1"], "primary must never be called without a key"


async def test_failsafe_is_tried_when_the_spend_cap_is_reached(tmp_path, monkeypatch):
    """R5: the cap stops paid calls, not the free failsafe."""
    llm = client(tmp_path, api_key="sk-test", fallback_api_key="or-test", config=FALLBACK_CONFIG)
    llm.usage.cost_usd = 5.0  # == daily_spend_cap_usd

    async def fallback_only(*args, base_url=None, **kwargs):
        assert base_url is not None, "primary must not be called once the cap is reached"
        return "failsafe prose", 0.0

    monkeypatch.setattr(llm, "_call", fallback_only)
    assert await llm.generate("explanation", "sys", "user") == "failsafe prose"


async def test_it_degrades_when_both_primary_and_failsafe_fail(tmp_path, monkeypatch):
    llm = client(tmp_path, api_key="sk-test", fallback_api_key="or-test", config=FALLBACK_CONFIG)

    async def always_fails(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(llm, "_call", always_fails)
    assert await llm.generate("explanation", "sys", "user") is None
    assert llm.usage.failures == 1
    assert llm.usage.fallback_calls == 0


async def test_no_failsafe_key_means_no_failsafe(tmp_path, monkeypatch):
    """`fallback.enabled: true` alone isn't enough — no OPENROUTER_API_KEY,
    no failsafe attempt."""
    llm = client(tmp_path, api_key="sk-test", fallback_api_key=None, config=FALLBACK_CONFIG)
    assert llm._fallback_model("explanation") is None

    calls = []

    async def primary_only(*args, base_url=None, **kwargs):
        calls.append(1)
        raise RuntimeError("nope")

    monkeypatch.setattr(llm, "_call", primary_only)
    await llm.generate("explanation", "sys", "user")
    assert len(calls) == 2, "just the two primary retries, no failsafe attempt"


async def test_a_role_missing_from_fallback_models_has_no_failsafe(tmp_path):
    config = json.loads(json.dumps(FALLBACK_CONFIG))
    del config["fallback"]["models"]["explanation"]
    llm = client(tmp_path, api_key="sk-test", fallback_api_key="or-test", config=config)
    assert llm._fallback_model("explanation") is None
    assert llm._fallback_model("router") is not None


class _FakeCompletions:
    """Records exactly what `_call` sent, so the OpenRouter request shape is
    asserted directly instead of trusting `_call`'s own kwargs-building."""

    def __init__(self, text="failsafe prose"):
        self.captured_kwargs: dict | None = None
        self._text = text

    async def create(self, **kwargs):
        self.captured_kwargs = kwargs
        message = type("Message", (), {"content": self._text})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice], "usage": None})()


class _FakeOpenAIClient:
    def __init__(self, text="failsafe prose"):
        self.completions = _FakeCompletions(text)
        self.chat = type("Chat", (), {"completions": self.completions})()


async def test_the_failsafe_calls_openrouters_free_models_router(tmp_path):
    """openrouter/free does its own free-model selection and fallback
    server-side, so this is a plain single-model request — no local list to
    build or maintain."""
    llm = client(tmp_path, api_key=None, fallback_api_key="or-test", config=FALLBACK_CONFIG)
    fake = _FakeOpenAIClient()
    llm._fallback_client = fake  # bypass real AsyncOpenAI construction

    assert await llm.generate("explanation", "sys", "user") == "failsafe prose"

    kwargs = fake.completions.captured_kwargs
    assert kwargs["model"] == "openrouter/free"
    assert "extra_body" not in kwargs


# --- cost and the daily spend cap (R5) ------------------------------------


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def test_cost_is_computed_from_configured_rates_and_token_counts():
    model_cfg = CONFIG["models"]["merge"]
    cost = LLMReasoningClient._cost(model_cfg, _Usage(prompt_tokens=1000, completion_tokens=1000))
    assert cost == pytest.approx(0.0025 + 0.01)


def test_cost_is_zero_when_the_provider_gives_no_usage():
    assert LLMReasoningClient._cost(CONFIG["models"]["merge"], None) == 0.0


async def test_a_successful_call_accumulates_cost_onto_usage(tmp_path, monkeypatch):
    llm = client(tmp_path, api_key="sk-test")

    async def costed(*args, **kwargs):
        return "prose", 1.23

    monkeypatch.setattr(llm, "_call", costed)
    await llm.generate("explanation", "sys", "user")
    assert llm.usage.cost_usd == pytest.approx(1.23)


async def test_the_spend_cap_degrades_further_calls_without_invoking_the_provider(tmp_path, monkeypatch):
    """R5: once the configured daily_spend_cap_usd is spent, generate() must
    degrade (SRS 3.4.3) rather than place another paid call.
    """
    llm = client(tmp_path, api_key="sk-test")
    llm.usage.cost_usd = 5.0  # == daily_spend_cap_usd in CONFIG

    calls = []

    async def spy(*args, **kwargs):
        calls.append(1)
        return "prose", 0.01

    monkeypatch.setattr(llm, "_call", spy)
    result = await llm.generate("explanation", "sys", "user")

    assert result is None
    assert calls == [], "cap must be checked before the provider is ever called"
    assert llm.usage.failures == 1


async def test_a_cache_hit_is_served_even_over_the_spend_cap(tmp_path, monkeypatch):
    """Cache hits cost nothing, so the cap must not block them."""
    llm = client(tmp_path, api_key="sk-test", cache=True)

    async def once(*args, **kwargs):
        return "the explanation", 0.0

    monkeypatch.setattr(llm, "_call", once)
    assert await llm.generate("explanation", "sys", "user") == "the explanation"

    llm.usage.cost_usd = 5.0
    assert await llm.generate("explanation", "sys", "user") == "the explanation"


async def test_a_zero_or_missing_cap_never_degrades(tmp_path, monkeypatch):
    llm = client(tmp_path, api_key="sk-test")
    llm.config["limits"] = {"request_timeout_s": 8.0, "max_retries": 1}  # no daily_spend_cap_usd
    llm.usage.cost_usd = 999.0

    async def costed(*args, **kwargs):
        return "prose", 0.0

    monkeypatch.setattr(llm, "_call", costed)
    assert await llm.generate("explanation", "sys", "user") == "prose"


# --- cache ---------------------------------------------------------------


def test_cache_key_depends_on_everything_that_changes_the_answer():
    key = PromptCache.key("gpt-4o", "sys", "user", 0.2)
    assert key != PromptCache.key("gpt-4o-mini", "sys", "user", 0.2)
    assert key != PromptCache.key("gpt-4o", "other", "user", 0.2)
    assert key != PromptCache.key("gpt-4o", "sys", "other", 0.2)
    assert key != PromptCache.key("gpt-4o", "sys", "user", 0.9)
    assert key == PromptCache.key("gpt-4o", "sys", "user", 0.2), "must be deterministic"


async def test_a_cache_hit_costs_no_call(tmp_path, monkeypatch):
    """What makes the demo fast and repeatable (R5)."""
    llm = client(tmp_path, api_key="sk-test", cache=True)
    calls = []

    async def once(*args, **kwargs):
        calls.append(1)
        return "the explanation", 0.0

    monkeypatch.setattr(llm, "_call", once)

    assert await llm.generate("explanation", "sys", "user") == "the explanation"
    assert await llm.generate("explanation", "sys", "user") == "the explanation"
    assert len(calls) == 1
    assert llm.usage.cache_hits == 1


async def test_the_cache_answers_even_with_no_api_key(tmp_path, monkeypatch):
    """A cached demo still shows prose after the key expires mid-viva."""
    warm = client(tmp_path, api_key="sk-test", cache=True)

    async def canned(*args, **kwargs):
        return "cached prose", 0.0

    monkeypatch.setattr(warm, "_call", canned)
    await warm.generate("explanation", "sys", "user")

    cold = client(tmp_path, api_key=None, cache=True)
    assert await cold.generate("explanation", "sys", "user") == "cached prose"


def test_a_corrupt_cache_entry_is_discarded_not_raised(tmp_path):
    cache = PromptCache(tmp_path / "c", ttl_hours=1)
    key = PromptCache.key("m", "s", "u", 0.0)
    cache.put(key, "fine")
    (cache.path / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None


def test_an_expired_entry_is_discarded(tmp_path):
    cache = PromptCache(tmp_path / "c", ttl_hours=0.0)
    key = PromptCache.key("m", "s", "u", 0.0)
    cache.put(key, "stale")
    assert cache.get(key) is None


# --- config and protocol -------------------------------------------------


def test_model_ids_come_from_config_not_code(tmp_path):
    """SRS 3.6.5 vendor neutrality: swapping models must not be a code change."""
    llm = client(tmp_path)
    assert llm._model("merge")["model"] == "gpt-4o"
    assert llm._model("router")["model"] == "gpt-4o-mini"
    assert llm._model("router")["temperature"] == 0.0, "routing must be deterministic"


def test_an_unknown_role_fails_loudly(tmp_path):
    with pytest.raises(KeyError, match="models.nonsense"):
        client(tmp_path)._model("nonsense")


def test_both_clients_satisfy_the_contract_protocol(tmp_path):
    assert isinstance(client(tmp_path), LLMReasoningClientProtocol)
    assert isinstance(FakeLLMClient(), LLMReasoningClientProtocol)


async def test_fake_client_records_calls_and_can_be_made_unavailable():
    fake = FakeLLMClient(response="prose")
    assert await fake.generate_explanation({"a": 1}) == "prose"
    assert fake.calls[0][0] == "explanation"

    degraded = FakeLLMClient(available=False)
    assert await degraded.generate_explanation({"a": 1}) == ""
    assert degraded.usage.failures == 1


# --- provider_status (SRS 3.5.4 admin dashboard) --------------------------


def test_provider_status_is_not_configured_with_no_keys(tmp_path):
    status = client(tmp_path, api_key=None).provider_status()
    assert status["openai"].configured is False
    assert status["openai"].status == "not_configured"
    assert status["openrouter"].configured is False
    assert status["openrouter"].status == "not_configured"


def test_provider_status_is_unknown_before_any_call(tmp_path):
    """Configured but never actually exercised since this process started --
    distinct from a real success or failure, neither of which happened yet.
    """
    status = client(tmp_path, api_key="sk-test").provider_status()
    assert status["openai"].configured is True
    assert status["openai"].status == "unknown"
    assert status["openai"].last_checked_at is None


async def test_provider_status_is_ok_after_a_successful_call(tmp_path, monkeypatch):
    llm = client(tmp_path, api_key="sk-test")

    async def succeeds(*args, **kwargs):
        return "prose", 0.0

    monkeypatch.setattr(llm, "_call", succeeds)
    await llm.generate("explanation", "sys", "user")

    status = llm.provider_status()["openai"]
    assert status.status == "ok"
    assert status.last_error is None
    assert status.last_checked_at is not None


async def test_provider_status_is_down_after_every_attempt_fails(tmp_path, monkeypatch):
    llm = client(tmp_path, api_key="sk-test")

    async def fails(*args, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(llm, "_call", fails)
    await llm.generate("explanation", "sys", "user")

    status = llm.provider_status()["openai"]
    assert status.status == "down"
    assert status.last_error == "rate limited"


async def test_provider_status_distinguishes_a_down_primary_from_an_ok_failsafe(tmp_path, monkeypatch):
    """The dashboard's whole point: "openai: down" next to "openrouter: ok"
    tells an admin GPT-4o needs attention while the free failsafe covers it.
    """
    llm = client(tmp_path, api_key="sk-test", fallback_api_key="or-test", config=FALLBACK_CONFIG)

    async def primary_fails_fallback_succeeds(*args, base_url=None, **kwargs):
        if base_url is None:
            raise RuntimeError("openai is down")
        return "failsafe prose", 0.0

    monkeypatch.setattr(llm, "_call", primary_fails_fallback_succeeds)
    await llm.generate("explanation", "sys", "user")

    status = llm.provider_status()
    assert status["openai"].status == "down"
    assert status["openai"].last_error == "openai is down"
    assert status["openrouter"].status == "ok"


def test_provider_status_reports_cap_reached_distinctly_from_down(tmp_path):
    """Calls are deliberately skipped to protect the budget (R5) -- an admin
    should not read that as "GPT-4o is broken", only that today's spend cap
    is spent. Doesn't need a real generate() call: the cap check reads
    `usage.cost_usd` live, same as generate() itself does.
    """
    llm = client(tmp_path, api_key="sk-test")
    llm.usage.cost_usd = 5.0  # == CONFIG's daily_spend_cap_usd

    assert llm.provider_status()["openai"].status == "cap_reached"
