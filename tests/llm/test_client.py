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
        "router": {"model": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 300},
        "merge": {"model": "gpt-4o", "temperature": 0.2, "max_tokens": 1200},
        "explanation": {"model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 600},
    },
    "limits": {"request_timeout_s": 8.0, "max_retries": 1},
    "cache": {"enabled": False, "path": ".cache/llm-test", "ttl_hours": 1},
}


def client(tmp_path, *, api_key=None, cache=False):
    config = json.loads(json.dumps(CONFIG))
    config["cache"] = {"enabled": cache, "path": str(tmp_path / "cache"), "ttl_hours": 1}
    return LLMReasoningClient(config=config, api_key=api_key)


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
        return "the explanation"

    monkeypatch.setattr(llm, "_call", once)

    assert await llm.generate("explanation", "sys", "user") == "the explanation"
    assert await llm.generate("explanation", "sys", "user") == "the explanation"
    assert len(calls) == 1
    assert llm.usage.cache_hits == 1


async def test_the_cache_answers_even_with_no_api_key(tmp_path, monkeypatch):
    """A cached demo still shows prose after the key expires mid-viva."""
    warm = client(tmp_path, api_key="sk-test", cache=True)

    async def canned(*args, **kwargs):
        return "cached prose"

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
