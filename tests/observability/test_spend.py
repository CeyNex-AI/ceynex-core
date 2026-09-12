"""Daily spend limits that hold across workers (D16).

The headline assertion is the one D15 could only disclose: with two uvicorn
workers, true spend could reach **twice** `daily_spend_cap_usd`, because each
worker counted only its own. Two clients sharing one Redis-backed counter — two
workers, in miniature — now stop at the cap, not double it.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest

from ceynex.llm import LLMReasoningClient
from ceynex.llm.client import _CallOutcome
from ceynex.observability import context, spend, trace
from ceynex.observability.spend import InProcessSpendCounter, RedisSpendCounter

CONFIG = {
    "provider": "openai",
    "models": {
        "merge": {
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 1200,
            "cost_per_1k_input_tokens": 0.0025,
            "cost_per_1k_output_tokens": 0.01,
        }
    },
    "limits": {
        "request_timeout_s": 8.0,
        "max_retries": 0,
        "daily_spend_cap_usd": 5.0,
        "per_user_daily_cap_usd": 1.0,
    },
    "cache": {"enabled": False, "path": ".cache/llm-test", "ttl_hours": 1},
}


class FakeRedis:
    """INCRBYFLOAT / MGET / EXPIRE, in memory, shared by whoever holds it."""

    def __init__(self):
        self.values: dict[str, float] = {}

    async def incrbyfloat(self, key, amount):
        self.values[key] = self.values.get(key, 0.0) + amount
        return self.values[key]

    async def expire(self, key, seconds):
        return True

    async def mget(self, keys):
        return [None if k not in self.values else str(self.values[k]).encode() for k in keys]


class DownRedis(FakeRedis):
    async def incrbyfloat(self, key, amount):
        raise ConnectionError("redis is down")

    async def mget(self, keys):
        raise ConnectionError("redis is down")


def _client(counter, *, fallback=False):
    config = json.loads(json.dumps(CONFIG))
    if fallback:
        config["fallback"] = {"enabled": True, "base_url": "https://openrouter.ai/api/v1",
                              "timeout_s": 5.0, "models": {"merge": "openrouter/free"}}
    return LLMReasoningClient(config=config, api_key="sk-test",
                              fallback_api_key="or-test" if fallback else None, spend=counter)


def _costing(llm, cost, providers=None):
    async def call(*args, base_url=None, **kwargs):
        if providers is not None:
            providers.append(base_url)
        return _CallOutcome("prose", 0.0 if base_url else cost, 0, 0)

    llm._call = call  # noqa: SLF001 - the provider round trip, stood in for
    return llm


async def _as(user_email, llm):
    token = context.install(context.RequestObservability(user_email=user_email))
    try:
        return await llm.generate("merge", "sys", f"user {user_email}")
    finally:
        context.reset(token)


# --- the counters ---------------------------------------------------------------


async def test_a_day_is_counted_deployment_wide_and_per_reader():
    counter = InProcessSpendCounter()
    await counter.add(0.4, "a@x")
    await counter.add(0.5, "b@x")
    await counter.add(0.1, None)

    assert (await counter.spent("a@x")).user_usd == pytest.approx(0.4)
    assert (await counter.spent("a@x")).total_usd == pytest.approx(1.0)
    assert (await counter.spent(None)).user_usd == 0.0


async def test_nothing_is_counted_for_a_free_call():
    counter = InProcessSpendCounter()
    await counter.add(0.0, "a@x")
    await counter.add(-1.0, "a@x")
    assert (await counter.spent("a@x")).total_usd == 0.0


async def test_the_limits_reset_at_midnight_utc(monkeypatch):
    counter = InProcessSpendCounter()
    monkeypatch.setattr(spend, "today", lambda now=None: "2026-09-10")
    await counter.add(4.0, "a@x")
    monkeypatch.setattr(spend, "today", lambda now=None: "2026-09-11")
    assert (await counter.spent("a@x")).total_usd == 0.0


def test_the_reset_time_is_the_next_utc_midnight():
    assert spend.resets_at(datetime(2026, 9, 10, 17, 30, tzinfo=UTC)) == datetime(
        2026, 9, 11, tzinfo=UTC
    )


async def test_two_workers_count_one_number_through_redis():
    redis = FakeRedis()
    worker_a, worker_b = RedisSpendCounter(redis), RedisSpendCounter(redis)
    await worker_a.add(3.0, "a@x")
    await worker_b.add(2.5, "a@x")
    for worker in (worker_a, worker_b):
        seen = await worker.spent("a@x")
        assert seen.total_usd == pytest.approx(5.5)
        assert seen.user_usd == pytest.approx(5.5)


async def test_a_redis_outage_falls_back_to_this_workers_own_count():
    """Fail open, but never to no cap at all: the worker's own spend still counts."""
    counter = RedisSpendCounter(DownRedis())
    await counter.add(2.0, "a@x")
    assert (await counter.spent("a@x")).total_usd == pytest.approx(2.0)


async def test_a_count_that_went_missing_cannot_make_spent_money_reappear():
    """A key that expired early, or a replica that lost writes, must not read
    lower than what this worker has itself spent today."""
    redis = FakeRedis()
    counter = RedisSpendCounter(redis)
    await counter.add(3.0, None)
    redis.values.clear()
    assert (await counter.spent(None)).total_usd == pytest.approx(3.0)


# --- enforcement ------------------------------------------------------------------


async def test_two_workers_stop_at_the_cap_not_at_twice_it():
    """D15's named weakness, closed. The same scenario, counted per worker,
    lets the second worker spend on past the cap."""
    shared = FakeRedis()
    worker_a = _costing(_client(RedisSpendCounter(shared)), cost=3.0)
    worker_b = _costing(_client(RedisSpendCounter(shared)), cost=3.0)
    assert await worker_a.generate("merge", "sys", "one") == "prose"   # 3.0 spent
    assert await worker_b.generate("merge", "sys", "two") == "prose"   # 6.0 spent
    assert await worker_a.generate("merge", "sys", "three") is None, "past the cap"
    assert await worker_b.generate("merge", "sys", "four") is None, "past the cap"

    # The old shape, for contrast: each worker only ever saw its own 3.0.
    alone_a = _costing(_client(InProcessSpendCounter()), cost=3.0)
    alone_b = _costing(_client(InProcessSpendCounter()), cost=3.0)
    await alone_a.generate("merge", "sys", "one")
    await alone_b.generate("merge", "sys", "two")
    assert await alone_b.generate("merge", "sys", "three") == "prose", "per-worker cap"


async def test_a_reader_over_their_budget_gets_the_free_failsafe_first():
    counter = InProcessSpendCounter()
    providers: list[str | None] = []
    llm = _costing(_client(counter, fallback=True), cost=0.6, providers=providers)

    await _as("a@x", llm)  # 0.6
    await _as("a@x", llm)  # 1.2 — over the 1.0 per-reader budget
    providers.clear()
    assert await _as("a@x", llm) == "prose"
    assert providers == ["https://openrouter.ai/api/v1"], "the paid primary was used"


async def test_a_reader_over_their_budget_degrades_when_there_is_no_failsafe():
    llm = _costing(_client(InProcessSpendCounter()), cost=0.6)
    await _as("a@x", llm)
    await _as("a@x", llm)
    assert await _as("a@x", llm) is None, "SRS 3.4.3: degrade, do not keep spending"


async def test_one_readers_budget_is_not_another_readers():
    llm = _costing(_client(InProcessSpendCounter()), cost=0.6)
    await _as("a@x", llm)
    await _as("a@x", llm)
    assert await _as("b@x", llm) == "prose"


async def test_the_trace_says_why_the_prose_is_missing():
    counter = InProcessSpendCounter()
    await counter.add(1.5, "a@x")
    llm = _costing(_client(counter), cost=0.1)
    sink = trace.TraceSink(request_id="r", loop=asyncio.get_running_loop())
    token = context.install(context.RequestObservability(trace=sink, user_email="a@x"))
    try:
        assert await llm.generate("merge", "sys", "user") is None
    finally:
        context.reset(token)

    budget = [e.payload for e in sink.history if e.kind == "budget"]
    assert budget and budget[0]["scope"] == "user"
    assert budget[0]["cap_usd"] == 1.0 and budget[0]["spent_usd"] == pytest.approx(1.5)
    assert budget[0]["resets_at"].endswith("00:00:00+00:00")


async def test_an_anonymous_call_meets_only_the_deployment_cap():
    """The eval harness and the demo CLI have no reader; a per-reader budget
    has nobody to apply to."""
    llm = _costing(_client(InProcessSpendCounter()), cost=0.6)
    for _ in range(3):
        assert await llm.generate("merge", "sys", "q") == "prose"
