"""Assertions for SRS 3.4.6 — rate limiting on POST /api/query.

The requirement protects availability (SRS 3.3.1) and response time (3.4.1) for
*other* callers, so the assertions that matter are the ones about isolation: one
caller exhausting its allowance must not affect anybody else, and dropping the
token must not hand you a fresh allowance.
"""

import pytest
from fastapi.testclient import TestClient

from ceynex.api import deps as deps_module
from ceynex.api import rate_limit
from ceynex.api.main import app
from ceynex.api.routes import query as query_routes

from .test_query import ANSWERED, FakeGraph, FakeKG, FakeLLM


def runtime():
    return deps_module.Runtime(kg=FakeKG(), llm=FakeLLM(), deps=None, graph=FakeGraph(ANSWERED))


@pytest.fixture
def client():
    deps_module.set_runtime(runtime())
    try:
        yield TestClient(app)
    finally:
        deps_module.set_runtime(None)


def post(client, **kwargs):
    return client.post("/api/query", json={"query": "cinnamon export trend"}, **kwargs)


# --- the window ----------------------------------------------------------


async def test_requests_within_the_limit_are_allowed():
    window = rate_limit.InProcessWindow()
    for _ in range(5):
        assert (await window.check("user:a@example.com", 5, 60)).allowed


async def test_the_request_past_the_limit_is_refused():
    window = rate_limit.InProcessWindow()
    for _ in range(5):
        await window.check("user:a@example.com", 5, 60)

    decision = await window.check("user:a@example.com", 5, 60)
    assert not decision.allowed
    assert decision.remaining == 0
    assert decision.retry_after_s >= 1, "a Retry-After of 0 tells the caller nothing"


async def test_one_caller_exhausting_its_allowance_does_not_affect_another():
    """The entire point of the requirement — see the module docstring."""
    window = rate_limit.InProcessWindow()
    for _ in range(5):
        await window.check("user:noisy@example.com", 5, 60)

    assert not (await window.check("user:noisy@example.com", 5, 60)).allowed
    assert (await window.check("user:quiet@example.com", 5, 60)).allowed


async def test_the_counter_resets_when_the_window_rolls_over(monkeypatch):
    """A refused caller gets its allowance back in the next window, not never.

    Time is moved rather than slept through: a test that waits 60 seconds for
    this would be dropped from the suite the first time someone was in a hurry.
    """
    now = 1_000_000.0
    monkeypatch.setattr(rate_limit.time, "time", lambda: now)

    window = rate_limit.InProcessWindow()
    for _ in range(3):
        await window.check("user:a@example.com", 3, 60)
    assert not (await window.check("user:a@example.com", 3, 60)).allowed

    now += 60
    assert (await window.check("user:a@example.com", 3, 60)).allowed


async def test_stale_windows_do_not_accumulate_forever(monkeypatch):
    """The counter dict is pruned, or a long-running process leaks one entry
    per distinct caller per window — invisible until it is not."""
    now = 1_000_000.0
    monkeypatch.setattr(rate_limit.time, "time", lambda: now)

    window = rate_limit.InProcessWindow()
    for i in range(1100):
        await window.check(f"ip:10.0.0.{i}", 10, 60)
    assert len(window._hits) > 1000

    now += 60
    await window.check("ip:10.0.0.1", 10, 60)
    assert len(window._hits) < 1000, "counters from the previous window were never dropped"


async def test_a_redis_outage_allows_the_request():
    """Failing open, deliberately — see rate_limit.py's "Failing open" section."""

    class BrokenRedis:
        async def incr(self, key):
            raise ConnectionError("redis is down")

        async def expire(self, key, seconds):  # pragma: no cover - never reached
            raise AssertionError

    decision = await rate_limit.RedisWindow(BrokenRedis()).check("user:a", 1, 60)
    assert decision.allowed


async def test_redis_sets_an_expiry_only_on_the_first_hit_of_a_window():
    """A second EXPIRE per request would be a wasted round trip on the hot path."""

    class RecordingRedis:
        def __init__(self):
            self.counts: dict[str, int] = {}
            self.expires: list[str] = []

        async def incr(self, key):
            self.counts[key] = self.counts.get(key, 0) + 1
            return self.counts[key]

        async def expire(self, key, seconds):
            self.expires.append(key)

    redis = RecordingRedis()
    window = rate_limit.RedisWindow(redis)
    for _ in range(4):
        await window.check("user:a@example.com", 10, 60)

    assert len(redis.expires) == 1


# --- identity ------------------------------------------------------------


def test_a_signed_in_user_is_counted_by_email_not_address():
    assert rate_limit.identity_of("a@example.com", "10.0.0.1") == "user:a@example.com"


def test_an_anonymous_caller_is_counted_by_address():
    """`POST /api/query` answers anonymous callers, so they must be limited too —
    otherwise dropping the token buys a fresh allowance."""
    assert rate_limit.identity_of(None, "10.0.0.1") == "ip:10.0.0.1"


def test_the_two_namespaces_cannot_collide():
    """One busy NAT address must not exhaust a signed-in user's own allowance."""
    assert rate_limit.identity_of("10.0.0.1", None) != rate_limit.identity_of(None, "10.0.0.1")


# --- client_ip (behind the nginx proxy) --------------------------------


class _Req:
    def __init__(self, headers: dict, host: str = "10.0.0.9"):
        self.headers = headers
        self.client = type("C", (), {"host": host})()


def test_client_ip_prefers_x_real_ip_over_forwarded_for():
    req = _Req({"x-real-ip": "203.0.113.7", "x-forwarded-for": "1.2.3.4"})
    assert rate_limit.client_ip(req) == "203.0.113.7"


def test_client_ip_takes_the_last_forwarded_for_hop_only():
    """`$proxy_add_x_forwarded_for` appends to a client-supplied value, so only
    the last hop — the one our nginx added — can be trusted."""
    req = _Req({"x-forwarded-for": "1.1.1.1, 2.2.2.2, 9.9.9.9"})
    assert rate_limit.client_ip(req) == "9.9.9.9"


def test_client_ip_falls_back_to_the_socket_peer_with_no_proxy_headers():
    assert rate_limit.client_ip(_Req({}, host="10.0.0.5")) == "10.0.0.5"


# --- the endpoint --------------------------------------------------------


def test_the_endpoint_refuses_with_429_and_a_retry_after_header(client, monkeypatch):
    monkeypatch.setattr(
        query_routes.settings,
        "load_config",
        lambda name: {"rate_limit": {"enabled": True, "query_per_minute": 3, "window_seconds": 60}},
    )

    for _ in range(3):
        assert post(client).status_code == 200

    refused = post(client)
    assert refused.status_code == 429
    assert refused.headers["Retry-After"].isdigit()
    assert "rate limit exceeded" in refused.json()["detail"]


def test_the_limit_can_be_switched_off_in_config(client, monkeypatch):
    monkeypatch.setattr(
        query_routes.settings,
        "load_config",
        lambda name: {
            "rate_limit": {"enabled": False, "query_per_minute": 1, "window_seconds": 60}
        },
    )

    for _ in range(4):
        assert post(client).status_code == 200


def test_the_shipped_config_is_readable_and_sane():
    """A rate limit nobody can parse is a rate limit that silently uses defaults."""
    from ceynex import settings

    config = settings.load_config("api")["rate_limit"]
    assert config["query_per_minute"] > 0
    assert config["window_seconds"] > 0
