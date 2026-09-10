"""Assertions for rate limiting on POST /api/auth/login and /api/auth/signup.

The window mechanics are covered by `test_rate_limit.py`; what matters here is
that the auth endpoints are wired to it, that an attempt is counted against
*both* the address and the email, and that the limit has its own config block
and its own identity namespace (so hammering login can't spend the query
allowance and vice versa).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ceynex.api import users as users_module
from ceynex.api.main import app
from ceynex.api.rate_limit import Decision
from ceynex.api.routes import auth as auth_routes


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def store(monkeypatch):
    """Minimal in-memory `users` so login/signup run without Postgres — the
    limiter fires before the credential check either way."""
    rows: dict[str, object] = {}

    def fake_create_user(email, password, role):
        email = email.strip().lower()
        if email in rows:
            raise users_module.EmailTakenError(email)
        u = users_module.User(
            id=len(rows) + 1, email=email, role=role,
            created_at="2026-01-01T00:00:00+00:00", disabled=False, token_epoch=0,
        )
        rows[email] = u
        return u

    def fake_authenticate(email, password):
        return rows.get(email.strip().lower())

    monkeypatch.setattr(users_module, "create_user", fake_create_user)
    monkeypatch.setattr(users_module, "authenticate", fake_authenticate)
    return rows


def _tighten(monkeypatch, per_minute):
    monkeypatch.setattr(
        auth_routes.settings, "load_config",
        lambda name: {"auth_rate_limit": {
            "enabled": True, "attempts_per_minute": per_minute, "window_seconds": 60,
        }},
    )


def login(client, email="a@ceynex.dev", password="whatever12"):
    return client.post("/api/auth/login", json={"email": email, "password": password})


# --- the limit -----------------------------------------------------------


def test_login_refuses_with_429_and_a_retry_after_once_the_limit_is_hit(client, store, monkeypatch):
    _tighten(monkeypatch, 3)
    for _ in range(3):
        assert login(client).status_code == 401  # wrong creds, but still counted
    refused = login(client)
    assert refused.status_code == 429
    assert refused.headers["Retry-After"].isdigit()
    assert "too many attempts" in refused.json()["detail"]


def test_signup_is_limited_too(client, store, monkeypatch):
    _tighten(monkeypatch, 2)
    body = lambda e: {"email": e, "password": "longenough1"}  # noqa: E731
    assert client.post("/api/auth/signup", json=body("one@ceynex.dev")).status_code == 201
    assert client.post("/api/auth/signup", json=body("two@ceynex.dev")).status_code == 201
    # third attempt from the same address trips the IP counter regardless of email
    assert client.post("/api/auth/signup", json=body("three@ceynex.dev")).status_code == 429


def test_varying_the_email_does_not_dodge_the_address_counter(client, store, monkeypatch):
    """The IP counter gates independently of the email — a stuffing script that
    rotates the target email still trips it."""
    _tighten(monkeypatch, 3)
    for i in range(3):
        login(client, email=f"target{i}@ceynex.dev")
    assert login(client, email="another@ceynex.dev").status_code == 429


def test_both_an_ip_and_a_normalised_email_identity_are_checked(client, store):
    seen: list[str] = []

    class Recorder:
        async def check(self, identity, limit, window_s):
            seen.append(identity)
            return Decision(allowed=True, limit=limit, remaining=limit, retry_after_s=0)

    auth_routes.set_window(Recorder())
    try:
        client.post("/api/auth/login", json={"email": "  Person@Ceynex.DEV ", "password": "x" * 10})
    finally:
        auth_routes.set_window(None)
    assert seen[0].startswith("auth:ip:")
    assert "auth:email:person@ceynex.dev" in seen  # normalised, from the body


def test_the_limit_can_be_switched_off_in_config(client, store, monkeypatch):
    monkeypatch.setattr(
        auth_routes.settings, "load_config",
        lambda name: {"auth_rate_limit": {"enabled": False, "attempts_per_minute": 1, "window_seconds": 60}},
    )
    for _ in range(5):
        assert login(client).status_code == 401


def test_the_shipped_config_is_readable_and_sane():
    from ceynex import settings

    config = settings.load_config("api")["auth_rate_limit"]
    assert config["attempts_per_minute"] > 0
    assert config["window_seconds"] > 0


def test_auth_limit_uses_its_own_namespace_not_the_query_one():
    """Prefix guard — an unprefixed identity would share Redis keys with
    /api/query and silently eat that allowance (see routes/news.py)."""
    from ceynex.api import rate_limit

    assert rate_limit.identity_of("a@x.dev", None) == "user:a@x.dev"
    # the auth route builds `auth:ip:...` / `auth:email:...`, never the bare form
