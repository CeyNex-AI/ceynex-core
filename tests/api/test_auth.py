"""Assertions for signup, login and token verification (SRS 3.1.11).

No real database: `ceynex.api.users`' persistence functions are monkeypatched
with an in-memory store, so the route wiring — status codes, the identical
error for a bad email vs a bad password, JWT issue/verify, `/me` — is exercised
without Postgres. `tests/api/test_users.py` covers the domain logic the same
DB-free way; `tests/api/test_users_integration.py` proves the real-Postgres
half. The app's lifespan is never entered (same pattern as test_query.py).
"""

from __future__ import annotations

import time

import jwt
import psycopg
import pytest
from fastapi.testclient import TestClient

from ceynex.api import users as users_module
from ceynex.api.main import app
from ceynex.settings import jwt_secret

PASSWORD = "correct-horse-battery"


@pytest.fixture
def store(monkeypatch):
    """A tiny in-memory stand-in for the `users` table. `create_user` /
    `authenticate` / `get_by_email` are patched at their real module attribute,
    which is what `ceynex.api.auth` and `routes/auth.py` resolve at call time."""
    rows: dict[str, users_module.User] = {}
    counter = {"n": 0}

    def fake_create_user(email, password, role):
        email = email.strip().lower()
        if role not in users_module.VALID_ROLES:
            raise users_module.InvalidRoleError(role)
        if len(password) < users_module.MIN_PASSWORD_LENGTH:
            raise users_module.WeakPasswordError("too short")
        if email in rows:
            raise users_module.EmailTakenError(email)
        counter["n"] += 1
        user = users_module.User(
            id=counter["n"], email=email, role=role, created_at="2026-01-01T00:00:00+00:00",
            disabled=False, token_epoch=0,
        )
        rows[email] = user
        rows[f"__pw__{email}"] = password  # type: ignore[assignment]
        return user

    def fake_authenticate(email, password):
        email = email.strip().lower()
        user = rows.get(email)
        if user is None or user.disabled:
            return None
        if rows.get(f"__pw__{email}") != password:
            return None
        return user

    def fake_get_by_email(email):
        return rows.get(email.strip().lower())

    monkeypatch.setattr(users_module, "create_user", fake_create_user)
    monkeypatch.setattr(users_module, "authenticate", fake_authenticate)
    monkeypatch.setattr(users_module, "get_by_email", fake_get_by_email)
    return rows


@pytest.fixture
def live_epoch(store, monkeypatch):
    """Make `verify_token`'s epoch check read the in-memory `store` instead of
    the conftest default (which just returns 0 for everyone). Only the
    invalidation tests want this."""

    def _epoch(email):
        user = store.get(email.strip().lower())
        return None if user is None or user.disabled else user.token_epoch

    monkeypatch.setattr(users_module, "current_token_epoch", _epoch)


def _bump_epoch(store, email="dev@ceynex.dev"):
    """Simulate a password/role change / disable having advanced the epoch."""
    u = store[email]
    store[email] = users_module.User(
        id=u.id, email=u.email, role=u.role, created_at=u.created_at,
        disabled=u.disabled, token_epoch=u.token_epoch + 1,
    )


@pytest.fixture
def client():
    return TestClient(app)


def signup(client, email="dev@ceynex.dev", password=PASSWORD):
    return client.post("/api/auth/signup", json={"email": email, "password": password})


def login(client, email="dev@ceynex.dev", password=PASSWORD):
    return client.post("/api/auth/login", json={"email": email, "password": password})


# --- signup ---------------------------------------------------------------


def test_signup_creates_an_account_at_the_default_role_and_logs_in(client, store):
    body = signup(client).json()
    assert body["email"] == "dev@ceynex.dev"
    assert body["role"] == users_module.DEFAULT_ROLE
    assert isinstance(body["token"], str) and body["token"]


def test_signup_then_login_round_trips(client, store):
    signup(client)
    assert login(client).json()["role"] == users_module.DEFAULT_ROLE


def test_a_second_signup_for_the_same_email_is_a_409(client, store):
    signup(client)
    assert signup(client).status_code == 409


def test_signup_is_case_and_whitespace_insensitive_on_email(client, store):
    signup(client, email="  Dev@CeyNex.DEV  ")
    assert login(client, email="dev@ceynex.dev").status_code == 200


def test_a_short_password_is_rejected_before_the_route_body_runs(client, store):
    assert client.post(
        "/api/auth/signup", json={"email": "x@ceynex.dev", "password": "short"}
    ).status_code == 422


# --- login --------------------------------------------------------------


def test_a_correct_login_returns_a_token_and_role(client, store):
    signup(client)
    body = login(client).json()
    assert body["email"] == "dev@ceynex.dev"
    assert isinstance(body["token"], str) and body["token"]


def test_a_wrong_password_is_rejected(client, store):
    signup(client)
    assert login(client, password="not-it").status_code == 401


def test_an_unknown_email_is_rejected(client, store):
    assert login(client, email="nobody@ceynex.dev").status_code == 401


def test_unknown_email_and_wrong_password_give_the_same_error(client, store):
    signup(client)
    wrong_password = login(client, password="nope").json()["detail"]
    unknown_email = login(client, email="nobody@ceynex.dev").json()["detail"]
    assert wrong_password == unknown_email


def test_a_disabled_account_cannot_log_in(client, store):
    signup(client)
    disabled = store["dev@ceynex.dev"]
    store["dev@ceynex.dev"] = users_module.User(
        id=disabled.id, email=disabled.email, role=disabled.role,
        created_at=disabled.created_at, disabled=True, token_epoch=disabled.token_epoch,
    )
    assert login(client).status_code == 401


def test_a_missing_field_is_rejected(client, store):
    assert client.post("/api/auth/login", json={"email": "dev@ceynex.dev"}).status_code == 422


# --- /me and token verification ---------------------------------------------


def test_me_returns_the_signed_in_user(client, store):
    token = signup(client).json()["token"]
    body = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert body == {"email": "dev@ceynex.dev", "role": users_module.DEFAULT_ROLE}


def test_me_without_a_token_is_rejected(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_with_a_garbage_token_is_rejected(client):
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


def test_me_with_a_token_signed_by_the_wrong_secret_is_rejected(client):
    forged = jwt.encode(
        {"sub": "dev@ceynex.dev", "role": "admin", "iat": 0, "exp": 9_999_999_999},
        "someone-elses-secret",
        algorithm="HS256",
    )
    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


def test_me_with_an_expired_token_is_rejected(client):
    expired = jwt.encode(
        {"sub": "dev@ceynex.dev", "role": "researcher", "iat": 0, "exp": int(time.time()) - 1},
        jwt_secret(),
        algorithm="HS256",
    )
    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401


def test_me_role_comes_from_the_token_claim_not_a_re_derivation(client, store):
    """A stale or tampered claim can't silently change what /me reports back
    within the token's life — the role is whatever was signed in."""
    signup(client)
    token = login(client).json()["token"]
    role = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["role"]
    assert role == users_module.DEFAULT_ROLE


# --- session invalidation (token_epoch) -----------------------------------


def test_a_token_stops_verifying_once_the_accounts_epoch_moves_on(client, store, live_epoch):
    """A password/role change or disable bumps `token_epoch`; a token minted
    against the old value is rejected at its next request, not 8 h later."""
    token = signup(client).json()["token"]
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    _bump_epoch(store)
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_a_token_for_a_since_disabled_account_stops_verifying(client, store, live_epoch):
    token = signup(client).json()["token"]
    u = store["dev@ceynex.dev"]
    store["dev@ceynex.dev"] = users_module.User(
        id=u.id, email=u.email, role=u.role, created_at=u.created_at,
        disabled=True, token_epoch=u.token_epoch,
    )
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_a_valid_token_still_verifies_when_the_epoch_check_cannot_reach_postgres(client, store, monkeypatch):
    """Fail open — a Postgres blip must not log everyone out. The token's
    signature was still good."""
    token = signup(client).json()["token"]

    def boom(email):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(users_module, "current_token_epoch", boom)
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200
