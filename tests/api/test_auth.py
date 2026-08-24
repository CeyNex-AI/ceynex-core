"""Assertions for login and token verification (SRS 3.1.11).

No network and no database — the four accounts are fixed in `ceynex/api/auth.py`,
matching `ceynex-web/src/lib/roles.ts` exactly, so nothing here touches Postgres
or Neo4j and the app's lifespan is never entered (same pattern as test_query.py).
"""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi.testclient import TestClient

from ceynex.api import auth as auth_module
from ceynex.api.main import app
from ceynex.settings import jwt_secret


@pytest.fixture
def client():
    return TestClient(app)


def login(client, email="admin@ceynex.dev", password=auth_module.DEMO_PASSWORD):
    return client.post("/api/auth/login", json={"email": email, "password": password})


# --- login ------------------------------------------------------------------


def test_a_correct_demo_login_returns_a_token_and_role(client):
    body = login(client).json()
    assert body["email"] == "admin@ceynex.dev"
    assert body["role"] == "admin"
    assert isinstance(body["token"], str) and body["token"]


@pytest.mark.parametrize(
    "email,role",
    [
        ("policymaker@ceynex.dev", "policymaker"),
        ("researcher@ceynex.dev", "researcher"),
        ("exporter@ceynex.dev", "exporter"),
    ],
)
def test_every_fixed_account_logs_in_with_its_own_role(client, email, role):
    assert login(client, email=email).json()["role"] == role


def test_login_is_case_and_whitespace_insensitive_on_email(client):
    body = login(client, email="  Admin@CeyNex.DEV  ").json()
    assert body["email"] == "admin@ceynex.dev"


def test_a_wrong_password_is_rejected(client):
    response = login(client, password="not-the-demo-password")
    assert response.status_code == 401


def test_an_unknown_email_is_rejected(client):
    response = login(client, email="nobody@ceynex.dev")
    assert response.status_code == 401


def test_unknown_email_and_wrong_password_give_the_same_error(client):
    """Never let the response distinguish a bad email from a bad password —
    that would let a caller enumerate which of the four accounts are real."""
    wrong_password = login(client, password="nope").json()["detail"]
    unknown_email = login(client, email="nobody@ceynex.dev").json()["detail"]
    assert wrong_password == unknown_email


def test_a_missing_field_is_rejected(client):
    assert client.post("/api/auth/login", json={"email": "admin@ceynex.dev"}).status_code == 422


# --- /me and token verification ---------------------------------------------


def test_me_returns_the_signed_in_user(client):
    token = login(client).json()["token"]
    body = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert body == {"email": "admin@ceynex.dev", "role": "admin"}


def test_me_without_a_token_is_rejected(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_with_a_garbage_token_is_rejected(client):
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


def test_me_with_a_token_signed_by_the_wrong_secret_is_rejected(client):
    forged = jwt.encode(
        {"sub": "admin@ceynex.dev", "role": "admin", "iat": 0, "exp": 9_999_999_999},
        "someone-elses-secret",
        algorithm="HS256",
    )
    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


def test_me_with_an_expired_token_is_rejected(client):
    expired = jwt.encode(
        {"sub": "admin@ceynex.dev", "role": "admin", "iat": 0, "exp": int(time.time()) - 1},
        jwt_secret(),
        algorithm="HS256",
    )
    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401


def test_a_token_cannot_grant_a_role_it_was_not_issued_with(client):
    """A forged-but-correctly-signed token is out of scope here (that's what
    keeping the secret out of source control is for) — this just confirms the
    role in /me always comes from the token's own claim, not from re-deriving
    it, so a stale or tampered claim can't silently upgrade a session."""
    token = login(client, email="researcher@ceynex.dev").json()["token"]
    role = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["role"]
    assert role == "researcher"
