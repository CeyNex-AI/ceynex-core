"""Assertions for notification preferences and API-key management —
Account.tsx's two remaining "planned, not built yet" stub items.

No real Postgres here — `ceynex.api.preferences`/`ceynex.api.api_keys`'
module functions are monkeypatched, matching `test_history.py`'s style.
`tests/api/test_account_integration.py` (docker-marked) proves the real
Postgres round trip and the API-key-as-bearer-token auth path.
"""

from __future__ import annotations

import psycopg
import pytest
from fastapi.testclient import TestClient

from ceynex.api import api_keys as api_keys_module
from ceynex.api import preferences as preferences_module
from ceynex.api.auth import issue_token
from ceynex.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def token_for(email="researcher@ceynex.dev", role="researcher"):
    return issue_token(email, role)


def auth_header(token=None):
    return {"Authorization": f"Bearer {token or token_for()}"}


# --- GET/PUT /api/account/preferences ---------------------------------------


def test_get_preferences_returns_the_signed_in_users_settings(client, monkeypatch):
    monkeypatch.setattr(
        preferences_module,
        "get_for_user",
        lambda user_email: preferences_module.Preferences(
            dq_flag_alerts=True, forecast_updates=False, weekly_digest=True
        ),
    )

    response = client.get("/api/account/preferences", headers=auth_header())

    assert response.status_code == 200
    assert response.json() == {
        "dq_flag_alerts": True,
        "forecast_updates": False,
        "weekly_digest": True,
    }


def test_get_preferences_without_a_token_is_rejected(client):
    assert client.get("/api/account/preferences").status_code == 401


def test_get_preferences_surfaces_a_postgres_outage_as_503(client, monkeypatch):
    def boom(user_email):
        raise psycopg.OperationalError("db down")

    monkeypatch.setattr(preferences_module, "get_for_user", boom)
    response = client.get("/api/account/preferences", headers=auth_header())
    assert response.status_code == 503


def test_put_preferences_passes_the_body_through_to_upsert(client, monkeypatch):
    captured = {}

    def fake_upsert(user_email, *, dq_flag_alerts, forecast_updates, weekly_digest):
        captured["user_email"] = user_email
        captured["values"] = (dq_flag_alerts, forecast_updates, weekly_digest)
        return preferences_module.Preferences(
            dq_flag_alerts=dq_flag_alerts,
            forecast_updates=forecast_updates,
            weekly_digest=weekly_digest,
        )

    monkeypatch.setattr(preferences_module, "upsert", fake_upsert)

    response = client.put(
        "/api/account/preferences",
        json={"dq_flag_alerts": False, "forecast_updates": True, "weekly_digest": True},
        headers=auth_header(),
    )

    assert response.status_code == 200
    assert captured["user_email"] == "researcher@ceynex.dev"
    assert captured["values"] == (False, True, True)
    assert response.json() == {
        "dq_flag_alerts": False,
        "forecast_updates": True,
        "weekly_digest": True,
    }


def test_put_preferences_without_a_token_is_rejected(client):
    body = {"dq_flag_alerts": True, "forecast_updates": True, "weekly_digest": False}
    assert client.put("/api/account/preferences", json=body).status_code == 401


# --- GET/POST /api/account/api-keys -----------------------------------------


def test_list_api_keys_returns_the_signed_in_users_keys(client, monkeypatch):
    monkeypatch.setattr(
        api_keys_module,
        "list_for_user",
        lambda user_email: [
            api_keys_module.ApiKeyEntry(
                id=1,
                label="CI pipeline",
                key_prefix="ck_Ab1Cd2E",
                created_at="2026-08-27T00:00:00+00:00",
                last_used_at=None,
                revoked=False,
            )
        ],
    )

    response = client.get("/api/account/api-keys", headers=auth_header())

    assert response.status_code == 200
    assert response.json()["keys"] == [
        {
            "id": 1,
            "label": "CI pipeline",
            "key_prefix": "ck_Ab1Cd2E",
            "created_at": "2026-08-27T00:00:00+00:00",
            "last_used_at": None,
            "revoked": False,
        }
    ]


def test_list_api_keys_without_a_token_is_rejected(client):
    assert client.get("/api/account/api-keys").status_code == 401


def test_create_api_key_returns_the_plaintext_key_once(client, monkeypatch):
    captured = {}

    def fake_create(user_email, label):
        captured["user_email"] = user_email
        captured["label"] = label
        return api_keys_module.NewApiKey(
            id=7,
            label=label,
            key="ck_the-real-secret-value",
            key_prefix="ck_the-rea",
            created_at="2026-08-27T00:00:00+00:00",
        )

    monkeypatch.setattr(api_keys_module, "create", fake_create)

    response = client.post(
        "/api/account/api-keys", json={"label": "CI pipeline"}, headers=auth_header()
    )

    assert response.status_code == 200
    body = response.json()
    assert body["key"] == "ck_the-real-secret-value"
    assert body["label"] == "CI pipeline"
    assert captured["user_email"] == "researcher@ceynex.dev"
    assert captured["label"] == "CI pipeline"


def test_create_api_key_rejects_an_empty_label(client):
    response = client.post(
        "/api/account/api-keys", json={"label": ""}, headers=auth_header()
    )
    assert response.status_code == 422


def test_create_api_key_without_a_token_is_rejected(client):
    assert client.post("/api/account/api-keys", json={"label": "x"}).status_code == 401


def test_revoke_api_key_marks_an_owned_key_revoked(client, monkeypatch):
    monkeypatch.setattr(api_keys_module, "revoke", lambda key_id, user_email: True)

    response = client.post("/api/account/api-keys/1/revoke", headers=auth_header())

    assert response.status_code == 200
    assert response.json() == {"id": 1, "revoked": True}


def test_revoke_api_key_for_someone_elses_or_a_nonexistent_key_is_a_404(client, monkeypatch):
    monkeypatch.setattr(api_keys_module, "revoke", lambda key_id, user_email: False)

    response = client.post("/api/account/api-keys/999/revoke", headers=auth_header())
    assert response.status_code == 404


def test_revoke_api_key_without_a_token_is_rejected(client):
    assert client.post("/api/account/api-keys/1/revoke").status_code == 401


# --- an API key itself authenticates like a token ---------------------------


def test_a_valid_api_key_authenticates_other_require_user_routes(client, monkeypatch):
    """The whole point of "programmatic access": a `ck_`-prefixed bearer
    token must work anywhere a login JWT does, not just on the account
    routes -- exercised here against /api/auth/me, the simplest
    `require_user` route."""
    monkeypatch.setattr(
        api_keys_module,
        "authenticate",
        lambda raw_key: api_keys_module.TokenPayload(email="exporter@ceynex.dev", role="exporter")
        if raw_key == "ck_valid-key"
        else None,
    )

    response = client.get(
        "/api/auth/me", headers={"Authorization": "Bearer ck_valid-key"}
    )

    assert response.status_code == 200
    assert response.json() == {"email": "exporter@ceynex.dev", "role": "exporter"}


def test_a_revoked_or_unknown_api_key_is_rejected_like_a_bad_token(client, monkeypatch):
    monkeypatch.setattr(api_keys_module, "authenticate", lambda raw_key: None)

    response = client.get(
        "/api/auth/me", headers={"Authorization": "Bearer ck_revoked-or-unknown"}
    )
    assert response.status_code == 401
