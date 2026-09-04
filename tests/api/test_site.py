"""Assertions for the site-wide theme setting: `GET /api/site/theme` is public,
`POST /api/site/theme` requires the admin role.

No real Postgres here -- `ceynex.api.site_settings`'s functions are
monkeypatched at their real module attribute, same pattern as
`test_admin.py`. `tests/api/test_site_integration.py` proves the real-Postgres
half of `ceynex/api/site_settings.py`.
"""

from __future__ import annotations

import psycopg
import pytest
from fastapi.testclient import TestClient

from ceynex.api import site_settings as site_settings_module
from ceynex.api.auth import DemoUser, issue_token
from ceynex.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def token_for(email: str, role: str) -> str:
    return issue_token(DemoUser(email=email, role=role, password_hash=b""))


def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {token_for('admin@ceynex.dev', 'admin')}"}


def researcher_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {token_for('researcher@ceynex.dev', 'researcher')}"}


def test_get_theme_is_public_and_needs_no_token(client, monkeypatch):
    monkeypatch.setattr(site_settings_module, "get_theme", lambda: "signal-deck")

    response = client.get("/api/site/theme")

    assert response.status_code == 200
    assert response.json() == {"theme": "signal-deck"}


def test_get_theme_defaults_to_classic(client, monkeypatch):
    monkeypatch.setattr(site_settings_module, "get_theme", lambda: "classic")

    response = client.get("/api/site/theme")

    assert response.json() == {"theme": "classic"}


def test_post_theme_rejects_a_non_admin_role(client):
    response = client.post(
        "/api/site/theme", json={"theme": "signal-deck"}, headers=researcher_headers()
    )
    assert response.status_code == 403


def test_post_theme_rejects_no_token(client):
    response = client.post("/api/site/theme", json={"theme": "signal-deck"})
    assert response.status_code == 401


def test_post_theme_as_admin_sets_it(client, monkeypatch):
    captured = {}

    def fake_set_theme(theme, *, updated_by):
        captured["theme"] = theme
        captured["updated_by"] = updated_by
        return theme

    monkeypatch.setattr(site_settings_module, "set_theme", fake_set_theme)

    response = client.post(
        "/api/site/theme", json={"theme": "signal-deck"}, headers=admin_headers()
    )

    assert response.status_code == 200
    assert response.json() == {"theme": "signal-deck"}
    assert captured == {"theme": "signal-deck", "updated_by": "admin@ceynex.dev"}


def test_post_theme_with_an_unknown_name_is_a_422(client, monkeypatch):
    def boom(theme, *, updated_by):
        raise ValueError(f"unknown theme {theme!r}; must be one of ('classic', 'signal-deck')")

    monkeypatch.setattr(site_settings_module, "set_theme", boom)

    response = client.post(
        "/api/site/theme", json={"theme": "neon"}, headers=admin_headers()
    )

    assert response.status_code == 422


def test_post_theme_outage_is_a_503(client, monkeypatch):
    def boom(theme, *, updated_by):
        raise psycopg.OperationalError("db down")

    monkeypatch.setattr(site_settings_module, "set_theme", boom)

    response = client.post(
        "/api/site/theme", json={"theme": "signal-deck"}, headers=admin_headers()
    )
    assert response.status_code == 503
