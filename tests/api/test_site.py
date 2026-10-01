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

from ceynex.api import audit as audit_module
from ceynex.api import site_settings as site_settings_module
from ceynex.api.auth import issue_token
from ceynex.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def captured_audit(monkeypatch):
    """A theme change writes an audit row first (SRS 3.4.7), so the real DB
    write is patched to a recorder, as in test_admin.py."""
    calls: list[dict[str, object]] = []

    def fake_record(*, actor_email: str, action: str, target: str | None) -> None:
        calls.append({"actor_email": actor_email, "action": action, "target": target})

    monkeypatch.setattr(audit_module, "record", fake_record)
    return calls


def token_for(email: str, role: str) -> str:
    return issue_token(email, role)


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


def test_post_theme_with_an_unknown_name_is_a_422_and_not_audited(client, captured_audit):
    response = client.post("/api/site/theme", json={"theme": "neon"}, headers=admin_headers())

    assert response.status_code == 422
    assert captured_audit == []


def test_post_theme_writes_an_audit_row_before_the_change(client, monkeypatch, captured_audit):
    order: list[str] = []
    monkeypatch.setattr(
        audit_module, "record", lambda **kw: (order.append("audit"), captured_audit.append(kw))
    )
    monkeypatch.setattr(
        site_settings_module,
        "set_theme",
        lambda theme, *, updated_by: (order.append("set"), theme)[1],
    )

    response = client.post(
        "/api/site/theme", json={"theme": "signal-deck"}, headers=admin_headers()
    )

    assert response.status_code == 200
    assert order == ["audit", "set"]
    assert captured_audit == [
        {"actor_email": "admin@ceynex.dev", "action": "set_site_theme", "target": "signal-deck"}
    ]


def test_post_theme_is_refused_when_the_audit_row_cannot_be_written(client, monkeypatch):
    def audit_down(**_kw):
        raise psycopg.OperationalError("db down")

    changed: list[str] = []
    monkeypatch.setattr(audit_module, "record", audit_down)
    monkeypatch.setattr(
        site_settings_module, "set_theme", lambda theme, *, updated_by: changed.append(theme)
    )

    response = client.post(
        "/api/site/theme", json={"theme": "signal-deck"}, headers=admin_headers()
    )

    assert response.status_code == 503
    assert "audit" in response.json()["detail"]
    assert changed == []


def test_post_theme_outage_is_a_503(client, monkeypatch):
    def boom(theme, *, updated_by):
        raise psycopg.OperationalError("db down")

    monkeypatch.setattr(site_settings_module, "set_theme", boom)

    response = client.post(
        "/api/site/theme", json={"theme": "signal-deck"}, headers=admin_headers()
    )
    assert response.status_code == 503
