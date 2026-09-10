"""Assertions for the usage surface (SRS 3.4.6, deviation D15).

The requirement here is easy to misread as already met. The rate limiter has
been enforcing since it shipped — but SRS 3.4.6 asks for restrictions to be
"disclosed to the user within the application rather than enforced silently",
and until there was a page saying so, it was enforcing silently. These tests are
mostly about that disclosure being honest, including about its own weakness.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ceynex.api.auth import DemoUser, issue_token
from ceynex.api.main import app

USER = "policymaker@ceynex.dev"
ADMIN = "admin@ceynex.dev"


def auth(email: str, role: str) -> dict[str, str]:
    token = issue_token(DemoUser(email=email, role=role, password_hash=b""))
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    return TestClient(app)


def test_usage_requires_a_signed_in_user(client):
    assert client.get("/api/usage/summary").status_code == 401


def test_a_reader_sees_their_own_usage(client):
    response = client.get("/api/usage/summary", headers=auth(USER, "policymaker"))
    assert response.status_code == 200
    assert response.json()["scope"] == "user"


def test_everyones_usage_is_admin_only(client):
    """It is everyone's data, so it takes the role that is allowed to see it."""
    assert client.get("/api/usage/all", headers=auth(USER, "policymaker")).status_code == 403
    assert client.get("/api/usage/all", headers=auth(ADMIN, "admin")).status_code == 200


def test_the_limits_page_states_the_limits_actually_in_force(client):
    """Read from config rather than restated, so the disclosure cannot drift
    from the enforcement it is describing."""
    body = client.get("/api/usage/limits", headers=auth(USER, "policymaker")).json()
    described = " ".join(item["key"] for item in body["limits"])
    assert "Questions" in described
    assert "Conversation turns" in described


def test_the_limits_page_admits_the_cap_is_per_worker(client):
    """`_cap_reached()` reads a per-process counter and the deployed image runs
    two workers, so real spend can reach twice the configured cap. Showing the
    cap as exact would be precisely the silent enforcement the requirement
    forbids — so the weakness travels in the response."""
    body = client.get("/api/usage/limits", headers=auth(USER, "policymaker")).json()
    assert body["cap_is_per_worker"] is True
    assert body["worker_count"] >= 2


def test_a_days_window_outside_the_allowed_range_is_rejected(client):
    assert client.get(
        "/api/usage/summary?days=0", headers=auth(USER, "policymaker")
    ).status_code == 422
    assert client.get(
        "/api/usage/summary?days=99999", headers=auth(USER, "policymaker")
    ).status_code == 422


def test_instructions_round_trip_and_are_capped(client):
    headers = auth(USER, "policymaker")
    assert client.get("/api/account/instructions", headers=headers).status_code == 200

    saved = client.put(
        "/api/account/instructions",
        json={"content": "  Answer in bullet points.  ", "enabled": True},
        headers=headers,
    )
    assert saved.status_code == 200
    assert saved.json()["content"] == "Answer in bullet points."

    # The cap is enforced by the schema, before anything reaches the database.
    too_long = client.put(
        "/api/account/instructions", json={"content": "x" * 5000}, headers=headers
    )
    assert too_long.status_code == 422


def test_instructions_require_a_signed_in_user(client):
    assert client.get("/api/account/instructions").status_code == 401
    assert client.put("/api/account/instructions", json={"content": "hi"}).status_code == 401
