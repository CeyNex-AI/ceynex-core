"""Assertions for the usage surface (SRS 3.4.6, deviation D15).

The requirement here is easy to misread as already met. The rate limiter has
been enforcing since it shipped — but SRS 3.4.6 asks for restrictions to be
"disclosed to the user within the application rather than enforced silently",
and until there was a page saying so, it was enforcing silently. These tests are
mostly about that disclosure being honest, including about its own weakness.
"""

from __future__ import annotations

import psycopg
import pytest
from fastapi.testclient import TestClient

from ceynex.api.auth import DemoUser, issue_token
from ceynex.api.main import app
from ceynex.observability import spend
from ceynex.observability.spend import InProcessSpendCounter, RedisSpendCounter

USER = "policymaker@ceynex.dev"
ADMIN = "admin@ceynex.dev"


def auth(email: str, role: str) -> dict[str, str]:
    token = issue_token(DemoUser(email=email, role=role, password_hash=b""))
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def ledger_and_counter(monkeypatch):
    """No database and no Redis: the ledger's reads answer from memory, the
    instruction store is a dict, and the spend counter is a fresh one per test.
    These tests used to read the real ledger, so with the docker stack down the
    first of them failed — a unit test that needs a database is not a unit test.
    The instructions round trip then repeated the same mistake through
    `PUT /api/account/instructions`, which is why the store is faked here too."""
    from ceynex.observability.ledger import UsageRollup

    state = {"spent": {None: 0.0}, "down": False, "instructions": {}}

    async def get_instruction(user_email):
        return state["instructions"].get(user_email, ("", True))

    async def save_instruction(user_email, content, enabled=True):
        if state["down"]:
            raise psycopg.OperationalError("instructions down")
        state["instructions"][user_email] = (content, enabled)

    monkeypatch.setattr("ceynex.chat.instructions.get", get_instruction)
    monkeypatch.setattr("ceynex.chat.instructions.save", save_instruction)

    async def by_day(user_email, *, days=30):
        if state["down"]:
            raise psycopg.OperationalError("ledger down")
        return [UsageRollup(key="2026-09-10", calls=2, tokens_in=10, tokens_out=5,
                            cost_usd=0.002)]

    async def by_role_and_model(user_email, *, days=30):
        if state["down"]:
            raise psycopg.OperationalError("ledger down")
        return []

    async def spent_today(user_email=None):
        if state["down"]:
            raise psycopg.OperationalError("ledger down")
        return state["spent"].get(user_email, 0.0)

    monkeypatch.setattr("ceynex.observability.ledger.by_day", by_day)
    monkeypatch.setattr("ceynex.observability.ledger.by_role_and_model", by_role_and_model)
    monkeypatch.setattr("ceynex.observability.ledger.spent_today", spent_today)
    counter = InProcessSpendCounter()
    spend.set_shared_counter(counter)
    yield state, counter
    spend.set_shared_counter(None)


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

    # And what was saved is what comes back.
    assert client.get("/api/account/instructions", headers=headers).json()["content"] == (
        "Answer in bullet points."
    )


def test_saving_instructions_during_a_postgres_outage_is_a_503(client, ledger_and_counter):
    """The same answer `PUT /api/account/preferences` gives the same outage. It
    used to be a 500: the driver's error escaped the handler untouched."""
    state, _ = ledger_and_counter
    state["down"] = True
    response = client.put(
        "/api/account/instructions",
        json={"content": "Be terse.", "enabled": True},
        headers=auth(USER, "policymaker"),
    )
    assert response.status_code == 503


def test_instructions_require_a_signed_in_user(client):
    assert client.get("/api/account/instructions").status_code == 401
    assert client.put("/api/account/instructions", json={"content": "hi"}).status_code == 401


# --- D16: the per-reader budget, and the shared count -----------------------


def test_the_limits_page_states_the_readers_own_budget_and_when_it_resets(client):
    body = client.get("/api/usage/limits", headers=auth(USER, "policymaker")).json()
    assert body["per_user_daily_cap_usd"] == 1.0
    assert body["resets_at"].endswith("T00:00:00+00:00")
    assert body["your_budget_spent"] is False


async def test_a_spent_budget_is_disclosed_rather_than_enforced_silently(client,
                                                                        ledger_and_counter):
    _, counter = ledger_and_counter
    await counter.add(1.25, USER)
    body = client.get("/api/usage/limits", headers=auth(USER, "policymaker")).json()
    assert body["your_budget_spent"] is True
    assert body["deployment_cap_spent"] is False


def test_the_page_reports_the_ledgers_accounting(client, ledger_and_counter):
    state, _ = ledger_and_counter
    state["spent"] = {None: 2.5, USER: 0.4}
    body = client.get("/api/usage/limits", headers=auth(USER, "policymaker")).json()
    assert body["spent_today_usd"] == 2.5
    assert body["spent_today_by_you_usd"] == 0.4


async def test_a_ledger_outage_reports_the_counter_rather_than_a_false_zero(
    client, ledger_and_counter
):
    state, counter = ledger_and_counter
    state["down"] = True
    await counter.add(0.7, USER)
    body = client.get("/api/usage/limits", headers=auth(USER, "policymaker")).json()
    assert body["spent_today_by_you_usd"] == pytest.approx(0.7)


def test_reading_spend_during_a_ledger_outage_is_a_503_not_a_zero(client, ledger_and_counter):
    """"You have spent nothing" is a claim, and during an outage a false one."""
    state, _ = ledger_and_counter
    state["down"] = True
    response = client.get("/api/usage/summary", headers=auth(USER, "policymaker"))
    assert response.status_code == 503


def test_a_shared_counter_is_not_described_as_per_worker(client):
    """D16's point: once Redis carries the count, the cap is one number across
    workers, and the page stops warning that it is not."""
    spend.set_shared_counter(RedisSpendCounter(client=None))
    body = client.get("/api/usage/limits", headers=auth(USER, "policymaker")).json()
    assert body["cap_is_per_worker"] is False
