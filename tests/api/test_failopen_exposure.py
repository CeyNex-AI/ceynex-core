"""What the token_epoch fail-open still exposes, pinned as a test.

`verify_token` makes one Postgres lookup per authed request to check the
account's `current_token_epoch`, so a password change, role change or disable
cuts a live session at its next request. If that lookup raises (Postgres
unreachable), a datastore blip must not log everyone out, so the token is
accepted on its signature, unless this worker has already seen the account's
epoch move past it.

Until 2026-10 the fail-open was total: during an outage a token that was
*already revoked* came back to life. SRS 3.1.6's plan says this path "should be
re-confirmed on a schedule rather than assumed permanently safe, since a
fail-open is exactly where a security regression could hide silently." These
tests are that re-confirmation, on every PR: they pin what is closed now, and
the one exposure left, so that widening it again fails here.
"""

from __future__ import annotations

import psycopg
import pytest
from fastapi.testclient import TestClient

from ceynex.api import users as users_module
from ceynex.api.auth import issue_token
from ceynex.api.main import app

EMAIL = "revoked@ceynex.dev"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def epoch_box(monkeypatch):
    """The account's current epoch, mutable by the test (None: disabled or gone).
    `verify_token` reads it through the patched `current_token_epoch`; raising is
    how a Postgres outage looks from inside verify_token."""
    box = {"value": 0, "raise": False}

    def _current(email: str):
        if box["raise"]:
            raise psycopg.OperationalError("connection refused")
        return box["value"]

    monkeypatch.setattr(users_module, "current_token_epoch", _current)
    return box


def _me(client: TestClient, token: str) -> int:
    return client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code


def test_revocation_is_enforced_while_postgres_is_reachable(client, epoch_box):
    token = issue_token(EMAIL, "researcher", token_epoch=0)
    assert _me(client, token) == 200
    epoch_box["value"] = 1  # a password/role change or disable advanced the epoch
    assert _me(client, token) == 401


def test_a_revoked_token_stays_revoked_through_an_outage(client, epoch_box):
    token = issue_token(EMAIL, "researcher", token_epoch=0)
    epoch_box["value"] = 1
    assert _me(client, token) == 401  # revoked while the DB is up

    epoch_box["raise"] = True  # Postgres goes away
    assert _me(client, token) == 401, "the outage must not bring a revoked token back"

    epoch_box["raise"] = False
    assert _me(client, token) == 401


def test_a_disabled_account_stays_out_through_an_outage(client, epoch_box):
    token = issue_token(EMAIL, "researcher", token_epoch=0)
    epoch_box["value"] = None  # disabled or deleted
    assert _me(client, token) == 401

    epoch_box["raise"] = True
    assert _me(client, token) == 401


def test_a_valid_session_survives_an_outage(client, epoch_box):
    """Degrade, don't fail: a blip must not sign everyone out."""
    token = issue_token(EMAIL, "researcher", token_epoch=0)
    assert _me(client, token) == 200

    epoch_box["raise"] = True
    assert _me(client, token) == 200


def test_the_remaining_exposure_an_account_this_worker_never_checked(client, epoch_box):
    """The exposure that is left, made explicit: a token whose epoch moved on
    while this worker never looked it up is accepted for as long as the lookup
    cannot reach Postgres. Bounded by the outage and by the token's 8 h TTL."""
    token = issue_token(EMAIL, "researcher", token_epoch=0)
    epoch_box["value"] = 1  # revoked, but this worker has not read it
    epoch_box["raise"] = True
    assert _me(client, token) == 200

    epoch_box["raise"] = False
    assert _me(client, token) == 401, "and enforced again the moment the DB is back"
