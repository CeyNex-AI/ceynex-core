"""What the token_epoch fail-open actually exposes, pinned as a test.

`verify_token` makes one Postgres lookup per authed request to check the
account's `current_token_epoch`, so a password change, role change or disable
cuts a live session at its next request. If that lookup raises (Postgres
unreachable) the token is accepted on its signature alone -- a deliberate
degrade-don't-fail choice so a datastore blip does not log everyone out.

`test_auth.py` already shows a *valid* token survives the blip. What no test
stated is the security cost of that choice: during the outage the same
fail-open also resurrects a token that was *already revoked*. SRS 3.1.6's plan
says this path "should be re-confirmed on a schedule rather than assumed
permanently safe, since a fail-open is exactly where a security regression
could hide silently." This test is that re-confirmation, and because it runs in
the unit suite it happens on every PR rather than on a calendar -- the exposure
window is bounded to a real Postgres outage, and a change that widened it (say,
fail-open on *any* exception, or skipping the epoch check when the DB is up)
would fail here.
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
    """The account's current epoch, mutable by the test. `verify_token` reads it
    through the patched `current_token_epoch`; raising is how a Postgres outage
    looks from inside verify_token."""
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


def test_fail_open_resurrects_a_revoked_token_during_a_postgres_outage(client, epoch_box):
    """The documented exposure, made explicit: a token revoked by an epoch bump
    is accepted again for as long as the epoch check cannot reach Postgres. This
    is the accepted availability/security trade-off, not a bug -- but it is now
    a tested, visible fact rather than an assumed-safe silent path."""
    token = issue_token(EMAIL, "researcher", token_epoch=0)
    epoch_box["value"] = 1
    assert _me(client, token) == 401  # revoked while the DB is up

    epoch_box["raise"] = True  # Postgres goes away
    assert _me(client, token) == 200  # revocation no longer enforced -- the exposure

    epoch_box["raise"] = False  # DB recovers
    assert _me(client, token) == 401  # and revocation is enforced again
