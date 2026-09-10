"""Real-Postgres assertions for notification preferences and API keys, same
spirit as `tests/api/test_history_integration.py`: `ensure_table`/CRUD
exercised against a live database instead of monkeypatched.

`preferences` has no notion of "valid account" (see its module docstring), so
these use a synthetic pytest-only email like `test_history_integration.py`
does. `api_keys.authenticate()` is different: it derives a role from
`auth.role_for_email`, which is DB-backed since RBAC -- so `clean_pytest_keys`
creates a real `users` row for `KEY_USER` (a distinctive pytest-only email,
not a demo account -- those are gone) and removes it afterwards along with the
keys.
"""

from __future__ import annotations

import contextlib

import psycopg
import pytest
from fastapi.testclient import TestClient

from ceynex.api import api_keys, preferences, users
from ceynex.api.auth import role_for_email
from ceynex.api.main import app
from ceynex.settings import postgres_dsn

PREFS_USER = "pytest-account@ceynex.dev"
KEY_USER = "pytest-account-integration@ceynex.dev"
KEY_LABEL = "pytest-account-integration key"


@pytest.fixture
def clean_prefs_user():
    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM notification_preferences WHERE user_email = %s", (PREFS_USER,))
            conn.commit()

    preferences.ensure_table()
    purge()
    yield
    purge()


@pytest.fixture
def clean_pytest_keys():
    """`KEY_USER` is a pytest-only account this fixture owns end to end: a real
    `users` row (so `api_keys.authenticate` can resolve a role), its keys, and
    its preferences row, all created here and removed afterwards."""

    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM api_keys WHERE user_email = %s", (KEY_USER,))
            conn.execute("DELETE FROM notification_preferences WHERE user_email = %s", (KEY_USER,))
            conn.execute("DELETE FROM users WHERE email = %s", (KEY_USER,))
            conn.commit()

    users.ensure_table()
    api_keys.ensure_table()
    purge()
    with contextlib.suppress(users.EmailTakenError):
        users.create_user(KEY_USER, "pytest-account-int-pw", "researcher")
    yield
    purge()


# --- preferences --------------------------------------------------------


@pytest.mark.integration
def test_preferences_default_before_any_upsert(clean_prefs_user):
    prefs = preferences.get_for_user(PREFS_USER)
    assert prefs == preferences.Preferences(
        dq_flag_alerts=True, forecast_updates=True, weekly_digest=False
    )


@pytest.mark.integration
def test_upsert_round_trips_and_a_second_call_updates_not_duplicates(clean_prefs_user):
    preferences.upsert(PREFS_USER, dq_flag_alerts=False, forecast_updates=True, weekly_digest=True)
    preferences.upsert(PREFS_USER, dq_flag_alerts=False, forecast_updates=False, weekly_digest=True)

    prefs = preferences.get_for_user(PREFS_USER)
    assert prefs == preferences.Preferences(
        dq_flag_alerts=False, forecast_updates=False, weekly_digest=True
    )
    with psycopg.connect(postgres_dsn()) as conn:
        (count,) = conn.execute(
            "SELECT count(*) FROM notification_preferences WHERE user_email = %s", (PREFS_USER,)
        ).fetchone()
    assert count == 1


# --- api keys ------------------------------------------------------------


@pytest.mark.integration
def test_a_created_key_authenticates_and_only_the_hash_is_stored(clean_pytest_keys):
    new_key = api_keys.create(KEY_USER, KEY_LABEL)
    assert new_key.key.startswith(api_keys.KEY_PREFIX)

    with psycopg.connect(postgres_dsn()) as conn:
        (stored,) = conn.execute("SELECT key_hash FROM api_keys WHERE id = %s", (new_key.id,)).fetchone()
    assert stored != new_key.key

    payload = api_keys.authenticate(new_key.key)
    assert payload is not None
    assert payload.email == KEY_USER
    assert payload.role == role_for_email(KEY_USER)


@pytest.mark.integration
def test_a_garbage_or_never_issued_key_does_not_authenticate(clean_pytest_keys):
    assert api_keys.authenticate("ck_" + "z" * 32) is None
    assert api_keys.authenticate("not-even-the-right-prefix") is None


@pytest.mark.integration
def test_a_revoked_key_stops_authenticating(clean_pytest_keys):
    new_key = api_keys.create(KEY_USER, KEY_LABEL)
    assert api_keys.revoke(new_key.id, KEY_USER) is True
    assert api_keys.authenticate(new_key.key) is None


@pytest.mark.integration
def test_revoking_someone_elses_key_does_nothing(clean_pytest_keys):
    new_key = api_keys.create(KEY_USER, KEY_LABEL)
    assert api_keys.revoke(new_key.id, "not-the-owner@ceynex.dev") is False
    assert api_keys.authenticate(new_key.key) is not None


@pytest.mark.integration
def test_list_for_user_reports_the_label_and_revoked_state(clean_pytest_keys):
    new_key = api_keys.create(KEY_USER, KEY_LABEL)
    api_keys.revoke(new_key.id, KEY_USER)

    entries = [e for e in api_keys.list_for_user(KEY_USER) if e.label == KEY_LABEL]
    assert len(entries) == 1
    assert entries[0].revoked is True


@pytest.mark.integration
def test_a_real_key_authenticates_a_require_user_route_end_to_end(clean_pytest_keys):
    """The end-to-end proof that "programmatic access" actually works: a
    freshly created key, used as a bearer token against a real route, with
    no monkeypatching of the auth path itself."""
    new_key = api_keys.create(KEY_USER, KEY_LABEL)

    with TestClient(app) as client:
        response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {new_key.key}"})

    assert response.status_code == 200
    assert response.json() == {"email": KEY_USER, "role": role_for_email(KEY_USER)}
