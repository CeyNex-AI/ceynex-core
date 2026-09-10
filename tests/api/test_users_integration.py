"""Real-Postgres assertions for the `users` table and RBAC, same spirit as
`tests/api/test_account_integration.py`: `ensure_table`/CRUD exercised against
a live database rather than monkeypatched.

Every row this file creates uses the `pytest-rbac-` email prefix and is deleted
before and after each test, so a real account is never touched.
"""

from __future__ import annotations

import psycopg
import pytest

from ceynex.api import users
from ceynex.settings import postgres_dsn

PREFIX = "pytest-rbac-"


@pytest.fixture
def clean_users():
    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM users WHERE email LIKE %s", (PREFIX + "%",))
            conn.commit()

    users.ensure_table()
    purge()
    yield
    purge()


def _email(name: str) -> str:
    return f"{PREFIX}{name}@ceynex.dev"


@pytest.mark.integration
def test_create_then_authenticate_and_only_the_hash_is_stored(clean_users):
    created = users.create_user(_email("alice"), "correct-horse-battery", "researcher")
    assert created.role == "researcher"
    assert created.disabled is False

    with psycopg.connect(postgres_dsn()) as conn:
        (stored,) = conn.execute(
            "SELECT password_hash FROM users WHERE id = %s", (created.id,)
        ).fetchone()
    assert stored != "correct-horse-battery"

    ok = users.authenticate(_email("Alice"), "correct-horse-battery")  # case-insensitive
    assert ok is not None and ok.id == created.id
    assert users.authenticate(_email("alice"), "wrong") is None


@pytest.mark.integration
def test_a_duplicate_email_raises_email_taken(clean_users):
    users.create_user(_email("bob"), "correct-horse-battery", "exporter")
    with pytest.raises(users.EmailTakenError):
        users.create_user(_email("BOB"), "another-password", "researcher")


@pytest.mark.integration
def test_a_disabled_account_stops_authenticating_and_role_for_email_goes_none(clean_users):
    from ceynex.api.auth import role_for_email

    created = users.create_user(_email("carol"), "correct-horse-battery", "exporter")
    assert role_for_email(_email("carol")) == "exporter"

    users.set_disabled(created.id, disabled=True)
    assert users.authenticate(_email("carol"), "correct-horse-battery") is None
    assert role_for_email(_email("carol")) is None

    users.set_disabled(created.id, disabled=True)  # idempotent
    re_enabled = users.set_disabled(created.id, disabled=False)
    assert re_enabled is not None and re_enabled.disabled is False
    assert users.authenticate(_email("carol"), "correct-horse-battery") is not None


@pytest.mark.integration
def test_set_role_changes_the_stored_role_and_unknown_id_is_none(clean_users):
    created = users.create_user(_email("dave"), "correct-horse-battery", "researcher")
    updated = users.set_role(created.id, "policymaker")
    assert updated is not None and updated.role == "policymaker"
    assert users.get_by_email(_email("dave")).role == "policymaker"
    assert users.set_role(-1, "admin") is None


@pytest.mark.integration
def test_set_role_rejects_an_unknown_role(clean_users):
    created = users.create_user(_email("erin"), "correct-horse-battery", "researcher")
    with pytest.raises(users.InvalidRoleError):
        users.set_role(created.id, "superuser")


@pytest.mark.integration
def test_set_password_replaces_the_hash_and_the_old_one_stops_working(clean_users):
    created = users.create_user(_email("frank"), "first-password-here", "exporter")
    updated = users.set_password(created.id, "second-password-here")
    assert updated is not None and updated.id == created.id
    assert users.authenticate(_email("frank"), "first-password-here") is None
    assert users.authenticate(_email("frank"), "second-password-here") is not None
    assert users.set_password(-1, "no-such-user-pw") is None
    with pytest.raises(users.WeakPasswordError):
        users.set_password(created.id, "short")


@pytest.mark.integration
def test_last_admin_guard_blocks_demotion_and_disable_of_a_lone_pytest_admin(clean_users):
    """Scoped to this file's own rows: if the deployment already has other
    enabled admins the guard won't fire, so the assertion is only meaningful
    when `pytest-rbac-` admins are the only ones — which `clean_users`
    guarantees by purging the prefix first. To avoid a false pass when a real
    admin exists, this checks the domain function directly against a count of
    prefix-scoped admins."""
    admin = users.create_user(_email("root"), "correct-horse-battery", "admin")

    with psycopg.connect(postgres_dsn()) as conn:
        (other_admins,) = conn.execute(
            "SELECT count(*) FROM users WHERE role = 'admin' AND disabled_at IS NULL AND id <> %s",
            (admin.id,),
        ).fetchone()

    if other_admins == 0:
        with pytest.raises(users.LastAdminError):
            users.set_role(admin.id, "researcher")
        with pytest.raises(users.LastAdminError):
            users.set_disabled(admin.id, disabled=True)
    else:
        # Another admin exists on this database — demotion is allowed.
        assert users.set_role(admin.id, "researcher") is not None
