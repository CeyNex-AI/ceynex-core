"""RBAC: the `users` domain helpers that need no database, and the
admin-only account/role routes (SRS 3.5.4).

DB-free, like `test_auth.py`: `ceynex.api.users`' persistence functions are
monkeypatched with an in-memory store, and `ceynex.api.audit.record` is stubbed
so the admin routes' write-audit-first step doesn't need Postgres.
`test_users_integration.py` exercises the real `users` table.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ceynex.api import audit as audit_module
from ceynex.api import users as users_module
from ceynex.api.auth import issue_token
from ceynex.api.main import app

ADMIN = {"Authorization": f"Bearer {issue_token('admin@ceynex.dev', 'admin')}"}
RESEARCHER = {"Authorization": f"Bearer {issue_token('r@ceynex.dev', 'researcher')}"}


# --- domain helpers that raise before any DB call -----------------------


def test_create_user_rejects_an_unknown_role_before_touching_the_db():
    with pytest.raises(users_module.InvalidRoleError):
        users_module.create_user("x@ceynex.dev", "longenough", "superuser")


def test_create_user_rejects_a_short_password_before_touching_the_db():
    with pytest.raises(users_module.WeakPasswordError):
        users_module.create_user("x@ceynex.dev", "short", "researcher")


def test_password_hash_round_trips_and_rejects_the_wrong_password():
    h = users_module._hash_password("correct-horse")
    assert h != "correct-horse"
    assert users_module._verify_password("correct-horse", h) is True
    assert users_module._verify_password("wrong", h) is False


def test_verify_password_on_a_garbage_hash_is_false_not_an_error():
    assert users_module._verify_password("anything", "not-a-bcrypt-hash") is False


# --- in-memory store for the route tests ------------------------------


@pytest.fixture
def store(monkeypatch):
    rows: dict[int, users_module.User] = {}
    counter = {"n": 0}

    def _by_email(email):
        email = email.strip().lower()
        return next((u for u in rows.values() if u.email == email), None)

    def _enabled_admins(excluding_id=None):
        return [
            u for u in rows.values()
            if u.role == "admin" and not u.disabled and u.id != excluding_id
        ]

    def fake_create_user(email, password, role):
        email = email.strip().lower()
        if role not in users_module.VALID_ROLES:
            raise users_module.InvalidRoleError(role)
        if len(password) < users_module.MIN_PASSWORD_LENGTH:
            raise users_module.WeakPasswordError("too short")
        if _by_email(email):
            raise users_module.EmailTakenError(email)
        counter["n"] += 1
        user = users_module.User(
            id=counter["n"], email=email, role=role,
            created_at="2026-01-01T00:00:00+00:00", disabled=False,
        )
        rows[user.id] = user
        return user

    def fake_list_users():
        return [
            users_module.UserSummary(
                id=u.id, email=u.email, role=u.role, created_at=u.created_at, disabled=u.disabled
            )
            for u in sorted(rows.values(), key=lambda u: u.id)
        ]

    def fake_set_role(user_id, role):
        if role not in users_module.VALID_ROLES:
            raise users_module.InvalidRoleError(role)
        user = rows.get(user_id)
        if user is None:
            return None
        if user.role == "admin" and role != "admin" and not _enabled_admins(excluding_id=user_id):
            raise users_module.LastAdminError("cannot remove the last enabled admin")
        updated = users_module.User(
            id=user.id, email=user.email, role=role,
            created_at=user.created_at, disabled=user.disabled,
        )
        rows[user_id] = updated
        return updated

    def fake_set_disabled(user_id, *, disabled):
        user = rows.get(user_id)
        if user is None:
            return None
        if disabled and user.role == "admin" and not _enabled_admins(excluding_id=user_id):
            raise users_module.LastAdminError("cannot disable the last enabled admin")
        updated = users_module.User(
            id=user.id, email=user.email, role=user.role,
            created_at=user.created_at, disabled=disabled,
        )
        rows[user_id] = updated
        return updated

    monkeypatch.setattr(users_module, "create_user", fake_create_user)
    monkeypatch.setattr(users_module, "list_users", fake_list_users)
    monkeypatch.setattr(users_module, "set_role", fake_set_role)
    monkeypatch.setattr(users_module, "set_disabled", fake_set_disabled)
    return rows


@pytest.fixture
def audit_calls(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        audit_module, "record",
        lambda *, actor_email, action, target: calls.append(
            {"actor_email": actor_email, "action": action, "target": target}
        ),
    )
    return calls


@pytest.fixture
def client():
    return TestClient(app)


def _seed_admin(store):
    return store.__setitem__(
        1,
        users_module.User(
            id=1, email="admin@ceynex.dev", role="admin",
            created_at="2026-01-01T00:00:00+00:00", disabled=False,
        ),
    )


# --- the role gate ----------------------------------------------------


def test_user_routes_reject_a_non_admin(client, store, audit_calls):
    assert client.get("/api/admin/users", headers=RESEARCHER).status_code == 403
    assert client.post(
        "/api/admin/users", json={"email": "a@b.dev", "password": "longenough", "role": "exporter"},
        headers=RESEARCHER,
    ).status_code == 403


def test_user_routes_reject_no_token(client, store, audit_calls):
    assert client.get("/api/admin/users").status_code == 401


# --- list + create --------------------------------------------------


def test_list_users_returns_every_row(client, store, audit_calls):
    _seed_admin(store)
    body = client.get("/api/admin/users", headers=ADMIN).json()
    assert [u["email"] for u in body["users"]] == ["admin@ceynex.dev"]


def test_admin_creates_a_user_at_a_chosen_role_and_audits_first(client, store, audit_calls):
    resp = client.post(
        "/api/admin/users",
        json={"email": "New@Ceynex.dev", "password": "longenough", "role": "policymaker"},
        headers=ADMIN,
    )
    assert resp.status_code == 201
    assert resp.json()["role"] == "policymaker"
    assert resp.json()["email"] == "new@ceynex.dev"
    assert audit_calls[0]["action"] == "create_user"


def test_creating_a_duplicate_is_a_409(client, store, audit_calls):
    body = {"email": "dup@ceynex.dev", "password": "longenough", "role": "exporter"}
    client.post("/api/admin/users", json=body, headers=ADMIN)
    assert client.post("/api/admin/users", json=body, headers=ADMIN).status_code == 409


def test_creating_with_an_unknown_role_is_a_422(client, store, audit_calls):
    resp = client.post(
        "/api/admin/users",
        json={"email": "x@ceynex.dev", "password": "longenough", "role": "root"},
        headers=ADMIN,
    )
    assert resp.status_code == 422


def test_creating_with_a_short_password_is_a_422(client, store, audit_calls):
    resp = client.post(
        "/api/admin/users",
        json={"email": "x@ceynex.dev", "password": "short", "role": "exporter"},
        headers=ADMIN,
    )
    assert resp.status_code == 422


# --- role changes -------------------------------------------------


def test_admin_changes_a_users_role(client, store, audit_calls):
    _seed_admin(store)
    created = client.post(
        "/api/admin/users",
        json={"email": "u@ceynex.dev", "password": "longenough", "role": "researcher"},
        headers=ADMIN,
    ).json()
    resp = client.post(
        f"/api/admin/users/{created['id']}/role", json={"role": "admin"}, headers=ADMIN
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "admin"


def test_changing_the_role_of_an_unknown_user_is_a_404(client, store, audit_calls):
    assert client.post(
        "/api/admin/users/999/role", json={"role": "admin"}, headers=ADMIN
    ).status_code == 404


def test_changing_to_an_unknown_role_is_a_422(client, store, audit_calls):
    _seed_admin(store)
    assert client.post(
        "/api/admin/users/1/role", json={"role": "wizard"}, headers=ADMIN
    ).status_code == 422


def test_demoting_the_last_admin_is_a_409(client, store, audit_calls):
    _seed_admin(store)
    resp = client.post("/api/admin/users/1/role", json={"role": "researcher"}, headers=ADMIN)
    assert resp.status_code == 409


# --- disable / enable -------------------------------------------


def test_admin_disables_and_re_enables_a_user(client, store, audit_calls):
    _seed_admin(store)
    created = client.post(
        "/api/admin/users",
        json={"email": "u@ceynex.dev", "password": "longenough", "role": "exporter"},
        headers=ADMIN,
    ).json()
    uid = created["id"]
    assert client.post(f"/api/admin/users/{uid}/disable", headers=ADMIN).json()["disabled"] is True
    assert client.post(f"/api/admin/users/{uid}/enable", headers=ADMIN).json()["disabled"] is False


def test_disabling_an_unknown_user_is_a_404(client, store, audit_calls):
    assert client.post("/api/admin/users/999/disable", headers=ADMIN).status_code == 404


def test_disabling_the_last_admin_is_a_409(client, store, audit_calls):
    _seed_admin(store)
    assert client.post("/api/admin/users/1/disable", headers=ADMIN).status_code == 409
