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
    passwords: dict[int, str] = {}
    counter = {"n": 0}

    def _by_email(email):
        email = email.strip().lower()
        return next((u for u in rows.values() if u.email == email), None)

    def _enabled_admins(excluding_id=None):
        return [
            u for u in rows.values()
            if u.role == "admin" and not u.disabled and u.id != excluding_id
        ]

    def _put(user, **changes):
        fields = {
            "id": user.id, "email": user.email, "role": user.role,
            "created_at": user.created_at, "disabled": user.disabled,
            "token_epoch": user.token_epoch,
        }
        fields.update(changes)
        updated = users_module.User(**fields)
        rows[updated.id] = updated
        return updated

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
            created_at="2026-01-01T00:00:00+00:00", disabled=False, token_epoch=0,
        )
        rows[user.id] = user
        passwords[user.id] = password
        return user

    def fake_authenticate(email, password):
        user = _by_email(email)
        if user is None or user.disabled:
            return None
        return user if passwords.get(user.id) == password else None

    def fake_set_password(user_id, new_password):
        if len(new_password) < users_module.MIN_PASSWORD_LENGTH:
            raise users_module.WeakPasswordError("too short")
        user = rows.get(user_id)
        if user is None:
            return None
        passwords[user_id] = new_password
        return _put(user, token_epoch=user.token_epoch + 1)

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
        return _put(user, role=role, token_epoch=user.token_epoch + 1)

    def fake_set_disabled(user_id, *, disabled):
        user = rows.get(user_id)
        if user is None:
            return None
        if disabled and user.role == "admin" and not _enabled_admins(excluding_id=user_id):
            raise users_module.LastAdminError("cannot disable the last enabled admin")
        bump = user.token_epoch + 1 if disabled else user.token_epoch
        return _put(user, disabled=disabled, token_epoch=bump)

    def fake_set_email(user_id, new_email):
        new_email = new_email.strip().lower()
        user = rows.get(user_id)
        if user is None:
            return None
        if user.email == new_email:
            return user
        if _by_email(new_email):
            raise users_module.EmailTakenError(new_email)
        return _put(user, email=new_email, token_epoch=user.token_epoch + 1)

    def fake_delete_user(user_id):
        user = rows.get(user_id)
        if user is None:
            return False
        if user.role == "admin" and not user.disabled and not _enabled_admins(excluding_id=user_id):
            raise users_module.LastAdminError("cannot delete the last enabled admin")
        del rows[user_id]
        passwords.pop(user_id, None)
        return True

    # The `admin@ceynex.dev` account the module-level ADMIN token belongs to,
    # at id 1, with `counter` past it so `create_user` never collides.
    rows[1] = users_module.User(
        id=1, email="admin@ceynex.dev", role="admin",
        created_at="2026-01-01T00:00:00+00:00", disabled=False, token_epoch=0,
    )
    passwords[1] = "admin-fixture-pw"
    counter["n"] = 1

    monkeypatch.setattr(users_module, "create_user", fake_create_user)
    monkeypatch.setattr(users_module, "list_users", fake_list_users)
    monkeypatch.setattr(users_module, "set_role", fake_set_role)
    monkeypatch.setattr(users_module, "set_disabled", fake_set_disabled)
    monkeypatch.setattr(users_module, "authenticate", fake_authenticate)
    monkeypatch.setattr(users_module, "set_password", fake_set_password)
    monkeypatch.setattr(users_module, "set_email", fake_set_email)
    monkeypatch.setattr(users_module, "delete_user", fake_delete_user)
    # `current_token_epoch` is left as the conftest stub (→ 0 for everyone) so
    # the module-level RESEARCHER token keeps verifying without a row. The
    # invalidation tests opt into a store-aware version via `live_epoch`.
    return rows


@pytest.fixture
def live_epoch(store, monkeypatch):
    def _epoch(email):
        email = email.strip().lower()
        user = next((u for u in store.values() if u.email == email), None)
        return None if user is None or user.disabled else user.token_epoch

    monkeypatch.setattr(users_module, "current_token_epoch", _epoch)


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
            created_at="2026-01-01T00:00:00+00:00", disabled=False, token_epoch=0,
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


# --- self-service password change (/api/account/password) --------------


def _signup(client, email="u@ceynex.dev", password="origpass12"):
    r = client.post("/api/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201
    return r.json()["token"]


def test_change_password_with_the_correct_current_one(client, store):
    token = _signup(client)
    r = client.post(
        "/api/account/password",
        json={"current_password": "origpass12", "new_password": "brandnew34"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.json()["email"] == "u@ceynex.dev"
    assert r.json()["token"] and r.json()["token"] != token  # a fresh token comes back
    # old password stops working, new one works
    assert client.post(
        "/api/auth/login", json={"email": "u@ceynex.dev", "password": "origpass12"}
    ).status_code == 401
    assert client.post(
        "/api/auth/login", json={"email": "u@ceynex.dev", "password": "brandnew34"}
    ).status_code == 200


def test_change_password_kills_the_old_token_and_the_returned_one_works(client, store, live_epoch):
    token = _signup(client)
    fresh = client.post(
        "/api/account/password",
        json={"current_password": "origpass12", "new_password": "brandnew34"},
        headers={"Authorization": f"Bearer {token}"},
    ).json()["token"]
    hdr = lambda t: {"Authorization": f"Bearer {t}"}  # noqa: E731
    assert client.get("/api/auth/me", headers=hdr(token)).status_code == 401
    assert client.get("/api/auth/me", headers=hdr(fresh)).status_code == 200


def test_change_password_with_a_wrong_current_one_is_403(client, store):
    token = _signup(client)
    r = client.post(
        "/api/account/password",
        json={"current_password": "not-it", "new_password": "brandnew34"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


def test_change_password_to_the_same_value_is_422(client, store):
    token = _signup(client)
    r = client.post(
        "/api/account/password",
        json={"current_password": "origpass12", "new_password": "origpass12"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 422


def test_change_password_below_the_length_floor_is_422(client, store):
    token = _signup(client)
    r = client.post(
        "/api/account/password",
        json={"current_password": "origpass12", "new_password": "short"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 422


def test_change_password_needs_a_token(client, store):
    assert client.post(
        "/api/account/password",
        json={"current_password": "x", "new_password": "longenough1"},
    ).status_code == 401


# --- admin password reset (/api/admin/users/{id}/password) ------------


def test_admin_resets_a_users_password_without_the_old_one(client, store, audit_calls):
    _seed_admin(store)
    created = client.post(
        "/api/admin/users",
        json={"email": "locked@ceynex.dev", "password": "forgotten1", "role": "exporter"},
        headers=ADMIN,
    ).json()
    r = client.post(
        f"/api/admin/users/{created['id']}/password",
        json={"password": "freshpass99"},
        headers=ADMIN,
    )
    assert r.status_code == 200
    assert audit_calls[-1]["action"] == "set_user_password"
    assert client.post(
        "/api/auth/login", json={"email": "locked@ceynex.dev", "password": "freshpass99"}
    ).status_code == 200


def test_admin_password_reset_of_an_unknown_user_is_404(client, store, audit_calls):
    assert client.post(
        "/api/admin/users/999/password", json={"password": "whatever12"}, headers=ADMIN
    ).status_code == 404


def test_admin_password_reset_rejects_a_non_admin(client, store, audit_calls):
    assert client.post(
        "/api/admin/users/1/password", json={"password": "whatever12"}, headers=RESEARCHER
    ).status_code == 403


# --- an admin action cuts the target's live session --------------------


def _bearer(client, email, password):
    return {
        "Authorization": "Bearer "
        + client.post("/api/auth/signup", json={"email": email, "password": password}).json()["token"]
    }


def test_a_role_change_cuts_the_targets_existing_session(client, store, audit_calls, live_epoch):
    _seed_admin(store)
    worker = _bearer(client, "worker@ceynex.dev", "workerpass1")
    assert client.get("/api/auth/me", headers=worker).status_code == 200
    uid = next(
        u["id"] for u in client.get("/api/admin/users", headers=ADMIN).json()["users"]
        if u["email"] == "worker@ceynex.dev"
    )
    client.post(f"/api/admin/users/{uid}/role", json={"role": "exporter"}, headers=ADMIN)
    assert client.get("/api/auth/me", headers=worker).status_code == 401


def test_disabling_a_user_cuts_their_existing_session(client, store, audit_calls, live_epoch):
    _seed_admin(store)
    worker = _bearer(client, "worker@ceynex.dev", "workerpass1")
    uid = next(
        u["id"] for u in client.get("/api/admin/users", headers=ADMIN).json()["users"]
        if u["email"] == "worker@ceynex.dev"
    )
    client.post(f"/api/admin/users/{uid}/disable", headers=ADMIN)
    assert client.get("/api/auth/me", headers=worker).status_code == 401
    # re-enable does not resurrect the old token
    client.post(f"/api/admin/users/{uid}/enable", headers=ADMIN)
    assert client.get("/api/auth/me", headers=worker).status_code == 401


# --- change email (/api/account/email) -------------------------------


def test_change_email_moves_the_account_and_returns_a_fresh_token(client, store, live_epoch):
    token = _signup(client, email="old@ceynex.dev", password="origpass12")
    r = client.post(
        "/api/account/email",
        json={"current_password": "origpass12", "new_email": "New@Ceynex.dev"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.json()["email"] == "new@ceynex.dev"  # normalised
    fresh = r.json()["token"]
    # old token dies (its sub is the old email + old epoch); fresh one works
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {fresh}"}).json()["email"] == "new@ceynex.dev"
    # can log in under the new address, not the old
    assert client.post("/api/auth/login", json={"email": "new@ceynex.dev", "password": "origpass12"}).status_code == 200
    assert client.post("/api/auth/login", json={"email": "old@ceynex.dev", "password": "origpass12"}).status_code == 401


def test_change_email_to_a_taken_address_is_409(client, store):
    _signup(client, email="taken@ceynex.dev", password="takenpass1")
    token = _signup(client, email="me@ceynex.dev", password="mypass1234")
    r = client.post(
        "/api/account/email",
        json={"current_password": "mypass1234", "new_email": "taken@ceynex.dev"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 409


def test_change_email_with_a_wrong_password_is_403(client, store):
    token = _signup(client, email="me@ceynex.dev", password="mypass1234")
    r = client.post(
        "/api/account/email",
        json={"current_password": "nope", "new_email": "new@ceynex.dev"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


def test_change_email_needs_a_token(client, store):
    assert client.post(
        "/api/account/email", json={"current_password": "x", "new_email": "a@b.dev"}
    ).status_code == 401


# --- delete account (DELETE /api/account) ----------------------------


def test_delete_account_removes_it_and_the_password_stops_working(client, store):
    token = _signup(client, email="bye@ceynex.dev", password="byepass123")
    r = client.request(
        "DELETE", "/api/account",
        json={"current_password": "byepass123"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    assert client.post(
        "/api/auth/login", json={"email": "bye@ceynex.dev", "password": "byepass123"}
    ).status_code == 401


def test_delete_account_with_a_wrong_password_is_403(client, store):
    token = _signup(client, email="stay@ceynex.dev", password="staypass12")
    r = client.request(
        "DELETE", "/api/account",
        json={"current_password": "wrong"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403
    assert client.post(
        "/api/auth/login", json={"email": "stay@ceynex.dev", "password": "staypass12"}
    ).status_code == 200


def test_deleting_the_last_admin_is_409(client, store):
    # the fixture-seeded admin (id 1) is the only admin; give the test a way to
    # authenticate as it
    r = client.request(
        "DELETE", "/api/account",
        json={"current_password": "admin-fixture-pw"},
        headers=ADMIN,
    )
    assert r.status_code == 409


def test_delete_account_needs_a_token(client, store):
    assert client.request(
        "DELETE", "/api/account", json={"current_password": "x"}
    ).status_code == 401
